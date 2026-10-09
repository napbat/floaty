//! The elementwise views of `Lanes` on the host unit, and the loops that
//! store, reduce, and convert their values with one check of the environment.
//!
//! A view computes a chunk of binary32 encodings from the chunks of its
//! operands, in the instruction set of the loop that reads it. Each
//! instruction gives the bits of the engine for operands that are not NaNs.
//! A NaN result keeps the NaN of the instruction, which the caller sends to
//! the engine. The minimum and maximum operations settle a pair with a NaN or
//! two zeros in integer instructions, because the instructions of the hosts
//! treat those pairs differently. The loops compute in the instruction set
//! that `dispatch` selects.
//!
//! Each step that rounds, in a view or in `to_int`, rounds in the direction
//! of the mode. To nearest even, it takes the forms of the environment. In
//! the other directions that `dispatch::direction` allows, it takes the
//! forms with a rounding control, which only the instruction sets of those
//! directions have.

use super::super::bits::nan_32;
use super::super::environment::packed;
use super::kernel::{encodings, singles};
use super::{Directed, Nearest, Task, any_lane, chunk, chunk_mut, dispatch, isa, run_task};
use crate::env::{Env, Rounding};
use crate::format::internal::MinMax;
use crate::host::{Direction, Isa, Load, Operation};

/// The sign bit of a binary32 encoding.
const SIGN: u32 = 0x8000_0000;

/// The encoding of 2^23, the least binary32 magnitude whose values are all
/// integral.
const INTEGRAL: u32 = 0x4B00_0000;

/// The encoding of 0.5.
const HALF: u32 = 0x3F00_0000;

/// The encoding of 1.
const ONE: u32 = 0x3F80_0000;

/// Returns the rounding direction of the mode when the mode, the
/// environment, and this processor allow the host paths of binary32, as
/// `dispatch::direction` states.
#[inline]
fn ready(env: &Env) -> Option<Rounding> {
    dispatch::direction(env)
}

/// Returns `operation` of each pair of binary32 encodings in the
/// instruction set `I`, rounded in the direction `D`.
#[inline]
pub fn binary<I: Isa, const C: usize, D: Direction>(
    x: [u32; C],
    y: [u32; C],
    operation: Operation,
    direction: D,
) -> Option<[u32; C]> {
    let (x, y) = (singles(x), singles(y));
    let results = if D::NEAREST {
        isa::binary::<I, C>(&x, &y, operation)
    } else {
        isa::binary_rounded::<I, C>(&x, &y, operation, direction.rounding())
    };
    Some(encodings(results?))
}

/// Returns the minimum or maximum operation `operation` of each pair of
/// binary32 encodings in the instruction set `I`: `Minimum`, `Maximum`,
/// `MinimumNumber`, or `MaximumNumber`. Returns `None` for another
/// operation.
// With `#[inline]`, LLVM leaves a copy of this function out of line in a
// loop of chunks. A copy for `V3` or `V4` then has no features of the set,
// and a call passes the lanes through memory. Inline in every loop,
// `maximum_of` on 1,280 values of the operations bench takes 1,049 ns for
// 1,514 ns in `Build`, and 1,106 ns for 1,441 ns in x86-64-v3.
#[allow(clippy::inline_always)]
#[inline(always)]
pub fn min_max<I: Isa, const C: usize>(
    x: [u32; C],
    y: [u32; C],
    operation: MinMax,
) -> Option<[u32; C]> {
    let number = match operation {
        MinMax::Minimum | MinMax::Maximum => false,
        MinMax::MinimumNumber | MinMax::MaximumNumber => true,
        _ => return None,
    };
    let mut results = encodings(isa::min_max::<I, C>(&singles(x), &singles(y), operation)?);
    results
        .iter_mut()
        .zip(x.iter().zip(&y))
        .for_each(|(result, (&left, &right))| {
            *result = settle(*result, left, right, operation.is_minimum(), number);
        });
    Some(results)
}

/// Returns the minimum or maximum of `left` and `right`: `selected`, the
/// lane of the instruction, unless the pair holds a NaN or two zeros. Of two
/// zeros, -0 is the smaller. A `number` operation gives the other operand of
/// one NaN, and the other operations give a NaN. The engine selects the NaN
/// of a NaN result.
///
/// The function selects with masks of all ones or all zeros, not branches,
/// so that LLVM computes more lanes of a chunk in vector instructions. With
/// branches, `maximum_of` on 1,280 values of the operations bench took
/// 1,102 ns in x86-64-v3 and 1,051 ns in `Build`; with masks it takes 549 ns
/// and 397 ns.
#[inline]
fn settle(selected: u32, left: u32, right: u32, minimum: bool, number: bool) -> u32 {
    let mask = |condition: bool| 0_u32.wrapping_sub(u32::from(condition));
    let (left_nan, right_nan) = (mask(nan_32(left)), mask(nan_32(right)));
    let zeros = mask((left | right) << 1 == 0);
    let zero = if minimum { left | right } else { left & right };
    // Of one NaN, a `number` operation takes the other operand, and the
    // other operations take the NaN. Of two NaNs, the result is `left`.
    let right_wins = (left_nan ^ right_nan) & if number { left_nan } else { !left_nan };
    let nan = (right & right_wins) | (left & !right_wins);
    let any_nan = left_nan | right_nan;
    let unordered = (nan & any_nan) | (selected & !any_nan);
    (zero & zeros) | (unordered & !zeros)
}

/// Returns each binary32 encoding rounded to an integral value in the
/// direction `rounding` in the instruction set `I`, or `None` for `ToOdd`.
/// The instructions of the direction round where the instruction set has
/// them. Otherwise the value rounds to nearest even, and an exact correction
/// moves the result by one in the direction.
#[inline]
pub fn round_to_integral<I: Isa, const C: usize>(
    x: [u32; C],
    rounding: Rounding,
) -> Option<[u32; C]> {
    I::run(move || {
        if let Some(rounded) = round_in_instructions::<I, C>(x, rounding) {
            return Some(rounded);
        }
        if rounding == Rounding::ToOdd {
            return None;
        }
        let even = match round_in_instructions::<I, C>(x, Rounding::TiesToEven) {
            Some(even) => even,
            None => nearest_even::<I, C>(x)?,
        };
        if rounding == Rounding::TiesToEven {
            return Some(even);
        }
        // The difference of a value and its nearest integer is exact: it
        // holds the low bits of the value, or the value itself below 1. An
        // infinity gives a NaN difference, and a large value a zero
        // difference, so neither moves.
        let differences = encodings(isa::binary::<I, C>(
            &singles(x),
            &singles(even),
            Operation::Sub,
        )?);
        // One loop for each direction selects each step with masks, which
        // LLVM computes in vector instructions. With the direction tested in
        // each lane, `TiesToAway` on 1,280 values took 0.96 ns per value in
        // x86-64-v3 on a Ryzen AI Max+ 395, against 0.28 ns.
        let steps = match rounding {
            Rounding::TiesToAway => {
                corrections(&x, differences, |c| c.tie & c.toward_zero & c.away)
            }
            Rounding::TiesTowardZero => {
                corrections(&x, differences, |c| c.tie & !c.toward_zero & c.back)
            }
            Rounding::TowardPositive => {
                corrections(&x, differences, |c| c.inexact & !c.negative & ONE)
            }
            Rounding::TowardNegative => {
                corrections(&x, differences, |c| c.inexact & c.negative & (ONE | SIGN))
            }
            Rounding::TowardZero => {
                corrections(&x, differences, |c| c.inexact & !c.toward_zero & c.back)
            }
            Rounding::AwayFromZero => {
                corrections(&x, differences, |c| c.inexact & c.toward_zero & c.away)
            }
            // The function returned above for these directions.
            Rounding::TiesToEven | Rounding::ToOdd => [0; C],
        };
        let moved = encodings(isa::binary::<I, C>(
            &singles(even),
            &singles(steps),
            Operation::Add,
        )?);
        Some(with_signs(x, moved))
    })
}

/// Returns each binary32 encoding rounded to an integral value by the
/// instructions of the direction `rounding`, or `None` where the
/// instruction set `I` has none.
#[inline]
fn round_in_instructions<I: Isa, const C: usize>(
    x: [u32; C],
    rounding: Rounding,
) -> Option<[u32; C]> {
    Some(encodings(isa::round::<I, C>(&singles(x), rounding)?))
}

/// Returns each binary32 encoding rounded to the nearest integral value, a
/// tie to even, by the sum and difference with 2^23 of the sign of the
/// value. The sum rounds to an integer for a magnitude below 2^23, and the
/// difference is exact. A larger magnitude, an infinity, and a NaN stay as
/// they are.
#[inline]
fn nearest_even<I: Isa, const C: usize>(x: [u32; C]) -> Option<[u32; C]> {
    let mut shifts = [0; C];
    shifts
        .iter_mut()
        .zip(&x)
        .for_each(|(shift, &value)| *shift = (value & SIGN) | INTEGRAL);
    let shifts = singles(shifts);
    let sums = isa::binary::<I, C>(&singles(x), &shifts, Operation::Add)?;
    let rounded = encodings(isa::binary::<I, C>(&sums, &shifts, Operation::Sub)?);
    Some(with_signs(x, rounded))
}

/// Returns `rounded` with the sign of each value of `x` where the magnitude
/// of the value is below 2^23, and the value itself elsewhere. An integral
/// result of a rounding has the sign of the value, and a value of a larger
/// magnitude is integral.
#[inline]
fn with_signs<const C: usize>(x: [u32; C], rounded: [u32; C]) -> [u32; C] {
    // Rust 1.89.0 (LLVM 20.1.7) and nightly 771916f90 (LLVM 23.1.0)
    // emit masked VPTERNLOGD with immediate 0xE4 for the combined selection.
    // That instruction needs 0xB8 for the emitted operand order.
    // Select the magnitude first so the sign copy remains unmasked.
    let mut results = rounded;
    results.iter_mut().zip(x).for_each(|(result, value)| {
        let magnitude = if value & !SIGN < INTEGRAL {
            *result
        } else {
            value
        };
        *result = (magnitude & !SIGN) | (value & SIGN);
    });
    results
}

/// The parts of the step, +0, +1, or -1, that moves the nearest even
/// integer of a value to its integer in another direction. Each condition
/// is a mask: all ones where it holds, and zero elsewhere.
#[derive(Clone, Copy)]
struct Correction {
    /// The value is not integral: the difference is not zero and not a NaN.
    inexact: u32,
    /// The value lies halfway between two integers.
    tie: u32,
    /// The rounding to nearest even moved the value toward zero.
    toward_zero: u32,
    /// The difference is negative: the rounding moved the value up.
    negative: u32,
    /// The encoding of the step away from zero.
    away: u32,
    /// The encoding of the step back toward zero.
    back: u32,
}

impl Correction {
    /// Returns the parts for `value` and `difference`: `value` minus its
    /// nearest even integer, exact, of a magnitude of at most 0.5, or a NaN.
    #[inline]
    fn new(value: u32, difference: u32) -> Self {
        let mask = |condition: bool| 0_u32.wrapping_sub(u32::from(condition));
        let magnitude = difference & !SIGN;
        Self {
            inexact: mask(magnitude != 0) & mask(magnitude <= HALF),
            tie: mask(magnitude == HALF),
            // The difference has the sign of the value.
            toward_zero: mask((difference ^ value) & SIGN == 0),
            negative: mask(difference & SIGN != 0),
            away: value & SIGN | ONE,
            back: (value ^ SIGN) & SIGN | ONE,
        }
    }
}

/// Returns the step of each value of `x`, as `step` selects it from the
/// [`Correction`] of the value and its difference in `differences`.
#[inline]
fn corrections<const C: usize>(
    x: &[u32; C],
    differences: [u32; C],
    step: impl Fn(Correction) -> u32,
) -> [u32; C] {
    let mut steps = [0; C];
    steps
        .iter_mut()
        .zip(x.iter().zip(differences))
        .for_each(|(result, (&value, difference))| {
            *result = step(Correction::new(value, difference));
        });
    steps
}

/// Returns each binary32 encoding converted to a 32-bit integer in the
/// direction `D`, in the instruction set `I`, or `None` where it has no
/// packed conversion. A lane that the scalar conversion must decide holds
/// `i32::MIN`: a NaN, a value outside the range of `i32`, and `-2^31`.
#[inline]
fn integers<I: Isa, const C: usize, D: Direction>(x: [u32; C], direction: D) -> Option<[i32; C]> {
    if !I::INTEGERS {
        return None;
    }
    if D::NEAREST {
        isa::to_int::<I, C>(&singles(x))
    } else {
        isa::to_int_rounded::<I, C>(&singles(x), direction.rounding())
    }
}

/// Returns `true` when a chunk of encodings holds a NaN, which the engine
/// selects. The test takes a slice, whose loop stays short until the
/// function inlines into a loop of chunks of a known length.
#[inline]
fn holds_nan(bits: &[u32]) -> bool {
    any_lane(bits.iter().copied(), nan_32)
}

/// Computes the values of `values` `N` at a time, and the values past the
/// last chunk of `N` [`PART`] at a time, with one check of the environment
/// for the call. Calls `each` with the index of the first value of each
/// chunk, the index past its last value, and its encodings, or `None` for a
/// chunk that holds a NaN or has no instruction. Returns `None`, and calls
/// `each` for no chunk, when the path does not apply.
#[inline]
pub fn store<const N: usize>(
    values: impl Load,
    env: &Env,
    each: impl FnMut(usize, usize, Option<&[u32]>),
) -> Option<()> {
    let rounding = ready(env)?;
    run_task(Store::<_, _, N> {
        values,
        rounding,
        each,
    });
    Some(())
}

/// The arguments of `store`, whose task calls `each` for each chunk in the
/// instruction set of the processor, with its features, in an environment
/// that allows the path, rounding in the direction `rounding`.
struct Store<V, E, const N: usize> {
    values: V,
    rounding: Rounding,
    each: E,
}

impl<V: Load, E: FnMut(usize, usize, Option<&[u32]>), const N: usize> Task for Store<V, E, N> {
    type Output = ();

    #[inline]
    fn run<I: Isa>(self) {
        let Self {
            values,
            rounding,
            each,
        } = self;
        if rounding == Rounding::TiesToEven {
            store_in::<I, N>(values, Nearest, each);
        } else {
            store_in::<I, N>(values, Directed(rounding), each);
        }
    }
}

/// Calls `each` as the task of `store` does, in the copy of the loop for the
/// direction type of `direction`.
#[inline]
fn store_in<I: Isa, const N: usize>(
    values: impl Load,
    direction: impl Direction,
    each: impl FnMut(usize, usize, Option<&[u32]>),
) {
    I::run(move || {
        // A closure at each call, not a shared function: with one codegen
        // unit, LLVM called a shared test out of line from the copy of each
        // instruction set, and the lanes went through memory. The
        // difference of two slices of 1,280 values then took 0.20 ns per
        // value in x86-64-v3 on a Ryzen AI Max+ 395, against 0.05 ns.
        in_value_chunks::<I, N, u32>(
            values,
            direction,
            |bits| (!holds_nan(&bits)).then_some(bits),
            |bits| (!holds_nan(&bits)).then_some(bits),
            each,
        );
    });
}

/// Converts the values of `values` to 32-bit integers in the direction of
/// the mode, in the chunks of `store`, with one check of the environment for
/// the call. Calls `each` with the index of the first value of each chunk,
/// the index past its last value, and its integers, or `None` for a chunk
/// without an instruction. A lane that the scalar conversion must decide
/// holds `i32::MIN`. Returns `None`, and calls `each` for no chunk, when the
/// path does not apply.
#[inline]
pub fn to_int<const N: usize>(
    values: impl Load,
    env: &Env,
    each: impl FnMut(usize, usize, Option<&[i32]>),
) -> Option<()> {
    if !packed::INTEGERS {
        return None;
    }
    let rounding = ready(env)?;
    run_task(ToInt::<_, _, N> {
        values,
        rounding,
        each,
    });
    Some(())
}

/// The arguments of `to_int`, whose task calls `each` for each chunk in the
/// instruction set of the processor, with its features, in an environment
/// that allows the path, rounding in the direction `rounding`.
struct ToInt<V, E, const N: usize> {
    values: V,
    rounding: Rounding,
    each: E,
}

impl<V: Load, E: FnMut(usize, usize, Option<&[i32]>), const N: usize> Task for ToInt<V, E, N> {
    type Output = ();

    #[inline]
    fn run<I: Isa>(self) {
        let Self {
            values,
            rounding,
            each,
        } = self;
        if rounding == Rounding::TiesToEven {
            to_int_in::<I, N>(values, Nearest, each);
        } else {
            to_int_in::<I, N>(values, Directed(rounding), each);
        }
    }
}

/// Calls `each` as the task of `to_int` does, in the copy of the loop for the
/// direction type of `direction`.
#[inline]
fn to_int_in<I: Isa, const N: usize>(
    values: impl Load,
    direction: impl Direction,
    each: impl FnMut(usize, usize, Option<&[i32]>),
) {
    I::run(move || {
        in_value_chunks::<I, N, i32>(
            values,
            direction,
            |x| integers::<I, N, _>(x, direction),
            |x| integers::<I, PART, _>(x, direction),
            each,
        );
    });
}

/// The lanes of each chunk past the last chunk of `N` values, where `N` is
/// larger. The kernels add their last values in chunks of eight lanes too.
/// A short vector then loads and computes its own values, not a chunk of
/// `N` lanes that it pads.
const PART: usize = 8;

/// Loads the values of `values` `C` at a time in the instruction set `I`,
/// rounding in the direction `direction`, and calls `each` with the index of
/// the first value of each chunk, the index past its last value, and
/// `compute` of its encodings, or `None` where a load or `compute` gives
/// `None`. The values past the last chunk of `C` take chunks of [`PART`] and
/// `compute_part` when `C` is larger.
#[inline]
fn in_value_chunks<I: Isa, const C: usize, T>(
    values: impl Load,
    direction: impl Direction,
    compute: impl Fn([u32; C]) -> Option<[T; C]>,
    compute_part: impl Fn([u32; PART]) -> Option<[T; PART]>,
    mut each: impl FnMut(usize, usize, Option<&[T]>),
) {
    let count = values.count();
    let full = count - count % C;
    for start in (0..full).step_by(C) {
        let chunk = values.load::<I, C>(start, direction).and_then(&compute);
        each(start, start + C, chunk.as_ref().map(|chunk| &chunk[..]));
    }
    if C > PART {
        in_rest::<I, PART, T>(values, full, direction, compute_part, each);
    } else {
        in_rest::<I, C, T>(values, full, direction, compute, each);
    }
}

/// Calls `each` for the values of `values` from `start` as
/// `in_value_chunks` does, `W` at a time, with +0 past the last value in
/// the last chunk.
#[inline]
fn in_rest<I: Isa, const W: usize, T>(
    values: impl Load,
    start: usize,
    direction: impl Direction,
    compute: impl Fn([u32; W]) -> Option<[T; W]>,
    mut each: impl FnMut(usize, usize, Option<&[T]>),
) {
    let count = values.count();
    for first in (start..count).step_by(W) {
        let end = count.min(first + W);
        let chunk = if end - first == W {
            values.load::<I, W>(first, direction)
        } else {
            values.load_rest::<I, W>(first, direction)
        };
        let chunk = chunk.and_then(&compute);
        each(
            first,
            end,
            chunk.as_ref().map(|chunk| &chunk[..end - first]),
        );
    }
}

/// Returns the minimum or maximum operation `operation` of the values of
/// `values` in the order of the reductions of `Lanes`: `N` lanes from the
/// identity of the operation, value `i` into lane `i % N`, and then the
/// lanes by halves. Returns `None` when the path does not apply, when an
/// operation has no instruction, and when the result is a NaN.
#[inline]
pub fn reduce<const N: usize>(values: impl Load, operation: MinMax, env: &Env) -> Option<u32> {
    let rounding = ready(env)?;
    run_task(Reduce::<_, N> {
        values,
        operation,
        rounding,
    })
}

/// The lanes of a reduction, on a boundary of 64 bytes: the line of the
/// cache, and the width of a 512-bit register. `reduce_into` stores a chunk
/// of lanes and loads it again for the next chunk. A chunk that crosses a
/// line makes the load wait for the store: the reduction of `maximum_of` on
/// 1,280 values took 293 or 435 ns in x86-64-v4 on a Ryzen AI Max+ 395 with
/// the lanes on a 4-byte boundary, as the build placed them, and 280 ns
/// aligned.
#[repr(C, align(64))]
struct Aligned<T>(T);

/// The arguments of `reduce`, whose task gives its result in the
/// instruction set of the processor, with its features, in an environment
/// that allows the path. The minimum and maximum select a value, and the
/// loads of `values` round in the direction `rounding`.
struct Reduce<V, const N: usize> {
    values: V,
    operation: MinMax,
    rounding: Rounding,
}

impl<V: Load, const N: usize> Task for Reduce<V, N> {
    type Output = Option<u32>;

    #[inline]
    fn run<I: Isa>(self) -> Option<u32> {
        let Self {
            values,
            operation,
            rounding,
        } = self;
        if rounding == Rounding::TiesToEven {
            reduce_in::<I, N>(values, operation, Nearest)
        } else {
            reduce_in::<I, N>(values, operation, Directed(rounding))
        }
    }
}

/// Returns the result of the task of `reduce` in the copy of the loop for
/// the direction type of `direction`.
#[inline]
fn reduce_in<I: Isa, const N: usize>(
    values: impl Load,
    operation: MinMax,
    direction: impl Direction,
) -> Option<u32> {
    I::run(move || {
        let identity = if operation.is_minimum() {
            0x7F80_0000
        } else {
            0xFF80_0000
        };
        let mut lanes = Aligned([identity; N]);
        let lanes = &mut lanes.0;
        if I::EXTRA_WIDE && N.is_multiple_of(16) {
            reduce_into::<I, N, 16>(lanes, values, operation, identity, direction)?;
        } else if N.is_multiple_of(8) {
            reduce_into::<I, N, 8>(lanes, values, operation, identity, direction)?;
        } else if N.is_multiple_of(4) {
            reduce_into::<I, N, 4>(lanes, values, operation, identity, direction)?;
        } else {
            reduce_into::<I, N, 1>(lanes, values, operation, identity, direction)?;
        }
        let mut half = N / 2;
        while half > 0 {
            let (low, high) = lanes.split_at_mut(half);
            for (low, &high) in low.iter_mut().zip(&*high) {
                let [result] = min_max::<I, 1>([*low], [high], operation)?;
                *low = result;
            }
            half /= 2;
        }
        (!nan_32(lanes[0])).then_some(lanes[0])
    })
}

/// Combines each value of `values` into its lane, `C` lanes at a time. The
/// values of each whole part of `N` combine at fixed offsets, so the lanes
/// stay in registers. The lanes past the last value of the last chunk take
/// the identity, which changes no lane. Each load rounds in the direction
/// `direction`.
#[inline]
fn reduce_into<I: Isa, const N: usize, const C: usize>(
    lanes: &mut [u32; N],
    values: impl Load,
    operation: MinMax,
    identity: u32,
    direction: impl Direction,
) -> Option<()> {
    let count = values.count();
    let full = count - count % N;
    for start in (0..full).step_by(N) {
        // One check of the bounds for each part serves every load from it.
        let part = values.part(start, N);
        for offset in (0..N).step_by(C) {
            let old: [u32; C] = *chunk(lanes, offset);
            let next = part.load::<I, C>(offset, direction)?;
            *chunk_mut(lanes, offset) = min_max::<I, C>(old, next, operation)?;
        }
    }
    // Value `full + i` combines into lane `i`.
    for offset in (0..count - full).step_by(C) {
        let start = full + offset;
        let next = if start + C <= count {
            values.load::<I, C>(start, direction)?
        } else {
            let mut rest = values.load_rest::<I, C>(start, direction)?;
            rest[count - start..].fill(identity);
            rest
        };
        let old: [u32; C] = *chunk(lanes, offset);
        *chunk_mut(lanes, offset) = min_max::<I, C>(old, next, operation)?;
    }
    Some(())
}
