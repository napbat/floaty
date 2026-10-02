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

use super::super::bits::nan_32;
use super::super::environment::packed;
use super::super::paths::ready_for;
use super::kernel::{encodings, singles};
use super::{any_lane, chunk, chunk_mut, dispatch, isa};
use crate::env::{Env, Rounding};
use crate::format::internal::MinMax;
use crate::host::{Host, Isa, Load, Operation};

/// The sign bit of a binary32 encoding.
const SIGN: u32 = 0x8000_0000;

/// The encoding of 2^23, the least binary32 magnitude whose values are all
/// integral.
const INTEGRAL: u32 = 0x4B00_0000;

/// The encoding of 0.5.
const HALF: u32 = 0x3F00_0000;

/// The encoding of 1.
const ONE: u32 = 0x3F80_0000;

/// Returns `true` when the environment allows the host paths of binary32.
#[inline]
fn ready(env: &Env) -> bool {
    ready_for(Host::Single, env, Host::Single.precision())
}

/// Returns `operation` of each pair of binary32 encodings in the
/// instruction set `I`.
#[inline]
pub fn binary<I: Isa, const C: usize>(
    x: [u32; C],
    y: [u32; C],
    operation: Operation,
) -> Option<[u32; C]> {
    Some(encodings(isa::binary::<I, C>(
        &singles(x),
        &singles(y),
        operation,
    )?))
}

/// Returns the minimum or maximum operation `operation` of each pair of
/// binary32 encodings in the instruction set `I`: `Minimum`, `Maximum`,
/// `MinimumNumber`, or `MaximumNumber`. Returns `None` for another
/// operation.
#[inline]
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
#[inline]
fn settle(selected: u32, left: u32, right: u32, minimum: bool, number: bool) -> u32 {
    let (left_nan, right_nan) = (nan_32(left), nan_32(right));
    if (left | right) << 1 == 0 {
        if minimum { left | right } else { left & right }
    } else if left_nan == right_nan {
        if left_nan { left } else { selected }
    } else if left_nan == number {
        right
    } else {
        left
    }
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
        let mut steps = [0; C];
        steps
            .iter_mut()
            .zip(x.iter().zip(differences))
            .for_each(|(step, (&value, difference))| {
                *step = correction(value, difference, rounding);
            });
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
    let mut results = rounded;
    results.iter_mut().zip(x).for_each(|(result, value)| {
        *result = if value & !SIGN < INTEGRAL {
            (*result & !SIGN) | (value & SIGN)
        } else {
            value
        };
    });
    results
}

/// Returns the encoding of the step, +0, +1, or -1, that moves the nearest
/// even integer of `value` to its integer in the direction `rounding`.
/// `difference` is `value` minus that integer, exact, of a magnitude of at
/// most 0.5, or a NaN.
#[inline]
fn correction(value: u32, difference: u32, rounding: Rounding) -> u32 {
    let magnitude = difference & !SIGN;
    let inexact = magnitude != 0 && magnitude <= HALF;
    let tie = magnitude == HALF;
    // The rounding to nearest even moved the value toward zero when the
    // difference has the sign of the value.
    let toward_zero = (difference ^ value) & SIGN == 0;
    let away = value & SIGN | ONE;
    let back = (value ^ SIGN) & SIGN | ONE;
    let negative = difference & SIGN != 0;
    let step = match rounding {
        Rounding::TiesToAway => (tie && toward_zero).then_some(away),
        Rounding::TiesTowardZero => (tie && !toward_zero).then_some(back),
        Rounding::TowardPositive => (inexact && !negative).then_some(ONE),
        Rounding::TowardNegative => (inexact && negative).then_some(ONE | SIGN),
        Rounding::TowardZero => (inexact && !toward_zero).then_some(back),
        Rounding::AwayFromZero => (inexact && toward_zero).then_some(away),
        Rounding::TiesToEven | Rounding::ToOdd => None,
    };
    step.unwrap_or(0)
}

/// Returns each binary32 encoding converted to a 32-bit integer, to nearest
/// even, in the instruction set `I`, or `None` where it has no packed
/// conversion. A lane that the scalar conversion must decide holds
/// `i32::MIN`: a NaN, a value outside the range of `i32`, and `-2^31`.
#[inline]
fn integers<I: Isa, const C: usize>(x: [u32; C]) -> Option<[i32; C]> {
    if !I::INTEGERS {
        return None;
    }
    isa::to_int::<I, C>(&singles(x))
}

/// Returns a chunk of encodings, or `None` when the chunk holds a NaN, which
/// the engine selects. The test takes a slice, whose loop stays short until
/// the function inlines into a loop of chunks of a known length.
#[inline]
fn without_nan<const C: usize>(bits: [u32; C]) -> Option<[u32; C]> {
    (!any_lane(bits[..].iter().copied(), nan_32)).then_some(bits)
}

/// Computes the values of `values` `N` at a time, with one check of the
/// environment for the call. Calls `each` with the index of the first value
/// of each chunk, the index past its last value, and its encodings, or
/// `None` for a chunk that holds a NaN or has no instruction. Returns
/// `None`, and calls `each` for no chunk, when the path does not apply.
#[inline]
pub fn store<const N: usize>(
    values: impl Load,
    env: &Env,
    each: impl FnMut(usize, usize, Option<&[u32]>),
) -> Option<()> {
    if !ready(env) {
        return None;
    }
    dispatch::store::<N>(values, each);
    Some(())
}

/// Calls `each` for each chunk as `store` does, in the instruction set `I`,
/// with its features, in an environment that allows the path.
#[inline]
pub(super) fn store_on<I: Isa, const N: usize>(
    values: impl Load,
    each: impl FnMut(usize, usize, Option<&[u32]>),
) {
    I::run(move || in_value_chunks::<I, N, u32>(values, without_nan, each));
}

/// Converts the values of `values` to 32-bit integers to nearest even, `N`
/// at a time, with one check of the environment for the call. Calls `each`
/// with the index of the first value of each chunk, the index past its last
/// value, and its integers, or `None` for a chunk without an instruction. A
/// lane that the scalar conversion must decide holds `i32::MIN`. Returns
/// `None`, and calls `each` for no chunk, when the path does not apply.
#[inline]
pub fn to_int<const N: usize>(
    values: impl Load,
    env: &Env,
    each: impl FnMut(usize, usize, Option<&[i32]>),
) -> Option<()> {
    if !packed::INTEGERS || !ready(env) {
        return None;
    }
    dispatch::to_int::<N>(values, each);
    Some(())
}

/// Calls `each` for each chunk as `to_int` does, in the instruction set `I`,
/// with its features, in an environment that allows the path.
#[inline]
pub(super) fn to_int_on<I: Isa, const N: usize>(
    values: impl Load,
    each: impl FnMut(usize, usize, Option<&[i32]>),
) {
    I::run(move || in_value_chunks::<I, N, i32>(values, integers::<I, N>, each));
}

/// Loads the values of `values` `C` at a time in the instruction set `I`,
/// with +0 past the last value in the last chunk, and calls `each` with the
/// index of the first value of each chunk, the index past its last value,
/// and `compute` of its encodings, or `None` where a load or `compute` gives
/// `None`.
#[inline]
fn in_value_chunks<I: Isa, const C: usize, T>(
    values: impl Load,
    compute: impl Fn([u32; C]) -> Option<[T; C]>,
    mut each: impl FnMut(usize, usize, Option<&[T]>),
) {
    let count = values.count();
    let full = count - count % C;
    for start in (0..full).step_by(C) {
        let chunk = values.load::<I, C>(start).and_then(&compute);
        each(start, start + C, chunk.as_ref().map(|chunk| &chunk[..]));
    }
    if full < count {
        let chunk = values.load_rest::<I, C>(full).and_then(&compute);
        each(
            full,
            count,
            chunk.as_ref().map(|chunk| &chunk[..count - full]),
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
    if !ready(env) {
        return None;
    }
    dispatch::reduce::<N>(values, operation)
}

/// Returns the result of `reduce` in the instruction set `I`, with its
/// features, in an environment that allows the path.
#[inline]
pub(super) fn reduce_on<I: Isa, const N: usize>(
    values: impl Load,
    operation: MinMax,
) -> Option<u32> {
    I::run(move || {
        let identity = if operation.is_minimum() {
            0x7F80_0000
        } else {
            0xFF80_0000
        };
        let mut lanes = [identity; N];
        if I::EXTRA_WIDE && N.is_multiple_of(16) {
            reduce_into::<I, N, 16>(&mut lanes, values, operation, identity)?;
        } else if N.is_multiple_of(8) {
            reduce_into::<I, N, 8>(&mut lanes, values, operation, identity)?;
        } else if N.is_multiple_of(4) {
            reduce_into::<I, N, 4>(&mut lanes, values, operation, identity)?;
        } else {
            reduce_into::<I, N, 1>(&mut lanes, values, operation, identity)?;
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
/// lanes past the last value of the last chunk take the identity, which
/// changes no lane.
#[inline]
fn reduce_into<I: Isa, const N: usize, const C: usize>(
    lanes: &mut [u32; N],
    values: impl Load,
    operation: MinMax,
    identity: u32,
) -> Option<()> {
    let count = values.count();
    for start in (0..count).step_by(C) {
        let next = if start + C <= count {
            values.load::<I, C>(start)?
        } else {
            let mut rest = values.load_rest::<I, C>(start)?;
            rest[count - start..].fill(identity);
            rest
        };
        let offset = start % N;
        let old: [u32; C] = *chunk(lanes, offset);
        *chunk_mut(lanes, offset) = min_max::<I, C>(old, next, operation)?;
    }
    Some(())
}
