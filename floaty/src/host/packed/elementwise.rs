//! The elementwise views of `Lanes` on the host unit, and the loops that
//! store, reduce, and convert their values with one check of the environment.
//!
//! A view computes a chunk of binary32 encodings from the chunks of its
//! operands. Each instruction gives the bits of the engine for operands that
//! are not NaNs. A NaN result keeps the NaN of the instruction, which the
//! caller sends to the engine. The minimum and maximum operations settle a
//! pair with a NaN or two zeros in integer instructions, because the
//! instructions of the hosts treat those pairs differently.

use super::super::bits::nan_32;
use super::super::environment::{self, packed};
use super::super::paths::{min_max_f32, ready_for};
use super::kernel::{encodings, singles};
use super::{any_lane, chunk, chunk_mut, in_chunks, indefinite_unless_i32};
use crate::env::{Env, Rounding};
use crate::format::internal::MinMax;
use crate::host::{Host, Load, Operation};

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

/// Returns `operation` of each pair of host values: eight lanes at a time
/// where the build has wide registers, then four, and then one.
#[inline]
fn apply<const C: usize>(x: [f32; C], y: [f32; C], operation: Operation) -> Option<[f32; C]> {
    let mut lanes = [0.0; C];
    in_chunks::<f32, C, 8, 4>(
        &mut lanes,
        |start| packed::binary_f32x8(*chunk(&x, start), *chunk(&y, start), operation),
        |start| {
            Some(packed::binary_f32x4(
                *chunk(&x, start),
                *chunk(&y, start),
                operation,
            ))
        },
        |index| Some(environment::binary_f32(x[index], y[index], operation)),
    )?;
    Some(lanes)
}

/// Returns `operation` of each pair of binary32 encodings.
#[inline]
pub fn binary<const C: usize>(x: [u32; C], y: [u32; C], operation: Operation) -> Option<[u32; C]> {
    Some(encodings(apply(singles(x), singles(y), operation)?))
}

/// Returns the minimum or maximum operation `operation` of each pair of
/// binary32 encodings: `Minimum`, `Maximum`, `MinimumNumber`, or
/// `MaximumNumber`. Returns `None` for another operation.
#[inline]
pub fn min_max<const C: usize>(x: [u32; C], y: [u32; C], operation: MinMax) -> Option<[u32; C]> {
    let number = match operation {
        MinMax::Minimum | MinMax::Maximum => false,
        MinMax::MinimumNumber | MinMax::MaximumNumber => true,
        _ => return None,
    };
    let (left, right) = (singles(x), singles(y));
    let mut lanes = [0.0; C];
    in_chunks::<f32, C, 8, 4>(
        &mut lanes,
        |start| packed::min_max_f32x8(*chunk(&left, start), *chunk(&right, start), operation),
        |start| {
            Some(packed::min_max_f32x4(
                *chunk(&left, start),
                *chunk(&right, start),
                operation,
            ))
        },
        |index| Some(min_max_f32(left[index], right[index], operation)),
    )?;
    let mut results = encodings(lanes);
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
/// direction `rounding`, or `None` for `ToOdd`. The instructions of the
/// direction round where the build has them. Otherwise the value rounds to
/// nearest even, and an exact correction moves the result by one in the
/// direction.
#[inline]
pub fn round_to_integral<const C: usize>(x: [u32; C], rounding: Rounding) -> Option<[u32; C]> {
    if let Some(rounded) = round_in_instructions(x, rounding) {
        return Some(rounded);
    }
    if rounding == Rounding::ToOdd {
        return None;
    }
    let even = match round_in_instructions(x, Rounding::TiesToEven) {
        Some(even) => even,
        None => nearest_even(x)?,
    };
    if rounding == Rounding::TiesToEven {
        return Some(even);
    }
    // The difference of a value and its nearest integer is exact: it holds
    // the low bits of the value, or the value itself below 1. An infinity
    // gives a NaN difference, and a large value a zero difference, so
    // neither moves.
    let differences = encodings(apply(singles(x), singles(even), Operation::Sub)?);
    let mut steps = [0; C];
    steps
        .iter_mut()
        .zip(x.iter().zip(differences))
        .for_each(|(step, (&value, difference))| *step = correction(value, difference, rounding));
    let moved = encodings(apply(singles(even), singles(steps), Operation::Add)?);
    Some(with_signs(x, moved))
}

/// Returns each binary32 encoding rounded to an integral value by the
/// instructions of the direction `rounding`, or `None` where the build has
/// none.
#[inline]
fn round_in_instructions<const C: usize>(x: [u32; C], rounding: Rounding) -> Option<[u32; C]> {
    let values = singles(x);
    let mut lanes = [0.0; C];
    in_chunks::<f32, C, 8, 4>(
        &mut lanes,
        |start| packed::round_f32x8(*chunk(&values, start), rounding),
        |start| packed::round_f32x4(*chunk(&values, start), rounding),
        |index| environment::round_f32(values[index], rounding),
    )?;
    Some(encodings(lanes))
}

/// Returns each binary32 encoding rounded to the nearest integral value, a
/// tie to even, by the sum and difference with 2^23 of the sign of the
/// value. The sum rounds to an integer for a magnitude below 2^23, and the
/// difference is exact. A larger magnitude, an infinity, and a NaN stay as
/// they are.
#[inline]
fn nearest_even<const C: usize>(x: [u32; C]) -> Option<[u32; C]> {
    let mut shifts = [0; C];
    shifts
        .iter_mut()
        .zip(&x)
        .for_each(|(shift, &value)| *shift = (value & SIGN) | INTEGRAL);
    let shifts = singles(shifts);
    let sums = apply(singles(x), shifts, Operation::Add)?;
    let rounded = encodings(apply(sums, shifts, Operation::Sub)?);
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
/// even, from the host unit, or `None` where the build has no packed
/// conversion. A lane that the scalar conversion must decide holds
/// `i32::MIN`: a NaN, a value outside the range of `i32`, and `-2^31`.
#[inline]
fn integers<const C: usize>(x: [u32; C]) -> Option<[i32; C]> {
    if !packed::INTEGERS {
        return None;
    }
    let values = singles(x);
    let mut lanes = [0; C];
    in_chunks::<i32, C, 8, 4>(
        &mut lanes,
        |start| packed::to_int_f32x8(*chunk(&values, start)),
        |start| Some(packed::to_int_f32x4(*chunk(&values, start))),
        |index| {
            Some(indefinite_unless_i32(environment::to_int_f32(
                values[index],
            )))
        },
    )?;
    Some(lanes)
}

/// Returns a chunk of encodings, or `None` when the chunk is `None` or holds
/// a NaN, which the engine selects.
#[inline]
fn without_nan<const N: usize>(bits: Option<[u32; N]>) -> Option<[u32; N]> {
    bits.filter(|bits| !any_lane(bits.iter().copied(), nan_32))
}

/// Computes the values of `values` `N` at a time, with one check of the
/// environment for every chunk. Calls `each` with the index of the first
/// value of each chunk and its encodings, or `None` for a chunk that holds a
/// NaN or has no instruction. The last chunk holds +0 past the last value.
/// Returns `None`, and calls `each` for no chunk, when the path does not
/// apply.
#[inline]
pub fn store<const N: usize>(
    values: impl Load,
    env: &Env,
    mut each: impl FnMut(usize, Option<[u32; N]>),
) -> Option<()> {
    if !ready(env) {
        return None;
    }
    let count = values.count();
    let full = count - count % N;
    for start in (0..full).step_by(N) {
        each(start, without_nan(values.load::<N>(start)));
    }
    if full < count {
        each(full, without_nan(values.load_rest::<N>(full)));
    }
    Some(())
}

/// Converts the values of `values` to 32-bit integers to nearest even, `N`
/// at a time, with one check of the environment for every chunk. Calls
/// `each` with the index of the first value of each chunk and its integers,
/// or `None` for a chunk without an instruction. A lane that the scalar
/// conversion must decide holds `i32::MIN`. Returns `None`, and calls `each`
/// for no chunk, when the path does not apply.
#[inline]
pub fn to_int<const N: usize>(
    values: impl Load,
    env: &Env,
    mut each: impl FnMut(usize, Option<[i32; N]>),
) -> Option<()> {
    if !packed::INTEGERS || !ready(env) {
        return None;
    }
    let count = values.count();
    let full = count - count % N;
    for start in (0..full).step_by(N) {
        each(start, values.load::<N>(start).and_then(integers));
    }
    if full < count {
        each(full, values.load_rest::<N>(full).and_then(integers));
    }
    Some(())
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
    let identity = if operation.is_minimum() {
        0x7F80_0000
    } else {
        0xFF80_0000
    };
    let mut lanes = [identity; N];
    if N.is_multiple_of(8) {
        reduce_into::<N, 8>(&mut lanes, values, operation, identity)?;
    } else if N.is_multiple_of(4) {
        reduce_into::<N, 4>(&mut lanes, values, operation, identity)?;
    } else {
        reduce_into::<N, 1>(&mut lanes, values, operation, identity)?;
    }
    let mut half = N / 2;
    while half > 0 {
        let (low, high) = lanes.split_at_mut(half);
        for (low, &high) in low.iter_mut().zip(&*high) {
            let [result] = min_max([*low], [high], operation)?;
            *low = result;
        }
        half /= 2;
    }
    (!nan_32(lanes[0])).then_some(lanes[0])
}

/// Combines each value of `values` into its lane, `C` lanes at a time. The
/// lanes past the last value of the last chunk take the identity, which
/// changes no lane.
#[inline]
fn reduce_into<const N: usize, const C: usize>(
    lanes: &mut [u32; N],
    values: impl Load,
    operation: MinMax,
    identity: u32,
) -> Option<()> {
    let count = values.count();
    for start in (0..count).step_by(C) {
        let next = if start + C <= count {
            values.load::<C>(start)?
        } else {
            let mut rest = values.load_rest::<C>(start)?;
            rest[count - start..].fill(identity);
            rest
        };
        let offset = start % N;
        let old: [u32; C] = *chunk(lanes, offset);
        *chunk_mut(lanes, offset) = min_max(old, next, operation)?;
    }
    Some(())
}
