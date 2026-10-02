//! The instruction set of the build, and the operations on arrays of lanes
//! in any instruction set.
//!
//! `Build` has the packed forms of the features that the build enables. Each
//! function below computes an array of any length in the chunks of an
//! instruction set: sixteen lanes at a time where it has 512-bit registers,
//! then eight where it has 256-bit registers, then four, and then each other
//! lane alone. A lone lane runs in the four-lane form with the lane in every
//! position. The scalar instructions of the build would mix legacy SSE
//! encodings into a larger instruction set, and each lane of a packed form
//! gives the bits of the scalar form.

use super::super::environment::packed;
use super::super::{Isa, Operation};
use super::kernel::encodings;
use super::{chunk, chunk_mut, in_chunks_on};
use crate::env::Rounding;
use crate::format::internal::MinMax;
use crate::sealed::Sealed;

/// The instruction set of the features that the build enables.
#[derive(Clone, Copy, Debug)]
pub struct Build;

impl Sealed for Build {}

impl Isa for Build {
    const WIDE: bool = packed::WIDE;
    const EXTRA_WIDE: bool = packed::EXTRA_WIDE;
    const HALF: bool = packed::HALF;
    const INTEGERS: bool = packed::INTEGERS;

    #[inline]
    fn run<R>(f: impl FnOnce() -> R) -> R {
        f()
    }

    #[inline]
    fn binary_f32x4(left: [f32; 4], right: [f32; 4], operation: Operation) -> Option<[f32; 4]> {
        Some(packed::binary_f32x4(left, right, operation))
    }

    #[inline]
    fn binary_f32x8(left: [f32; 8], right: [f32; 8], operation: Operation) -> Option<[f32; 8]> {
        packed::binary_f32x8(left, right, operation)
    }

    #[inline]
    fn binary_f32x16(left: [f32; 16], right: [f32; 16], operation: Operation) -> Option<[f32; 16]> {
        packed::binary_f32x16(left, right, operation)
    }

    #[inline]
    fn mul_add_f32x4(left: [f32; 4], right: [f32; 4], addend: [f32; 4]) -> Option<[f32; 4]> {
        packed::mul_add_f32x4(left, right, addend)
    }

    #[inline]
    fn mul_add_f32x8(left: [f32; 8], right: [f32; 8], addend: [f32; 8]) -> Option<[f32; 8]> {
        packed::mul_add_f32x8(left, right, addend)
    }

    #[inline]
    fn mul_add_f32x16(left: [f32; 16], right: [f32; 16], addend: [f32; 16]) -> Option<[f32; 16]> {
        packed::mul_add_f32x16(left, right, addend)
    }

    #[inline]
    fn min_max_f32x4(left: [f32; 4], right: [f32; 4], operation: MinMax) -> Option<[f32; 4]> {
        Some(packed::min_max_f32x4(left, right, operation))
    }

    #[inline]
    fn min_max_f32x8(left: [f32; 8], right: [f32; 8], operation: MinMax) -> Option<[f32; 8]> {
        packed::min_max_f32x8(left, right, operation)
    }

    #[inline]
    fn min_max_f32x16(left: [f32; 16], right: [f32; 16], operation: MinMax) -> Option<[f32; 16]> {
        packed::min_max_f32x16(left, right, operation)
    }

    #[inline]
    fn round_f32x4(value: [f32; 4], rounding: Rounding) -> Option<[f32; 4]> {
        packed::round_f32x4(value, rounding)
    }

    #[inline]
    fn round_f32x8(value: [f32; 8], rounding: Rounding) -> Option<[f32; 8]> {
        packed::round_f32x8(value, rounding)
    }

    #[inline]
    fn round_f32x16(value: [f32; 16], rounding: Rounding) -> Option<[f32; 16]> {
        packed::round_f32x16(value, rounding)
    }

    #[inline]
    fn to_int_f32x4(value: [f32; 4]) -> Option<[i32; 4]> {
        Some(packed::to_int_f32x4(value))
    }

    #[inline]
    fn to_int_f32x8(value: [f32; 8]) -> Option<[i32; 8]> {
        packed::to_int_f32x8(value)
    }

    #[inline]
    fn to_int_f32x16(value: [f32; 16]) -> Option<[i32; 16]> {
        packed::to_int_f32x16(value)
    }

    #[inline]
    fn from_int_x4(value: [i32; 4]) -> Option<[f32; 4]> {
        Some(packed::from_int_x4(value))
    }

    #[inline]
    fn from_int_x8(value: [i32; 8]) -> Option<[f32; 8]> {
        packed::from_int_x8(value)
    }

    #[inline]
    fn from_int_x16(value: [i32; 16]) -> Option<[f32; 16]> {
        packed::from_int_x16(value)
    }

    #[inline]
    fn widen_halves_x4(value: [u16; 4]) -> Option<[f32; 4]> {
        packed::widen_halves_x4(value)
    }

    #[inline]
    fn widen_halves_x8(value: [u16; 8]) -> Option<[f32; 8]> {
        packed::widen_halves_x8(value)
    }

    #[inline]
    fn widen_halves_x16(value: [u16; 16]) -> Option<[f32; 16]> {
        packed::widen_halves_x16(value)
    }

    #[inline]
    fn narrow_halves_x4(value: [f32; 4]) -> Option<[u16; 4]> {
        packed::narrow_halves_x4(value)
    }

    #[inline]
    fn narrow_halves_x8(value: [f32; 8]) -> Option<[u16; 8]> {
        packed::narrow_halves_x8(value)
    }

    #[inline]
    fn narrow_halves_x16(value: [f32; 16]) -> Option<[u16; 16]> {
        packed::narrow_halves_x16(value)
    }

    #[inline]
    fn widen_x2(value: [f32; 2]) -> Option<[f64; 2]> {
        Some(packed::widen_x2(value))
    }

    #[inline]
    fn widen_x4(value: [f32; 4]) -> Option<[f64; 4]> {
        packed::widen_x4(value)
    }

    #[inline]
    fn narrow_x2(value: [f64; 2]) -> Option<[f32; 2]> {
        Some(packed::narrow_x2(value))
    }

    #[inline]
    fn narrow_x4(value: [f64; 4]) -> Option<[f32; 4]> {
        packed::narrow_x4(value)
    }
}

/// Computes the `N` lanes of `lanes` in the chunks of the instruction set
/// `I`, as the module states. Each function takes the index of its first
/// lane. Returns `None` when a function has no instruction.
#[inline]
fn in_single_chunks<I: Isa, T: Copy, const N: usize>(
    lanes: &mut [T; N],
    extra_wide: impl Fn(usize) -> Option<[T; 16]>,
    wide: impl Fn(usize) -> Option<[T; 8]>,
    narrow: impl Fn(usize) -> Option<[T; 4]>,
    lane: impl Fn(usize) -> Option<T>,
) -> Option<()> {
    let extra_wide_end = if I::EXTRA_WIDE { N - N % 16 } else { 0 };
    let wide_end = if I::WIDE {
        N - (N - extra_wide_end) % 8
    } else {
        extra_wide_end
    };
    let narrow_end = N - (N - wide_end) % 4;
    for start in (0..extra_wide_end).step_by(16) {
        *chunk_mut(lanes, start) = extra_wide(start)?;
    }
    for start in (extra_wide_end..wide_end).step_by(8) {
        *chunk_mut(lanes, start) = wide(start)?;
    }
    for start in (wide_end..narrow_end).step_by(4) {
        *chunk_mut(lanes, start) = narrow(start)?;
    }
    for (index, value) in lanes.iter_mut().enumerate().skip(narrow_end) {
        *value = lane(index)?;
    }
    Some(())
}

/// Returns `operation` of each pair of lanes of `x` and `y`.
#[inline]
pub fn binary<I: Isa, const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    operation: Operation,
) -> Option<[f32; N]> {
    let mut lanes = [0.0; N];
    in_single_chunks::<I, f32, N>(
        &mut lanes,
        |start| I::binary_f32x16(*chunk(x, start), *chunk(y, start), operation),
        |start| I::binary_f32x8(*chunk(x, start), *chunk(y, start), operation),
        |start| I::binary_f32x4(*chunk(x, start), *chunk(y, start), operation),
        |index| Some(I::binary_f32x4([x[index]; 4], [y[index]; 4], operation)?[0]),
    )?;
    Some(lanes)
}

/// Returns `x * y + z` of each triple of lanes, rounded once.
#[inline]
pub fn mul_add<I: Isa, const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    z: &[f32; N],
) -> Option<[f32; N]> {
    let mut lanes = [0.0; N];
    in_single_chunks::<I, f32, N>(
        &mut lanes,
        |start| I::mul_add_f32x16(*chunk(x, start), *chunk(y, start), *chunk(z, start)),
        |start| I::mul_add_f32x8(*chunk(x, start), *chunk(y, start), *chunk(z, start)),
        |start| I::mul_add_f32x4(*chunk(x, start), *chunk(y, start), *chunk(z, start)),
        |index| Some(I::mul_add_f32x4([x[index]; 4], [y[index]; 4], [z[index]; 4])?[0]),
    )?;
    Some(lanes)
}

/// Returns the lane of each pair of `x` and `y` that the minimum or maximum
/// instruction of `operation` selects.
#[inline]
pub fn min_max<I: Isa, const N: usize>(
    x: &[f32; N],
    y: &[f32; N],
    operation: MinMax,
) -> Option<[f32; N]> {
    let mut lanes = [0.0; N];
    in_single_chunks::<I, f32, N>(
        &mut lanes,
        |start| I::min_max_f32x16(*chunk(x, start), *chunk(y, start), operation),
        |start| I::min_max_f32x8(*chunk(x, start), *chunk(y, start), operation),
        |start| I::min_max_f32x4(*chunk(x, start), *chunk(y, start), operation),
        |index| Some(I::min_max_f32x4([x[index]; 4], [y[index]; 4], operation)?[0]),
    )?;
    Some(lanes)
}

/// Returns each lane rounded to an integral value in the direction
/// `rounding`, or `None` where the instruction set has no instruction for
/// the direction.
#[inline]
pub fn round<I: Isa, const N: usize>(x: &[f32; N], rounding: Rounding) -> Option<[f32; N]> {
    let mut lanes = [0.0; N];
    in_single_chunks::<I, f32, N>(
        &mut lanes,
        |start| I::round_f32x16(*chunk(x, start), rounding),
        |start| I::round_f32x8(*chunk(x, start), rounding),
        |start| I::round_f32x4(*chunk(x, start), rounding),
        |index| Some(I::round_f32x4([x[index]; 4], rounding)?[0]),
    )?;
    Some(lanes)
}

/// Returns each lane rounded to a 32-bit integer to nearest even, with the
/// integer indefinite for a lane that the scalar conversion must decide.
#[inline]
pub fn to_int<I: Isa, const N: usize>(x: &[f32; N]) -> Option<[i32; N]> {
    let mut lanes = [0; N];
    in_single_chunks::<I, i32, N>(
        &mut lanes,
        |start| I::to_int_f32x16(*chunk(x, start)),
        |start| I::to_int_f32x8(*chunk(x, start)),
        |start| I::to_int_f32x4(*chunk(x, start)),
        |index| Some(I::to_int_f32x4([x[index]; 4])?[0]),
    )?;
    Some(lanes)
}

/// Returns each 32-bit integer rounded to binary32.
#[inline]
pub fn from_int<I: Isa, const N: usize>(x: &[i32; N]) -> Option<[f32; N]> {
    let mut lanes = [0.0; N];
    in_single_chunks::<I, f32, N>(
        &mut lanes,
        |start| I::from_int_x16(*chunk(x, start)),
        |start| I::from_int_x8(*chunk(x, start)),
        |start| I::from_int_x4(*chunk(x, start)),
        |index| Some(I::from_int_x4([x[index]; 4])?[0]),
    )?;
    Some(lanes)
}

/// Returns the binary32 encoding of each binary16 encoding widened exactly,
/// or `None` where the instruction set has no widening instruction. Each
/// chunk becomes encodings as it leaves its instruction, so the lanes stay
/// in vectors up to the store of the caller.
#[inline]
pub fn widen_halves<I: Isa, const N: usize>(x: &[u16; N]) -> Option<[u32; N]> {
    let mut lanes = [0; N];
    in_single_chunks::<I, u32, N>(
        &mut lanes,
        |start| I::widen_halves_x16(*chunk(x, start)).map(encodings),
        |start| I::widen_halves_x8(*chunk(x, start)).map(encodings),
        |start| I::widen_halves_x4(*chunk(x, start)).map(encodings),
        |index| Some(I::widen_halves_x4([x[index]; 4])?[0].to_bits()),
    )?;
    Some(lanes)
}

/// Returns each lane rounded to binary16 to nearest even, or `None` where
/// the instruction set has no rounding instruction.
#[inline]
pub fn narrow_halves<I: Isa, const N: usize>(x: &[f32; N]) -> Option<[u16; N]> {
    let mut halves = [0; N];
    in_single_chunks::<I, u16, N>(
        &mut halves,
        |start| I::narrow_halves_x16(*chunk(x, start)),
        |start| I::narrow_halves_x8(*chunk(x, start)),
        |start| I::narrow_halves_x4(*chunk(x, start)),
        |index| Some(I::narrow_halves_x4([x[index]; 4])?[0]),
    )?;
    Some(halves)
}

/// Returns each binary32 lane widened exactly to binary64: four lanes at a
/// time where the instruction set has 256-bit registers, then two, and then
/// one.
#[inline]
pub fn widen<I: Isa, const N: usize>(x: &[f32; N]) -> Option<[f64; N]> {
    let mut lanes = [0.0; N];
    in_chunks_on::<I, f64, N, 4, 2>(
        &mut lanes,
        |start| I::widen_x4(*chunk(x, start)),
        |start| I::widen_x2(*chunk(x, start)),
        |index| Some(I::widen_x2([x[index]; 2])?[0]),
    )?;
    Some(lanes)
}

/// Returns each binary64 lane rounded to binary32, as `widen` chunks the
/// lanes.
#[inline]
pub fn narrow<I: Isa, const N: usize>(x: &[f64; N]) -> Option<[f32; N]> {
    let mut lanes = [0.0; N];
    in_chunks_on::<I, f32, N, 4, 2>(
        &mut lanes,
        |start| I::narrow_x4(*chunk(x, start)),
        |start| I::narrow_x2(*chunk(x, start)),
        |index| Some(I::narrow_x2([x[index]; 2])?[0]),
    )?;
    Some(lanes)
}
