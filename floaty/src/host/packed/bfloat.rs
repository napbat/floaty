//! The packed paths of bfloat16 lanes, which a shift widens to binary32.
//!
//! A bfloat16 encoding is the high half of a binary32 encoding, so a shift
//! widens each lane exactly, with no floating-point instruction. The
//! integral value of a bfloat16 value in each direction is a bfloat16
//! value, so the low 16 bits of the binary32 integral value are zero, and a
//! shift narrows it exactly.
//!
//! The arithmetic computes in the packed binary32 instructions and rounds
//! each binary32 result to bfloat16 in integer instructions. binary32 holds
//! `2p + 2` bits of bfloat16, so the two roundings give the correctly rounded
//! result, as the module `paths` states. The lanes after the last chunk take
//! the scalar path.

use super::super::Operation;
use super::super::bits::nan_bfloat;
use super::super::environment::{self, packed};
use super::super::narrow::round_to_bfloat;
use super::{chunk, encodings_u16, in_chunks, lanes_u16, singles};
use crate::env::{Mode, Rounding};
use crate::float::Float;
use crate::format::Standard;

/// Returns the binary32 value of a bfloat16 encoding.
#[inline]
fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Returns the bfloat16 encoding of a binary32 value whose low 16 bits are
/// zero, as those of the integral value of a bfloat16 value are.
#[inline]
fn narrow_integral(value: f32) -> u16 {
    u16::try_from(value.to_bits() >> 16).expect("a shift by 16 leaves 16 bits")
}

/// Returns the bfloat16 encoding of a binary32 value rounded to nearest
/// even. A NaN gives a NaN.
#[inline]
fn narrow(value: f32) -> u16 {
    round_to_bfloat(value.to_bits())
}

/// Returns `operation` of each pair of bfloat16 lanes, or `None` when a lane
/// is a NaN.
#[inline]
pub(super) fn binary<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: Operation,
) -> Option<[Float<S, W, M>; N]> {
    let (x, y) = (encodings_u16(left)?, encodings_u16(right)?);
    lanes_u16(left, nan_bfloat, |lanes| {
        in_chunks::<u16, N, 8, 4>(
            lanes,
            |start| {
                let (a, b) = (chunk(x, start).map(widen), chunk(y, start).map(widen));
                Some(packed::binary_f32x8(a, b, operation)?.map(narrow))
            },
            |start| {
                let (a, b) = (chunk(x, start).map(widen), chunk(y, start).map(widen));
                Some(packed::binary_f32x4(a, b, operation).map(narrow))
            },
            |index| {
                let (a, b) = (widen(x[index]), widen(y[index]));
                Some(narrow(environment::binary_f32(a, b, operation)))
            },
        )
    })
}

/// Returns the square root of each bfloat16 lane, or `None` when a lane is
/// a NaN.
#[inline]
pub(super) fn sqrt<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[Float<S, W, M>; N]> {
    let x = encodings_u16(value)?;
    lanes_u16(value, nan_bfloat, |lanes| {
        in_chunks::<u16, N, 8, 4>(
            lanes,
            |start| Some(packed::sqrt_f32x8(chunk(x, start).map(widen))?.map(narrow)),
            |start| Some(packed::sqrt_f32x4(chunk(x, start).map(widen)).map(narrow)),
            |index| Some(narrow(environment::sqrt_f32(widen(x[index])))),
        )
    })
}

/// Returns the bfloat16 encoding of each binary32 lane rounded to nearest
/// even. A NaN lane gives a NaN.
#[inline]
pub(super) fn from_singles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[u16; N]> {
    Some(singles(value)?.map(narrow))
}

/// Returns each bfloat16 lane rounded to an integral value in the direction
/// `rounding`, or `None` when the build has no instruction or a lane is a
/// NaN.
#[inline]
pub(super) fn round_to_integral<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    rounding: Rounding,
) -> Option<[Float<S, W, M>; N]> {
    let x = encodings_u16(value)?;
    lanes_u16(value, nan_bfloat, |lanes| {
        in_chunks::<u16, N, 8, 4>(
            lanes,
            |start| {
                Some(
                    packed::round_f32x8(chunk(x, start).map(widen), rounding)?.map(narrow_integral),
                )
            },
            |start| {
                Some(
                    packed::round_f32x4(chunk(x, start).map(widen), rounding)?.map(narrow_integral),
                )
            },
            |index| {
                Some(narrow_integral(environment::round_f32(
                    widen(x[index]),
                    rounding,
                )?))
            },
        )
    })
}

/// Returns each bfloat16 lane widened exactly to binary32.
#[inline]
pub(super) fn to_singles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[f32; N]> {
    let x = encodings_u16(value)?;
    let mut lanes = [0.0; N];
    lanes
        .iter_mut()
        .zip(x)
        .for_each(|(lane, &bits)| *lane = widen(bits));
    Some(lanes)
}
