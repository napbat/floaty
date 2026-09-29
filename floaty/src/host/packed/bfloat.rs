//! The packed paths of bfloat16 lanes, which a shift widens to binary32.
//!
//! A bfloat16 encoding is the high half of a binary32 encoding, so a shift
//! widens each lane exactly, with no floating-point instruction. The
//! integral value of a bfloat16 value in each direction is a bfloat16
//! value, so the low 16 bits of the binary32 integral value are zero, and a
//! shift narrows it exactly. The arithmetic of bfloat16 lanes takes the
//! scalar path of each lane: no x86-64 instruction below `AVX512_BF16` rounds
//! binary32 to bfloat16, and `VCVTNEPS2BF16` reads a subnormal input as
//! zero. The AArch64 build with `FEAT_BF16` has `BFCVTN`, which no packed
//! path uses yet.

use super::super::environment::{self, packed};
use super::super::paths::nan_bfloat;
use super::{chunk, encodings_u16, in_chunks, lanes_u16};
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

/// Returns each bfloat16 lane widened exactly to binary64, or `None` when
/// the build has no instruction.
#[inline]
pub(super) fn to_doubles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[f64; N]> {
    let singles = to_singles(value)?;
    let mut lanes = [0.0; N];
    in_chunks::<f64, N, 4, 2>(
        &mut lanes,
        |start| packed::widen_x4(*chunk(&singles, start)),
        |start| Some(packed::widen_x2(*chunk(&singles, start))),
        |index| Some(environment::widen_single(singles[index])),
    )?;
    Some(lanes)
}
