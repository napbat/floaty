//! The packed paths of binary16 lanes, which compute in binary32.
//!
//! Each chunk widens its lanes to binary32 exactly, computes in the packed
//! binary32 instructions, and rounds the binary32 results to binary16 to
//! nearest even. Each lane gives the bits of the scalar binary16 path, which
//! runs the same instructions on one lane: binary32 holds 2p + 2 bits of
//! binary16, so the two roundings of add, subtract, multiply, divide, and
//! square root give the correctly rounded result, and an integral binary16
//! value narrows exactly. The lanes after the last chunk take the scalar
//! path.

use super::super::Operation;
use super::super::environment::{self, packed};
use super::{chunk, half_lanes, halves, in_chunks, singles};
use crate::env::Mode;
use crate::float::Float;
use crate::format::Standard;

/// Returns `operation` of each pair of binary16 lanes, or `None` when the
/// build has no instruction or a lane is a NaN.
#[inline]
pub(super) fn binary<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: Operation,
) -> Option<[Float<S, W, M>; N]> {
    let (x, y) = (halves(left)?, halves(right)?);
    half_lanes(left, |lanes| {
        in_chunks::<u16, N, 8, 4>(
            lanes,
            |start| {
                let a = packed::widen_halves_x8(*chunk(x, start))?;
                let b = packed::widen_halves_x8(*chunk(y, start))?;
                packed::narrow_halves_x8(packed::binary_f32x8(a, b, operation)?)
            },
            |start| {
                let a = packed::widen_halves_x4(*chunk(x, start))?;
                let b = packed::widen_halves_x4(*chunk(y, start))?;
                packed::narrow_halves_x4(packed::binary_f32x4(a, b, operation))
            },
            |index| {
                let a = environment::widen_half(x[index])?;
                let b = environment::widen_half(y[index])?;
                environment::narrow_half(environment::binary_f32(a, b, operation))
            },
        )
    })
}

/// Returns the square root of each binary16 lane, or `None` when the build
/// has no instruction or a lane is a NaN.
#[inline]
pub(super) fn sqrt<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[Float<S, W, M>; N]> {
    let x = halves(value)?;
    half_lanes(value, |lanes| {
        in_chunks::<u16, N, 8, 4>(
            lanes,
            |start| {
                let a = packed::widen_halves_x8(*chunk(x, start))?;
                packed::narrow_halves_x8(packed::sqrt_f32x8(a)?)
            },
            |start| {
                let a = packed::widen_halves_x4(*chunk(x, start))?;
                packed::narrow_halves_x4(packed::sqrt_f32x4(a))
            },
            |index| {
                let a = environment::widen_half(x[index])?;
                environment::narrow_half(environment::sqrt_f32(a))
            },
        )
    })
}

/// Returns each binary16 lane rounded to an integral value to nearest even,
/// or `None` when the build has no instruction or a lane is a NaN.
#[inline]
pub(super) fn round_to_integral<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[Float<S, W, M>; N]> {
    let x = halves(value)?;
    half_lanes(value, |lanes| {
        in_chunks::<u16, N, 8, 4>(
            lanes,
            |start| {
                let a = packed::widen_halves_x8(*chunk(x, start))?;
                packed::narrow_halves_x8(packed::round_f32x8(a)?)
            },
            |start| {
                let a = packed::widen_halves_x4(*chunk(x, start))?;
                packed::narrow_halves_x4(packed::round_f32x4(a)?)
            },
            |index| {
                let a = environment::widen_half(x[index])?;
                environment::narrow_half(environment::round_f32(a)?)
            },
        )
    })
}

/// Returns each binary16 lane widened exactly to binary32, or `None` when
/// the build has no instruction.
#[inline]
pub(super) fn to_singles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[f32; N]> {
    let x = halves(value)?;
    let mut lanes = [0.0; N];
    in_chunks::<f32, N, 8, 4>(
        &mut lanes,
        |start| packed::widen_halves_x8(*chunk(x, start)),
        |start| packed::widen_halves_x4(*chunk(x, start)),
        |index| environment::widen_half(x[index]),
    )?;
    Some(lanes)
}

/// Returns each binary16 lane widened exactly to binary64, or `None` when
/// the build has no instruction.
#[inline]
pub(super) fn to_doubles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[f64; N]> {
    let x = halves(value)?;
    let mut lanes = [0.0; N];
    in_chunks::<f64, N, 4, 2>(
        &mut lanes,
        |start| packed::widen_x4(packed::widen_halves_x4(*chunk(x, start))?),
        |start| {
            let [a0, a1] = *chunk(x, start);
            let [b0, b1, _, _] = packed::widen_halves_x4([a0, a1, 0, 0])?;
            Some(packed::widen_x2([b0, b1]))
        },
        |index| {
            Some(environment::widen_single(environment::widen_half(
                x[index],
            )?))
        },
    )?;
    Some(lanes)
}

/// Returns the binary16 encoding of each binary32 lane rounded to nearest
/// even, or `None` when the build has no instruction.
#[inline]
pub(super) fn from_singles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[u16; N]> {
    let x = singles(value)?;
    let mut lanes = [0; N];
    in_chunks::<u16, N, 8, 4>(
        &mut lanes,
        |start| packed::narrow_halves_x8(*chunk(x, start)),
        |start| packed::narrow_halves_x4(*chunk(x, start)),
        |index| environment::narrow_half(x[index]),
    )?;
    Some(lanes)
}
