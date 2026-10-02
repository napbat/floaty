//! The packed paths of binary16 lanes, which compute in binary32, and in
//! binary16 with `FEAT_FP16`.
//!
//! Each chunk widens its lanes to binary32 exactly, computes in the packed
//! binary32 instructions, and rounds the binary32 results to binary16 to
//! nearest even. Each lane gives the bits of the scalar binary16 path, which
//! runs the same instructions on one lane: binary32 holds 2p + 2 bits of
//! binary16, so the two roundings of add, subtract, multiply, divide, and
//! square root give the correctly rounded result, and an integral binary16
//! value narrows exactly. The lanes after the last chunk take the scalar
//! path.
//!
//! With `FEAT_FP16`, each chunk of eight lanes computes in the binary16
//! instructions, which round each lane once. The fused multiply-add takes
//! the packed path only then.

use core::cmp::Ordering;

use super::super::Operation;
use super::super::bits::{min_max_differs_16, nan_16};
use super::super::environment::{self, packed};
use super::super::paths::min_max_f32;
use super::{any_lane, chunk, encodings_u16, in_chunks, in_chunks_where, lanes_u16, singles};
use crate::env::{Mode, Rounding};
use crate::float::Float;
use crate::format::Standard;
use crate::format::internal::MinMax;

/// Computes binary16 lanes as `in_chunks` does. A chunk of eight lanes runs
/// `eight` in a build with `FEAT_FP16` or with wide registers, and a chunk of
/// four lanes runs `four`.
#[inline]
fn in_half_chunks<T: Copy, const N: usize>(
    lanes: &mut [T; N],
    eight: impl Fn(usize) -> Option<[T; 8]>,
    four: impl Fn(usize) -> Option<[T; 4]>,
    lane: impl Fn(usize) -> Option<T>,
) -> Option<()> {
    let eight = (packed::NATIVE_HALF || packed::WIDE).then_some(eight);
    in_chunks_where(lanes, eight, four, lane)
}

/// Returns `operation` of each pair of binary16 lanes, or `None` when the
/// build has no instruction or a lane is a NaN.
#[inline]
pub(super) fn binary<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: Operation,
) -> Option<[Float<S, W, M>; N]> {
    let (x, y) = (encodings_u16(left)?, encodings_u16(right)?);
    lanes_u16(left, nan_16, |lanes| {
        in_half_chunks(
            lanes,
            |start| {
                if packed::NATIVE_HALF {
                    return packed::binary_f16x8(*chunk(x, start), *chunk(y, start), operation);
                }
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
                let a = environment::widen_half(x[index]);
                let b = environment::widen_half(y[index]);
                Some(environment::narrow_half(environment::binary_f32(
                    a, b, operation,
                )))
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
    let x = encodings_u16(value)?;
    lanes_u16(value, nan_16, |lanes| {
        in_half_chunks(
            lanes,
            |start| {
                if packed::NATIVE_HALF {
                    return packed::sqrt_f16x8(*chunk(x, start));
                }
                let a = packed::widen_halves_x8(*chunk(x, start))?;
                packed::narrow_halves_x8(packed::sqrt_f32x8(a)?)
            },
            |start| {
                let a = packed::widen_halves_x4(*chunk(x, start))?;
                packed::narrow_halves_x4(packed::sqrt_f32x4(a))
            },
            |index| {
                let a = environment::widen_half(x[index]);
                Some(environment::narrow_half(environment::sqrt_f32(a)))
            },
        )
    })
}

/// Returns each binary16 lane rounded to an integral value in the direction
/// `rounding`, or `None` when the build has no instruction or a lane is a
/// NaN. The integral value in each direction is a binary16 value, so the
/// narrowing is exact.
#[inline]
pub(super) fn round_to_integral<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    rounding: Rounding,
) -> Option<[Float<S, W, M>; N]> {
    let x = encodings_u16(value)?;
    lanes_u16(value, nan_16, |lanes| {
        in_half_chunks(
            lanes,
            |start| {
                if packed::NATIVE_HALF {
                    return packed::round_f16x8(*chunk(x, start), rounding);
                }
                let a = packed::widen_halves_x8(*chunk(x, start))?;
                packed::narrow_halves_x8(packed::round_f32x8(a, rounding)?)
            },
            |start| {
                let a = packed::widen_halves_x4(*chunk(x, start))?;
                packed::narrow_halves_x4(packed::round_f32x4(a, rounding)?)
            },
            |index| {
                let a = environment::widen_half(x[index]);
                Some(environment::narrow_half(environment::round_f32(
                    a, rounding,
                )?))
            },
        )
    })
}

/// Returns `left * right + addend` of each triple of binary16 lanes, each
/// rounded once, or `None` when the build has no instruction or a lane is a
/// NaN. A chunk of eight lanes runs `FMLA` on binary16 lanes, and every
/// other lane runs the scalar fused multiply-add of the host paths. The
/// packed paths take this function only with `FEAT_FP16`.
#[inline]
pub(super) fn mul_add<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    addend: &[Float<S, W, M>; N],
) -> Option<[Float<S, W, M>; N]> {
    let (x, y, z) = (
        encodings_u16(left)?,
        encodings_u16(right)?,
        encodings_u16(addend)?,
    );
    let fused = |index: usize| environment::mul_add_f16(x[index], y[index], z[index]);
    lanes_u16(left, nan_16, |lanes| {
        in_half_chunks(
            lanes,
            |start| packed::mul_add_f16x8(*chunk(x, start), *chunk(y, start), *chunk(z, start)),
            |start| Some(core::array::from_fn(|lane| fused(start + lane))),
            |index| Some(fused(index)),
        )
    })
}

/// Writes the order of each pair of binary16 lanes to `orders`, or returns
/// `None` when the build has no instruction. A chunk of eight lanes compares
/// in the binary16 instructions, and every other lane compares widened
/// exactly to binary32. The packed paths take this function only with
/// `FEAT_FP16`.
#[inline]
pub(super) fn compare<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    orders: &mut [Option<Ordering>; N],
) -> Option<()> {
    let (x, y) = (encodings_u16(left)?, encodings_u16(right)?);
    in_half_chunks(
        orders,
        |start| Some(packed::compare_f16x8(*chunk(x, start), *chunk(y, start))?.orders()),
        |start| {
            let a = packed::widen_halves_x4(*chunk(x, start))?;
            let b = packed::widen_halves_x4(*chunk(y, start))?;
            Some(packed::compare_f32x4(a, b).orders())
        },
        |index| {
            let a = environment::widen_half(x[index]);
            let b = environment::widen_half(y[index]);
            Some(environment::compare_f32(a, b))
        },
    )
}

/// Returns the minimum or maximum operation `operation` of each pair of
/// binary16 lanes, or `None` when the build has no instruction, or when a
/// pair holds a NaN or two zeros. A chunk of eight lanes runs `FMIN` or
/// `FMAX` on binary16 lanes. Every other lane runs on the lanes widened
/// exactly to binary32, and its result is an operand, so it narrows exactly.
/// The packed paths take this function only with `FEAT_FP16`.
#[inline]
pub(super) fn min_max<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: MinMax,
) -> Option<[Float<S, W, M>; N]> {
    let (x, y) = (encodings_u16(left)?, encodings_u16(right)?);
    if any_lane(x.iter().zip(y), |(&a, &b)| min_max_differs_16(a, b)) {
        return None;
    }
    lanes_u16(left, nan_16, |lanes| {
        in_half_chunks(
            lanes,
            |start| packed::min_max_f16x8(*chunk(x, start), *chunk(y, start), operation),
            |start| {
                let a = packed::widen_halves_x4(*chunk(x, start))?;
                let b = packed::widen_halves_x4(*chunk(y, start))?;
                packed::narrow_halves_x4(packed::min_max_f32x4(a, b, operation))
            },
            |index| {
                let a = environment::widen_half(x[index]);
                let b = environment::widen_half(y[index]);
                Some(environment::narrow_half(min_max_f32(a, b, operation)))
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
    let x = encodings_u16(value)?;
    let mut lanes = [0.0; N];
    in_chunks::<f32, N, 8, 4>(
        &mut lanes,
        |start| packed::widen_halves_x8(*chunk(x, start)),
        |start| packed::widen_halves_x4(*chunk(x, start)),
        |index| Some(environment::widen_half(x[index])),
    )?;
    Some(lanes)
}

/// Returns each binary16 lane widened exactly to binary64, or `None` when
/// the build has no instruction.
#[inline]
pub(super) fn to_doubles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[f64; N]> {
    let x = encodings_u16(value)?;
    let mut lanes = [0.0; N];
    in_chunks::<f64, N, 4, 2>(
        &mut lanes,
        |start| packed::widen_x4(packed::widen_halves_x4(*chunk(x, start))?),
        |start| {
            let [a0, a1] = *chunk(x, start);
            let [b0, b1, _, _] = packed::widen_halves_x4([a0, a1, 0, 0])?;
            Some(packed::widen_x2([b0, b1]))
        },
        |index| Some(environment::widen_single(environment::widen_half(x[index]))),
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
        |index| Some(environment::narrow_half(x[index])),
    )?;
    Some(lanes)
}
