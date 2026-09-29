//! The packed host paths of `Lanes`, the same on every architecture that has
//! them.
//!
//! Each entry point checks the mode and the environment once for all lanes.
//! It computes full chunks of lanes with packed instructions: wide chunks
//! where the build has wide registers, then narrow chunks, and then the other
//! lanes with the scalar instructions of the host paths. It returns `None`
//! when the path does not apply, and when a lane is a NaN. The caller then
//! computes each lane by its scalar operation, which sends a NaN to the
//! engine. The result of the path then stays in registers. A NaN path that
//! read the lanes of the result kept them in memory.
//!
//! The instructions read the lanes where they are: `Float` has the layout of
//! its encoding, so an array of binary32 values is an array of `f32`. A copy
//! of the lanes in smaller parts made each load of a chunk wait for the
//! stores, which took more time than the arithmetic.

use super::Operation;
use super::environment::{self, packed};
use super::paths::{nan_32, nan_64, precision_of, ready_for};
use crate::env::{Env, Mode};
use crate::float::Float;
use crate::format::Standard;
use crate::format::internal::Host;

/// Returns `true` when the packed paths compute lanes of the host kind
/// `host`. The answer is a constant.
#[must_use]
pub const fn lanes_of(host: Host) -> bool {
    matches!(host, Host::Single | Host::Double)
}

/// Returns `true` when the packed paths convert lanes from the host kind
/// `from` to the host kind `to`. The answer is a constant.
#[must_use]
pub const fn converts(from: Host, to: Host) -> bool {
    matches!(
        (from, to),
        (Host::Single, Host::Double) | (Host::Double, Host::Single)
    )
}

/// Returns `true` when a value of the float type has the size and at least
/// the alignment of `T`. The test is a constant.
#[inline]
const fn fits<S: Standard<W>, const W: usize, M: Mode, T>() -> bool {
    size_of::<Float<S, W, M>>() == size_of::<T>() && align_of::<Float<S, W, M>>() >= align_of::<T>()
}

/// Returns an array of values of a format with 32-bit storage as an array
/// of `f32`, or `None` for another format.
#[inline]
fn singles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    lanes: &[Float<S, W, M>; N],
) -> Option<&[f32; N]> {
    if !fits::<S, W, M, f32>() {
        return None;
    }
    // SAFETY: `Float` is `repr(transparent)` over its encoding, which has the
    // size and at least the alignment of an `f32`, as `fits` shows. Every bit
    // pattern is an `f32`.
    Some(unsafe { &*lanes.as_ptr().cast::<[f32; N]>() })
}

/// Returns an array of values of a format with 32-bit storage as a mutable
/// array of `f32`, or `None` for another format.
#[inline]
fn singles_mut<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    lanes: &mut [Float<S, W, M>; N],
) -> Option<&mut [f32; N]> {
    if !fits::<S, W, M, f32>() {
        return None;
    }
    // SAFETY: as in `singles`. Every `f32` is also an encoding of the format.
    Some(unsafe { &mut *lanes.as_mut_ptr().cast::<[f32; N]>() })
}

/// Returns an array of values of a format with 64-bit storage as an array
/// of `f64`, or `None` for another format.
#[inline]
fn doubles<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    lanes: &[Float<S, W, M>; N],
) -> Option<&[f64; N]> {
    if !fits::<S, W, M, f64>() {
        return None;
    }
    // SAFETY: as in `singles`, with an `f64`.
    Some(unsafe { &*lanes.as_ptr().cast::<[f64; N]>() })
}

/// Returns an array of values of a format with 64-bit storage as a mutable
/// array of `f64`, or `None` for another format.
#[inline]
fn doubles_mut<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    lanes: &mut [Float<S, W, M>; N],
) -> Option<&mut [f64; N]> {
    if !fits::<S, W, M, f64>() {
        return None;
    }
    // SAFETY: as in `singles_mut`, with an `f64`.
    Some(unsafe { &mut *lanes.as_mut_ptr().cast::<[f64; N]>() })
}

/// Returns the chunk of `C` lanes of `values` from lane `start`.
#[inline]
fn chunk<T, const N: usize, const C: usize>(values: &[T; N], start: usize) -> &[T; C] {
    values[start..start + C]
        .try_into()
        .expect("the caller asks only for a chunk inside the lanes")
}

/// Returns the mutable chunk of `C` lanes of `values` from lane `start`.
#[inline]
fn chunk_mut<T, const N: usize, const C: usize>(values: &mut [T; N], start: usize) -> &mut [T; C] {
    (&mut values[start..start + C])
        .try_into()
        .expect("the caller asks only for a chunk inside the lanes")
}

/// Computes the `N` lanes of `lanes`: `WIDE` lanes at a time where the build
/// has wide registers, then `NARROW` lanes at a time, and then one lane at a
/// time. Each function takes the index of its first lane. Returns `None`
/// when a function has no instruction.
#[inline]
fn in_chunks<T: Copy, const N: usize, const WIDE: usize, const NARROW: usize>(
    lanes: &mut [T; N],
    wide: impl Fn(usize) -> Option<[T; WIDE]>,
    narrow: impl Fn(usize) -> Option<[T; NARROW]>,
    lane: impl Fn(usize) -> Option<T>,
) -> Option<()> {
    let wide_end = if packed::WIDE { N - N % WIDE } else { 0 };
    let narrow_end = N - (N - wide_end) % NARROW;
    for start in (0..wide_end).step_by(WIDE) {
        *chunk_mut(lanes, start) = wide(start)?;
    }
    for start in (wide_end..narrow_end).step_by(NARROW) {
        *chunk_mut(lanes, start) = narrow(start)?;
    }
    for (index, value) in lanes.iter_mut().enumerate().skip(narrow_end) {
        *value = lane(index)?;
    }
    Some(())
}

/// Computes binary32 lanes into a copy of `first` with `compute`, which
/// writes the `f32` lanes. Returns `None` when a lane is a NaN.
#[inline]
fn single_lanes<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    first: &[Float<S, W, M>; N],
    compute: impl FnOnce(&mut [f32; N]) -> Option<()>,
) -> Option<[Float<S, W, M>; N]> {
    let mut lanes = *first;
    let values = singles_mut(&mut lanes)?;
    compute(values)?;
    // A fold without a branch lets LLVM test all lanes at once.
    let nan = values
        .iter()
        .fold(false, |nan, value| nan | nan_32(value.to_bits()));
    (!nan).then_some(lanes)
}

/// Computes binary64 lanes into a copy of `first` with `compute`, which
/// writes the `f64` lanes. Returns `None` when a lane is a NaN.
#[inline]
fn double_lanes<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    first: &[Float<S, W, M>; N],
    compute: impl FnOnce(&mut [f64; N]) -> Option<()>,
) -> Option<[Float<S, W, M>; N]> {
    let mut lanes = *first;
    let values = doubles_mut(&mut lanes)?;
    compute(values)?;
    // A fold without a branch lets LLVM test all lanes at once.
    let nan = values
        .iter()
        .fold(false, |nan, value| nan | nan_64(value.to_bits()));
    (!nan).then_some(lanes)
}

/// Returns `operation` of each pair of lanes from the host unit.
#[inline]
pub fn binary<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: Operation,
    env: &Env,
) -> Option<[Float<S, W, M>; N]> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let (x, y) = (singles(left)?, singles(right)?);
            single_lanes(left, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start| packed::binary_f32x8(*chunk(x, start), *chunk(y, start), operation),
                    |start| {
                        Some(packed::binary_f32x4(
                            *chunk(x, start),
                            *chunk(y, start),
                            operation,
                        ))
                    },
                    |index| Some(environment::binary_f32(x[index], y[index], operation)),
                )
            })
        }
        Host::Double => {
            let (x, y) = (doubles(left)?, doubles(right)?);
            double_lanes(left, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start| packed::binary_f64x4(*chunk(x, start), *chunk(y, start), operation),
                    |start| {
                        Some(packed::binary_f64x2(
                            *chunk(x, start),
                            *chunk(y, start),
                            operation,
                        ))
                    },
                    |index| Some(environment::binary_f64(x[index], y[index], operation)),
                )
            })
        }
        Host::None | Host::Half | Host::BFloat | Host::Extended => None,
    }
}

/// Returns the square root of each lane from the host unit.
#[inline]
pub fn sqrt<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    env: &Env,
) -> Option<[Float<S, W, M>; N]> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let x = singles(value)?;
            single_lanes(value, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start| packed::sqrt_f32x8(*chunk(x, start)),
                    |start| Some(packed::sqrt_f32x4(*chunk(x, start))),
                    |index| Some(environment::sqrt_f32(x[index])),
                )
            })
        }
        Host::Double => {
            let x = doubles(value)?;
            double_lanes(value, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start| packed::sqrt_f64x4(*chunk(x, start)),
                    |start| Some(packed::sqrt_f64x2(*chunk(x, start))),
                    |index| Some(environment::sqrt_f64(x[index])),
                )
            })
        }
        Host::None | Host::Half | Host::BFloat | Host::Extended => None,
    }
}

/// Returns each lane rounded to an integral value to nearest even from the
/// host unit.
#[inline]
pub fn round_to_integral<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    env: &Env,
) -> Option<[Float<S, W, M>; N]> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let x = singles(value)?;
            single_lanes(value, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start| packed::round_f32x8(*chunk(x, start)),
                    |start| packed::round_f32x4(*chunk(x, start)),
                    |index| environment::round_f32(x[index]),
                )
            })
        }
        Host::Double => {
            let x = doubles(value)?;
            double_lanes(value, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start| packed::round_f64x4(*chunk(x, start)),
                    |start| packed::round_f64x2(*chunk(x, start)),
                    |index| environment::round_f64(x[index]),
                )
            })
        }
        Host::None | Host::Half | Host::BFloat | Host::Extended => None,
    }
}

/// Returns `left * right + addend` of each triple of lanes, each rounded
/// once, from the host unit.
#[inline]
pub fn mul_add<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    addend: &[Float<S, W, M>; N],
    env: &Env,
) -> Option<[Float<S, W, M>; N]> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let (x, y, z) = (singles(left)?, singles(right)?, singles(addend)?);
            single_lanes(left, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start| {
                        packed::mul_add_f32x8(*chunk(x, start), *chunk(y, start), *chunk(z, start))
                    },
                    |start| {
                        packed::mul_add_f32x4(*chunk(x, start), *chunk(y, start), *chunk(z, start))
                    },
                    |index| environment::mul_add_f32(x[index], y[index], z[index]),
                )
            })
        }
        Host::Double => {
            let (x, y, z) = (doubles(left)?, doubles(right)?, doubles(addend)?);
            double_lanes(left, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start| {
                        packed::mul_add_f64x4(*chunk(x, start), *chunk(y, start), *chunk(z, start))
                    },
                    |start| {
                        packed::mul_add_f64x2(*chunk(x, start), *chunk(y, start), *chunk(z, start))
                    },
                    |index| environment::mul_add_f64(x[index], y[index], z[index]),
                )
            })
        }
        Host::None | Host::Half | Host::BFloat | Host::Extended => None,
    }
}

/// Returns the encoding of each lane converted to the host kind `to` by the
/// host unit. The packed paths convert between binary32 and binary64. A
/// conversion gives a NaN only for a NaN lane, so the result is `None` for
/// a NaN lane.
#[inline]
pub fn convert<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    to: Host,
    env: &Env,
) -> Option<[u64; N]> {
    if !ready_for(to, env, precision_of(to)) {
        return None;
    }
    match (S::HOST, to) {
        (Host::Single, Host::Double) => {
            let x = singles(value)?;
            let mut lanes = [0.0; N];
            in_chunks::<f64, N, 4, 2>(
                &mut lanes,
                |start| packed::widen_x4(*chunk(x, start)),
                |start| Some(packed::widen_x2(*chunk(x, start))),
                |index| Some(environment::widen_single(x[index])),
            )?;
            let lanes = lanes.map(f64::to_bits);
            let nan = lanes.iter().fold(false, |nan, &bits| nan | nan_64(bits));
            (!nan).then_some(lanes)
        }
        (Host::Double, Host::Single) => {
            let x = doubles(value)?;
            let mut lanes = [0.0; N];
            in_chunks::<f32, N, 4, 2>(
                &mut lanes,
                |start| packed::narrow_x4(*chunk(x, start)),
                |start| Some(packed::narrow_x2(*chunk(x, start))),
                |index| Some(environment::narrow_double(x[index])),
            )?;
            let bits = lanes.map(f32::to_bits);
            let nan = bits.iter().fold(false, |nan, &bits| nan | nan_32(bits));
            (!nan).then_some(bits.map(u64::from))
        }
        _ => None,
    }
}
