//! The packed host paths of `Lanes`, the same on every architecture that has
//! them.
//!
//! Each entry point checks the mode and the environment once for all lanes.
//! It computes full chunks of lanes with packed instructions: wide chunks
//! where the build has wide registers, then narrow chunks, and then the other
//! lanes with the scalar instructions of the host paths. It returns `None`
//! when the path does not apply. A NaN lane keeps the NaN of the host, and
//! the result says that a lane is a NaN, so the caller computes each NaN lane
//! in the engine.
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

/// The lanes that a packed host path computes.
#[derive(Clone, Copy, Debug)]
pub struct Packed<L, const N: usize> {
    /// The lanes.
    pub lanes: [L; N],
    /// `true` when a lane is a NaN, which the engine must compute again.
    pub nan: bool,
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
/// time. Each chunk function takes the index of its first lane and the chunk
/// to write. Returns `None` when a function has no instruction.
#[inline]
fn in_chunks<T: Copy, const N: usize, const WIDE: usize, const NARROW: usize>(
    lanes: &mut [T; N],
    wide: impl Fn(usize, &mut [T; WIDE]) -> Option<()>,
    narrow: impl Fn(usize, &mut [T; NARROW]) -> Option<()>,
    lane: impl Fn(usize) -> Option<T>,
) -> Option<()> {
    let wide_end = if packed::WIDE { N - N % WIDE } else { 0 };
    let narrow_end = N - (N - wide_end) % NARROW;
    for start in (0..wide_end).step_by(WIDE) {
        wide(start, chunk_mut(lanes, start))?;
    }
    for start in (wide_end..narrow_end).step_by(NARROW) {
        narrow(start, chunk_mut(lanes, start))?;
    }
    for (index, value) in lanes.iter_mut().enumerate().skip(narrow_end) {
        *value = lane(index)?;
    }
    Some(())
}

/// Computes binary32 lanes into a copy of `first` with `compute`, which
/// writes the `f32` lanes, and returns them with a NaN test.
#[inline]
fn single_lanes<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    first: &[Float<S, W, M>; N],
    compute: impl FnOnce(&mut [f32; N]) -> Option<()>,
) -> Option<Packed<Float<S, W, M>, N>> {
    let mut lanes = *first;
    let values = singles_mut(&mut lanes)?;
    compute(values)?;
    // A fold without a branch lets LLVM test all lanes at once.
    let nan = values
        .iter()
        .fold(false, |nan, value| nan | nan_32(value.to_bits()));
    Some(Packed { lanes, nan })
}

/// Computes binary64 lanes into a copy of `first` with `compute`, which
/// writes the `f64` lanes, and returns them with a NaN test.
#[inline]
fn double_lanes<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    first: &[Float<S, W, M>; N],
    compute: impl FnOnce(&mut [f64; N]) -> Option<()>,
) -> Option<Packed<Float<S, W, M>, N>> {
    let mut lanes = *first;
    let values = doubles_mut(&mut lanes)?;
    compute(values)?;
    // A fold without a branch lets LLVM test all lanes at once.
    let nan = values
        .iter()
        .fold(false, |nan, value| nan | nan_64(value.to_bits()));
    Some(Packed { lanes, nan })
}

/// Returns `operation` of each pair of lanes from the host unit.
#[inline]
pub fn binary<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: Operation,
    env: &Env,
) -> Option<Packed<Float<S, W, M>, N>> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let (x, y) = (singles(left)?, singles(right)?);
            single_lanes(left, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start, out| {
                        packed::binary_f32(chunk(x, start), chunk(y, start), out, operation)
                    },
                    |start, out| {
                        packed::binary_f32(chunk(x, start), chunk(y, start), out, operation)
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
                    |start, out| {
                        packed::binary_f64(chunk(x, start), chunk(y, start), out, operation)
                    },
                    |start, out| {
                        packed::binary_f64(chunk(x, start), chunk(y, start), out, operation)
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
) -> Option<Packed<Float<S, W, M>, N>> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let x = singles(value)?;
            single_lanes(value, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start, out| packed::sqrt_f32(chunk(x, start), out),
                    |start, out| packed::sqrt_f32(chunk(x, start), out),
                    |index| Some(environment::sqrt_f32(x[index])),
                )
            })
        }
        Host::Double => {
            let x = doubles(value)?;
            double_lanes(value, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start, out| packed::sqrt_f64(chunk(x, start), out),
                    |start, out| packed::sqrt_f64(chunk(x, start), out),
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
) -> Option<Packed<Float<S, W, M>, N>> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let x = singles(value)?;
            single_lanes(value, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start, out| packed::round_f32(chunk(x, start), out),
                    |start, out| packed::round_f32(chunk(x, start), out),
                    |index| environment::round_f32(x[index]),
                )
            })
        }
        Host::Double => {
            let x = doubles(value)?;
            double_lanes(value, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start, out| packed::round_f64(chunk(x, start), out),
                    |start, out| packed::round_f64(chunk(x, start), out),
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
) -> Option<Packed<Float<S, W, M>, N>> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::Single => {
            let (x, y, z) = (singles(left)?, singles(right)?, singles(addend)?);
            single_lanes(left, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start, out| {
                        packed::mul_add_f32(chunk(x, start), chunk(y, start), chunk(z, start), out)
                    },
                    |start, out| {
                        packed::mul_add_f32(chunk(x, start), chunk(y, start), chunk(z, start), out)
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
                    |start, out| {
                        packed::mul_add_f64(chunk(x, start), chunk(y, start), chunk(z, start), out)
                    },
                    |start, out| {
                        packed::mul_add_f64(chunk(x, start), chunk(y, start), chunk(z, start), out)
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
/// conversion gives a NaN only for a NaN lane.
#[inline]
pub fn convert<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    to: Host,
    env: &Env,
) -> Option<Packed<u64, N>> {
    if !ready_for(to, env, precision_of(to)) {
        return None;
    }
    match (S::HOST, to) {
        (Host::Single, Host::Double) => {
            let x = singles(value)?;
            let mut lanes = [0.0; N];
            in_chunks::<f64, N, 4, 2>(
                &mut lanes,
                |start, out| packed::widen(chunk(x, start), out),
                |start, out| packed::widen(chunk(x, start), out),
                |index| Some(environment::widen_single(x[index])),
            )?;
            let lanes = lanes.map(f64::to_bits);
            let nan = lanes.iter().any(|&bits| nan_64(bits));
            Some(Packed { lanes, nan })
        }
        (Host::Double, Host::Single) => {
            let x = doubles(value)?;
            let mut lanes = [0.0; N];
            in_chunks::<f32, N, 4, 2>(
                &mut lanes,
                |start, out| packed::narrow(chunk(x, start), out),
                |start, out| packed::narrow(chunk(x, start), out),
                |index| Some(environment::narrow_double(x[index])),
            )?;
            let bits = lanes.map(f32::to_bits);
            let nan = bits.iter().any(|&bits| nan_32(bits));
            Some(Packed {
                lanes: bits.map(u64::from),
                nan,
            })
        }
        _ => None,
    }
}
