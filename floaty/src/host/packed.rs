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

mod bfloat;
mod extended;
mod half;
mod order;

use core::cmp::Ordering;

use super::bits::{nan_16, nan_32, nan_64, nan_bfloat};
use super::environment::{self, packed};
use super::paths::{ready_for, ready_for_integral};
use super::{Kind, Operation};
use crate::env::{Env, Mode};
use crate::float::Float;
use crate::format::Standard;
use crate::format::internal::MinMax;
use crate::host::Host;

/// Returns `true` when the packed paths compute the operations of `kind` on
/// lanes of the host kind `host`: the build has the scalar path, and the
/// packed paths cover the lanes. binary16 lanes compute in binary32 where
/// the build widens and narrows them, and bfloat16 lanes compute in binary32
/// in every build. Neither has the fused multiply-add: two roundings of it
/// through binary32 can differ from one. x87 extended lanes run the scalar
/// instructions after one check of the control word. The answer is a
/// constant.
#[must_use]
pub const fn available(host: Host, kind: Kind) -> bool {
    if !super::available(host, kind) {
        return false;
    }
    match (host, kind) {
        (
            Host::Single | Host::Double,
            Kind::Arithmetic
            | Kind::SquareRoot
            | Kind::FusedMultiplyAdd
            | Kind::RoundToIntegral
            | Kind::Comparison,
        )
        | (
            Host::BFloat,
            Kind::Arithmetic | Kind::SquareRoot | Kind::RoundToIntegral | Kind::Comparison,
        )
        | (Host::Extended, Kind::Arithmetic | Kind::SquareRoot) => true,
        (Host::Single | Host::Double, Kind::ToInt) => packed::INTEGERS,
        (
            Host::Half,
            Kind::Arithmetic | Kind::SquareRoot | Kind::RoundToIntegral | Kind::Comparison,
        ) => packed::HALF,
        _ => false,
    }
}

/// Returns `true` when the packed paths convert lanes from the host kind
/// `from` to the host kind `to`: the build has the scalar conversion, and the
/// packed paths cover the lanes. The answer is a constant. binary64 lanes do
/// not round to binary16 or bfloat16: two roundings through binary32 can
/// differ from one.
#[must_use]
pub const fn convertible(from: Host, to: Host) -> bool {
    if !super::convertible(from, to) {
        return false;
    }
    match (from, to) {
        (Host::Single, Host::Double | Host::BFloat)
        | (Host::Double, Host::Single)
        | (Host::BFloat, Host::Single | Host::Double) => true,
        (Host::Half, Host::Single | Host::Double) | (Host::Single, Host::Half) => packed::HALF,
        _ => false,
    }
}

/// Returns `true` when `test` holds for any item. The fold has no branch,
/// so LLVM tests all lanes at once.
#[inline]
fn any_lane<T>(items: impl IntoIterator<Item = T>, test: impl Fn(T) -> bool) -> bool {
    items
        .into_iter()
        .fold(false, |found, item| found | test(item))
}

/// The masks of a packed comparison. A lane of `less` is not zero when the
/// left lane is less, a lane of `greater` when the left lane is greater, and
/// a lane of `unordered` when the pair is unordered.
#[derive(Clone, Copy, Debug)]
pub struct Masks<T, const N: usize> {
    /// The lanes where the left lane is less.
    pub less: [T; N],
    /// The lanes where the left lane is greater.
    pub greater: [T; N],
    /// The lanes where the pair is unordered.
    pub unordered: [T; N],
}

impl<T: Copy + Default + PartialEq, const N: usize> Masks<T, N> {
    /// Returns the order of each pair of lanes: `None` for an unordered
    /// pair.
    #[must_use]
    #[inline]
    pub fn orders(self) -> [Option<Ordering>; N] {
        let zero = T::default();
        let mut orders = [None; N];
        orders
            .iter_mut()
            .zip(self.less.iter().zip(&self.greater).zip(&self.unordered))
            .for_each(|(order, ((&less, &greater), &unordered))| {
                // A clear `less` orders after a set one, so the comparison of
                // the two tests gives the order without a branch.
                *order = (unordered == zero).then(|| (less == zero).cmp(&(greater == zero)));
            });
        orders
    }
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

/// Returns an array of values of a format with 16-bit storage as an array
/// of `u16` encodings, or `None` for another format.
#[inline]
fn encodings_u16<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    lanes: &[Float<S, W, M>; N],
) -> Option<&[u16; N]> {
    if !fits::<S, W, M, u16>() {
        return None;
    }
    // SAFETY: as in `singles`, with a `u16`.
    Some(unsafe { &*lanes.as_ptr().cast::<[u16; N]>() })
}

/// Returns an array of values of a format with 16-bit storage as a mutable
/// array of `u16` encodings, or `None` for another format.
#[inline]
fn encodings_u16_mut<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    lanes: &mut [Float<S, W, M>; N],
) -> Option<&mut [u16; N]> {
    if !fits::<S, W, M, u16>() {
        return None;
    }
    // SAFETY: as in `singles_mut`, with a `u16`.
    Some(unsafe { &mut *lanes.as_mut_ptr().cast::<[u16; N]>() })
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
    let nan = any_lane(values.iter().map(|value| value.to_bits()), nan_32);
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
    let nan = any_lane(values.iter().map(|value| value.to_bits()), nan_64);
    (!nan).then_some(lanes)
}

/// Computes lanes of a format with 16-bit storage into a copy of `first`
/// with `compute`, which writes the encodings of the lanes. Returns `None`
/// when a lane is a NaN, which `is_nan` tests on an encoding.
#[inline]
fn lanes_u16<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    first: &[Float<S, W, M>; N],
    is_nan: impl Fn(u16) -> bool,
    compute: impl FnOnce(&mut [u16; N]) -> Option<()>,
) -> Option<[Float<S, W, M>; N]> {
    let mut lanes = *first;
    let values = encodings_u16_mut(&mut lanes)?;
    compute(values)?;
    let nan = any_lane(values.iter().copied(), is_nan);
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
        Host::Half => half::binary(left, right, operation),
        Host::BFloat => bfloat::binary(left, right, operation),
        Host::Extended => extended::binary(left, right, operation),
        Host::None => None,
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
        Host::Half => half::sqrt(value),
        Host::BFloat => bfloat::sqrt(value),
        Host::Extended => extended::sqrt(value),
        Host::None => None,
    }
}

/// Returns each lane rounded to an integral value in the rounding direction
/// of `env` from the host unit.
#[inline]
pub fn round_to_integral<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    env: &Env,
) -> Option<[Float<S, W, M>; N]> {
    if !ready_for_integral(S::HOST, env, S::PRECISION) {
        return None;
    }
    let rounding = env.rounding;
    match S::HOST {
        Host::Single => {
            let x = singles(value)?;
            single_lanes(value, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start| packed::round_f32x8(*chunk(x, start), rounding),
                    |start| packed::round_f32x4(*chunk(x, start), rounding),
                    |index| environment::round_f32(x[index], rounding),
                )
            })
        }
        Host::Double => {
            let x = doubles(value)?;
            double_lanes(value, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start| packed::round_f64x4(*chunk(x, start), rounding),
                    |start| packed::round_f64x2(*chunk(x, start), rounding),
                    |index| environment::round_f64(x[index], rounding),
                )
            })
        }
        Host::Half => half::round_to_integral(value, rounding),
        Host::BFloat => bfloat::round_to_integral(value, rounding),
        Host::None | Host::Extended => None,
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
        // Two roundings of a fused multiply-add through binary32 can differ
        // from one.
        Host::None | Host::Half | Host::BFloat | Host::Extended => None,
    }
}

/// Returns the encoding of each lane converted to the host kind `to` by the
/// host unit. The packed paths convert between binary32 and binary64, from
/// binary16 and bfloat16 to both, and from binary32 to binary16 and to
/// bfloat16. A
/// conversion gives a NaN only for a NaN lane, so the result is `None` for
/// a NaN lane.
#[inline]
pub fn convert<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    to: Host,
    env: &Env,
) -> Option<[u64; N]> {
    if !ready_for(to, env, to.precision()) {
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
            let nan = any_lane(lanes.iter().copied(), nan_64);
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
            let nan = any_lane(bits.iter().copied(), nan_32);
            (!nan).then_some(bits.map(u64::from))
        }
        (Host::Half, Host::Single) => {
            let bits = half::to_singles(value)?.map(f32::to_bits);
            let nan = any_lane(bits.iter().copied(), nan_32);
            (!nan).then_some(bits.map(u64::from))
        }
        (Host::Half, Host::Double) => {
            let lanes = half::to_doubles(value)?.map(f64::to_bits);
            let nan = any_lane(lanes.iter().copied(), nan_64);
            (!nan).then_some(lanes)
        }
        (Host::Single, Host::Half) => {
            let bits = half::from_singles(value)?;
            let nan = any_lane(bits.iter().copied(), nan_16);
            (!nan).then_some(bits.map(u64::from))
        }
        (Host::BFloat, Host::Single) => {
            let bits = bfloat::to_singles(value)?.map(f32::to_bits);
            let nan = any_lane(bits.iter().copied(), nan_32);
            (!nan).then_some(bits.map(u64::from))
        }
        (Host::BFloat, Host::Double) => {
            let lanes = bfloat::to_doubles(value)?.map(f64::to_bits);
            let nan = any_lane(lanes.iter().copied(), nan_64);
            (!nan).then_some(lanes)
        }
        (Host::Single, Host::BFloat) => {
            let bits = bfloat::from_singles(value)?;
            let nan = any_lane(bits.iter().copied(), nan_bfloat);
            (!nan).then_some(bits.map(u64::from))
        }
        _ => None,
    }
}

/// Returns the 32-bit integer of a scalar conversion to a 64-bit integer, or
/// the integer indefinite `i32::MIN` when the scalar conversion must decide
/// the lane.
#[inline]
fn indefinite_unless_i32(integer: Option<i64>) -> i32 {
    integer
        .and_then(|integer| i32::try_from(integer).ok())
        .unwrap_or(i32::MIN)
}

/// Returns each lane rounded to a 32-bit integer to nearest even by the host
/// unit, or `None` when the path does not apply. A lane that the scalar
/// conversion must decide is `None`: a NaN, a value outside the range of
/// `i32`, and `-2^31` itself. The host unit gives each of them as the
/// integer indefinite, `i32::MIN`.
#[inline]
pub fn to_int_i32<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
    env: &Env,
) -> Option<[Option<i32>; N]> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    let mut lanes = [0; N];
    match S::HOST {
        Host::Single => {
            let x = singles(value)?;
            in_chunks::<i32, N, 8, 4>(
                &mut lanes,
                |start| packed::to_int_f32x8(*chunk(x, start)),
                |start| Some(packed::to_int_f32x4(*chunk(x, start))),
                |index| Some(indefinite_unless_i32(environment::to_int_f32(x[index]))),
            )?;
        }
        Host::Double => {
            let x = doubles(value)?;
            in_chunks::<i32, N, 4, 2>(
                &mut lanes,
                |start| packed::to_int_f64x4(*chunk(x, start)),
                |start| Some(packed::to_int_f64x2(*chunk(x, start))),
                |index| Some(indefinite_unless_i32(environment::to_int_f64(x[index]))),
            )?;
        }
        Host::None | Host::Half | Host::BFloat | Host::Extended => return None,
    }
    Some(lanes.map(|integer| (integer != i32::MIN).then_some(integer)))
}

/// Returns the order of each pair of lanes from the host unit, as the quiet
/// predicates give it: `None` in a lane for an unordered pair. A comparison
/// gives no NaN, so every lane gives the result of the engine.
#[inline]
pub fn compare<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    env: &Env,
) -> Option<[Option<Ordering>; N]> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    order::compare(left, right)
}

/// Returns the minimum or maximum operation `operation` of each pair of
/// lanes from the host unit, or `None` when a pair holds a NaN or two zeros.
#[inline]
pub fn min_max<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: MinMax,
    env: &Env,
) -> Option<[Float<S, W, M>; N]> {
    // The instructions compare values, not magnitudes.
    if operation.is_magnitude() || !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    order::min_max(left, right, operation)
}
