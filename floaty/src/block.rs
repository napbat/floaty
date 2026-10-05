//! Blocks: a chain of binary32 or binary64 steps for each lane of slices,
//! with one check of the environment and one selection of the instruction
//! set for the whole call.
//!
//! A [`Chain`] states the steps of one lane once, generic over [`Steps`], so
//! one chain runs in binary32 and in binary64. [`Float::map`] runs the chain
//! for each lane of slices, and [`Float::evaluate`] runs it for one lane.
//! Each step gives the result of its entry point in the mode of the type:
//! every operation of the format from values of the format to one value of
//! the format. So a block gives the bits of the same steps one at a time, on
//! every host.
//!
//! Where the build has a host path, and the mode and the environment allow
//! it, the steps run as Rust operations on `f32` or `f64` after one check of
//! the environment, in the instruction set that the processor selects. LLVM
//! then sees the whole chain and computes many lanes in vector
//! instructions. A lane whose result is a NaN runs the chain again in the
//! engine, which selects the NaN by the rule of the mode. So does a lane
//! with a step that no instruction of the set computes exactly, such as
//! `exp`, or `mul_add` where the instruction set has no fused multiply-add,
//! and a lane where a step reads a bit of a NaN that can change a result
//! that is not a NaN. Elsewhere every lane runs the steps of the type.
//!
//! ```
//! use floaty::block::{Chain, Steps};
//! use floaty::{F32, F64};
//!
//! /// `(x - mean) * scale + offset`, with the product and the sum rounded
//! /// once.
//! struct Normalize;
//!
//! impl Chain<1, 3> for Normalize {
//!     fn apply<S: Steps>(&self, [x]: [S; 1], [mean, scale, offset]: [S; 3]) -> S {
//!         (x - mean).mul_add(scale, offset)
//!     }
//! }
//!
//! let x = [1.5_f32, 2.5, -0.5];
//! let parameters = [0.5_f32, 2.0, 1.0].map(F32::from);
//! let mut out = [0.0_f32; 3];
//! F32::map(&Normalize, [&x[..]], parameters, &mut out[..]);
//! assert_eq!(out, [3.0, 5.0, -1.0]);
//!
//! let one = F32::evaluate(&Normalize, [F32::from(1.5_f32)], parameters);
//! assert_eq!(one.to_bits(), 3.0_f32.to_bits());
//!
//! // The same chain in binary64.
//! let x = [1.5_f64, 2.5, -0.5];
//! let parameters = [0.5_f64, 2.0, 1.0].map(F64::from);
//! let mut out = [0.0_f64; 3];
//! F64::map(&Normalize, [&x[..]], parameters, &mut out[..]);
//! assert_eq!(out, [3.0, 5.0, -1.0]);
//! ```

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::env::{Mode, Rounding};
use crate::float::Float;
use crate::format::Binary;
use crate::host;
use crate::lanes::Element;
use crate::sealed::Sealed;

/// The binary32 type of the mode `M`.
type Single<M> = Float<Binary<8>, 32, M>;

/// The binary64 type of the mode `M`.
type Double<M> = Float<Binary<11>, 64, M>;

/// The steps of a [`Chain`]: every operation of a binary format from values
/// of the format to one value of the format, each rounded once to nearest
/// even as its entry point rounds. Each step names the method of [`Float`]
/// that gives its result. The binary32 and binary64 types of floaty
/// implement the trait, and so do the lane types of the host path of a
/// block. The trait is sealed.
pub trait Steps:
    Sealed
    + Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Returns `self * multiplier + addend`, rounded once, as
    /// [`Float::mul_add`] gives it.
    #[must_use]
    fn mul_add(self, multiplier: Self, addend: Self) -> Self;

    /// Returns the square root, as [`Float::sqrt`] gives it.
    #[must_use]
    fn sqrt(self) -> Self;

    /// Returns the absolute value, as [`Float::abs`] gives it.
    #[must_use]
    fn abs(self) -> Self;

    /// Returns the value with the sign of `sign`, as [`Float::copy_sign`]
    /// gives it.
    #[must_use]
    fn copy_sign(self, sign: Self) -> Self;

    /// Returns the least value above `self`, as [`Float::next_up`] gives
    /// it.
    #[must_use]
    fn next_up(self) -> Self;

    /// Returns the greatest value below `self`, as [`Float::next_down`]
    /// gives it.
    #[must_use]
    fn next_down(self) -> Self;

    /// Returns the integral value in the direction of the mode, as
    /// [`Float::round_to_integral`] gives it.
    #[must_use]
    fn round_to_integral(self) -> Self;

    /// Returns the integral value in the direction `rounding`, as
    /// [`Float::round_to_integral_with`] gives it.
    #[must_use]
    fn round_to_integral_by(self, rounding: Rounding) -> Self;

    /// Returns the IEEE 754 remainder, as [`Float::remainder`] gives it.
    #[must_use]
    fn remainder(self, divisor: Self) -> Self;

    /// Returns the truncated remainder, as [`Float::truncated_remainder`]
    /// gives it.
    #[must_use]
    fn truncated_remainder(self, divisor: Self) -> Self;

    /// Returns `self * 2^scale`, as [`Float::scale_b`] gives it.
    #[must_use]
    fn scale_b(self, scale: i32) -> Self;

    /// Returns the exponent of the leading bit, as [`Float::log_b`] gives
    /// it.
    #[must_use]
    fn log_b(self) -> Self;

    /// Returns `e^self`, as [`Float::exp`] gives it.
    #[must_use]
    fn exp(self) -> Self;

    /// Returns the natural logarithm, as [`Float::log`] gives it.
    #[must_use]
    fn log(self) -> Self;

    /// Returns `(1 + self)^n`, as [`Float::compound`] gives it.
    #[must_use]
    fn compound(self, n: i64) -> Self;

    /// Returns `sqrt(self^2 + other^2)`, as [`Float::hypot`] gives it.
    #[must_use]
    fn hypot(self, other: Self) -> Self;

    /// Returns `self^n`, as [`Float::pown`] gives it.
    #[must_use]
    fn pown(self, n: i64) -> Self;

    /// Returns `self^(1/n)`, as [`Float::rootn`] gives it.
    #[must_use]
    fn rootn(self, n: i64) -> Self;

    /// Returns `1 / sqrt(self)`, as [`Float::reciprocal_sqrt`] gives it.
    #[must_use]
    fn reciprocal_sqrt(self) -> Self;

    /// Returns the IEEE 754-2019 `minimum`, as [`Float::minimum`] gives it.
    #[must_use]
    fn minimum(self, other: Self) -> Self;

    /// Returns the IEEE 754-2019 `maximum`, as [`Float::maximum`] gives it.
    #[must_use]
    fn maximum(self, other: Self) -> Self;

    /// Returns the IEEE 754-2019 `minimumNumber`, as
    /// [`Float::minimum_number`] gives it.
    #[must_use]
    fn minimum_number(self, other: Self) -> Self;

    /// Returns the IEEE 754-2019 `maximumNumber`, as
    /// [`Float::maximum_number`] gives it.
    #[must_use]
    fn maximum_number(self, other: Self) -> Self;

    /// Returns the IEEE 754-2008 `minNum`, as [`Float::min_num`] gives it.
    #[must_use]
    fn min_num(self, other: Self) -> Self;

    /// Returns the IEEE 754-2008 `maxNum`, as [`Float::max_num`] gives it.
    #[must_use]
    fn max_num(self, other: Self) -> Self;

    /// Returns the IEEE 754-2019 `minimumMagnitude`, as
    /// [`Float::minimum_magnitude`] gives it.
    #[must_use]
    fn minimum_magnitude(self, other: Self) -> Self;

    /// Returns the IEEE 754-2019 `maximumMagnitude`, as
    /// [`Float::maximum_magnitude`] gives it.
    #[must_use]
    fn maximum_magnitude(self, other: Self) -> Self;

    /// Returns the IEEE 754-2019 `minimumMagnitudeNumber`, as
    /// [`Float::minimum_magnitude_number`] gives it.
    #[must_use]
    fn minimum_magnitude_number(self, other: Self) -> Self;

    /// Returns the IEEE 754-2019 `maximumMagnitudeNumber`, as
    /// [`Float::maximum_magnitude_number`] gives it.
    #[must_use]
    fn maximum_magnitude_number(self, other: Self) -> Self;
}

/// Implements each step of [`Steps`] by its entry point, the method of the
/// same name.
macro_rules! entry_points {
    ($($name:ident($($argument:ident: $type:ty),*);)*) => {
        $(
            #[inline]
            fn $name(self, $($argument: $type),*) -> Self {
                Self::$name(self, $($argument),*)
            }
        )*
    };
}

/// Implements [`Steps`] for each binary type of floaty in each mode.
macro_rules! binary_steps {
    ($($type:ident),*) => {
        $(
            impl<M: Mode> Steps for $type<M> {
                entry_points! {
                    mul_add(multiplier: Self, addend: Self);
                    sqrt();
                    abs();
                    copy_sign(sign: Self);
                    next_up();
                    next_down();
                    round_to_integral();
                    remainder(divisor: Self);
                    truncated_remainder(divisor: Self);
                    scale_b(scale: i32);
                    log_b();
                    exp();
                    log();
                    compound(n: i64);
                    hypot(other: Self);
                    pown(n: i64);
                    rootn(n: i64);
                    reciprocal_sqrt();
                    minimum(other: Self);
                    maximum(other: Self);
                    minimum_number(other: Self);
                    maximum_number(other: Self);
                    min_num(other: Self);
                    max_num(other: Self);
                    minimum_magnitude(other: Self);
                    maximum_magnitude(other: Self);
                    minimum_magnitude_number(other: Self);
                    maximum_magnitude_number(other: Self);
                }

                #[inline]
                fn round_to_integral_by(self, rounding: Rounding) -> Self {
                    self.round_to_integral_with(rounding).0
                }
            }
        )*
    };
}

binary_steps!(Single, Double);

/// The steps of one lane of a block, over `IN` values of the lane and `P`
/// parameters of the whole call.
///
/// A chain is a function of its values and parameters. It must reach its
/// result only through the steps of `S`: a block runs it once on the host
/// unit, and again in the engine for a lane that needs it.
pub trait Chain<const IN: usize, const P: usize> {
    /// Returns the result of a lane from its values `x` and the parameters
    /// `p`.
    fn apply<S: Steps>(&self, x: [S; IN], p: [S; P]) -> S;
}

/// Implements the entry points of blocks for each binary type of floaty in
/// each mode.
macro_rules! blocks {
    ($($type:ident),*) => {
        $(
            impl<M: Mode> $type<M> {
                /// Writes the result of `chain` for lane `i` into `out[i]`:
                /// the chain takes value `i` of each slice of `x`, and the
                /// parameters `p`. The call checks the environment and
                /// selects the instruction set once, as the
                /// [module](crate::block) states. Each result is the result
                /// of the steps of the type.
                ///
                /// # Panics
                ///
                /// Panics when a slice of `x` and `out` differ in length.
                pub fn map<C, E, O, const IN: usize, const P: usize>(
                    chain: &C,
                    x: [&[E]; IN],
                    p: [Self; P],
                    out: &mut [O],
                ) where
                    C: Chain<IN, P>,
                    E: Element<Self>,
                    O: Element<Self>,
                {
                    let x = x.map(E::floats);
                    let out = O::floats_mut(out);
                    assert!(
                        x.iter().all(|values| values.len() == out.len()),
                        "each input holds one value for each output"
                    );
                    match host::block::map(chain, x, p, out) {
                        Some(false) => {}
                        Some(true) => replay(chain, x, p, out, Self::is_nan),
                        None => replay(chain, x, p, out, |_| true),
                    }
                }

                /// Returns the result of `chain` for the values `x` and the
                /// parameters `p`, with one check of the environment, as
                /// [`map`](Self::map) gives it for one lane.
                #[must_use]
                pub fn evaluate<C: Chain<IN, P>, const IN: usize, const P: usize>(
                    chain: &C,
                    x: [Self; IN],
                    p: [Self; P],
                ) -> Self {
                    host::block::evaluate(chain, x, p).unwrap_or_else(|| chain.apply(x, p))
                }
            }
        )*
    };
}

blocks!(Single, Double);

/// Writes the result of `chain` in the steps of the type into each element
/// of `out` that `select` accepts.
fn replay<C: Chain<IN, P>, F: Steps, const IN: usize, const P: usize>(
    chain: &C,
    x: [&[F]; IN],
    p: [F; P],
    out: &mut [F],
    select: impl Fn(F) -> bool,
) {
    for (index, result) in out.iter_mut().enumerate() {
        if select(*result) {
            *result = chain.apply(x.map(|values| values[index]), p);
        }
    }
}
