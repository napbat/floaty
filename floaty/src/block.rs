//! Blocks: a chain of binary32 steps for each lane of slices, with one check
//! of the environment and one selection of the instruction set for the
//! whole call.
//!
//! A [`Chain`] states the steps of one lane once, generic over [`Steps`].
//! [`Float::map`] runs the chain for each lane of slices, and
//! [`Float::evaluate`] runs it for one lane. Each step gives the result of
//! its entry point in the mode of the type: the operators, `mul_add`,
//! `sqrt`, `abs`, `minimum`, `maximum`, `minimum_number`, and
//! `maximum_number`. So a block gives the bits of the same steps one at a
//! time, on every host.
//!
//! Where the build has a host path, and the mode and the environment allow
//! it, the steps run as Rust operations on `f32` after one check of the
//! environment, in the instruction set that the processor selects. LLVM
//! then sees the whole chain and computes many lanes in vector
//! instructions. A lane whose result is a NaN runs the chain again in the
//! engine, which selects the NaN by the rule of the mode. So does a lane
//! with a `mul_add` where the instruction set has no fused multiply-add.
//! Elsewhere every lane runs the steps of the type.
//!
//! ```
//! use floaty::F32;
//! use floaty::block::{Chain, Steps};
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
//! ```

use core::ops::{Add, Div, Mul, Neg, Sub};

use crate::env::Mode;
use crate::float::Float;
use crate::format::Binary;
use crate::host;
use crate::lanes::Element;
use crate::sealed::Sealed;

/// The binary32 type of the mode `M`.
type Single<M> = Float<Binary<8>, 32, M>;

/// The steps of a [`Chain`]: the arithmetic, the fused multiply-add, the
/// square root, the absolute value, and the minimum and maximum operations
/// of binary32, each rounded once to nearest even as its entry point
/// rounds. The binary32 types of floaty implement the trait, and so does
/// the lane type of the host path of a block. The trait is sealed.
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
}

impl<M: Mode> Steps for Single<M> {
    #[inline]
    fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        Self::mul_add(self, multiplier, addend)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::sqrt(self)
    }

    #[inline]
    fn abs(self) -> Self {
        Self::abs(self)
    }

    #[inline]
    fn minimum(self, other: Self) -> Self {
        Self::minimum(self, other)
    }

    #[inline]
    fn maximum(self, other: Self) -> Self {
        Self::maximum(self, other)
    }

    #[inline]
    fn minimum_number(self, other: Self) -> Self {
        Self::minimum_number(self, other)
    }

    #[inline]
    fn maximum_number(self, other: Self) -> Self {
        Self::maximum_number(self, other)
    }
}

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

impl<M: Mode> Float<Binary<8>, 32, M> {
    /// Writes the result of `chain` for lane `i` into `out[i]`: the chain
    /// takes value `i` of each slice of `x`, and the parameters `p`. The
    /// call checks the environment and selects the instruction set once, as
    /// the [module](crate::block) states. Each result is the result of the
    /// steps of the type.
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

    /// Returns the result of `chain` for the values `x` and the parameters
    /// `p`, with one check of the environment, as [`map`](Self::map) gives
    /// it for one lane.
    #[must_use]
    pub fn evaluate<C: Chain<IN, P>, const IN: usize, const P: usize>(
        chain: &C,
        x: [Self; IN],
        p: [Self; P],
    ) -> Self {
        host::block::evaluate(chain, x, p).unwrap_or_else(|| chain.apply(x, p))
    }
}

/// Writes the result of `chain` in the steps of the type into each element
/// of `out` that `select` accepts.
fn replay<C: Chain<IN, P>, M: Mode, const IN: usize, const P: usize>(
    chain: &C,
    x: [&[Single<M>]; IN],
    p: [Single<M>; P],
    out: &mut [Single<M>],
    select: impl Fn(Single<M>) -> bool,
) {
    for (index, result) in out.iter_mut().enumerate() {
        if select(*result) {
            *result = chain.apply(x.map(|values| values[index]), p);
        }
    }
}
