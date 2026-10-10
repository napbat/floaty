//! The trigonometric functions of IEEE 754-2019 section 9.2: `sin`, `cos`,
//! and `tan`.
//!
//! The module `reduction` reduces `|x|` to the angle `r = pi f / 2` within
//! about `pi / 4` of zero, and its quadrant `q`. The series of `sin` and
//! `cos` at `r` give the value: `sin |x|` is `sin r`, `cos r`, `-sin r`, or
//! `-cos r` for `q` from 0 to 3, `cos |x|` is `cos r`, `-sin r`, `-cos r`, or
//! `sin r`, and `tan |x|` is `tan r` for an even `q` and `-1 / tan r` for an
//! odd one. `sin` and `tan` take the sign of `x`.
//!
//! A trigonometric function of a nonzero rational is irrational
//! (Lindemann), so the value lies on no point of a grid. An argument below
//! `2^-ceil((p + 5) / 2)`, or `10^-ceil((p + 3) / 2)`, takes its truncation
//! without a ball: `sin` lies just below `|x|`, `tan` just above it, and
//! `cos` just below 1, because the rest of each series lies below `|x|^3`,
//! or below `x^2` for `cos`, less than one unit of the truncation.
//!
//! The bits of 2/pi of the reduction reach every argument below
//! `2^(2^22)`, every finite value of binary512 and of decimal128. A larger
//! argument, which only a binary format with 24 exponent bits or more holds,
//! gives the default NaN and signals invalid.

use super::ball::Ball;
use super::reduction::{REACH, Reduced, reduce};
use super::{
    Argument, Elementary, Function, Radix, Special, Target, beside, leading, near_one, series, ziv,
};
use crate::exact::Unrounded;
use crate::limbs::{Limbs, Widen};

/// A trigonometric function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Circular {
    /// `sin`.
    Sin,
    /// `cos`.
    Cos,
    /// `tan`.
    Tan,
}

impl Circular {
    /// Returns the result at a zero with the sign `negative`: the zero for
    /// `sin` and `tan`, and 1 for `cos`.
    pub(super) fn at_zero<L>(self, negative: bool) -> Special<L> {
        match self {
            Self::Sin | Self::Tan => Special::Zero { negative },
            Self::Cos => Special::One { negative: false },
        }
    }
}

/// Returns the result at a finite argument, or the argument to evaluate. An
/// argument of `2^(2^22)` or more lies past the bits of the reduction.
pub(super) fn special<L: Limbs>(x: Argument<L>, radix: Radix) -> Special<L> {
    if leading(&x, radix) >= REACH {
        Special::Invalid
    } else {
        Special::Evaluate(x)
    }
}

/// Returns `function` of an argument that [`special`] gives, truncated to
/// `p + 3` bits or `p + 2` digits with the sticky bit set.
pub(super) fn evaluate<L: Elementary>(
    function: Circular,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    if leading(x, target.radix) < -target.half_tiny() {
        let toward_zero = match function {
            Circular::Cos => return near_one(true, target),
            Circular::Sin => true,
            Circular::Tan => false,
        };
        return beside(
            x.negative,
            x.significand.resize(),
            i64::from(x.exponent),
            toward_zero,
            target,
        );
    }
    let value = Value {
        function,
        x: *x,
        radix: target.radix,
    };
    ziv::<L, _>(&value, target)
}

/// A trigonometric function at an argument.
struct Value<L> {
    /// The function.
    function: Circular,
    /// The argument.
    x: Argument<L>,
    /// The radix of the argument.
    radix: Radix,
}

impl<L: Limbs> Function for Value<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let Reduced { quadrant, angle } = reduce::<L, W>(&self.x, self.radix)?;
        let value = match self.function {
            Circular::Tan => {
                let (sine, cosine) = (series::sine(&angle), series::cosine(&angle));
                if quadrant % 2 == 0 {
                    sine.div(&cosine)?
                } else {
                    cosine.div(&sine)?.negate()
                }
            }
            Circular::Sin | Circular::Cos => {
                // `cos |x|` is `sin` a quadrant further on.
                let quadrant = if self.function == Circular::Cos {
                    (quadrant + 1) & 3
                } else {
                    quadrant
                };
                let value = if quadrant % 2 == 0 {
                    series::sine(&angle)
                } else {
                    series::cosine(&angle)
                };
                if quadrant >= 2 { value.negate() } else { value }
            }
        };
        Some(if self.x.negative && self.function != Circular::Cos {
            value.negate()
        } else {
            value
        })
    }
}
