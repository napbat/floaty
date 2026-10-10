//! The inverse trigonometric functions of IEEE 754-2019 section 9.2, `asin`,
//! `acos`, and `atan`, and their forms scaled by pi, `asinPi`, `acosPi`, and
//! `atanPi`, which divide the angle by pi.
//!
//! The functions evaluate in ball arithmetic from `atan`, in forms that keep
//! the error relative to the value near each zero of the function:
//!
//! - `atan |x|`;
//! - `asin |x| = atan(|x| / sqrt((1 - |x|) (1 + |x|)))`;
//! - `acos x = 2 atan(sqrt((1 - x) / (1 + x)))`.
//!
//! `asin` and `atan` take the sign of `x`. `1 - x` and `1 + x` are exact,
//! because a ball of a decimal argument near ±1 would lose digits of them.
//!
//! An inverse trigonometric function of a nonzero rational is irrational,
//! and so is `acos` of a rational other than 1 (Lindemann). Divided by pi,
//! an angle is rational only where the sine or the tangent is 0, ±1/2, or
//! ±1 (Niven). At ±1/2, the angle is a multiple of 1/6, which lies on no
//! point of a binary or decimal grid, so a ball decides it. The other exact
//! angles, at 0, ±1, and infinities, are [`Special::Angle`]: a multiple of
//! `pi / 4`, or of 1/4 scaled by pi.
//!
//! Three kinds of argument take their truncation without a ball:
//!
//! - An argument below `2^-ceil((p + 5) / 2)`, or `10^-ceil((p + 3) / 2)`,
//!   gives `asin` just above `|x|` and `atan` just below it. The rest of
//!   each series lies below `|x|^3`, less than one unit of the truncation.
//! - An argument below `RADIX^-(t + 1)`, with `t` the digits of the
//!   truncation, gives `acosPi = 1/2 - asin(x) / pi` just beside 1/2, below
//!   it for `x > 0`. `|x| / pi` lies below one unit there.
//! - An argument from `RADIX^(t + 1)` gives `|atanPi x| = 1/2 - atan(1 / |x|)
//!   / pi` just below 1/2.

use core::cmp::Ordering;

use super::ball::Ball;
use super::constants::pi;
use super::exact::{self, one_plus};
use super::{
    Argument, Elementary, Function, Radix, Special, Target, argument, beside, compare_one, leading,
    number, series, ziv,
};
use crate::exact::Unrounded;
use crate::limbs::{Limbs, Widen};

/// An inverse trigonometric function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Inverse {
    /// `asin`, or `asinPi`.
    Sin,
    /// `acos`, or `acosPi`.
    Cos,
    /// `atan`, or `atanPi`.
    Tan,
}

impl Inverse {
    /// Returns the result at a zero with the sign `negative`: the zero for
    /// `asin` and `atan`, and a right angle for `acos`.
    pub(super) fn at_zero<L>(self, negative: bool, scaled: bool) -> Special<L> {
        match self {
            Self::Sin | Self::Tan => Special::Zero { negative },
            Self::Cos => Special::Angle { eighths: 2, scaled },
        }
    }

    /// Returns the result at an infinity with the sign `negative`: a right
    /// angle with the sign for `atan`, and invalid otherwise.
    pub(super) fn at_infinity<L>(self, negative: bool, scaled: bool) -> Special<L> {
        match self {
            Self::Tan => Special::Angle {
                eighths: if negative { -2 } else { 2 },
                scaled,
            },
            Self::Sin | Self::Cos => Special::Invalid,
        }
    }
}

/// Returns the result at a finite argument, or the argument to evaluate.
/// `asin` and `acos` take `|x| <= 1`. At ±1, `asin` gives a right angle with
/// the sign, `acos` gives +0 at 1 and a straight angle at -1, and `atan`
/// gives half a right angle with the sign.
pub(super) fn special<L: Limbs>(
    function: Inverse,
    scaled: bool,
    x: Argument<L>,
    radix: Radix,
) -> Special<L> {
    let sign = if x.negative { -1 } else { 1 };
    match (function, compare_one(&x, radix)) {
        (Inverse::Sin | Inverse::Cos, Ordering::Greater) => Special::Invalid,
        (Inverse::Sin, Ordering::Equal) => Special::Angle {
            eighths: 2 * sign,
            scaled,
        },
        (Inverse::Cos, Ordering::Equal) if x.negative => Special::Angle { eighths: 4, scaled },
        (Inverse::Cos, Ordering::Equal) => Special::Zero { negative: false },
        (Inverse::Tan, Ordering::Equal) => Special::Angle {
            eighths: sign,
            scaled,
        },
        _ => Special::Evaluate(x),
    }
}

/// Returns `eighths pi / 4`, or `eighths / 4` when `scaled` is set, truncated
/// to `p + 3` bits or `p + 2` digits with the sticky bit set, or exact.
pub(super) fn angle<L: Elementary>(
    eighths: i8,
    scaled: bool,
    target: &Target,
) -> Unrounded<L::Second> {
    let negative = eighths < 0;
    let count = eighths.unsigned_abs();
    if scaled {
        // `count / 4` is `count 2^-2`, or `25 count 10^-2`.
        let value = match target.radix {
            Radix::Binary => u64::from(count),
            Radix::Decimal => 25 * u64::from(count),
        };
        let value = <L::Second as Limbs>::ZERO.with_limb(0, value);
        return exact::exact(negative, value, -2, target);
    }
    ziv::<L, _>(&Angle { negative, count }, target)
}

/// Returns `function` of an argument that [`special`] gives, divided by pi
/// when `scaled` is set, truncated to `p + 3` bits or `p + 2` digits with the
/// sticky bit set.
pub(super) fn evaluate<L: Elementary>(
    function: Inverse,
    scaled: bool,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    let radix = target.radix;
    let leading = leading(x, radix);
    let kept = i64::from(target.kept());
    if !scaled && function != Inverse::Cos && leading < -target.half_tiny() {
        return beside(
            x.negative,
            x.significand.resize(),
            i64::from(x.exponent),
            function == Inverse::Tan,
            target,
        );
    }
    // 1/2 is `1 * 2^-1`, or `5 * 10^-1`.
    let half = <L::Second as Limbs>::ZERO.with_limb(
        0,
        match radix {
            Radix::Binary => 1,
            Radix::Decimal => 5,
        },
    );
    match function {
        Inverse::Cos if scaled && leading < -(kept + 1) => {
            return beside(false, half, -1, !x.negative, target);
        }
        Inverse::Tan if scaled && leading > kept => {
            return beside(x.negative, half, -1, true, target);
        }
        _ => {}
    }
    let value = Value {
        function,
        scaled,
        x: *x,
        radix,
    };
    ziv::<L, _>(&value, target)
}

/// The angle `count pi / 4` with a sign.
struct Angle {
    /// The sign.
    negative: bool,
    /// The count of eighths of a turn.
    count: u8,
}

impl Function for Angle {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let value = pi::<W>()
            .mul(&Ball::integer(i64::from(self.count)))
            .scale(-2);
        Some(if self.negative { value.negate() } else { value })
    }
}

/// An inverse trigonometric function at an argument.
struct Value<L> {
    /// The function.
    function: Inverse,
    /// `true` for the form scaled by pi.
    scaled: bool,
    /// The argument.
    x: Argument<L>,
    /// The radix of the argument.
    radix: Radix,
}

impl<L: Limbs> Value<L> {
    /// Returns the ball of `1 + x`, or of `1 - x` when `minus` is set, exact
    /// when it fits the width `W`. `|x|` is below 1.
    fn one_plus<W: Widen>(&self, minus: bool) -> Option<Ball<W>> {
        let x = Argument {
            negative: self.x.negative != minus,
            ..self.x
        };
        match one_plus::<L, W>(&x, self.radix) {
            Some((value, exponent)) => number(false, value, exponent, self.radix),
            None => Some(Ball::one().add(&argument::<L, W>(&x, self.radix)?)),
        }
    }
}

impl<L: Limbs> Function for Value<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let value = match self.function {
            Inverse::Tan => {
                let magnitude = Argument {
                    negative: false,
                    ..self.x
                };
                series::arctangent(&argument::<L, W>(&magnitude, self.radix)?)?
            }
            Inverse::Sin => {
                let magnitude = Argument {
                    negative: false,
                    ..self.x
                };
                let y = argument::<L, W>(&magnitude, self.radix)?;
                let rest = self.one_plus::<W>(!self.x.negative)?;
                let sum = self.one_plus::<W>(self.x.negative)?;
                let cosine = rest.mul(&sum).sqrt()?;
                series::arctangent(&y.div(&cosine)?)?
            }
            Inverse::Cos => {
                let rest = self.one_plus::<W>(true)?;
                let sum = self.one_plus::<W>(false)?;
                series::arctangent(&rest.div(&sum)?.sqrt()?)?.scale(1)
            }
        };
        let value = if self.scaled {
            value.div(&pi())?
        } else {
            value
        };
        Some(if self.x.negative && self.function != Inverse::Cos {
            value.negate()
        } else {
            value
        })
    }
}
