//! The hyperbolic functions and their inverses of IEEE 754-2019 section 9.2.
//!
//! The functions evaluate in ball arithmetic from `e^x`, `e^x - 1`,
//! `ln(1 + x)`, and the square root, in forms that keep the error relative
//! to the value near each zero of the function:
//!
//! - `sinh |x| = (u + u / (u + 1)) / 2`, with `u = e^|x| - 1`;
//! - `cosh x = (v + 1 / v) / 2`, with `v = e^|x|`;
//! - `tanh |x| = u / (u + 2)`, with `u = e^(2|x|) - 1`;
//! - `asinh |x| = ln(1 + |x| + x^2 / (1 + sqrt(1 + x^2)))`;
//! - `acosh x = ln(1 + e + sqrt(e (e + 2)))`, with `e = x - 1`;
//! - `atanh |x| = ln(1 + 2|x| / (1 - |x|)) / 2`.
//!
//! The odd functions take the sign of `x`. `x - 1` and `1 - |x|` are exact,
//! because a ball of a decimal argument near 1 would lose digits of them.
//!
//! A hyperbolic function of a nonzero rational is irrational: each one is a
//! rational function of `e^x`, or the logarithm of an algebraic number other
//! than 1. So the value lies on no point of a grid. Three kinds of argument
//! take their truncation without a ball:
//!
//! - An argument below `2^-ceil((p + 5) / 2)`, or `10^-ceil((p + 3) / 2)`,
//!   gives `sinh` and `atanh` just above `|x|`, `tanh` and `asinh` just below
//!   it, and `cosh` just above 1. The rest of each series lies below `|x|^3`,
//!   or below `x^2` for `cosh`, less than one unit of the truncation.
//! - An argument from `(p + 4) ln 2 / 2`, or `(p + 3) ln 10 / 2`, gives
//!   `|tanh x|` just below 1, because `1 - |tanh x|` is below `2 e^-2|x|`.
//! - An argument far past the range gives an overflow of `sinh` and `cosh`,
//!   as of `e^x`.

use core::cmp::Ordering;

use super::ball::Ball;
use super::exact::{self, minus_one, one_plus};
use super::{
    Argument, Base, Elementary, Function, Radix, Special, Target, argument, beside, compare_one,
    leading, near_one, number, out_of_range, past, series, ziv,
};
use crate::exact::Unrounded;
use crate::limbs::{Limbs, Widen};

/// A hyperbolic function or its inverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Hyperbolic {
    /// `sinh`.
    Sinh,
    /// `cosh`.
    Cosh,
    /// `tanh`.
    Tanh,
    /// `asinh`.
    Asinh,
    /// `acosh`.
    Acosh,
    /// `atanh`.
    Atanh,
}

impl Hyperbolic {
    /// Returns the result at a zero with the sign `negative`: the zero for an
    /// odd function, 1 for `cosh`, and invalid for `acosh`.
    pub(super) fn at_zero<L>(self, negative: bool) -> Special<L> {
        match self {
            Self::Cosh => Special::One { negative: false },
            Self::Acosh => Special::Invalid,
            Self::Sinh | Self::Tanh | Self::Asinh | Self::Atanh => Special::Zero { negative },
        }
    }

    /// Returns the result at an infinity with the sign `negative`: ±1 for
    /// `tanh`, invalid for `atanh` and for `acosh` of -inf, and an infinity
    /// otherwise.
    pub(super) fn at_infinity<L>(self, negative: bool) -> Special<L> {
        match self {
            Self::Sinh | Self::Asinh => Special::Infinity { negative },
            Self::Cosh => Special::Infinity { negative: false },
            Self::Tanh => Special::One { negative },
            Self::Acosh if !negative => Special::Infinity { negative: false },
            Self::Acosh | Self::Atanh => Special::Invalid,
        }
    }

    /// Returns `true` for an odd function, which takes the sign of `x`.
    fn is_odd(self) -> bool {
        !matches!(self, Self::Cosh | Self::Acosh)
    }
}

/// Returns the result at a finite argument, or the argument to evaluate.
/// `acosh` takes `x >= 1` and gives +0 at 1. `atanh` takes `|x| <= 1` and
/// has a pole at ±1.
pub(super) fn special<L: Limbs>(function: Hyperbolic, x: Argument<L>, radix: Radix) -> Special<L> {
    match function {
        Hyperbolic::Acosh => match (x.negative, compare_one(&x, radix)) {
            (true, _) | (false, Ordering::Less) => Special::Invalid,
            (false, Ordering::Equal) => Special::Zero { negative: false },
            (false, Ordering::Greater) => Special::Evaluate(x),
        },
        Hyperbolic::Atanh => match compare_one(&x, radix) {
            Ordering::Greater => Special::Invalid,
            Ordering::Equal => Special::Pole {
                negative: x.negative,
            },
            Ordering::Less => Special::Evaluate(x),
        },
        Hyperbolic::Sinh | Hyperbolic::Cosh | Hyperbolic::Tanh | Hyperbolic::Asinh => {
            Special::Evaluate(x)
        }
    }
}

/// Returns `function` of an argument that [`special`] gives, truncated to
/// `p + 3` bits or `p + 2` digits with the sticky bit set.
pub(super) fn evaluate<L: Elementary>(
    function: Hyperbolic,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    let radix = target.radix;
    // `|x| < RADIX^-half` puts `x^2` below `RADIX^-tiny`.
    let leading = leading(x, radix);
    let half = (target.tiny() + 1) / 2;
    if leading < -half {
        let toward_zero = match function {
            Hyperbolic::Cosh => return near_one(false, target),
            Hyperbolic::Sinh | Hyperbolic::Atanh => false,
            Hyperbolic::Tanh | Hyperbolic::Asinh => true,
            Hyperbolic::Acosh => unreachable!("acosh takes an argument above 1"),
        };
        let exponent = i64::from(x.exponent);
        return beside(
            x.negative,
            x.significand.resize(),
            exponent,
            toward_zero,
            target,
        );
    }
    match function {
        Hyperbolic::Sinh | Hyperbolic::Cosh
            if past(leading, 4 * u64::from(target.range), radix) =>
        {
            return Unrounded {
                negative: function == Hyperbolic::Sinh && x.negative,
                ..out_of_range(true, target)
            };
        }
        Hyperbolic::Tanh if exact::at_least(x, radix, near_one_bound(target)) => {
            return Unrounded {
                negative: x.negative,
                ..near_one(true, target)
            };
        }
        _ => {}
    }
    let value = Value {
        function,
        x: *x,
        radix,
    };
    ziv::<L, _>(&value, target)
}

/// Returns the least `|x|` from which `1 - |tanh x|` lies below one unit of
/// the truncation of 1 below it: from `(kept + 1) ln(RADIX) / 2`,
/// `2 e^-2|x|` is at most `2 RADIX^-(kept + 1)`, below `RADIX^-kept`.
fn near_one_bound(target: &Target) -> u64 {
    let (numerator, denominator) = Base::E.log_of(target.radix);
    (u64::from(target.kept()) + 1) * numerator / denominator / 2 + 1
}

/// A hyperbolic function at an argument.
struct Value<L> {
    /// The function.
    function: Hyperbolic,
    /// The argument.
    x: Argument<L>,
    /// The radix of the argument.
    radix: Radix,
}

impl<L: Limbs> Function for Value<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let magnitude = Argument {
            negative: false,
            ..self.x
        };
        let y = argument::<L, W>(&magnitude, self.radix)?;
        let one = Ball::one();
        let value = match self.function {
            Hyperbolic::Sinh => {
                let u = series::exp_minus_one(&y);
                u.add(&u.div(&u.add(&one))?).scale(-1)
            }
            Hyperbolic::Cosh => {
                let v = series::exp(&y);
                v.add(&one.div(&v)?).scale(-1)
            }
            Hyperbolic::Tanh => {
                let u = series::exp_minus_one(&y.scale(1));
                u.div(&u.add(&Ball::integer(2)))?
            }
            Hyperbolic::Asinh => {
                let square = y.mul(&y);
                let root = one.add(&square).sqrt()?;
                series::log_one_plus(&y.add(&square.div(&one.add(&root))?))?
            }
            Hyperbolic::Acosh => {
                let e = match minus_one::<L, W>(&self.x, self.radix) {
                    Some((value, exponent)) => number(false, value, exponent, self.radix)?,
                    None => y.sub(&one),
                };
                let root = e.mul(&e.add(&Ball::integer(2))).sqrt()?;
                series::log_one_plus(&e.add(&root))?
            }
            Hyperbolic::Atanh => {
                let negated = Argument {
                    negative: true,
                    ..self.x
                };
                let rest = match one_plus::<L, W>(&negated, self.radix) {
                    Some((value, exponent)) => number(false, value, exponent, self.radix)?,
                    None => one.sub(&y),
                };
                series::log_one_plus(&y.scale(1).div(&rest)?)?.scale(-1)
            }
        };
        Some(if self.x.negative && self.function.is_odd() {
            value.negate()
        } else {
            value
        })
    }
}
