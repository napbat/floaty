//! `atan2(y, x)` and `atan2Pi(y, x)` of IEEE 754-2019 section 9.2: the
//! angle of the point `(x, y)`, divided by pi for `atan2Pi`.
//!
//! The special cases of section 9.2.1 give an exact zero, or an angle of
//! [`super::Special::Angle`]: a multiple of `pi / 4`, or of 1/4 scaled by
//! pi. So does `|y| = |x|`, at `±pi / 4` and `±3 pi / 4`. Every other pair
//! of finite nonzero arguments evaluates `t = |y| / |x|` in ball arithmetic:
//! the angle is `atan t` for `x > 0` and `pi - atan t` for `x < 0`, with the
//! sign of `y`. `atan t` of a rational `t` other than 0 and 1 is irrational
//! (Lindemann), and so is the angle divided by pi (Niven).
//!
//! Three kinds of argument take their truncation without a ball:
//!
//! - `atan2` with `x > 0` and `t` below `RADIX^-ceil((t' + p + 2) / 2)`,
//!   with `t'` the digits of the truncation, lies below `t` by less than
//!   `t^3 / 3`. An exact quotient of the significands gives `t` to `t' + 1`
//!   digits or more. `t` lies on that grid, or off it by at least one unit
//!   over the significand of `x`, more than `t^3 / 3`. So the value
//!   truncates to one unit below the quotient, or to the quotient.
//! - `atan2Pi` with `x < 0` and `t` below `RADIX^-t'` lies just below 1 in
//!   magnitude.
//! - `atan2Pi` with `|x| / |y|` below `RADIX^-(t' + 1)` lies just beside 1/2
//!   in magnitude: below it for `x > 0`, and above it for `x < 0`.

use core::cmp::Ordering;

use super::ball::Ball;
use super::bivariate::{Outcome, Point, point};
use super::constants::pi;
use super::exact::clamped;
use super::inverse::angle;
use super::{
    Argument, Elementary, Function, Radix, Target, argument, beside, digit_count, leading, power,
    series, ziv,
};
use crate::exact::Unrounded;
use crate::limbs::{self, Limbs, Widen};
use crate::unpacked::Unpacked;

/// Returns `atan2(y, x)`, or `atan2Pi(y, x)` when `scaled` is set, for
/// operands that are not NaNs.
pub(super) fn atan2<L: Elementary>(
    scaled: bool,
    y: &Unpacked<L>,
    x: &Unpacked<L>,
    target: &Target,
) -> Outcome<L::Second> {
    let (y_negative, y) = point(y);
    let (x_negative, x) = point(x);
    // An angle of `eighths pi / 4`, with the sign of `y`.
    let quadrant = |eighths: i8| {
        let eighths = if y_negative { -eighths } else { eighths };
        Outcome::Truncated(angle::<L>(eighths, scaled, target))
    };
    match (y, x) {
        (Point::Zero, _) | (Point::Finite(_), Point::Infinity) => {
            if x_negative {
                quadrant(4)
            } else {
                Outcome::Zero {
                    negative: y_negative,
                }
            }
        }
        (Point::Infinity, Point::Infinity) => quadrant(if x_negative { 3 } else { 1 }),
        (Point::Infinity, _) | (Point::Finite(_), Point::Zero) => quadrant(2),
        (Point::Finite(y), Point::Finite(x)) => {
            match compare_magnitudes::<L>(&y, &x, target.radix) {
                Ordering::Equal => quadrant(if x_negative { 3 } else { 1 }),
                _ => Outcome::Truncated(evaluate::<L>(scaled, &y, &x, target)),
            }
        }
    }
}

/// Returns the angle of two finite nonzero arguments with `|y| != |x|`,
/// truncated to `p + 3` bits or `p + 2` digits with the sticky bit set.
fn evaluate<L: Elementary>(
    scaled: bool,
    y: &Argument<L>,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    let radix = target.radix;
    let kept = i64::from(target.kept());
    // `RADIX^(spread - 1) < |y| / |x| < RADIX^(spread + 1)`.
    let spread = leading(y, radix) - leading(x, radix);
    let one = <L::Second as Limbs>::ZERO.with_bit(0);
    // 1/2 is `1 * 2^-1`, or `5 * 10^-1`.
    let half = <L::Second as Limbs>::ZERO.with_limb(
        0,
        match radix {
            Radix::Binary => 1,
            Radix::Decimal => 5,
        },
    );
    if scaled && x.negative && spread < -kept {
        return beside(y.negative, one, 0, true, target);
    }
    if scaled && spread > kept + 1 {
        return beside(y.negative, half, -1, !x.negative, target);
    }
    let tiny = (kept + i64::from(target.precision) + 3) / 2;
    if !scaled && !x.negative && spread < -tiny {
        return below_quotient::<L>(y, x, target);
    }
    let value = Value {
        scaled,
        y: *y,
        x: *x,
        radix,
    };
    ziv::<L, _>(&value, target)
}

/// Returns the truncation of `atan(|y| / |x|)` with the sign of `y` for a
/// tiny quotient: the quotient `q` of `|y| RADIX^s` by `|x|`, with `t' + 1`
/// or `t' + 2` digits, and the sticky bit, or `q - 1` when the division is
/// exact.
fn below_quotient<L: Elementary>(
    y: &Argument<L>,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    let radix = target.radix;
    let length = |value: &L| match radix {
        Radix::Binary => value.bit_length(),
        Radix::Decimal => digit_count(value),
    };
    // `|y| RADIX^s / |x|` lies in `[RADIX^t', RADIX^(t' + 2))`.
    let shift = target.kept() + 1 + length(&x.significand) - length(&y.significand);
    let numerator = match radix {
        Radix::Binary => y.significand.resize::<L::Second>().shl(shift),
        Radix::Decimal => limbs::multiply_fit(
            y.significand.resize::<L::Second>(),
            power::<L::Second>(radix, shift),
        ),
    };
    let (quotient, rest) = limbs::divide(numerator, x.significand.resize::<L::Second>());
    let significand = if rest.is_zero() {
        quotient.sub(<L::Second as Limbs>::ZERO.with_bit(0))
    } else {
        quotient
    };
    let exponent = i64::from(y.exponent) - i64::from(x.exponent) - i64::from(shift);
    Unrounded {
        negative: y.negative,
        exponent: clamped(i128::from(exponent), target),
        significand,
        sticky: true,
    }
}

/// Compares the magnitudes of two finite nonzero arguments.
fn compare_magnitudes<L: Elementary>(a: &Argument<L>, b: &Argument<L>, radix: Radix) -> Ordering {
    let order = leading(a, radix).cmp(&leading(b, radix));
    if order != Ordering::Equal {
        return order;
    }
    // With equal leading exponents, the exponents differ by less than the
    // precision, so the significand of the larger exponent moves down to the
    // smaller one in the second working width.
    let difference = i64::from(a.exponent) - i64::from(b.exponent);
    let shift = u32::try_from(difference.unsigned_abs())
        .expect("the exponents of equal leading exponents differ by less than the precision");
    let scaled = |value: &L| match radix {
        Radix::Binary => value.resize::<L::Second>().shl(shift),
        Radix::Decimal => limbs::multiply_fit(
            value.resize::<L::Second>(),
            power::<L::Second>(radix, shift),
        ),
    };
    if difference >= 0 {
        scaled(&a.significand).compare(&b.significand.resize())
    } else {
        a.significand
            .resize::<L::Second>()
            .compare(&scaled(&b.significand))
    }
}

/// `atan2` or `atan2Pi` at two finite nonzero arguments.
struct Value<L> {
    /// `true` for `atan2Pi`.
    scaled: bool,
    /// The first argument.
    y: Argument<L>,
    /// The second argument.
    x: Argument<L>,
    /// The radix of the arguments.
    radix: Radix,
}

impl<L: Limbs> Function for Value<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let magnitude = |x: &Argument<L>| Argument {
            negative: false,
            ..*x
        };
        let a = argument::<L, W>(&magnitude(&self.y), self.radix)?;
        let b = argument::<L, W>(&magnitude(&self.x), self.radix)?;
        let angle = series::arctangent(&a.div(&b)?)?;
        let angle = if self.x.negative {
            pi::<W>().sub(&angle)
        } else {
            angle
        };
        let value = if self.scaled {
            angle.div(&pi())?
        } else {
            angle
        };
        Some(if self.y.negative {
            value.negate()
        } else {
            value
        })
    }
}
