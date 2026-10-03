//! The exponential and the natural logarithm of IEEE 754-2019 section 9.2,
//! correctly rounded in every rounding direction, for the binary and the
//! decimal formats.
//!
//! The functions evaluate in ball arithmetic: a center and a radius that
//! bounds the distance to the true value. Every step truncates its center
//! and adds a bound on its error to the radius, so the ball holds the true
//! value. `e^x` of a nonzero rational `x` and `ln x` of a positive rational
//! `x` other than 1 are irrational (Lindemann), so the true value never lies
//! on a point of a grid. When every value of the ball has one truncation to
//! `p + 3` bits or `p + 2` digits, that truncation with a sticky bit rounds
//! as the true value does in every direction, with every flag. This is
//! Ziv's strategy: otherwise the evaluation repeats at a wider precision.
//!
//! The first working precision has twice the bits of the storage of the
//! format, and the second four times. For binary64 the second has 256 bits.
//! The hardest binary64 cases that Lefèvre and Muller list ("Some Worst
//! Cases for the Table Maker's Dilemma", 2000) need fewer than 160. No input
//! of any format is known to need more than the second precision. If one
//! did, the result would round from the center of the second ball.
//!
//! An argument of `exp` below `2^-(p + 4)`, or `10^-(p + 3)`, gives a result
//! within one grid step of 1, so its truncation follows from the sign of the
//! argument alone. An argument far past the range of the format gives an
//! overflow or an underflow without an evaluation.

mod ball;
mod constants;
mod series;

use self::ball::{Ball, truncation};
use crate::exact::Unrounded;
use crate::limbs::{self, Limbs, Widen};

/// The working widths of the elementary functions for the limbs of a
/// storage: the first has twice the bits, and the second four times.
pub trait Elementary: Widen {
    /// The first working width.
    type First: Widen;
    /// The second working width.
    type Second: Widen;
}

macro_rules! elementary {
    ($($limbs:literal => $first:literal, $second:literal),*) => {
        $(
            impl Elementary for [u64; $limbs] {
                type First = [u64; $first];
                type Second = [u64; $second];
            }
        )*
    };
}

elementary!(
    1 => 2, 4,
    2 => 4, 8,
    3 => 6, 12,
    4 => 8, 16,
    5 => 10, 20,
    6 => 12, 24,
    7 => 14, 28,
    8 => 16, 32
);

/// The radix of an argument and of its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Radix {
    /// Radix 2.
    Binary,
    /// Radix 10.
    Decimal,
}

/// The format of a result.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Target {
    /// The radix of the format and of the argument.
    pub(crate) radix: Radix,
    /// The precision in digits of the radix.
    pub(crate) precision: u32,
    /// A bound on the exponents that the format reaches, as a power of the
    /// radix: `max(emax, p - emin) + 3`. A result above `RADIX^range`
    /// overflows, and one below `RADIX^-range` underflows, in every
    /// direction.
    pub(crate) range: u32,
}

/// A nonzero finite argument, `significand * RADIX^exponent`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Argument<L> {
    /// The sign.
    pub(crate) negative: bool,
    /// The exponent, as a power of the radix.
    pub(crate) exponent: i32,
    /// The significand.
    pub(crate) significand: L,
}

/// Returns `e^x`, truncated for the rounding routine of the format, with the
/// sticky bit set.
pub(crate) fn exp<L: Elementary>(x: &Argument<L>, target: &Target) -> Unrounded<L::Second> {
    // `RADIX^leading <= |x| < RADIX^(leading + 1)`.
    let leading = leading(x, target.radix);
    let precision = i64::from(target.precision);
    let tiny = match target.radix {
        Radix::Binary => leading < -(precision + 4),
        Radix::Decimal => leading < -(precision + 3),
    };
    if tiny {
        return near_one(x.negative, target);
    }
    // Past `4 * range`, `e^x` overflows or underflows the range, because
    // `ln 10 < 4` and `ln 2 < 4`. Below it, `|x / ln 2|` stays far inside an
    // `i32`.
    let limit = 4 * u64::from(target.range);
    let huge = match target.radix {
        Radix::Binary => leading >= i64::from(64 - limit.leading_zeros()),
        Radix::Decimal => leading >= i64::from(digit_count(&[limit])),
    };
    if huge {
        return out_of_range(!x.negative, target);
    }
    ziv::<L, _>(&Exp(*x, target.radix), target)
}

/// Returns `ln x` for a positive finite `x` other than 1, truncated for the
/// rounding routine of the format, with the sticky bit set.
pub(crate) fn log<L: Elementary>(x: &Argument<L>, target: &Target) -> Unrounded<L::Second> {
    debug_assert!(!x.negative, "the logarithm takes a positive argument");
    ziv::<L, _>(&Log(*x, target.radix), target)
}

/// A function that gives the ball of its value at any working width.
trait Function {
    /// Returns the ball of the value at the width `W`, or `None` when the
    /// width cannot bound it.
    fn ball<W: Widen>(&self) -> Option<Ball<W>>;
}

/// The function `e^x`.
struct Exp<L>(Argument<L>, Radix);

/// The function `ln x`.
struct Log<L>(Argument<L>, Radix);

impl<L: Limbs> Function for Exp<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        Some(series::exp(&argument::<L, W>(&self.0, self.1)?))
    }
}

impl<L: Limbs> Function for Log<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let Argument {
            exponent,
            significand,
            ..
        } = self.0;
        match self.1 {
            Radix::Binary => series::log(&Ball::new(false, significand, i64::from(exponent))),
            Radix::Decimal => {
                series::log_decimal(&Ball::new(false, significand, 0), i64::from(exponent))
            }
        }
    }
}

/// Returns the ball of an argument at the width `W`. A binary argument is
/// exact. A decimal one multiplies or divides by a power of 10.
fn argument<L: Limbs, W: Widen>(x: &Argument<L>, radix: Radix) -> Option<Ball<W>> {
    match radix {
        Radix::Binary => Some(Ball::new(x.negative, x.significand, i64::from(x.exponent))),
        Radix::Decimal => {
            let coefficient = Ball::new(x.negative, x.significand, 0);
            let power = series::power_of_ten::<W>(u64::from(x.exponent.unsigned_abs()));
            if x.exponent >= 0 {
                Some(coefficient.mul(&power))
            } else {
                coefficient.div(&power)
            }
        }
    }
}

/// Evaluates a function at the first working width, and at the second when
/// the first cannot decide the truncation.
fn ziv<L: Elementary, F: Function>(function: &F, target: &Target) -> Unrounded<L::Second> {
    let first = function
        .ball::<L::First>()
        .and_then(|ball| truncate(&ball, target));
    if let Some(Unrounded {
        negative,
        exponent,
        significand,
        sticky,
    }) = first
    {
        return Unrounded {
            negative,
            exponent,
            significand: significand.resize(),
            sticky,
        };
    }
    let ball = function
        .ball::<L::Second>()
        .expect("the second width bounds every argument that the first one takes");
    truncate(&ball, target).unwrap_or_else(|| {
        truncate(&ball.center_only(), target)
            .expect("the center of a result is a nonzero number of the working range")
    })
}

/// Returns the truncation of every value of a ball to `p + 3` bits or
/// `p + 2` digits, with the sticky bit set, when it is the same for all of
/// them.
fn truncate<W: Widen>(ball: &Ball<W>, target: &Target) -> Option<Unrounded<W>> {
    let center = ball.center();
    if center.is_zero() {
        return None;
    }
    let precision = i64::from(target.precision);
    let (lowest, significand) = match target.radix {
        Radix::Binary => {
            let lowest = center.top() - precision - 2;
            (lowest, truncation(ball, lowest)?)
        }
        Radix::Decimal => {
            // `floor(top * log10(2))`, with `log10(2)` to 32 bits. The
            // decimal exponent of the value lies at the estimate or one
            // above it, so the truncation has `p + 2` to `p + 4` digits.
            let estimate = (i128::from(center.top()) * 1_292_913_986).div_euclid(1 << 32);
            let lowest = i64::try_from(estimate).ok()? - precision - 2;
            let power = series::power_of_ten::<W>(lowest.unsigned_abs());
            let scaled = if lowest >= 0 {
                ball.abs().div(&power)?
            } else {
                ball.abs().mul(&power)
            };
            (lowest, truncation(&scaled, 0)?)
        }
    };
    Some(Unrounded {
        negative: center.is_negative(),
        exponent: i32::try_from(lowest).ok()?,
        significand,
        sticky: true,
    })
}

/// Returns `RADIX^digits` in the limbs `L`.
fn power<L: Limbs>(radix: Radix, digits: u32) -> L {
    match radix {
        Radix::Binary => L::ZERO.with_bit(digits),
        Radix::Decimal => (0..digits).fold(L::ZERO.with_bit(0), |power, _| {
            limbs::multiply_small(power, 10)
        }),
    }
}

/// Returns the truncation of `e^x` for an argument below the tiny bound.
///
/// For `x > 0`, `e^x` lies above 1 by less than `2|x|`, which is below one
/// unit of the `p + 3` bits, or `p + 2` digits, of 1. For `x < 0`, `e^x`
/// lies below 1 by less than `|x|`, so it truncates to all ones, or nines,
/// one digit further down.
fn near_one<L: Limbs>(negative: bool, target: &Target) -> Unrounded<L> {
    let precision = target.precision;
    let digits = match target.radix {
        Radix::Binary => precision + 2,
        Radix::Decimal => precision + 1,
    };
    let (exponent, significand) = if negative {
        let below: L = power(target.radix, digits + 1);
        (digits + 1, below.sub(L::ZERO.with_bit(0)))
    } else {
        (digits, power(target.radix, digits))
    };
    Unrounded {
        negative: false,
        exponent: -i32::try_from(exponent).expect("a precision fits an i32"),
        significand,
        sticky: true,
    }
}

/// Returns a value past the range of the format: above `RADIX^range` for an
/// overflow, or below `RADIX^-range` for an underflow.
fn out_of_range<L: Limbs>(overflow: bool, target: &Target) -> Unrounded<L> {
    let digits = match target.radix {
        Radix::Binary => target.precision + 2,
        Radix::Decimal => target.precision + 1,
    };
    let range = i32::try_from(target.range).expect("a range fits an i32");
    let digits_i32 = i32::try_from(digits).expect("a precision fits an i32");
    Unrounded {
        negative: false,
        exponent: if overflow { range } else { -range - digits_i32 },
        significand: power(target.radix, digits),
        sticky: true,
    }
}

/// Returns the exponent `e` with `RADIX^e <= |x| < RADIX^(e + 1)`.
fn leading<L: Limbs>(x: &Argument<L>, radix: Radix) -> i64 {
    let length = match radix {
        Radix::Binary => x.significand.bit_length(),
        Radix::Decimal => digit_count(&x.significand),
    };
    i64::from(x.exponent) + i64::from(length) - 1
}

/// Returns the number of decimal digits of an integer below 2^128.
fn digit_count<L: Limbs>(value: &L) -> u32 {
    let value = value.resize::<[u64; 3]>();
    let mut power = [1_u64, 0, 0];
    let mut count = 0;
    while value.compare(&power).is_ge() {
        power = limbs::multiply_small(power, 10);
        count += 1;
    }
    count
}
