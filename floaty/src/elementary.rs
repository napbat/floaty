//! The exponentials and the logarithms of IEEE 754-2019 section 9.2, in base
//! e, 2, and 10, and their forms shifted by one, `b^x - 1` and
//! `log_b(1 + x)`, correctly rounded in every rounding direction for the
//! binary and the decimal formats. Also `compound` for the binary formats.
//!
//! The functions evaluate in ball arithmetic: a center and a radius that
//! bounds the distance to the true value. Every step truncates its center
//! and adds a bound on its error to the radius, so the ball holds the true
//! value. `e^x` of a nonzero rational `x` and `ln x` of a positive rational
//! `x` other than 1 are irrational (Lindemann), so the true value never lies
//! on a point of a grid. The module `exact` gives the values in base 2 and
//! 10 that are rational. When every value of the ball has one truncation to
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
//! A ball cannot decide a value that lies closer to a point of the grid than
//! its radius. So four kinds of argument take their truncation without a
//! ball:
//!
//! - An argument below `2^-(p + 5)`, or `10^-(p + 3)`, gives `b^x` within
//!   one grid step of 1, `e^x - 1` and `ln(1 + x)` within one grid step of
//!   `x`. The side of the point follows from the sign of the argument.
//! - A negative argument whose `b^x` lies below one unit of the truncation
//!   of 1 gives `b^x - 1` just above -1.
//! - An argument `b^k` at or above `2^(p + 5)`, or `10^(p + 3)`, gives
//!   `log_b(1 + x)` just above the integer `k`.
//! - An argument far past the range of the format gives an overflow or an
//!   underflow.
//!
//! `compound(x, n) = e^(n ln(1 + x))` has a rational result, which can lie
//! on a grid. Its module computes those results without a ball.

mod ball;
mod compound;
mod constants;
mod exact;
mod series;

pub(crate) use self::compound::compound;

use core::cmp::Ordering;

use self::ball::{Ball, truncation};
use self::constants::{ln2, ln10};
use crate::exact::Unrounded;
use crate::limbs::{self, Limbs, Widen};
use crate::unpacked::Unpacked;

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

/// An exponential or a logarithm of IEEE 754-2019 section 9.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transcendental {
    /// `exp`: `e^x`.
    Exp,
    /// `expm1`: `e^x - 1`.
    ExpM1,
    /// `exp2`: `2^x`.
    Exp2,
    /// `exp2m1`: `2^x - 1`.
    Exp2M1,
    /// `exp10`: `10^x`.
    Exp10,
    /// `exp10m1`: `10^x - 1`.
    Exp10M1,
    /// `log`: `ln x`.
    Log,
    /// `log2`: `log_2 x`.
    Log2,
    /// `log10`: `log_10 x`.
    Log10,
    /// `logp1`: `ln(1 + x)`.
    LogP1,
    /// `log2p1`: `log_2(1 + x)`.
    Log2P1,
    /// `log10p1`: `log_10(1 + x)`.
    Log10P1,
}

impl Transcendental {
    /// Returns `true` for an exponential.
    fn is_exponential(self) -> bool {
        matches!(
            self,
            Self::Exp | Self::ExpM1 | Self::Exp2 | Self::Exp2M1 | Self::Exp10 | Self::Exp10M1
        )
    }

    /// Returns `true` for a form shifted by one: `b^x - 1` or `log_b(1 + x)`.
    fn is_shifted(self) -> bool {
        matches!(
            self,
            Self::ExpM1 | Self::Exp2M1 | Self::Exp10M1 | Self::LogP1 | Self::Log2P1 | Self::Log10P1
        )
    }

    /// Returns the base.
    fn base(self) -> Base {
        match self {
            Self::Exp | Self::ExpM1 | Self::Log | Self::LogP1 => Base::E,
            Self::Exp2 | Self::Exp2M1 | Self::Log2 | Self::Log2P1 => Base::Two,
            Self::Exp10 | Self::Exp10M1 | Self::Log10 | Self::Log10P1 => Base::Ten,
        }
    }
}

/// The base of an exponential or a logarithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Base {
    /// e.
    E,
    /// 2.
    Two,
    /// 10.
    Ten,
}

impl Base {
    /// Returns `true` when the base is the radix of the format.
    fn is_radix(self, radix: Radix) -> bool {
        matches!(
            (self, radix),
            (Self::Two, Radix::Binary) | (Self::Ten, Radix::Decimal)
        )
    }

    /// Returns `y ln b`.
    fn times<W: Widen>(self, y: &Ball<W>) -> Ball<W> {
        match self {
            Self::E => *y,
            Self::Two => y.mul(&ln2()),
            Self::Ten => y.mul(&ln10()),
        }
    }

    /// Returns `y / ln b`.
    fn over<W: Widen>(self, y: &Ball<W>) -> Option<Ball<W>> {
        match self {
            Self::E => Some(*y),
            Self::Two => y.div(&ln2()),
            Self::Ten => y.div(&ln10()),
        }
    }

    /// Returns a rational upper bound `(numerator, denominator)` on
    /// `log_b RADIX`, above it by less than 10^-4 of its value.
    fn log_of(self, radix: Radix) -> (u64, u64) {
        match (self, radix) {
            (Self::E, Radix::Binary) => (6_932, 10_000),
            (Self::Two, Radix::Binary) | (Self::Ten, Radix::Decimal) => (1, 1),
            (Self::Ten, Radix::Binary) => (30_103, 100_000),
            (Self::E, Radix::Decimal) => (23_026, 10_000),
            (Self::Two, Radix::Decimal) => (33_220, 10_000),
        }
    }
}

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

impl Target {
    /// Returns the digits of the truncation: `p + 3` bits or `p + 2` digits.
    fn kept(&self) -> u32 {
        match self.radix {
            Radix::Binary => self.precision + 3,
            Radix::Decimal => self.precision + 2,
        }
    }

    /// Returns `t` such that an argument below `RADIX^-t` is tiny: `b^x`
    /// lies within one grid step of 1, and `e^x - 1` and `ln(1 + x)` within
    /// one grid step of `x`. `|b^x - 1|` is below `|x| ln 10 (1 + |x|)`, and
    /// `ln 10 < 4`.
    fn tiny(&self) -> i64 {
        let precision = i64::from(self.precision);
        match self.radix {
            Radix::Binary => precision + 5,
            Radix::Decimal => precision + 3,
        }
    }
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

/// The result of a function at an operand that is not a NaN, by the special
/// cases of IEEE 754-2019 section 9.2.1.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Special<L> {
    /// A finite argument of the domain of the function, which [`evaluate`]
    /// takes.
    Evaluate(Argument<L>),
    /// The exact result 1 with the sign `negative`.
    One {
        /// The sign.
        negative: bool,
    },
    /// The exact result zero with the sign `negative`.
    Zero {
        /// The sign.
        negative: bool,
    },
    /// An infinity with the sign `negative`.
    Infinity {
        /// The sign.
        negative: bool,
    },
    /// -inf at a pole, which signals divide-by-zero.
    Pole,
    /// An argument outside the domain, which gives the default NaN and
    /// signals invalid.
    Invalid,
}

/// Returns the result of `function` at an operand that is not a NaN, or the
/// argument to evaluate.
///
/// A shifted form keeps the zero of its argument. `b^-inf - 1` is -1. A
/// logarithm of 1 is +0, of a zero is a pole, and of a negative argument is
/// invalid. A shifted logarithm of -1 is a pole, and of an argument below -1
/// is invalid.
pub(crate) fn special<L: Limbs>(
    function: Transcendental,
    x: Unpacked<L>,
    radix: Radix,
) -> Special<L> {
    let exponential = function.is_exponential();
    let shifted = function.is_shifted();
    match x {
        Unpacked::Zero { negative, .. } => match (exponential, shifted) {
            (_, true) => Special::Zero { negative },
            (true, false) => Special::One { negative: false },
            (false, false) => Special::Pole,
        },
        Unpacked::Infinity { negative: false } => Special::Infinity { negative: false },
        Unpacked::Infinity { negative: true } => match (exponential, shifted) {
            (true, false) => Special::Zero { negative: false },
            (true, true) => Special::One { negative: true },
            (false, _) => Special::Invalid,
        },
        Unpacked::Finite {
            negative,
            exponent,
            significand,
        } => {
            let x = Argument {
                negative,
                exponent,
                significand,
            };
            if exponential {
                return Special::Evaluate(x);
            }
            match (shifted, negative, compare_one(&x, radix)) {
                (false, true, _) | (true, true, Ordering::Greater) => Special::Invalid,
                (false, false, Ordering::Equal) => Special::Zero { negative: false },
                (true, true, Ordering::Equal) => Special::Pole,
                _ => Special::Evaluate(x),
            }
        }
        Unpacked::Nan { .. } | Unpacked::Unsupported => {
            unreachable!("the NaN rule handles every NaN and unsupported operand")
        }
    }
}

/// Returns `function` of an argument that [`special`] gives, truncated for
/// the rounding routine of the format: exact with the sticky bit clear when
/// the value is rational, and to `p + 3` bits or `p + 2` digits with the
/// sticky bit set otherwise.
pub(crate) fn evaluate<L: Elementary>(
    function: Transcendental,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    let base = function.base();
    if function.is_exponential() {
        exponential::<L>(x, base, function.is_shifted(), target)
    } else {
        logarithm::<L>(x, base, function.is_shifted(), target)
    }
}

/// Returns `b^x`, or `b^x - 1` when `minus_one` is set.
fn exponential<L: Elementary>(
    x: &Argument<L>,
    base: Base,
    minus_one: bool,
    target: &Target,
) -> Unrounded<L::Second> {
    // `RADIX^leading <= |x| < RADIX^(leading + 1)`.
    let leading = leading(x, target.radix);
    if leading < -target.tiny() {
        if !minus_one {
            return near_one(x.negative, target);
        }
        if base == Base::E {
            // `e^x - 1` lies above `x` by less than `x^2`.
            return beside(
                x.negative,
                x.significand.resize(),
                i64::from(x.exponent),
                x.negative,
                target,
            );
        }
        // `b^x - 1` is near `x ln b`, which lies on no grid.
    }
    if minus_one && x.negative && exact::at_least(x, target.radix, near_minus_one(base, target)) {
        // `b^x` lies below one unit of the truncation of 1.
        return Unrounded {
            negative: true,
            ..near_one(true, target)
        };
    }
    // Past `4 * range`, `b^x` overflows or underflows the range, because
    // `2^4 > 10`, and `10^x` does past `range`. Below that bound, `|x ln b|`
    // stays below `8 range`, as for `e^x`, far inside an `i32`.
    let factor = if base == Base::Ten { 1 } else { 4 };
    let limit = factor * u64::from(target.range);
    let huge = match target.radix {
        Radix::Binary => leading >= i64::from(64 - limit.leading_zeros()),
        Radix::Decimal => leading >= i64::from(digit_count(&[limit])),
    };
    if huge {
        return out_of_range(!x.negative, target);
    }
    if base != Base::E
        && let Some(n) = exact::integer(x, target.radix)
        && let Some(value) = exact::power_value(base, n, minus_one, target)
    {
        return value;
    }
    let function = Exponential {
        x: *x,
        radix: target.radix,
        base,
        minus_one,
    };
    ziv::<L, _>(&function, target)
}

/// Returns the least `|x|` from which `b^-|x|` lies below one unit of the
/// truncation of 1 below it: `2^-(p + 3)`, or `10^-(p + 2)`. Below that
/// bound, `b^-|x|` is above a tenth of the unit, so a ball decides the
/// truncation of `b^x - 1`.
fn near_minus_one(base: Base, target: &Target) -> u64 {
    let (numerator, denominator) = base.log_of(target.radix);
    u64::from(target.kept()) * numerator / denominator + 1
}

/// Returns `log_b x`, or `log_b(1 + x)` when `plus_one` is set.
fn logarithm<L: Elementary>(
    x: &Argument<L>,
    base: Base,
    plus_one: bool,
    target: &Target,
) -> Unrounded<L::Second> {
    if plus_one && base == Base::E && leading(x, target.radix) < -target.tiny() {
        // `ln(1 + x)` lies below `x` by less than `x^2`.
        return beside(
            x.negative,
            x.significand.resize(),
            i64::from(x.exponent),
            !x.negative,
            target,
        );
    }
    if base != Base::E
        && let Some(k) = exact::logarithm::<L, L::Second>(x, base, plus_one, target.radix)
    {
        return exact::integer_value(k);
    }
    if plus_one
        && base != Base::E
        && leading(x, target.radix) >= target.tiny()
        && let Some(k) = exact::logarithm::<L, L::Second>(x, base, false, target.radix)
    {
        // `log_b(1 + x)` lies above the integer `k = log_b x` by less than
        // `1 / (x ln 2)`, below one unit of the truncation of `k`.
        let k = L::Second::ZERO.with_limb(0, k.unsigned_abs());
        return beside(false, k, 0, false, target);
    }
    let function = Logarithm {
        x: *x,
        radix: target.radix,
        base,
        plus_one,
    };
    ziv::<L, _>(&function, target)
}

/// A function that gives the ball of its value at any working width.
trait Function {
    /// Returns the ball of the value at the width `W`, or `None` when the
    /// width cannot bound it.
    fn ball<W: Widen>(&self) -> Option<Ball<W>>;
}

/// The function `b^x`, or `b^x - 1`.
struct Exponential<L> {
    /// The argument.
    x: Argument<L>,
    /// The radix of the argument.
    radix: Radix,
    /// The base.
    base: Base,
    /// `true` for `b^x - 1`.
    minus_one: bool,
}

/// The function `log_b x`, or `log_b(1 + x)`.
struct Logarithm<L> {
    /// The argument.
    x: Argument<L>,
    /// The radix of the argument.
    radix: Radix,
    /// The base.
    base: Base,
    /// `true` for `log_b(1 + x)`.
    plus_one: bool,
}

impl<L: Limbs> Function for Exponential<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let y = self.base.times(&argument::<L, W>(&self.x, self.radix)?);
        Some(if self.minus_one {
            series::exp_minus_one(&y)
        } else {
            series::exp(&y)
        })
    }
}

impl<L: Limbs> Function for Logarithm<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let log = if self.plus_one {
            series::log_one_plus(&argument::<L, W>(&self.x, self.radix)?)?
        } else {
            natural_log::<L, W>(&self.x, self.radix)?
        };
        self.base.over(&log)
    }
}

/// Returns the ball of `ln x` of a positive argument at the width `W`. A
/// decimal argument `c 10^q` gives `ln c + q ln 10`.
fn natural_log<L: Limbs, W: Widen>(x: &Argument<L>, radix: Radix) -> Option<Ball<W>> {
    let exponent = i64::from(x.exponent);
    match radix {
        Radix::Binary => series::log(&Ball::new(false, x.significand, exponent)),
        Radix::Decimal => series::log_decimal(&Ball::new(false, x.significand, 0), exponent),
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

/// Returns the truncation of `e^x` for an argument below the tiny bound:
/// just above 1 for `x > 0`, and just below it for `x < 0`.
fn near_one<L: Limbs>(negative: bool, target: &Target) -> Unrounded<L> {
    beside(false, L::ZERO.with_bit(0), 0, negative, target)
}

/// Returns the truncation of a value beside the nonzero number
/// `significand * RADIX^exponent`: closer to it than one unit of its
/// truncation to `p + 3` bits, or `p + 2` digits, toward zero when
/// `toward_zero` is set, and away from zero otherwise.
///
/// The truncation of the number itself is the truncation away from zero.
/// Toward zero, it is also the truncation when the number lies off the grid.
/// For a number on the grid, it is one unit below the number, or all ones,
/// or nines, one digit further down when the number is a power of the
/// radix.
fn beside<L: Limbs>(
    negative: bool,
    significand: L,
    exponent: i64,
    toward_zero: bool,
    target: &Target,
) -> Unrounded<L> {
    let kept = target.kept();
    let length = match target.radix {
        Radix::Binary => significand.bit_length(),
        Radix::Decimal => digit_count(&significand),
    };
    let (scaled, lowest, on_grid) = if let Some(shift) = kept.checked_sub(length) {
        let scaled = match target.radix {
            Radix::Binary => significand.shl(shift),
            Radix::Decimal => limbs::multiply_fit(significand, power(Radix::Decimal, shift)),
        };
        (scaled, exponent - i64::from(shift), true)
    } else {
        let shift = length - kept;
        let (scaled, dropped) = match target.radix {
            Radix::Binary => (significand.shr(shift), significand.any_below(shift)),
            Radix::Decimal => {
                let (quotient, rest) = limbs::divide(significand, power(Radix::Decimal, shift));
                (quotient, !rest.is_zero())
            }
        };
        (scaled, exponent + i64::from(shift), !dropped)
    };
    let one = L::ZERO.with_bit(0);
    let (significand, lowest) = if !toward_zero || !on_grid {
        (scaled, lowest)
    } else if scaled == power(target.radix, kept - 1) {
        (power::<L>(target.radix, kept).sub(one), lowest - 1)
    } else {
        (scaled.sub(one), lowest)
    };
    Unrounded {
        negative,
        exponent: i32::try_from(lowest).expect("the exponent of an operand fits an i32"),
        significand,
        sticky: true,
    }
}

/// Returns a value past the range of the format: above `RADIX^range` for an
/// overflow, or below `RADIX^-range` for an underflow.
fn out_of_range<L: Limbs>(overflow: bool, target: &Target) -> Unrounded<L> {
    let digits = target.kept() - 1;
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

/// Compares the magnitude of an argument with 1.
pub(crate) fn compare_one<L: Limbs>(x: &Argument<L>, radix: Radix) -> Ordering {
    match radix {
        Radix::Binary => {
            let top = x.significand.bit_length() - 1;
            match (i64::from(x.exponent) + i64::from(top)).cmp(&0) {
                // `1 <= |x| < 2`: `|x|` is 1 when its significand has one bit.
                Ordering::Equal if x.significand.any_below(top) => Ordering::Greater,
                order => order,
            }
        }
        Radix::Decimal => {
            // A coefficient has at most 34 digits, below 2^113. A positive
            // exponent puts a coefficient at or above 10.
            let coefficient = limbs::to_u128(&x.significand);
            if x.exponent >= 0 {
                return if x.exponent == 0 {
                    coefficient.cmp(&1)
                } else {
                    Ordering::Greater
                };
            }
            10_u128
                .checked_pow(x.exponent.unsigned_abs())
                .map_or(Ordering::Less, |unit| coefficient.cmp(&unit))
        }
    }
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
