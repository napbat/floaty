//! The oracle of `atan2` and `atan2Pi` of IEEE 754-2019 section 9.2 on the
//! binary and the decimal formats.
//!
//! The oracle states the special cases of section 9.2.1 itself: an exact
//! zero, and for `atan2Pi` the exact multiples of 1/4, at zeros, infinities,
//! and `|y| = |x|`. MPFR's `atan2` and `atan2pi` give every other value.
//! Inside a quadrant, `atan2` is monotonic in each argument, so the least
//! and the greatest value at the four corners of the bounds of two decimal
//! arguments bound the value.
//!
//! MPFR's `atan2pi` takes a working precision near `|log2 t|` bits for a
//! quotient `t = |y| / |x|` far from 1 where the value lies near 1/2 or 1.
//! There the oracle takes the value from a stand-in on the same side of
//! each point of the grid: for two binary arguments past `2^(±(p + 11))`,
//! and for two decimal arguments past `2^(±(4p + 70))`, where MPFR bounds
//! of a decimal argument would need such a precision for `atan2` too. The
//! module `decimal` takes its stand-ins in the same way.

use floaty::format::Standard;
use floaty::{Env, Flags, Float};
use rug::float::Round;
use rug::ops::PowAssignRound;
use rug::{Float as BigFloat, Integer, Rational};

use super::decimal::{DecimalRead, decimal_nan, decimal_truncation, read_decimal};
use super::{Shape, power, truncation};
use crate::mpfr::decimal::{self, DecimalFormat, DecimalValue};
use crate::mpfr::{self, Format, Input, Operand, Read};
use crate::operations::{Outcome, special_operands};

/// A function of two arguments of the oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bivariate {
    /// `atan2(y, x)`.
    Atan2,
    /// `atan2(y, x) / pi`.
    Atan2Pi,
    /// `pow(x, y)`.
    Pow,
    /// `powr(x, y)`.
    Powr,
}

impl Bivariate {
    /// `atan2` and `atan2Pi`.
    pub const ATAN2: [Self; 2] = [Self::Atan2, Self::Atan2Pi];

    /// `pow` and `powr`.
    pub const POWER: [Self; 2] = [Self::Pow, Self::Powr];

    /// Returns floaty's result of the function on its first and second
    /// operands in `env`, `y` and `x` for `atan2`, and the flags.
    #[must_use]
    pub fn floaty<S: Standard<W>, const W: usize>(
        self,
        first: Float<S, W>,
        second: Float<S, W>,
        env: Env,
    ) -> (Float<S, W>, Flags) {
        match self {
            Self::Atan2 => first.atan2_with(second, env),
            Self::Atan2Pi => first.atan2_pi_with(second, env),
            Self::Pow => first.pow_with(second, env),
            Self::Powr => first.powr_with(second, env),
        }
    }

    /// Returns `true` for `pow` and `powr`.
    fn is_power(self) -> bool {
        matches!(self, Self::Pow | Self::Powr)
    }

    /// Returns the value at the first and the second operand that MPFR
    /// rounds to `precision` bits in the direction `round`. `first` must have
    /// at most `precision` bits.
    pub(super) fn mpfr(
        self,
        first: &BigFloat,
        second: &BigFloat,
        precision: u32,
        round: Round,
    ) -> BigFloat {
        let mut value = BigFloat::with_val(precision, first);
        let _ = match self {
            Self::Atan2 => value.atan2_round(second, round),
            Self::Atan2Pi => value.atan2_pi_round(second, round),
            Self::Pow | Self::Powr => value.pow_assign_round(second, round),
        };
        value
    }

    /// Returns the result of the special cases of IEEE 754-2019 section
    /// 9.2.1, or [`Special::Evaluate`]. `equal` is set for two finite nonzero
    /// arguments of one magnitude.
    fn special(self, y: &Shape, x: &Shape, equal: bool) -> Special {
        let (y_negative, x_negative) = (y.is_negative(), x.is_negative());
        let quarters = |n: i8| match self {
            Self::Atan2Pi => Special::Quarters(if y_negative { -n } else { n }),
            Self::Atan2 | Self::Pow | Self::Powr => Special::Evaluate,
        };
        match (y, x) {
            (Shape::Zero(_), _) | (Shape::Finite { .. }, Shape::Infinity(_)) => {
                if x_negative {
                    quarters(4)
                } else {
                    Special::Zero(y_negative)
                }
            }
            (Shape::Infinity(_), Shape::Infinity(_)) => quarters(if x_negative { 3 } else { 1 }),
            (Shape::Infinity(_), _) | (Shape::Finite { .. }, Shape::Zero(_)) => quarters(2),
            (Shape::Finite { .. }, Shape::Finite { .. }) if equal => {
                quarters(if x_negative { 3 } else { 1 })
            }
            (Shape::Finite { .. }, Shape::Finite { .. }) => Special::Evaluate,
        }
    }
}

/// The result of a special case of `atan2` or `atan2Pi`.
enum Special {
    /// No special case: MPFR gives the value.
    Evaluate,
    /// An exact zero with the sign.
    Zero(bool),
    /// The exact `n / 4` of `atan2Pi`, which rounds as any exact value does.
    Quarters(i8),
}

impl Shape {
    /// Returns `true` for an operand with the sign bit set.
    pub(super) fn is_negative(&self) -> bool {
        match *self {
            Self::Zero(negative) | Self::Infinity(negative) | Self::Finite { negative, .. } => {
                negative
            }
        }
    }

    /// Returns the shape of an operand of a binary format that is not a NaN.
    fn of(x: &BigFloat) -> Self {
        if x.is_zero() {
            Self::Zero(x.is_sign_negative())
        } else if x.is_infinite() {
            Self::Infinity(x.is_sign_negative())
        } else {
            Self::finite(x)
        }
    }

    /// Returns the zero or the infinity of the shape as a binary value, or
    /// `None` for a finite value.
    fn special_value(&self) -> Option<BigFloat> {
        let value = match *self {
            Self::Zero(false) => rug::float::Special::Zero,
            Self::Zero(true) => rug::float::Special::NegZero,
            Self::Infinity(false) => rug::float::Special::Infinity,
            Self::Infinity(true) => rug::float::Special::NegInfinity,
            Self::Finite { .. } => return None,
        };
        Some(BigFloat::with_val(2, value))
    }
}

/// Returns the expected result of `function` on two operands of a binary
/// format, `y` first, and the flags.
#[must_use]
pub fn expected_bivariate<const N: usize>(
    function: Bivariate,
    y: &Operand<N>,
    x: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    if function.is_power() {
        return power::expected(function, y, x, format, env);
    }
    let mut flags = Flags::NONE;
    let reads = [y.read(env, &mut flags), x.read(env, &mut flags)];
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    let [Read::Number(y), Read::Number(x)] = &reads else {
        unreachable!("every special operand has a result");
    };
    let equal = y.is_normal() && x.is_normal() && *y.as_abs() == *x.as_abs();
    let input = match function.special(&Shape::of(y), &Shape::of(x), equal) {
        Special::Zero(negative) => return (Outcome::from_value(format.zero(negative)), flags),
        Special::Quarters(n) => Input {
            negative: n < 0,
            exponent: -2,
            significand: Integer::from(n.unsigned_abs()),
            sticky: false,
        },
        Special::Evaluate => {
            binary_stand_in(function, y, x, format.precision).unwrap_or_else(|| {
                truncation(format.precision, |precision, round| {
                    function.mpfr(y, x, precision, round)
                })
            })
        }
    };
    let (value, more) = mpfr::round(&input, format, env);
    (Outcome::from_value(value), flags | more)
}

/// Returns a stand-in for `atan2Pi` at two finite nonzero binary arguments,
/// truncated to `p + 3` bits with the sticky bit, with the sign of `y`:
///
/// - with `x < 0` and `|y| / |x|` below `2^-(p + 11)`, `1 - atan(t) / pi`
///   lies within `t / 3` below 1, so it truncates to all ones below 1;
/// - with `|x| / |y|` below `2^-(p + 11)`, the value lies within
///   `|x| / (3 |y|)` of 1/2, below it for `x > 0`, where it truncates to all
///   ones below 1/2, and above it for `x < 0`, where it truncates to 1/2.
///
/// Every other pair returns `None`.
fn binary_stand_in(
    function: Bivariate,
    y: &BigFloat,
    x: &BigFloat,
    precision: u32,
) -> Option<Input> {
    if function != Bivariate::Atan2Pi {
        return None;
    }
    // `2^(spread - 1) < |y| / |x| < 2^(spread + 1)`.
    let spread = i64::from(y.get_exp()?) - i64::from(x.get_exp()?);
    let bound = i64::from(precision) + 12;
    let digits = i32::try_from(precision).ok()?;
    let ones = (Integer::from(1) << (precision + 3)) - 1u32;
    let (exponent, significand) = match (spread <= -bound, spread >= bound) {
        (true, _) if x.is_sign_negative() => (-(digits + 3), ones),
        (_, true) if x.is_sign_negative() => (-(digits + 3), Integer::from(1) << (precision + 2)),
        (_, true) => (-(digits + 4), ones),
        _ => return None,
    };
    Some(Input {
        negative: y.is_sign_negative(),
        exponent,
        significand,
        sticky: true,
    })
}

/// Returns the expected result of `function` on two operands of a decimal
/// format, `y` first, and the flags.
///
/// # Panics
///
/// Panics for an unsupported encoding, which a decimal format does not have.
#[must_use]
pub fn expected_decimal_bivariate(
    function: Bivariate,
    y: &Operand<2>,
    x: &Operand<2>,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    if function.is_power() {
        return power::expected_decimal(function, y, x, format, env);
    }
    let mut flags = Flags::NONE;
    let reads = [
        read_decimal(y, env, &mut flags),
        read_decimal(x, env, &mut flags),
    ];
    let nans: Vec<_> = reads
        .iter()
        .filter_map(|read| match read {
            DecimalRead::Nan(nan) => Some(nan.clone()),
            DecimalRead::Number(..) => None,
        })
        .collect();
    if !nans.is_empty() {
        let (value, more) = decimal_nan(&nans, env);
        return (value, flags | more);
    }
    let [
        DecimalRead::Number(y_shape, y),
        DecimalRead::Number(x_shape, x),
    ] = reads
    else {
        unreachable!("an operand that is not a NaN is a number");
    };
    let equal = matches!((&y, &x), (Some(y), Some(x)) if *y.as_abs() == *x.as_abs());
    let (value, more) = match function.special(&y_shape, &x_shape, equal) {
        Special::Zero(negative) => (
            DecimalValue::Zero {
                negative,
                exponent: 0,
            },
            Flags::NONE,
        ),
        Special::Quarters(n) => decimal::rational_to_decimal(&Rational::from((n, 4)), format, env),
        Special::Evaluate => {
            if let (Some(y), Some(x)) = (&y, &x)
                && let Some(value) = stand_in(function, y, x, format.precision)
            {
                decimal::rational_to_decimal(&value, format, env)
            } else {
                let operands = [(&y_shape, y.as_ref()), (&x_shape, x.as_ref())];
                let value = decimal_truncation(format.precision, |precision, round| {
                    corners(function, operands, precision, round)
                });
                decimal::to_decimal(&value, format, env)
            }
        }
    };
    (value, flags | more)
}

/// Returns the least value of `function` at the four corners of the bounds
/// of two decimal operands for `Round::Down`, and the greatest for
/// `Round::Up`. A zero or an infinity is its own bound.
fn corners(
    function: Bivariate,
    [y, x]: [(&Shape, Option<&Rational>); 2],
    precision: u32,
    round: Round,
) -> BigFloat {
    let bound = |(shape, value): (&Shape, Option<&Rational>), direction| {
        shape.special_value().unwrap_or_else(|| {
            let value = value.expect("a finite operand has a value");
            BigFloat::with_val_round(precision, value, direction).0
        })
    };
    [Round::Down, Round::Up]
        .into_iter()
        .flat_map(|y_round| {
            [Round::Down, Round::Up].map(|x_round| {
                function.mpfr(&bound(y, y_round), &bound(x, x_round), precision, round)
            })
        })
        .reduce(|chosen, value| {
            let better = match round {
                Round::Down => value < chosen,
                _ => value > chosen,
            };
            if better { value } else { chosen }
        })
        .expect("a box has four corners")
}

/// Returns a stand-in for the value at two finite nonzero decimal arguments
/// with `t = |y| / |x|` past `2^-(4p + 70)` or its reciprocal, where MPFR
/// bounds would need a working precision near `log2(1 / t)` bits.
///
/// - `atan t` of `atan2` with `x > 0` lies short of `t` by less than `t^3`,
///   as `t - t 2^-(4p + 70)` does, closer to `t` than one unit of `p + 2`
///   digits.
/// - `1 - atan(t) / pi` of `atan2Pi` with `x < 0` lies within `t / 3` below
///   1, as `1 - t / 4` does.
/// - `atan2Pi` of a large `t` lies within `1 / (3t)` of 1/2: below it for
///   `x > 0`, as `1/2 - 1 / (4t)` does, and above it for `x < 0`, as
///   `1/2 + 1 / (4t)` does.
///
/// Each takes the sign of `y`. Every other value returns `None`.
fn stand_in(function: Bivariate, y: &Rational, x: &Rational, precision: u32) -> Option<Rational> {
    let tiny = Rational::from((1, Integer::from(1) << (4 * precision + 70)));
    let t = Rational::from(y.abs_ref()) / Rational::from(x.abs_ref());
    let negative_x = *x < 0;
    let magnitude = match function {
        Bivariate::Atan2 if !negative_x && t < tiny => t.clone() - t * &tiny,
        Bivariate::Atan2Pi if negative_x && t < tiny => 1u32 - t / 4u32,
        Bivariate::Atan2Pi if t.clone().recip() < tiny => {
            let half = Rational::from((1, 2));
            let step = t.recip() / 4u32;
            if negative_x { half + step } else { half - step }
        }
        _ => return None,
    };
    Some(if *y < 0 { -magnitude } else { magnitude })
}
