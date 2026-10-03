//! An oracle for `exp` and `log` of IEEE 754-2019 section 9.2 on the binary
//! and the decimal formats.
//!
//! MPFR evaluates the function with directed rounding, at a working precision
//! that doubles until the bounds below and above decide the truncation of the
//! value to `p + 3` bits, or to `p + 2` digits. `e^x` of a nonzero rational
//! `x` and `ln x` of a positive rational other than 1 are irrational, so
//! that truncation with a sticky bit rounds as the true value does. For a
//! binary format, [`crate::mpfr::round`] rounds it. For a decimal format,
//! [`decimal::to_decimal`] rounds a binary value strictly inside the
//! truncation interval, which rounds as the true value does. A decimal
//! argument `C * 10^q` enters MPFR as two bounds, and both functions
//! increase, so the bounds of the arguments give bounds of the results.
//!
//! An argument of `exp` at or above `2^24` in magnitude, or `2^16` for a
//! decimal format, gives a result past the range of every format, which the
//! sign of the argument decides without MPFR. IEEE 754-2019 section 9.2.1
//! gives the special cases, and floaty's operand rule gives the NaN and
//! unsupported cases. A decimal result that is exact, `e^0 = 1`, `ln 1 = 0`,
//! or `e^-inf = 0`, has the exponent 0, by the rule of floaty.

use floaty::{Decoded, Env, Flags};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer, Rational};

use super::{Outcome, default_nan, special_operands};
use crate::mpfr::decimal::{self, DecimalFormat, DecimalValue};
use crate::mpfr::{self, Format, Input, Nan, Operand, Read, select_nan};

/// A function of the oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Function {
    /// `e^x`.
    Exp,
    /// The natural logarithm `ln x`.
    Log,
}

impl Function {
    /// Both functions.
    pub const ALL: [Self; 2] = [Self::Exp, Self::Log];
}

/// The exponent of MPFR at or past which an argument of `exp` overflows or
/// underflows every binary format: `|x| >= 2^24`. binary512, the widest
/// format, overflows at about `2^21.5`.
const OUT_OF_RANGE: i32 = 25;

/// The exponent of MPFR at or past which an argument of `exp` overflows or
/// underflows every decimal format: `|x| >= 2^16`. decimal128 underflows to
/// zero past about `-14_300`.
const DECIMAL_OUT_OF_RANGE: i32 = 17;

/// The working precision past which the oracle stops: no result of a format
/// needs anything near it.
const PRECISION_LIMIT: u32 = 1 << 20;

/// Returns the expected result of `function` on an operand, and the flags.
#[must_use]
pub fn expected<const N: usize>(
    function: Function,
    x: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [x.read(env, &mut flags)];
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    let [Read::Number(value)] = &reads else {
        unreachable!("every special operand has a result");
    };
    let (outcome, more) = match function {
        Function::Exp => exp(value, format, env),
        Function::Log => log(value, format, env),
    };
    (outcome, flags | more)
}

/// Returns the expected `e^x` of a number.
fn exp(x: &BigFloat, format: &Format, env: &Env) -> (Outcome, Flags) {
    if x.is_zero() {
        return (Outcome::Finite(BigFloat::with_val(1, 1)), Flags::NONE);
    }
    if x.is_infinite() {
        let value = if x.is_sign_negative() {
            format.zero(false)
        } else {
            format.infinity(false, env)
        };
        return (Outcome::from_value(value), Flags::NONE);
    }
    let input = if x.get_exp().is_some_and(|exponent| exponent >= OUT_OF_RANGE) {
        // A value far past the range: `2^(p + 2)` times `2^(±2^28)`, with the
        // sticky bit.
        Input {
            negative: false,
            exponent: if x.is_sign_negative() {
                -(1 << 28)
            } else {
                1 << 28
            },
            significand: Integer::from(1) << (format.precision + 2),
            sticky: true,
        }
    } else {
        truncation(format.precision, |precision, round| {
            BigFloat::with_val_round(precision, x.exp_ref(), round).0
        })
    };
    let (value, flags) = mpfr::round(&input, format, env);
    (Outcome::from_value(value), flags)
}

/// Returns the expected `ln x` of a number.
fn log(x: &BigFloat, format: &Format, env: &Env) -> (Outcome, Flags) {
    if x.is_zero() {
        let value = format.infinity(true, env);
        return (Outcome::from_value(value), Flags::DIVIDE_BY_ZERO);
    }
    if x.is_sign_negative() {
        return (default_nan(format, env), Flags::INVALID);
    }
    if x.is_infinite() {
        return (
            Outcome::from_value(format.infinity(false, env)),
            Flags::NONE,
        );
    }
    if *x == 1 {
        return (Outcome::from_value(format.zero(false)), Flags::NONE);
    }
    let input = truncation(format.precision, |precision, round| {
        BigFloat::with_val_round(precision, x.ln_ref(), round).0
    });
    let (value, flags) = mpfr::round(&input, format, env);
    (Outcome::from_value(value), flags)
}

/// Returns the truncation of an irrational value to `precision + 3` bits,
/// with the sticky bit set. `evaluate` gives the value rounded at a working
/// precision in a direction. The working precision doubles until the bounds
/// decide the truncation.
///
/// # Panics
///
/// Panics when the working precision reaches [`PRECISION_LIMIT`].
fn truncation(precision: u32, evaluate: impl Fn(u32, Round) -> BigFloat) -> Input {
    let mut working = precision + 64;
    loop {
        let (low, high) = (evaluate(working, Round::Down), evaluate(working, Round::Up));
        if let Some(input) = common_truncation(&low, &high, precision) {
            return input;
        }
        working *= 2;
        assert!(
            working < PRECISION_LIMIT,
            "the bounds decide the truncation"
        );
    }
}

/// Returns the truncation to `precision + 3` bits of every value strictly
/// between two finite bounds of one sign, or `None` when the bounds do not
/// decide it. A bound can be a power of two while the value is in the
/// binade below, as `e^x` is for a tiny negative `x`.
fn common_truncation(low: &BigFloat, high: &BigFloat, precision: u32) -> Option<Input> {
    let negative = low.is_sign_negative();
    if negative != high.is_sign_negative() || low.is_zero() || high.is_zero() {
        return None;
    }
    let (small, large) = if negative { (high, low) } else { (low, high) };
    // The value is in the binade of `small` or above it. The magnitude of
    // `small` lies in `[2^top, 2^(top + 1))`.
    let top = small.get_exp()? - 1;
    let lowest = top - i32::try_from(precision + 2).ok()?;
    let below = (small.clone().abs() >> lowest).floor().to_integer()?;
    let above = (large.clone().abs() >> lowest).ceil().to_integer()? - 1u32;
    (below == above).then_some(Input {
        negative,
        exponent: lowest,
        significand: below,
        sticky: true,
    })
}

/// A decimal operand as an operation reads it, after denormals-are-zero.
enum DecimalRead {
    /// A zero.
    Zero,
    /// An infinity with its sign.
    Infinity(bool),
    /// A nonzero finite value.
    Finite(Rational),
}

/// Returns the expected result of `function` on a decimal operand, and the
/// flags.
///
/// # Panics
///
/// Panics for an unsupported encoding, which a decimal format does not have.
#[must_use]
pub fn expected_decimal(
    function: Function,
    x: &Operand<2>,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let mut flags = if x.subnormal {
        Flags::DENORMAL_INPUT
    } else {
        Flags::NONE
    };
    let read = match x.decoded {
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => {
            let offered = Nan {
                negative,
                signaling,
                payload: Integer::from_digits(&payload, Order::Lsf),
            };
            let nan = select_nan(&[offered], env);
            if signaling {
                flags |= Flags::INVALID;
            }
            let value = DecimalValue::Nan {
                negative: nan.negative,
                payload: nan.payload,
            };
            return (value, flags);
        }
        Decoded::Zero { .. } => DecimalRead::Zero,
        Decoded::Finite { .. } if x.subnormal && env.denormals_are_zero => DecimalRead::Zero,
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let coefficient = Integer::from_digits(&significand, Order::Lsf);
            let power = Integer::from(Integer::u_pow_u(10, exponent.unsigned_abs()));
            let magnitude = if exponent >= 0 {
                Rational::from(coefficient * power)
            } else {
                Rational::from((coefficient, power))
            };
            DecimalRead::Finite(if negative { -magnitude } else { magnitude })
        }
        Decoded::Infinity { negative } => DecimalRead::Infinity(negative),
        Decoded::Unsupported => panic!("a decimal format has no unsupported encoding"),
    };
    let (value, more) = match function {
        Function::Exp => decimal_exp(read, format, env),
        Function::Log => decimal_log(read, format, env),
    };
    (value, flags | more)
}

/// Returns the exact decimal result `coefficient * 10^0`.
fn exact_decimal(coefficient: u32) -> DecimalValue {
    if coefficient == 0 {
        DecimalValue::Zero {
            negative: false,
            exponent: 0,
        }
    } else {
        DecimalValue::Finite {
            negative: false,
            coefficient: Integer::from(coefficient),
            exponent: 0,
        }
    }
}

/// Returns the expected `e^x` of a decimal number.
fn decimal_exp(x: DecimalRead, format: DecimalFormat, env: &Env) -> (DecimalValue, Flags) {
    let x = match x {
        DecimalRead::Zero => return (exact_decimal(1), Flags::NONE),
        DecimalRead::Infinity(false) => {
            return (DecimalValue::Infinity { negative: false }, Flags::NONE);
        }
        DecimalRead::Infinity(true) => return (exact_decimal(0), Flags::NONE),
        DecimalRead::Finite(x) => x,
    };
    let bound = BigFloat::with_val(64, &x);
    let value = if bound
        .get_exp()
        .is_some_and(|exponent| exponent >= DECIMAL_OUT_OF_RANGE)
    {
        // A value far past the range of every decimal format.
        let scale = if x < 0 { -(1 << 16) } else { 1 << 16 };
        BigFloat::with_val(2, 1) << scale
    } else {
        decimal_truncation(format.precision, |precision, round| {
            let argument = BigFloat::with_val_round(precision, &x, round).0;
            BigFloat::with_val_round(precision, argument.exp_ref(), round).0
        })
    };
    decimal::to_decimal(&value, format, env)
}

/// Returns the expected `ln x` of a decimal number.
fn decimal_log(x: DecimalRead, format: DecimalFormat, env: &Env) -> (DecimalValue, Flags) {
    let nan = DecimalValue::Nan {
        negative: env.nan.default_negative,
        payload: Integer::ZERO,
    };
    let x = match x {
        DecimalRead::Zero => {
            return (
                DecimalValue::Infinity { negative: true },
                Flags::DIVIDE_BY_ZERO,
            );
        }
        DecimalRead::Infinity(false) => {
            return (DecimalValue::Infinity { negative: false }, Flags::NONE);
        }
        DecimalRead::Infinity(true) => return (nan, Flags::INVALID),
        DecimalRead::Finite(x) if x < 0 => return (nan, Flags::INVALID),
        DecimalRead::Finite(x) if x == 1 => return (exact_decimal(0), Flags::NONE),
        DecimalRead::Finite(x) => x,
    };
    let value = decimal_truncation(format.precision, |precision, round| {
        let argument = BigFloat::with_val_round(precision, &x, round).0;
        BigFloat::with_val_round(precision, argument.ln_ref(), round).0
    });
    decimal::to_decimal(&value, format, env)
}

/// Returns a binary value with the truncation to `precision + 2` digits of
/// an irrational value, strictly inside the truncation interval. `evaluate`
/// gives a bound of the value at a working precision in a direction. The
/// working precision doubles until the bounds decide the truncation.
///
/// # Panics
///
/// Panics when the working precision reaches [`PRECISION_LIMIT`].
fn decimal_truncation(precision: u32, evaluate: impl Fn(u32, Round) -> BigFloat) -> BigFloat {
    let mut working = 4 * precision + 64;
    loop {
        let (low, high) = (evaluate(working, Round::Down), evaluate(working, Round::Up));
        if let Some(value) = inside_truncation(&low, &high, precision) {
            return value;
        }
        working *= 2;
        assert!(
            working < PRECISION_LIMIT,
            "the bounds decide the truncation"
        );
    }
}

/// Returns the midpoint of two finite bounds of one sign when every value
/// strictly between them has one truncation to `precision + 2` digits, or
/// `None`. The midpoint has that truncation too.
fn inside_truncation(low: &BigFloat, high: &BigFloat, precision: u32) -> Option<BigFloat> {
    let negative = low.is_sign_negative();
    if negative != high.is_sign_negative() || low.is_zero() || high.is_zero() {
        return None;
    }
    let (small, large) = if negative { (high, low) } else { (low, high) };
    // MPFR writes `|small|` as `0.d * 10^exponent`, so the value is in the
    // decade `[10^top, 10^(top + 1))` of `small` or above it.
    let (_, _, exponent) = small.to_sign_string_exp_round(10, Some(1), Round::Zero);
    let top = i64::from(exponent?) - 1;
    let lowest = top - i64::from(precision) - 1;
    let unit = Integer::from(Integer::u_pow_u(
        10,
        u32::try_from(lowest.unsigned_abs()).ok()?,
    ));
    let scaled = |bound: &BigFloat| {
        let (numerator, denominator) = bound.to_rational()?.abs().into_numer_denom();
        Some(if lowest >= 0 {
            (numerator, denominator * &unit)
        } else {
            (numerator * &unit, denominator)
        })
    };
    let (numerator, denominator) = scaled(small)?;
    let below = numerator.div_rem_floor(denominator).0;
    let (numerator, denominator) = scaled(large)?;
    let above = numerator.div_rem_ceil(denominator).0 - 1u32;
    if below != above {
        return None;
    }
    // The bounds lie in one decade, at most four binades apart.
    let bits = small.prec().max(large.prec()) + 8;
    Some(BigFloat::with_val(bits, small + large) >> 1u32)
}
