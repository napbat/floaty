//! The oracle of the elementary functions on the decimal formats.
//!
//! A decimal argument `C * 10^q` enters MPFR as two bounds. A function that
//! increases takes its bounds at the bounds of the argument. Another takes
//! the least and the greatest MPFR value at the two bounds. GMP gives every
//! rational result exactly, and [`decimal::to_decimal`] rounds a binary
//! value strictly inside the truncation interval of an irrational one.

use std::cell::RefCell;
use std::collections::HashMap;

use floaty::{Decoded, Env, Flags};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer, Rational};

use super::{Function, PRECISION_LIMIT, Shape, Special, quarter, reverse};
use crate::mpfr::decimal::{self, DecimalFormat, DecimalValue};
use crate::mpfr::{Nan, Operand, select_nan};

thread_local! {
    /// The binary value inside the truncation interval of each decimal value
    /// that MPFR bounds, by function, argument, and precision. Each behavior
    /// and each encoding of a test reads the same value, and `sin` near
    /// `10^6144` takes MPFR about a millisecond at each bound.
    static TRUNCATIONS: RefCell<HashMap<(Function, Rational, u32), BigFloat>> =
        RefCell::new(HashMap::new());
}

/// Returns the bound `4 R` past which an argument of an exponential
/// overflows or underflows a decimal format, with
/// `R = max(emax, p - emin) + 3`: `b^|x| >= 2^(4R) = 16^R > 10^R`, and every
/// finite value of the format lies between `10^-R` and `10^R`.
fn decimal_out_of_range(format: DecimalFormat) -> i64 {
    let emin = 1 - i64::from(format.emax);
    let reach = i64::from(format.emax).max(i64::from(format.precision) - emin);
    4 * (reach + 3)
}

/// A decimal operand as the oracle reads it.
pub(super) enum DecimalRead {
    /// A NaN, which the NaN rule selects from.
    Nan(Nan),
    /// A number, with its value when it is finite and nonzero.
    Number(Shape, Option<Rational>),
}

/// Reads a decimal operand: a subnormal signals denormal input, and is a
/// zero under DAZ.
///
/// # Panics
///
/// Panics for an unsupported encoding, which a decimal format does not have.
pub(super) fn read_decimal(x: &Operand<2>, env: &Env, flags: &mut Flags) -> DecimalRead {
    if x.subnormal {
        *flags |= Flags::DENORMAL_INPUT;
    }
    match x.decoded {
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => DecimalRead::Nan(Nan {
            negative,
            signaling,
            payload: Integer::from_digits(&payload, Order::Lsf),
        }),
        Decoded::Zero { negative, .. } => DecimalRead::Number(Shape::Zero(negative), None),
        Decoded::Finite { negative, .. } if x.subnormal && env.denormals_are_zero => {
            DecimalRead::Number(Shape::Zero(negative), None)
        }
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
            let value = if negative { -magnitude } else { magnitude };
            DecimalRead::Number(Shape::finite(&value), Some(value))
        }
        Decoded::Infinity { negative } => DecimalRead::Number(Shape::Infinity(negative), None),
        Decoded::Unsupported => panic!("a decimal format has no unsupported encoding"),
    }
}

/// Returns the NaN that the rule selects from the NaN operands, in operand
/// order, and invalid for a signaling one. At least one operand is a NaN.
pub(super) fn decimal_nan(nans: &[Nan], env: &Env) -> (DecimalValue, Flags) {
    let nan = select_nan(nans, env);
    let flags = if nans.iter().any(|nan| nan.signaling) {
        Flags::INVALID
    } else {
        Flags::NONE
    };
    let value = DecimalValue::Nan {
        negative: nan.negative,
        payload: nan.payload,
    };
    (value, flags)
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
    let mut flags = Flags::NONE;
    let (value, more) = match read_decimal(x, env, &mut flags) {
        DecimalRead::Nan(nan) => decimal_nan(&[nan], env),
        DecimalRead::Number(shape, value) => {
            decimal_number(function, &shape, value.as_ref(), format, env)
        }
    };
    (value, flags | more)
}

/// Returns the expected result of `function` on a decimal operand that is
/// not a NaN, and the flags.
fn decimal_number(
    function: Function,
    shape: &Shape,
    value: Option<&Rational>,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let mut special = function.special(shape);
    if matches!(special, Special::Evaluate)
        && function.is_pi_scaled()
        && let Some(reduced) = value.and_then(quarter)
        && let Some(at_quarter) = function.at_quarter(&reduced)
    {
        special = at_quarter;
    }
    if let Some(result) = decimal_special(special, format, env) {
        return result;
    }
    let Some(x) = value else {
        // `acos(±0) = pi / 2` and `atan(±inf) = ±pi / 2`: MPFR bounds the
        // angle.
        let argument = match shape {
            Shape::Zero(false) => BigFloat::with_val(2, rug::float::Special::Zero),
            Shape::Zero(true) => BigFloat::with_val(2, rug::float::Special::NegZero),
            Shape::Infinity(false) => BigFloat::with_val(2, rug::float::Special::Infinity),
            Shape::Infinity(true) => BigFloat::with_val(2, rug::float::Special::NegInfinity),
            Shape::Finite { .. } => unreachable!("a finite operand has a value"),
        };
        let value = decimal_truncation(format.precision, |precision, round| {
            function.mpfr(&argument, precision, round)
        });
        return decimal::to_decimal(&value, format, env);
    };
    if let Some(result) = decimal_stand_in(function, x, format, env) {
        return result;
    }
    if let Some(value) = function.rational(x) {
        return decimal::rational_to_decimal(&value, format, env);
    }
    // `cosh` is even, and increases with `|x|`, so its argument enters MPFR
    // as `|x|`. A function that does not increase takes the least and the
    // greatest value at the two bounds of the argument, which lie closer
    // together than any extremum or pole.
    let x = if function == Function::Cosh {
        x.clone().abs()
    } else {
        x.clone()
    };
    let key = (function, x, format.precision);
    if let Some(value) = TRUNCATIONS.with_borrow(|map| map.get(&key).cloned()) {
        return decimal::to_decimal(&value, format, env);
    }
    let x = &key.1;
    let value = decimal_truncation(format.precision, |precision, round| {
        let argument = BigFloat::with_val_round(precision, x, round).0;
        let value = function.mpfr(&argument, precision, round);
        if function.is_increasing() {
            return value;
        }
        let other = BigFloat::with_val_round(precision, x, reverse(round)).0;
        let other = function.mpfr(&other, precision, round);
        match round {
            Round::Down if other < value => other,
            Round::Up if other > value => other,
            _ => value,
        }
    });
    let result = decimal::to_decimal(&value, format, env);
    TRUNCATIONS.with_borrow_mut(|map| map.insert(key, value));
    result
}

/// Returns the decimal result of a special case, or `None` to evaluate.
fn decimal_special(
    special: Special,
    format: DecimalFormat,
    env: &Env,
) -> Option<(DecimalValue, Flags)> {
    let value = match special {
        Special::Evaluate => return None,
        Special::One(negative) => DecimalValue::Finite {
            negative,
            coefficient: Integer::from(1),
            exponent: 0,
        },
        Special::Zero(negative) => DecimalValue::Zero {
            negative,
            exponent: 0,
        },
        Special::Infinity(negative) => DecimalValue::Infinity { negative },
        Special::Pole(negative) => {
            let infinity = DecimalValue::Infinity { negative };
            return Some((infinity, Flags::DIVIDE_BY_ZERO));
        }
        Special::Invalid => {
            let nan = DecimalValue::Nan {
                negative: env.nan.default_negative,
                payload: Integer::ZERO,
            };
            return Some((nan, Flags::INVALID));
        }
        Special::Quarters(n) => {
            let value = Rational::from((n, 4));
            return Some(decimal::rational_to_decimal(&value, format, env));
        }
    };
    Some((value, Flags::NONE))
}

/// Returns the decimal result at an argument whose value the oracle takes
/// from a bound, where MPFR bounds of a decimal argument would need a
/// working precision near `|x|` or `log2(1 / |x|)` bits. A stand-in value
/// lies in the same unit of `p + 2` digits as the value, on the same side of
/// each point of the grid, so it rounds as the value does.
fn decimal_stand_in(
    function: Function,
    x: &Rational,
    format: DecimalFormat,
    env: &Env,
) -> Option<(DecimalValue, Flags)> {
    let negative = *x < 0;
    let bound = decimal_out_of_range(format);
    let precision = format.precision;
    // `b^x <= 2^x` lies below `2^-(4p + 16)`, below `10^-(p + 2)`, and so
    // does `1 - |tanh x| < 2 e^-2|x|` from `|x| = 2p + 12`. So `b^x - 1` and
    // `tanh x` lie inside ±1 by less than one unit of `p + 2` digits, as
    // `±(1 - 2^-(4p + 20))` does.
    let shifted_negative = function.is_exponential() && function.is_shifted() && negative;
    let near_minus_one = shifted_negative && *x <= -i64::from(4 * precision + 16);
    let near_one = function == Function::Tanh && x.clone().abs() >= 2 * precision + 12;
    if near_minus_one || near_one {
        let tiny = BigFloat::with_val(2, 1) >> (4 * precision + 20);
        let value = BigFloat::with_val(4 * precision + 24, 1u32 - &tiny);
        let value = if negative { -value } else { value };
        return Some(decimal::to_decimal(&value, format, env));
    }
    // `2^(±4R)` lies past the range of the format, as the value does: `b^x`
    // overflows or underflows by the sign of `x`, and `sinh` and `cosh`
    // overflow, `sinh` with the sign of `x`.
    let overflows =
        function.is_exponential() || matches!(function, Function::Sinh | Function::Cosh);
    if x.clone().abs() >= bound && !shifted_negative && overflows {
        let scale = i32::try_from(bound).expect("a decimal bound fits an i32");
        let underflow = negative && function.is_exponential();
        let value = BigFloat::with_val(2, 1) << if underflow { -scale } else { scale };
        let value = if negative && function == Function::Sinh {
            -value
        } else {
            value
        };
        return Some(decimal::to_decimal(&value, format, env));
    }
    // `e^x - 1` lies above `x`, and `ln(1 + x)` below it, by less than
    // `x^2`. `sinh x`, `atanh x`, `asin x`, and `tan x` lie beyond `x`, and
    // `tanh x`, `asinh x`, `atan x`, and `sin x` short of it, by less than
    // `|x|^3`. Below `|x| = 2^-(4p + 70)`, `x ± |x| 2^-(4p + 70)` lies on the
    // same side of `x` and closer to it than one unit of `p + 2` digits, as
    // the value does, and GMP gives its digits.
    let tiny = Rational::from((1, Integer::from(1) << (4 * precision + 70)));
    let half = Rational::from((1, 2));
    // `acosPi x = 1/2 - asin(x) / pi` lies within `|x| / 3` of 1/2, and
    // `|atanPi x| = 1/2 - atan(1 / |x|) / pi` within `1 / (3 |x|)` of it.
    // There `1/2 - x / 4`, and `±(1/2 - 1 / (4 |x|))` past `2^(4p + 70)`,
    // lie on the same side of 1/2 and closer to it than one unit, as the
    // values do.
    if function == Function::AtanPi && x.clone().abs().recip() < tiny {
        let magnitude = half - (x.clone().abs() * 4u32).recip();
        let value = if negative { -magnitude } else { magnitude };
        return Some(decimal::rational_to_decimal(&value, format, env));
    }
    if x.clone().abs() >= tiny {
        return None;
    }
    let step = x.clone() * &tiny;
    let value = match function {
        Function::ExpM1 => x.clone() + step.abs(),
        Function::LogP1 => x.clone() - step.abs(),
        Function::Sinh | Function::Atanh | Function::Asin | Function::Tan => x.clone() + step,
        Function::Tanh | Function::Asinh | Function::Atan | Function::Sin => x.clone() - step,
        Function::AcosPi => half - x.clone() / 4u32,
        _ => return None,
    };
    Some(decimal::rational_to_decimal(&value, format, env))
}

/// Returns a binary value with the truncation to `precision + 2` digits of
/// an irrational value, strictly inside the truncation interval. `evaluate`
/// gives a bound of the value at a working precision in a direction. The
/// working precision doubles until the bounds decide the truncation.
///
/// # Panics
///
/// Panics when the working precision reaches [`PRECISION_LIMIT`].
pub(super) fn decimal_truncation(
    precision: u32,
    evaluate: impl Fn(u32, Round) -> BigFloat,
) -> BigFloat {
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
