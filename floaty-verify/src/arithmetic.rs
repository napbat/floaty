//! An arithmetic oracle: MPFR computes the finite results, and the special
//! values follow IEEE 754 and the rules of floaty's `Env`.
//!
//! MPFR computes each result at the format precision plus 64 bits, rounded
//! toward zero, with a sticky bit for the lost bits. The rounding oracle in
//! [`crate::mpfr`] then rounds that value. The extra bits make the second
//! rounding give the correctly rounded result, by the round-to-odd property.
//!
//! The oracle does not select NaNs: TestFloat and the processor check the NaN
//! rules. Its NaN result stands for any NaN; the tests that use it check that
//! floaty returns a NaN operand made quiet, or the default NaN.

use core::cmp::Ordering;

use floaty::env::InvalidProduct;
use floaty::{Decoded, Env, Flags, Rounding};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

use crate::mpfr::{self, Format, Input, Specials, Value};

/// An arithmetic operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`.
    Div,
    /// The square root of `a`.
    Sqrt,
    /// `a * b + c`, rounded once.
    MulAdd,
}

/// An operand: its decoded value, and whether its encoding is subnormal.
#[derive(Clone, Copy, Debug)]
pub struct Operand<const N: usize> {
    /// The decoded value.
    pub decoded: Decoded<N>,
    /// `true` when the encoding is subnormal.
    pub subnormal: bool,
}

/// A value that is not a NaN, as the oracle computes with it.
#[derive(Clone, Debug)]
enum Number {
    Zero(bool),
    Finite(BigFloat),
    Infinity(bool),
}

impl Number {
    fn negative(&self) -> bool {
        match self {
            Self::Zero(negative) | Self::Infinity(negative) => *negative,
            Self::Finite(value) => value.is_sign_negative(),
        }
    }
}

/// Returns the expected result and flags of an operation. `operands` holds one,
/// two, or three operands in operand order.
///
/// # Panics
///
/// Panics for an unsupported operand, which these formats do not have.
#[must_use]
pub fn compute<const N: usize>(
    operation: Operation,
    operands: &[Operand<N>],
    format: &Format,
    env: &Env,
) -> (Value, Flags) {
    let mut flags = Flags::NONE;
    let mut numbers = Vec::new();
    let mut signaling = false;
    let mut nan = false;
    for operand in operands {
        if operand.subnormal {
            flags |= Flags::DENORMAL_INPUT;
        }
        match operand.decoded {
            Decoded::Nan {
                signaling: sign, ..
            } => {
                nan = true;
                signaling |= sign;
                numbers.push(Number::Zero(false));
            }
            Decoded::Zero { negative, .. } => numbers.push(Number::Zero(negative)),
            Decoded::Finite { negative, .. } if operand.subnormal && env.denormals_are_zero => {
                numbers.push(Number::Zero(negative));
            }
            Decoded::Finite {
                negative,
                exponent,
                significand,
            } => {
                let integer = Integer::from_digits(&significand, Order::Lsf);
                let bits = integer.significant_bits().max(1);
                let mut value = BigFloat::with_val(bits, integer) << exponent;
                if negative {
                    value = -value;
                }
                numbers.push(Number::Finite(value));
            }
            Decoded::Infinity { negative } => numbers.push(Number::Infinity(negative)),
            Decoded::Unsupported => panic!("the oracle has no rule for an unsupported operand"),
        }
    }
    if signaling {
        flags |= Flags::INVALID;
    }
    if nan {
        // The invalid product 0 * inf with a NaN addend signals invalid
        // unless the rule is `YieldsToNan`. A NaN factor makes the product a
        // NaN, not 0 * inf.
        let product_nan = operands[..2.min(operands.len())]
            .iter()
            .any(|operand| matches!(operand.decoded, Decoded::Nan { .. }));
        if operation == Operation::MulAdd
            && env.nan.invalid_product != InvalidProduct::YieldsToNan
            && !product_nan
            && is_zero_times_infinity(&numbers[0], &numbers[1])
        {
            flags |= Flags::INVALID;
        }
        return (Value::Nan { negative: false }, flags);
    }
    let (value, special) = match operation {
        Operation::Add => add(&numbers[0], &numbers[1], format, env),
        Operation::Sub => add(&numbers[0], &negate(&numbers[1]), format, env),
        Operation::Mul => mul(&numbers[0], &numbers[1], format, env),
        Operation::Div => div(&numbers[0], &numbers[1], format, env),
        Operation::Sqrt => sqrt(&numbers[0], format, env),
        Operation::MulAdd => mul_add(&numbers[0], &numbers[1], &numbers[2], format, env),
    };
    (value, flags | special)
}

fn negate(number: &Number) -> Number {
    match number {
        Number::Zero(negative) => Number::Zero(!negative),
        Number::Finite(value) => Number::Finite(-value.clone()),
        Number::Infinity(negative) => Number::Infinity(!negative),
    }
}

fn is_zero_times_infinity(first: &Number, second: &Number) -> bool {
    matches!(
        (first, second),
        (Number::Zero(_), Number::Infinity(_)) | (Number::Infinity(_), Number::Zero(_))
    )
}

fn invalid() -> (Value, Flags) {
    (Value::Nan { negative: false }, Flags::INVALID)
}

fn zero(negative: bool, format: &Format) -> Value {
    Value::Zero {
        negative: negative && format.specials != Specials::Fnuz,
    }
}

/// Returns an infinity, or for a format without one the NaN or, when the
/// behavior saturates, the largest finite value.
fn infinity(negative: bool, format: &Format, env: &Env) -> Value {
    if format.specials == Specials::Ieee {
        return Value::Infinity { negative };
    }
    if !env.saturate {
        return Value::Nan { negative: false };
    }
    let precision = format.precision_in(env);
    let significand = if format.specials == Specials::NoInf && precision == format.precision {
        (Integer::from(1) << precision) - 2u32
    } else {
        (Integer::from(1) << precision) - 1u32
    };
    let shift = format.emax - i32::try_from(precision - 1).expect("a precision fits an i32");
    let mut value = BigFloat::with_val(precision, significand) << shift;
    if negative {
        value = -value;
    }
    Value::Finite(value)
}

/// The sign of an exact zero sum of values with different signs.
fn zero_sum_sign(env: &Env) -> bool {
    env.rounding == Rounding::TowardNegative
}

/// Rounds an MPFR result, computed at the format precision plus 64 bits
/// toward zero, to the format.
fn finish(value: &BigFloat, ordering: Ordering, format: &Format, env: &Env) -> (Value, Flags) {
    let (integer, exponent) = value
        .to_integer_exp()
        .expect("the result is finite and nonzero");
    let input = Input {
        negative: value.is_sign_negative(),
        exponent,
        significand: integer.abs(),
        sticky: ordering != Ordering::Equal,
    };
    mpfr::round(&input, format, env)
}

/// The working precision of MPFR.
fn working(format: &Format) -> u32 {
    format.precision + 64
}

fn add(first: &Number, second: &Number, format: &Format, env: &Env) -> (Value, Flags) {
    match (first, second) {
        (Number::Infinity(a), Number::Infinity(b)) if a != b => invalid(),
        (Number::Infinity(negative), _) | (_, Number::Infinity(negative)) => {
            (infinity(*negative, format, env), Flags::NONE)
        }
        (Number::Zero(a), Number::Zero(b)) => {
            let negative = if a == b { *a } else { zero_sum_sign(env) };
            (zero(negative, format), Flags::NONE)
        }
        (Number::Zero(_), Number::Finite(value)) | (Number::Finite(value), Number::Zero(_)) => {
            finish(value, Ordering::Equal, format, env)
        }
        (Number::Finite(a), Number::Finite(b)) => {
            let (sum, ordering) = BigFloat::with_val_round(working(format), a + b, Round::Zero);
            if sum.is_zero() {
                return (zero(zero_sum_sign(env), format), Flags::NONE);
            }
            finish(&sum, ordering, format, env)
        }
    }
}

fn mul(first: &Number, second: &Number, format: &Format, env: &Env) -> (Value, Flags) {
    let negative = first.negative() != second.negative();
    match (first, second) {
        _ if is_zero_times_infinity(first, second) => invalid(),
        (Number::Infinity(_), _) | (_, Number::Infinity(_)) => {
            (infinity(negative, format, env), Flags::NONE)
        }
        (Number::Zero(_), _) | (_, Number::Zero(_)) => (zero(negative, format), Flags::NONE),
        (Number::Finite(a), Number::Finite(b)) => {
            let (product, ordering) = BigFloat::with_val_round(working(format), a * b, Round::Zero);
            finish(&product, ordering, format, env)
        }
    }
}

fn div(first: &Number, second: &Number, format: &Format, env: &Env) -> (Value, Flags) {
    let negative = first.negative() != second.negative();
    match (first, second) {
        (Number::Infinity(_), Number::Infinity(_)) | (Number::Zero(_), Number::Zero(_)) => {
            invalid()
        }
        (Number::Infinity(_), _) => (infinity(negative, format, env), Flags::NONE),
        (_, Number::Infinity(_)) | (Number::Zero(_), _) => (zero(negative, format), Flags::NONE),
        (_, Number::Zero(_)) => (infinity(negative, format, env), Flags::DIVIDE_BY_ZERO),
        (Number::Finite(a), Number::Finite(b)) => {
            let (quotient, ordering) =
                BigFloat::with_val_round(working(format), a / b, Round::Zero);
            finish(&quotient, ordering, format, env)
        }
    }
}

fn sqrt(value: &Number, format: &Format, env: &Env) -> (Value, Flags) {
    match value {
        Number::Zero(negative) => (zero(*negative, format), Flags::NONE),
        Number::Infinity(false) => (infinity(false, format, env), Flags::NONE),
        Number::Infinity(true) => invalid(),
        Number::Finite(value) if value.is_sign_negative() => invalid(),
        Number::Finite(value) => {
            let (root, ordering) =
                BigFloat::with_val_round(working(format), value.sqrt_ref(), Round::Zero);
            finish(&root, ordering, format, env)
        }
    }
}

fn mul_add(
    first: &Number,
    second: &Number,
    addend: &Number,
    format: &Format,
    env: &Env,
) -> (Value, Flags) {
    if is_zero_times_infinity(first, second) {
        return invalid();
    }
    let product_negative = first.negative() != second.negative();
    let product_infinite =
        matches!(first, Number::Infinity(_)) || matches!(second, Number::Infinity(_));
    let product_zero = matches!(first, Number::Zero(_)) || matches!(second, Number::Zero(_));
    match addend {
        _ if product_infinite => match addend {
            Number::Infinity(negative) if *negative != product_negative => invalid(),
            _ => (infinity(product_negative, format, env), Flags::NONE),
        },
        Number::Infinity(negative) => (infinity(*negative, format, env), Flags::NONE),
        Number::Zero(negative) if product_zero => {
            let negative = if *negative == product_negative {
                *negative
            } else {
                zero_sum_sign(env)
            };
            (zero(negative, format), Flags::NONE)
        }
        Number::Finite(value) if product_zero => finish(value, Ordering::Equal, format, env),
        _ => {
            let (Number::Finite(a), Number::Finite(b)) = (first, second) else {
                unreachable!("a finite product has finite operands");
            };
            // The exact product needs twice the precision; MPFR rounds the sum
            // of it and the addend once.
            let zero_addend = BigFloat::new(2);
            let c = match addend {
                Number::Finite(value) => value,
                _ => &zero_addend,
            };
            let (result, ordering) =
                BigFloat::with_val_round(working(format), a.mul_add_ref(b, c), Round::Zero);
            if result.is_zero() {
                return (zero(zero_sum_sign(env), format), Flags::NONE);
            }
            finish(&result, ordering, format, env)
        }
    }
}
