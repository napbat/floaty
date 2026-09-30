//! An arithmetic oracle: MPFR computes the finite results, and the special
//! values follow IEEE 754 and the rules of floaty's `Env`.
//!
//! MPFR computes each result at the format precision plus 64 bits, rounded
//! toward zero, with a sticky bit for the lost bits. The rounding oracle in
//! [`crate::mpfr`] then rounds that value. The extra bits make the second
//! rounding give the correctly rounded result, by the round-to-odd property.
//!
//! A NaN result follows the NaN rule of the `Env`, as its documentation
//! states. The rule has a propagation rule and a default NaN. For a fused
//! multiply-add, it also has a rule for `0 * inf + NaN` and an order of the
//! NaN operands. An overflow in a format without an infinity gives the NaN of
//! that format, with the sign of the result.

use core::cmp::Ordering;

use floaty::env::{FusedNanOrder, InvalidProduct, NanPropagation};
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

/// The expected result of an operation.
#[derive(Clone, Debug, PartialEq)]
pub struct Expected<const N: usize> {
    /// The value. A NaN has the sign that the NaN rule gives.
    pub value: Value,
    /// The payload of a NaN value, and zero for every other value.
    pub payload: [u64; N],
    /// The flags.
    pub flags: Flags,
}

/// A NaN that the propagation rule can select.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Candidate<const N: usize> {
    negative: bool,
    signaling: bool,
    payload: [u64; N],
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
/// Panics for an unsupported operand, which these formats do not have, and
/// for a NaN rule that a later floaty adds.
#[must_use]
pub fn compute<const N: usize>(
    operation: Operation,
    operands: &[Operand<N>],
    format: &Format,
    env: &Env,
) -> Expected<N> {
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
        let nan = propagate(operation, operands, &numbers, env);
        return Expected {
            value: Value::Nan {
                negative: nan.negative || format.specials == Specials::Fnuz,
            },
            payload: nan.payload,
            flags,
        };
    }
    let (value, special) = match operation {
        Operation::Add => add(&numbers[0], &numbers[1], format, env),
        Operation::Sub => add(&numbers[0], &negate(&numbers[1]), format, env),
        Operation::Mul => mul(&numbers[0], &numbers[1], format, env),
        Operation::Div => div(&numbers[0], &numbers[1], format, env),
        Operation::Sqrt => sqrt(&numbers[0], format, env),
        Operation::MulAdd => mul_add(&numbers[0], &numbers[1], &numbers[2], format, env),
    };
    Expected {
        value,
        payload: [0; N],
        flags: flags | special,
    }
}

/// Returns an operand as a NaN that the propagation rule can select, or
/// `None` for a number.
fn candidate<const N: usize>(operand: &Operand<N>) -> Option<Candidate<N>> {
    match operand.decoded {
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => Some(Candidate {
            negative,
            signaling,
            payload,
        }),
        _ => None,
    }
}

/// Selects a NaN among `offered`, in the order of the list, by the
/// propagation rule of `env`, and makes it quiet. `DefaultNan` gives the
/// default NaN. `LargerSignificand` takes a quiet NaN before a signaling NaN,
/// then the larger payload, then the positive sign.
fn select<const N: usize>(offered: &[Candidate<N>], env: &Env) -> Candidate<N> {
    let first = offered
        .first()
        .expect("the rule selects from at least one NaN");
    let chosen = match env.nan.propagation {
        NanPropagation::DefaultNan => Candidate {
            negative: env.nan.default_negative,
            signaling: false,
            payload: [0; N],
        },
        NanPropagation::SignalingFirst => {
            *offered.iter().find(|nan| nan.signaling).unwrap_or(first)
        }
        NanPropagation::FirstOperand => *first,
        NanPropagation::LargerSignificand => *offered
            .iter()
            .max_by(|a, b| {
                (!a.signaling)
                    .cmp(&!b.signaling)
                    .then_with(|| a.payload.iter().rev().cmp(b.payload.iter().rev()))
                    .then_with(|| (!a.negative).cmp(&!b.negative))
            })
            .expect("the rule selects from at least one NaN"),
        _ => panic!("the oracle knows every propagation rule"),
    };
    Candidate {
        signaling: false,
        ..chosen
    }
}

/// Returns the NaN result of an operation with a NaN operand.
///
/// A fused multiply-add whose product is the invalid `0 * inf` follows the
/// rule for `0 * inf + NaN`. `Signals` offers the default NaN and then the
/// addend. The other rules offer the addend alone.
///
/// Otherwise the fused order decides. `ProductFirst` selects among the
/// factors and makes that NaN quiet. It then selects between that NaN and the
/// addend. `AddendFirst` offers the addend and the factors. `AddendSecond`
/// offers the first factor, the addend, and the second factor.
fn propagate<const N: usize>(
    operation: Operation,
    operands: &[Operand<N>],
    numbers: &[Number],
    env: &Env,
) -> Candidate<N> {
    let offer = |indices: &[usize]| -> Vec<Candidate<N>> {
        indices
            .iter()
            .filter_map(|&index| candidate(&operands[index]))
            .collect()
    };
    if operation != Operation::MulAdd {
        let every: Vec<usize> = (0..operands.len()).collect();
        return select(&offer(&every), env);
    }
    let product = offer(&[0, 1]);
    if product.is_empty() && is_zero_times_infinity(&numbers[0], &numbers[1]) {
        return match env.nan.invalid_product {
            InvalidProduct::Signals => {
                let default = Candidate {
                    negative: env.nan.default_negative,
                    signaling: false,
                    payload: [0; N],
                };
                let mut offered = vec![default];
                offered.extend(offer(&[2]));
                select(&offered, env)
            }
            InvalidProduct::YieldsToNan | InvalidProduct::SignalsAndYieldsToNan => {
                select(&offer(&[2]), env)
            }
            _ => panic!("the oracle knows every rule for 0 * inf + NaN"),
        };
    }
    match env.nan.fused_order {
        FusedNanOrder::ProductFirst => {
            let mut offered = Vec::new();
            if !product.is_empty() {
                offered.push(select(&product, env));
            }
            offered.extend(offer(&[2]));
            select(&offered, env)
        }
        FusedNanOrder::AddendFirst => select(&offer(&[2, 0, 1]), env),
        FusedNanOrder::AddendSecond => select(&offer(&[0, 2, 1]), env),
        _ => panic!("the oracle knows every fused NaN order"),
    }
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

/// Returns the result of an invalid operation: the default NaN, or positive
/// zero in a format without a NaN. The one NaN of `Fnuz` is negative.
fn invalid(format: &Format, env: &Env) -> (Value, Flags) {
    if format.specials == Specials::Finite {
        return (Value::Zero { negative: false }, Flags::INVALID);
    }
    let negative = env.nan.default_negative || format.specials == Specials::Fnuz;
    (Value::Nan { negative }, Flags::INVALID)
}

fn zero(negative: bool, format: &Format) -> Value {
    Value::Zero {
        negative: negative && format.specials != Specials::Fnuz,
    }
}

/// Returns an infinity, or for a format without one the NaN or, when the
/// behavior saturates or the format has no NaN, the largest finite value.
fn infinity(negative: bool, format: &Format, env: &Env) -> Value {
    if format.specials == Specials::Ieee {
        return Value::Infinity { negative };
    }
    if !env.saturate && format.specials != Specials::Finite {
        return Value::Nan {
            negative: negative || format.specials == Specials::Fnuz,
        };
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
        (Number::Infinity(a), Number::Infinity(b)) if a != b => invalid(format, env),
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
        _ if is_zero_times_infinity(first, second) => invalid(format, env),
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
            invalid(format, env)
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
        Number::Infinity(true) => invalid(format, env),
        Number::Finite(value) if value.is_sign_negative() => invalid(format, env),
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
        return invalid(format, env);
    }
    let product_negative = first.negative() != second.negative();
    let product_infinite =
        matches!(first, Number::Infinity(_)) || matches!(second, Number::Infinity(_));
    let product_zero = matches!(first, Number::Zero(_)) || matches!(second, Number::Zero(_));
    match addend {
        _ if product_infinite => match addend {
            Number::Infinity(negative) if *negative != product_negative => invalid(format, env),
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
