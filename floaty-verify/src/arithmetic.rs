//! An arithmetic oracle: MPFR computes the finite results, and the special
//! values follow IEEE 754 and the rules of floaty's `Env`.
//!
//! MPFR computes each result at the format precision plus 64 bits, rounded
//! toward zero, with a sticky bit for the lost bits. The rounding oracle in
//! [`crate::mpfr`] then rounds that value. The extra bits make the second
//! rounding give the correctly rounded result, by the round-to-odd property.
//!
//! An unsupported operand, such as an x87 unnormal, signals invalid and
//! gives the default NaN, as the documentation of `Class::Unsupported`
//! states. A NaN result follows the NaN rule of the `Env`, as its
//! documentation states. The rule has a propagation rule and a default NaN. For a fused
//! multiply-add, it also has a rule for `0 * inf + NaN` and an order of the
//! NaN operands. An overflow in a format without an infinity gives the NaN of
//! that format, with the sign of the result.

use core::cmp::Ordering;

use floaty::env::{FusedNanOrder, InvalidProduct};
use floaty::{Env, Flags, Rounding};
use rug::Float as BigFloat;
use rug::float::Round;

use crate::encodings::to_limbs;
use crate::mpfr::{self, Format, Input, Nan, Operand, Read, Value, select_nan};

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

/// A value that is not a NaN, as the oracle computes with it.
#[derive(Clone, Debug)]
enum Number {
    Zero(bool),
    Finite(BigFloat),
    Infinity(bool),
}

impl Number {
    /// Returns an MPFR number as a zero, a finite value, or an infinity.
    fn of(value: &BigFloat) -> Self {
        if value.is_zero() {
            Self::Zero(value.is_sign_negative())
        } else if value.is_infinite() {
            Self::Infinity(value.is_sign_negative())
        } else {
            Self::Finite(value.clone())
        }
    }

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
/// Panics for a NaN rule that a later floaty adds.
#[must_use]
pub fn compute<const N: usize>(
    operation: Operation,
    operands: &[Operand<N>],
    format: &Format,
    env: &Env,
) -> Expected<N> {
    let mut flags = Flags::NONE;
    let reads: Vec<Read> = operands
        .iter()
        .map(|operand| operand.read(env, &mut flags))
        .collect();
    if reads.iter().any(|read| matches!(read, Read::Unsupported)) {
        return Expected {
            value: format.nan(Nan::default_of(env).negative),
            payload: [0; N],
            flags: flags | Flags::INVALID,
        };
    }
    let numbers: Vec<Number> = reads
        .iter()
        .map(|read| match read {
            Read::Number(value) => Number::of(value),
            Read::Nan(_) | Read::Unsupported => Number::Zero(false),
        })
        .collect();
    if reads.iter().any(Read::is_signaling) {
        flags |= Flags::INVALID;
    }
    if reads.iter().any(|read| matches!(read, Read::Nan(_))) {
        // The invalid product 0 * inf with a NaN addend signals invalid
        // unless the rule is `YieldsToNan`. A NaN factor makes the product a
        // NaN, not 0 * inf.
        let product_nan = reads[..2.min(reads.len())]
            .iter()
            .any(|read| matches!(read, Read::Nan(_)));
        if operation == Operation::MulAdd
            && env.nan.invalid_product != InvalidProduct::YieldsToNan
            && !product_nan
            && is_zero_times_infinity(&numbers[0], &numbers[1])
        {
            flags |= Flags::INVALID;
        }
        let nan = propagate(operation, &reads, &numbers, env);
        return Expected {
            value: format.nan(nan.negative),
            payload: to_limbs(&nan.payload),
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
fn candidate(read: &Read) -> Option<Nan> {
    match read {
        Read::Nan(nan) => Some(nan.clone()),
        Read::Number(_) | Read::Unsupported => None,
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
fn propagate(operation: Operation, operands: &[Read], numbers: &[Number], env: &Env) -> Nan {
    let offer = |indices: &[usize]| -> Vec<Nan> {
        indices
            .iter()
            .filter_map(|&index| candidate(&operands[index]))
            .collect()
    };
    if operation != Operation::MulAdd {
        let every: Vec<usize> = (0..operands.len()).collect();
        return select_nan(&offer(&every), env);
    }
    let product = offer(&[0, 1]);
    if product.is_empty() && is_zero_times_infinity(&numbers[0], &numbers[1]) {
        return match env.nan.invalid_product {
            InvalidProduct::Signals => {
                let mut offered = vec![Nan::default_of(env)];
                offered.extend(offer(&[2]));
                select_nan(&offered, env)
            }
            InvalidProduct::YieldsToNan | InvalidProduct::SignalsAndYieldsToNan => {
                select_nan(&offer(&[2]), env)
            }
            _ => panic!("the oracle knows every rule for 0 * inf + NaN"),
        };
    }
    match env.nan.fused_order {
        FusedNanOrder::ProductFirst => {
            let mut offered = Vec::new();
            if !product.is_empty() {
                offered.push(select_nan(&product, env));
            }
            offered.extend(offer(&[2]));
            select_nan(&offered, env)
        }
        FusedNanOrder::AddendFirst => select_nan(&offer(&[2, 0, 1]), env),
        FusedNanOrder::AddendSecond => select_nan(&offer(&[0, 2, 1]), env),
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

/// Returns the result of an invalid operation: the default NaN of the
/// format, by [`Format::nan`].
fn invalid(format: &Format, env: &Env) -> (Value, Flags) {
    (format.nan(env.nan.default_negative), Flags::INVALID)
}

/// The sign of an exact zero sum of values with different signs.
fn zero_sum_sign(env: &Env) -> bool {
    env.rounding == Rounding::TowardNegative
}

/// Rounds an MPFR result, computed at the format precision plus 64 bits
/// toward zero, to the format.
pub(crate) fn finish(
    value: &BigFloat,
    ordering: Ordering,
    format: &Format,
    env: &Env,
) -> (Value, Flags) {
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
pub(crate) fn working(format: &Format) -> u32 {
    format.precision + 64
}

fn add(first: &Number, second: &Number, format: &Format, env: &Env) -> (Value, Flags) {
    match (first, second) {
        (Number::Infinity(a), Number::Infinity(b)) if a != b => invalid(format, env),
        (Number::Infinity(negative), _) | (_, Number::Infinity(negative)) => {
            (format.infinity(*negative, env), Flags::NONE)
        }
        (Number::Zero(a), Number::Zero(b)) => {
            let negative = if a == b { *a } else { zero_sum_sign(env) };
            (format.zero(negative), Flags::NONE)
        }
        (Number::Zero(_), Number::Finite(value)) | (Number::Finite(value), Number::Zero(_)) => {
            finish(value, Ordering::Equal, format, env)
        }
        (Number::Finite(a), Number::Finite(b)) => {
            let (sum, ordering) = BigFloat::with_val_round(working(format), a + b, Round::Zero);
            if sum.is_zero() {
                return (format.zero(zero_sum_sign(env)), Flags::NONE);
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
            (format.infinity(negative, env), Flags::NONE)
        }
        (Number::Zero(_), _) | (_, Number::Zero(_)) => (format.zero(negative), Flags::NONE),
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
        (Number::Infinity(_), _) => (format.infinity(negative, env), Flags::NONE),
        (_, Number::Infinity(_)) | (Number::Zero(_), _) => (format.zero(negative), Flags::NONE),
        (_, Number::Zero(_)) => (format.infinity(negative, env), Flags::DIVIDE_BY_ZERO),
        (Number::Finite(a), Number::Finite(b)) => {
            let (quotient, ordering) =
                BigFloat::with_val_round(working(format), a / b, Round::Zero);
            finish(&quotient, ordering, format, env)
        }
    }
}

fn sqrt(value: &Number, format: &Format, env: &Env) -> (Value, Flags) {
    match value {
        Number::Zero(negative) => (format.zero(*negative), Flags::NONE),
        Number::Infinity(false) => (format.infinity(false, env), Flags::NONE),
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
            _ => (format.infinity(product_negative, env), Flags::NONE),
        },
        Number::Infinity(negative) => (format.infinity(*negative, env), Flags::NONE),
        Number::Zero(negative) if product_zero => {
            let negative = if *negative == product_negative {
                *negative
            } else {
                zero_sum_sign(env)
            };
            (format.zero(negative), Flags::NONE)
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
                return (format.zero(zero_sum_sign(env)), Flags::NONE);
            }
            finish(&result, ordering, format, env)
        }
    }
}
