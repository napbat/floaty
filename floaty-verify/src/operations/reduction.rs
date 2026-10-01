//! An oracle for the reduction operations of IEEE 754-2019 section 9.4 on
//! the binary formats, by the documented rules of floaty.
//!
//! - `sum`, `sumAbs`, `sumSquare`, and `dot` round their exact result once.
//!   MPFR's `mpfr_sum` and `mpfr_dot` give the exact result rounded toward
//!   zero with the ternary value, and [`crate::mpfr::round`] rounds it.
//! - The scaled products round each step: MPFR computes each exact step,
//!   scales it into `[1, 2)`, and the rounding oracle rounds it.
//!
//! ISO/IEC TS 18661-4 gives the special cases. floaty documents the rest on
//! `Float::sum_with`, `Float::dot_with`, and `floaty::Scaled`.

use floaty::format::{Encoding, Standard, Storage, Width};
use floaty::{Binary, Env, Flags, Float, Rounding, Scaled};
use rug::Float as BigFloat;
use rug::float::Round;

use super::{Outcome, propagate};
use crate::arithmetic::{finish, working};
use crate::mpfr::{Format, Operand, Read, Value};

/// A sum of one vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Summand {
    /// `sum`.
    Value,
    /// `sumAbs`.
    Magnitude,
    /// `sumSquare`.
    Square,
}

impl Summand {
    /// Every sum of one vector.
    pub const ALL: [Self; 3] = [Self::Value, Self::Magnitude, Self::Square];

    /// Runs the sum of floaty.
    #[must_use]
    pub fn apply<const E: u32, Enc: Encoding, const W: usize>(
        self,
        values: &[Float<Binary<E, Enc>, W>],
        env: Env,
    ) -> (Float<Binary<E, Enc>, W>, Flags)
    where
        Width<W>: Storage,
        Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
    {
        let values = values.iter().copied();
        match self {
            Self::Value => Float::<Binary<E, Enc>, W>::sum_with(values, env),
            Self::Magnitude => Float::<Binary<E, Enc>, W>::sum_abs_with(values, env),
            Self::Square => Float::<Binary<E, Enc>, W>::sum_square_with(values, env),
        }
    }
}

/// A scaled product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Product {
    /// `scaledProd`: the operands are the factors.
    Values,
    /// `scaledProdSum`: each factor is the sum of two operands in order.
    Sums,
    /// `scaledProdDiff`: each factor is the difference of two operands in
    /// order.
    Differences,
}

impl Product {
    /// Every scaled product.
    pub const ALL: [Self; 3] = [Self::Values, Self::Sums, Self::Differences];

    /// Runs the scaled product of floaty. A product of pairs reads the
    /// operands two by two.
    #[must_use]
    pub fn apply<const E: u32, Enc: Encoding, const W: usize>(
        self,
        operands: &[Float<Binary<E, Enc>, W>],
        env: Env,
    ) -> (Scaled<Float<Binary<E, Enc>, W>>, Flags)
    where
        Width<W>: Storage,
        Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
    {
        type Value<const E: u32, Enc, const W: usize> = Float<Binary<E, Enc>, W>;
        let pairs = operands.as_chunks::<2>().0.iter().map(|&[x, y]| (x, y));
        match self {
            Self::Values => Value::<E, Enc, W>::scaled_product_with(operands.iter().copied(), env),
            Self::Sums => Value::<E, Enc, W>::scaled_product_sum_with(pairs, env),
            Self::Differences => Value::<E, Enc, W>::scaled_product_difference_with(pairs, env),
        }
    }
}

/// Returns the expected sum of the values, magnitudes, or squares of
/// `operands`, and the flags.
#[must_use]
pub fn sum<const N: usize>(
    operands: &[Operand<N>],
    summand: Summand,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads: Vec<Read> = operands
        .iter()
        .map(|operand| operand.read(env, &mut flags))
        .collect();
    let numbers: Vec<&BigFloat> = reads
        .iter()
        .filter_map(|read| match read {
            Read::Number(value) => Some(value),
            Read::Nan(_) | Read::Unsupported => None,
        })
        .collect();
    if reads.iter().any(|read| matches!(read, Read::Unsupported)) {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    let signed = summand == Summand::Value;
    let infinite = |negative| {
        numbers
            .iter()
            .any(|value| value.is_infinite() && (!signed || value.is_sign_negative() == negative))
    };
    let signaling = reads.iter().any(Read::is_signaling);
    let nan = reads.iter().any(|read| matches!(read, Read::Nan(_)));
    // `sumAbs` and `sumSquare` give +inf before a quiet NaN.
    if nan && (signed || signaling || !infinite(false)) {
        let (value, special) = propagate(&reads, format, env);
        return (value, flags | special);
    }
    match (infinite(false), signed && infinite(true)) {
        (true, true) => return (default_nan(format, env), flags | Flags::INVALID),
        (true, false) => return (infinity(false, format, env), flags),
        (false, true) => return (infinity(true, format, env), flags),
        (false, false) => {}
    }
    let terms: Vec<BigFloat> = numbers
        .iter()
        .map(|&value| match summand {
            Summand::Value => value.clone(),
            Summand::Magnitude => value.clone().abs(),
            Summand::Square => BigFloat::with_val(2 * format.precision, value.square_ref()),
        })
        .collect();
    let zero_signs: Vec<bool> = terms
        .iter()
        .filter(|term| term.is_zero())
        .map(BigFloat::is_sign_negative)
        .collect();
    let (outcome, round_flags) = round_terms(&terms, &zero_signs, format, env);
    (outcome, flags | round_flags)
}

/// Returns the expected sum of the products of `pairs`, and the flags.
#[must_use]
pub fn dot<const N: usize>(
    pairs: &[(Operand<N>, Operand<N>)],
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads: Vec<Read> = pairs
        .iter()
        .flat_map(|(x, y)| [x, y])
        .map(|operand| operand.read(env, &mut flags))
        .collect();
    if reads.iter().any(|read| matches!(read, Read::Unsupported)) {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    let numbers: Vec<(&BigFloat, &BigFloat)> = reads
        .as_chunks::<2>()
        .0
        .iter()
        .filter_map(|pair| match pair {
            [Read::Number(x), Read::Number(y)] => Some((x, y)),
            _ => None,
        })
        .collect();
    let invalid_product = numbers
        .iter()
        .any(|(x, y)| (x.is_infinite() && y.is_zero()) || (x.is_zero() && y.is_infinite()));
    if reads.iter().any(|read| matches!(read, Read::Nan(_))) {
        let (value, special) = propagate(&reads, format, env);
        let invalid = if invalid_product {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        return (value, flags | special | invalid);
    }
    if invalid_product {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    let infinite = |negative| {
        numbers.iter().any(|(x, y)| {
            (x.is_infinite() || y.is_infinite())
                && (x.is_sign_negative() != y.is_sign_negative()) == negative
        })
    };
    match (infinite(false), infinite(true)) {
        (true, true) => return (default_nan(format, env), flags | Flags::INVALID),
        (true, false) => return (infinity(false, format, env), flags),
        (false, true) => return (infinity(true, format, env), flags),
        (false, false) => {}
    }
    let terms: Vec<BigFloat> = numbers
        .iter()
        .map(|&(x, y)| BigFloat::with_val(2 * format.precision, x * y))
        .collect();
    let zero_signs: Vec<bool> = numbers
        .iter()
        .filter(|(x, y)| x.is_zero() || y.is_zero())
        .map(|(x, y)| x.is_sign_negative() != y.is_sign_negative())
        .collect();
    let (outcome, round_flags) = round_terms(&terms, &zero_signs, format, env);
    (outcome, flags | round_flags)
}

/// Rounds the exact sum of `terms` once. An exact zero sum of nonzero terms
/// is +0, or -0 toward negative. Zero terms alone, with the signs
/// `zero_signs`, give -0 when every one is negative, the zero of a sum of
/// zeros of different signs when the signs differ, and +0 otherwise.
fn round_terms(
    terms: &[BigFloat],
    zero_signs: &[bool],
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let (value, ordering) =
        BigFloat::with_val_round(working(format), BigFloat::sum(terms.iter()), Round::Zero);
    if !value.is_zero() {
        let (value, flags) = finish(&value, ordering, format, env);
        return (Outcome::from_value(value), flags);
    }
    let toward_negative = env.rounding == Rounding::TowardNegative;
    let negative = if terms.iter().any(|term| !term.is_zero()) {
        toward_negative
    } else if zero_signs.iter().all(|&negative| negative) {
        !zero_signs.is_empty()
    } else {
        zero_signs.iter().any(|&negative| negative) && toward_negative
    };
    (Outcome::from_value(format.zero(negative)), Flags::NONE)
}

/// Returns the expected scaled product, its scale, and the flags.
/// `operands` holds the factors, or for a product of pairs the operands of
/// the factors two by two.
///
/// # Panics
///
/// Panics when a step rounds to a value that is not finite, which does not
/// happen in `[1, 2]`.
#[must_use]
pub fn scaled_product<const N: usize>(
    operands: &[Operand<N>],
    product: Product,
    format: &Format,
    env: &Env,
) -> (Outcome, i64, Flags) {
    let mut flags = Flags::NONE;
    let reads: Vec<Read> = operands
        .iter()
        .map(|operand| operand.read(env, &mut flags))
        .collect();
    if reads.iter().any(|read| matches!(read, Read::Unsupported)) {
        return (default_nan(format, env), 0, flags | Flags::INVALID);
    }
    if reads.iter().any(|read| matches!(read, Read::Nan(_))) {
        let (value, special) = propagate(&reads, format, env);
        return (value, 0, flags | special);
    }
    let number = |read: &Read| match read {
        Read::Number(value) => value.clone(),
        Read::Nan(_) | Read::Unsupported => unreachable!("the special operands have a result"),
    };
    // Each factor as its operands, with a negated subtrahend.
    let factors: Vec<Vec<BigFloat>> = match product {
        Product::Values => reads.iter().map(|read| vec![number(read)]).collect(),
        Product::Sums | Product::Differences => reads
            .as_chunks::<2>()
            .0
            .iter()
            .map(|[first, second]| {
                let second = number(second);
                let second = if product == Product::Differences {
                    -second
                } else {
                    second
                };
                vec![number(first), second]
            })
            .collect(),
    };
    let mut negative = false;
    let (mut zero, mut infinite, mut invalid) = (false, false, false);
    for factor in &factors {
        let infinities: Vec<&BigFloat> =
            factor.iter().filter(|value| value.is_infinite()).collect();
        let factor_negative = if let Some(first) = infinities.first() {
            invalid |= infinities
                .iter()
                .any(|value| value.is_sign_negative() != first.is_sign_negative());
            infinite = true;
            first.is_sign_negative()
        } else {
            let exact = BigFloat::with_val(2, BigFloat::sum(factor.iter()));
            if exact.is_zero() {
                zero = true;
                zero_sign(factor, env)
            } else {
                exact.is_sign_negative()
            }
        };
        negative ^= factor_negative;
    }
    if invalid || (zero && infinite) {
        return (default_nan(format, env), 0, flags | Flags::INVALID);
    }
    if infinite {
        return (infinity(negative, format, env), 0, flags);
    }
    if zero {
        return (Outcome::from_value(format.zero(negative)), 0, flags);
    }
    let mut value = BigFloat::with_val(format.precision, 1);
    let mut scale: i64 = 0;
    for factor in &factors {
        let products: Vec<BigFloat> = factor
            .iter()
            .map(|operand| BigFloat::with_val(2 * format.precision, &value * operand))
            .collect();
        let (mut exact, ordering) =
            BigFloat::with_val_round(working(format), BigFloat::sum(products.iter()), Round::Zero);
        // MPFR's exponent e places the value in [2^(e - 1), 2^e).
        let shift = exact.get_exp().expect("a step is finite and nonzero") - 1;
        exact >>= shift;
        let (rounded, step_flags) = finish(&exact, ordering, format, env);
        if step_flags.contains(Flags::INEXACT) {
            flags |= Flags::INEXACT;
        }
        let Value::Finite(mut rounded) = rounded else {
            panic!("a value in [1, 2] rounds to a finite value");
        };
        let carry = i32::from(rounded.clone().abs() >= 2);
        rounded >>= carry;
        scale = match scale.checked_add(i64::from(shift) + i64::from(carry)) {
            Some(scale) => scale,
            None => return (default_nan(format, env), 0, flags | Flags::INVALID),
        };
        value = rounded;
    }
    (Outcome::Finite(value), scale, flags)
}

/// Returns the sign of an exact zero sum of the operands of a factor: the
/// sign of a zero factor, or the rule of an exact zero sum.
fn zero_sign(factor: &[BigFloat], env: &Env) -> bool {
    let toward_negative = env.rounding == Rounding::TowardNegative;
    if factor.iter().all(BigFloat::is_zero) {
        let negative = factor
            .iter()
            .filter(|value| value.is_sign_negative())
            .count();
        negative == factor.len() || (negative > 0 && toward_negative)
    } else {
        toward_negative
    }
}

/// Returns the default NaN of the NaN rule.
fn default_nan(format: &Format, env: &Env) -> Outcome {
    Outcome::from_value(format.nan(env.nan.default_negative))
}

/// Returns an infinity, or what a format without one gives for it.
fn infinity(negative: bool, format: &Format, env: &Env) -> Outcome {
    Outcome::from_value(format.infinity(negative, env))
}
