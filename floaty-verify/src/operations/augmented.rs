//! An oracle for the augmented operations of IEEE 754-2019 section 9.5:
//! `augmentedAddition`, `augmentedSubtraction`, and
//! `augmentedMultiplication`.
//!
//! IEEE 754-2019 defines the head as the exact result rounded by
//! `roundTiesTowardZero`, and the tail as the exact result minus the head,
//! rounded the same way. E. J. Riedy and J. Demmel, "Augmented Arithmetic
//! Operations Proposed for IEEE-754 2018", ARITH 25 (2018), section III and
//! tables II and III, give the special cases of the draft. A NaN operand or
//! an invalid operation gives one NaN in both results. An infinite operand,
//! a zero head, and an overflow give the head in both results. The pair
//! signals inexact only when `head + tail` differs from the exact result.
//! The `fp-ieee` 0.1.0.6 Haskell package, file `test/AugmentedArithSpec.hs`,
//! gives an exact tail of zero the sign of the head.
//!
//! floaty documents the rest on `floaty::Augmented`. The other fields of the
//! behavior apply to both roundings. The flags describe the pair: the head
//! adds only `TINY`, and `ROUNDED_UP` compares `head + tail` with the exact
//! result.
//!
//! MPFR computes the exact result. `mpfr_sum` rounds the error of the head
//! correctly, however far apart the operands are.

use floaty::format::{Encoding, Standard, Storage, Width};
use floaty::{Augmented, Binary, Env, Flags, Float, Rounding};
use rug::Float as BigFloat;
use rug::float::Round;

use super::{Outcome, special_operands};
use crate::arithmetic::{finish, working};
use crate::mpfr::{Format, Operand, Read, Value};

/// An augmented operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Augmentation {
    /// `augmentedAddition`.
    Addition,
    /// `augmentedSubtraction`.
    Subtraction,
    /// `augmentedMultiplication`.
    Multiplication,
}

impl Augmentation {
    /// Every augmented operation.
    pub const ALL: [Self; 3] = [Self::Addition, Self::Subtraction, Self::Multiplication];

    /// Runs the operation of floaty on `x` and `y` with `env`.
    #[must_use]
    pub fn apply<const E: u32, Enc: Encoding, const W: usize>(
        self,
        x: Float<Binary<E, Enc>, W>,
        y: Float<Binary<E, Enc>, W>,
        env: Env,
    ) -> (Augmented<Float<Binary<E, Enc>, W>>, Flags)
    where
        Width<W>: Storage,
        Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
    {
        match self {
            Self::Addition => x.augmented_add_with(y, env),
            Self::Subtraction => x.augmented_sub_with(y, env),
            Self::Multiplication => x.augmented_mul_with(y, env),
        }
    }
}

/// Returns the expected head, tail, and flags of an augmented operation.
///
/// # Panics
///
/// Panics when an inexact head leaves no error, which does not happen.
#[must_use]
pub fn augmented<const N: usize>(
    operation: Augmentation,
    x: &Operand<N>,
    y: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Outcome, Flags) {
    let env = env.with_rounding(Rounding::TiesTowardZero);
    let mut flags = Flags::NONE;
    let operands = [x.read(&env, &mut flags), y.read(&env, &mut flags)];
    if let Some((nan, special)) = special_operands(&operands, format, &env) {
        return (nan.clone(), nan, flags | special);
    }
    let [Read::Number(a), Read::Number(b)] = &operands else {
        unreachable!("every special operand has a result");
    };
    let b = if operation == Augmentation::Subtraction {
        -b.clone()
    } else {
        b.clone()
    };
    let (head, tail, result) = match operation {
        Augmentation::Addition | Augmentation::Subtraction => add(a, &b, format, &env),
        Augmentation::Multiplication => multiply(a, &b, format, &env),
    };
    (head, tail, flags | result)
}

/// Returns the results of `a + b`.
fn add(a: &BigFloat, b: &BigFloat, format: &Format, env: &Env) -> (Outcome, Outcome, Flags) {
    if a.is_infinite() && b.is_infinite() && a.is_sign_negative() != b.is_sign_negative() {
        return twice(format.nan(env.nan.default_negative), Flags::INVALID);
    }
    if a.is_infinite() || b.is_infinite() {
        let negative = if a.is_infinite() { a } else { b }.is_sign_negative();
        return twice(format.infinity(negative, env), Flags::NONE);
    }
    if a.is_zero() && b.is_zero() {
        // roundTiesTowardZero gives -0 only for the sum of two -0.
        let negative = a.is_sign_negative() && b.is_sign_negative();
        return twice(format.zero(negative), Flags::NONE);
    }
    augment(&[a.clone(), b.clone()], format, env)
}

/// Returns the results of `a * b`.
fn multiply(a: &BigFloat, b: &BigFloat, format: &Format, env: &Env) -> (Outcome, Outcome, Flags) {
    let negative = a.is_sign_negative() != b.is_sign_negative();
    if (a.is_infinite() && b.is_zero()) || (a.is_zero() && b.is_infinite()) {
        return twice(format.nan(env.nan.default_negative), Flags::INVALID);
    }
    if a.is_infinite() || b.is_infinite() {
        return twice(format.infinity(negative, env), Flags::NONE);
    }
    if a.is_zero() || b.is_zero() {
        return twice(format.zero(negative), Flags::NONE);
    }
    // Two significands of p bits give an exact product of 2p bits.
    let product = BigFloat::with_val(2 * format.precision, a * b);
    augment(&[product], format, env)
}

/// Returns one result in both fields.
fn twice(value: Value, flags: Flags) -> (Outcome, Outcome, Flags) {
    let outcome = Outcome::from_value(value);
    (outcome.clone(), outcome, flags)
}

/// Returns the head and the tail of the exact sum of `terms`.
fn augment(terms: &[BigFloat], format: &Format, env: &Env) -> (Outcome, Outcome, Flags) {
    let sum_of = |terms: &[BigFloat]| {
        BigFloat::with_val_round(working(format), BigFloat::sum(terms.iter()), Round::Zero)
    };
    let (value, ordering) = sum_of(terms);
    if value.is_zero() {
        // An exact zero sum of nonzero values is +0 in roundTiesTowardZero.
        return twice(format.zero(false), Flags::NONE);
    }
    let (head, head_flags) = finish(&value, ordering, format, env);
    let head_value = match &head {
        Value::Finite(head_value) if !head_flags.contains(Flags::OVERFLOW) => head_value.clone(),
        _ => return twice(head, head_flags),
    };
    if !head_flags.contains(Flags::INEXACT) {
        let tail = format.zero(head_value.is_sign_negative());
        return (
            Outcome::from_value(head),
            Outcome::from_value(tail),
            head_flags,
        );
    }
    let mut error_terms = terms.to_vec();
    error_terms.push(-head_value.clone());
    let (error, ordering) = sum_of(&error_terms);
    assert!(!error.is_zero(), "an inexact head leaves a nonzero error");
    let (tail, tail_flags) = finish(&error, ordering, format, env);
    let tail_value = match &tail {
        Value::Finite(tail_value) => tail_value.clone(),
        _ => BigFloat::new(working(format)),
    };
    // The sign of `head + tail - exact` is exact at any precision.
    let mut excess_terms = vec![head_value.clone(), tail_value];
    excess_terms.extend(terms.iter().map(|term| -term.clone()));
    let excess = BigFloat::with_val(working(format), BigFloat::sum(excess_terms.iter()));
    let rounded_up =
        !excess.is_zero() && excess.is_sign_negative() == head_value.is_sign_negative();
    let mut flags = tail_flags.difference(Flags::ROUNDED_UP);
    if head_flags.contains(Flags::TINY) {
        flags |= Flags::TINY;
    }
    if rounded_up {
        flags |= Flags::ROUNDED_UP;
    }
    (Outcome::from_value(head), Outcome::from_value(tail), flags)
}
