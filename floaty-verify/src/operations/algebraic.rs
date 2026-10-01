//! An oracle for the algebraic functions of IEEE 754-2019 section 9.2 on the
//! binary formats: `hypot` and `rSqrt`.
//!
//! MPFR's `mpfr_hypot` and `mpfr_rec_sqrt` give the exact result rounded
//! toward zero with the ternary value, and [`crate::mpfr::round`] rounds it.
//! IEEE 754-2019 section 9.2.1 gives the special cases: `hypot` of an
//! infinity is +inf even with a quiet NaN, and `rSqrt` of a zero is an
//! infinity of its sign with divide-by-zero.

use floaty::{Env, Flags};
use rug::Float as BigFloat;
use rug::float::Round;

use super::{Outcome, propagate, special_operands};
use crate::arithmetic::{finish, working};
use crate::mpfr::{Format, Operand, Read};

/// Returns the expected `hypot` of two operands, and the flags.
#[must_use]
pub fn hypot<const N: usize>(
    x: &Operand<N>,
    y: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [x.read(env, &mut flags), y.read(env, &mut flags)];
    if reads.iter().any(|read| matches!(read, Read::Unsupported)) {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    let infinite = reads
        .iter()
        .any(|read| matches!(read, Read::Number(value) if value.is_infinite()));
    let nan = reads.iter().any(|read| matches!(read, Read::Nan(_)));
    if reads.iter().any(Read::is_signaling) || (nan && !infinite) {
        let (value, special) = propagate(&reads, format, env);
        return (value, flags | special);
    }
    if infinite {
        return (Outcome::from_value(format.infinity(false, env)), flags);
    }
    let [Read::Number(a), Read::Number(b)] = &reads else {
        unreachable!("every special operand has a result");
    };
    if a.is_zero() && b.is_zero() {
        return (Outcome::from_value(format.zero(false)), flags);
    }
    let (value, ordering) = BigFloat::with_val_round(working(format), a.hypot_ref(b), Round::Zero);
    let (value, round_flags) = finish(&value, ordering, format, env);
    (Outcome::from_value(value), flags | round_flags)
}

/// Returns the expected `rSqrt` of an operand, and the flags.
#[must_use]
pub fn reciprocal_sqrt<const N: usize>(
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
    if value.is_zero() {
        let infinity = format.infinity(value.is_sign_negative(), env);
        return (Outcome::from_value(infinity), flags | Flags::DIVIDE_BY_ZERO);
    }
    if value.is_sign_negative() {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    if value.is_infinite() {
        return (Outcome::from_value(format.zero(false)), flags);
    }
    let (root, ordering) =
        BigFloat::with_val_round(working(format), value.recip_sqrt_ref(), Round::Zero);
    let (root, round_flags) = finish(&root, ordering, format, env);
    (Outcome::from_value(root), flags | round_flags)
}

/// Returns the default NaN of the NaN rule.
fn default_nan(format: &Format, env: &Env) -> Outcome {
    Outcome::from_value(format.nan(env.nan.default_negative))
}
