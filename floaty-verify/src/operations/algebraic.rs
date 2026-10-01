//! An oracle for the algebraic functions of IEEE 754-2019 section 9.2 on the
//! binary formats: `hypot`, `rSqrt`, `pown`, and `rootn`.
//!
//! MPFR's `mpfr_hypot`, `mpfr_rec_sqrt`, `mpfr_pow_si`, and `mpfr_rootn_si`
//! give the exact result rounded toward zero with the ternary value, and
//! [`crate::mpfr::round`] rounds it. IEEE 754-2019 section 9.2.1 gives the
//! special cases. floaty documents the rest: past `|n| = 64`, `pown` rounds
//! each product of squaring, and `rootn` gives the default NaN.

use floaty::{Env, Flags};
use rug::Float as BigFloat;
use rug::float::Round;
use rug::ops::Pow;

use super::{Outcome, propagate, special_operands};
use crate::arithmetic::{finish, working};
use crate::mpfr::{self, Format, Input, Operand, Read, Value};

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

/// The largest `|n|` that `pown` and `rootn` round once.
const EXACT_LIMIT: u64 = 64;

/// The bound of floaty on the exponent of a result past the range of every
/// format.
const EXPONENT_CLAMP: i128 = 1 << 30;

/// Returns the expected `pown` of an operand, and the flags.
///
/// # Panics
///
/// Panics when a step rounds to a value that is not finite, which does not
/// happen in `[1, 2]`.
#[must_use]
pub fn pown<const N: usize>(
    x: &Operand<N>,
    n: i64,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [x.read(env, &mut flags)];
    let special = reads[0].is_signaling() || matches!(reads[0], Read::Unsupported);
    if n == 0 && !special {
        let one = BigFloat::with_val(format.precision, 1);
        return (Outcome::Finite(one), flags);
    }
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    let [Read::Number(value)] = &reads else {
        unreachable!("every special operand has a result");
    };
    let negative = value.is_sign_negative() && n % 2 != 0;
    if value.is_zero() {
        if n > 0 {
            return (Outcome::from_value(format.zero(negative)), flags);
        }
        let infinity = format.infinity(negative, env);
        return (Outcome::from_value(infinity), flags | Flags::DIVIDE_BY_ZERO);
    }
    if value.is_infinite() {
        let result = if n > 0 {
            format.infinity(negative, env)
        } else {
            format.zero(negative)
        };
        return (Outcome::from_value(result), flags);
    }
    let (result, round_flags) = if n.unsigned_abs() <= EXACT_LIMIT {
        let power = i32::try_from(n).expect("the power is at most 64");
        let (exact, ordering) =
            BigFloat::with_val_round(working(format), value.pow(power), Round::Zero);
        finish(&exact, ordering, format, env)
    } else {
        pown_steps(value, n, format, env)
    };
    (Outcome::from_value(result), flags | round_flags)
}

/// Returns `x^n` for `|n| > 64` by the rule of floaty: squaring from the top
/// bit of `|n|`, each product rounded to the precision without an exponent
/// limit, the last product of a positive `n` rounded into the format, and the
/// reciprocal of the power rounded into the format for a negative `n`.
fn pown_steps(x: &BigFloat, n: i64, format: &Format, env: &Env) -> (Value, Flags) {
    // The value scaled exactly into [1, 2), and the scale.
    let scaled = |value: &BigFloat| {
        let shift = value
            .get_exp()
            .expect("a finite nonzero value has an exponent")
            - 1;
        (
            BigFloat::with_val(value.prec(), value >> shift),
            i128::from(shift),
        )
    };
    let base = scaled(x);
    let mut value = base.clone();
    let mut inexact = false;
    let power = n.unsigned_abs();
    let bits = u64::BITS - power.leading_zeros();
    let mut step = |(first, first_scale): &(BigFloat, i128),
                    (second, second_scale): &(BigFloat, i128),
                    last: bool| {
        let product = BigFloat::with_val(2 * format.precision, first * second);
        let scale = first_scale + second_scale;
        if last {
            return Err(round_scaled(&product, scale, false, format, env));
        }
        let (normalized, shift) = scaled(&product);
        let (rounded, step_flags) = finish(&normalized, core::cmp::Ordering::Equal, format, env);
        inexact |= step_flags.contains(Flags::INEXACT);
        let Value::Finite(rounded) = rounded else {
            panic!("a value in [1, 2] rounds to a finite value");
        };
        let (rounded, carry) = scaled(&rounded);
        Ok((rounded, scale + shift + carry))
    };
    let with_inexact = |(result, flags): (Value, Flags), inexact: bool| {
        let earlier = if inexact { Flags::INEXACT } else { Flags::NONE };
        (result, flags | earlier)
    };
    for bit in (0..bits - 1).rev() {
        let multiply = power >> bit & 1 == 1;
        let last = bit == 0 && n > 0;
        value = match step(&value, &value, !multiply && last) {
            Ok(next) => next,
            Err(result) => return with_inexact(result, inexact),
        };
        if multiply {
            value = match step(&value, &base, last) {
                Ok(next) => next,
                Err(result) => return with_inexact(result, inexact),
            };
        }
    }
    let (mantissa, scale) = value;
    let reciprocal = BigFloat::with_val(working(format), 1);
    let (reciprocal, ordering) =
        BigFloat::with_val_round(working(format), &reciprocal / &mantissa, Round::Zero);
    let sticky = ordering != core::cmp::Ordering::Equal;
    with_inexact(
        round_scaled(&reciprocal, -scale, sticky, format, env),
        inexact,
    )
}

/// Rounds `value * 2^scale` into the format, with a sticky bit, and clamps
/// the exponent as floaty does.
fn round_scaled(
    value: &BigFloat,
    scale: i128,
    sticky: bool,
    format: &Format,
    env: &Env,
) -> (Value, Flags) {
    let (integer, exponent) = value
        .to_integer_exp()
        .expect("the value is finite and nonzero");
    let exponent = (i128::from(exponent) + scale).clamp(-EXPONENT_CLAMP, EXPONENT_CLAMP);
    let input = Input {
        negative: value.is_sign_negative(),
        exponent: i32::try_from(exponent).expect("the clamp fits an i32"),
        significand: integer.abs(),
        sticky,
    };
    mpfr::round(&input, format, env)
}

/// Returns the expected `rootn` of an operand, and the flags.
///
/// # Panics
///
/// Never: a root that MPFR computes is at most 64 in magnitude.
#[must_use]
pub fn rootn<const N: usize>(
    x: &Operand<N>,
    n: i64,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [x.read(env, &mut flags)];
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    if n == 0 || n.unsigned_abs() > EXACT_LIMIT {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    let [Read::Number(value)] = &reads else {
        unreachable!("every special operand has a result");
    };
    let odd = n % 2 != 0;
    let negative = value.is_sign_negative();
    if value.is_zero() {
        if n > 0 {
            return (Outcome::from_value(format.zero(negative && odd)), flags);
        }
        let infinity = format.infinity(negative && odd, env);
        return (Outcome::from_value(infinity), flags | Flags::DIVIDE_BY_ZERO);
    }
    if negative && !odd {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    if value.is_infinite() {
        let result = if n > 0 {
            format.infinity(negative, env)
        } else {
            format.zero(negative)
        };
        return (Outcome::from_value(result), flags);
    }
    let root = i32::try_from(n).expect("the root is at most 64");
    let (exact, ordering) =
        BigFloat::with_val_round(working(format), value.root_i_ref(root), Round::Zero);
    let (result, round_flags) = finish(&exact, ordering, format, env);
    (Outcome::from_value(result), flags | round_flags)
}

/// Returns the default NaN of the NaN rule.
fn default_nan(format: &Format, env: &Env) -> Outcome {
    Outcome::from_value(format.nan(env.nan.default_negative))
}
