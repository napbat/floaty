//! An oracle for the algebraic functions of IEEE 754-2019 section 9.2 on the
//! binary formats: `hypot`, `rSqrt`, `pown`, `rootn`, and `compound`.
//!
//! MPFR's `mpfr_hypot`, `mpfr_rec_sqrt`, `mpfr_pow_si`, and `mpfr_rootn_si`
//! give the exact result rounded toward zero with the ternary value, and
//! [`crate::mpfr::round`] rounds it. IEEE 754-2019 section 9.2.1 gives the
//! special cases. floaty documents the rest: past `|n| = 64`, `pown` rounds
//! each product of squaring, and `rootn` gives the default NaN.
//!
//! [`compound`] computes `(1 + x)^n` exactly in GMP when the power has few
//! bits, and bounds it with MPFR otherwise.

use floaty::{Env, Flags};
use rug::float::Round;
use rug::ops::Pow;
use rug::{Float as BigFloat, Integer};

use super::elementary::truncation;
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

/// The bits of an exact power `M^|n|` up to which the `compound` oracle
/// computes it in GMP.
const COMPOUND_EXACT_BITS: u64 = 1 << 16;

/// The bound of the oracle on the exponent of an exact power. A clamped
/// exponent rounds as the true one does.
const COMPOUND_CLAMP: i128 = 1 << 28;

/// Returns the expected `compound(x, n) = (1 + x)^n` of an operand, and the
/// flags.
///
/// The special cases follow IEEE 754-2019 section 9.2.1, as
/// `mpfr_compound_si` does: `n = 0` gives 1 for a quiet NaN and for every
/// value at or above -1, a value below -1 gives the default NaN with
/// invalid, and -1 gives +0 for `n > 0` and +inf with divide-by-zero for
/// `n < 0`. Every other result is an exact power in GMP, the exact power
/// `x^n` of a large `x` with a sticky bit, or a power that MPFR bounds.
#[must_use]
pub fn compound<const N: usize>(
    x: &Operand<N>,
    n: i64,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [x.read(env, &mut flags)];
    let one = || Outcome::Finite(BigFloat::with_val(1, 1));
    if n == 0 && matches!(&reads[0], Read::Nan(nan) if !nan.signaling) {
        return (one(), flags);
    }
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    let [Read::Number(value)] = &reads else {
        unreachable!("every special operand has a result");
    };
    if *value < -1 {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    if n == 0 || value.is_zero() {
        return (one(), flags);
    }
    if value.is_infinite() {
        let result = if n > 0 {
            format.infinity(false, env)
        } else {
            format.zero(false)
        };
        return (Outcome::from_value(result), flags);
    }
    if *value == -1 {
        if n > 0 {
            return (Outcome::from_value(format.zero(false)), flags);
        }
        let infinity = format.infinity(false, env);
        return (Outcome::from_value(infinity), flags | Flags::DIVIDE_BY_ZERO);
    }
    let (result, round_flags) = mpfr::round(&compound_input(value, n, format), format, env);
    (Outcome::from_value(result), flags | round_flags)
}

/// Returns `(1 + x)^n` of a finite `x > -1` other than zero and an `n` other
/// than zero, as an input of the rounding oracle.
///
/// With `1 + x = M 2^E` and `M` odd, a power of two is exact. A power
/// `M^|n|` of at most [`COMPOUND_EXACT_BITS`] bits is exact in GMP: `M^n
/// 2^(E n)` for `n > 0`, and the quotient `2^s / M^|n|` with a sticky bit for
/// `n < 0`. Every other power is off the grid of `p + 3` bits, so bounds
/// decide its truncation: `e^(n log1p(x))` with `mpfr_log1p`, `mpfr_mul_si`,
/// and `mpfr_exp` rounded in the direction of each bound, at a precision
/// that doubles until the bounds agree.
///
/// # Panics
///
/// Panics when `x` is not finite, or not above -1.
fn compound_input(x: &BigFloat, n: i64, format: &Format) -> Input {
    let sum = x.to_rational().expect("a finite value is rational") + 1u32;
    let (numerator, denominator) = sum.into_numer_denom();
    let zeros = numerator.find_one(0).expect("1 + x is above zero");
    let odd = numerator >> zeros;
    let exponent = i128::from(zeros) - i128::from(denominator.significant_bits() - 1);
    let scale = exponent * i128::from(n);
    if odd == 1 {
        return Input {
            negative: false,
            exponent: clamp_exponent(scale),
            significand: Integer::from(1),
            sticky: false,
        };
    }
    let count = n.unsigned_abs();
    if u64::from(odd.significant_bits()).saturating_mul(count) <= COMPOUND_EXACT_BITS {
        let power = odd.pow(u32::try_from(count).expect("an exact power has a small count"));
        if n > 0 {
            return Input {
                negative: false,
                exponent: clamp_exponent(scale),
                significand: power,
                sticky: false,
            };
        }
        let shift = power.significant_bits() + format.precision + 3;
        let (quotient, rest) = (Integer::from(1) << shift).div_rem(power);
        return Input {
            negative: false,
            exponent: clamp_exponent(scale - i128::from(shift)),
            significand: quotient,
            sticky: rest != 0,
        };
    }
    if let Some(input) = dominant_input(x, n, format) {
        return input;
    }
    let estimate = BigFloat::with_val(64, x.ln_1p_ref()) * n;
    // Past `|n ln(1 + x)| = 4 R`, `(1 + x)^n` lies past `2^(±4R)`, outside
    // the range of the format.
    if estimate.clone().abs() >= 4 * format.reach() {
        // A value far past the range: `2^(p + 2)` times `2^(±2^28)`.
        return Input {
            negative: false,
            exponent: if estimate.is_sign_negative() {
                -(1 << 28)
            } else {
                1 << 28
            },
            significand: Integer::from(1) << (format.precision + 2),
            sticky: true,
        };
    }
    // Conflict: MPFR 4.2.2 `mpfr_compound_si` gives a wrong bound in a hard
    // case. For x = (2^205 + 1) 2^43485 and n = -3, `(1 + x)^-3` lies about
    // 4.3e-42 ulp above a number of 270 bits, by exact GMP rationals. At 270
    // and at 1080 bits, `mpfr_compound_si` toward +inf returns that number,
    // below the exact value, with a positive ternary value, and toward -inf
    // one ulp lower. Resolution: the oracle takes its bounds from
    // `mpfr_log1p`, `mpfr_mul_si`, and `mpfr_exp`, which give correct
    // directed bounds, and does not call `mpfr_compound_si`.
    truncation(format.precision, |precision, round| {
        // `e^t` increases with `t = n ln(1 + x)`, which increases with
        // `ln(1 + x)` for `n > 0` and decreases for `n < 0`.
        let inner = match (n > 0, round) {
            (false, Round::Up) => Round::Down,
            (false, _) => Round::Up,
            (true, round) => round,
        };
        let log = BigFloat::with_val_round(precision, x.ln_1p_ref(), inner).0;
        let t = BigFloat::with_val_round(precision, &log * n, round).0;
        BigFloat::with_val_round(precision, t.exp_ref(), round).0
    })
}

/// Returns an exponent clamped to [`COMPOUND_CLAMP`].
fn clamp_exponent(exponent: i128) -> i32 {
    i32::try_from(exponent.clamp(-COMPOUND_CLAMP, COMPOUND_CLAMP)).expect("the clamp fits an i32")
}

/// Returns `(1 + x)^n` for a large `x`, from the exact power `x^n`, or
/// `None` when `x` is not large enough or `x^n` has too many bits.
///
/// `(1 + x)^n` is `x^n (1 + d)`, where `d` has the sign of `n` and `|d|` is
/// below `2 |n| / x`. With `x = m 2^e`, `m` odd, and `x` at least
/// `2^(bits(m^|n|) + p + 8 + bits(n))`, `|d|` is below
/// `2^-(bits(m^|n|) + p + 7)`. Then `x^n` scaled to an integer of at least
/// `p + 3` bits stays within one unit of `(1 + x)^n`: it is a lower bound for
/// `n > 0`, and the quotient `2^s / m^|n|` or one below it, at a power of
/// two, is for `n < 0`. So that integer with the sticky bit rounds as
/// `(1 + x)^n` does.
fn dominant_input(x: &BigFloat, n: i64, format: &Format) -> Option<Input> {
    if x.is_sign_negative() {
        return None;
    }
    let (integer, exponent) = x.to_integer_exp()?;
    let zeros = integer.find_one(0)?;
    let odd = integer >> zeros;
    let count = n.unsigned_abs();
    if u64::from(odd.significant_bits()).saturating_mul(count) > COMPOUND_EXACT_BITS {
        return None;
    }
    let power = odd.pow(u32::try_from(count).ok()?);
    let bits = power.significant_bits();
    let count_bits = u64::BITS - count.leading_zeros();
    let top = i64::from(x.get_exp()?) - 1;
    let bound = i64::from(bits) + i64::from(format.precision) + 8 + i64::from(count_bits);
    if top < bound {
        return None;
    }
    let scale = (i128::from(exponent) + i128::from(zeros)) * i128::from(n);
    Some(if n > 0 {
        let shift = format.precision + 5;
        Input {
            negative: false,
            exponent: clamp_exponent(scale - i128::from(shift)),
            significand: power << shift,
            sticky: true,
        }
    } else {
        let shift = bits + format.precision + 3;
        let (quotient, rest) = (Integer::from(1) << shift).div_rem(power);
        Input {
            negative: false,
            exponent: clamp_exponent(scale - i128::from(shift)),
            significand: if rest == 0 { quotient - 1u32 } else { quotient },
            sticky: true,
        }
    })
}
