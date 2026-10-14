//! The oracle of `y log2(x)` and `y log2(1 + x)` on the binary formats,
//! which floaty's `mul_log2_with` and `mul_log2_p1_with` round once.
//!
//! The oracle states the special cases of floaty's documentation itself: the
//! cases of the logarithm and of the product together, with divide-by-zero
//! for an infinity from a finite `y` at the pole. MPFR's `log2` and
//! `log2p1` bound the logarithm at a working precision, and the product
//! with the exact `y` rounds outward, so the bounds hold the value. The
//! working precision doubles until the bounds decide the truncation. A
//! power of 2 gives an exact logarithm, where the bounds meet at the exact
//! product.
//!
//! `log2(1 + 2^k)` lies above `k` by less than `2^-k / ln 2`, so MPFR needs
//! more than `k` bits there. For `k` from `p + 11`, the oracle takes the
//! bounds `k` and `k + 2^-(p + 10)` instead, whose truncations of the
//! product agree.

use floaty::{Env, Flags};
use rug::Float as BigFloat;
use rug::float::Round;

use super::truncation;
use crate::mpfr::{self, Format, Operand, Read};
use crate::operations::{Outcome, default_nan, special_operands};

/// The logarithm of an operand that is not a NaN.
enum Logarithm {
    /// An exact zero with the sign.
    Zero(bool),
    /// Negative infinity at a pole.
    Pole,
    /// Positive infinity.
    Infinity,
    /// The logarithm of a negative number.
    Invalid,
    /// A finite nonzero logarithm with the sign.
    Finite(bool),
}

/// Returns `log2(x)`, or `log2(1 + x)` when `plus_one` is set, as a
/// [`Logarithm`].
fn logarithm(plus_one: bool, x: &BigFloat) -> Logarithm {
    let negative = x.is_sign_negative();
    if x.is_zero() {
        return if plus_one {
            Logarithm::Zero(negative)
        } else {
            Logarithm::Pole
        };
    }
    if x.is_infinite() {
        return if negative {
            Logarithm::Invalid
        } else {
            Logarithm::Infinity
        };
    }
    if !plus_one {
        return if negative {
            Logarithm::Invalid
        } else if *x == 1 {
            Logarithm::Zero(false)
        } else {
            Logarithm::Finite(*x < 1)
        };
    }
    if *x < -1 {
        Logarithm::Invalid
    } else if *x == -1 {
        Logarithm::Pole
    } else {
        Logarithm::Finite(negative)
    }
}

/// Returns the expected result of `y log2(x)`, or of `y log2(1 + x)` when
/// `plus_one` is set, and the flags.
#[must_use]
pub fn expected_log_product<const N: usize>(
    plus_one: bool,
    y: &Operand<N>,
    x: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [y.read(env, &mut flags), x.read(env, &mut flags)];
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    let [Read::Number(y), Read::Number(x)] = &reads else {
        unreachable!("every special operand has a result");
    };
    let y_negative = y.is_sign_negative();
    let zero = |negative| Outcome::from_value(format.zero(negative));
    let infinity = |negative| Outcome::from_value(format.infinity(negative, env));
    let invalid = (default_nan(format, env), flags | Flags::INVALID);
    match (logarithm(plus_one, x), y.is_zero(), y.is_infinite()) {
        (Logarithm::Invalid, ..)
        | (Logarithm::Zero(_), _, true)
        | (Logarithm::Pole | Logarithm::Infinity, true, _) => invalid,
        (Logarithm::Zero(negative), ..) | (Logarithm::Finite(negative), true, _) => {
            (zero(y_negative != negative), flags)
        }
        (Logarithm::Pole, _, false) => (infinity(!y_negative), flags | Flags::DIVIDE_BY_ZERO),
        (Logarithm::Pole, _, true) => (infinity(!y_negative), flags),
        (Logarithm::Infinity, ..) => (infinity(y_negative), flags),
        (Logarithm::Finite(negative), _, true) => (infinity(y_negative != negative), flags),
        (Logarithm::Finite(_), false, false) => {
            let input = truncation(format.precision, |precision, round| {
                bound(plus_one, y, x, (precision, format.precision), round)
            });
            let (value, more) = mpfr::round(&input, format, env);
            (Outcome::from_value(value), flags | more)
        }
    }
}

/// Returns a bound of `y log2(x)`, or of `y log2(1 + x)`, at the working
/// precision: below the value for `Round::Down`, and above it for
/// `Round::Up`. `precision` holds the working precision and the precision of
/// the format.
fn bound(
    plus_one: bool,
    y: &BigFloat,
    x: &BigFloat,
    (working, format_precision): (u32, u32),
    round: Round,
) -> BigFloat {
    // A negative `y` takes the other bound of the logarithm.
    let toward = if (round == Round::Down) == y.is_sign_negative() {
        Round::Up
    } else {
        Round::Down
    };
    let log = if let Some(k) = plus_one
        .then(|| power_exponent(x, format_precision))
        .flatten()
    {
        let k = BigFloat::with_val(working, k);
        if toward == Round::Down {
            k
        } else {
            k + (BigFloat::with_val(working, 1) >> (format_precision + 10))
        }
    } else {
        let mut log = BigFloat::with_val(working, x);
        let _ = if plus_one {
            log.log2_1p_round(toward)
        } else {
            log.log2_round(toward)
        };
        log
    };
    BigFloat::with_val_round(working, &log * y, round).0
}

/// Returns `k` for `x = 2^k` with `k >= p + 11`, where `log2(1 + x)` lies
/// within `2^-(p + 10)` above `k`.
fn power_exponent(x: &BigFloat, precision: u32) -> Option<i32> {
    let (significand, exponent) = x.to_integer_exp()?;
    let length = i32::try_from(significand.significant_bits()).ok()?;
    let k = exponent + length - 1;
    let threshold = i32::try_from(precision + 11).ok()?;
    (significand.is_power_of_two() && k >= threshold).then_some(k)
}
