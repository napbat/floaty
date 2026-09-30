//! The MPFR oracle of a conversion from a binary value to a decimal format.
//!
//! MPFR gives the leading digits of the exact binary value, truncated and
//! rounded away from zero, and the digit after them. The rounding directions
//! follow from those digits by their IEEE 754 definitions, and IBM's round
//! to prepare for shorter precision for `ToOdd`. The exponent follows the
//! IEEE 754 preferred exponent rules, with preferred exponent 0.

use floaty::{Decoded, Env, Flags, Rounding};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

use super::{Parameters, limit_precision, overflows_to_infinity};

/// The parameters of a decimal format.
#[derive(Clone, Copy, Debug)]
pub struct DecimalFormat {
    /// The precision in digits.
    pub precision: u32,
    /// The largest adjusted exponent.
    pub emax: i32,
}

impl DecimalFormat {
    /// Returns the parameters of the decimal type `T`.
    #[must_use]
    pub fn of<T: Parameters>() -> Self {
        Self {
            precision: T::PRECISION,
            emax: T::EMAX,
        }
    }

    /// Returns the width `t` of the trailing significand field, by IEEE
    /// 754-2019 Table 3.6: 20, 50, or 110 bits.
    ///
    /// # Panics
    ///
    /// Panics for a precision that no decimal interchange format has.
    #[must_use]
    pub fn trailing_bits(self) -> u32 {
        match self.precision {
            7 => 20,
            16 => 50,
            34 => 110,
            _ => panic!("a decimal interchange format has 7, 16, or 34 digits"),
        }
    }

    /// Returns the largest canonical NaN payload, `10^(p - 1) - 1`.
    #[must_use]
    pub fn largest_payload(self) -> Integer {
        Integer::from(Integer::u_pow_u(10, self.precision - 1)) - 1u32
    }

    /// Returns the smallest and the largest exponent of a full coefficient.
    #[must_use]
    pub fn exponents(self) -> (i64, i64) {
        let digits = i64::from(self.precision);
        (
            i64::from(1 - self.emax) - digits + 1,
            i64::from(self.emax) - digits + 1,
        )
    }
}

/// A decimal result in a form that compares across implementations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecimalValue {
    /// A zero with its exponent.
    Zero {
        /// The sign.
        negative: bool,
        /// The exponent, the quantum of the zero.
        exponent: i64,
    },
    /// A nonzero finite value, `coefficient * 10^exponent`.
    Finite {
        /// The sign.
        negative: bool,
        /// The coefficient.
        coefficient: Integer,
        /// The exponent.
        exponent: i64,
    },
    /// An infinity.
    Infinity {
        /// The sign.
        negative: bool,
    },
    /// A quiet NaN with its payload.
    Nan {
        /// The sign.
        negative: bool,
        /// The payload value.
        payload: Integer,
    },
}

impl DecimalValue {
    /// Converts a decoded floaty decimal value.
    ///
    /// # Panics
    ///
    /// Panics for a signaling NaN, which a conversion never gives, and for an
    /// unsupported encoding, which a decimal format does not have.
    #[must_use]
    pub fn from_decoded(decoded: Decoded<2>) -> Self {
        match decoded {
            Decoded::Zero { negative, exponent } => Self::Zero {
                negative,
                exponent: i64::from(exponent),
            },
            Decoded::Finite {
                negative,
                exponent,
                significand,
            } => Self::Finite {
                negative,
                coefficient: Integer::from_digits(&significand, Order::Lsf),
                exponent: i64::from(exponent),
            },
            Decoded::Infinity { negative } => Self::Infinity { negative },
            Decoded::Nan {
                negative,
                signaling,
                payload,
            } => {
                assert!(!signaling, "a conversion quiets a NaN");
                Self::Nan {
                    negative,
                    payload: Integer::from_digits(&payload, Order::Lsf),
                }
            }
            Decoded::Unsupported => panic!("a decimal format has no unsupported encoding"),
        }
    }
}

/// Where the dropped digits lie against half a unit of the last kept digit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dropped {
    Nothing,
    BelowHalf,
    Half,
    AboveHalf,
}

/// Returns the `digits` leading digits of `value` as an integer, truncated,
/// and `true` when they hold `value` exactly. MPFR writes `|value|` as
/// `0.d1 d2 ... * 10^exponent`; the function also returns that exponent.
fn leading_digits(value: &BigFloat, digits: usize) -> (Integer, i64, bool) {
    let (_, toward, exponent) = value.to_sign_string_exp_round(10, Some(digits), Round::Zero);
    let (_, away, away_exponent) =
        value.to_sign_string_exp_round(10, Some(digits), Round::AwayZero);
    let exponent = exponent.expect("a nonzero finite value has an exponent");
    let integer = Integer::from_str_radix(&toward, 10).expect("MPFR writes decimal digits");
    let exact = toward == away && Some(exponent) == away_exponent;
    (integer, i64::from(exponent), exact)
}

/// Returns the leading `kept_digits` digits of `exact`, truncated, and where
/// the dropped digits lie against half a unit of the last kept digit.
fn cut(exact: &BigFloat, kept_digits: i64) -> (Integer, Dropped) {
    if kept_digits < 0 {
        return (Integer::ZERO, Dropped::BelowHalf);
    }
    if kept_digits == 0 {
        let (first, _, exact_first) = leading_digits(exact, 1);
        let dropped = match (first.to_u32().expect("a digit fits a u32"), exact_first) {
            (5, true) => Dropped::Half,
            (0..=4, _) => Dropped::BelowHalf,
            _ => Dropped::AboveHalf,
        };
        return (Integer::ZERO, dropped);
    }
    let count = usize::try_from(kept_digits).expect("a digit count fits a usize");
    let (kept, _, exact_digits) = leading_digits(exact, count);
    let (longer, _, exact_longer) = leading_digits(exact, count + 1);
    let next = (longer % 10u32).to_u32().expect("a digit fits a u32");
    let dropped = match (exact_digits, next, exact_longer) {
        (true, _, _) => Dropped::Nothing,
        (false, 5, true) => Dropped::Half,
        (false, 0..=4, _) => Dropped::BelowHalf,
        _ => Dropped::AboveHalf,
    };
    (kept, dropped)
}

/// Returns `true` when the rounding direction adds one unit to the kept
/// digits, by the definition of the direction.
fn rounds_up(rounding: Rounding, negative: bool, dropped: Dropped, kept: &Integer) -> bool {
    let inexact = dropped != Dropped::Nothing;
    let last = (kept.clone() % 10u32).to_u32().expect("a digit fits a u32");
    match rounding {
        Rounding::TiesToEven => {
            dropped == Dropped::AboveHalf || (dropped == Dropped::Half && last % 2 == 1)
        }
        Rounding::TiesToAway => matches!(dropped, Dropped::Half | Dropped::AboveHalf),
        Rounding::TiesTowardZero => dropped == Dropped::AboveHalf,
        Rounding::TowardPositive => inexact && !negative,
        Rounding::TowardNegative => inexact && negative,
        Rounding::TowardZero => false,
        Rounding::AwayFromZero => inexact,
        // IBM's round to prepare for shorter precision.
        Rounding::ToOdd => inexact && (last == 0 || last == 5),
        _ => panic!("the oracle knows every rounding direction"),
    }
}

/// Returns the expected decimal result of converting a nonzero finite
/// binary value, `exact`, and the flags.
///
/// # Panics
///
/// Panics for a zero, an infinity, or a NaN.
#[must_use]
pub fn to_decimal(exact: &BigFloat, format: DecimalFormat, env: &Env) -> (DecimalValue, Flags) {
    let negative = exact.is_sign_negative();
    let precision = limit_precision(format.precision, env);
    let emin = i64::from(1 - format.emax);
    let (full_lowest, full_highest) = format.exponents();
    let (_, top_plus_one, _) = leading_digits(exact, 1);
    let top = top_plus_one - 1;
    let position = (top - i64::from(precision) + 1).max(emin - i64::from(precision) + 1);
    let (kept, dropped) = cut(exact, top - position + 1);
    let inexact = dropped != Dropped::Nothing;
    let step = rounds_up(env.rounding, negative, dropped, &kept);
    let (mut coefficient, mut exponent) = (kept + u32::from(step), position);
    if coefficient == Integer::from(Integer::u_pow_u(10, precision)) {
        coefficient /= 10u32;
        exponent += 1;
    }
    let tiny = top < emin;
    let mut flags = Flags::NONE;
    let zero = DecimalValue::Zero {
        negative,
        exponent: full_lowest,
    };
    if tiny {
        flags |= Flags::TINY;
        if env.flush_to_zero {
            return (zero, flags | Flags::UNDERFLOW | Flags::INEXACT);
        }
    }
    if inexact {
        flags |= Flags::INEXACT;
        if tiny {
            flags |= Flags::UNDERFLOW;
        }
    }
    if step {
        flags |= Flags::ROUNDED_UP;
    }
    if coefficient == 0 {
        return (zero, flags);
    }
    let digits = |value: &Integer| i64::try_from(value.to_string().len()).expect("a count fits");
    if exponent + digits(&coefficient) - 1 > i64::from(format.emax) {
        return overflow(negative, precision, format, env);
    }
    let full = i64::from(format.precision);
    if inexact {
        // An inexact result has the least possible exponent.
        while exponent > full_lowest && digits(&coefficient) < full {
            coefficient *= 10u32;
            exponent -= 1;
        }
    } else {
        // An exact result has the exponent nearest the preferred exponent 0.
        while exponent < 0 && coefficient.is_divisible_u(10) {
            coefficient /= 10u32;
            exponent += 1;
        }
        while exponent > 0 && digits(&coefficient) < full {
            coefficient *= 10u32;
            exponent -= 1;
        }
    }
    assert!(
        exponent <= full_highest,
        "the value is below the overflow bound"
    );
    let finite = DecimalValue::Finite {
        negative,
        coefficient,
        exponent,
    };
    (finite, flags)
}

/// Returns the result of an overflow: an infinity, or the largest value of
/// `precision` digits in the directions that round toward zero.
fn overflow(
    negative: bool,
    precision: u32,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let flags = Flags::OVERFLOW | Flags::INEXACT;
    if overflows_to_infinity(env.rounding, negative) {
        return (
            DecimalValue::Infinity { negative },
            flags | Flags::ROUNDED_UP,
        );
    }
    let full = format.precision;
    let coefficient = Integer::from(Integer::u_pow_u(10, full))
        - Integer::from(Integer::u_pow_u(10, full - precision));
    let (_, highest) = format.exponents();
    let largest = DecimalValue::Finite {
        negative,
        coefficient,
        exponent: highest,
    };
    (largest, flags)
}

/// Aligns the high-order bits of a field of `from` bits with a field of `to`
/// bits: a wider field gains low-order zeros, and a narrower field drops the
/// low-order bits.
#[must_use]
pub fn align(payload: Integer, from: u32, to: u32) -> Integer {
    if to >= from {
        payload << (to - from)
    } else {
        payload >> (from - to)
    }
}

/// Returns the payload of a decimal NaN converted from a binary NaN with
/// `payload`, a field of `bits` bits. The payload bits align with the
/// trailing significand field of the decimal format, and a payload above
/// `10^(p - 1) - 1` is non-canonical and becomes zero, as the Intel decimal
/// library does.
#[must_use]
pub fn decimal_payload(payload: Integer, bits: u32, format: DecimalFormat) -> Integer {
    let aligned = align(payload, bits, format.trailing_bits());
    if aligned > format.largest_payload() {
        Integer::ZERO
    } else {
        aligned
    }
}
