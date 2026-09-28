//! A rounding oracle built from MPFR operations.
//!
//! MPFR rounds to any precision in five directions and emulates a subnormal
//! range. The oracle composes those operations to give the IEEE 754 result
//! of rounding an exact value to a binary format: the value, and the flags.
//! Round to nearest with ties away from zero, and round to odd, come from the
//! two directed results, as `DESIGN.md` describes.

use core::cmp::Ordering;

use floaty::env::Tininess;
use floaty::{Decoded, Env, Flags, Rounding};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

/// How a format encodes its special values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Specials {
    /// IEEE 754 infinities and NaNs.
    Ieee,
    /// No infinity; the all-ones significand at `emax` is the NaN.
    NoInf,
    /// No infinity and no negative zero; the one NaN is negative.
    Fnuz,
}

/// The parameters of a format, as the oracle uses them.
#[derive(Clone, Copy, Debug)]
pub struct Format {
    /// The precision.
    pub precision: u32,
    /// The exponent of the smallest normal value.
    pub emin: i32,
    /// The exponent of the largest finite value.
    pub emax: i32,
    /// The special values.
    pub specials: Specials,
}

impl Format {
    /// Returns the precision that `env` rounds to: the precision limit of the
    /// behavior, when it is below the format precision.
    #[must_use]
    pub fn precision_in(&self, env: &Env) -> u32 {
        env.precision
            .map_or(self.precision, |limit| limit.get().min(self.precision))
    }
}

/// A rounded value in a form that compares across implementations.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    /// A zero.
    Zero {
        /// The sign.
        negative: bool,
    },
    /// A nonzero finite value.
    Finite(BigFloat),
    /// An infinity.
    Infinity {
        /// The sign.
        negative: bool,
    },
    /// A NaN.
    Nan {
        /// The sign.
        negative: bool,
    },
}

impl Value {
    /// Converts a decoded floaty value.
    ///
    /// # Panics
    ///
    /// Panics for an unsupported encoding, which rounding never gives.
    #[must_use]
    pub fn from_decoded<const N: usize>(decoded: Decoded<N>) -> Self {
        match decoded {
            Decoded::Zero { negative } => Self::Zero { negative },
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
                Self::Finite(value)
            }
            Decoded::Infinity { negative } => Self::Infinity { negative },
            Decoded::Nan { negative, .. } => Self::Nan { negative },
            Decoded::Unsupported => panic!("rounding never gives an unsupported encoding"),
        }
    }
}

/// An exact value to round: `significand * 2^exponent`, plus a sticky bit.
#[derive(Clone, Debug)]
pub struct Input {
    /// The sign.
    pub negative: bool,
    /// The weight of the lowest significand bit.
    pub exponent: i32,
    /// The significand.
    pub significand: Integer,
    /// The sticky bit.
    pub sticky: bool,
}

/// Returns the expected result and flags of rounding `input` to `format`.
///
/// # Panics
///
/// Panics when the input breaks the sticky contract of `floaty::Exact`.
#[must_use]
pub fn round(input: &Input, format: &Format, env: &Env) -> (Value, Flags) {
    let negative = input.negative;
    if input.significand.is_zero() {
        return (zero(negative, format), Flags::NONE);
    }
    let precision = format.precision_in(env);
    let width = input.significand.significant_bits();
    assert!(
        !input.sticky || width >= precision + 2,
        "the input keeps the sticky contract"
    );

    // A sticky value lies strictly between two multiples of 2^exponent. Its
    // midpoint rounds the same way, because every rounding position is at
    // least two bits above the lowest bit.
    let (integer, exponent) = if input.sticky {
        (
            (input.significand.clone() << 1u32) + 1u32,
            input.exponent - 1,
        )
    } else {
        (input.significand.clone(), input.exponent)
    };
    let mut exact = BigFloat::with_val(integer.significant_bits(), integer) << exponent;
    if negative {
        exact = -exact;
    }

    let smallest_normal = BigFloat::with_val(2, 1) << format.emin;
    let unbounded = round_to(&exact, precision, None, env.rounding);
    let tiny = match env.tininess {
        Tininess::BeforeRounding => exact.clone().abs() < smallest_normal,
        Tininess::AfterRounding => unbounded.clone().abs() < smallest_normal,
    };
    let rounded = round_to(&exact, precision, Some(format.emin), env.rounding);

    let mut flags = Flags::NONE;
    if tiny {
        flags |= Flags::TINY;
        if env.flush_to_zero {
            return (
                zero(negative, format),
                flags | Flags::UNDERFLOW | Flags::INEXACT,
            );
        }
    }
    let inexact = rounded != exact;
    if inexact {
        flags |= Flags::INEXACT;
        if tiny {
            flags |= Flags::UNDERFLOW;
        }
    }
    if rounded.is_zero() {
        return (zero(negative, format), flags);
    }
    if rounded.clone().abs() > largest(format, precision) {
        return overflow(negative, format, precision, env);
    }
    if rounded.clone().abs() > exact.abs() {
        flags |= Flags::ROUNDED_UP;
    }
    (Value::Finite(rounded), flags)
}

fn zero(negative: bool, format: &Format) -> Value {
    Value::Zero {
        negative: negative && format.specials != Specials::Fnuz,
    }
}

fn nan(negative: bool, format: &Format) -> Value {
    Value::Nan {
        negative: negative || format.specials == Specials::Fnuz,
    }
}

/// Returns the largest finite magnitude at `precision` bits.
fn largest(format: &Format, precision: u32) -> BigFloat {
    let significand = if format.specials == Specials::NoInf && precision == format.precision {
        (Integer::from(1) << precision) - 2u32
    } else {
        (Integer::from(1) << precision) - 1u32
    };
    let shift = format.emax - i32::try_from(precision - 1).expect("a precision fits an i32");
    BigFloat::with_val(precision, significand) << shift
}

fn overflow(negative: bool, format: &Format, precision: u32, env: &Env) -> (Value, Flags) {
    let flags = Flags::OVERFLOW | Flags::INEXACT;
    let to_infinity = match env.rounding {
        Rounding::NearestEven | Rounding::NearestAway => true,
        Rounding::TowardPositive => !negative,
        Rounding::TowardNegative => negative,
        _ => false,
    };
    if to_infinity {
        if format.specials == Specials::Ieee {
            return (Value::Infinity { negative }, flags | Flags::ROUNDED_UP);
        }
        if !env.saturate {
            return (nan(negative, format), flags);
        }
    }
    let mut value = largest(format, precision);
    if negative {
        value = -value;
    }
    (Value::Finite(value), flags)
}

/// Rounds to `precision` bits. With `emin`, the result keeps the subnormal
/// quantum `2^(emin - precision + 1)`, as MPFR's subnormal emulation gives.
fn round_to(exact: &BigFloat, precision: u32, emin: Option<i32>, rounding: Rounding) -> BigFloat {
    let directed = |round: Round| {
        let (mut value, ordering) = BigFloat::with_val_round(precision, exact, round);
        if let Some(emin) = emin {
            // MPFR writes a value as m * 2^e with 0.5 <= m < 1, so the
            // smallest normal value 2^emin has the MPFR exponent emin + 1.
            value.subnormalize_round(emin + 1, ordering, round);
        }
        value
    };
    match rounding {
        Rounding::NearestEven => directed(Round::Nearest),
        Rounding::TowardPositive => directed(Round::Up),
        Rounding::TowardNegative => directed(Round::Down),
        Rounding::TowardZero => directed(Round::Zero),
        Rounding::NearestAway => {
            let (toward, away) = (directed(Round::Zero), directed(Round::AwayZero));
            if toward == away {
                return toward;
            }
            let working = precision + 2;
            let midpoint: BigFloat = (BigFloat::with_val(working, &toward) + &away) >> 1u32;
            if exact.clone().abs() >= midpoint.abs() {
                away
            } else {
                toward
            }
        }
        Rounding::ToOdd => {
            let toward = directed(Round::Zero);
            if &toward == exact || is_odd(&toward, precision, emin) {
                toward
            } else {
                directed(Round::AwayZero)
            }
        }
        _ => panic!("the oracle knows every rounding direction"),
    }
}

/// Returns `true` when the lowest significand bit of `value` is set, at the
/// quantum of its binade or at the subnormal quantum.
fn is_odd(value: &BigFloat, precision: u32, emin: Option<i32>) -> bool {
    let Some((integer, exponent)) = value.to_integer_exp() else {
        return false;
    };
    let top =
        exponent + i32::try_from(integer.significant_bits()).expect("the width fits an i32") - 1;
    let below = i32::try_from(precision - 1).expect("a precision fits an i32");
    let binade = top - below;
    let subnormal = emin.map_or(i32::MIN, |emin| emin - below);
    let quantum = binade.max(subnormal);
    match quantum.cmp(&exponent) {
        Ordering::Less => false,
        Ordering::Equal => integer.is_odd(),
        Ordering::Greater => {
            let shift = u32::try_from(quantum - exponent).expect("a positive shift");
            (integer.abs() >> shift).is_odd()
        }
    }
}

/// Returns the expected result and flags of converting a decoded value to
/// `format`, by the conversion rules in `DESIGN.md`. MPFR rounds the finite
/// values. `subnormal` says that the source encoding is subnormal.
///
/// The caller decodes the source with floaty's own `decode` and `classify`,
/// which the step 1 oracles check. The special-value rules are the rules of
/// the design, not an independent reference.
///
/// # Panics
///
/// Panics for an unsupported source encoding, which the caller checks
/// against the processor instead.
#[must_use]
pub fn convert<const N: usize>(
    source: &Decoded<N>,
    subnormal: bool,
    format: &Format,
    env: &Env,
) -> (Value, Flags) {
    let input = if subnormal {
        Flags::DENORMAL_INPUT
    } else {
        Flags::NONE
    };
    match *source {
        Decoded::Zero { negative } => (zero(negative, format), Flags::NONE),
        Decoded::Finite { negative, .. } if subnormal && env.denormals_are_zero => {
            (zero(negative, format), input)
        }
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let exact = Input {
                negative,
                exponent,
                significand: Integer::from_digits(&significand, Order::Lsf),
                sticky: false,
            };
            let (value, flags) = round(&exact, format, env);
            (value, flags | input)
        }
        Decoded::Infinity { negative } if format.specials == Specials::Ieee => {
            (Value::Infinity { negative }, Flags::NONE)
        }
        Decoded::Infinity { negative } if env.saturate => {
            let mut value = largest(format, format.precision_in(env));
            if negative {
                value = -value;
            }
            (Value::Finite(value), Flags::INVALID)
        }
        Decoded::Infinity { negative } => (nan(negative, format), Flags::INVALID),
        Decoded::Nan {
            negative,
            signaling,
            ..
        } => {
            let flags = if signaling {
                Flags::INVALID
            } else {
                Flags::NONE
            };
            (nan(negative, format), flags)
        }
        Decoded::Unsupported => panic!("the oracle has no rule for an unsupported encoding"),
    }
}
