//! A rounding oracle built from MPFR operations.
//!
//! MPFR rounds to any precision in five directions and emulates a subnormal
//! range. The oracle composes those operations to give the IEEE 754 result
//! of rounding an exact value to a binary format: the value, and the flags.
//! The directions that MPFR lacks come from the results toward zero and away
//! from zero: round to nearest with ties away from zero or toward zero, by
//! the midpoint, and round to odd, by the parity.
//!
//! The module also holds the special-value rules that every oracle of the
//! harness shares: the zero, the NaN, and the largest value of a [`Format`],
//! and the NaN that a NaN rule selects, [`select_nan`]. [`Operand::read`]
//! gives the one rule by which every oracle reads an operand.

use core::cmp::Ordering;
use core::num::NonZeroU32;

pub mod decimal;
mod operand;

use floaty::env::{Mode, NanPropagation, NanRule, Tininess};
use floaty::format::Standard;
use floaty::{Decoded, Env, Flags, Float, Rounding};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

pub(crate) use operand::signed_zero;
pub use operand::{Operand, Read};

pub use crate::formats::Specials;

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

/// A floaty `Float` type, whose format parameters the oracles read. The
/// trait is sealed.
pub trait Parameters: sealed::Sealed {
    /// The precision in digits of the radix.
    const PRECISION: u32;
    /// The exponent of the smallest normal value, as a power of the radix.
    const EMIN: i32;
    /// The exponent of the largest finite value, as a power of the radix.
    const EMAX: i32;
}

impl<S: Standard<W>, const W: usize, M: Mode> Parameters for Float<S, W, M> {
    const PRECISION: u32 = Float::<S, W, M>::PRECISION;
    const EMIN: i32 = Float::<S, W, M>::EMIN;
    const EMAX: i32 = Float::<S, W, M>::EMAX;
}

mod sealed {
    use floaty::Float;
    use floaty::env::Mode;
    use floaty::format::Standard;

    pub trait Sealed {}

    impl<S: Standard<W>, const W: usize, M: Mode> Sealed for Float<S, W, M> {}
}

impl Format {
    /// Returns the parameters of the binary type `T` with the special values
    /// `specials`.
    #[must_use]
    pub const fn of<T: Parameters>(specials: Specials) -> Self {
        Self {
            precision: T::PRECISION,
            emin: T::EMIN,
            emax: T::EMAX,
            specials,
        }
    }

    /// Returns the precision that `env` rounds to: the precision limit of the
    /// behavior, when it is below the format precision.
    #[must_use]
    pub fn precision_in(&self, env: &Env) -> u32 {
        limit_precision(self.precision, env)
    }

    /// Returns a zero with the sign `negative`. A format without a negative
    /// zero, `Fnuz`, gives `+0`.
    #[must_use]
    pub fn zero(&self, negative: bool) -> Value {
        Value::Zero {
            negative: negative && self.specials != Specials::Fnuz,
        }
    }

    /// Returns the NaN with the sign `negative`. The one NaN of `Fnuz` is
    /// negative, and a format without a NaN, `Finite`, gives `+0`.
    #[must_use]
    pub fn nan(&self, negative: bool) -> Value {
        if self.specials == Specials::Finite {
            return Value::Zero { negative: false };
        }
        Value::Nan {
            negative: negative || self.specials == Specials::Fnuz,
        }
    }

    /// Returns the largest finite magnitude at `precision` bits. At the full
    /// precision, the all-ones significand at `emax` of `NoInf` is the NaN,
    /// so the largest significand is one below it.
    ///
    /// # Panics
    ///
    /// Panics for a precision that does not fit an `i32`.
    #[must_use]
    pub fn largest(&self, precision: u32) -> BigFloat {
        let significand = if self.specials == Specials::NoInf && precision == self.precision {
            (Integer::from(1) << precision) - 2u32
        } else {
            (Integer::from(1) << precision) - 1u32
        };
        let shift = self.emax - i32::try_from(precision - 1).expect("a precision fits an i32");
        BigFloat::with_val(precision, significand) << shift
    }

    /// Returns the largest finite value at `precision` bits, by
    /// [`Format::largest`], with the sign `negative`.
    #[must_use]
    pub fn signed_largest(&self, negative: bool, precision: u32) -> Value {
        let largest = self.largest(precision);
        Value::Finite(if negative { -largest } else { largest })
    }

    /// Returns the value of an infinite result with the sign `negative` in
    /// `env`. A saturating behavior gives the largest finite value of the
    /// precision of `env`, in every format, as OCP FP8 saturation and PTX
    /// `.satfinite` do. Otherwise an IEEE format keeps the infinity, a format
    /// without an infinity gives its NaN, by [`Format::nan`], and a format
    /// with neither gives the largest finite value.
    #[must_use]
    pub fn infinity(&self, negative: bool, env: &Env) -> Value {
        if env.saturate || self.specials == Specials::Finite {
            return self.signed_largest(negative, self.precision_in(env));
        }
        if self.specials == Specials::Ieee {
            return Value::Infinity { negative };
        }
        self.nan(negative)
    }
}

/// Returns the precision that `env` rounds to in a format of `precision`
/// digits: the precision limit of the behavior, when it is below.
pub(crate) fn limit_precision(precision: u32, env: &Env) -> u32 {
    env.precision
        .map_or(precision, |limit| limit.get().min(precision))
}

/// Returns `true` when an overflow with the sign `negative` rounds to an
/// infinity in the direction `rounding`, and `false` when it rounds to the
/// largest finite value.
///
/// # Panics
///
/// Panics for a rounding direction that a later floaty adds.
pub(crate) fn overflows_to_infinity(rounding: Rounding, negative: bool) -> bool {
    match rounding {
        Rounding::TiesToEven
        | Rounding::TiesToAway
        | Rounding::TiesTowardZero
        | Rounding::AwayFromZero => true,
        Rounding::TowardPositive => !negative,
        Rounding::TowardNegative => negative,
        Rounding::TowardZero | Rounding::ToOdd => false,
        _ => panic!("the oracle knows every rounding direction"),
    }
}

/// Returns `significand * 2^exponent` exactly, with the sign `negative`. The
/// significand is in limbs from the low bits up.
#[must_use]
pub fn exact(negative: bool, exponent: i32, significand: &[u64]) -> BigFloat {
    let integer = Integer::from_digits(significand, Order::Lsf);
    let bits = integer.significant_bits().max(1);
    let value = BigFloat::with_val(bits, integer) << exponent;
    if negative { -value } else { value }
}

/// A NaN that a NaN rule can select: a NaN operand, or the default NaN.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Nan {
    /// The sign.
    pub negative: bool,
    /// `true` for a signaling NaN.
    pub signaling: bool,
    /// The payload, below the quiet bit.
    pub payload: Integer,
}

impl Nan {
    /// Returns the default NaN of the NaN rule of `env`: a quiet NaN with the
    /// default sign of the rule and a zero payload.
    #[must_use]
    pub fn default_of(env: &Env) -> Self {
        Self {
            negative: env.nan.default_negative,
            signaling: false,
            payload: Integer::ZERO,
        }
    }
}

/// Selects a NaN among `offered`, in the order of the list, by the
/// propagation rule of `env`, and makes it quiet.
///
/// The rules are those of the documentation of `NanPropagation`:
/// `SignalingFirst` takes the first signaling NaN, or else the first NaN.
/// `FirstOperand` takes the first NaN. `LargerSignificand` takes a quiet NaN
/// before a signaling NaN, then the larger payload, then the positive sign.
/// `DefaultNan` gives the default NaN of the rule, with a zero payload.
///
/// # Panics
///
/// Panics when `offered` is empty, and for a propagation rule that a later
/// floaty adds.
#[must_use]
pub fn select_nan(offered: &[Nan], env: &Env) -> Nan {
    let first = offered
        .first()
        .expect("the rule selects from at least one NaN");
    let chosen = match env.nan.propagation {
        NanPropagation::DefaultNan => return Nan::default_of(env),
        NanPropagation::SignalingFirst => offered.iter().find(|nan| nan.signaling).unwrap_or(first),
        NanPropagation::FirstOperand => first,
        NanPropagation::LargerSignificand => offered
            .iter()
            .max_by_key(|&nan| (!nan.signaling, &nan.payload, !nan.negative))
            .expect("the rule selects from at least one NaN"),
        _ => panic!("the oracle knows every propagation rule"),
    };
    Nan {
        signaling: false,
        ..chosen.clone()
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
            Decoded::Zero { negative, .. } => Self::Zero { negative },
            Decoded::Finite {
                negative,
                exponent,
                significand,
            } => Self::Finite(exact(negative, exponent, &significand)),
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
        return (format.zero(negative), Flags::NONE);
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

    let mut flags = match underflow_flags(tiny, rounded != exact, env) {
        Underflow::Flushed(flags) => return (format.zero(negative), flags),
        Underflow::Kept(flags) => flags,
    };
    if rounded.is_zero() {
        return (format.zero(negative), flags);
    }
    if rounded.clone().abs() > format.largest(precision) {
        return overflow(negative, format, precision, env);
    }
    if rounded.clone().abs() > exact.abs() {
        flags |= Flags::ROUNDED_UP;
    }
    (Value::Finite(rounded), flags)
}

/// The flags of a rounded result, and whether flush-to-zero replaces it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Underflow {
    /// Flush-to-zero replaces the tiny result with a zero of its sign, with
    /// these flags.
    Flushed(Flags),
    /// The result stays, with these flags.
    Kept(Flags),
}

/// Returns the flags of a rounded result by its tininess and its exactness,
/// for a binary and a decimal destination alike. A tiny result signals
/// `TINY`, and `UNDERFLOW` when it is inexact. Flush-to-zero replaces a tiny
/// result with a zero, which signals `UNDERFLOW` and `INEXACT`.
pub(crate) fn underflow_flags(tiny: bool, inexact: bool, env: &Env) -> Underflow {
    let mut flags = Flags::NONE;
    if tiny {
        flags |= Flags::TINY;
        if env.flush_to_zero {
            return Underflow::Flushed(flags | Flags::UNDERFLOW | Flags::INEXACT);
        }
    }
    if inexact {
        flags |= Flags::INEXACT;
        if tiny {
            flags |= Flags::UNDERFLOW;
        }
    }
    Underflow::Kept(flags)
}

/// Every rounding direction of floaty, which the oracle knows.
pub const DIRECTIONS: [Rounding; 8] = [
    Rounding::TiesToEven,
    Rounding::TiesToAway,
    Rounding::TiesTowardZero,
    Rounding::TowardPositive,
    Rounding::TowardNegative,
    Rounding::TowardZero,
    Rounding::AwayFromZero,
    Rounding::ToOdd,
];

fn overflow(negative: bool, format: &Format, precision: u32, env: &Env) -> (Value, Flags) {
    let flags = Flags::OVERFLOW | Flags::INEXACT;
    let to_infinity = overflows_to_infinity(env.rounding, negative);
    // A saturating behavior gives the largest finite value, as the OCP FP8
    // saturation mode does for an overflow.
    if to_infinity && !env.saturate && format.specials != Specials::Finite {
        if format.specials == Specials::Ieee {
            return (Value::Infinity { negative }, flags | Flags::ROUNDED_UP);
        }
        return (format.nan(negative), flags);
    }
    (format.signed_largest(negative, precision), flags)
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
        Rounding::TiesToEven => directed(Round::Nearest),
        Rounding::TowardPositive => directed(Round::Up),
        Rounding::TowardNegative => directed(Round::Down),
        Rounding::TowardZero => directed(Round::Zero),
        Rounding::AwayFromZero => directed(Round::AwayZero),
        Rounding::TiesToAway | Rounding::TiesTowardZero => {
            let (toward, away) = (directed(Round::Zero), directed(Round::AwayZero));
            if toward == away {
                return toward;
            }
            let working = precision + 2;
            let midpoint: BigFloat = (BigFloat::with_val(working, &toward) + &away) >> 1u32;
            let (magnitude, midpoint) = (exact.clone().abs(), midpoint.abs());
            let tie_goes_away = rounding == Rounding::TiesToAway;
            if magnitude > midpoint || (magnitude == midpoint && tie_goes_away) {
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

/// The behaviors of the conversion tests between binary formats: every
/// rounding direction, both tininess rules, flush-to-zero,
/// denormals-are-zero, saturation, precision limits, and the default NaN.
pub const CONVERSION_BEHAVIORS: [Env; 10] = [
    Env::IEEE,
    Env::IEEE
        .with_rounding(Rounding::TowardZero)
        .with_tininess(Tininess::BeforeRounding),
    Env::IEEE
        .with_rounding(Rounding::TowardPositive)
        .with_flush_to_zero(true),
    Env::IEEE.with_rounding(Rounding::ToOdd).with_saturate(true),
    Env::IEEE
        .with_rounding(Rounding::TiesToAway)
        .with_denormals_are_zero(true),
    Env::IEEE
        .with_rounding(Rounding::TowardNegative)
        .with_precision(NonZeroU32::new(3)),
    // A saturated infinity and a saturated overflow both use the limit.
    Env::IEEE
        .with_saturate(true)
        .with_precision(NonZeroU32::new(2)),
    Env::IEEE
        .with_rounding(Rounding::TiesTowardZero)
        .with_flush_to_zero(true),
    Env::IEEE.with_rounding(Rounding::AwayFromZero),
    Env::IEEE.with_nan(NanRule::new(NanPropagation::DefaultNan).with_default_negative(true)),
];

/// Returns the expected result and flags of converting an operand to
/// `format`, by floaty's conversion rules. MPFR rounds the finite values.
///
/// The special-value rules are floaty's own rules, not an independent
/// reference. An unsupported x87 encoding gives the default NaN and signals
/// invalid, as the documentation of `convert_with` states. The x87 hardware
/// tests check that rule for binary32 and binary64 destinations.
///
/// # Panics
///
/// Never: MPFR gives an integer significand for every finite value.
#[must_use]
pub fn convert<const N: usize>(source: &Operand<N>, format: &Format, env: &Env) -> (Value, Flags) {
    let mut flags = Flags::NONE;
    match source.read(env, &mut flags) {
        Read::Number(value) if value.is_zero() => (format.zero(value.is_sign_negative()), flags),
        Read::Number(value) if value.is_infinite() => {
            // A format without an infinity cannot hold the infinity.
            if format.specials != Specials::Ieee {
                flags |= Flags::INVALID;
            }
            (format.infinity(value.is_sign_negative(), env), flags)
        }
        Read::Number(value) => {
            let (significand, exponent) = value.to_integer_exp().expect("the value is finite");
            let exact = Input {
                negative: value.is_sign_negative(),
                exponent,
                significand: significand.abs(),
                sticky: false,
            };
            let (value, round_flags) = round(&exact, format, env);
            (value, flags | round_flags)
        }
        Read::Nan(nan) => {
            // A format without a NaN cannot hold the NaN.
            if nan.signaling || format.specials == Specials::Finite {
                flags |= Flags::INVALID;
            }
            // `DefaultNan` gives the default NaN. The other rules keep the sign.
            (format.nan(select_nan(&[nan], env).negative), flags)
        }
        Read::Unsupported => (
            format.nan(Nan::default_of(env).negative),
            flags | Flags::INVALID,
        ),
    }
}
