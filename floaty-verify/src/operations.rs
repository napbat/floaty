//! An oracle for the operations beyond arithmetic: comparison, total order,
//! minimum and maximum, rounding to an integral value, conversion to and from
//! an integer, the remainder, scaling by a power of two, the next value up or
//! down, and the sign operations.
//!
//! MPFR computes each finite result. The operations that do not round get the
//! exact MPFR value, and the others round with the oracle in [`crate::mpfr`].
//! IEEE 754-2019 and the documented rules of floaty give the special values,
//! the flags, and the NaN that each rule selects.
//!
//! The oracle reads its operands by [`Operand::read`], as
//! [`crate::arithmetic`] does.

use core::cmp::Ordering;
use core::num::NonZeroU32;

use floaty::env::{Mode, NanPropagation, NanRule, Tininess};
use floaty::format::Standard;
use floaty::{Decoded, Env, Flags, Float, Rounding};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

use crate::encodings::Layout;
use crate::mpfr::{
    self, Format, Input, Nan, Operand, Read, Specials, Value, exact, select_nan, signed_zero,
};

pub mod check;
pub mod compare;
pub mod integral;

/// The behaviors of the operation tests. Together they use every rounding
/// direction, every NaN rule, both tininess rules, flush-to-zero,
/// denormals-are-zero, saturation, and a precision limit. Each operation
/// ignores the fields that its documentation says do not apply to it, so
/// every behavior also checks that those fields change nothing.
pub const BEHAVIORS: [Env; 10] = [
    Env::IEEE,
    Env::IEEE
        .with_rounding(Rounding::TiesToAway)
        .with_denormals_are_zero(true)
        .with_nan(NanRule::new(NanPropagation::FirstOperand).with_default_negative(true)),
    Env::IEEE
        .with_rounding(Rounding::TowardPositive)
        .with_saturate(true)
        .with_nan(NanRule::new(NanPropagation::LargerSignificand)),
    Env::IEEE
        .with_rounding(Rounding::TowardNegative)
        .with_flush_to_zero(true)
        .with_tininess(Tininess::BeforeRounding),
    Env::IEEE
        .with_rounding(Rounding::TowardZero)
        .with_nan(NanRule::new(NanPropagation::DefaultNan).with_default_negative(true)),
    Env::IEEE
        .with_rounding(Rounding::ToOdd)
        .with_precision(NonZeroU32::new(2))
        .with_saturate(true),
    Env::IEEE
        .with_denormals_are_zero(true)
        .with_flush_to_zero(true),
    Env::IEEE
        .with_precision(NonZeroU32::new(2))
        .with_tininess(Tininess::BeforeRounding),
    Env::IEEE
        .with_rounding(Rounding::TiesTowardZero)
        .with_tininess(Tininess::BeforeRounding),
    Env::IEEE
        .with_rounding(Rounding::AwayFromZero)
        .with_saturate(true),
];

/// The expected result of an operation that returns a float.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
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
        /// `true` for a signaling NaN. No expected result is signaling.
        signaling: bool,
        /// The payload, below the quiet bit.
        payload: Integer,
    },
}

impl Outcome {
    /// Converts a decoded floaty result.
    ///
    /// # Panics
    ///
    /// Panics for an unsupported encoding, which no operation returns.
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
            Decoded::Nan {
                negative,
                signaling,
                payload,
            } => Self::Nan {
                negative,
                signaling,
                payload: Integer::from_digits(&payload, Order::Lsf),
            },
            Decoded::Unsupported => panic!("no operation returns an unsupported encoding"),
        }
    }

    /// Converts a result of the rounding oracle. Its NaN is quiet and has no
    /// payload.
    fn from_value(value: Value) -> Self {
        match value {
            Value::Zero { negative } => Self::Zero { negative },
            Value::Finite(value) => Self::Finite(value),
            Value::Infinity { negative } => Self::Infinity { negative },
            Value::Nan { negative } => Self::Nan {
                negative,
                signaling: false,
                payload: Integer::ZERO,
            },
        }
    }
}

/// Returns the outcome of a floaty result, for comparison with the oracle.
#[must_use]
pub fn outcome<S: Standard<W>, const W: usize, M: Mode>(value: Float<S, W, M>) -> Outcome {
    Outcome::from_decoded(value.decode::<8>())
}

/// An encoding with its decoded operand, for the operations that read the
/// bits: the total order and the sign operations.
#[derive(Clone, Debug)]
pub struct Sample<const N: usize> {
    /// The encoding.
    pub bits: Integer,
    /// The field layout of the encoding.
    pub layout: Layout,
    /// `true` when the encoding is canonical.
    pub canonical: bool,
    /// The decoded operand.
    pub operand: Operand<N>,
}

impl Sample<8> {
    /// Makes a sample from an encoding, its field layout, and the floaty value
    /// of that encoding.
    ///
    /// # Panics
    ///
    /// Panics when the width of the layout is not the width of the format.
    #[must_use]
    pub fn new<S: Standard<W>, const W: usize, M: Mode>(
        bits: Integer,
        layout: Layout,
        value: Float<S, W, M>,
    ) -> Self {
        assert_eq!(
            usize::try_from(layout.width).ok(),
            Some(W),
            "the layout has the width of the format"
        );
        Self {
            bits,
            layout,
            canonical: value.is_canonical(),
            operand: Operand::of(value),
        }
    }
}

/// Returns the outcome of a number: a zero, a finite value, or an infinity.
/// A format without a negative zero gives `+0`.
fn number(value: &BigFloat, format: &Format) -> Outcome {
    if value.is_zero() {
        Outcome::from_value(format.zero(value.is_sign_negative()))
    } else if value.is_infinite() {
        Outcome::Infinity {
            negative: value.is_sign_negative(),
        }
    } else {
        Outcome::Finite(value.clone())
    }
}

/// Returns the default NaN of the NaN rule, by [`Format::nan`].
fn default_nan(format: &Format, env: &Env) -> Outcome {
    Outcome::from_value(format.nan(env.nan.default_negative))
}

/// Returns the NaN that the NaN rule of `env` selects among the operands by
/// [`select_nan`], and `INVALID` when an operand is a signaling NaN. The
/// result is the NaN of the format, by [`Format::nan`], with the payload of
/// the selected NaN. At least one operand is a NaN.
fn propagate(operands: &[Read], format: &Format, env: &Env) -> (Outcome, Flags) {
    let nans: Vec<Nan> = operands
        .iter()
        .filter_map(|operand| match operand {
            Read::Nan(nan) => Some(nan.clone()),
            Read::Number(_) | Read::Unsupported => None,
        })
        .collect();
    let flags = if nans.iter().any(|nan| nan.signaling) {
        Flags::INVALID
    } else {
        Flags::NONE
    };
    let selected = select_nan(&nans, env);
    let nan = match format.nan(selected.negative) {
        Value::Nan { negative } => Outcome::Nan {
            negative,
            signaling: false,
            payload: selected.payload,
        },
        value => Outcome::from_value(value),
    };
    (nan, flags)
}

/// Returns the result of an operation with an unsupported or a NaN operand,
/// or `None` when every operand is a number. An unsupported operand signals
/// invalid and gives the default NaN, before any NaN operand.
fn special_operands(operands: &[Read], format: &Format, env: &Env) -> Option<(Outcome, Flags)> {
    if operands
        .iter()
        .any(|operand| matches!(operand, Read::Unsupported))
    {
        return Some((default_nan(format, env), Flags::INVALID));
    }
    if operands
        .iter()
        .any(|operand| matches!(operand, Read::Nan(_)))
    {
        return Some(propagate(operands, format, env));
    }
    None
}

/// Returns `TINY` for a nonzero finite value below the smallest normal
/// magnitude. floaty reports `TINY` for a subnormal remainder.
fn tiny(value: &BigFloat, format: &Format) -> Flags {
    let smallest_normal = BigFloat::with_val(2, 1) << format.emin;
    if value.is_normal() && *value.as_abs() < smallest_normal {
        Flags::TINY
    } else {
        Flags::NONE
    }
}

/// Returns the expected result and flags of `remainder_with`.
///
/// IEEE 754-2019 section 5.3.1: the remainder is `x - n * y` with `n` the
/// integer nearest `x / y`, and the even one at a tie. MPFR computes it
/// exactly. The remainder of a finite `x` by an infinity is `x`, and a zero
/// remainder has the sign of `x`. Section 7.2 makes an infinite `x` or a zero
/// `y` invalid. floaty reports `TINY` for a subnormal result; the
/// rounding direction, the precision limit, and flush-to-zero do not apply.
///
/// # Panics
///
/// Panics when MPFR does not give an exact remainder at the format precision.
#[must_use]
pub fn remainder<const N: usize>(
    first: &Operand<N>,
    second: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    exact_remainder(first, second, format, env, |x, y| {
        BigFloat::with_val_round(format.precision, x.remainder_ref(y), Round::Nearest)
    })
}

/// Returns the expected result and flags of `truncated_remainder_with`.
///
/// The remainder is `x - n * y` with `n` the integer part of `x / y`, as C
/// `fmod` and MPFR `mpfr_fmod` define it. The other rules are those of
/// [`remainder`].
///
/// # Panics
///
/// Panics when MPFR does not give an exact remainder at the format precision.
#[must_use]
pub fn truncated_remainder<const N: usize>(
    first: &Operand<N>,
    second: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    exact_remainder(first, second, format, env, |x, y| {
        BigFloat::with_val_round(format.precision, x % y, Round::Nearest)
    })
}

/// Returns the expected result and flags of a remainder of finite operands
/// that `compute` gives exactly, with the special cases of IEEE 754.
fn exact_remainder<const N: usize>(
    first: &Operand<N>,
    second: &Operand<N>,
    format: &Format,
    env: &Env,
    compute: impl Fn(&BigFloat, &BigFloat) -> (BigFloat, Ordering),
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let operands = [first.read(env, &mut flags), second.read(env, &mut flags)];
    if let Some((nan, special)) = special_operands(&operands, format, env) {
        return (nan, flags | special);
    }
    let [Read::Number(x), Read::Number(y)] = &operands else {
        unreachable!("every special operand has a result");
    };
    if x.is_infinite() || y.is_zero() {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    if x.is_zero() || y.is_infinite() {
        return (number(x, format), flags | tiny(x, format));
    }
    let (result, ordering) = compute(x, y);
    assert_eq!(ordering, Ordering::Equal, "the remainder is exact");
    let result = if result.is_zero() {
        signed_zero(x.is_sign_negative())
    } else {
        result
    };
    (number(&result, format), flags | tiny(&result, format))
}

/// The largest scale magnitude that the oracle applies. The exponent range
/// of every format spans less than `2^23`, so a larger scale overflows or
/// underflows past every rounding boundary, and gives the same result as the
/// clamp of floaty at `2^30`. The smaller limit keeps the MPFR exponent
/// inside its default range.
const SCALE_LIMIT: i32 = 1 << 24;

/// Returns the expected result and flags of `scale_b_with`.
///
/// IEEE 754-2019 section 5.3.3 `scaleB`: the exact value `x * 2^n` rounds
/// once, with every rounding field of the behavior. A zero or an infinity
/// does not change.
///
/// # Panics
///
/// Panics when MPFR gives no significand for a finite value, which does not
/// happen.
#[must_use]
pub fn scale_b<const N: usize>(
    operand: &Operand<N>,
    scale: i32,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let operands = [operand.read(env, &mut flags)];
    if let Some((nan, special)) = special_operands(&operands, format, env) {
        return (nan, flags | special);
    }
    let [Read::Number(value)] = &operands else {
        unreachable!("every special operand has a result");
    };
    if !value.is_normal() {
        return (number(value, format), flags);
    }
    let (significand, exponent) = value.to_integer_exp().expect("the value is finite");
    let input = Input {
        negative: value.is_sign_negative(),
        exponent: exponent + scale.clamp(-SCALE_LIMIT, SCALE_LIMIT),
        significand: significand.abs(),
        sticky: false,
    };
    let (result, round_flags) = mpfr::round(&input, format, env);
    (Outcome::from_value(result), flags | round_flags)
}

/// The direction of `next_up` and `next_down`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// `nextUp`: toward positive infinity.
    Up,
    /// `nextDown`: toward negative infinity.
    Down,
}

/// Returns the expected result and flags of `next_up_with` or
/// `next_down_with`.
///
/// IEEE 754-2019 section 5.3.1: `nextUp(x)` is the least value above `x`,
/// and `nextDown(x)` the greatest value below `x`. The oracle rounds `x`
/// plus or minus a tiny amount toward the direction, with the rounding
/// oracle. The amount is far below the spacing of the values near `x`, so
/// the rounded value is the neighbor. An infinity reads as a value past the
/// largest finite value. In a format without an infinity, the value past the
/// largest finite value is the NaN, or with `saturate` set the largest
/// finite value itself. In a format with an infinity, it is the infinity
/// with or without `saturate`. Only a signaling NaN or an unsupported encoding
/// signals.
///
/// # Panics
///
/// Panics when the precision of the format does not fit an `i32`.
#[must_use]
pub fn next<const N: usize>(
    operand: &Operand<N>,
    direction: Direction,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let operands = [operand.read(env, &mut flags)];
    if let Some((nan, special)) = special_operands(&operands, format, env) {
        return (nan, flags | special);
    }
    let [Read::Number(value)] = &operands else {
        unreachable!("every special operand has a result");
    };
    let up = direction == Direction::Up;
    let lowest = format.emin
        - i32::try_from(format.precision - 1).expect("a precision of at most 512 bits fits an i32");
    if value.is_zero() {
        // The neighbors of a zero are the smallest subnormal values.
        let smallest = BigFloat::with_val(2, 1) << lowest;
        return (
            Outcome::Finite(if up { smallest } else { -smallest }),
            flags,
        );
    }
    let (significand, exponent) = if value.is_infinite() {
        (Integer::from(1), format.emax + 2)
    } else {
        let (significand, exponent) = value.to_integer_exp().expect("the value is finite");
        (significand.abs(), exponent)
    };
    // The significand with `precision + 2` more bits, and the sticky bit,
    // lies strictly between the value and its neighbors. Its magnitude grows
    // for a step away from zero, and shrinks for a step toward zero.
    let shift = format.precision + 2;
    let wider = significand << shift;
    let away_from_zero = up != value.is_sign_negative();
    let input = Input {
        negative: value.is_sign_negative(),
        exponent: exponent
            - i32::try_from(shift).expect("a precision of at most 512 bits fits an i32"),
        significand: if away_from_zero { wider } else { wider - 1u32 },
        sticky: true,
    };
    let rounding = if up {
        Rounding::TowardPositive
    } else {
        Rounding::TowardNegative
    };
    // The step does not round, so the precision limit does not apply. A
    // neighbor has the full precision of the format, and so does the largest
    // finite value that a saturated step past it gives. The step overflows
    // nothing, so saturation applies only where no value lies past the
    // largest finite value: an infinity is the neighbor in IEEE 754.
    let neighbor = Env::IEEE
        .with_rounding(rounding)
        .with_saturate(env.saturate && format.specials != Specials::Ieee);
    let (result, _) = mpfr::round(&input, format, &neighbor);
    (Outcome::from_value(result), flags)
}

/// Returns the expected encoding of a sign operation: the encoding with its
/// sign bit set to `negative`. A zero and the NaN of `Fnuz` have one
/// encoding each, so they do not change.
#[must_use]
pub fn with_sign(sample: &Sample<8>, specials: Specials, negative: bool) -> Integer {
    let sign = sample.layout.width - 1;
    let mut bits = sample.bits.clone();
    let magnitude_zero = bits.clone().keep_bits(sign).is_zero();
    if specials == Specials::Fnuz && magnitude_zero {
        return bits;
    }
    bits.set_bit(sign, negative);
    bits
}
