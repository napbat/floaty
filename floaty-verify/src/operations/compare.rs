//! The oracle of comparison, total order, and the minimum and maximum
//! operations.

use core::cmp::Ordering;

use floaty::{Decoded, Env, Flags, TotalOrder};
use rug::Integer;
use rug::integer::Order;

use super::{Outcome, Sample, default_nan, number, propagate};
use crate::encodings::IntegerBit;
use crate::mpfr::{Format, Operand, Read};

/// Returns the expected order and flags of `compare_quiet_with`.
///
/// IEEE 754-2019 section 5.11: zeros of either sign are equal, a NaN is
/// unordered, and a quiet predicate signals invalid only for a signaling
/// NaN. floaty adds that an unsupported encoding is unordered and
/// signals invalid.
#[must_use]
pub fn compare_quiet<const N: usize>(
    first: &Operand<N>,
    second: &Operand<N>,
    env: &Env,
) -> (Option<Ordering>, Flags) {
    let mut flags = Flags::NONE;
    let operands = [first.read(env, &mut flags), second.read(env, &mut flags)];
    if operands
        .iter()
        .any(|operand| matches!(operand, Read::Unsupported) || operand.is_signaling())
    {
        return (None, flags | Flags::INVALID);
    }
    match &operands {
        [Read::Number(a), Read::Number(b)] => (a.partial_cmp(b), flags),
        _ => (None, flags),
    }
}

/// Returns the expected order and flags of `compare_signaling_with`.
///
/// IEEE 754-2019 section 5.11: a signaling predicate signals invalid for
/// every unordered pair.
#[must_use]
pub fn compare_signaling<const N: usize>(
    first: &Operand<N>,
    second: &Operand<N>,
    env: &Env,
) -> (Option<Ordering>, Flags) {
    let (order, flags) = compare_quiet(first, second, env);
    match order {
        Some(_) => (order, flags),
        None => (None, flags | Flags::INVALID),
    }
}

/// Returns the expected order of `total_cmp`.
///
/// Two canonical encodings order by IEEE 754-2019 section 5.10: a negative
/// NaN below every number, and every number below a positive NaN. Numbers
/// order by value, with `-0` below `+0`, as MPFR's `mpfr_total_order_p`
/// orders them. Among positive NaNs a signaling NaN
/// orders below a quiet NaN, and a smaller payload below a larger one; the
/// negative NaNs order in reverse. The NaN of `Fnuz` is negative.
///
/// An encoding that is not canonical orders by the rules of `TotalOrder`.
/// With `TotalOrder::Encoding` it orders by the sign bit, and then by the
/// magnitude bits. With `TotalOrder::Datum` an x87 pseudo-denormal is its
/// datum, the equal normal value, and an unsupported encoding orders by the
/// sign bit and the magnitude bits of its own encoding and the canonical
/// encoding of the other operand.
///
/// # Panics
///
/// Panics for an unsupported encoding that `canonical` does not exclude.
#[must_use]
pub fn total_order<const N: usize>(
    first: &Sample<N>,
    second: &Sample<N>,
    order: TotalOrder,
) -> Ordering {
    let datum = |sample: &Sample<N>| !matches!(sample.operand.decoded, Decoded::Unsupported);
    match order {
        TotalOrder::Encoding if !first.canonical || !second.canonical => {
            return by_bits(&first.bits, &second.bits, first.layout.width);
        }
        TotalOrder::Datum if !datum(first) || !datum(second) => {
            return by_bits(
                &canonical_bits(first),
                &canonical_bits(second),
                first.layout.width,
            );
        }
        _ => {}
    }
    let rank = |sample: &Sample<N>| match sample.operand.decoded {
        Decoded::Nan {
            negative: true,
            signaling,
            ref payload,
        } => (
            0,
            Some((!signaling, Integer::from_digits(payload, Order::Lsf))),
        ),
        Decoded::Nan {
            negative: false,
            signaling,
            ref payload,
        } => (
            2,
            Some((!signaling, Integer::from_digits(payload, Order::Lsf))),
        ),
        Decoded::Unsupported => panic!("an unsupported encoding is not canonical"),
        _ => (1, None),
    };
    let ((first_rank, first_nan), (second_rank, second_nan)) = (rank(first), rank(second));
    first_rank.cmp(&second_rank).then_with(|| match first_rank {
        0 => second_nan.cmp(&first_nan),
        2 => first_nan.cmp(&second_nan),
        _ => {
            let mut flags = Flags::NONE;
            let operands = [
                first.operand.read(&Env::IEEE, &mut flags),
                second.operand.read(&Env::IEEE, &mut flags),
            ];
            let [Read::Number(a), Read::Number(b)] = &operands else {
                unreachable!("the rank holds numbers");
            };
            a.total_cmp(b)
        }
    })
}

/// Returns the canonical encoding of a datum, or the encoding of an
/// unsupported operand. The only binary encoding of a datum that is not
/// canonical is a pseudo-denormal of a format with a stored integer bit, x87
/// extended: exponent field 0 with the integer bit set. The normal encoding
/// of its value has exponent field 1, so it sets the lowest bit of the field.
fn canonical_bits<const N: usize>(sample: &Sample<N>) -> Integer {
    if sample.canonical || matches!(sample.operand.decoded, Decoded::Unsupported) {
        return sample.bits.clone();
    }
    assert_eq!(
        sample.layout.integer_bit,
        IntegerBit::Explicit,
        "only an encoding with a stored integer bit is not canonical"
    );
    let mut twin = sample.bits.clone();
    twin.set_bit(sample.layout.fraction_bits(), true);
    twin
}

/// Orders two encodings of `width` bits by the sign bit and then the
/// magnitude bits.
fn by_bits(first: &Integer, second: &Integer, width: u32) -> Ordering {
    let key = |bits: &Integer| {
        let sign = width - 1;
        let mut magnitude = bits.clone();
        magnitude.set_bit(sign, false);
        (bits.get_bit(sign), magnitude)
    };
    let ((first_negative, first_magnitude), (second_negative, second_magnitude)) =
        (key(first), key(second));
    match (first_negative, second_negative) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => first_magnitude.cmp(&second_magnitude),
        (true, true) => second_magnitude.cmp(&first_magnitude),
    }
}

/// A minimum or maximum operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinMax {
    /// IEEE 754-2019 `minimum`.
    Minimum,
    /// IEEE 754-2019 `maximum`.
    Maximum,
    /// IEEE 754-2019 `minimumNumber`.
    MinimumNumber,
    /// IEEE 754-2019 `maximumNumber`.
    MaximumNumber,
    /// IEEE 754-2008 `minNum`.
    MinNum,
    /// IEEE 754-2008 `maxNum`.
    MaxNum,
    /// IEEE 754-2019 `minimumMagnitude`.
    MinimumMagnitude,
    /// IEEE 754-2019 `maximumMagnitude`.
    MaximumMagnitude,
    /// IEEE 754-2019 `minimumMagnitudeNumber`.
    MinimumMagnitudeNumber,
    /// IEEE 754-2019 `maximumMagnitudeNumber`.
    MaximumMagnitudeNumber,
}

impl MinMax {
    /// Every operation.
    pub const ALL: [Self; 10] = [
        Self::Minimum,
        Self::Maximum,
        Self::MinimumNumber,
        Self::MaximumNumber,
        Self::MinNum,
        Self::MaxNum,
        Self::MinimumMagnitude,
        Self::MaximumMagnitude,
        Self::MinimumMagnitudeNumber,
        Self::MaximumMagnitudeNumber,
    ];
}

/// Returns the expected result and flags of a minimum or maximum operation.
///
/// IEEE 754-2019 section 9.6: `minimum` gives a NaN for a NaN operand, and
/// `minimumNumber` gives the number and signals invalid for a signaling NaN.
/// `minimumMagnitude` and `minimumMagnitudeNumber` give the operand of the
/// smaller magnitude, and `minimum` or `minimumNumber` of operands of one
/// magnitude. IEEE 754-2008 section 5.3.1: `minNum` gives the number for a
/// quiet NaN and a NaN for a signaling NaN. floaty orders `-0` below `+0` in
/// every family, and returns the operand in its canonical encoding.
///
/// # Panics
///
/// Never: MPFR compares the magnitudes of two numbers.
#[must_use]
pub fn min_max<const N: usize>(
    operation: MinMax,
    first: &Operand<N>,
    second: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let operands = [first.read(env, &mut flags), second.read(env, &mut flags)];
    if operands
        .iter()
        .any(|operand| matches!(operand, Read::Unsupported))
    {
        return (default_nan(format, env), flags | Flags::INVALID);
    }
    let signaling = operands.iter().any(Read::is_signaling);
    let (a, b) = match &operands {
        [Read::Number(a), Read::Number(b)] => (a, b),
        [Read::Number(value), _] | [_, Read::Number(value)] => {
            let gives_nan = match operation {
                MinMax::Minimum
                | MinMax::Maximum
                | MinMax::MinimumMagnitude
                | MinMax::MaximumMagnitude => true,
                MinMax::MinNum | MinMax::MaxNum => signaling,
                MinMax::MinimumNumber
                | MinMax::MaximumNumber
                | MinMax::MinimumMagnitudeNumber
                | MinMax::MaximumMagnitudeNumber => false,
            };
            if !gives_nan {
                let invalid = if signaling {
                    Flags::INVALID
                } else {
                    Flags::NONE
                };
                return (number(value, format), flags | invalid);
            }
            let (nan, special) = propagate(&operands, format, env);
            return (nan, flags | special);
        }
        _ => {
            let (nan, special) = propagate(&operands, format, env);
            return (nan, flags | special);
        }
    };
    // MPFR's total order orders numbers by value, with -0 below +0.
    let order = a.total_cmp(b);
    let magnitude = matches!(
        operation,
        MinMax::MinimumMagnitude
            | MinMax::MaximumMagnitude
            | MinMax::MinimumMagnitudeNumber
            | MinMax::MaximumMagnitudeNumber
    );
    let order = if magnitude {
        a.cmp_abs(b)
            .expect("two numbers compare in magnitude")
            .then(order)
    } else {
        order
    };
    let minimum = matches!(
        operation,
        MinMax::Minimum
            | MinMax::MinimumNumber
            | MinMax::MinNum
            | MinMax::MinimumMagnitude
            | MinMax::MinimumMagnitudeNumber
    );
    let chosen = match (minimum, order) {
        (true, Ordering::Greater) | (false, Ordering::Less) => b,
        _ => a,
    };
    (number(chosen, format), flags)
}
