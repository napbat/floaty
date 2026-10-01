//! Comparison, total order, and the minimum and maximum operations of the
//! decimal formats.
//!
//! Members of one cohort, such as 1.0 and 1.00, compare equal. The total
//! order and the minimum and maximum operations order them by sign and then
//! by exponent, as IEEE 754-2019 section 5.10 and decNumber do: for a positive
//! value the smaller exponent orders first, and for a negative value the
//! larger one.

use core::cmp::Ordering;

use super::digits::{digit_count, power_of_ten};
use super::{DecimalLayout, Wide};
use crate::env::{Env, Flags, TotalOrder};
use crate::format::internal::MinMax;
use crate::format::{DecimalEncoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan;
use crate::unpacked::Unpacked;

/// Orders the magnitudes of two numbers: zeros, finite values, or
/// infinities. Members of one cohort are equal.
fn compare_magnitudes<L: Widen>(first: &Unpacked<L>, second: &Unpacked<L>) -> Ordering {
    let rank = |value: &Unpacked<L>| match value {
        Unpacked::Zero { .. } => 0,
        Unpacked::Finite { .. } => 1,
        Unpacked::Infinity { .. } => 2,
        Unpacked::Nan { .. } | Unpacked::Unsupported => {
            unreachable!("the caller orders only numbers")
        }
    };
    let (
        Unpacked::Finite {
            exponent: first_exponent,
            significand: first_significand,
            ..
        },
        Unpacked::Finite {
            exponent: second_exponent,
            significand: second_significand,
            ..
        },
    ) = (*first, *second)
    else {
        return rank(first).cmp(&rank(second));
    };
    let (first_coefficient, second_coefficient): (Wide<L>, Wide<L>) =
        (first_significand.resize(), second_significand.resize());
    let top = |exponent: i32, coefficient: &Wide<L>| {
        i64::from(exponent) + i64::from(digit_count(coefficient))
    };
    let order =
        top(first_exponent, &first_coefficient).cmp(&top(second_exponent, &second_coefficient));
    if order != Ordering::Equal {
        return order;
    }
    // Equal leading weights: align the coefficients at the smaller exponent.
    // The exponents differ by less than the precision.
    let lowest = first_exponent.min(second_exponent);
    let align = |exponent: i32, coefficient: Wide<L>| {
        let shift = u32::try_from(exponent - lowest).expect("the shift is not negative");
        limbs::multiply_fit(coefficient, power_of_ten(shift))
    };
    align(first_exponent, first_coefficient).compare(&align(second_exponent, second_coefficient))
}

/// Returns `true` for a negative finite value or a negative infinity.
fn below_zero<L>(value: &Unpacked<L>) -> bool {
    matches!(
        value,
        Unpacked::Finite { negative: true, .. } | Unpacked::Infinity { negative: true }
    )
}

/// Orders two numbers by value. Zeros of either sign and members of one
/// cohort are equal.
fn compare_numbers<L: Widen>(first: &Unpacked<L>, second: &Unpacked<L>) -> Ordering {
    match (below_zero(first), below_zero(second)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => compare_magnitudes(first, second),
        (true, true) => compare_magnitudes(second, first),
    }
}

/// Orders two numbers of equal value: by sign, and then by exponent, the
/// smaller exponent first for a positive value.
fn order_in_cohort<L>(first: &Unpacked<L>, second: &Unpacked<L>) -> Ordering {
    let key = |value: &Unpacked<L>| match *value {
        Unpacked::Zero { negative, exponent }
        | Unpacked::Finite {
            negative, exponent, ..
        } => (negative, exponent),
        Unpacked::Infinity { negative } => (negative, 0),
        Unpacked::Nan { .. } | Unpacked::Unsupported => {
            unreachable!("the caller orders only numbers")
        }
    };
    let ((first_negative, first_exponent), (second_negative, second_exponent)) =
        (key(first), key(second));
    match (first_negative, second_negative) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => first_exponent.cmp(&second_exponent),
        (true, true) => second_exponent.cmp(&first_exponent),
    }
}

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Compares two values as the IEEE 754 quiet predicates do. `None` means
    /// that the values are unordered. A signaling NaN signals invalid.
    pub fn compare<L: Widen>(left: L, right: L, env: &Env) -> (Option<Ordering>, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        if first.is_nan() || second.is_nan() {
            if first.is_signaling() || second.is_signaling() {
                flags |= Flags::INVALID;
            }
            return (None, flags);
        }
        (Some(compare_numbers(&first, &second)), flags)
    }

    /// Orders two encodings as IEEE 754 `totalOrder` does.
    ///
    /// A negative NaN orders first, then the numbers, then a positive NaN.
    /// Members of one cohort order by exponent. A signaling NaN orders nearer
    /// the numbers than a quiet NaN, and a larger payload farther. Two
    /// encodings of one datum, such as a non-canonical infinity and the
    /// canonical one, are equal with [`TotalOrder::Datum`], as IEEE 754-2019
    /// section 5.10 states. With [`TotalOrder::Encoding`] they order by
    /// their bits below the sign, reversed for a negative sign.
    pub fn total_cmp<L: Widen>(left: L, right: L, order: TotalOrder) -> Ordering {
        let negative = |bits: &L| bits.bit(Self::WIDTH - 1);
        match (negative(&left), negative(&right)) {
            (true, false) => return Ordering::Less,
            (false, true) => return Ordering::Greater,
            _ => {}
        }
        let (first, second) = (Self::decode(left), Self::decode(right));
        let magnitude = match (first, second) {
            (
                Unpacked::Nan {
                    signaling: first_signaling,
                    payload: first_payload,
                    ..
                },
                Unpacked::Nan {
                    signaling: second_signaling,
                    payload: second_payload,
                    ..
                },
            ) => second_signaling
                .cmp(&first_signaling)
                .then_with(|| first_payload.compare(&second_payload)),
            (Unpacked::Nan { .. }, _) => Ordering::Greater,
            (_, Unpacked::Nan { .. }) => Ordering::Less,
            _ => compare_magnitudes(&first, &second).then_with(|| {
                let exponent = |value: &Unpacked<L>| match *value {
                    Unpacked::Zero { exponent, .. } | Unpacked::Finite { exponent, .. } => exponent,
                    _ => 0,
                };
                exponent(&first).cmp(&exponent(&second))
            }),
        };
        let magnitude = match order {
            TotalOrder::Datum => magnitude,
            TotalOrder::Encoding => magnitude.then_with(|| {
                left.low_bits(Self::WIDTH - 1)
                    .compare(&right.low_bits(Self::WIDTH - 1))
            }),
        };
        if negative(&left) {
            magnitude.reverse()
        } else {
            magnitude
        }
    }

    /// Returns the minimum or the maximum of one family. The NaN cases follow
    /// `nan::min_max`. Equal values order by sign and then by exponent, so
    /// the maximum of 1.0 and 1.00 is 1.0. A magnitude operation orders the
    /// magnitudes first, and then the values.
    pub fn min_max<L: Widen>(left: L, right: L, operation: MinMax, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::min_max(&first, &second, operation, env) {
            return Self::exact(value, flags | special);
        }
        let order = compare_numbers(&first, &second).then_with(|| order_in_cohort(&first, &second));
        let order = if operation.is_magnitude() {
            compare_numbers(&first.abs(), &second.abs()).then(order)
        } else {
            order
        };
        let take_first = if operation.is_minimum() {
            order != Ordering::Greater
        } else {
            order != Ordering::Less
        };
        Self::exact(if take_first { first } else { second }, flags)
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use crate::env::Env;
    use crate::exact::Exact;
    use crate::float::D64Bid;

    fn number(negative: bool, coefficient: u64, exponent: i32) -> D64Bid {
        let exact = Exact {
            negative,
            exponent,
            significand: [coefficient],
            sticky: false,
        };
        D64Bid::round(exact, Env::IEEE).0
    }

    #[test]
    fn cohort_members_compare_equal_and_order_by_exponent() {
        let (one_tenth, one_hundredth) = (number(false, 10, -1), number(false, 100, -2));
        assert_eq!(one_tenth.partial_cmp(&one_hundredth), Some(Ordering::Equal));
        assert_eq!(one_hundredth.total_cmp(one_tenth), Ordering::Less);
        assert_eq!(
            one_tenth.maximum(one_hundredth).to_bits(),
            one_tenth.to_bits()
        );
        assert_eq!(
            one_tenth.minimum(one_hundredth).to_bits(),
            one_hundredth.to_bits()
        );
        let (minus_tenth, minus_hundredth) = (number(true, 10, -1), number(true, 100, -2));
        assert_eq!(minus_tenth.total_cmp(minus_hundredth), Ordering::Less);
        assert_eq!(
            minus_tenth.maximum(minus_hundredth).to_bits(),
            minus_hundredth.to_bits()
        );
        assert!(number(false, 2, 0) > number(false, 19, -1));
        // Two encodings of one datum are equal: an infinity with trailing
        // bits, and a coefficient above 10^16 - 1, which reads as zero.
        let infinity = D64Bid::from_bits(0x7800_0000_0000_0000);
        let wide_infinity = D64Bid::from_bits(0x7800_0000_0000_0001);
        assert_eq!(wide_infinity.total_cmp(infinity), Ordering::Equal);
        let encoding = crate::TotalOrder::Encoding;
        assert_eq!(
            wide_infinity.total_cmp_with(infinity, encoding),
            Ordering::Greater
        );
        assert_eq!(
            (-wide_infinity).total_cmp_with(-infinity, encoding),
            Ordering::Less
        );
        let zero = D64Bid::from_bits(0x31C0_0000_0000_0000);
        let wide_zero = D64Bid::from_bits(0x6C77_FFFF_FFFF_FFFF);
        assert_eq!(wide_zero.total_cmp(zero), Ordering::Equal);
        assert_eq!((-wide_zero).total_cmp(-zero), Ordering::Equal);
        assert_eq!(number(true, 0, 5), number(false, 0, -3));
    }
}
