//! Comparison, total order, and the minimum and maximum operations of the
//! binary formats.
//!
//! None of these operations rounds. The minimum and maximum operations return
//! one operand in its canonical encoding, as IEEE 754 requires, or a NaN.

use core::cmp::Ordering;

use super::Layout;
use crate::env::{Env, Flags, TotalOrder};
use crate::format::EncodingKind;
use crate::format::internal::MinMax;
use crate::format::{Encoding, Storage, Width};
use crate::limbs::Limbs;
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

/// Orders the magnitudes of two numbers: zeros, finite values, or
/// infinities.
///
/// A canonical finite value has a full-precision significand unless it is
/// subnormal, and a subnormal value has the smallest exponent. So the exponent
/// orders two magnitudes first, and the significand orders them at one
/// exponent.
fn compare_magnitudes<L: Limbs>(first: &Unpacked<L>, second: &Unpacked<L>) -> Ordering {
    let key = |value: &Unpacked<L>| match *value {
        Unpacked::Zero { .. } => (0, 0, L::ZERO),
        Unpacked::Finite {
            exponent,
            significand,
            ..
        } => (1, exponent, significand),
        Unpacked::Infinity { .. } => (2, 0, L::ZERO),
        Unpacked::Nan { .. } | Unpacked::Unsupported => {
            unreachable!("the caller orders only numbers")
        }
    };
    let (first_rank, first_exponent, first_significand) = key(first);
    let (second_rank, second_exponent, second_significand) = key(second);
    first_rank
        .cmp(&second_rank)
        .then(first_exponent.cmp(&second_exponent))
        .then_with(|| first_significand.compare(&second_significand))
}

/// A number whose encoding orders as its value: the numbers of the formats
/// with an implicit integer bit, and the canonical x87 numbers. The sign and
/// the magnitude bits order such numbers without a decode.
struct Ordered<L> {
    /// `true` for a negative sign.
    negative: bool,
    /// The encoding without the sign bit.
    magnitude: L,
    /// `true` for a subnormal number, which reports a denormal input.
    subnormal: bool,
}

impl<L: Limbs> Ordered<L> {
    /// Orders two numbers by sign and magnitude, with `-0` below `+0`.
    #[inline]
    fn total(&self, other: &Self) -> Ordering {
        match (self.negative, other.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => self.magnitude.compare(&other.magnitude),
            (true, true) => other.magnitude.compare(&self.magnitude),
        }
    }

    /// Orders two numbers as the comparison predicates do: zeros of either
    /// sign are equal.
    #[inline]
    fn numeric(&self, other: &Self) -> Ordering {
        if self.magnitude.is_zero() && other.magnitude.is_zero() {
            Ordering::Equal
        } else {
            self.total(other)
        }
    }
}

/// Orders two numbers. Zeros of either sign are equal.
fn compare_numbers<L: Limbs>(first: &Unpacked<L>, second: &Unpacked<L>) -> Ordering {
    let below_zero = |value: &Unpacked<L>| match *value {
        Unpacked::Finite { negative, .. } | Unpacked::Infinity { negative } => negative,
        _ => false,
    };
    match (below_zero(first), below_zero(second)) {
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => compare_magnitudes(first, second),
        (true, true) => compare_magnitudes(second, first),
    }
}

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns a number whose encoding orders as its value, or `None` for a
    /// NaN, an unsupported x87 encoding, and an x87 pseudo-denormal. The test
    /// reads only the fields, so two such numbers compare without a decode.
    #[inline]
    fn ordered<L: Limbs>(bits: L) -> Option<Ordered<L>> {
        let negative = bits.bit(Self::WIDTH - 1);
        let magnitude = bits.low_bits(Self::WIDTH - 1);
        let field = bits.field(Self::FRACTION_BITS, E);
        let fraction = bits.low_bits(Self::FRACTION_BITS);
        let number = match Enc::KIND {
            // A NaN has the largest exponent field and a fraction other than
            // zero.
            EncodingKind::Ieee => field != Self::FIELD_MAX || fraction.is_zero(),
            // The one NaN of each sign has every magnitude bit set.
            EncodingKind::NoInf => magnitude != L::ones(Self::WIDTH - 1),
            // The one NaN has the sign bit set and a zero magnitude.
            EncodingKind::Fnuz => !negative || !magnitude.is_zero(),
            // Every encoding is a number.
            EncodingKind::Finite => true,
            // The integer bit, bit 63, is set exactly when the exponent field
            // is not zero. An infinity has a zero fraction below it.
            EncodingKind::X87 => {
                if field == 0 {
                    !fraction.bit(63)
                } else if field == Self::FIELD_MAX {
                    fraction.bit(63) && fraction.low_bits(63).is_zero()
                } else {
                    fraction.bit(63)
                }
            }
        };
        number.then(|| Ordered {
            negative,
            magnitude,
            subnormal: field == 0 && !fraction.is_zero(),
        })
    }

    /// Returns two numbers whose encodings order as their values, and the
    /// flags of their decode, or `None` when the full decode must run. DAZ
    /// makes a subnormal operand a zero, so that operand takes the decode.
    ///
    /// A format wider than 128 bits takes the decode too. Its magnitudes span
    /// four or eight limbs, and their copies cost more than the decode saves.
    #[inline]
    fn ordered_pair<L: Limbs>(
        left: L,
        right: L,
        env: &Env,
    ) -> Option<(Ordered<L>, Ordered<L>, Flags)> {
        if Self::WIDTH > 128 {
            return None;
        }
        let (first, second) = (Self::ordered(left)?, Self::ordered(right)?);
        let subnormal = first.subnormal || second.subnormal;
        if subnormal && env.denormals_are_zero {
            return None;
        }
        let flags = if subnormal {
            Flags::DENORMAL_INPUT
        } else {
            Flags::NONE
        };
        Some((first, second, flags))
    }

    /// Compares two values as the IEEE 754 quiet predicates do. `None` means
    /// that the values are unordered.
    ///
    /// A signaling NaN or an unsupported operand signals invalid. The
    /// signaling predicates also signal invalid for a quiet NaN; the caller
    /// adds that flag.
    #[inline]
    pub fn compare<L: Limbs>(left: L, right: L, env: &Env) -> (Option<Ordering>, Flags) {
        match Self::ordered_pair(left, right, env) {
            Some((first, second, flags)) => (Some(first.numeric(&second)), flags),
            None => Self::compare_decoded(left, right, env),
        }
    }

    /// Compares two values after the full decode, for a pair that
    /// `ordered_pair` does not order.
    #[inline(never)]
    fn compare_decoded<L: Limbs>(left: L, right: L, env: &Env) -> (Option<Ordering>, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        if matches!(first, Unpacked::Unsupported) || matches!(second, Unpacked::Unsupported) {
            return (None, flags | Flags::INVALID);
        }
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
    /// The order compares the sign, and then the magnitude bits: the exponent
    /// field and the fraction. For an IEEE 754 encoding this is the order of
    /// `totalOrder`: a negative NaN first, then the numbers, then a positive
    /// NaN. A signaling NaN is nearer the numbers than a quiet NaN. The one
    /// NaN of `Fnuz` has only the sign bit set; it orders first, as a negative
    /// NaN does.
    ///
    /// An x87 pseudo-denormal has a canonical twin: the normal encoding with
    /// exponent field 1. With [`TotalOrder::Datum`] the pseudo-denormal
    /// orders as its twin, and with [`TotalOrder::Encoding`] by its own bits.
    /// An unsupported x87 encoding is no datum, so it orders by its bits.
    pub fn total_cmp<L: Limbs>(left: L, right: L, order: TotalOrder) -> Ordering {
        let canonical = |bits: L| {
            if order != TotalOrder::Datum || !matches!(Enc::KIND, EncodingKind::X87) {
                return bits;
            }
            match Self::decode(bits) {
                Unpacked::Unsupported => bits,
                value => Self::encode(value),
            }
        };
        let (left, right) = (canonical(left), canonical(right));
        let key = |bits: L| {
            let magnitude = bits.low_bits(Self::WIDTH - 1);
            let negative = bits.bit(Self::WIDTH - 1);
            if matches!(Enc::KIND, EncodingKind::Fnuz) && negative && magnitude.is_zero() {
                // Above every magnitude, so the NaN orders first.
                (true, L::ones(Self::WIDTH))
            } else {
                (negative, magnitude)
            }
        };
        let ((first_negative, first), (second_negative, second)) = (key(left), key(right));
        match (first_negative, second_negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (false, false) => first.compare(&second),
            (true, true) => second.compare(&first),
        }
    }

    /// Returns the minimum or the maximum of one family.
    ///
    /// `minimum` and `maximum` return a NaN for a NaN operand.
    /// `minimumNumber` and `maximumNumber` return the number, and signal
    /// invalid for a signaling NaN. `minNum` and `maxNum` return the number
    /// for a quiet NaN, and a NaN for a signaling NaN. The NaN rule selects a
    /// NaN result. Every family orders `-0` below `+0`. A magnitude operation
    /// orders the magnitudes first, and then the values.
    #[inline]
    pub fn min_max<L: Limbs>(left: L, right: L, operation: MinMax, env: &Env) -> (L, Flags) {
        if operation.is_magnitude() {
            return Self::min_max_decoded(left, right, operation, env);
        }
        let Some((first, second, flags)) = Self::ordered_pair(left, right, env) else {
            return Self::min_max_decoded(left, right, operation, env);
        };
        // An ordered encoding is canonical, so the result is the operand.
        let order = first.total(&second);
        let take_first = if operation.is_minimum() {
            order != Ordering::Greater
        } else {
            order != Ordering::Less
        };
        (if take_first { left } else { right }, flags)
    }

    /// Returns the minimum or the maximum after the full decode, for a pair
    /// that `ordered_pair` does not order.
    #[inline(never)]
    fn min_max_decoded<L: Limbs>(left: L, right: L, operation: MinMax, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        if matches!(first, Unpacked::Unsupported) || matches!(second, Unpacked::Unsupported) {
            return Self::exact(default_nan(env), flags | Flags::INVALID);
        }
        if let Some((value, special)) = nan::min_max(&first, &second, operation, env) {
            return Self::exact(value, flags | special);
        }
        let order = match compare_numbers(&first, &second) {
            // Zeros of different signs: -0 orders below +0.
            Ordering::Equal => match (first, second) {
                (
                    Unpacked::Zero {
                        negative: first_negative,
                        ..
                    },
                    Unpacked::Zero {
                        negative: second_negative,
                        ..
                    },
                ) => second_negative.cmp(&first_negative),
                _ => Ordering::Equal,
            },
            order => order,
        };
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

    use crate::env::{Env, Flags};
    use crate::float::{F8E4M3Fnuz, F32};

    const QUIET: u32 = 0x7FC0_0001;
    const SIGNALING: u32 = 0x7F80_0001;

    fn f32(bits: u32) -> F32 {
        F32::from_bits(bits)
    }

    #[test]
    fn quiet_and_signaling_predicates_differ_only_for_a_quiet_nan() {
        let (one, zero) = (f32(0x3F80_0000), f32(0));
        assert_eq!(
            zero.compare_quiet_with(f32(0x8000_0000), Env::IEEE),
            (Some(Ordering::Equal), Flags::NONE)
        );
        assert_eq!(
            one.compare_quiet_with(f32(QUIET), Env::IEEE),
            (None, Flags::NONE)
        );
        assert_eq!(
            one.compare_signaling_with(f32(QUIET), Env::IEEE),
            (None, Flags::INVALID)
        );
        assert_eq!(
            one.compare_quiet_with(f32(SIGNALING), Env::IEEE),
            (None, Flags::INVALID)
        );
        let daz = Env::IEEE.with_denormals_are_zero(true);
        assert_eq!(
            f32(1).compare_quiet_with(zero, daz),
            (Some(Ordering::Equal), Flags::DENORMAL_INPUT)
        );
        assert!(f32(0xBF80_0000) < one && f32(1) > zero);
    }

    #[test]
    fn total_order_places_nans_by_sign_kind_and_payload() {
        let ordered = [
            0xFFC0_0002,
            0xFFC0_0001,
            0xFF80_0001,
            0xFF80_0000,
            0xBF80_0000,
            0x8000_0000,
            0,
            1,
            0x7F80_0000,
            0x7F80_0001,
            0x7FC0_0000,
            0x7FC0_0001,
        ];
        for pair in ordered.windows(2) {
            assert_eq!(
                f32(pair[0]).total_cmp(f32(pair[1])),
                Ordering::Less,
                "{pair:x?}"
            );
        }
        let nan = F8E4M3Fnuz::from_bits(0x80);
        assert_eq!(nan.total_cmp(F8E4M3Fnuz::from_bits(0xFF)), Ordering::Less);
    }

    #[test]
    fn each_minimum_family_treats_nans_and_zeros_by_its_standard() {
        let (one, negative_zero, zero) = (f32(0x3F80_0000), f32(0x8000_0000), f32(0));
        let bits = |(value, flags): (F32, Flags)| (value.to_bits(), flags);
        assert_eq!(
            bits(zero.minimum_with(negative_zero, Env::IEEE)),
            (0x8000_0000, Flags::NONE)
        );
        assert_eq!(
            bits(negative_zero.max_num_with(zero, Env::IEEE)),
            (0, Flags::NONE)
        );
        assert_eq!(
            bits(one.minimum_with(f32(QUIET), Env::IEEE)),
            (QUIET, Flags::NONE)
        );
        assert_eq!(
            bits(one.minimum_number_with(f32(QUIET), Env::IEEE)),
            (0x3F80_0000, Flags::NONE)
        );
        assert_eq!(
            bits(one.minimum_number_with(f32(SIGNALING), Env::IEEE)),
            (0x3F80_0000, Flags::INVALID)
        );
        assert_eq!(
            bits(one.min_num_with(f32(QUIET), Env::IEEE)),
            (0x3F80_0000, Flags::NONE)
        );
        assert_eq!(
            bits(one.min_num_with(f32(SIGNALING), Env::IEEE)),
            (0x7FC0_0001, Flags::INVALID)
        );
        assert_eq!(
            bits(f32(QUIET).max_num_with(f32(0x7FC0_0002), Env::IEEE)).0,
            QUIET
        );
    }

    #[test]
    fn an_x87_pseudo_denormal_orders_as_its_twin() {
        use crate::TotalOrder;
        use crate::float::F80;
        // Exponent field 0 with the integer bit set, and the normal encoding
        // of the same value with exponent field 1.
        let pseudo = F80::from_bits((1 << 63) | 5);
        let twin = F80::from_bits((1 << 64) | (1 << 63) | 5);
        assert_eq!(pseudo.total_cmp(twin), Ordering::Equal);
        assert_eq!(
            pseudo.total_cmp_with(twin, TotalOrder::Encoding),
            Ordering::Less
        );
        let above = F80::from_bits((1 << 64) | (1 << 63) | 6);
        assert_eq!(pseudo.total_cmp(above), Ordering::Less);
        // An unnormal is no datum, so it orders by its bits in both rules.
        let unnormal = F80::from_bits((2 << 64) | 5);
        assert_eq!(unnormal.total_cmp(twin), Ordering::Greater);
        assert_eq!(
            unnormal.total_cmp_with(twin, TotalOrder::Encoding),
            Ordering::Greater
        );
    }
}
