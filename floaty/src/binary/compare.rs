//! Comparison, total order, and the minimum and maximum operations of the
//! binary formats.
//!
//! None of these operations rounds. The minimum and maximum operations return
//! one operand in its canonical encoding, as IEEE 754 requires, or a NaN.

use core::cmp::Ordering;

use super::nan::{self, default_nan};
use super::{EncodingKind, Layout, Unpacked};
use crate::env::{Env, Flags};
use crate::format::internal::MinMax;
use crate::format::{Encoding, Storage, Width};
use crate::limbs::Limbs;

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
    /// Compares two values as the IEEE 754 quiet predicates do. `None` means
    /// that the values are unordered.
    ///
    /// A signaling NaN or an unsupported operand signals invalid. The
    /// signaling predicates also signal invalid for a quiet NaN; the caller
    /// adds that flag.
    pub fn compare<L: Limbs>(left: L, right: L, env: &Env) -> (Option<Ordering>, Flags) {
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
    /// NaN does. An x87 encoding that is not canonical orders by its bits.
    pub fn total_cmp<L: Limbs>(left: L, right: L) -> Ordering {
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
    /// NaN result. Every family orders `-0` below `+0`.
    pub fn min_max<L: Limbs>(left: L, right: L, operation: MinMax, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        if matches!(first, Unpacked::Unsupported) || matches!(second, Unpacked::Unsupported) {
            return Self::exact(default_nan(env), flags | Flags::INVALID);
        }
        let signaling = first.is_signaling() || second.is_signaling();
        match (first.is_nan(), second.is_nan()) {
            (false, false) => {}
            (true, true) => {
                let (value, special) = nan::propagate(&first, &second, env);
                return Self::exact(value, flags | special);
            }
            (true, false) | (false, true) => {
                let returns_nan = match operation {
                    MinMax::Minimum | MinMax::Maximum => true,
                    MinMax::MinNum | MinMax::MaxNum => signaling,
                    MinMax::MinimumNumber | MinMax::MaximumNumber => false,
                };
                if returns_nan {
                    let (value, special) = nan::propagate(&first, &second, env);
                    return Self::exact(value, flags | special);
                }
                if signaling {
                    flags |= Flags::INVALID;
                }
                let number = if first.is_nan() { second } else { first };
                return Self::exact(number, flags);
            }
        }
        let order = match compare_numbers(&first, &second) {
            // Zeros of different signs: -0 orders below +0.
            Ordering::Equal => match (first, second) {
                (
                    Unpacked::Zero {
                        negative: first_negative,
                    },
                    Unpacked::Zero {
                        negative: second_negative,
                    },
                ) => second_negative.cmp(&first_negative),
                _ => Ordering::Equal,
            },
            order => order,
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
}
