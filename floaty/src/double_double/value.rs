//! The exact value `hi + lo` of a pair, and the operations that only read
//! it: decoding, the sign, the comparisons, the class, and the total order.

use core::cmp::Ordering;

use super::{Algorithm, DoubleDouble, glibc};
use crate::env::{Flags, Mode};
use crate::float::{Class, Decoded, F64};
use crate::limbs::Limbs;
use crate::unpacked::Unpacked;

/// `|hi + lo| * 2^1074`. Every finite pair is a multiple of 2^-1074 below
/// 2^1025, so 2,099 bits hold the exact value of every finite pair.
pub(super) type Magnitude = [u64; 33];

/// The weight of the lowest bit of a [`Magnitude`].
const LOWEST: i32 = -1074;

/// The encoding bits of the exponent field of binary64, all ones in a NaN
/// and an infinity.
const SPECIAL_FIELD: u64 = 0x7FF0_0000_0000_0000;

/// The bit length of `2^-969 * 2^1074`, the magnitude of `LDBL_MIN`: a
/// finite value with a shorter [`Magnitude`] is below `LDBL_MIN`.
const NORMAL_BITS: u32 = 106;

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// Returns the value with both halves negated when the exact value has a
    /// negative sign: a negative number, `-0`, a negative infinity, or a NaN
    /// with a negative sign. The operation signals nothing.
    #[must_use]
    pub fn abs(self) -> Self {
        if self.is_sign_negative() { -self } else { self }
    }

    /// Returns the exact value `hi + lo`.
    ///
    /// A NaN or an infinite high half gives that value. With a finite high
    /// half, a NaN or an infinite low half gives that value. Otherwise a
    /// finite value has an odd significand, and a zero takes the sign of the
    /// high half. 33 limbs hold every significand.
    #[must_use]
    pub fn decode(self) -> Decoded<33> {
        match self.exact() {
            Unpacked::Zero { negative, .. } => Decoded::Zero {
                negative,
                exponent: 0,
            },
            Unpacked::Finite {
                negative,
                significand,
                ..
            } => {
                let (zeros, exponent) = lowest_bit(&significand);
                Decoded::Finite {
                    negative,
                    exponent,
                    significand: significand.shr(zeros),
                }
            }
            Unpacked::Infinity { negative } => Decoded::Infinity { negative },
            Unpacked::Nan {
                negative,
                signaling,
                payload,
            } => Decoded::Nan {
                negative,
                signaling,
                payload,
            },
            Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
        }
    }

    /// Compares the exact values as the IEEE 754 quiet predicates do. `None`
    /// means unordered. A signaling NaN signals invalid.
    #[must_use]
    pub fn compare_quiet(self, other: Self) -> (Option<Ordering>, Flags) {
        let (order, signaling) = self.compare(other);
        let flags = if signaling {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        (order, flags)
    }

    /// Compares the exact values as the IEEE 754 signaling predicates do.
    /// `None` means unordered. Every NaN signals invalid.
    #[must_use]
    pub fn compare_signaling(self, other: Self) -> (Option<Ordering>, Flags) {
        let (order, _) = self.compare(other);
        let flags = if order.is_none() {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        (order, flags)
    }

    /// Returns the order of the exact values, and `true` when a value is a
    /// signaling NaN.
    fn compare(self, other: Self) -> (Option<Ordering>, bool) {
        let (first, second) = (self.exact(), other.exact());
        if first.is_nan() || second.is_nan() {
            return (None, first.is_signaling() || second.is_signaling());
        }
        // The values order as in the total order, but `-0` equals `+0`.
        if matches!(
            (&first, &second),
            (Unpacked::Zero { .. }, Unpacked::Zero { .. })
        ) {
            return (Some(Ordering::Equal), false);
        }
        (Some(order_values(&first, &second)), false)
    }

    /// Returns the exact value of the pair. The half of
    /// [`special_half`](Self::special_half) makes the value that half.
    /// Otherwise the value is the exact sum, and a zero sum takes the sign of
    /// the high half. A nonzero finite value has the exponent `-1074`, and its
    /// significand is its [`Magnitude`].
    pub(super) fn exact(self) -> Unpacked<Magnitude> {
        if let Some(half) = self.special_half() {
            return special(half.decode::<1>());
        }
        sum(self.hi.decode::<1>(), self.lo.decode::<1>())
    }

    /// Returns the half that makes the value a NaN or an infinity: a NaN or
    /// infinite high half, or, with a finite high half, a NaN or infinite low
    /// half. A pair of finite halves gives `None`.
    pub(super) fn special_half(self) -> Option<F64> {
        let special = |half: F64| half.to_bits() & SPECIAL_FIELD == SPECIAL_FIELD;
        if special(self.hi) {
            Some(self.hi)
        } else if special(self.lo) {
            Some(self.lo)
        } else {
            None
        }
    }

    /// Returns the class of the exact value.
    ///
    /// A NaN or an infinite high half gives its class. With a finite high
    /// half, a NaN or an infinite low half gives its class. A finite value
    /// below `2^-969` in magnitude, the `LDBL_MIN` of the IBM `long double`,
    /// is subnormal: below it, the low half of a pair cannot hold 53 bits.
    /// A pair whose halves cancel, such as `(1, -1)`, is a zero.
    #[must_use]
    pub fn classify(self) -> Class {
        match self.exact() {
            Unpacked::Zero { .. } => Class::Zero,
            Unpacked::Finite { significand, .. } => {
                if significand.bit_length() < NORMAL_BITS {
                    Class::Subnormal
                } else {
                    Class::Normal
                }
            }
            Unpacked::Infinity { .. } => Class::Infinite,
            Unpacked::Nan {
                signaling: true, ..
            } => Class::SignalingNan,
            Unpacked::Nan {
                signaling: false, ..
            } => Class::QuietNan,
            Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
        }
    }

    /// Returns `true` when the exact value has a negative sign: a negative
    /// number, `-0`, a negative infinity, or a NaN with a negative sign.
    #[must_use]
    pub fn is_sign_negative(self) -> bool {
        has_negative_sign(&self.exact())
    }

    /// Returns `true` when the exact value has a positive sign.
    #[must_use]
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` for a quiet or a signaling NaN.
    #[must_use]
    pub fn is_nan(self) -> bool {
        self.classify().is_nan()
    }

    /// Returns `true` for a signaling NaN.
    #[must_use]
    pub fn is_signaling_nan(self) -> bool {
        self.classify() == Class::SignalingNan
    }

    /// Returns `true` for a positive or a negative infinity.
    #[must_use]
    pub fn is_infinite(self) -> bool {
        self.classify() == Class::Infinite
    }

    /// Returns `true` for a zero, a subnormal, or a normal value.
    #[must_use]
    pub fn is_finite(self) -> bool {
        self.classify().is_finite()
    }

    /// Returns `true` for a zero.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.classify() == Class::Zero
    }

    /// Returns `true` for a subnormal value.
    #[must_use]
    pub fn is_subnormal(self) -> bool {
        self.classify() == Class::Subnormal
    }

    /// Returns `true` for a normal value.
    #[must_use]
    pub fn is_normal(self) -> bool {
        self.classify() == Class::Normal
    }

    /// Returns `true` when the pair is canonical, as `iscanonicall` of the
    /// IBM `long double` in glibc 2.43 decides.
    ///
    /// A pair with a zero low half is canonical, and so is a pair with a NaN
    /// high half. A pair with an infinite high half and a nonzero low half is
    /// not. Otherwise the low half must be below half an ulp of the high half
    /// in magnitude, or exactly half an ulp with an even high half. Without
    /// a precision limit, the rounding to a pair of the type documentation
    /// gives a canonical pair.
    #[must_use]
    pub fn is_canonical(self) -> bool {
        glibc::is_canonical((self.hi, self.lo))
    }

    /// Orders the pairs by a total order: by their exact values as IEEE 754
    /// `totalOrder` orders values, then by the halves.
    ///
    /// A negative NaN orders first, then the numbers from negative infinity
    /// to positive infinity with `-0` below `+0`, then a positive NaN. A
    /// signaling NaN orders nearer the numbers than a quiet NaN, and a larger
    /// payload farther. The NaN and the sign of a zero are those of the
    /// exact value. Two pairs of one exact value, such as `(1, 0)` and
    /// `(2, -1)`, order by the binary64 total order of their high halves,
    /// then of their low halves. The operation signals nothing.
    #[must_use]
    pub fn total_cmp(self, other: Self) -> Ordering {
        order_values(&self.exact(), &other.exact())
            .then_with(|| self.hi.total_cmp(other.hi))
            .then_with(|| self.lo.total_cmp(other.lo))
    }
}

/// Returns the NaN or the infinity of a half as an exact value.
fn special(half: Decoded<1>) -> Unpacked<Magnitude> {
    match half {
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => Unpacked::Nan {
            negative,
            signaling,
            payload: payload.resize(),
        },
        Decoded::Infinity { negative } => Unpacked::Infinity { negative },
        _ => unreachable!("the caller passes a NaN or an infinity"),
    }
}

/// Returns the sign and `|half| * 2^1074` of a zero or finite half.
fn scaled(half: Decoded<1>) -> (bool, Magnitude) {
    match half {
        Decoded::Zero { negative, .. } => (negative, Magnitude::ZERO),
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let shift = u32::try_from(exponent - LOWEST)
                .expect("a binary64 exponent is at or above the lowest weight");
            (negative, significand.resize::<Magnitude>().shl(shift))
        }
        _ => unreachable!("the caller passes a zero or a finite half"),
    }
}

/// Returns the exact sum of a zero or finite high half and a zero or finite
/// low half. A zero sum takes the sign of the high half.
fn sum(hi: Decoded<1>, lo: Decoded<1>) -> Unpacked<Magnitude> {
    let ((hi_negative, hi_magnitude), (lo_negative, lo_magnitude)) = (scaled(hi), scaled(lo));
    let (negative, magnitude) = if hi_negative == lo_negative {
        (hi_negative, hi_magnitude.add(lo_magnitude))
    } else {
        match hi_magnitude.compare(&lo_magnitude) {
            Ordering::Greater => (hi_negative, hi_magnitude.sub(lo_magnitude)),
            Ordering::Less => (lo_negative, lo_magnitude.sub(hi_magnitude)),
            Ordering::Equal => return Unpacked::zero(hi_negative),
        }
    };
    if magnitude.is_zero() {
        return Unpacked::zero(hi_negative);
    }
    Unpacked::Finite {
        negative,
        exponent: LOWEST,
        significand: magnitude,
    }
}

/// Returns the number of zero bits below the lowest set bit of a nonzero
/// magnitude, and the weight of that bit.
fn lowest_bit(magnitude: &Magnitude) -> (u32, i32) {
    let (index, limb) = magnitude
        .iter()
        .enumerate()
        .find(|(_, limb)| **limb != 0)
        .expect("the magnitude is not zero");
    let zeros = u32::try_from(index).expect("a limb index fits a u32") * 64 + limb.trailing_zeros();
    let weight = LOWEST + i32::try_from(zeros).expect("a bit index of 33 limbs fits an i32");
    (zeros, weight)
}

/// Returns `true` when an exact value has a negative sign, a NaN included.
fn has_negative_sign(value: &Unpacked<Magnitude>) -> bool {
    match *value {
        Unpacked::Zero { negative, .. }
        | Unpacked::Finite { negative, .. }
        | Unpacked::Infinity { negative }
        | Unpacked::Nan { negative, .. } => negative,
        Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
    }
}

/// Orders two exact values as IEEE 754 `totalOrder` orders values of one
/// format.
pub(super) fn order_values(first: &Unpacked<Magnitude>, second: &Unpacked<Magnitude>) -> Ordering {
    let (first_negative, second_negative) = (has_negative_sign(first), has_negative_sign(second));
    if first_negative != second_negative {
        return if first_negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let order = rank(first)
        .cmp(&rank(second))
        .then_with(|| magnitude(first).compare(&magnitude(second)));
    if first_negative {
        order.reverse()
    } else {
        order
    }
}

/// Returns the place of a class of values in the total order of the
/// magnitudes: zero, a finite value, an infinity, a signaling NaN, a quiet
/// NaN.
fn rank(value: &Unpacked<Magnitude>) -> u8 {
    match value {
        Unpacked::Zero { .. } => 0,
        Unpacked::Finite { .. } => 1,
        Unpacked::Infinity { .. } => 2,
        Unpacked::Nan {
            signaling: true, ..
        } => 3,
        Unpacked::Nan {
            signaling: false, ..
        } => 4,
        Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
    }
}

/// Returns the magnitude of a finite value, or the payload of a NaN, which
/// orders values of one rank.
fn magnitude(value: &Unpacked<Magnitude>) -> Magnitude {
    match *value {
        Unpacked::Finite { significand, .. } => significand,
        Unpacked::Nan { payload, .. } => payload,
        Unpacked::Zero { .. } | Unpacked::Infinity { .. } | Unpacked::Unsupported => {
            Magnitude::ZERO
        }
    }
}

impl<Alg: Algorithm, M: Mode> PartialEq for DoubleDouble<Alg, M> {
    /// Compares the exact values as the quiet equality predicate does.
    fn eq(&self, other: &Self) -> bool {
        self.compare_quiet(*other).0 == Some(Ordering::Equal)
    }
}

impl<Alg: Algorithm, M: Mode> PartialOrd for DoubleDouble<Alg, M> {
    /// Orders the exact values as the quiet predicates do.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.compare_quiet(*other).0
    }
}

#[cfg(test)]
mod tests {
    use core::cmp::Ordering;

    use crate::double_double::{DoubleDouble, Gcc};
    use crate::env::{Flags, Rounding};
    use crate::float::{Class, Decoded, F32, F64};

    const ONE: u64 = 0x3FF0_0000_0000_0000;

    fn gcc(hi: u64, lo: u64) -> DoubleDouble<Gcc> {
        DoubleDouble::from_parts(F64::from_bits(hi), F64::from_bits(lo))
    }

    fn bits(value: DoubleDouble<Gcc>) -> (u64, u64) {
        (value.hi().to_bits(), value.lo().to_bits())
    }

    #[test]
    fn the_exact_value_decodes_converts_and_compares() {
        let value = gcc(ONE, 0x3C30_0000_0000_0000);
        assert_eq!(
            value.decode(),
            Decoded::Finite {
                negative: false,
                exponent: -60,
                significand: {
                    let mut limbs = [0; 33];
                    limbs[0] = (1 << 60) + 1;
                    limbs
                },
            }
        );
        let single: F32 = value.convert();
        assert_eq!(single.to_bits(), 0x3F80_0000);
        let (up, flags) = value.convert_with::<F32>(Rounding::TowardPositive);
        assert_eq!(
            (up.to_bits(), flags),
            (0x3F80_0001, Flags::INEXACT | Flags::ROUNDED_UP)
        );
        assert!(value > gcc(ONE, 0));
        assert_eq!(
            gcc(ONE, 0).partial_cmp(&gcc(ONE, 0x8000_0000_0000_0000)),
            Some(Ordering::Equal)
        );
        // (1, -1) is zero, with the sign of the high half.
        let cancelled = gcc(ONE, 0xBFF0_0000_0000_0000);
        assert_eq!(
            cancelled.decode(),
            Decoded::Zero {
                negative: false,
                exponent: 0
            }
        );
        assert_eq!(bits(-cancelled), (0xBFF0_0000_0000_0000, ONE));
    }

    #[test]
    fn abs_reads_the_sign_of_the_exact_value() {
        // (-0, 1) is 1, so it keeps its halves; (+0, -1) is -1.
        let negative_zero = 0x8000_0000_0000_0000;
        assert_eq!(bits(gcc(negative_zero, ONE).abs()), (negative_zero, ONE));
        assert_eq!(
            bits(gcc(0, ONE | negative_zero).abs()),
            (negative_zero, ONE)
        );
        assert_eq!(bits(gcc(negative_zero, 0).abs()), (0, negative_zero));
    }

    #[test]
    fn the_class_follows_the_exact_value() {
        // 2^-969 is the smallest normal value, and 2^-969 - 2^-1074 is below it.
        assert_eq!(gcc(0x0360_0000_0000_0000, 0).classify(), Class::Normal);
        assert_eq!(
            gcc(0x0360_0000_0000_0000, 0x8000_0000_0000_0001).classify(),
            Class::Subnormal
        );
        // The halves of (1, -1) cancel.
        assert!(gcc(ONE, 0xBFF0_0000_0000_0000).is_zero());
        // A NaN low half makes the value a NaN, and an infinite high half hides it.
        assert!(gcc(ONE, 0x7FF0_0000_0000_0001).is_signaling_nan());
        assert!(gcc(0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000).is_infinite());
        assert!(gcc(0xFFF0_0000_0000_0000, 0x7FF8_0000_0000_0000).is_sign_negative());
    }

    #[test]
    fn a_canonical_low_half_is_at_most_half_an_ulp() {
        let half_ulp = 0x3CA0_0000_0000_0000; // 2^-53
        assert!(gcc(ONE, half_ulp).is_canonical());
        assert!(!gcc(0x3FF0_0000_0000_0001, half_ulp).is_canonical());
        assert!(!gcc(ONE, 0x3CB0_0000_0000_0000).is_canonical());
        assert!(gcc(ONE, 0x3C9F_FFFF_FFFF_FFFF).is_canonical());
        assert!(!gcc(0x7FF0_0000_0000_0000, 1).is_canonical());
        assert!(gcc(0x7FF8_0000_0000_0000, 1).is_canonical());
        assert!(!gcc(0, 1).is_canonical());
    }

    #[test]
    fn pairs_of_one_value_order_by_their_halves() {
        // (1, 0) and (2, -1) hold 1.
        let (one, twin) = (
            gcc(ONE, 0),
            gcc(0x4000_0000_0000_0000, 0xBFF0_0000_0000_0000),
        );
        assert_eq!(one.total_cmp(twin), Ordering::Less);
        assert_eq!(twin.total_cmp(one), Ordering::Greater);
        assert_eq!(bits(one.minimum(twin)), bits(one));
        assert_eq!(bits(one.maximum(twin)), bits(twin));
        // -0 orders below +0.
        let (negative, positive) = (gcc(1 << 63, 0), gcc(0, 0));
        assert_eq!(bits(positive.minimum(negative)), bits(negative));
    }
}
