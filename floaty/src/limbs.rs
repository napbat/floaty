//! Fixed-width integer arithmetic on `[u64; N]` limbs.
//!
//! Limb 0 holds the least significant 64 bits. A bit position counts from the
//! least significant bit of limb 0.

use core::cmp::Ordering;
use core::fmt::Debug;
use core::hash::Hash;

/// An unsigned integer of `64 * N` bits, stored as little-endian limbs.
///
/// Every `[u64; N]` implements this trait.
pub trait Limbs: Copy + Eq + Hash + Debug {
    /// The value zero.
    const ZERO: Self;

    /// The width in bits.
    const BITS: u32;

    /// Returns `true` when every bit is zero.
    fn is_zero(&self) -> bool;

    /// Returns the bit at `position`, or `false` at or above [`BITS`](Self::BITS).
    fn bit(&self, position: u32) -> bool;

    /// Returns the number of bits up to and including the highest set bit,
    /// or zero for the value zero.
    fn bit_length(&self) -> u32;

    /// Returns `true` when a bit below `count` is set.
    fn any_below(&self, count: u32) -> bool;

    /// Shifts right by `count` bits. A count at or above the width gives zero.
    #[must_use]
    fn shr(self, count: u32) -> Self;

    /// Shifts left by `count` bits. The bits shifted out of the width are lost.
    #[must_use]
    fn shl(self, count: u32) -> Self;

    /// Adds one, and wraps to zero above the largest value.
    #[must_use]
    fn increment(self) -> Self;

    /// Adds `other`. The sum must fit.
    #[must_use]
    fn add(self, other: Self) -> Self;

    /// Subtracts `other`, which must not be larger.
    #[must_use]
    fn sub(self, other: Self) -> Self;

    /// Compares two values.
    fn compare(&self, other: &Self) -> Ordering;

    /// Shifts right by `count` bits and sets the lowest bit when a shifted-out
    /// bit is set. The lowest bit then marks a value that is not exact.
    #[must_use]
    fn shr_jam(self, count: u32) -> Self;

    /// Returns a copy with the bit at `position` set.
    #[must_use]
    fn with_bit(self, position: u32) -> Self;

    /// Returns the `width`-bit field that starts at bit `low`.
    ///
    /// `width` must be from 1 to 64.
    fn field(&self, low: u32, width: u32) -> u64;

    /// Returns a copy with the `width`-bit field at bit `low` set to `value`.
    ///
    /// `width` must be from 1 to 64, and `value` must fit in `width` bits.
    #[must_use]
    fn with_field(self, low: u32, width: u32, value: u64) -> Self;

    /// Returns a copy that keeps only the bits below `count`.
    #[must_use]
    fn low_bits(self, count: u32) -> Self;

    /// Returns a value with every bit below `count` set.
    fn ones(count: u32) -> Self;

    /// Converts to another limb width. The value must fit in the target.
    fn resize<Target: Limbs>(self) -> Target;

    /// Returns a copy with limb `index` set to `value`. `index` must be below
    /// the limb count.
    #[must_use]
    fn with_limb(self, index: usize, value: u64) -> Self;
}

impl<const N: usize> Limbs for [u64; N] {
    const ZERO: Self = [0; N];

    const BITS: u32 = {
        let mut bits = 0;
        let mut index = 0;
        while index < N {
            bits += 64;
            index += 1;
        }
        bits
    };

    fn is_zero(&self) -> bool {
        self.iter().all(|&limb| limb == 0)
    }

    fn bit(&self, position: u32) -> bool {
        let (index, offset) = split(position);
        self.get(index)
            .is_some_and(|limb| (limb >> offset) & 1 == 1)
    }

    fn bit_length(&self) -> u32 {
        self.iter().rposition(|&limb| limb != 0).map_or(0, |index| {
            let above = u32::try_from(index).expect("a limb index fits a u32") * 64;
            above + 64 - self[index].leading_zeros()
        })
    }

    fn any_below(&self, count: u32) -> bool {
        !self.low_bits(count).is_zero()
    }

    fn shr(self, count: u32) -> Self {
        let (limbs, offset) = split(count);
        let mut result = Self::ZERO;
        for (index, slot) in result.iter_mut().enumerate() {
            let low = self.get(index + limbs).copied().unwrap_or(0);
            let high = self.get(index + limbs + 1).copied().unwrap_or(0);
            *slot = if offset == 0 {
                low
            } else {
                (low >> offset) | (high << (64 - offset))
            };
        }
        result
    }

    fn shl(self, count: u32) -> Self {
        let (limbs, offset) = split(count);
        let mut result = Self::ZERO;
        for (index, slot) in result.iter_mut().enumerate() {
            let Some(source) = index.checked_sub(limbs) else {
                continue;
            };
            let low = source.checked_sub(1).map_or(0, |below| self[below]);
            *slot = if offset == 0 {
                self[source]
            } else {
                (self[source] << offset) | (low >> (64 - offset))
            };
        }
        result
    }

    fn increment(mut self) -> Self {
        for limb in &mut self {
            let (sum, carry) = limb.overflowing_add(1);
            *limb = sum;
            if !carry {
                break;
            }
        }
        self
    }

    fn add(mut self, other: Self) -> Self {
        let mut carry = false;
        for (limb, &addend) in self.iter_mut().zip(&other) {
            let (sum, first) = limb.overflowing_add(addend);
            let (sum, second) = sum.overflowing_add(u64::from(carry));
            *limb = sum;
            carry = first || second;
        }
        debug_assert!(!carry, "the sum fits");
        self
    }

    fn sub(mut self, other: Self) -> Self {
        let mut borrow = false;
        for (limb, &subtrahend) in self.iter_mut().zip(&other) {
            let (difference, first) = limb.overflowing_sub(subtrahend);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            *limb = difference;
            borrow = first || second;
        }
        debug_assert!(!borrow, "the subtrahend is not larger");
        self
    }

    fn compare(&self, other: &Self) -> Ordering {
        self.iter().rev().cmp(other.iter().rev())
    }

    fn shr_jam(self, count: u32) -> Self {
        let lost = self.any_below(count);
        let shifted = self.shr(count);
        if lost { shifted.with_bit(0) } else { shifted }
    }

    fn with_bit(mut self, position: u32) -> Self {
        let (index, offset) = split(position);
        self[index] |= 1 << offset;
        self
    }

    fn field(&self, low: u32, width: u32) -> u64 {
        debug_assert!((1..=64).contains(&width), "a field is 1 to 64 bits wide");
        let (index, offset) = split(low);
        let mut value = self[index] >> offset;
        if offset != 0 && offset + width > 64 {
            value |= self[index + 1] << (64 - offset);
        }
        value & mask(width)
    }

    fn with_field(mut self, low: u32, width: u32, value: u64) -> Self {
        debug_assert!((1..=64).contains(&width), "a field is 1 to 64 bits wide");
        debug_assert!(value & !mask(width) == 0, "the value fits in the field");
        let (index, offset) = split(low);
        let field_mask = mask(width);
        self[index] = (self[index] & !(field_mask << offset)) | (value << offset);
        if offset != 0 && offset + width > 64 {
            let spill = 64 - offset;
            self[index + 1] = (self[index + 1] & !(field_mask >> spill)) | (value >> spill);
        }
        self
    }

    fn low_bits(mut self, count: u32) -> Self {
        let (boundary, offset) = split(count);
        for (index, limb) in self.iter_mut().enumerate() {
            if index > boundary || (index == boundary && offset == 0) {
                *limb = 0;
            } else if index == boundary {
                *limb &= mask(offset);
            }
        }
        self
    }

    fn ones(count: u32) -> Self {
        let (boundary, offset) = split(count);
        let mut value = Self::ZERO;
        for (index, limb) in value.iter_mut().enumerate() {
            if index < boundary {
                *limb = u64::MAX;
            } else if index == boundary && offset != 0 {
                *limb = mask(offset);
            }
        }
        value
    }

    fn resize<Target: Limbs>(self) -> Target {
        self.iter()
            .enumerate()
            .fold(Target::ZERO, |target, (index, &limb)| {
                if limb == 0 {
                    target
                } else {
                    target.with_limb(index, limb)
                }
            })
    }

    fn with_limb(mut self, index: usize, value: u64) -> Self {
        self[index] = value;
        self
    }
}

/// A limb array with a double-width type for products.
///
/// Stable Rust cannot name `[u64; 2 * N]` for a generic `N`, so this table
/// names the double width of each limb count that a format uses.
pub trait Widen: Limbs {
    /// The limb array of twice the width.
    type Double: Limbs;

    /// Returns the full product.
    fn widening_mul(self, other: Self) -> Self::Double;
}

macro_rules! widen {
    ($($limbs:literal => $double:literal),*) => {
        $(
            impl Widen for [u64; $limbs] {
                type Double = [u64; $double];

                fn widening_mul(self, other: Self) -> [u64; $double] {
                    multiply(&self, &other)
                }
            }
        )*
    };
}

widen!(1 => 2, 2 => 4, 3 => 6, 4 => 8, 5 => 10, 6 => 12, 7 => 14, 8 => 16);

/// Returns the full product of two limb arrays. `D` must be twice `N`.
fn multiply<const N: usize, const D: usize>(left: &[u64; N], right: &[u64; N]) -> [u64; D] {
    debug_assert!(D == 2 * N, "the product has twice the limbs");
    let mut product = [0_u64; D];
    // The schoolbook product adds each partial product into two limbs, so the
    // loops index the result.
    for (index, &multiplicand) in left.iter().enumerate() {
        let mut carry = 0_u64;
        for (offset, &multiplier) in right.iter().enumerate() {
            let slot = &mut product[index + offset];
            let wide = u128::from(multiplicand) * u128::from(multiplier)
                + u128::from(*slot)
                + u128::from(carry);
            let [low, high] = split_u128(wide);
            *slot = low;
            carry = high;
        }
        product[index + N] = carry;
    }
    product
}

/// Splits a `u128` into its low and high 64 bits.
#[inline]
fn split_u128(value: u128) -> [u64; 2] {
    let low = u64::try_from(value & u128::from(u64::MAX)).expect("the mask keeps 64 bits");
    let high = u64::try_from(value >> 64).expect("the shift keeps 64 bits");
    [low, high]
}

/// Divides `numerator` by a nonzero `divisor`. Returns the quotient and the
/// remainder.
pub fn divide<L: Limbs>(numerator: L, divisor: L) -> (L, L) {
    debug_assert!(!divisor.is_zero(), "the divisor is not zero");
    let mut quotient = L::ZERO;
    let mut remainder = L::ZERO;
    for position in (0..numerator.bit_length()).rev() {
        remainder = remainder.shl(1);
        if numerator.bit(position) {
            remainder = remainder.with_bit(0);
        }
        if remainder.compare(&divisor) != Ordering::Less {
            remainder = remainder.sub(divisor);
            quotient = quotient.with_bit(position);
        }
    }
    (quotient, remainder)
}

/// Returns the integer square root of `value`, rounded down, and `true` when
/// the root is not exact.
pub fn square_root<L: Limbs>(value: L) -> (L, bool) {
    let mut remainder = value;
    let mut root = L::ZERO;
    let length = value.bit_length();
    if length == 0 {
        return (root, false);
    }
    // The highest power of four at or below the value.
    let mut bit = L::ZERO.with_bit((length - 1) & !1);
    while !bit.is_zero() {
        let trial = root.add(bit);
        root = root.shr(1);
        if remainder.compare(&trial) != Ordering::Less {
            remainder = remainder.sub(trial);
            root = root.add(bit);
        }
        bit = bit.shr(2);
    }
    (root, !remainder.is_zero())
}

/// Splits a bit position into a limb index and a bit offset in that limb.
#[inline]
fn split(position: u32) -> (usize, u32) {
    let index = usize::try_from(position / 64).expect("a limb index fits a usize");
    (index, position % 64)
}

/// Returns a mask of the low `width` bits. `width` must be from 1 to 64.
#[inline]
fn mask(width: u32) -> u64 {
    u64::MAX >> (64 - width)
}

#[cfg(test)]
mod tests {
    use super::{Limbs, Widen};

    #[test]
    fn field_reads_across_a_limb_boundary() {
        let value = [0xF000_0000_0000_0000, 0x0000_0000_0000_000A];
        assert_eq!(value.field(60, 8), 0xAF);
        assert_eq!(value.field(0, 64), 0xF000_0000_0000_0000);
        assert_eq!(value.field(64, 4), 0xA);
    }

    #[test]
    fn with_field_writes_across_a_limb_boundary() {
        let value = [u64::MAX, u64::MAX].with_field(60, 8, 0x5A);
        assert_eq!(value, [0xAFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFF5]);
        assert_eq!(value.field(60, 8), 0x5A);
        assert_eq!([0u64; 1].with_field(0, 64, u64::MAX), [u64::MAX]);
    }

    #[test]
    fn low_bits_and_ones_agree() {
        for count in 0..=192 {
            let all: [u64; 3] = [u64::MAX; 3];
            assert_eq!(
                all.low_bits(count),
                <[u64; 3]>::ones(count),
                "count {count}"
            );
        }
        assert_eq!(<[u64; 2]>::ones(64), [u64::MAX, 0]);
        assert_eq!([u64::MAX].low_bits(0), [0]);
    }

    #[test]
    fn resize_keeps_the_value() {
        let narrow: [u64; 2] = [7, 9];
        let wide: [u64; 4] = narrow.resize();
        assert_eq!(wide, [7, 9, 0, 0]);
        assert_eq!(wide.resize::<[u64; 2]>(), narrow);
    }

    #[test]
    fn shifts_move_bits_across_limbs() {
        let value: [u64; 3] = [0x8000_0000_0000_0001, 0x1, 0];
        assert_eq!(value.shl(1), [0x2, 0x3, 0]);
        assert_eq!(value.shl(64), [0, 0x8000_0000_0000_0001, 0x1]);
        assert_eq!(value.shl(127), [0, 1 << 63, 0xC000_0000_0000_0000]);
        assert_eq!(value.shr(1), [0xC000_0000_0000_0000, 0, 0]);
        assert_eq!(value.shr(63), [0x3, 0, 0]);
        assert_eq!(value.shr(64), [0x1, 0, 0]);
        assert_eq!(value.shr(192), [0; 3]);
        assert_eq!(value.shr(1000), [0; 3]);
        assert_eq!(value.shl(192), [0; 3]);
    }

    #[test]
    fn bit_length_and_any_below() {
        assert_eq!([0_u64; 2].bit_length(), 0);
        assert_eq!([1_u64, 0].bit_length(), 1);
        assert_eq!([0, 1_u64 << 63].bit_length(), 128);
        let value: [u64; 2] = [0x10, 0];
        assert!(!value.any_below(4) && value.any_below(5) && value.any_below(500));
        assert!(!value.bit(128) && !value.bit(4000));
        assert_eq!(<[u64; 5]>::BITS, 320);
    }

    #[test]
    fn add_sub_and_compare_carry_across_limbs() {
        let a: [u64; 2] = [u64::MAX, 1];
        let b: [u64; 2] = [1, 0];
        assert_eq!(a.add(b), [0, 2]);
        assert_eq!([0, 2].sub(b), a);
        assert_eq!(a.compare(&b), core::cmp::Ordering::Greater);
        assert_eq!(b.compare(&a), core::cmp::Ordering::Less);
        assert_eq!(a.compare(&a), core::cmp::Ordering::Equal);
        assert_eq!([0b1011_u64].shr_jam(2), [0b11]);
        assert_eq!([0b1000_u64].shr_jam(2), [0b10]);
        assert_eq!([1_u64, 0].shr_jam(200), [1, 0]);
    }

    #[test]
    fn widening_multiply_divide_and_square_root() {
        let product = [u64::MAX, u64::MAX].widening_mul([u64::MAX, u64::MAX]);
        // (2^128 - 1)^2 = 2^256 - 2^129 + 1.
        assert_eq!(product, [1, 0, u64::MAX - 1, u64::MAX]);
        let (quotient, remainder) = super::divide([100_u64, 0], [7, 0]);
        assert_eq!((quotient, remainder), ([14, 0], [2, 0]));
        let (quotient, remainder) = super::divide(product, [u64::MAX, u64::MAX, 0, 0]);
        assert_eq!((quotient, remainder), ([u64::MAX, u64::MAX, 0, 0], [0; 4]));
        assert_eq!(super::square_root([144_u64]), ([12], false));
        assert_eq!(super::square_root([145_u64]), ([12], true));
        assert_eq!(super::square_root([0_u64, 1]), ([1 << 32, 0], false));
        assert_eq!(
            super::square_root(product),
            ([u64::MAX, u64::MAX, 0, 0], false)
        );
    }

    #[test]
    fn increment_carries() {
        assert_eq!([u64::MAX, 0].increment(), [0, 1]);
        assert_eq!([u64::MAX, u64::MAX].increment(), [0, 0]);
        assert_eq!([5_u64].increment(), [6]);
    }

    #[test]
    fn bits_set_and_read() {
        let value = <[u64; 2]>::ZERO.with_bit(0).with_bit(127);
        assert!(value.bit(0) && value.bit(127) && !value.bit(64));
        assert_eq!(value, [1, 1 << 63]);
        assert!(!value.is_zero());
        assert!(<[u64; 2]>::ZERO.is_zero());
    }
}
