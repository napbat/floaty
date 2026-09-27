//! Fixed-width integer arithmetic on `[u64; N]` limbs.
//!
//! Limb 0 holds the least significant 64 bits. A bit position counts from the
//! least significant bit of limb 0.

use core::fmt::Debug;
use core::hash::Hash;

/// An unsigned integer of `64 * N` bits, stored as little-endian limbs.
///
/// Every `[u64; N]` implements this trait.
pub trait Limbs: Copy + Eq + Hash + Debug {
    /// The value zero.
    const ZERO: Self;

    /// Returns `true` when every bit is zero.
    fn is_zero(&self) -> bool;

    /// Returns the bit at `position`.
    fn bit(&self, position: u32) -> bool;

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

    fn is_zero(&self) -> bool {
        self.iter().all(|&limb| limb == 0)
    }

    fn bit(&self, position: u32) -> bool {
        let (index, offset) = split(position);
        (self[index] >> offset) & 1 == 1
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
    use super::Limbs;

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
    fn bits_set_and_read() {
        let value = <[u64; 2]>::ZERO.with_bit(0).with_bit(127);
        assert!(value.bit(0) && value.bit(127) && !value.bit(64));
        assert_eq!(value, [1, 1 << 63]);
        assert!(!value.is_zero());
        assert!(<[u64; 2]>::ZERO.is_zero());
    }
}
