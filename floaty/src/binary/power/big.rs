//! A nonnegative integer of up to 32,768 bits, for the exact powers of
//! `pown` and `rootn`.
//!
//! The exact powers have at most 64 times the bits of a significand and a
//! few bits more: 31,977 bits for binary512. The integer keeps its limbs in a
//! fixed array, and counts the limbs in use, so an operation costs what its
//! operands need.

use core::cmp::Ordering;

use crate::limbs::{Limbs, split_u128};

/// The limb count of the array.
const CAPACITY: usize = 512;

/// The width of the array in bits.
const CAPACITY_BITS: u32 = 64 * 512;

/// A nonnegative integer. Limb 0 holds the least significant 64 bits, and
/// the limbs from `length` up are zero.
#[derive(Clone)]
pub(super) struct Big {
    limbs: [u64; CAPACITY],
    length: usize,
}

impl Big {
    /// Returns the value of a limb array.
    pub(super) fn of<L: Limbs>(value: &L) -> Self {
        let mut big = Self::zero();
        let count = usize::try_from(L::BITS / 64).expect("a limb count fits a usize");
        for (index, slot) in big.limbs.iter_mut().take(count).enumerate() {
            *slot = value.limb(index);
        }
        big.length = count;
        big.trim();
        big
    }

    /// Returns `2^exponent`.
    pub(super) fn power_of_two(exponent: u32) -> Self {
        let index = usize::try_from(exponent / 64).expect("an index fits a usize");
        assert!(index < CAPACITY, "the exact powers fit the capacity");
        let mut big = Self::zero();
        big.limbs[index] = 1 << (exponent % 64);
        big.length = index + 1;
        big
    }

    fn zero() -> Self {
        Self {
            limbs: [0; CAPACITY],
            length: 0,
        }
    }

    /// Drops the zero limbs at the top from the count.
    fn trim(&mut self) {
        while self.length > 0 && self.limbs[self.length - 1] == 0 {
            self.length -= 1;
        }
    }

    /// Returns the number of bits up to the highest set bit.
    pub(super) fn bit_length(&self) -> u32 {
        match self.length.checked_sub(1) {
            None => 0,
            Some(top) => {
                let top_bits = 64 - self.limbs[top].leading_zeros();
                u32::try_from(top).expect("an index fits a u32") * 64 + top_bits
            }
        }
    }

    /// Returns `true` when a bit below `count` is set.
    pub(super) fn any_below(&self, count: u32) -> bool {
        let whole = usize::try_from(count / 64).expect("an index fits a usize");
        let partial = count % 64;
        self.limbs[..whole.min(self.length)]
            .iter()
            .any(|&limb| limb != 0)
            || (partial != 0
                && whole < self.length
                && self.limbs[whole] & ((1 << partial) - 1) != 0)
    }

    /// Returns `self >> count` as a limb array. The shifted value must fit.
    pub(super) fn shr_to<L: Limbs>(&self, count: u32) -> L {
        debug_assert!(
            self.bit_length().saturating_sub(count) <= L::BITS,
            "the shifted value fits the limb array"
        );
        let whole = usize::try_from(count / 64).expect("an index fits a usize");
        let partial = count % 64;
        let limb = |index: usize| self.limbs.get(index).copied().unwrap_or(0);
        let count = usize::try_from(L::BITS / 64).expect("a limb count fits a usize");
        (0..count).fold(L::ZERO, |value, index| {
            let low = limb(whole + index) >> partial;
            let high = if partial == 0 {
                0
            } else {
                limb(whole + index + 1) << (64 - partial)
            };
            value.with_limb(index, low | high)
        })
    }

    /// Returns `self << count`.
    pub(super) fn shl(&self, count: u32) -> Self {
        if self.length == 0 {
            return Self::zero();
        }
        assert!(
            self.bit_length() + count <= CAPACITY_BITS,
            "the exact powers fit the capacity"
        );
        let whole = usize::try_from(count / 64).expect("an index fits a usize");
        let partial = count % 64;
        let length = (self.length + whole + 1).min(CAPACITY);
        let mut shifted = Self::zero();
        // Each limb takes bits from two source limbs, so the loop indexes.
        for index in 0..self.length {
            let limb = self.limbs[index];
            shifted.limbs[index + whole] |= limb << partial;
            if partial != 0 && index + whole + 1 < CAPACITY {
                shifted.limbs[index + whole + 1] |= limb >> (64 - partial);
            }
        }
        shifted.length = length;
        shifted.trim();
        shifted
    }

    /// Returns the product.
    pub(super) fn mul(&self, other: &Self) -> Self {
        let mut product = Self::zero();
        let length = self.length + other.length;
        assert!(length <= CAPACITY, "the exact powers fit the capacity");
        // The schoolbook product adds each partial product into two limbs,
        // so the loops index the result.
        for (index, &multiplicand) in self.limbs[..self.length].iter().enumerate() {
            let mut carry = 0_u64;
            for (offset, &multiplier) in other.limbs[..other.length].iter().enumerate() {
                let slot = &mut product.limbs[index + offset];
                let wide = u128::from(multiplicand) * u128::from(multiplier)
                    + u128::from(*slot)
                    + u128::from(carry);
                let [low, high] = split_u128(wide);
                *slot = low;
                carry = high;
            }
            product.limbs[index + other.length] = carry;
        }
        product.length = length;
        product.trim();
        product
    }

    /// Returns `self^exponent`.
    pub(super) fn pow(&self, exponent: u32) -> Self {
        let mut result = Self::power_of_two(0);
        for bit in (0..u32::BITS - exponent.leading_zeros()).rev() {
            result = result.mul(&result);
            if (exponent >> bit) & 1 == 1 {
                result = result.mul(self);
            }
        }
        result
    }

    /// Compares two values.
    pub(super) fn compare(&self, other: &Self) -> Ordering {
        self.length.cmp(&other.length).then_with(|| {
            self.limbs[..self.length]
                .iter()
                .rev()
                .cmp(other.limbs[..other.length].iter().rev())
        })
    }
}
