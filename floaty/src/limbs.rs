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

    /// Returns limb `index`, or zero at or above the limb count.
    fn limb(&self, index: usize) -> u64;
}

/// Returns the limb of a one-limb array.
#[inline]
fn narrow_to_u64<const N: usize>(limbs: &[u64; N]) -> u64 {
    debug_assert!(N == 1, "the array has one limb");
    limbs.first().copied().unwrap_or(0)
}

/// Returns a `u64` as a one-limb array.
#[inline]
fn narrow_from_u64<const N: usize>(value: u64) -> [u64; N] {
    debug_assert!(N == 1, "the array has one limb");
    let mut limbs = [0; N];
    if let Some(slot) = limbs.first_mut() {
        *slot = value;
    }
    limbs
}

/// Returns an array of at most two limbs as a `u128`.
#[inline]
fn narrow_to_u128<const N: usize>(limbs: &[u64; N]) -> u128 {
    debug_assert!(N <= 2, "the array has at most two limbs");
    let low = limbs.first().copied().unwrap_or(0);
    let high = limbs.get(1).copied().unwrap_or(0);
    u128::from(low) | (u128::from(high) << 64)
}

/// Returns the low `64 * N` bits of a `u128` as an array of at most two limbs.
#[inline]
fn narrow_from_u128<const N: usize>(value: u128) -> [u64; N] {
    debug_assert!(N <= 2, "the array has at most two limbs");
    let mut limbs = [0; N];
    for (slot, limb) in limbs.iter_mut().zip(split_u128(value)) {
        *slot = limb;
    }
    limbs
}

/// An array of one limb computes on a native `u64`, and an array of two limbs
/// on a native `u128`. The native types give the same values as the limb
/// loops, faster. `N` is a constant, so the compiler keeps one code path.
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

    #[inline]
    fn bit_length(&self) -> u32 {
        if N == 1 {
            return 64 - narrow_to_u64(self).leading_zeros();
        }
        if N <= 2 {
            return 128 - narrow_to_u128(self).leading_zeros();
        }
        self.iter().rposition(|&limb| limb != 0).map_or(0, |index| {
            let above = u32::try_from(index).expect("a limb index fits a u32") * 64;
            above + 64 - self[index].leading_zeros()
        })
    }

    #[inline]
    fn any_below(&self, count: u32) -> bool {
        if N == 1 {
            let value = narrow_to_u64(self);
            if count >= 64 {
                return value != 0;
            }
            return value & ((1 << count) - 1) != 0;
        }
        if N <= 2 {
            let value = narrow_to_u128(self);
            if count >= 128 {
                return value != 0;
            }
            return value & ((1 << count) - 1) != 0;
        }
        !self.low_bits(count).is_zero()
    }

    #[inline]
    fn shr(self, count: u32) -> Self {
        if N == 1 {
            return narrow_from_u64(narrow_to_u64(&self).checked_shr(count).unwrap_or(0));
        }
        if N <= 2 {
            return narrow_from_u128(narrow_to_u128(&self).checked_shr(count).unwrap_or(0));
        }
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

    #[inline]
    fn shl(self, count: u32) -> Self {
        if N == 1 {
            return narrow_from_u64(narrow_to_u64(&self).checked_shl(count).unwrap_or(0));
        }
        if N <= 2 {
            return narrow_from_u128(narrow_to_u128(&self).checked_shl(count).unwrap_or(0));
        }
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

    #[inline]
    fn increment(mut self) -> Self {
        if N == 1 {
            return narrow_from_u64(narrow_to_u64(&self).wrapping_add(1));
        }
        if N <= 2 {
            return narrow_from_u128(narrow_to_u128(&self).wrapping_add(1));
        }
        for limb in &mut self {
            let (sum, carry) = limb.overflowing_add(1);
            *limb = sum;
            if !carry {
                break;
            }
        }
        self
    }

    #[inline]
    fn add(mut self, other: Self) -> Self {
        if N == 1 {
            return narrow_from_u64(narrow_to_u64(&self) + narrow_to_u64(&other));
        }
        if N <= 2 {
            let sum = narrow_to_u128(&self) + narrow_to_u128(&other);
            debug_assert!(N == 2 || sum >> 64 == 0, "the sum fits");
            return narrow_from_u128(sum);
        }
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

    #[inline]
    fn sub(mut self, other: Self) -> Self {
        if N == 1 {
            return narrow_from_u64(narrow_to_u64(&self) - narrow_to_u64(&other));
        }
        if N <= 2 {
            return narrow_from_u128(narrow_to_u128(&self) - narrow_to_u128(&other));
        }
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

    #[inline]
    fn compare(&self, other: &Self) -> Ordering {
        if N == 1 {
            return narrow_to_u64(self).cmp(&narrow_to_u64(other));
        }
        if N <= 2 {
            return narrow_to_u128(self).cmp(&narrow_to_u128(other));
        }
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

    #[inline]
    fn low_bits(mut self, count: u32) -> Self {
        if N == 1 {
            let value = narrow_to_u64(&self);
            let kept = if count >= 64 {
                value
            } else {
                value & ((1 << count) - 1)
            };
            return narrow_from_u64(kept);
        }
        if N <= 2 {
            let value = narrow_to_u128(&self);
            let kept = if count >= 128 {
                value
            } else {
                value & ((1 << count) - 1)
            };
            return narrow_from_u128(kept);
        }
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

    fn limb(&self, index: usize) -> u64 {
        self.get(index).copied().unwrap_or(0)
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

                #[inline]
                fn widening_mul(self, other: Self) -> [u64; $double] {
                    multiply(&self, &other)
                }
            }
        )*
    };
}

widen!(1 => 2, 2 => 4, 3 => 6, 4 => 8, 5 => 10, 6 => 12, 7 => 14, 8 => 16);

/// Returns the full product of two limb arrays. `D` must be twice `N`.
#[inline]
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
pub fn split_u128(value: u128) -> [u64; 2] {
    let low = u64::try_from(value & u128::from(u64::MAX)).expect("the mask keeps 64 bits");
    let high = u64::try_from(value >> 64).expect("the shift keeps 64 bits");
    [low, high]
}

/// The buffer of `divide` for values of up to 1,024 bits: the double width of
/// the widest storage type, and one limb more.
const SMALL_BUFFER: usize = 17;

/// The buffer of `divide` for the wide values of a conversion between radixes:
/// up to 16,384 bits, and one limb more.
const LARGE_BUFFER: usize = 257;

/// Returns the low 64 bits of a `u128`.
#[inline]
fn low_u64(value: u128) -> u64 {
    split_u128(value)[0]
}

/// Returns a value of at most 128 bits as a `u128`.
#[inline]
pub fn to_u128<L: Limbs>(value: &L) -> u128 {
    debug_assert!(value.bit_length() <= 128, "the value fits 128 bits");
    u128::from(value.limb(0)) | (u128::from(value.limb(1)) << 64)
}

/// Returns a `u128` as limbs. The value must fit `L`.
#[inline]
pub fn from_u128<L: Limbs>(value: u128) -> L {
    let [low, high] = split_u128(value);
    let low_limb = L::ZERO.with_limb(0, low);
    if high == 0 {
        low_limb
    } else {
        low_limb.with_limb(1, high)
    }
}

/// Returns `value * factor`. The product must fit `L`.
pub fn multiply_small<L: Limbs>(value: L, factor: u64) -> L {
    let count = limb_count(&value);
    let mut product = L::ZERO;
    let mut carry = 0_u64;
    // The carry passes from each limb to the next, so the loop indexes.
    for index in 0..count {
        let [low, high] =
            split_u128(u128::from(value.limb(index)) * u128::from(factor) + u128::from(carry));
        product = product.with_limb(index, low);
        carry = high;
    }
    if carry != 0 {
        product = product.with_limb(count, carry);
    }
    product
}

/// Returns `left * right`. The product must fit `L`.
///
/// Values of at most 128 bits use the native `u128` product. A wider value
/// adds one partial product for each limb of `right`. Each partial product is
/// at most the whole product, so each one fits too.
pub fn multiply_fit<L: Limbs>(left: L, right: L) -> L {
    if L::BITS <= 128 {
        let product = to_u128(&left) * to_u128(&right);
        return from_u128(product);
    }
    (0..limb_count(&right)).fold(L::ZERO, |product, index| {
        let shift = u32::try_from(index * 64).expect("a limb offset fits a u32");
        let partial = multiply_small(left, right.limb(index));
        debug_assert!(partial.bit_length() + shift <= L::BITS, "the product fits");
        product.add(partial.shl(shift))
    })
}

/// Returns the number of limbs up to the highest nonzero limb.
#[inline]
fn limb_count<L: Limbs>(value: &L) -> usize {
    usize::try_from(value.bit_length().div_ceil(64)).expect("a limb count fits a usize")
}

/// Divides `value` by a nonzero `divisor` of one limb. Returns the quotient
/// and the remainder. The division needs no buffer, so it takes a value of
/// any width.
pub fn divide_small<L: Limbs>(value: L, divisor: u64) -> (L, u64) {
    debug_assert!(divisor != 0, "the divisor is not zero");
    let by = u128::from(divisor);
    let mut quotient = L::ZERO;
    let mut rest = 0_u128;
    // The remainder passes from each limb to the next lower one, so the loop
    // indexes.
    for index in (0..limb_count(&value)).rev() {
        let current = (rest << 64) | u128::from(value.limb(index));
        quotient = quotient.with_limb(index, low_u64(current / by));
        rest = current % by;
    }
    (quotient, low_u64(rest))
}

/// Divides `numerator` by a nonzero `divisor`. Returns the quotient and the
/// remainder.
///
/// Values of at most 64 bits use the native `u64` division, and values of at
/// most 128 bits the native `u128` division. Wider values use Knuth's
/// Algorithm D, one quotient limb per step.
pub fn divide<L: Limbs>(numerator: L, divisor: L) -> (L, L) {
    debug_assert!(!divisor.is_zero(), "the divisor is not zero");
    if numerator.bit_length() <= 64 {
        if divisor.bit_length() > 64 {
            // The divisor is larger than the numerator.
            return (L::ZERO, numerator);
        }
        // The native 64-bit division is faster than the 128-bit one.
        let (wide, by) = (numerator.limb(0), divisor.limb(0));
        return (
            L::ZERO.with_limb(0, wide / by),
            L::ZERO.with_limb(0, wide % by),
        );
    }
    if numerator.bit_length() <= 128 {
        if divisor.bit_length() > 128 {
            // The divisor is larger than the numerator.
            return (L::ZERO, numerator);
        }
        let (wide, by) = (to_u128(&numerator), to_u128(&divisor));
        return (from_u128(wide / by), from_u128(wide % by));
    }
    if L::BITS <= 1024 {
        long_divide::<L, SMALL_BUFFER>(numerator, divisor)
    } else {
        long_divide::<L, LARGE_BUFFER>(numerator, divisor)
    }
}

/// Divides by Knuth's Algorithm D (The Art of Computer Programming, Volume 2,
/// section 4.3.1). The divisor is normalized so that its top limb has its
/// high bit set; each quotient limb estimate is then at most two too large.
///
/// `BUFFER` holds the numerator and one limb more.
fn long_divide<L: Limbs, const BUFFER: usize>(numerator: L, divisor: L) -> (L, L) {
    let length = limb_count(&divisor);
    let total = limb_count(&numerator);
    assert!(
        total < BUFFER,
        "the buffer holds the numerator and one limb more"
    );
    if total < length {
        return (L::ZERO, numerator);
    }
    if length == 1 {
        // One 128-by-64-bit division for each limb.
        let by = u128::from(divisor.limb(0));
        let mut quotient = L::ZERO;
        let mut rest = 0_u128;
        for index in (0..total).rev() {
            let current = (rest << 64) | u128::from(numerator.limb(index));
            quotient = quotient.with_limb(index, low_u64(current / by));
            rest = current % by;
        }
        return (quotient, from_u128(rest));
    }
    let shift = divisor.limb(length - 1).leading_zeros();
    let shifted = |value: &L, index: usize| {
        let low = if shift == 0 || index == 0 {
            0
        } else {
            value.limb(index - 1) >> (64 - shift)
        };
        (value.limb(index) << shift) | low
    };
    let mut by = [0_u64; BUFFER];
    for (index, limb) in by.iter_mut().enumerate().take(length) {
        *limb = shifted(&divisor, index);
    }
    let mut rest = [0_u64; BUFFER];
    for (index, limb) in rest.iter_mut().enumerate().take(total + 1) {
        *limb = shifted(&numerator, index);
    }
    let (top, next) = (u128::from(by[length - 1]), u128::from(by[length - 2]));
    let mut quotient = L::ZERO;
    // Each step divides the top limbs of the rest and subtracts the estimate
    // times the divisor. The loops carry between limbs, so they index.
    for step in (0..=total - length).rev() {
        let leading = (u128::from(rest[step + length]) << 64) | u128::from(rest[step + length - 1]);
        let mut estimate = leading / top;
        let mut remainder = leading % top;
        while estimate >> 64 != 0
            || estimate * next > ((remainder << 64) | u128::from(rest[step + length - 2]))
        {
            estimate -= 1;
            remainder += top;
            if remainder >> 64 != 0 {
                break;
            }
        }
        let mut carry = 0_u64;
        let mut borrow = false;
        for index in 0..length {
            let [low, high] = split_u128(estimate * u128::from(by[index]) + u128::from(carry));
            carry = high;
            let (difference, first) = rest[step + index].overflowing_sub(low);
            let (difference, second) = difference.overflowing_sub(u64::from(borrow));
            rest[step + index] = difference;
            borrow = first || second;
        }
        let (difference, first) = rest[step + length].overflowing_sub(carry);
        let (difference, second) = difference.overflowing_sub(u64::from(borrow));
        rest[step + length] = difference;
        let mut digit = low_u64(estimate);
        if first || second {
            // The estimate was one too large: add the divisor back.
            digit -= 1;
            let mut carry = false;
            for index in 0..length {
                let (sum, first) = rest[step + index].overflowing_add(by[index]);
                let (sum, second) = sum.overflowing_add(u64::from(carry));
                rest[step + index] = sum;
                carry = first || second;
            }
            rest[step + length] = rest[step + length].wrapping_add(u64::from(carry));
        }
        quotient = quotient.with_limb(step, digit);
    }
    let normalized = rest
        .iter()
        .take(length)
        .enumerate()
        .fold(L::ZERO, |value, (index, &limb)| {
            value.with_limb(index, limb)
        });
    (quotient, normalized.shr(shift))
}

/// Returns the integer square root of `value`, rounded down, and `true` when
/// the root is not exact.
///
/// Values of at most 64 bits use the native `u64` square root, and values of
/// at most 128 bits the native `u128` square root. For a wider value,
/// Newton's iteration starts above the root, from the root of the top 128
/// bits, and decreases to it.
pub fn square_root<L: Limbs>(value: L) -> (L, bool) {
    let length = value.bit_length();
    if length <= 64 {
        let narrow = value.limb(0);
        let root = narrow.isqrt();
        return (L::ZERO.with_limb(0, root), root * root != narrow);
    }
    if length <= 128 {
        let wide = to_u128(&value);
        let root = wide.isqrt();
        return (from_u128(root), root * root != wide);
    }
    // An even shift keeps the root of the shifted value at half the shift.
    let shift = (length - 127) & !1;
    let top = to_u128(&value.shr(shift));
    let mut root = from_u128::<L>(top.isqrt() + 1).shl(shift / 2);
    loop {
        let (quotient, rest) = divide(value, root);
        let next = root.add(quotient).shr(1);
        if next.compare(&root) != Ordering::Less {
            // The root is exact when the value divides into it evenly.
            return (root, quotient != root || !rest.is_zero());
        }
        root = next;
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
    fn one_limb_operations_take_every_count() {
        let value: [u64; 1] = [0x8000_0000_0000_0001];
        assert_eq!(value.shl(0), value);
        assert_eq!(value.shl(63), [1 << 63]);
        assert_eq!(value.shl(64), [0]);
        assert_eq!(value.shl(1000), [0]);
        assert_eq!(value.shr(63), [1]);
        assert_eq!(value.shr(64), [0]);
        assert_eq!(value.shr(1000), [0]);
        assert!(!value.any_below(0) && value.any_below(1) && value.any_below(64));
        assert!(!<[u64; 1]>::ZERO.any_below(1000) && value.any_below(1000));
        assert_eq!(value.low_bits(1), [1]);
        assert_eq!(value.low_bits(64), value);
        assert_eq!(value.low_bits(100), value);
        assert_eq!((value.bit_length(), [0_u64].bit_length()), (64, 0));
        assert_eq!([u64::MAX].increment(), [0]);
        assert_eq!([3_u64].add([4]), [7]);
        assert_eq!([7_u64].sub([4]), [3]);
        assert_eq!([3_u64].compare(&[4]), core::cmp::Ordering::Less);
        // Division and the square root of values below 2^64 on a wider
        // array.
        assert_eq!(
            super::divide([100_u64, 0], [1 << 40, 0]),
            ([0, 0], [100, 0])
        );
        assert_eq!(
            super::divide([u64::MAX, 0], [0, 1]),
            ([0, 0], [u64::MAX, 0])
        );
        assert_eq!(
            super::square_root([u64::MAX, 0]),
            ([u64::from(u32::MAX), 0], true)
        );
    }

    #[test]
    fn a_product_that_fits_needs_no_wider_type() {
        assert_eq!(super::multiply_fit([6_u64], [7]), [42]);
        assert_eq!(
            super::multiply_fit([u64::MAX, 0], [u64::MAX, 0]),
            [1, u64::MAX - 1]
        );
        // (2^128 - 1) * (2^64 + 3) across four limbs.
        let product = super::multiply_fit([u64::MAX, u64::MAX, 0, 0], [3, 1, 0, 0]);
        assert_eq!(product, [u64::MAX - 2, u64::MAX - 1, 2, 1]);
        assert_eq!(
            product,
            [u64::MAX, u64::MAX, 0, 0]
                .widening_mul([3, 1, 0, 0])
                .resize::<[u64; 4]>()
        );
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
    fn long_division_matches_the_definition() {
        // (2^128 - 1)^2 + 5 divided by 2^128 - 1. The top limb of the divisor
        // needs no normalization.
        let product = [u64::MAX, u64::MAX].widening_mul([u64::MAX, u64::MAX]);
        let with_rest = product.add([5, 0, 0, 0]);
        assert_eq!(
            super::divide(with_rest, [u64::MAX, u64::MAX, 0, 0]),
            ([u64::MAX, u64::MAX, 0, 0], [5, 0, 0, 0])
        );
        // A quotient limb whose estimate is too large exercises the add-back
        // step: 2^192 / (2^128 + 1).
        let (quotient, remainder) = super::divide([0, 0, 0, 1], [1, 0, 1, 0]);
        // 2^192 = (2^64 - 1)(2^128 + 1) + (2^128 - 2^64 + 1).
        assert_eq!(quotient, [u64::MAX, 0, 0, 0]);
        assert_eq!(remainder, [1, u64::MAX, 0, 0]);
        // One-limb divisor of a wide numerator.
        assert_eq!(
            super::divide([7, 0, 0, 1], [2, 0, 0, 0]),
            ([3, 0, 1 << 63, 0], [1, 0, 0, 0])
        );
        // A wide square root by Newton's iteration.
        let (root, inexact) = super::square_root(product);
        assert_eq!((root, inexact), ([u64::MAX, u64::MAX, 0, 0], false));
        let (root, inexact) = super::square_root(with_rest);
        assert_eq!((root, inexact), ([u64::MAX, u64::MAX, 0, 0], true));
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
