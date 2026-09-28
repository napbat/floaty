//! Exact arithmetic for the conversions between the binary and the decimal
//! formats.
//!
//! A conversion reduces a value to a few digits more than the precision by a
//! power of 5 and a shift. 10^6144, the scale of decimal128, is about
//! 2^20410, and the reduced numbers stay below 16,384 bits: 5^6216 has 14,434
//! bits. A conversion whose numbers fit 1,024 bits computes in that width.
//!
//! The functions work in place on `[u64; N]`, because the by-value limb
//! operations copy the whole array for each limb.

use crate::limbs::{self, Limbs};

/// The limb count of a conversion whose numbers fit 1,024 bits.
pub const SMALL: usize = 16;

/// The limb count that holds the numbers of every conversion: 16,384 bits.
pub const LARGE: usize = 256;

/// The largest power of 5 in a `u64`: 5^27.
const FIVE_STEP: u64 = 7_450_580_596_923_828_125;

/// Returns `true` when numbers of `bits` bits fit the small width.
pub fn fits_small(bits: i64) -> bool {
    bits <= i64::from(<[u64; SMALL]>::BITS)
}

/// Returns the number of limbs up to the highest nonzero limb.
fn used<const N: usize>(value: &[u64; N]) -> usize {
    value
        .iter()
        .rposition(|&limb| limb != 0)
        .map_or(0, |top| top + 1)
}

/// Multiplies `value` by `factor` in place. The product must fit.
fn scale<const N: usize>(value: &mut [u64; N], factor: u64) {
    let used = used(value);
    let mut carry = 0_u64;
    for limb in value.iter_mut().take(used) {
        let [low, high] =
            limbs::split_u128(u128::from(*limb) * u128::from(factor) + u128::from(carry));
        *limb = low;
        carry = high;
    }
    if carry != 0 {
        *value.get_mut(used).expect("the product fits the width") = carry;
    }
}

/// Returns 5^`exponent`. The power must fit `N` limbs.
pub fn power_of_five<const N: usize>(exponent: u32) -> [u64; N] {
    let mut power: [u64; N] = limbs::from_u128(1);
    for _ in 0..exponent / 27 {
        scale(&mut power, FIVE_STEP);
    }
    scale(&mut power, 5_u64.pow(exponent % 27));
    power
}

/// Returns `value * factor`. The product must fit `N` limbs.
pub fn multiply<const N: usize, L: Limbs>(value: &[u64; N], factor: &L) -> [u64; N] {
    let count =
        usize::try_from(factor.bit_length().div_ceil(64)).expect("a limb count fits a usize");
    let length = used(value);
    let mut product = [0_u64; N];
    // Row `index` writes the limbs from `index` to `index + length`, so the
    // carry lands on a limb that no earlier row wrote.
    for index in 0..count {
        let digit = factor.limb(index);
        let mut carry = 0_u64;
        let row = product
            .iter_mut()
            .skip(index)
            .zip(value.iter().take(length));
        for (slot, &limb) in row {
            let [low, high] = limbs::split_u128(
                u128::from(limb) * u128::from(digit) + u128::from(*slot) + u128::from(carry),
            );
            *slot = low;
            carry = high;
        }
        if carry != 0 {
            *product
                .get_mut(index + length)
                .expect("the product fits the width") = carry;
        }
    }
    product
}

/// Shifts `value` left by `shift` bits, or right for a negative shift.
/// Returns the result, and `true` when a right shift lost a nonzero bit.
pub fn shift<const N: usize>(value: &[u64; N], shift: i64) -> ([u64; N], bool) {
    if shift >= 0 {
        let count = u32::try_from(shift).expect("a left shift fits the width");
        debug_assert!(
            value.bit_length() + count <= <[u64; N]>::BITS,
            "the shifted value fits"
        );
        (value.shl(count), false)
    } else {
        let count = u32::try_from(-shift).unwrap_or(u32::MAX);
        (value.shr(count), value.any_below(count))
    }
}

/// Returns the number of bits of 5^`exponent`, or one more, for an exponent
/// below 10,000.
pub fn power_of_five_bits(exponent: u32) -> i64 {
    // 2.3220 is above log2(5) by less than 1e-4, so the estimate is exact or
    // one more for every exponent below 10,000.
    (i64::from(exponent) * 23_220).div_euclid(10_000) + 1
}

/// Returns an estimate of `floor(top * log10(2)) + 1`, the number of decimal
/// digits of `2^top` above the decimal point. A value of `top + 1` bits has
/// this many digits or one more.
///
/// The factor 78913 / 2^18 is below log10(2) by less than 1e-6. The estimate
/// is never above the exact count for a positive `top`, and never below it
/// for a negative `top`. It is within one of the exact count while `|top|` is
/// below 1,000,000.
pub fn digits_estimate(top: i64) -> i64 {
    ((top * 78_913) >> 18) + 1
}

#[cfg(test)]
mod tests {
    use super::{LARGE, SMALL, multiply, power_of_five, power_of_five_bits, shift};
    use crate::limbs::{self, Limbs};

    #[test]
    fn powers_products_and_shifts() {
        assert_eq!(limbs::to_u128(&power_of_five::<SMALL>(3)), 125);
        assert_eq!(
            limbs::to_u128(&power_of_five::<SMALL>(27)),
            7_450_580_596_923_828_125
        );
        let large = power_of_five::<LARGE>(6216);
        assert_eq!(large.bit_length(), 14_434);
        let estimate = power_of_five_bits(6216);
        assert!((14_434..=14_435).contains(&estimate));
        let product = multiply(&power_of_five::<SMALL>(30), &[3_u64, 1]);
        let expected = limbs::multiply_small(power_of_five::<SMALL>(30), 3)
            .add(power_of_five::<SMALL>(30).shl(64));
        assert_eq!(product, expected);
        let (shifted, lost) = shift(&limbs::from_u128::<[u64; SMALL]>(0b1011), -2);
        assert_eq!((limbs::to_u128(&shifted), lost), (0b10, true));
    }
}
