//! Decimal digits of binary integers: powers of 10, digit counts, and the
//! last digit.

use crate::limbs::{self, Limbs};

/// The powers of 10 that fit a `u128`: 10^0 to 10^38.
const POWERS: [u128; 39] = {
    let mut table = [1; 39];
    let mut index = 1;
    while index < 39 {
        table[index] = table[index - 1] * 10;
        index += 1;
    }
    table
};

/// The largest power of 10 in a `u64`: 10^19.
const LARGE_STEP: u64 = 10_000_000_000_000_000_000;

/// Returns 10^`exponent`. The power must fit `L`.
pub fn power_of_ten<L: Limbs>(exponent: u32) -> L {
    let index = usize::try_from(exponent).expect("an exponent fits a usize");
    if let Some(&power) = POWERS.get(index) {
        return limbs::from_u128(power);
    }
    let mut power: L = limbs::from_u128(POWERS[38]);
    let mut left = exponent - 38;
    while left >= 19 {
        power = limbs::multiply_small(power, LARGE_STEP);
        left -= 19;
    }
    let small = u64::try_from(power_of_ten_u128(left)).expect("10^18 fits a u64");
    limbs::multiply_small(power, small)
}

/// Returns 10^`exponent` as a `u128`. The exponent is at most 38.
pub const fn power_of_ten_u128(exponent: u32) -> u128 {
    10_u128.pow(exponent)
}

/// log10(2) as a fraction of 2^64, rounded down.
const LOG10_2: u64 = 0x4D10_4D42_7DE7_FBCC;

/// Returns the number of decimal digits of `value`, or 0 for zero.
pub fn digit_count<L: Limbs>(value: &L) -> u32 {
    let bits = value.bit_length();
    if bits == 0 {
        return 0;
    }
    // With n = bits - 1, the value has floor(n * log10(2)) + 1 digits, or
    // one more. `LOG10_2` is below log10(2) by less than 2^-64, so the
    // estimate is below floor(n * log10(2)) + 1 only when the fraction of
    // n * log10(2) is below 2^-32. The value then has the smaller count. So
    // the count is `estimate` or `estimate + 1` for every u32 bit count.
    let product = u128::from(bits - 1) * u128::from(LOG10_2);
    let estimate = u32::try_from((product >> 64) + 1).expect("a digit count fits a u32");
    // value >= 10^estimate exactly when value / 10 >= 10^(estimate - 1), a
    // power that fits because it is at most 2^(bits - 1).
    let (tenth, _) = limbs::divide_small(*value, 10);
    if tenth.compare(&power_of_ten(estimate - 1)).is_ge() {
        estimate + 1
    } else {
        estimate
    }
}

/// Returns the number of decimal digits of a `u128`, or 0 for zero.
pub fn digit_count_u128(value: u128) -> u32 {
    let count = POWERS.iter().take_while(|&&power| power <= value).count();
    u32::try_from(count).expect("a digit count fits a u32")
}

/// Returns the last decimal digit of `value`.
pub fn last_digit<L: Limbs>(value: &L) -> u64 {
    // 2^64 = 6 (mod 10), and 6 * 6 = 6 (mod 10), so every limb above the
    // first adds six times its value modulo 10.
    let count = usize::try_from(L::BITS / 64).expect("a limb count fits a usize");
    let rest: u128 = (1..count)
        .map(|index| 6 * u128::from(value.limb(index) % 10))
        .sum();
    let sum = u128::from(value.limb(0) % 10) + rest;
    u64::try_from(sum % 10).expect("a digit fits a u64")
}

#[cfg(test)]
mod tests {
    use super::{digit_count, last_digit, power_of_ten};
    use crate::limbs::Limbs;

    #[test]
    fn digits_of_wide_values() {
        assert_eq!(digit_count(&[0_u64]), 0);
        assert_eq!(digit_count(&[9_u64]), 1);
        assert_eq!(digit_count(&[10_u64]), 2);
        assert_eq!(digit_count(&[u64::MAX]), 20);
        let big: [u64; 4] = power_of_ten(60);
        assert_eq!(digit_count(&big), 61);
        assert_eq!(digit_count(&big.sub([1, 0, 0, 0])), 60);
        assert_eq!(last_digit(&big.sub([1, 0, 0, 0])), 9);
        assert_eq!(last_digit(&[0_u64, 1]), 6);
    }
}
