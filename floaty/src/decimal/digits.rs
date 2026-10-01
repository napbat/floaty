//! Decimal digits of binary integers: powers of 10, digit counts, and the
//! last digit.

use crate::limbs::{self, Divisor, Limbs, high_u64, low_u64};

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

/// The powers of 10 from 10^39 to 10^77 in four limbs, low limb first.
/// 10^77 is the largest power of 10 below 2^256.
const WIDE_POWERS: [[u64; 4]; 39] = {
    let mut table = [[0; 4]; 39];
    let mut power = [low_u64(POWERS[38]), high_u64(POWERS[38]), 0, 0];
    let mut index = 0;
    while index < 39 {
        // Multiplies `power` by 10. `u128::from` is not callable in a
        // constant; the casts widen.
        let mut carry = 0;
        let mut limb = 0;
        while limb < 4 {
            let product = power[limb] as u128 * 10 + carry as u128;
            power[limb] = low_u64(product);
            carry = high_u64(product);
            limb += 1;
        }
        table[index] = power;
        index += 1;
    }
    table
};

/// For each bit length from 0 to 128: the digit count of the smallest value
/// of that length, and the power of 10 that adds a digit, or 0 when no value
/// of that length reaches the next power.
const BIT_LENGTH_DIGITS: [(u32, u128); 129] = {
    let mut table = [(0, 0); 129];
    let mut bits = 1;
    while bits <= 128 {
        // The smallest value of the length is 2^(bits - 1).
        let smallest = 1_u128 << (bits - 1);
        // `usize::from` is not callable in a constant; the casts widen.
        let mut count = 1_u32;
        while count < 39 && POWERS[count as usize] <= smallest {
            count += 1;
        }
        // The power at `count` is above `smallest`. A value of the length
        // reaches it when the power has the same length.
        let threshold = if count < 39 && (bits == 128 || POWERS[count as usize] >> bits == 0) {
            POWERS[count as usize]
        } else {
            0
        };
        table[bits] = (count, threshold);
        bits += 1;
    }
    table
};

/// The largest power of 10 in a `u64`: 10^19.
const LARGE_STEP: u64 = 10_000_000_000_000_000_000;

/// The powers of 10 that fit a `u64`, 10^0 to 10^19, as divisors.
const DIVISORS: [Divisor; 20] = {
    let mut table = [Divisor::new(1); 20];
    let mut power = 1;
    let mut index = 1;
    while index < 20 {
        power *= 10;
        table[index] = Divisor::new(power);
        index += 1;
    }
    table
};

/// Returns 10^`exponent` as a divisor. The exponent is at most 19.
#[inline]
pub fn power_divisor(exponent: u32) -> Divisor {
    DIVISORS[usize::try_from(exponent).expect("an exponent fits a usize")]
}

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

/// Returns half of 10^`exponent`, 5 * 10^(`exponent` - 1), for an exponent
/// from 1 to 19.
#[inline]
pub fn half_power(exponent: u32) -> u64 {
    let index = usize::try_from(exponent - 1).expect("an exponent fits a usize");
    u64::try_from(POWERS[index] * 5).expect("half of 10^19 fits a u64")
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
    if bits <= 128 {
        let (count, threshold) =
            BIT_LENGTH_DIGITS[usize::try_from(bits).expect("a bit length fits a usize")];
        let reaches = threshold != 0 && limbs::to_u128(value) >= threshold;
        return count + u32::from(reaches);
    }
    // With n = bits - 1, the value has floor(n * log10(2)) + 1 digits, or
    // one more. `LOG10_2` is below log10(2) by less than 2^-64, so the
    // estimate is below floor(n * log10(2)) + 1 only when the fraction of
    // n * log10(2) is below 2^-32. The value then has the smaller count. So
    // the count is `estimate` or `estimate + 1` for every u32 bit count.
    let product = u128::from(bits - 1) * u128::from(LOG10_2);
    let estimate = u32::try_from((product >> 64) + 1).expect("a digit count fits a u32");
    if bits <= 256 {
        // The estimate is from 39 to 77 here, so the table holds its power.
        let index = usize::try_from(estimate - 39).expect("a digit count fits a usize");
        let reaches = (0..4)
            .rev()
            .map(|limb| value.limb(limb))
            .cmp(WIDE_POWERS[index].iter().rev().copied())
            .is_ge();
        return estimate + u32::from(reaches);
    }
    // value >= 10^estimate exactly when value / 10 >= 10^(estimate - 1), a
    // power that fits because it is at most 2^(bits - 1).
    let (tenth, _) = limbs::divide_small(*value, power_divisor(1));
    if tenth.compare(&power_of_ten(estimate - 1)).is_ge() {
        estimate + 1
    } else {
        estimate
    }
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
    use super::{digit_count, half_power, last_digit, power_of_ten};
    use crate::limbs::Limbs;

    #[test]
    fn half_powers_span_every_chunk() {
        assert_eq!(half_power(1), 5);
        assert_eq!(half_power(4), 5_000);
        assert_eq!(half_power(19), 5_000_000_000_000_000_000);
    }

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

    #[test]
    fn digit_counts_change_at_each_power_of_ten() {
        // Eight limbs reach the division past 256 bits, four limbs the table
        // of wide powers, and two limbs the table of `u128` powers.
        let one = [1, 0, 0, 0, 0, 0, 0, 0];
        for exponent in 0..=100 {
            let power: [u64; 8] = power_of_ten(exponent);
            let below = power.sub(one);
            assert_eq!(digit_count(&power), exponent + 1, "10^{exponent}");
            assert_eq!(digit_count(&below), exponent, "10^{exponent} - 1");
            if power.bit_length() <= 256 {
                let four: [u64; 4] = power.resize();
                assert_eq!(digit_count(&four), exponent + 1, "10^{exponent}");
                assert_eq!(digit_count(&below.resize::<[u64; 4]>()), exponent);
            }
            if power.bit_length() <= 128 {
                let narrow: [u64; 2] = power.resize();
                assert_eq!(digit_count(&narrow), exponent + 1, "10^{exponent}");
                assert_eq!(digit_count(&below.resize::<[u64; 2]>()), exponent);
            }
        }
        // 2^128 - 1 and 2^128 have 39 digits, on each side of the u128 range.
        assert_eq!(digit_count(&[u64::MAX, u64::MAX]), 39);
        assert_eq!(digit_count(&[0_u64, 0, 1, 0]), 39);
        // 2^256 - 1 and 2^256 have 78 digits, on each side of the wide table.
        assert_eq!(digit_count(&[u64::MAX; 4]), 78);
        assert_eq!(digit_count(&[0_u64, 0, 0, 0, 1, 0, 0, 0]), 78);
    }

    #[test]
    fn digit_counts_hold_at_both_ends_of_each_bit_length() {
        // Counts the digits by repeated division, independent of the tables.
        let reference = |mut value: u128| {
            let mut count = 0;
            while value != 0 {
                value /= 10;
                count += 1;
            }
            count
        };
        for bits in 1..=128 {
            let smallest = 1_u128 << (bits - 1);
            let largest = smallest | (smallest - 1);
            for value in [smallest, largest] {
                let limbs: [u64; 2] = crate::limbs::from_u128(value);
                assert_eq!(digit_count(&limbs), reference(value), "{value}");
            }
        }
    }
}
