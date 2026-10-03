//! The constants of the elementary functions, `ln 2` and `ln 10`, as
//! truncated binary integers.
//!
//! The tables come from the arbitrary-precision `ln` of Python's `decimal`
//! module at 1,100 digits, truncated. The unit tests recompute every bit
//! from series in integer arithmetic. `ln 2` has 2,176 bits, enough for the
//! 2,048-bit working precision of binary512 and for a reduction by `k ln 2`
//! with `|k|` below 2^31. `ln 10` has 640 bits, enough for the 512-bit
//! working precision of decimal128 and for `q ln 10` with `|q|` below 2^13.

use super::ball::{Ball, Radius};
use crate::limbs::Widen;

/// `floor(ln 2 * 2^2176)`, least significant limb first.
const LN2: [u64; 34] = [
    0x7598_A195_1AE2_73EE,
    0x4D16_2DB3_B365_853D,
    0x5F50_B518_5064_C18B,
    0x078F_735D_1B2D_B31B,
    0xAE31_3CDB_6C60_6CB1,
    0x955D_5179_B1E1_7B9D,
    0x0C48_0A54_1735_0D2C,
    0x074D_B601_5CFE_7AA3,
    0x6A9C_7F8A_5E14_8E82,
    0x2566_9B33_3564_A337,
    0x4C1A_1E0B_D1D6_095D,
    0xCCCC_4E65_9393_514C,
    0xC943_E732_B479_CD33,
    0x1746_0775_DB89_90E5,
    0x7D2E_23DE_1400_B396,
    0xEE56_9D6D_FC1E_FA15,
    0x610D_30F8_8FE5_51A2,
    0x07F4_CA11_FB5B_FB90,
    0xDA2D_97C5_0F3F_D5C6,
    0x655F_A187_2F20_E3A2,
    0xF5DF_A6BD_3830_3248,
    0x72CE_87B1_9D65_48CA,
    0x256F_A0EC_7657_F74B,
    0xB9EA_9BC3_B136_603B,
    0x1ACB_DA11_317C_387E,
    0x3E96_CA16_224A_E8C5,
    0x2757_3B29_1169_B825,
    0xED2E_AE35_C138_2144,
    0x5595_52FB_4AFA_1B10,
    0xE7B8_7620_6DEB_AC98,
    0x8A0D_175B_8BAA_FA2B,
    0x40F3_4326_7298_B62D,
    0xC9E3_B398_03F2_F6AF,
    0xB172_17F7_D1CF_79AB,
];

/// `floor(ln 10 * 2^638)`, least significant limb first.
const LN10: [u64; 10] = [
    0x765A_A6C3_B0D8_31FB,
    0x782C_F8A2_8A8C_911E,
    0xFB8F_7884_02E5_16D6,
    0x2C62_2418_410B_E2DA,
    0xCC70_CBC0_2C5F_0D68,
    0x962F_02D7_B1A8_105C,
    0x83C6_1E82_01F0_2D72,
    0xE28F_ECF9_DA5D_F90E,
    0xEA56_D62B_82D3_0A28,
    0x935D_8DDD_AAA8_AC16,
];

/// The weight of the lowest bit of [`LN2`].
const LN2_LOWEST: i64 = -2176;

/// The weight of the lowest bit of [`LN10`].
const LN10_LOWEST: i64 = -638;

/// Returns `ln 2` at the width `W`. The radius covers the truncation of the
/// table and of the width.
pub(super) fn ln2<W: Widen>() -> Ball<W> {
    Ball::new(false, LN2, LN2_LOWEST).widened(Radius::power_of_two(LN2_LOWEST))
}

/// Returns `ln 10` at the width `W`.
pub(super) fn ln10<W: Widen>() -> Ball<W> {
    Ball::new(false, LN10, LN10_LOWEST).widened(Radius::power_of_two(LN10_LOWEST))
}

#[cfg(test)]
mod tests {
    use super::{LN2, LN10};
    use crate::limbs::{self, Divisor, Limbs};

    /// The bits below the lowest bit of a table that the series carry. Each
    /// term rounds down by less than one unit there, and fewer than 2^12
    /// terms lose less than one unit of the table.
    const GUARD: u32 = 64;

    /// Returns `floor(ln 2 * 2^bits)` with the guard bits, from
    /// `ln 2 = sum over k >= 1 of 1 / (k 2^k)`. Terms past `bits` vanish.
    fn ln2_series<L: Limbs>(bits: u32) -> L {
        let total = bits + GUARD;
        (1..total).fold(L::ZERO, |sum, k| {
            let numerator = L::ZERO.with_bit(total - k);
            sum.add(limbs::divide_small(numerator, Divisor::new(u64::from(k))).0)
        })
    }

    #[test]
    fn ln2_matches_its_series() {
        let sum = ln2_series::<[u64; 36]>(2176);
        assert_eq!(sum.shr(GUARD).resize::<[u64; 34]>(), LN2);
    }

    #[test]
    fn ln10_matches_its_series() {
        // ln 10 = 3 ln 2 + ln(5/4), and ln(5/4) = 2 atanh(1/9), the sum over
        // j >= 0 of 2 / ((2j + 1) 9^(2j + 1)). Each power divides the last
        // by 81, and floors of floors are floors of the exact quotients.
        let bits = 638;
        let total = bits + GUARD;
        let three_ln2 = limbs::multiply_small(ln2_series::<[u64; 12]>(bits), 3);
        let mut power = limbs::divide_small([0_u64; 12].with_bit(total + 1), Divisor::new(9)).0;
        let mut sum = [0_u64; 12];
        for j in 0_u64.. {
            if power.is_zero() {
                break;
            }
            sum = sum.add(limbs::divide_small(power, Divisor::new(2 * j + 1)).0);
            power = limbs::divide_small(power, Divisor::new(81)).0;
        }
        let ln10 = three_ln2.add(sum).shr(GUARD).resize::<[u64; 10]>();
        assert_eq!(ln10, LN10);
    }
}
