//! The constants of the elementary functions, `ln 2` and `ln 10`, as
//! truncated binary integers.
//!
//! The tables come from the arbitrary-precision `ln` of Python's `decimal`
//! module at 1,100 digits, truncated. The unit tests recompute every bit
//! from series in integer arithmetic. Each table has 2,176 bits, enough for
//! the 2,048-bit working precision of binary512 and for a product by an
//! integer below 2^31: the reduction by `k ln 2`, the term `q ln 10` of a
//! decimal logarithm, and the factor `ln 2` or `ln 10` of `exp2`, `exp10`,
//! and their logarithms.

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

/// `floor(ln 10 * 2^2174)`, least significant limb first.
const LN10: [u64; 34] = [
    0x3982_A78C_A45D_DFC8,
    0xEF99_C8E5_F697_4F36,
    0x4CB0_466D_61BA_648E,
    0xAD6B_FBFF_D821_BA0A,
    0xE38A_5700_FFDE_2DB1,
    0xF77E_3760_4E94_3960,
    0xA949_EAAA_DF69_E8A5,
    0xE880_47F1_7B0D_9B50,
    0x3848_C8D2_5FAF_1BCA,
    0x3DFD_3C51_748E_6D6E,
    0xAF88_486E_A9B7_401E,
    0xF47F_A96D_EB27_1060,
    0x4576_5CDE_2683_39DB,
    0xE40B_F3CC_1E14_126A,
    0xDB1D_28EA_57D4_FDC0,
    0xA47E_CB26_978C_5D4F,
    0x9CD5_B42E_6A27_1619,
    0xE247_8FCA_AD3A_EE98,
    0x469E_A58E_9305_E981,
    0x5B08_B057_D5ED_E20F,
    0x8E93_368D_4478_9C4F,
    0xCA67_B35B_2360_5085,
    0x5161_BB49_D219_C7BB,
    0xEF66_CEB0_4AB3_C6FA,
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

/// `floor(pi * 2^2174)`, least significant limb first.
const PI: [u64; 34] = [
    0x8AEA_7157_5D06_0C7D,
    0xECFB_8504_58DB_EF0A,
    0xA855_21AB_DF1C_BA64,
    0xAD33_170D_0450_7A33,
    0x1572_8E5A_8AAA_C42D,
    0x15D2_2618_98FA_0510,
    0x3995_497C_EA95_6AE5,
    0xDE2B_CBF6_9558_1718,
    0xB5C5_5DF0_6F4C_52C9,
    0x9B27_83A2_EC07_A28F,
    0xE39E_772C_180E_8603,
    0x3290_5E46_2E36_CE3B,
    0xF174_6C08_CA18_217C,
    0x670C_354E_4ABC_9804,
    0x9ED5_2907_7096_966D,
    0x1C62_F356_2085_52BB,
    0x8365_5D23_DCA3_AD96,
    0x6916_3FA8_FD24_CF5F,
    0x98DA_4836_1C55_D39A,
    0xC200_7CB8_A163_BF05,
    0x4928_6651_ECE4_5B3D,
    0xAE9F_2411_7C4B_1FE6,
    0xEE38_6BFB_5A89_9FA5,
    0x0BFF_5CB6_F406_B7ED,
    0xF44C_42E9_A637_ED6B,
    0xE485_B576_625E_7EC6,
    0x4FE1_356D_6D51_C245,
    0x302B_0A6D_F25F_1437,
    0xEF95_19B3_CD3A_431B,
    0x514A_0879_8E34_04DD,
    0x020B_BEA6_3B13_9B22,
    0x2902_4E08_8A67_CC74,
    0xC4C6_628B_80DC_1CD1,
    0xC90F_DAA2_2168_C234,
];

/// The weight of the lowest bit of [`LN2`].
const LN2_LOWEST: i64 = -2176;

/// The weight of the lowest bit of [`LN10`].
const LN10_LOWEST: i64 = -2174;

/// The weight of the lowest bit of [`PI`].
const PI_LOWEST: i64 = -2174;

/// Returns `ln 2` at the width `W`. The radius covers the truncation of the
/// table and of the width.
pub(super) fn ln2<W: Widen>() -> Ball<W> {
    Ball::new(false, LN2, LN2_LOWEST).widened(Radius::power_of_two(LN2_LOWEST))
}

/// Returns `ln 10` at the width `W`.
pub(super) fn ln10<W: Widen>() -> Ball<W> {
    Ball::new(false, LN10, LN10_LOWEST).widened(Radius::power_of_two(LN10_LOWEST))
}

/// Returns `pi` at the width `W`.
pub(super) fn pi<W: Widen>() -> Ball<W> {
    Ball::new(false, PI, PI_LOWEST).widened(Radius::power_of_two(PI_LOWEST))
}

#[cfg(test)]
mod tests {
    use super::{LN2, LN10, PI};
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
        let bits = 2174;
        let total = bits + GUARD;
        let three_ln2 = limbs::multiply_small(ln2_series::<[u64; 36]>(bits), 3);
        let mut power = limbs::divide_small([0_u64; 36].with_bit(total + 1), Divisor::new(9)).0;
        let mut sum = [0_u64; 36];
        for j in 0_u64.. {
            if power.is_zero() {
                break;
            }
            sum = sum.add(limbs::divide_small(power, Divisor::new(2 * j + 1)).0);
            power = limbs::divide_small(power, Divisor::new(81)).0;
        }
        let ln10 = three_ln2.add(sum).shr(GUARD).resize::<[u64; 34]>();
        assert_eq!(ln10, LN10);
    }

    /// Returns `floor(atan(1 / n) * 2^total)` with the guard bits of
    /// `total`, from the alternating series of `1 / ((2j + 1) n^(2j + 1))`.
    /// The positive and the negative terms add up apart, and each floor
    /// loses less than one unit.
    fn atan_inverse_series(n: u64, total: u32) -> [u64; 36] {
        let square = Divisor::new(n * n);
        let mut power = limbs::divide_small([0_u64; 36].with_bit(total), Divisor::new(n)).0;
        let (mut positive, mut negative) = ([0_u64; 36], [0_u64; 36]);
        for j in 0_u64.. {
            if power.is_zero() {
                break;
            }
            let term = limbs::divide_small(power, Divisor::new(2 * j + 1)).0;
            if j % 2 == 0 {
                positive = positive.add(term);
            } else {
                negative = negative.add(term);
            }
            power = limbs::divide_small(power, square).0;
        }
        positive.sub(negative)
    }

    #[test]
    fn pi_matches_its_series() {
        // Machin's formula: pi = 16 atan(1/5) - 4 atan(1/239).
        let total = 2174 + GUARD;
        let fifth = limbs::multiply_small(atan_inverse_series(5, total), 16);
        let part = limbs::multiply_small(atan_inverse_series(239, total), 4);
        let pi = fifth.sub(part).shr(GUARD).resize::<[u64; 34]>();
        assert_eq!(pi, PI);
    }
}
