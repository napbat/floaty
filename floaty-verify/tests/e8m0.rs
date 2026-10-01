//! Checks the E8M0 recipe of the README against `ml_dtypes` 0.6.0: the
//! binary32 value of every code, and the code of binary32 inputs at every
//! exponent field and every rounding case, in `data/mx-e8m0.bin`.
//!
//! The recipe converts binary32 to binary64 with a precision limit of one
//! bit, rounding a tie away from zero. It maps an overflow, a zero, a
//! negative value, and a NaN to the NaN code 255, and a value below 2^-127 to
//! code 0.

use core::num::NonZeroU32;

use floaty::{Env, Exact, F32, F64, Rounding};

const TABLE: &[u8] = include_bytes!("../data/mx-e8m0.bin");

/// The binary32 fraction fields of the inputs, in the order of
/// `E8M0_FRACTIONS` in `scripts/generate_mx_reference.py`.
const FRACTIONS: [u32; 9] = [
    0x00_0000, 0x00_0001, 0x3F_FFFF, 0x40_0000, 0x40_0001, 0x5F_FFFF, 0x60_0000, 0x60_0001,
    0x7F_FFFF,
];

/// Returns the E8M0 code of a binary32 value, by the recipe.
fn to_scale(value: F32) -> u8 {
    let one_bit = Env::IEEE
        .with_rounding(Rounding::TiesToAway)
        .with_precision(NonZeroU32::new(1));
    let (power, _) = value.convert_with::<F64>(one_bit);
    if power.is_nan() || power.is_sign_negative() || power.is_zero() {
        return 0xFF;
    }
    let exponent = i32::try_from(power.to_bits() >> 52).expect("11 bits") - 1023;
    if exponent > 127 {
        return 0xFF;
    }
    u8::try_from(exponent.max(-127) + 127).expect("the code is below 255")
}

/// Returns the binary32 value of an E8M0 code, by the recipe.
fn from_scale(code: u8) -> F32 {
    if code == 0xFF {
        return F32::from_bits(0x7FC0_0000);
    }
    let power = Exact {
        negative: false,
        exponent: i32::from(code) - 127,
        significand: [1],
        sticky: false,
    };
    F32::round(power, Env::IEEE).0
}

/// Returns the binary32 inputs of the table, in the order of `e8m0_inputs`
/// in `scripts/generate_mx_reference.py`.
fn inputs() -> impl Iterator<Item = u32> {
    (0..=0xFF_u32).flat_map(|field| {
        FRACTIONS
            .into_iter()
            .flat_map(move |fraction| [0, 1].map(|sign| sign << 31 | field << 23 | fraction))
    })
}

#[test]
fn every_code_reads_like_ml_dtypes() {
    let (values, _) = TABLE.split_at(1024);
    for (code, &bytes) in (0..=u8::MAX).zip(values.as_chunks::<4>().0) {
        let expected = u32::from_le_bytes(bytes);
        assert_eq!(from_scale(code).to_bits(), expected, "code {code}");
    }
}

#[test]
fn every_rounding_case_of_binary32_converts_like_ml_dtypes() {
    let (_, codes) = TABLE.split_at(1024);
    assert_eq!(codes.len(), 256 * FRACTIONS.len() * 2, "one code per input");
    for (bits, &expected) in inputs().zip(codes) {
        let ours = to_scale(F32::from_bits(bits));
        // Conflict: ml_dtypes 0.6.0 rounds a binary32 subnormal by its bits,
        // as if code 0 were zero. So an input above 2^-127 and below
        // 1.5 * 2^-127 gives code 1 (2^-126), although 2^-127 is nearer:
        // 0x0040_0001 is 2^-127 + 2^-149. The rule of the recipe, the nearest
        // power of two, gives code 0: the distance to 2^-127 is below
        // 2^-128, and the distance to 2^-126 is above 2^-128. The test
        // checks those inputs by that rule.
        if (0x0040_0001..0x0060_0000).contains(&bits) {
            assert_eq!((ours, expected), (0, 1), "{bits:#010x}");
        } else {
            assert_eq!(ours, expected, "{bits:#010x}");
        }
    }
}
