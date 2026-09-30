//! Compares the conversion of every binary16 encoding to the FP8 formats
//! with `ml_dtypes`, rounding to nearest even, in `data/fp8-from-f16.bin`.
//!
//! `ml_dtypes` gives every NaN result a canonical payload, so a NaN result
//! only has to be a NaN with the same sign. `conversions-mpfr.rs` checks the
//! payload rule of floaty, binary16 to E5M2 included.

use floaty::{Env, F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F16};

const TABLE: &[u8] = include_bytes!("../data/fp8-from-f16.bin");

/// Checks every binary16 encoding against the table block of one format.
macro_rules! check_format {
    ($alias:ty, $block:literal) => {{
        let block = &TABLE[$block * 65536..($block + 1) * 65536];
        for (bits, &expected) in (0..=u16::MAX).zip(block) {
            let (ours, _): ($alias, _) = F16::from_bits(bits).convert_with(Env::IEEE);
            let expected = <$alias>::from_bits(expected);
            let context = format!("{} from F16 {bits:#06x}", stringify!($alias));
            if expected.is_nan() {
                assert!(ours.is_nan(), "{context}: {ours:?} is a NaN");
                assert_eq!(
                    ours.is_sign_negative(),
                    expected.is_sign_negative(),
                    "{context}: sign"
                );
            } else {
                assert_eq!(ours.to_bits(), expected.to_bits(), "{context}");
            }
        }
    }};
}

#[test]
fn every_binary16_encoding_converts_like_ml_dtypes() {
    assert_eq!(
        TABLE.len(),
        4 * 65536,
        "the table has one block for each format"
    );
    check_format!(F8E4M3Fn, 0);
    check_format!(F8E5M2, 1);
    check_format!(F8E4M3Fnuz, 2);
    check_format!(F8E5M2Fnuz, 3);
}
