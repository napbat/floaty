//! Compares the FP8 formats with the `ml_dtypes` reference table in
//! `data/fp8-reference.txt`.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{Class, Decoded};
use floaty_verify::formats::Specials;
use floaty_verify::ml_dtypes::{Row, check, rows};
use floaty_verify::shape::{self, Payload, trailing_payload};
use rug::Integer;

const TABLE: &str = include_str!("../data/fp8-reference.txt");

/// Returns the class and the decoded value of the encoding `bits` of the
/// FP8 format `alias`, after the shape checks of the encoding.
///
/// # Panics
///
/// Panics for a format that is not in the FP8 list.
fn decode(alias: &str, bits: u8) -> (Class, Decoded<1>) {
    macro_rules! decode_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            if alias == stringify!($alias) {
                let value = floaty::$alias::from_bits(bits);
                let context = format!("{alias} {bits:#04x}");
                assert_eq!(
                    value.is_sign_negative(),
                    bits & 0x80 != 0,
                    "{context} sign bit"
                );
                // Only the formats with IEEE NaNs carry a payload.
                let payload = if matches!($specials, Specials::Ieee) {
                    trailing_payload(&Integer::from(bits), floaty::$alias::PRECISION)
                } else {
                    Payload::None
                };
                let decoded = value.decode::<1>();
                shape::check(
                    &decoded,
                    value.classify(),
                    floaty::$alias::PRECISION,
                    floaty::$alias::EMIN,
                    &payload,
                    &context,
                );
                return (value.classify(), decoded);
            }
        };
    }
    floaty_verify::for_each_fp8_format!(decode_listed);
    panic!("unknown format {alias}")
}

/// Returns the precision, `emax`, and `emin` of the FP8 format `alias`.
///
/// # Panics
///
/// Panics for a format that is not in the FP8 list.
fn parameters(alias: &str) -> (u32, i32, i32) {
    macro_rules! parameters_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            if alias == stringify!($alias) {
                return (
                    floaty::$alias::PRECISION,
                    floaty::$alias::EMAX,
                    floaty::$alias::EMIN,
                );
            }
        };
    }
    floaty_verify::for_each_fp8_format!(parameters_listed);
    panic!("unknown format {alias}")
}

#[test]
fn every_fp8_encoding_matches_ml_dtypes() {
    let mut encodings = 0;
    let mut formats = 0;
    for row in rows(TABLE) {
        match row {
            Row::Format {
                alias,
                precision,
                emax,
                emin,
            } => {
                let ours = parameters(alias);
                assert_eq!(ours, (precision, emax, emin), "{alias} parameters");
                formats += 1;
            }
            Row::Encoding {
                alias,
                bits,
                class,
                sign,
                value,
            } => {
                let (ours, decoded) = decode(alias, bits);
                check(
                    &format!("{alias} {bits:#04x}"),
                    ours,
                    decoded,
                    (class, sign, value),
                );
                encodings += 1;
            }
        }
    }
    assert_eq!(
        (formats, encodings),
        (7, 7 * 256),
        "the table covers every encoding"
    );
}
