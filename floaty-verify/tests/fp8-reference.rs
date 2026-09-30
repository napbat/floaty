//! Compares the FP8 formats with the `ml_dtypes` reference table in
//! `data/fp8-reference.txt`.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz};
use floaty_verify::ml_dtypes::{Row, check, rows};
use floaty_verify::shape::{self, Payload, trailing_payload};
use rug::Integer;

const TABLE: &str = include_str!("../data/fp8-reference.txt");

macro_rules! dispatch {
    ($alias:expr, $bits:expr, $($name:ident),*) => {
        match $alias {
            $(stringify!($name) => {
                let value = $name::from_bits($bits);
                let context = format!("{} {:#04x}", $alias, $bits);
                assert_eq!(value.is_sign_negative(), $bits & 0x80 != 0, "{context} sign bit");
                // Only E5M2 has IEEE NaNs, which carry a payload.
                let payload = if $alias == "F8E5M2" {
                    trailing_payload(&Integer::from($bits), $name::PRECISION)
                } else {
                    Payload::None
                };
                let decoded = value.decode::<1>();
                shape::check(&decoded, value.classify(), $name::PRECISION, $name::EMIN, &payload, &context);
                (value.classify(), decoded)
            })*
            other => panic!("unknown format {other}"),
        }
    };
}

macro_rules! parameters {
    ($alias:expr, $($name:ident),*) => {
        match $alias {
            $(stringify!($name) => ($name::PRECISION, $name::EMAX, $name::EMIN),)*
            other => panic!("unknown format {other}"),
        }
    };
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
                let ours = parameters!(alias, F8E4M3Fn, F8E5M2, F8E4M3Fnuz, F8E5M2Fnuz);
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
                let (ours, decoded) =
                    dispatch!(alias, bits, F8E4M3Fn, F8E5M2, F8E4M3Fnuz, F8E5M2Fnuz);
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
        (4, 4 * 256),
        "the table covers every encoding"
    );
}
