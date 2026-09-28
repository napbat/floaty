//! Compares the FP8 formats with the `ml_dtypes` reference table in
//! `data/fp8-reference.txt`.

use floaty::{Class, Decoded, F8E4M3, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz};
use floaty_verify::shape::{self, Payload, trailing_payload};
use rug::Integer;

const TABLE: &str = include_str!("../data/fp8-reference.txt");

/// One line of the table: the parameters of a format, or one encoding.
enum Row<'a> {
    Format {
        alias: &'a str,
        precision: u32,
        emax: i32,
        emin: i32,
    },
    Encoding {
        alias: &'a str,
        bits: u8,
        class: &'a str,
        sign: &'a str,
        value: &'a str,
    },
}

fn rows() -> impl Iterator<Item = Row<'static>> {
    TABLE
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            let fields: Vec<&str> = line.split(' ').collect();
            match fields.as_slice() {
                ["format", alias, precision, emax, emin] => Row::Format {
                    alias,
                    precision: precision.parse().expect("the precision is an integer"),
                    emax: emax.parse().expect("emax is an integer"),
                    emin: emin.parse().expect("emin is an integer"),
                },
                [alias, bits, class, sign, value] => Row::Encoding {
                    alias,
                    bits: u8::from_str_radix(bits.trim_start_matches("0x"), 16)
                        .expect("the encoding is a hexadecimal byte"),
                    class,
                    sign,
                    value,
                },
                _ => panic!("malformed reference line: {line}"),
            }
        })
}

/// Parses the output of Python `float.hex` for a finite value into a sign, an
/// integer significand, and the weight of its lowest bit.
fn parse_hex(text: &str) -> (bool, u128, i32) {
    let (negative, body) = text
        .strip_prefix('-')
        .map_or((false, text), |rest| (true, rest));
    let body = body
        .strip_prefix("0x")
        .expect("a hexadecimal float starts with 0x");
    let (digits, exponent) = body
        .split_once('p')
        .expect("a hexadecimal float has an exponent");
    let (integer, fraction) = digits.split_once('.').unwrap_or((digits, ""));
    let significand = u128::from_str_radix(&format!("{integer}{fraction}"), 16)
        .expect("the significand is hexadecimal");
    let exponent: i32 = exponent.parse().expect("the exponent is decimal");
    let fraction_bits = 4 * i32::try_from(fraction.len()).expect("the fraction is short");
    (negative, significand, exponent - fraction_bits)
}

/// Removes trailing zero bits, so that equal values compare equal.
fn normalize(significand: u128, exponent: i32) -> (u128, i32) {
    let zeros = significand.trailing_zeros();
    (
        significand >> zeros,
        exponent + i32::try_from(zeros).expect("at most 128 zeros"),
    )
}

/// Checks one encoding against its table row.
fn check(context: &str, ours: Class, decoded: Decoded<1>, row: (&str, &str, &str)) {
    let (class, sign, value) = row;
    match class {
        "nan" => assert!(
            matches!(ours, Class::QuietNan | Class::SignalingNan),
            "{context}"
        ),
        "zero" => assert_eq!(ours, Class::Zero, "{context}"),
        "subnormal" => assert_eq!(ours, Class::Subnormal, "{context}"),
        "normal" => assert_eq!(ours, Class::Normal, "{context}"),
        "infinite" => assert_eq!(ours, Class::Infinite, "{context}"),
        _ => panic!("unknown class {class} for {context}"),
    }
    let negative = match decoded {
        Decoded::Zero { negative, .. }
        | Decoded::Infinity { negative }
        | Decoded::Nan { negative, .. } => negative,
        Decoded::Finite {
            negative,
            exponent,
            significand: [significand],
        } => {
            let (_, expected, expected_exponent) = parse_hex(value);
            assert_eq!(
                normalize(u128::from(significand), exponent),
                normalize(expected, expected_exponent),
                "{context} value"
            );
            negative
        }
        Decoded::Unsupported => panic!("{context} is not unsupported"),
    };
    assert_eq!(negative, sign == "-", "{context} sign");
}

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
    for row in rows() {
        match row {
            Row::Format {
                alias,
                precision,
                emax,
                emin,
            } => {
                let ours = parameters!(alias, F8E4M3, F8E5M2, F8E4M3Fnuz, F8E5M2Fnuz);
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
                    dispatch!(alias, bits, F8E4M3, F8E5M2, F8E4M3Fnuz, F8E5M2Fnuz);
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
