//! Reads the reference tables that the `ml_dtypes` scripts in `scripts/`
//! generate: `data/fp8-reference.txt` and `data/mx-reference.txt`.

use floaty::{Class, Decoded};

/// One line of the table: the parameters of a format, or one encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Row<'a> {
    /// The parameters of a format, as `ml_dtypes.finfo` gives them.
    Format {
        /// The floaty alias of the format.
        alias: &'a str,
        /// The precision in bits.
        precision: u32,
        /// The exponent of the largest finite value.
        emax: i32,
        /// The exponent of the smallest normal value.
        emin: i32,
    },
    /// One encoding of a format.
    Encoding {
        /// The floaty alias of the format.
        alias: &'a str,
        /// The encoding.
        bits: u8,
        /// `zero`, `subnormal`, `normal`, `infinite`, or `nan`.
        class: &'a str,
        /// `+` or `-`.
        sign: &'a str,
        /// The value as Python `float.hex` writes it, or `nan`.
        value: &'a str,
    },
}

/// Returns the rows of a table, without its comments.
///
/// # Panics
///
/// Panics for a malformed line.
pub fn rows(table: &str) -> impl Iterator<Item = Row<'_>> {
    table
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
///
/// # Panics
///
/// Panics for text that is not a finite hexadecimal float.
#[must_use]
pub fn parse_hex(text: &str) -> (bool, u128, i32) {
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
///
/// # Panics
///
/// Panics for a significand of zero, which has no trailing bit.
#[must_use]
pub fn normalize(significand: u128, exponent: i32) -> (u128, i32) {
    let zeros = significand.trailing_zeros();
    (
        significand >> zeros,
        exponent + i32::try_from(zeros).expect("at most 128 zeros"),
    )
}

/// Checks one encoding against its table row: its class, sign, and value.
///
/// # Panics
///
/// Panics when the encoding differs from the row.
pub fn check(context: &str, ours: Class, decoded: Decoded<1>, row: (&str, &str, &str)) {
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
