//! Reads `readtest.in`, the tests of the Intel Decimal Floating-Point Math
//! Library.
//!
//! `build.rs` extracts the file from the pinned archive to [`PATH`]. The
//! library's `TESTS/readtest.c` reads it, and this module follows that
//! program. A test line is
//!
//! ```text
//! function rounding operand... result flags annotation... -- comment
//! ```
//!
//! - `rounding` is a `BID_ROUNDING_*` value from 0 to 4.
//! - An operand or a result is a [`Field`]: hexadecimal digits in brackets,
//!   or a decimal string or integer.
//! - `flags` is the expected status in hexadecimal, with or without `0x`,
//!   as [`Flags`] bits.
//! - The annotations `ulp=`, `underflow_before_only`, `str_prefix=`, and
//!   `longintsize=` end the test.
//! - `ulp=` gives the error of the expected result in units in the last
//!   place. `readtest.c` uses it only for the functions that it compares
//!   within an error bound: the functions that are not correctly rounded, and
//!   `fmod`. floaty does not have these functions. `readtest.c` compares the
//!   other functions exactly and ignores `ulp=` on their lines, for example
//!   on lines of `sqrt`, `scalbn`, `scalbln`, `ldexp`, and `logb`. The tests
//!   also ignore it.
//! - `str_prefix=` gives the white space before a string operand of the
//!   `strtod` and `wcstod` functions, which the tests do not run.
//! - A line with `underflow_before_only` expects underflow only when the
//!   library detects tininess before rounding, which this build does.
//! - `--` starts a comment, even inside a token.
//!
//! `readtest.c` converts a decimal string operand of a decimal format with
//! the `from_string` function of that format, rounding to nearest, and a
//! decimal string result with the rounding direction of the line.
//! [`Field::decimal`] makes the same conversion.

use std::fmt;
use std::fs;

use crate::intel_decimal::{Flags, Format, Integer, Rounding};

/// The path of the extracted `readtest.in`.
pub const PATH: &str = env!("FLOATY_INTEL_READTEST");

/// An operand or a result of a test line.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Field {
    /// The hexadecimal digits in brackets, without the brackets. The two
    /// halves of a 128-bit value can have a comma between them.
    Hex(String),
    /// A decimal string or an integer.
    Text(String),
}

/// A field that does not have the form its function needs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FieldError {
    /// The field.
    pub field: Field,
    /// The form that the function needs.
    pub expected: &'static str,
}

impl fmt::Display for FieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?} is not {}", self.field, self.expected)
    }
}

impl std::error::Error for FieldError {}

impl Field {
    fn error(&self, expected: &'static str) -> FieldError {
        FieldError {
            field: self.clone(),
            expected,
        }
    }

    /// Returns the digits of a hexadecimal field.
    fn hex_digits(&self) -> Result<&str, FieldError> {
        match self {
            Self::Hex(digits) => Ok(digits),
            Self::Text(_) => Err(self.error("hexadecimal")),
        }
    }

    /// Reads a hexadecimal field with one `sscanf` conversion of at most
    /// `width` digits, as `%08x` and `%016llx` do.
    fn scan(&self, width: usize) -> Result<u128, FieldError> {
        scan_hex(self.hex_digits()?, width)
            .map(|(value, _)| value)
            .ok_or_else(|| self.error("hexadecimal"))
    }

    /// Reads a hexadecimal field with two conversions: at most `high`
    /// digits for the bits above bit 64, then at most 16 digits for the low
    /// 64 bits, as `%016llx%016llx` and `%04x%016llx` do. With `comma`, a
    /// comma can also separate the halves, as `%016llx,%016llx` allows.
    fn scan_pair(&self, high: usize, comma: bool) -> Result<u128, FieldError> {
        let error = || self.error("two hexadecimal halves");
        let (high, rest) = scan_hex(self.hex_digits()?, high).ok_or_else(error)?;
        if let Some((low, _)) = scan_hex(rest, 16) {
            return Ok((high << 64) | low);
        }
        let rest = rest.strip_prefix(',').filter(|_| comma).ok_or_else(error)?;
        let (low, _) = scan_hex(rest, 16).ok_or_else(error)?;
        Ok((high << 64) | low)
    }

    /// Returns the encoding of format `F`: the bits in brackets, or a
    /// decimal string that `F::from_string` converts with `rounding`.
    ///
    /// # Errors
    ///
    /// Returns a [`FieldError`] when the field has no hexadecimal digits.
    pub fn decimal<F: Format>(&self, rounding: Rounding) -> Result<F::Bits, FieldError> {
        let bits = match (self, size_of::<F::Bits>()) {
            (Self::Text(text), _) => return Ok(F::from_string(text, rounding).value),
            (Self::Hex(_), 4) => self.scan(8)?,
            (Self::Hex(_), 8) => self.scan(16)?,
            (Self::Hex(_), _) => self.scan_pair(16, true)?,
        };
        F::Bits::try_from(bits).map_err(|_| self.error("an encoding of the format"))
    }

    /// Returns an integer of type `integer`: decimal digits, or its two's
    /// complement bits in brackets.
    ///
    /// `readtest.c` reads decimal digits with `%d` and its relatives, which
    /// read the leading integer of a string: the exponent operand `1.0` of a
    /// `scalbn` line reads as 1. It reads an unsigned integer with `%u` or
    /// `%llu`, which accept a minus sign and wrap the value as `strtoul`
    /// does: the expected result `-2147483648` of a `uint32` conversion is
    /// `0x80000000`.
    ///
    /// # Errors
    ///
    /// Returns a [`FieldError`] when the value does not fit `integer`.
    pub fn integer(&self, integer: Integer) -> Result<i128, FieldError> {
        let (bits, signed): (usize, bool) = match integer {
            Integer::Int8 => (8, true),
            Integer::Int16 => (16, true),
            Integer::Int32 => (32, true),
            Integer::Int64 => (64, true),
            Integer::UInt8 => (8, false),
            Integer::UInt16 => (16, false),
            Integer::UInt32 => (32, false),
            Integer::UInt64 => (64, false),
        };
        let (minimum, maximum) = if signed {
            (-(1_i128 << (bits - 1)), (1_i128 << (bits - 1)) - 1)
        } else {
            (0, (1_i128 << bits) - 1)
        };
        let value = match self {
            Self::Text(text) => {
                let value = leading_integer(text).ok_or_else(|| self.error("an integer"))?;
                // The variable that `readtest.c` reads into has 32 or 64
                // bits.
                if !signed && value < 0 {
                    value + (1_i128 << bits.max(32))
                } else {
                    value
                }
            }
            // `readtest.c` rejects an `int8` field in brackets.
            Self::Hex(_) if integer == Integer::Int8 => return Err(self.error("decimal")),
            Self::Hex(_) => {
                // A conversion reads at most 16 digits, so the value fits.
                let raw = i128::try_from(self.scan(bits / 4)?)
                    .map_err(|_| self.error("an integer of the type"))?;
                if signed && raw > maximum {
                    raw - (1_i128 << bits)
                } else {
                    raw
                }
            }
        };
        if (minimum..=maximum).contains(&value) {
            Ok(value)
        } else {
            Err(self.error("an integer of the type"))
        }
    }

    /// Returns binary32 bits: the bits in brackets, or a decimal string
    /// rounded to nearest, as `strtof` does.
    ///
    /// # Errors
    ///
    /// Returns a [`FieldError`] when the field is not binary32.
    pub fn binary32(&self) -> Result<u32, FieldError> {
        match self {
            Self::Text(text) => text
                .parse::<f32>()
                .map(f32::to_bits)
                .map_err(|_| self.error("a binary32 string")),
            Self::Hex(_) => u32::try_from(self.scan(8)?).map_err(|_| self.error("binary32")),
        }
    }

    /// Returns binary64 bits: the bits in brackets, or a decimal string
    /// rounded to nearest, as `strtod` does.
    ///
    /// # Errors
    ///
    /// Returns a [`FieldError`] when the field is not binary64.
    pub fn binary64(&self) -> Result<u64, FieldError> {
        match self {
            Self::Text(text) => text
                .parse::<f64>()
                .map(f64::to_bits)
                .map_err(|_| self.error("a binary64 string")),
            Self::Hex(_) => u64::try_from(self.scan(16)?).map_err(|_| self.error("binary64")),
        }
    }

    /// Returns the bits of an x87 extended value in brackets: 4 digits of
    /// sign and exponent, then 16 digits of significand.
    ///
    /// `readtest.c` ignores the digits after them. Lines 126386 and 126511
    /// to 126515 have 33 and 32 digits.
    ///
    /// # Errors
    ///
    /// Returns a [`FieldError`] for a decimal field, which `readtest.in`
    /// does not use for binary80, or a field with too few digits.
    pub fn binary80(&self) -> Result<u128, FieldError> {
        self.scan_pair(4, false)
    }

    /// Returns the bits of a binary128 value in brackets: two halves of 16
    /// digits.
    ///
    /// # Errors
    ///
    /// Returns a [`FieldError`] for a decimal field, which `readtest.c`
    /// rejects, or a field with too few digits.
    pub fn binary128(&self) -> Result<u128, FieldError> {
        self.scan_pair(16, false)
    }
}

/// Reads at most `width` hexadecimal digits from the start of `text`, as a
/// `sscanf` conversion with that width does. Returns the value and the text
/// after the digits, or `None` when `text` does not start with a digit.
fn scan_hex(text: &str, width: usize) -> Option<(u128, &str)> {
    let length = text
        .bytes()
        .take(width)
        .take_while(u8::is_ascii_hexdigit)
        .count();
    if length == 0 {
        return None;
    }
    let value = u128::from_str_radix(&text[..length], 16).ok()?;
    Some((value, &text[length..]))
}

/// Reads the integer at the start of `text`, as `sscanf` with `%d` does: a
/// sign and digits, up to the first other character.
fn leading_integer(text: &str) -> Option<i128> {
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    let length = digits.bytes().take_while(u8::is_ascii_digit).count();
    let magnitude: i128 = digits[..length].parse().ok()?;
    Some(if negative { -magnitude } else { magnitude })
}

/// One test line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Line {
    /// The line number in the file, from 1.
    pub number: usize,
    /// The library function, for example `bid64_add`.
    pub function: String,
    /// The rounding direction of the function.
    pub rounding: Rounding,
    /// The operands.
    pub operands: Vec<Field>,
    /// The expected result.
    pub result: Field,
    /// The expected flags.
    pub flags: Flags,
    /// Whether the expected underflow flag needs tininess before rounding.
    pub underflow_before_only: bool,
    /// The size of `long int` that the expected result needs, from
    /// `longintsize=`.
    pub long_int_size: Option<u32>,
}

/// A line of `readtest.in` that is not a test or a comment.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParseError {
    /// The line number, from 1.
    pub line: usize,
    /// The text of the line.
    pub text: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "line {} is not a test: {}", self.line, self.text)
    }
}

impl std::error::Error for ParseError {}

/// The annotations that end the test part of a line.
const ANNOTATIONS: [&str; 4] = [
    "ulp=",
    "underflow_before_only",
    "str_prefix=",
    "longintsize=",
];

/// Parses one line. Returns `None` for an empty line or a comment.
fn parse_line(number: usize, line: &str) -> Option<Result<Line, ParseError>> {
    let content = line.split("--").next().unwrap_or_default().trim_end();
    if content.trim_start().is_empty() {
        return None;
    }
    let error = || ParseError {
        line: number,
        text: line.to_owned(),
    };
    let end = ANNOTATIONS
        .iter()
        .filter_map(|annotation| content.find(annotation))
        .min()
        .unwrap_or(content.len());
    let (body, annotations) = content.split_at(end);
    let long_int_size = match annotations.split_once("longintsize=") {
        Some((_, size)) => {
            let digits = size.split_whitespace().next().unwrap_or_default();
            match digits.parse() {
                Ok(size) => Some(size),
                Err(_) => return Some(Err(error())),
            }
        }
        None => None,
    };
    // `readtest.c` reads at most seven tokens with `sscanf`. Line 2367
    // has an eighth, the misspelled annotation `undefrlow_before_only`,
    // which it ignores.
    let tokens: Vec<&str> = body.split_whitespace().take(7).collect();
    let &[function, rounding, ref fields @ .., result, flags] = tokens.as_slice() else {
        return Some(Err(error()));
    };
    if !(1..=3).contains(&fields.len()) {
        return Some(Err(error()));
    }
    let rounding = rounding.parse().ok().and_then(Rounding::from_code);
    // `sscanf` with `%x` reads an optional `0x` prefix.
    let flags = flags.strip_prefix("0x").unwrap_or(flags);
    let flags = u32::from_str_radix(flags, 16).ok();
    let (Some(rounding), Some(flags)) = (rounding, flags) else {
        return Some(Err(error()));
    };
    // `readtest.c` does not read the closing bracket. Line 11605 lacks one.
    let field = |token: &str| match token.strip_prefix('[') {
        Some(inner) => Field::Hex(inner.strip_suffix(']').unwrap_or(inner).to_owned()),
        None => Field::Text(token.to_owned()),
    };
    let operands = fields.iter().map(|token| field(token)).collect();
    let result = field(result);
    Some(Ok(Line {
        number,
        function: function.to_owned(),
        rounding,
        operands,
        result,
        flags: Flags::from_bits(flags),
        underflow_before_only: annotations.contains("underflow_before_only"),
        long_int_size,
    }))
}

/// Parses the text of `readtest.in`.
///
/// # Errors
///
/// Returns a [`ParseError`] at the first line that is not empty, a
/// comment, or a test.
pub fn parse(text: &str) -> Result<Vec<Line>, ParseError> {
    text.lines()
        .enumerate()
        .filter_map(|(index, line)| parse_line(index + 1, line.trim_end_matches('\r')))
        .collect()
}

/// Reads and parses `readtest.in`.
///
/// # Panics
///
/// Panics when the file cannot be read or parsed. The pinned file parses.
#[must_use]
pub fn read() -> Vec<Line> {
    let text = fs::read_to_string(PATH).expect("build.rs extracts readtest.in");
    parse(&text).unwrap_or_else(|error| panic!("{PATH}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::{Field, Flags, Integer, Line, Rounding, parse};
    use crate::intel_decimal::{Bid64, Bid128};

    #[test]
    fn parses_fields_annotations_and_comments() {
        let text = "\
-- a comment\r
bid128_abs 0 [20491165061c532a,535089a5c8f9da39] [20491165061c532a535089a5c8f9da39] 00\r
\r
bid64_add 2 1.5 [31c0000000000001] 2.5 20 underflow_before_only -- a comment\r
bid128_lrint 0 [00000000000000000000000000000001] 0 20 longintsize=32 -- MinDen\r
";
        let lines = parse(text).expect("the sample parses");
        assert_eq!(lines.len(), 3);
        assert_eq!(
            lines[1],
            Line {
                number: 4,
                function: String::from("bid64_add"),
                rounding: Rounding::TowardPositive,
                operands: vec![
                    Field::Text(String::from("1.5")),
                    Field::Hex(String::from("31c0000000000001"))
                ],
                result: Field::Text(String::from("2.5")),
                flags: Flags::INEXACT,
                underflow_before_only: true,
                long_int_size: None,
            }
        );
        assert_eq!(lines[2].long_int_size, Some(32));
        assert_eq!(
            lines[0].operands[0].decimal::<Bid128>(Rounding::TiesToEven),
            Ok(0x2049_1165_061c_532a_5350_89a5_c8f9_da39)
        );
        assert_eq!(
            lines[1].operands[0].decimal::<Bid64>(Rounding::TiesToEven),
            Ok(0x31a0_0000_0000_000f)
        );
    }

    #[test]
    fn reads_integers_and_binary_fields() {
        let hex = |digits: &str| Field::Hex(digits.to_owned());
        let text = |digits: &str| Field::Text(digits.to_owned());
        assert_eq!(text("-128").integer(Integer::Int8), Ok(-128));
        assert!(text("128").integer(Integer::Int8).is_err());
        assert_eq!(hex("ffffffff").integer(Integer::Int32), Ok(-1));
        assert_eq!(hex("ffffffff").integer(Integer::UInt32), Ok(0xffff_ffff));
        assert_eq!(
            text("-2147483648").integer(Integer::UInt32),
            Ok(0x8000_0000)
        );
        assert_eq!(
            text("-1").integer(Integer::UInt64),
            Ok(0xffff_ffff_ffff_ffff)
        );
        assert!(text("-1").integer(Integer::UInt8).is_err());
        assert_eq!(hex("1234567").binary32(), Ok(0x0123_4567));
        assert_eq!(text("1.0").integer(Integer::Int32), Ok(1));
        assert_eq!(text("-9.999999e-95").integer(Integer::Int32), Ok(-9));
        assert!(text("x").integer(Integer::Int32).is_err());
        assert_eq!(
            text("+0.00000100000E0").binary64(),
            Ok(0x3eb0_c6f7_a0b5_ed8d)
        );
        assert_eq!(
            hex("3fff8000000000000000").binary80(),
            Ok(0x3fff_8000_0000_0000_0000)
        );
        assert_eq!(
            hex("407396FF33D4AAC3D365000000000000").binary80(),
            Ok(0x4073_96ff_33d4_aac3_d365)
        );
        assert!(hex("3fff").binary80().is_err());
        assert!(hex("1234").binary128().is_err());
        assert!(hex("1,2").binary128().is_err());
        assert_eq!(
            hex("1,2").decimal::<Bid128>(Rounding::TiesToEven),
            Ok((1 << 64) | 2)
        );
    }
}
