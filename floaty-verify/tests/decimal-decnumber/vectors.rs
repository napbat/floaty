//! decTest's `dd`, `dq`, and `ds` files against `D64Dpd`, `D128Dpd`, and
//! `D32Dpd`.
//!
//! Two kinds of case need no string conversion:
//!
//! - `canonical x -> e`: `x` is canonical exactly when it is `e`, so floaty's
//!   `is_canonical` must give `x == e`. floaty's conversion to the same
//!   format is IEEE 754 `convertFormat`, which gives the canonical encoding
//!   of a number or an infinity, `e`. It quiets a signaling NaN and signals
//!   invalid, so for a NaN the result is the quiet NaN with the sign and
//!   the payload of `e`.
//! - `apply #x -> r`: decNumber decodes the encoding `#x` to the number
//!   `r`. floaty's `decode` of `#x` must give the sign, the coefficient, the
//!   exponent, and the NaN kind and payload of decNumber's string of `r`.
//!
//! The `ds` files have only such cases and text conversions.

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Decoded, Env, Flags};
use floaty_verify::decnumber::{self, Arithmetic, Double, Format, Quad, Single, Unary};
use floaty_verify::dectest::{self, Expected, Operand, Operation, Test};

use super::operands::Number;
use super::{
    Answer, DpdFloat, Tally, comparison, describe, direction, excluded, flags_of, ieee,
    keeps_encoding, noted, run_floaty, unmapped,
};

/// Returns whether floaty's answer matches the expected result of a vector.
fn matches<F: Format>(
    operation: &Operation,
    expected: &Expected<F::Bits>,
    actual: &Answer<F::Bits>,
) -> bool {
    match (expected, actual) {
        (Expected::Undefined, _) => true,
        (Expected::Text(text), Answer::Text(actual)) => text == actual,
        (Expected::Encoding(bits) | Expected::Number(bits), Answer::Order(order)) => {
            comparison::<F>(*bits) == Answer::Order(*order)
        }
        // A quiet operation keeps a non-canonical encoding, so it only has
        // to give the same number. Every other operation gives the
        // canonical encoding of the number.
        (Expected::Number(bits), Answer::Encoding(actual)) if keeps_encoding(operation) => {
            F::to_string(*bits) == F::to_string(*actual)
        }
        (Expected::Encoding(bits) | Expected::Number(bits), Answer::Encoding(actual)) => {
            bits == actual
        }
        _ => false,
    }
}

/// Splits a `u128` into the two limbs of a `Decoded<2>`, low limb first.
fn limbs(value: u128) -> [u64; 2] {
    let low = u64::try_from(value & u128::from(u64::MAX)).expect("the mask keeps 64 bits");
    let high = u64::try_from(value >> 64).expect("the shift keeps 64 bits");
    [low, high]
}

/// Returns the value of decNumber's scientific string of an encoding, in
/// the form of floaty's `decode`.
fn decoded<F: Format>(bits: F::Bits) -> Decoded<2> {
    let text = F::to_string(bits);
    if let Some(number) = Number::parse(&text) {
        return if number.coefficient == 0 {
            Decoded::Zero {
                negative: number.negative,
                exponent: number.exponent,
            }
        } else {
            Decoded::Finite {
                negative: number.negative,
                exponent: number.exponent,
                significand: limbs(number.coefficient),
            }
        };
    }
    let (negative, magnitude) = match text.strip_prefix('-') {
        Some(magnitude) => (true, magnitude),
        None => (false, text.as_str()),
    };
    if magnitude == "Infinity" {
        return Decoded::Infinity { negative };
    }
    let (signaling, payload) = match magnitude.strip_prefix("sNaN") {
        Some(payload) => (true, payload),
        None => (
            false,
            magnitude
                .strip_prefix("NaN")
                .expect("decNumber writes a number, an infinity, or a NaN"),
        ),
    };
    let payload = if payload.is_empty() {
        0
    } else {
        payload.parse().expect("a payload has at most 33 digits")
    };
    Decoded::Nan {
        negative,
        signaling,
        payload: limbs(payload),
    }
}

/// Returns the quiet NaN with the sign and the payload of a NaN, from
/// decNumber's string of the NaN.
fn quiet<F: Format>(nan: F::Bits) -> F::Bits {
    let text = F::to_string(nan).replacen("sNaN", "NaN", 1);
    F::from_string(&text, decnumber::Rounding::HalfEven).value
}

/// Checks a `canonical` case, as the module documentation describes.
fn check_canonical<F: Arithmetic, const W: usize>(x: F::Bits, expected: F::Bits) -> Option<String>
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let value = DpdFloat::<W>::from_bits(x);
    let canonical = x == expected;
    let (converted, flags) = value.convert_with::<DpdFloat<W>>(Env::IEEE);
    let (want, want_flags) = if F::class(expected) == "sNaN" {
        (quiet::<F>(expected), Flags::INVALID)
    } else {
        (expected, Flags::NONE)
    };
    let flags = ieee(flags);
    if value.is_canonical() == canonical && converted.to_bits() == want && flags == want_flags {
        return None;
    }
    Some(format!(
        "is_canonical gives {}, expected {canonical}; convert gives {} {flags:?}, expected {} \
         {want_flags:?}",
        value.is_canonical(),
        describe::<F>(&Answer::Encoding(converted.to_bits())),
        describe::<F>(&Answer::Encoding(want)),
    ))
}

/// Checks an `apply` case with an explicit encoding, as the module
/// documentation describes.
fn check_decode<F: Format, const W: usize>(x: F::Bits, expected: F::Bits) -> Option<String>
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let actual = DpdFloat::<W>::from_bits(x).decode::<2>();
    let want = decoded::<F>(expected);
    (actual != want).then(|| format!("decode gives {actual:?}, expected {want:?}"))
}

/// Returns the encodings of a case whose operands are all encodings.
fn encodings<B: Copy>(operands: &[Operand<B>]) -> Option<Vec<B>> {
    operands
        .iter()
        .map(|operand| match operand {
            Operand::Encoding(bits) => Some(*bits),
            Operand::Text(_) | Operand::Null => None,
        })
        .collect()
}

/// Checks a decTest vector. Returns a failure message, or why the vector
/// is skipped.
fn check_vector<F: Arithmetic, const W: usize>(
    test: &Test,
    tally: &mut Tally,
) -> Result<Option<String>, String>
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let case = test.encode::<F>().map_err(|error| error.skip_reason())?;
    if case.operands.contains(&Operand::Null) {
        return Err(String::from("null reference operand"));
    }
    let failure = |message: String| format!("{} {:?}: {message}", test.id, test.operands);
    match (
        &test.operation,
        encodings(&case.operands).as_deref(),
        &case.expected,
    ) {
        (
            Operation::Unary(Unary::Canonical),
            Some(&[x]),
            Expected::Encoding(expected) | Expected::Number(expected),
        ) => {
            assert!(
                test.conditions.is_empty(),
                "a canonical case raises nothing"
            );
            return Ok(check_canonical::<F, W>(x, *expected).map(failure));
        }
        (
            Operation::Apply,
            Some(&[x]),
            Expected::Encoding(expected) | Expected::Number(expected),
        ) => {
            assert!(
                flags_of(test.conditions).is_empty(),
                "the decoding of an encoding raises no IEEE 754 condition"
            );
            return Ok(check_decode::<F, W>(x, *expected).map(failure));
        }
        _ => {}
    }
    if let Some(reason) = unmapped(&test.operation) {
        return Err(String::from(reason));
    }
    let rounding = direction(test.context.rounding);
    let operands = encodings(&case.operands)
        .expect("only the text conversions and null references have other operands");
    if let Some(reason) = excluded::<F>(&test.operation, &operands, test.conditions) {
        return Err(String::from(reason));
    }
    if let Some(note) = noted::<F>(&test.operation, &operands) {
        tally.note(note);
    }
    let (answer, flags) = run_floaty::<F, W>(&test.operation, &operands, rounding);
    let flags = ieee(flags);
    let expected_flags = flags_of(test.conditions);
    if matches::<F>(&test.operation, &case.expected, &answer) && flags == expected_flags {
        return Ok(None);
    }
    Ok(Some(format!(
        "{} {:?} {:?} {:?} -> {} {}: floaty gives {} {flags:?}, expected {expected_flags:?}",
        test.id,
        test.operation,
        test.context.rounding,
        test.operands,
        test.result,
        test.conditions,
        describe::<F>(&answer),
    )))
}

/// Checks every vector of the decTest files with `prefix` in format `F`.
fn run_vectors<F: Arithmetic, const W: usize>(prefix: &str) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut tally = Tally::default();
    for path in dectest::files(prefix) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("a decTest name is text");
        for test in dectest::read(name) {
            match check_vector::<F, W>(&test, &mut tally) {
                Ok(None) => tally.passed += 1,
                Ok(Some(failure)) => tally.fail(|| format!("{name}:{} {failure}", test.line)),
                Err(reason) => tally.skip(reason),
            }
        }
    }
    tally
}

/// The skip reason of the text conversions.
const TEXT: &str = "apply of a string, tosci, toeng: text conversions; floaty has no strings";

/// Returns the skip counts of the `dd` or `dq` files. The two sets have
/// the same vectors except for the text conversions, the logical
/// operations, and `reduce`.
fn skips(
    text: usize,
    logical: usize,
    divide_integer: usize,
    reduce: usize,
    scale_limit: usize,
) -> [(&'static str, usize); 12] {
    [
        (
            "abs, minus, plus: decNumber rounds these; floaty has the quiet sign operations",
            160,
        ),
        (TEXT, text),
        (
            "divideint: an integer quotient, not IEEE 754",
            divide_integer,
        ),
        (
            "fma: signaling NaN factor and addend; floaty takes SoftFloat's NaN order",
            6,
        ),
        ("logical operation: floaty has none", logical),
        (
            "maxmag, minmag, nexttoward: floaty has no such operation",
            774,
        ),
        ("null reference operand", 43),
        ("reduce: not an IEEE 754 operation", reduce),
        (
            "remainder, remaindernear: decNumber's Division_impossible; floaty follows IEEE 754",
            14,
        ),
        ("scaleb: a NaN scale operand", 17),
        (
            "scaleb: a scale operand beyond decNumber's limit of 2 * (emax + p)",
            scale_limit,
        ),
        (
            "scaleb: a scale operand that is not an integer with exponent 0",
            21,
        ),
    ]
}

#[test]
fn dectest_decimal64() {
    let tally = run_vectors::<Double, 64>("dd");
    tally.report("decTest dd, D64Dpd");
    // The archive is pinned, so the counts are exact.
    tally.assert_counts(10_397, &skips(1109, 1377, 371, 133, 8), &[]);
}

#[test]
fn dectest_decimal128() {
    let tally = run_vectors::<Quad, 128>("dq");
    tally.report("decTest dq, D128Dpd");
    tally.assert_counts(10_433, &skips(1088, 1735, 372, 133, 4), &[]);
}

#[test]
fn dectest_decimal32_encodings() {
    let tally = run_vectors::<Single, 32>("ds");
    tally.report("decTest ds, D32Dpd");
    tally.assert_counts(175, &[(TEXT, 1002)], &[]);
}
