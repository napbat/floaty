//! Self-tests of the decimal oracles: the C libraries against their own test
//! files.
//!
//! - decNumber runs every case of decTest's `dd`, `dq`, and `ds` files that
//!   its `decDouble`, `decQuad`, and `decSingle` functions support.
//! - The Intel Decimal Floating-Point Math Library runs every line of its
//!   `readtest.in` for the functions that `floaty_verify::intel_decimal`
//!   wraps.
//!
//! A pass shows that the downloads, the builds, the FFI, and the parsers are
//! right. The skip counts show which cases later floaty tests can use.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use std::collections::BTreeMap;

use floaty_verify::decnumber::{Arithmetic, Double, Format, Outcome, Quad, Single, Status};
use floaty_verify::dectest::{self, Case, Expected, Operand, Operation, Test};

/// The result that decNumber gives for a case.
#[derive(Debug)]
enum Actual<B> {
    /// An encoding.
    Encoding(B),
    /// Text: a class name, `0` or `1`, or a converted string.
    Text(String),
}

/// Counts of the cases of one run.
#[derive(Default)]
struct Tally {
    passed: usize,
    failures: Vec<String>,
    skipped: BTreeMap<String, usize>,
}

impl Tally {
    fn skip(&mut self, reason: impl Into<String>) {
        *self.skipped.entry(reason.into()).or_default() += 1;
    }

    /// Prints the counts, and the failures in full.
    fn report(&self, title: &str) {
        println!(
            "{title}: {} passed, {} failed",
            self.passed,
            self.failures.len()
        );
        for (reason, count) in &self.skipped {
            println!("  skipped {count:6}: {reason}");
        }
        for failure in &self.failures {
            println!("  FAILED {failure}");
        }
    }
}

/// Runs a conversion test, `apply`, `tosci`, or `toeng`, in format `F`.
fn convert<F: Format>(test: &Test, operand: &Operand<F::Bits>) -> Option<Outcome<Actual<F::Bits>>> {
    let rounding = test.context.rounding;
    let number = match operand {
        Operand::Text(string) => F::from_string(string, rounding),
        // An explicit encoding decodes without loss and then converts in
        // the context, as a string of its value does.
        Operand::Encoding(bits) => F::from_string(&F::to_string(*bits), rounding),
        Operand::Null => return None,
    };
    let value = match test.operation {
        Operation::Apply => Actual::Encoding(number.value),
        Operation::ToSci => Actual::Text(F::to_string(number.value)),
        Operation::ToEng => Actual::Text(F::to_eng_string(number.value)),
        _ => return None,
    };
    Some(Outcome {
        value,
        status: number.status,
    })
}

/// Runs a test that needs arithmetic in format `F`.
fn compute<F: Arithmetic>(test: &Test, case: &Case<F::Bits>) -> Option<Outcome<Actual<F::Bits>>> {
    let rounding = test.context.rounding;
    let encodings: Vec<F::Bits> = case
        .operands
        .iter()
        .map(|operand| match operand {
            Operand::Encoding(bits) => Some(*bits),
            Operand::Text(_) | Operand::Null => None,
        })
        .collect::<Option<_>>()?;
    let encoding = |outcome: Outcome<F::Bits>| Outcome {
        value: Actual::Encoding(outcome.value),
        status: outcome.status,
    };
    let textual = |value: String| Outcome {
        value: Actual::Text(value),
        status: Status::NONE,
    };
    Some(match (&test.operation, encodings.as_slice()) {
        (Operation::Unary(operation), &[x]) => encoding(F::unary(*operation, x, rounding)),
        (Operation::Binary(operation), &[x, y]) => encoding(F::binary(*operation, x, y, rounding)),
        (Operation::Fma, &[x, y, z]) => encoding(F::fma(x, y, z, rounding)),
        (Operation::Class, &[x]) => textual(F::class(x).to_owned()),
        (Operation::SameQuantum, &[x, y]) => {
            textual(String::from(if F::same_quantum(x, y) { "1" } else { "0" }))
        }
        _ => return None,
    })
}

/// Returns whether the result matches the expected result of a case.
fn matches<F: Format>(expected: &Expected<F::Bits>, actual: &Actual<F::Bits>) -> bool {
    match (expected, actual) {
        (Expected::Undefined, _) => true,
        (Expected::Encoding(expected), Actual::Encoding(actual)) => expected == actual,
        // The scientific string names the sign, the coefficient, the
        // exponent, and the payload, so equal strings are the same number.
        (Expected::Number(expected), Actual::Encoding(actual)) => {
            F::to_string(*expected) == F::to_string(*actual)
        }
        (Expected::Text(expected), Actual::Text(actual)) => expected == actual,
        _ => false,
    }
}

/// Runs every test of the decTest files with `prefix` in format `F`.
/// `compute` runs the tests that are not conversions.
fn run_dectest<F: Format>(
    prefix: &str,
    compute: impl Fn(&Test, &Case<F::Bits>) -> Option<Outcome<Actual<F::Bits>>>,
) -> Tally {
    let mut tally = Tally::default();
    for path in dectest::files(prefix) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("a decTest name is text");
        for test in dectest::read(name) {
            let case = match test.encode::<F>() {
                Ok(case) => case,
                Err(error) => {
                    tally.skip(error.skip_reason());
                    continue;
                }
            };
            if case.operands.contains(&Operand::Null) {
                tally.skip("null reference operand");
                continue;
            }
            let outcome = if test.operation.converts_text() {
                convert::<F>(&test, &case.operands[0])
            } else {
                compute(&test, &case)
            };
            let Some(outcome) = outcome else {
                tally.skip(format!("{:?} has no {} function", test.operation, F::NAME));
                continue;
            };
            let conditions = test.conditions.without(Status::UNREPORTED);
            if matches::<F>(&case.expected, &outcome.value) && outcome.status == conditions {
                tally.passed += 1;
            } else {
                tally.failures.push(format!(
                    "{name}:{} {} {:?} {:?} -> {} {}: got {:?} {}",
                    test.line,
                    test.id,
                    test.operation,
                    test.operands,
                    test.result,
                    test.conditions,
                    outcome.value,
                    outcome.status,
                ));
            }
        }
    }
    tally
}

#[test]
fn dectest_decsingle() {
    let tally = run_dectest::<Single>("ds", |_, _| None);
    tally.report("decTest ds, decSingle");
    assert_eq!(tally.failures, Vec::<String>::new());
    // The archive is pinned, so the counts are exact.
    assert_eq!(tally.passed, 1177);
    assert!(tally.skipped.is_empty());
}

#[test]
fn dectest_decdouble() {
    let tally = run_dectest::<Double>("dd", compute::<Double>);
    tally.report("decTest dd, decDouble");
    assert_eq!(tally.failures, Vec::<String>::new());
    assert_eq!(tally.passed, 14_387);
    assert_eq!(tally.skipped.values().sum::<usize>(), 43);
}

#[test]
fn dectest_decquad() {
    let tally = run_dectest::<Quad>("dq", compute::<Quad>);
    tally.report("decTest dq, decQuad");
    assert_eq!(tally.failures, Vec::<String>::new());
    assert_eq!(tally.passed, 14_757);
    assert_eq!(tally.skipped.values().sum::<usize>(), 43);
}

/// The Intel library against its `readtest.in`.
mod intel {
    use floaty_verify::intel_decimal::{
        self, Bid32, Bid64, Bid128, Flags, Format, Integer, Outcome, Predicate, Rounding,
    };
    use floaty_verify::readtest::{self, Field, Line};

    use super::Tally;

    /// A result or an expected result.
    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Value {
        /// An encoding, or the bits of a binary value.
        Bits(u128),
        /// An integer, or a predicate as 0 or 1.
        Integer(i128),
        /// A string.
        Text(String),
    }

    /// The form of an expected result.
    #[derive(Clone, Copy)]
    enum Kind {
        /// An encoding of the format.
        Decimal,
        /// An encoding of the format, or another encoding that compares
        /// equal to it, as `readtest.c` accepts for `minnum` and its
        /// relatives.
        EqualDecimal,
        /// An integer of a type.
        Integer(Integer),
        /// binary32 bits.
        Binary32,
        /// binary64 bits.
        Binary64,
        /// x87 extended bits.
        Binary80,
        /// binary128 bits.
        Binary128,
        /// A string.
        Text,
    }

    /// The comparison of one line.
    struct Verdict {
        passed: bool,
        actual: Outcome<Value>,
        expected: Value,
    }

    /// The comparison of a line, or why its fields cannot be read.
    type Checked = Result<Verdict, String>;

    /// Returns an operand, by [`Line::operand`].
    fn field(line: &Line, index: usize) -> Result<&Field, String> {
        line.operand(index).map_err(|error| error.to_string())
    }

    /// Returns a decimal operand, by [`Line::decimal_operand`].
    fn operand<F: Format>(line: &Line, index: usize) -> Result<F::Bits, String> {
        line.decimal_operand::<F>(index)
            .map_err(|error| error.to_string())
    }

    fn expected<F: Format>(line: &Line, kind: Kind) -> Result<Value, String> {
        let result = &line.result;
        let value = match kind {
            Kind::Decimal | Kind::EqualDecimal => result
                .decimal::<F>(line.rounding)
                .map(|bits| Value::Bits(bits.into())),
            Kind::Integer(integer) => result.integer(integer).map(Value::Integer),
            Kind::Binary32 => result.binary32().map(|bits| Value::Bits(bits.into())),
            Kind::Binary64 => result.binary64().map(|bits| Value::Bits(bits.into())),
            Kind::Binary80 => result.binary80().map(Value::Bits),
            Kind::Binary128 => result.binary128().map(Value::Bits),
            Kind::Text => match result {
                Field::Text(text) => Ok(Value::Text(text.clone())),
                Field::Hex(_) => return Err(String::from("the result is not a string")),
            },
        };
        value.map_err(|error| error.to_string())
    }

    /// Compares an outcome of format `F` with the expected result and flags
    /// of a line.
    fn verdict<F: Format>(line: &Line, actual: Outcome<Value>, kind: Kind) -> Checked {
        let expected = expected::<F>(line, kind)?;
        let same = actual.value == expected
            || match kind {
                Kind::EqualDecimal => equal::<F>(&actual.value, &expected),
                Kind::Text => same_number::<F>(&actual.value, &expected, line.rounding),
                _ => false,
            };
        Ok(Verdict {
            passed: same && actual.flags == line.flags,
            actual,
            expected,
        })
    }

    fn equal<F: Format>(actual: &Value, expected: &Value) -> bool {
        let (&Value::Bits(actual), &Value::Bits(expected)) = (actual, expected) else {
            return false;
        };
        let (Ok(actual), Ok(expected)) = (F::Bits::try_from(actual), F::Bits::try_from(expected))
        else {
            return false;
        };
        F::compare(actual, expected, Predicate::QuietEqual).value
    }

    /// Returns whether two strings convert to the same encoding. `readtest.c`
    /// compares the results of `to_string` so: `+10E-97` matches the
    /// expected `+1.0e-96`.
    fn same_number<F: Format>(actual: &Value, expected: &Value, rounding: Rounding) -> bool {
        let (Value::Text(actual), Value::Text(expected)) = (actual, expected) else {
            return false;
        };
        F::from_string(actual, rounding).value == F::from_string(expected, rounding).value
    }

    fn bits<B: Into<u128>>(outcome: Outcome<B>) -> Outcome<Value> {
        outcome.map(|bits| Value::Bits(bits.into()))
    }

    fn integer<I: Into<i128>>(outcome: Outcome<I>) -> Outcome<Value> {
        outcome.map(|value| Value::Integer(value.into()))
    }

    /// Wraps the result of a function that reports no flags.
    fn exact(value: Value) -> Outcome<Value> {
        Outcome {
            value,
            flags: Flags::NONE,
        }
    }

    /// A function with one operand and a rounding direction.
    type Rounded1<B> = fn(B, Rounding) -> Outcome<B>;
    /// A function with two operands and a rounding direction.
    type Rounded2<B> = fn(B, B, Rounding) -> Outcome<B>;
    /// A function with one operand that reports flags.
    type Flagged1<B> = fn(B) -> Outcome<B>;
    /// A function with two operands that reports flags.
    type Flagged2<B> = fn(B, B) -> Outcome<B>;

    fn rounded_1<F: Format>(line: &Line, function: Rounded1<F::Bits>) -> Checked {
        let x = operand::<F>(line, 0)?;
        verdict::<F>(line, bits(function(x, line.rounding)), Kind::Decimal)
    }

    fn rounded_2<F: Format>(line: &Line, function: Rounded2<F::Bits>) -> Checked {
        let (x, y) = (operand::<F>(line, 0)?, operand::<F>(line, 1)?);
        verdict::<F>(line, bits(function(x, y, line.rounding)), Kind::Decimal)
    }

    fn flagged_1<F: Format>(line: &Line, function: Flagged1<F::Bits>) -> Checked {
        let x = operand::<F>(line, 0)?;
        verdict::<F>(line, bits(function(x)), Kind::Decimal)
    }

    fn flagged_2<F: Format>(line: &Line, function: Flagged2<F::Bits>, kind: Kind) -> Checked {
        let (x, y) = (operand::<F>(line, 0)?, operand::<F>(line, 1)?);
        verdict::<F>(line, bits(function(x, y)), kind)
    }

    fn relation<F: Format>(line: &Line, function: fn(F::Bits, F::Bits) -> bool) -> Checked {
        let (x, y) = (operand::<F>(line, 0)?, operand::<F>(line, 1)?);
        let value = Value::Integer(i128::from(function(x, y)));
        verdict::<F>(line, exact(value), Kind::Integer(Integer::Int32))
    }

    fn sign<F: Format>(line: &Line, function: fn(F::Bits) -> F::Bits) -> Checked {
        let x = operand::<F>(line, 0)?;
        verdict::<F>(line, exact(Value::Bits(function(x).into())), Kind::Decimal)
    }

    fn from_integer<F: Format, I: TryFrom<i128>>(
        line: &Line,
        integer: Integer,
        function: fn(I, Rounding) -> Outcome<F::Bits>,
    ) -> Checked {
        let value = line
            .integer_operand(0, integer)
            .map_err(|error| error.to_string())?;
        let value = I::try_from(value).map_err(|_| String::from("the integer does not fit"))?;
        verdict::<F>(line, bits(function(value, line.rounding)), Kind::Decimal)
    }

    /// Runs a line of a function of format `F`, named without its prefix.
    /// Returns `None` for a function that is not wrapped.
    fn run_format<F: Format>(name: &str, line: &Line) -> Option<Checked> {
        Some(match name {
            "add" => rounded_2::<F>(line, F::add),
            "sub" => rounded_2::<F>(line, F::sub),
            "mul" => rounded_2::<F>(line, F::mul),
            "div" => rounded_2::<F>(line, F::div),
            "quantize" => rounded_2::<F>(line, F::quantize),
            "sqrt" => rounded_1::<F>(line, F::sqrt),
            "round_integral_exact" => rounded_1::<F>(line, F::round_integral_exact),
            "nearbyint" => rounded_1::<F>(line, F::nearbyint),
            "round_integral_nearest_even" => {
                flagged_1::<F>(line, |x| F::round_integral(x, Rounding::TiesToEven))
            }
            "round_integral_nearest_away" => {
                flagged_1::<F>(line, |x| F::round_integral(x, Rounding::TiesToAway))
            }
            "round_integral_zero" => {
                flagged_1::<F>(line, |x| F::round_integral(x, Rounding::TowardZero))
            }
            "round_integral_negative" => {
                flagged_1::<F>(line, |x| F::round_integral(x, Rounding::TowardNegative))
            }
            "round_integral_positive" => {
                flagged_1::<F>(line, |x| F::round_integral(x, Rounding::TowardPositive))
            }
            "logb" => flagged_1::<F>(line, F::logb),
            "quantum" => flagged_1::<F>(line, F::quantum),
            "nextup" => flagged_1::<F>(line, F::nextup),
            "nextdown" => flagged_1::<F>(line, F::nextdown),
            "rem" => flagged_2::<F>(line, F::rem, Kind::Decimal),
            "minnum" => flagged_2::<F>(line, F::minnum, Kind::EqualDecimal),
            "maxnum" => flagged_2::<F>(line, F::maxnum, Kind::EqualDecimal),
            "minnum_mag" => flagged_2::<F>(line, F::minnum_mag, Kind::EqualDecimal),
            "maxnum_mag" => flagged_2::<F>(line, F::maxnum_mag, Kind::EqualDecimal),
            "totalOrder" => relation::<F>(line, F::total_order),
            "totalOrderMag" => relation::<F>(line, F::total_order_mag),
            "sameQuantum" => relation::<F>(line, F::same_quantum),
            "abs" => sign::<F>(line, F::abs),
            "negate" => sign::<F>(line, F::negate),
            "from_int32" => from_integer::<F, i32>(line, Integer::Int32, F::from_int32),
            "from_uint32" => from_integer::<F, u32>(line, Integer::UInt32, F::from_uint32),
            "from_int64" => from_integer::<F, i64>(line, Integer::Int64, F::from_int64),
            "from_uint64" => from_integer::<F, u64>(line, Integer::UInt64, F::from_uint64),
            _ => return run_format_other::<F>(name, line),
        })
    }

    /// Runs the functions of [`run_format`] that need more than one shape.
    fn run_format_other<F: Format>(name: &str, line: &Line) -> Option<Checked> {
        let rounding = line.rounding;
        let int32 = Kind::Integer(Integer::Int32);
        let x = || operand::<F>(line, 0);
        let y = || operand::<F>(line, 1);
        let run = || -> Option<Checked> {
            Some(match name {
                "fma" => (|| {
                    let outcome = F::fma(x()?, y()?, operand::<F>(line, 2)?, rounding);
                    verdict::<F>(line, bits(outcome), Kind::Decimal)
                })(),
                "scalbn" => (|| {
                    let n = line
                        .integer_operand(1, Integer::Int32)
                        .map_err(|error| error.to_string())?;
                    let n =
                        i32::try_from(n).map_err(|_| String::from("the exponent does not fit"))?;
                    verdict::<F>(line, bits(F::scalbn(x()?, n, rounding)), Kind::Decimal)
                })(),
                "ilogb" => x().and_then(|x| verdict::<F>(line, integer(F::ilogb(x)), int32)),
                "quantexp" => x().and_then(|x| verdict::<F>(line, integer(F::quantexp(x)), int32)),
                "copySign" => (|| {
                    let value = Value::Bits(F::copy_sign(x()?, y()?).into());
                    verdict::<F>(line, exact(value), Kind::Decimal)
                })(),
                "class" => x().and_then(|x| {
                    let value = Value::Integer(i128::from(F::class(x).code()));
                    verdict::<F>(line, exact(value), int32)
                }),
                "isCanonical" => x().and_then(|x| {
                    let value = Value::Integer(i128::from(F::is_canonical(x)));
                    verdict::<F>(line, exact(value), int32)
                }),
                "from_string" => (|| {
                    // `readtest.c` passes a string operand unchanged, and
                    // the string of an encoding operand.
                    let text = match field(line, 0)? {
                        Field::Text(text) => text.clone(),
                        Field::Hex(_) => F::to_string(x()?).value,
                    };
                    verdict::<F>(line, bits(F::from_string(&text, rounding)), Kind::Decimal)
                })(),
                "to_string" => x()
                    .and_then(|x| verdict::<F>(line, F::to_string(x).map(Value::Text), Kind::Text)),
                "to_binary32" => x().and_then(|x| {
                    verdict::<F>(line, bits(F::to_binary32(x, rounding)), Kind::Binary32)
                }),
                "to_binary64" => x().and_then(|x| {
                    verdict::<F>(line, bits(F::to_binary64(x, rounding)), Kind::Binary64)
                }),
                "to_binary80" => x().and_then(|x| {
                    verdict::<F>(line, bits(F::to_binary80(x, rounding)), Kind::Binary80)
                }),
                "to_binary128" => x().and_then(|x| {
                    verdict::<F>(line, bits(F::to_binary128(x, rounding)), Kind::Binary128)
                }),
                _ => return None,
            })
        };
        if let Some(checked) = run() {
            return Some(checked);
        }
        if let Some(predicate) = Predicate::ALL
            .into_iter()
            .find(|predicate| predicate.name() == name)
        {
            return Some((|| {
                let outcome = F::compare(x()?, y()?, predicate);
                verdict::<F>(line, integer(outcome), int32)
            })());
        }
        let (integer_type, direction, inexact) = readtest::to_integer_name(name)?;
        Some(x().and_then(|x| {
            let outcome = F::to_integer(x, integer_type, direction, inexact);
            verdict::<F>(line, integer(outcome), Kind::Integer(integer_type))
        }))
    }

    /// Runs a conversion from binary format `source`, for example `64`, to
    /// format `F`. Returns `None` for another source.
    fn from_binary<F: Format>(source: &str, line: &Line) -> Option<Checked> {
        let rounding = line.rounding;
        let x = match field(line, 0) {
            Ok(x) => x,
            Err(error) => return Some(Err(error)),
        };
        let outcome = match source {
            "32" => x.binary32().map(|bits| F::from_binary32(bits, rounding)),
            "64" => x.binary64().map(|bits| F::from_binary64(bits, rounding)),
            "80" => x.binary80().map(|bits| F::from_binary80(bits, rounding)),
            "128" => x.binary128().map(|bits| F::from_binary128(bits, rounding)),
            _ => return None,
        };
        Some(
            outcome
                .map_err(|error| error.to_string())
                .and_then(|outcome| verdict::<F>(line, bits(outcome), Kind::Decimal)),
        )
    }

    /// Runs a conversion between two widths.
    fn between_widths(line: &Line) -> Option<Checked> {
        let rounding = line.rounding;
        Some(match line.function.as_str() {
            "bid32_to_bid64" => operand::<Bid32>(line, 0).and_then(|x| {
                verdict::<Bid64>(line, bits(intel_decimal::bid32_to_bid64(x)), Kind::Decimal)
            }),
            "bid32_to_bid128" => operand::<Bid32>(line, 0).and_then(|x| {
                verdict::<Bid128>(line, bits(intel_decimal::bid32_to_bid128(x)), Kind::Decimal)
            }),
            "bid64_to_bid32" => operand::<Bid64>(line, 0).and_then(|x| {
                verdict::<Bid32>(
                    line,
                    bits(intel_decimal::bid64_to_bid32(x, rounding)),
                    Kind::Decimal,
                )
            }),
            "bid64_to_bid128" => operand::<Bid64>(line, 0).and_then(|x| {
                verdict::<Bid128>(line, bits(intel_decimal::bid64_to_bid128(x)), Kind::Decimal)
            }),
            "bid128_to_bid32" => operand::<Bid128>(line, 0).and_then(|x| {
                verdict::<Bid32>(
                    line,
                    bits(intel_decimal::bid128_to_bid32(x, rounding)),
                    Kind::Decimal,
                )
            }),
            "bid128_to_bid64" => operand::<Bid128>(line, 0).and_then(|x| {
                verdict::<Bid64>(
                    line,
                    bits(intel_decimal::bid128_to_bid64(x, rounding)),
                    Kind::Decimal,
                )
            }),
            _ => return None,
        })
    }

    /// The functions that are not correctly rounded. `readtest.c` accepts
    /// an error of up to a few units in the last place for them.
    const TRANSCENDENTAL: [&str; 27] = [
        "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "cos", "cosh", "erf",
        "erfc", "exp", "exp10", "exp2", "expm1", "hypot", "lgamma", "log", "log10", "log1p",
        "log2", "pow", "sin", "sinh", "tan", "tanh",
    ];

    /// Returns why a function is not wrapped.
    fn unwrapped_reason(function: &str) -> &'static str {
        let name = ["bid32_", "bid64_", "bid128_"]
            .into_iter()
            .find_map(|prefix| function.strip_prefix(prefix));
        match name {
            Some(name) if TRANSCENDENTAL.contains(&name) => {
                "not wrapped: a function that is not correctly rounded"
            }
            Some(_) => "not wrapped: a C99 or TR 24732 function, or a predicate that class covers",
            None if function.starts_with("bid64") || function.starts_with("bid128") => {
                "not wrapped: an operation on operands of another format"
            }
            None if function.contains("strtod")
                || function.contains("wcstod")
                || function == "str64" =>
            {
                "not wrapped: a C string function"
            }
            None => "not wrapped: a status flag or rounding mode function",
        }
    }

    /// Runs a line. Returns `None` for a function that is not wrapped.
    fn run_line(line: &Line) -> Option<Checked> {
        let function = line.function.as_str();
        if let Some(checked) = between_widths(line) {
            return Some(checked);
        }
        if let Some(name) = function.strip_prefix("bid32_") {
            return run_format::<Bid32>(name, line);
        }
        if let Some(name) = function.strip_prefix("bid64_") {
            return run_format::<Bid64>(name, line);
        }
        if let Some(name) = function.strip_prefix("bid128_") {
            return run_format::<Bid128>(name, line);
        }
        let dpd = |to: bool| -> Option<Checked> {
            let width = function.strip_prefix(if to { "bid_to_dpd" } else { "bid_dpd_to_bid" })?;
            Some(match (width, to) {
                ("32", true) => sign::<Bid32>(line, Bid32::to_dpd),
                ("64", true) => sign::<Bid64>(line, Bid64::to_dpd),
                ("128", true) => sign::<Bid128>(line, Bid128::to_dpd),
                ("32", false) => sign::<Bid32>(line, Bid32::from_dpd),
                ("64", false) => sign::<Bid64>(line, Bid64::from_dpd),
                ("128", false) => sign::<Bid128>(line, Bid128::from_dpd),
                _ => return None,
            })
        };
        if let Some(checked) = dpd(true).or_else(|| dpd(false)) {
            return Some(checked);
        }
        let (source, target) = function.strip_prefix("binary")?.split_once("_to_bid")?;
        match target {
            "32" => from_binary::<Bid32>(source, line),
            "64" => from_binary::<Bid64>(source, line),
            "128" => from_binary::<Bid128>(source, line),
            _ => None,
        }
    }

    #[test]
    fn readtest_lines() {
        let mut tally = Tally::default();
        let mut underflow_before_only = 0;
        for line in readtest::read() {
            let Some(checked) = run_line(&line) else {
                tally.skip(unwrapped_reason(&line.function));
                continue;
            };
            let describe = |detail: String| {
                format!(
                    "readtest.in:{} {} {} {:?} -> {:?} {:#04x}: {detail}",
                    line.number,
                    line.function,
                    line.rounding.code(),
                    line.operands,
                    line.result,
                    line.flags.bits(),
                )
            };
            match checked {
                Ok(verdict) if verdict.passed => {
                    tally.passed += 1;
                    underflow_before_only += usize::from(line.underflow_before_only);
                }
                Ok(verdict) => tally.failures.push(describe(format!(
                    "got {:?} {:#04x}, expected {:?}",
                    verdict.actual.value,
                    verdict.actual.flags.bits(),
                    verdict.expected
                ))),
                Err(error) => tally.failures.push(describe(error)),
            }
        }
        tally.report("readtest.in, Intel library");
        println!("  {underflow_before_only} passing lines need tininess before rounding");
        assert_eq!(tally.failures, Vec::<String>::new());
        // The archive is pinned, so the counts are exact.
        assert_eq!(tally.passed, 99_467);
        assert_eq!(underflow_before_only, 165);
    }
}
