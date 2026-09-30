//! Compares floaty's operations on `D32Bid`, `D64Bid`, and `D128Bid` with
//! the `readtest.in` vectors of the Intel Decimal Floating-Point Math Library
//! 2.0 Update 2. `decimal-intel-conversions.rs` checks the conversion lines.
//!
//! Each line of a function that floaty has runs in the format of the line.
//! The test compares the result bits and the flags. These rules map floaty to
//! the library:
//!
//! - floaty runs with the behavior of [`intel_decimal::env`]: the rounding
//!   direction of the line and the NaN rule of the library.
//! - The flags compare as the five IEEE flags. The library sets its denormal
//!   flag only in a conversion from a binary format, so the test drops
//!   `DENORMAL_INPUT`, and the library must not set its denormal flag.
//! - `nearbyint`, the `round_integral_*` functions other than
//!   `round_integral_exact`, and the conversions to integers without `x` do
//!   not signal inexact. floaty always reports `INEXACT`, as
//!   `round_to_integral_with` and `to_int_with` state, so the test drops it
//!   for those functions.
//! - `lrint` and `llrint` are the `to_int64_x*` conversion in the direction
//!   of the line, and `lround` and `llround` are `to_int64_rninta`, as the
//!   library sources `bid*_lrintd.c` and `bid*_lround.c` define them.
//!   `readtest.c` skips a line whose `longintsize=` differs from the host
//!   `long`, 64 bits here, and so does this test.
//! - An invalid conversion to an integer gives the integer indefinite of
//!   [`intel_decimal::Integer::indefinite`]. floaty gives `ToInt::OutOfRange`
//!   or `ToInt::Nan` instead.
//! - `minnum` and `maxnum` of two operands that compare equal can return
//!   either operand, and `readtest.c` accepts any result that compares equal
//!   to the expected one. Where the operands and the expected result compare
//!   equal, floaty's rule decides the bits instead: see
//!   [`intel_decimal::equal_operand_choice`]. The test wants the exact bits
//!   of the line otherwise.
//! - `fma(x, y, z)` runs as `y.mul_add_with(x, z)`; see [`fused`].
//! - `quantexp` reads the exponent from `decode`. An infinity or a NaN gives
//!   `i32::MIN` and invalid, as the library source `bid*_quantexpd.c` does.
//! - `scalbln` takes a 64-bit scale. floaty takes an `i32`, and a scale
//!   beyond `2^30` acts as `2^30`, so the test clamps the scale to `i32`.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::cmp::Ordering;
use std::collections::BTreeMap;
use std::panic;

use floaty::format::{Bid, Decimal, Standard, Storage, Width};
use floaty::{Decoded, Env, Flags, Float, Integer};
use floaty_verify::intel_decimal::{
    self, Bid32, Bid64, Bid128, Class as IntelClass, EQUAL_OPERANDS, Extremum, Format, Inexact,
    Integer as IntelInteger, Outcome, Predicate, Rounding as IntelRounding, Signals, Value,
    encoding, to_integer,
};
use floaty_verify::readtest::{self, Line};

/// The storage of a format of width `W`.
type Bits<const W: usize> = <Width<W> as Storage>::Bits;

/// The BID format of width `W`.
type BidFloat<const W: usize> = Float<Decimal<Bid>, W>;

/// The form of an expected result.
#[derive(Clone, Copy, Debug)]
enum Kind {
    /// An encoding of the decimal format of the line.
    Decimal,
    /// An integer of a type.
    Integer(IntelInteger),
}

/// How a decimal result compares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Match {
    /// The bits are equal.
    Bits,
    /// The bits are equal. When the operands compare equal, the result of
    /// the minimum or the maximum follows floaty's rule.
    EqualOperands(Extremum),
}

/// The comparison of one line.
struct Verdict {
    passed: bool,
    actual: Outcome<Value>,
    expected: Outcome<Value>,
    /// The rule that changes the expected result of the line, if any.
    rule: Option<&'static str>,
}

/// The comparison of a line, or why its fields cannot be read.
type Checked = Result<Verdict, String>;

/// Counts of the lines of one run.
#[derive(Default)]
struct Tally {
    passed: BTreeMap<String, usize>,
    failures: Vec<String>,
    skipped: BTreeMap<Skip, usize>,
}

impl Tally {
    /// Prints the counts, and the failures in full.
    fn report(&self) {
        println!(
            "readtest.in, floaty: {} passed, {} failed",
            self.passed.values().sum::<usize>(),
            self.failures.len()
        );
        for (group, count) in &self.passed {
            println!("  passed  {count:6}: {group}");
        }
        for (skip, count) in &self.skipped {
            println!("  skipped {count:6}: {}", skip.reason());
        }
        for failure in &self.failures {
            println!("  FAILED {failure}");
        }
    }
}

/// Returns a decimal operand, by [`Line::decimal_operand`].
fn operand<F: Format>(line: &Line, index: usize) -> Result<F::Bits, String> {
    line.decimal_operand::<F>(index)
        .map_err(|error| error.to_string())
}

/// Returns an integer operand of type `integer`, by [`Line::integer_operand`].
fn integer_operand(line: &Line, index: usize, integer: IntelInteger) -> Result<i128, String> {
    line.integer_operand(index, integer)
        .map_err(|error| error.to_string())
}

fn expected<F: Format>(line: &Line, kind: Kind) -> Result<Value, String> {
    let result = &line.result;
    let value = match kind {
        Kind::Decimal => result
            .decimal::<F>(line.rounding)
            .map(|bits| Value::Bits(bits.into())),
        Kind::Integer(integer) => result.integer(integer).map(Value::Integer),
    };
    value.map_err(|error| error.to_string())
}

/// Compares a floaty result of a line of format `F` with its expected
/// result and flags.
fn check<F: Format>(
    line: &Line,
    value: Value,
    flags: Flags,
    signals: Signals,
    kind: Kind,
) -> Checked {
    let expected = Outcome {
        value: expected::<F>(line, kind)?,
        flags: line.flags,
    };
    let actual = Outcome {
        value,
        flags: signals.map(flags),
    };
    Ok(Verdict {
        passed: actual == expected,
        actual,
        expected,
        rule: None,
    })
}

/// Returns the behavior of the library in `rounding`.
fn env(rounding: IntelRounding) -> Env {
    intel_decimal::env(rounding)
}

/// Returns the bits of a decimal result.
fn bits<const W: usize>(value: BidFloat<W>) -> Value
where
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    Value::Bits(value.to_bits().into())
}

/// Runs an operation with one decimal operand and a decimal result.
fn unary<F, const W: usize>(
    line: &Line,
    signals: Signals,
    operation: impl FnOnce(BidFloat<W>, Env) -> (BidFloat<W>, Flags),
) -> Checked
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let x = BidFloat::<W>::from_bits(operand::<F>(line, 0)?);
    let (result, flags) = operation(x, env(line.rounding));
    check::<F>(line, bits(result), flags, signals, Kind::Decimal)
}

/// Runs an operation with two decimal operands and a decimal result.
fn binary<F, const W: usize>(
    line: &Line,
    matching: Match,
    operation: impl FnOnce(BidFloat<W>, BidFloat<W>, Env) -> (BidFloat<W>, Flags),
) -> Checked
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let (x, y) = (operand::<F>(line, 0)?, operand::<F>(line, 1)?);
    let (result, flags) = operation(
        BidFloat::<W>::from_bits(x),
        BidFloat::<W>::from_bits(y),
        env(line.rounding),
    );
    let mut verdict = check::<F>(line, bits(result), flags, Signals::Ieee, Kind::Decimal)?;
    let Match::EqualOperands(extremum) = matching else {
        return Ok(verdict);
    };
    let Value::Bits(expected) = verdict.expected.value else {
        unreachable!("a decimal result is an encoding");
    };
    let expected = encoding::<F>(expected);
    // `readtest.in` is the oracle for the value and the flags. IEEE 754 lets
    // either operand be the result, so floaty's rule, with the order of the
    // library's `totalOrder`, decides the bits.
    let equal = |a: F::Bits, b: F::Bits| F::compare(a, b, Predicate::QuietEqual).value;
    if equal(x, y) && equal(expected, x) {
        let choice = intel_decimal::equal_operand_choice::<F>(x, y, extremum);
        if choice != expected {
            verdict.expected.value = Value::Bits(choice.into());
            verdict.rule = Some(EQUAL_OPERANDS);
        }
        verdict.passed = verdict.actual == verdict.expected;
    }
    Ok(verdict)
}

/// Runs an operation with one decimal operand and an integer result, or a
/// predicate as 0 or 1.
fn property<F, const W: usize>(
    line: &Line,
    operation: impl FnOnce(BidFloat<W>, Env) -> (i128, Flags),
) -> Checked
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let x = BidFloat::<W>::from_bits(operand::<F>(line, 0)?);
    let (value, flags) = operation(x, env(line.rounding));
    let kind = Kind::Integer(IntelInteger::Int32);
    check::<F>(line, Value::Integer(value), flags, Signals::Ieee, kind)
}

/// Runs a predicate of two decimal operands.
fn relation<F, const W: usize>(
    line: &Line,
    operation: impl FnOnce(BidFloat<W>, BidFloat<W>, Env) -> (bool, Flags),
) -> Checked
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let x = BidFloat::<W>::from_bits(operand::<F>(line, 0)?);
    let y = BidFloat::<W>::from_bits(operand::<F>(line, 1)?);
    let (holds, flags) = operation(x, y, env(line.rounding));
    let kind = Kind::Integer(IntelInteger::Int32);
    check::<F>(
        line,
        Value::Integer(i128::from(holds)),
        flags,
        Signals::Ieee,
        kind,
    )
}

/// Evaluates a comparison predicate with floaty.
fn compare<const W: usize>(
    x: BidFloat<W>,
    y: BidFloat<W>,
    predicate: Predicate,
    env: Env,
) -> (bool, Flags)
where
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let (order, flags) = if predicate.is_signaling() {
        x.compare_signaling_with(y, env)
    } else {
        x.compare_quiet_with(y, env)
    };
    (predicate.holds(order), flags)
}

/// Returns the library class of a value.
fn class<const W: usize>(x: BidFloat<W>) -> IntelClass
where
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    IntelClass::from_floaty(x.classify(), x.is_sign_negative())
        .expect("a decimal value has a class of the library")
}

/// Returns the exponent of a finite value, as `quantexp` does.
fn quantum_exponent<const W: usize>(x: BidFloat<W>) -> (i128, Flags)
where
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    match x.decode::<2>() {
        Decoded::Zero { exponent, .. } | Decoded::Finite { exponent, .. } => {
            (i128::from(exponent), Flags::NONE)
        }
        _ => (i128::from(i32::MIN), Flags::INVALID),
    }
}

/// Runs a conversion to an integer type.
fn convert_to_integer<F, const W: usize>(
    line: &Line,
    integer: IntelInteger,
    rounding: IntelRounding,
    inexact: Inexact,
) -> Checked
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let x = BidFloat::<W>::from_bits(operand::<F>(line, 0)?);
    let env = env(line.rounding).with_rounding(rounding.into());
    let (value, flags) = match integer {
        IntelInteger::Int8 => to_integer::<i8, W>(x, integer, env),
        IntelInteger::Int16 => to_integer::<i16, W>(x, integer, env),
        IntelInteger::Int32 => to_integer::<i32, W>(x, integer, env),
        IntelInteger::Int64 => to_integer::<i64, W>(x, integer, env),
        IntelInteger::UInt8 => to_integer::<u8, W>(x, integer, env),
        IntelInteger::UInt16 => to_integer::<u16, W>(x, integer, env),
        IntelInteger::UInt32 => to_integer::<u32, W>(x, integer, env),
        IntelInteger::UInt64 => to_integer::<u64, W>(x, integer, env),
    };
    let signals = match inexact {
        Inexact::Ignored => Signals::NoInexact,
        Inexact::Signaled => Signals::Ieee,
    };
    check::<F>(
        line,
        Value::Integer(value),
        flags,
        signals,
        Kind::Integer(integer),
    )
}

/// Runs a conversion from an integer type.
fn convert_from_integer<F, I, const W: usize>(line: &Line, integer: IntelInteger) -> Checked
where
    F: Format<Bits = Bits<W>>,
    I: Integer + TryFrom<i128>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let value = integer_operand(line, 0, integer)?;
    let value = I::try_from(value).map_err(|_| String::from("the integer does not fit"))?;
    let (result, flags) = BidFloat::<W>::from_int_with(value, env(line.rounding));
    check::<F>(line, bits(result), flags, Signals::Ieee, Kind::Decimal)
}

/// Runs a scaling by a power of 10, with a scale of type `integer`.
fn scale<F, const W: usize>(line: &Line, integer: IntelInteger) -> Checked
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let x = BidFloat::<W>::from_bits(operand::<F>(line, 0)?);
    let n = integer_operand(line, 1, integer)?;
    let n = i32::try_from(n).unwrap_or(if n < 0 { i32::MIN } else { i32::MAX });
    let (result, flags) = x.scale_b_with(n, env(line.rounding));
    check::<F>(line, bits(result), flags, Signals::Ieee, Kind::Decimal)
}

/// Returns the integer type, direction, and inexact rule of a name such as
/// `to_uint16_xfloor`, by [`readtest::to_integer_name`], or of `lrint`,
/// `llrint`, `lround`, and `llround` in the direction of the line.
fn to_integer_name(name: &str, line: &Line) -> Option<(IntelInteger, IntelRounding, Inexact)> {
    match name {
        "lrint" | "llrint" => Some((IntelInteger::Int64, line.rounding, Inexact::Signaled)),
        "lround" | "llround" => Some((
            IntelInteger::Int64,
            IntelRounding::TiesToAway,
            Inexact::Ignored,
        )),
        _ => readtest::to_integer_name(name),
    }
}

/// Runs a line of a function of format `F`, named without its prefix.
/// Returns `None` for a function that floaty does not have.
fn run_format<F, const W: usize>(name: &str, line: &Line) -> Option<Checked>
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let quiet = Signals::NoInexact;
    let fixed = |rounding: IntelRounding| {
        move |x: BidFloat<W>, env: Env| x.round_to_integral_with(env.with_rounding(rounding.into()))
    };
    Some(match name {
        "add" => binary::<F, W>(line, Match::Bits, BidFloat::add_with),
        "sub" => binary::<F, W>(line, Match::Bits, BidFloat::sub_with),
        "mul" => binary::<F, W>(line, Match::Bits, BidFloat::mul_with),
        "div" => binary::<F, W>(line, Match::Bits, BidFloat::div_with),
        "quantize" => binary::<F, W>(line, Match::Bits, BidFloat::quantize_with),
        "rem" => binary::<F, W>(line, Match::Bits, BidFloat::remainder_with),
        "fmod" => binary::<F, W>(line, Match::Bits, BidFloat::truncated_remainder_with),
        "minnum" => binary::<F, W>(
            line,
            Match::EqualOperands(Extremum::Minimum),
            BidFloat::min_num_with,
        ),
        "maxnum" => binary::<F, W>(
            line,
            Match::EqualOperands(Extremum::Maximum),
            BidFloat::max_num_with,
        ),
        "sqrt" => unary::<F, W>(line, Signals::Ieee, BidFloat::sqrt_with),
        "logb" => unary::<F, W>(line, Signals::Ieee, BidFloat::log_b_with),
        "quantum" => unary::<F, W>(line, Signals::Ieee, BidFloat::quantum_with),
        "nextup" => unary::<F, W>(line, Signals::Ieee, BidFloat::next_up_with),
        "nextdown" => unary::<F, W>(line, Signals::Ieee, BidFloat::next_down_with),
        "round_integral_exact" => {
            unary::<F, W>(line, Signals::Ieee, BidFloat::round_to_integral_with)
        }
        "nearbyint" => unary::<F, W>(line, quiet, BidFloat::round_to_integral_with),
        "round_integral_nearest_even" => {
            unary::<F, W>(line, quiet, fixed(IntelRounding::TiesToEven))
        }
        "round_integral_nearest_away" => {
            unary::<F, W>(line, quiet, fixed(IntelRounding::TiesToAway))
        }
        "round_integral_zero" => unary::<F, W>(line, quiet, fixed(IntelRounding::TowardZero)),
        "round_integral_negative" => {
            unary::<F, W>(line, quiet, fixed(IntelRounding::TowardNegative))
        }
        "round_integral_positive" => {
            unary::<F, W>(line, quiet, fixed(IntelRounding::TowardPositive))
        }
        "abs" => unary::<F, W>(line, Signals::Ieee, |x, _| (x.abs(), Flags::NONE)),
        "negate" => unary::<F, W>(line, Signals::Ieee, |x, _| (-x, Flags::NONE)),
        "copy" => unary::<F, W>(line, Signals::Ieee, |x, _| (x, Flags::NONE)),
        "copySign" => binary::<F, W>(line, Match::Bits, |x, y, _| (x.copy_sign(y), Flags::NONE)),
        "fma" => fused::<F, W>(line),
        _ => return run_format_other::<F, W>(name, line),
    })
}

/// Runs the functions of [`run_format`] with results that are not decimal
/// encodings.
fn run_format_other<F, const W: usize>(name: &str, line: &Line) -> Option<Checked>
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let flag = |test: fn(BidFloat<W>) -> bool| {
        move |x: BidFloat<W>, _: Env| (i128::from(test(x)), Flags::NONE)
    };
    Some(match name {
        "scalbn" | "ldexp" => scale::<F, W>(line, IntelInteger::Int32),
        "scalbln" => scale::<F, W>(line, IntelInteger::Int64),
        "quantexp" | "llquantexp" => property::<F, W>(line, |x, _| quantum_exponent(x)),
        "class" => property::<F, W>(line, |x, _| (i128::from(class(x).code()), Flags::NONE)),
        "isCanonical" => property::<F, W>(line, flag(BidFloat::is_canonical)),
        "isSigned" => property::<F, W>(line, flag(BidFloat::is_sign_negative)),
        "isNormal" => property::<F, W>(line, flag(BidFloat::is_normal)),
        "isSubnormal" => property::<F, W>(line, flag(BidFloat::is_subnormal)),
        "isFinite" => property::<F, W>(line, flag(BidFloat::is_finite)),
        "isZero" => property::<F, W>(line, flag(BidFloat::is_zero)),
        "isInf" => property::<F, W>(line, flag(BidFloat::is_infinite)),
        "isSignaling" => property::<F, W>(line, flag(BidFloat::is_signaling_nan)),
        "isNaN" => property::<F, W>(line, flag(BidFloat::is_nan)),
        "radix" => property::<F, W>(line, |_, _| (i128::from(BidFloat::<W>::RADIX), Flags::NONE)),
        "sameQuantum" => relation::<F, W>(line, |x, y, _| (x.same_quantum(y), Flags::NONE)),
        "totalOrder" => relation::<F, W>(line, |x, y, _| {
            (x.total_cmp(y) != Ordering::Greater, Flags::NONE)
        }),
        "totalOrderMag" => relation::<F, W>(line, |x, y, _| {
            (x.abs().total_cmp(y.abs()) != Ordering::Greater, Flags::NONE)
        }),
        "from_int32" => convert_from_integer::<F, i32, W>(line, IntelInteger::Int32),
        "from_uint32" => convert_from_integer::<F, u32, W>(line, IntelInteger::UInt32),
        "from_int64" => convert_from_integer::<F, i64, W>(line, IntelInteger::Int64),
        "from_uint64" => convert_from_integer::<F, u64, W>(line, IntelInteger::UInt64),
        _ => {
            if let Some(predicate) = Predicate::ALL
                .into_iter()
                .find(|predicate| predicate.name() == name)
            {
                return Some(relation::<F, W>(line, |x, y, env| {
                    compare(x, y, predicate, env)
                }));
            }
            let (integer, rounding, inexact) = to_integer_name(name, line)?;
            convert_to_integer::<F, W>(line, integer, rounding, inexact)
        }
    })
}

/// Runs `fma`: `x * y + z`, rounded once.
///
/// The library takes the NaN of `y`, then of `z`, then of `x`. floaty takes
/// the NaN of the first factor, then of the second, then of the addend, so
/// the test passes the factors as `y * x`. [`fma_nan_order`] finds the
/// lines where the orders still differ.
fn fused<F, const W: usize>(line: &Line) -> Checked
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let x = BidFloat::<W>::from_bits(operand::<F>(line, 0)?);
    let y = BidFloat::<W>::from_bits(operand::<F>(line, 1)?);
    let z = BidFloat::<W>::from_bits(operand::<F>(line, 2)?);
    let (result, flags) = y.mul_add_with(x, z, env(line.rounding));
    check::<F>(line, bits(result), flags, Signals::Ieee, Kind::Decimal)
}

/// Returns whether an `fma` line has NaN operands `x` and `z` and a number
/// `y`. The library then returns the NaN of `z`, and floaty the NaN of the
/// factor `x`: `FusedNanOrder::ProductFirst` selects the NaN of the factors
/// before the NaN of the addend.
fn fma_nan_order<F: Format>(line: &Line) -> Result<bool, String> {
    let nan = |index| -> Result<bool, String> {
        let class = F::class(operand::<F>(line, index)?);
        Ok(matches!(
            class,
            IntelClass::QuietNan | IntelClass::SignalingNan
        ))
    };
    Ok(nan(0)? && !nan(1)? && nan(2)?)
}

/// Why a line does not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Skip {
    /// A function that is not correctly rounded.
    Transcendental,
    /// A function on operands of more than one format.
    MixedFormats,
    /// A string conversion. floaty defers decimal strings.
    Strings,
    /// `minnum_mag` or `maxnum_mag`.
    MagnitudeMinMax,
    /// Another C99 or TR 24732 function.
    OtherFunction,
    /// A function of the status flags, the rounding mode, or conformance.
    Environment,
    /// A line for a 32-bit `long`.
    LongIntSize,
    /// An `fma` line whose NaN operand order floaty cannot follow.
    FmaNanOrder,
    /// A conversion between formats.
    Conversion,
}

impl Skip {
    fn reason(self) -> &'static str {
        match self {
            Self::Transcendental => "a function that is not correctly rounded, which floaty defers",
            Self::MixedFormats => "an operation on operands of different formats",
            Self::Strings => "a string conversion, which floaty defers",
            Self::MagnitudeMinMax => "minnum_mag or maxnum_mag, which floaty does not have",
            Self::OtherFunction => {
                "a C99 or TR 24732 function that floaty does not have: fdim, frexp, ilogb, \
                 inf, modf, nextafter, nexttoward"
            }
            Self::Environment => {
                "a status flag, rounding mode, or conformance query function of the library"
            }
            Self::LongIntSize => "longintsize=32, which readtest.c skips on a 64-bit long host",
            Self::Conversion => "a conversion between formats: decimal-intel-conversions.rs",
            Self::FmaNanOrder => {
                "fma with NaN x and z and a number y: the library returns the NaN of z, floaty \
                 the NaN of the factors first (FusedNanOrder::ProductFirst, SoftFloat's order)"
            }
        }
    }
}

/// The functions that are not correctly rounded.
const TRANSCENDENTAL: [&str; 27] = [
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "cos", "cosh", "erf",
    "erfc", "exp", "exp10", "exp2", "expm1", "hypot", "lgamma", "log", "log10", "log1p", "log2",
    "pow", "sin", "sinh", "tan", "tanh",
];

/// Returns why a line of a function that floaty does not have is skipped.
fn skip_reason(function: &str) -> Skip {
    let name = ["bid32_", "bid64_", "bid128_"]
        .into_iter()
        .find_map(|prefix| function.strip_prefix(prefix));
    match name {
        Some(name) if TRANSCENDENTAL.contains(&name) || name == "tgamma" => Skip::Transcendental,
        Some("from_string" | "to_string" | "nan") => Skip::Strings,
        Some("minnum_mag" | "maxnum_mag") => Skip::MagnitudeMinMax,
        Some(_) => Skip::OtherFunction,
        None if function.starts_with("bid64") || function.starts_with("bid128") => {
            Skip::MixedFormats
        }
        None if function.contains("strtod")
            || function.contains("wcstod")
            || function == "str64" =>
        {
            Skip::Strings
        }
        None => Skip::Environment,
    }
}

/// Returns whether a function converts between two formats: two decimal
/// widths, BID and DPD, or a decimal and a binary format.
fn is_conversion(function: &str) -> bool {
    let decimal = |name: &str| {
        ["32", "64", "128"]
            .into_iter()
            .any(|width| name == format!("bid{width}"))
    };
    function.starts_with("bid_to_dpd")
        || function.starts_with("bid_dpd_to_bid")
        || function.split_once("_to_").is_some_and(|(from, to)| {
            (decimal(from) || from.starts_with("binary"))
                && (decimal(to) || to.starts_with("binary"))
        })
}

/// Runs a line, or returns why it does not run.
fn run_line(line: &Line) -> Result<Checked, Skip> {
    if line.long_int_size.is_some_and(|size| size != 64) {
        return Err(Skip::LongIntSize);
    }
    let function = line.function.as_str();
    let order = match function {
        "bid32_fma" => fma_nan_order::<Bid32>(line),
        "bid64_fma" => fma_nan_order::<Bid64>(line),
        "bid128_fma" => fma_nan_order::<Bid128>(line),
        _ => Ok(false),
    };
    match order {
        Ok(true) => return Err(Skip::FmaNanOrder),
        Ok(false) => {}
        Err(error) => return Ok(Err(error)),
    }
    if is_conversion(function) {
        return Err(Skip::Conversion);
    }
    let (width, name) = function
        .strip_prefix("bid")
        .and_then(|rest| rest.split_once('_'))
        .ok_or_else(|| skip_reason(function))?;
    let checked = match width {
        "32" => run_format::<Bid32, 32>(name, line),
        "64" => run_format::<Bid64, 64>(name, line),
        "128" => run_format::<Bid128, 128>(name, line),
        _ => None,
    };
    checked.ok_or_else(|| skip_reason(function))
}

/// Returns the group of a function for the counts: its name without the
/// width.
fn group(function: &str) -> String {
    function
        .replace("bid128", "bidN")
        .replace("bid64", "bidN")
        .replace("bid32", "bidN")
}

/// Runs a line, and reports a panic of floaty as a failure.
fn run_guarded(line: &Line) -> Result<Checked, Skip> {
    panic::catch_unwind(|| run_line(line)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("a panic without a message");
        Ok(Err(format!("floaty panics: {message}")))
    })
}

#[test]
fn readtest_lines() {
    let mut tally = Tally::default();
    // A panic of floaty is a failure of its line. The report prints it, so
    // the default hook stays quiet during the run.
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    for line in readtest::read() {
        let checked = match run_guarded(&line) {
            Ok(checked) => checked,
            Err(skip) => {
                *tally.skipped.entry(skip).or_default() += 1;
                continue;
            }
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
                let group = match verdict.rule {
                    Some(rule) => format!("{} by {rule}", group(&line.function)),
                    None => group(&line.function),
                };
                *tally.passed.entry(group).or_default() += 1;
            }
            Ok(verdict) => tally.failures.push(describe(format!(
                "floaty {:x?} {:#04x}, expected {:x?} {:#04x}",
                verdict.actual.value,
                verdict.actual.flags.bits(),
                verdict.expected.value,
                verdict.expected.flags.bits(),
            ))),
            Err(error) => tally.failures.push(describe(error)),
        }
    }
    panic::set_hook(hook);
    tally.report();
    assert!(
        tally.failures.is_empty(),
        "{} lines fail",
        tally.failures.len()
    );
    // The archive is pinned, so the counts are exact.
    assert_eq!(tally.passed.values().sum::<usize>(), 65_528);
    let skipped = [
        (Skip::Transcendental, 7_563),
        (Skip::MixedFormats, 1_977),
        (Skip::Strings, 427),
        (Skip::MagnitudeMinMax, 498),
        (Skip::OtherFunction, 1_986),
        (Skip::Environment, 217),
        (Skip::LongIntSize, 6_652),
        (Skip::FmaNanOrder, 64),
        (Skip::Conversion, 41_505),
    ];
    assert_eq!(tally.skipped, BTreeMap::from(skipped));
    let decided = |group: &str| {
        tally
            .passed
            .get(&format!("{group} by {EQUAL_OPERANDS}"))
            .copied()
    };
    assert_eq!(decided("bidN_minnum"), Some(15));
    assert_eq!(decided("bidN_maxnum"), Some(14));
}
