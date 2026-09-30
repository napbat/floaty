//! Compares comparison, remainder, round to integral, and the conversions
//! between floats and 32- and 64-bit integers with Berkeley TestFloat, for
//! binary16, binary32, binary64, x87 extended precision, and binary128.
//!
//! The ARM generator checks each rounding direction of TestFloat. The
//! default-NaN, 8086, and 8086-SSE generators check the other NaN rules at
//! the default behavior.
//! x87 extended precision also runs the remainder and round to integral at
//! precision control 32 and 64 bits. SoftFloat and floaty ignore precision
//! control for these operations, so the results do not change.
//!
//! A conversion to an integer gives a `ToInt`. Each SoftFloat specialization
//! encodes an out-of-range value and a NaN as its instruction set does. The
//! tests map `ToInt` to the ARM and to the x86 integers, as a consumer does.
//! TestFloat 3e's `testfloat_gen` has no `_r_minMag` conversions. The
//! `-rminMag` runs of the conversions check the same results.
//!
//! Comparisons and the remainder use TestFloat level 1. The operations with
//! one operand use level 2.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::cmp::Ordering;
use core::fmt::Debug;

use floaty::format::Standard;
use floaty::{Binary, Env, Flags, Float, Integer, Rounding, ToInt, X87};
use floaty_verify::testfloat::{
    self, Exactness, Generator, Level, Options, PRECISION_CONTROL, PrecisionControl, ROUNDINGS,
    Run, fields, flag_bits, quiet_extended_nan,
};

/// The storage of a format that TestFloat covers. It converts to and from the
/// `u128` of a hexadecimal field.
trait Field: TryFrom<u128, Error: Debug> + Into<u128> {}

impl<T: TryFrom<u128, Error: Debug> + Into<u128>> Field for T {}

/// The kind of a format under test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    /// An IEEE 754 interchange format.
    Interchange,
    /// x87 extended precision.
    Extended,
}

/// A format under test: its TestFloat name and its kind.
#[derive(Clone, Copy, Debug)]
struct Format {
    name: &'static str,
    family: Family,
}

impl Format {
    /// Returns the precision control settings. Only x87 extended precision
    /// has precision control.
    fn precisions(self) -> &'static [PrecisionControl] {
        match self.family {
            Family::Interchange => &[PrecisionControl::FULL],
            Family::Extended => &PRECISION_CONTROL,
        }
    }

    /// Returns the generators of the NaN rules other than the ARM rule whose
    /// rule covers the format.
    fn nan_generators(self) -> impl Iterator<Item = Generator> {
        Generator::OTHER_NAN_RULES
            .into_iter()
            .filter(move |generator| {
                self.family == Family::Interchange || generator.covers_extended
            })
    }

    /// Returns the result bits that floaty must give for the result field of
    /// a case. SoftFloat's ARM specialization does not quiet a signaling x87
    /// NaN; [`quiet_extended_nan`] corrects the expected NaN.
    fn expected(self, bits: u128) -> u128 {
        match self.family {
            Family::Interchange => bits,
            Family::Extended => quiet_extended_nan(bits),
        }
    }
}

/// Returns the runs of `generator` in every rounding direction with each
/// exactness, at the precision control `precision`.
fn directions(
    generator: Generator,
    exactness: &[Exactness],
    precision: PrecisionControl,
) -> Vec<Run> {
    exactness
        .iter()
        .flat_map(|&exactness| {
            ROUNDINGS.map(|(rounding, _)| {
                let options = Options {
                    rounding: Some(rounding),
                    exactness: Some(exactness),
                    precision: Some(precision),
                    ..Options::default()
                };
                Run::new(generator, options)
            })
        })
        .collect()
}

/// Returns the runs of an operation on floats of `format`: the ARM generator
/// in every rounding direction with each exactness, and the other NaN
/// generators at the default behavior, at every precision control setting of
/// the format.
fn float_runs(format: Format, exactness: &[Exactness]) -> Vec<Run> {
    let mut runs = Vec::new();
    for &precision in format.precisions() {
        runs.extend(directions(Generator::ARM, exactness, precision));
        for generator in format.nan_generators() {
            let options = Options {
                rounding: Some(Rounding::TiesToEven),
                exactness: Some(Exactness::Exact),
                precision: Some(precision),
                ..Options::default()
            };
            runs.push(Run::new(generator, options));
        }
    }
    runs
}

/// Runs `function` at TestFloat `level` for each run, and calls `compare` on
/// each case.
fn check(function: &str, level: Level, runs: &[Run], compare: impl Fn(&str, &Run)) {
    for run in runs {
        let count = testfloat::run(run.generator, level, function, &run.options, None, |line| {
            compare(line, run);
        });
        assert!(count > 0, "{function} {:?} gave test cases", run.options);
    }
}

/// Checks floaty's result bits and flags against the expected result and
/// flags of a case.
fn assert_case(function: &str, line: &str, run: &Run, ours: (u128, Flags), expected: [u128; 2]) {
    let (bits, flags) = ours;
    let context = format!("{function} {:?} {:?} {line}", run.options, run.env);
    assert_eq!(bits, expected[0], "{context}: result");
    assert_eq!(
        flag_bits(run.exactness.reported(flags)),
        expected[1],
        "{context}: flags {flags:?}"
    );
}

/// Makes a value of the format from a hexadecimal field.
fn decode<S: Standard<W, Bits: Field>, const W: usize>(bits: u128) -> Float<S, W> {
    let bits = bits
        .try_into()
        .expect("TestFloat gives an encoding of the format under test");
    Float::from_bits(bits)
}

/// The comparison form that a TestFloat predicate uses.
#[derive(Clone, Copy, Debug)]
enum Form {
    /// Signals invalid for a signaling NaN only.
    Quiet,
    /// Signals invalid for every NaN.
    Signaling,
}

/// The relation that a TestFloat predicate tests.
#[derive(Clone, Copy, Debug)]
enum Relation {
    Equal,
    LessOrEqual,
    Less,
}

impl Relation {
    /// Returns `true` when `order` satisfies the relation. An unordered pair
    /// satisfies no relation.
    fn holds(self, order: Option<Ordering>) -> bool {
        match self {
            Self::Equal => order == Some(Ordering::Equal),
            Self::LessOrEqual => matches!(order, Some(Ordering::Less | Ordering::Equal)),
            Self::Less => order == Some(Ordering::Less),
        }
    }

    /// Evaluates the relation with the `PartialEq` and `PartialOrd`
    /// operators.
    fn by_operator<T: Copy + PartialOrd>(self, a: T, b: T) -> bool {
        match self {
            Self::Equal => a == b,
            Self::LessOrEqual => a <= b,
            Self::Less => a < b,
        }
    }
}

/// The TestFloat comparison predicates: the function suffix, the form, and
/// the relation.
const PREDICATES: [(&str, Form, Relation); 6] = [
    ("eq", Form::Quiet, Relation::Equal),
    ("le", Form::Signaling, Relation::LessOrEqual),
    ("lt", Form::Signaling, Relation::Less),
    ("eq_signaling", Form::Signaling, Relation::Equal),
    ("le_quiet", Form::Quiet, Relation::LessOrEqual),
    ("lt_quiet", Form::Quiet, Relation::Less),
];

/// Checks the six comparison predicates of a format with the default mode.
/// The operators must give the value of each predicate, because a signaling
/// predicate has the value of its quiet form.
fn check_comparisons<S: Standard<W, Bits: Field>, const W: usize>(format: Format) {
    let run = Run::new(Generator::ARM, Options::default());
    for (suffix, form, relation) in PREDICATES {
        let function = format!("{}_{suffix}", format.name);
        check(
            &function,
            Level::One,
            core::slice::from_ref(&run),
            |line, run| {
                let [a, b, result, flags] = fields::<4>(line);
                let (a, b) = (decode::<S, W>(a), decode::<S, W>(b));
                let (order, ours_flags) = match form {
                    Form::Quiet => a.compare_quiet_with(b, run.env),
                    Form::Signaling => a.compare_signaling_with(b, run.env),
                };
                let ours = u128::from(relation.holds(order));
                assert_case(&function, line, run, (ours, ours_flags), [result, flags]);
                assert_eq!(
                    u128::from(relation.by_operator(a, b)),
                    result,
                    "{function} {line}: operator"
                );
            },
        );
    }
}

/// Checks the remainder of a format. The remainder is exact, so every
/// rounding direction gives the same result.
fn check_remainder<S: Standard<W, Bits: Field>, const W: usize>(format: Format) {
    let function = format!("{}_rem", format.name);
    let runs = float_runs(format, &[Exactness::Exact]);
    check(&function, Level::One, &runs, |line, run| {
        let [a, b, result, flags] = fields::<4>(line);
        let (ours, ours_flags) = decode::<S, W>(a).remainder_with(decode(b), run.env);
        let ours = (ours.to_bits().into(), ours_flags);
        assert_case(&function, line, run, ours, [format.expected(result), flags]);
    });
}

/// Checks round to integral of a format, as `roundToIntegralExact` with
/// `-exact` and as the operations without inexact with `-notexact`.
/// The runs of the default mode also check `round_to_integral`, which returns
/// no flags and takes a host path where the build has one.
fn check_round_to_integral<S: Standard<W, Bits: Field>, const W: usize>(format: Format) {
    let function = format!("{}_roundToInt", format.name);
    let runs = float_runs(format, &[Exactness::Exact, Exactness::NotExact]);
    let defaults = core::cell::Cell::new(0_usize);
    check(&function, Level::Two, &runs, |line, run| {
        let [a, result, flags] = fields::<3>(line);
        let (ours, ours_flags) = decode::<S, W>(a).round_to_integral_with(run.env);
        let ours = (ours.to_bits().into(), ours_flags);
        assert_case(&function, line, run, ours, [format.expected(result), flags]);
        if run.env == Env::IEEE {
            let default: u128 = decode::<S, W>(a).round_to_integral().to_bits().into();
            assert_eq!(
                default,
                format.expected(result),
                "{function} {line}: default mode"
            );
            defaults.set(defaults.get() + 1);
        }
    });
    assert!(
        defaults.get() > 0,
        "{function}: the default mode rounded test cases"
    );
}

/// Returns the TestFloat name of an integer type, such as `ui32`.
fn integer_name<I: Integer>() -> String {
    let sign = if I::SIGNED { "i" } else { "ui" };
    format!("{sign}{}", I::BITS)
}

/// Returns the smallest and the largest value of an integer type of at most
/// 64 bits.
fn integer_range<I: Integer>() -> (i128, i128) {
    if I::SIGNED {
        (-(1 << (I::BITS - 1)), (1 << (I::BITS - 1)) - 1)
    } else {
        (0, (1 << I::BITS) - 1)
    }
}

/// Returns the two's complement bits of a value of an integer type of at
/// most 64 bits, as TestFloat writes them.
fn integer_bits<I: Integer>(value: i128) -> u128 {
    u128::try_from(value.rem_euclid(1 << I::BITS)).expect("a Euclidean remainder is not negative")
}

/// Reads a hexadecimal field that holds the two's complement bits of an
/// integer type of at most 64 bits.
fn integer_from_field<I: Integer + TryFrom<i128, Error: Debug>>(bits: u128) -> I {
    let value = i128::try_from(bits).expect("TestFloat gives an integer of at most 64 bits");
    let value = if I::SIGNED && value >> (I::BITS - 1) == 1 {
        value - (1 << I::BITS)
    } else {
        value
    };
    I::try_from(value).expect("TestFloat gives a value of the integer type under test")
}

/// How an instruction set encodes the special results of a conversion to an
/// integer.
#[derive(Clone, Copy, Debug)]
enum IntegerEncoding {
    /// SoftFloat's ARM-VFPv2 specialization: an out-of-range value saturates
    /// to the end of the range on its side, and a NaN gives zero.
    Saturating,
    /// SoftFloat's 8086-SSE specialization: an out-of-range value and a NaN
    /// give the integer indefinite. That is the smallest signed value, or all
    /// ones for an unsigned type.
    Indefinite,
}

impl IntegerEncoding {
    /// Returns the two's complement bits that the instruction set gives for a
    /// conversion result.
    fn bits<I: Integer + Into<i128>>(self, result: ToInt<I>) -> u128 {
        let (smallest, largest) = integer_range::<I>();
        let value = match (self, result) {
            (_, ToInt::Value(value)) => value.into(),
            (Self::Saturating, ToInt::OutOfRange { negative: true }) => smallest,
            (Self::Saturating, ToInt::OutOfRange { negative: false }) => largest,
            (Self::Saturating, ToInt::Nan) => 0,
            (Self::Indefinite, ToInt::OutOfRange { .. } | ToInt::Nan) => {
                if I::SIGNED {
                    smallest
                } else {
                    largest
                }
            }
        };
        integer_bits::<I>(value)
    }
}

/// Checks the conversion of a format to the integer type `I`. The ARM
/// generator checks every direction with and without inexact. The 8086-SSE
/// generator checks every direction with the x86 encoding of the special
/// results, which is the only difference.
fn check_to_integer<S, const W: usize, I>(format: Format)
where
    S: Standard<W, Bits: Field>,
    I: Integer + Into<i128>,
{
    let function = format!("{}_to_{}", format.name, integer_name::<I>());
    let exactness = [Exactness::Exact, Exactness::NotExact];
    let arm = directions(Generator::ARM, &exactness, PrecisionControl::FULL);
    let sse = directions(Generator::SSE, &[Exactness::Exact], PrecisionControl::FULL);
    let defaults = core::cell::Cell::new(0_usize);
    for (runs, encoding) in [
        (arm, IntegerEncoding::Saturating),
        (sse, IntegerEncoding::Indefinite),
    ] {
        check(&function, Level::Two, &runs, |line, run| {
            let [a, result, flags] = fields::<3>(line);
            let (ours, ours_flags) = decode::<S, W>(a).to_int_with::<I>(run.env);
            let ours = (encoding.bits(ours), ours_flags);
            assert_case(&function, line, run, ours, [result, flags]);
            // `to_int` returns no flags and takes a host path where the build
            // has one.
            if run.env == Env::IEEE {
                let default = encoding.bits(decode::<S, W>(a).to_int::<I>());
                assert_eq!(default, result, "{function} {line}: default mode");
                defaults.set(defaults.get() + 1);
            }
        });
    }
    assert!(
        defaults.get() > 0,
        "{function}: the default mode converted test cases"
    );
}

/// Checks the conversion of the integer type `I` to a format, in every
/// direction.
///
/// The runs do not use precision control. floaty applies the precision limit
/// to a conversion from an integer, and SoftFloat's conversions of an
/// integer to x87 extended precision ignore it.
fn check_from_integer<S, const W: usize, I>(format: Format)
where
    S: Standard<W, Bits: Field>,
    I: Integer + TryFrom<i128, Error: Debug>,
{
    let function = format!("{}_to_{}", integer_name::<I>(), format.name);
    let runs = directions(Generator::ARM, &[Exactness::Exact], PrecisionControl::FULL);
    let defaults = core::cell::Cell::new(0_usize);
    check(&function, Level::Two, &runs, |line, run| {
        let [a, result, flags] = fields::<3>(line);
        let value = integer_from_field::<I>(a);
        let (ours, ours_flags) = Float::<S, W>::from_int_with(value, run.env);
        let ours = (ours.to_bits().into(), ours_flags);
        assert_case(&function, line, run, ours, [result, flags]);
        // `from_int` returns no flags and takes a host path where the build
        // has one.
        if run.env == Env::IEEE {
            let default: u128 = Float::<S, W>::from_int(value).to_bits().into();
            assert_eq!(default, result, "{function} {line}: default mode");
            defaults.set(defaults.get() + 1);
        }
    });
    assert!(
        defaults.get() > 0,
        "{function}: the default mode converted test cases"
    );
}

/// Defines the tests of one format.
macro_rules! format_tests {
    ($module:ident, $standard:ty, $width:literal, $format:expr) => {
        mod $module {
            use super::*;

            const FORMAT: Format = $format;

            #[test]
            fn compare() {
                check_comparisons::<$standard, $width>(FORMAT);
            }

            #[test]
            fn remainder() {
                check_remainder::<$standard, $width>(FORMAT);
            }

            #[test]
            fn round_to_integral() {
                check_round_to_integral::<$standard, $width>(FORMAT);
            }

            #[test]
            fn to_integer() {
                check_to_integer::<$standard, $width, i32>(FORMAT);
                check_to_integer::<$standard, $width, i64>(FORMAT);
                check_to_integer::<$standard, $width, u32>(FORMAT);
                check_to_integer::<$standard, $width, u64>(FORMAT);
            }

            #[test]
            fn from_integer() {
                check_from_integer::<$standard, $width, i32>(FORMAT);
                check_from_integer::<$standard, $width, i64>(FORMAT);
                check_from_integer::<$standard, $width, u32>(FORMAT);
                check_from_integer::<$standard, $width, u64>(FORMAT);
            }
        }
    };
}

format_tests!(
    binary16,
    Binary<5>,
    16,
    Format {
        name: "f16",
        family: Family::Interchange,
    }
);
format_tests!(
    binary32,
    Binary<8>,
    32,
    Format {
        name: "f32",
        family: Family::Interchange,
    }
);
format_tests!(
    binary64,
    Binary<11>,
    64,
    Format {
        name: "f64",
        family: Family::Interchange,
    }
);
format_tests!(
    extended,
    Binary<15, X87>,
    80,
    Format {
        name: "extF80",
        family: Family::Extended,
    }
);
format_tests!(
    binary128,
    Binary<15>,
    128,
    Format {
        name: "f128",
        family: Family::Interchange,
    }
);
