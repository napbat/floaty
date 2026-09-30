//! Runs the Berkeley TestFloat generator, `testfloat_gen`, and reads its test
//! cases.
//!
//! The build script builds one generator for each SoftFloat NaN
//! specialization. Each line of the output holds the operands, the expected
//! result, and the expected flags, all in hexadecimal.

use core::fmt::Debug;
use core::num::NonZeroU32;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use floaty::env::{NanPropagation, NanRule, Tininess};
use floaty::{Env, Flags, Rounding};

/// A `testfloat_gen` built against one SoftFloat NaN specialization, and the
/// NaN rule that the specialization follows.
#[derive(Clone, Copy, Debug)]
pub struct Generator {
    /// The path of the generator.
    pub path: &'static str,
    /// The NaN rule of the specialization.
    pub rule: NanRule,
    /// Whether the rule holds for x87 extended precision. The 8086-SSE
    /// specialization has the x87 code of the 8086 specialization, so its
    /// rule does not.
    pub covers_extended: bool,
}

impl Generator {
    /// The ARM-VFPv2 specialization: a signaling NaN first and a positive
    /// default NaN. `Env::IEEE` uses this rule.
    pub const ARM: Self = Self {
        path: env!("FLOATY_TESTFLOAT_GEN_ARM"),
        rule: NanRule::new(NanPropagation::SignalingFirst),
        covers_extended: true,
    };

    /// The ARM-VFPv2-defaultNaN specialization: the default NaN for every
    /// NaN result.
    pub const ARM_DEFAULT_NAN: Self = Self {
        path: env!("FLOATY_TESTFLOAT_GEN_ARM_DEFAULT_NAN"),
        rule: NanRule::new(NanPropagation::DefaultNan),
        covers_extended: true,
    };

    /// The 8086 specialization: the larger significand and a negative default
    /// NaN.
    pub const X87: Self = Self {
        path: env!("FLOATY_TESTFLOAT_GEN_X87"),
        rule: NanRule::new(NanPropagation::LargerSignificand).with_default_negative(true),
        covers_extended: true,
    };

    /// The 8086-SSE specialization: the first NaN operand and a negative
    /// default NaN. SoftFloat signals invalid for `0 * inf + NaN`; the
    /// processor does not, and [`crate::x86::sse_env`] follows the processor.
    pub const SSE: Self = Self {
        path: env!("FLOATY_TESTFLOAT_GEN_SSE"),
        rule: NanRule::new(NanPropagation::FirstOperand).with_default_negative(true),
        covers_extended: false,
    };

    /// Every generator.
    pub const ALL: [Self; 4] = [Self::ARM, Self::ARM_DEFAULT_NAN, Self::X87, Self::SSE];

    /// The generators of the NaN rules other than the rule of `Env::IEEE`.
    /// A NaN rule does not depend on the rounding direction, so a test can
    /// run them at the default behavior only.
    pub const OTHER_NAN_RULES: [Self; 3] = [Self::ARM_DEFAULT_NAN, Self::X87, Self::SSE];
}

/// The rounding directions and their `testfloat_gen` options.
pub const ROUNDINGS: [(Rounding, &str); 6] = [
    (Rounding::TiesToEven, "-rnear_even"),
    (Rounding::TiesToAway, "-rnear_maxMag"),
    (Rounding::TowardZero, "-rminMag"),
    (Rounding::TowardNegative, "-rmin"),
    (Rounding::TowardPositive, "-rmax"),
    (Rounding::ToOdd, "-rodd"),
];

/// The tininess rules and their `testfloat_gen` options.
pub const TININESS: [(Tininess, &str); 2] = [
    (Tininess::BeforeRounding, "-tininessbefore"),
    (Tininess::AfterRounding, "-tininessafter"),
];

/// A TestFloat test level. Level 2 has more cases than level 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// `-level 1`.
    One,
    /// `-level 2`.
    Two,
}

/// Whether TestFloat raises inexact when a result rounds to an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exactness {
    /// `-exact`: the expected flags include inexact, as floaty reports it.
    Exact,
    /// `-notexact`: the expected flags never include inexact. The IEEE 754
    /// operation that does not signal inexact is floaty's operation with
    /// `INEXACT` ignored.
    NotExact,
}

impl Exactness {
    /// Returns the part of floaty's flags that a TestFloat case reports.
    #[must_use]
    pub fn reported(self, flags: Flags) -> Flags {
        match self {
            Self::Exact => flags,
            Self::NotExact => flags.difference(Flags::INEXACT),
        }
    }
}

/// An x87 precision-control setting of `testfloat_gen`, with the precision
/// limit of floaty that matches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PrecisionControl {
    /// The precision limit, or `None` for the full 64 bits.
    pub limit: Option<NonZeroU32>,
    /// The `testfloat_gen` option.
    option: &'static str,
}

impl PrecisionControl {
    /// `-precision80`: the full precision. The generator ignores the option
    /// for the formats other than x87 extended precision.
    pub const FULL: Self = Self {
        limit: None,
        option: "-precision80",
    };
}

/// Every x87 precision-control setting: 80, 32, and 64 bits.
pub const PRECISION_CONTROL: [PrecisionControl; 3] = [
    PrecisionControl::FULL,
    PrecisionControl {
        limit: NonZeroU32::new(24),
        option: "-precision32",
    },
    PrecisionControl {
        limit: NonZeroU32::new(53),
        option: "-precision64",
    },
];

/// The options of one `testfloat_gen` run beside its level. The generator
/// takes its default for an option at `None`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// The rounding direction, one of [`ROUNDINGS`].
    pub rounding: Option<Rounding>,
    /// The tininess rule.
    pub tininess: Option<Tininess>,
    /// Whether a rounding to an integer raises inexact.
    pub exactness: Option<Exactness>,
    /// The x87 precision control.
    pub precision: Option<PrecisionControl>,
}

impl Options {
    /// Returns the arguments of `testfloat_gen` for `function` at `level`.
    ///
    /// # Panics
    ///
    /// Panics for a rounding direction that is not in [`ROUNDINGS`].
    fn arguments<'a>(&self, level: Level, function: &'a str) -> Vec<&'a str> {
        let level = match level {
            Level::One => "1",
            Level::Two => "2",
        };
        let rounding = self.rounding.map(|rounding| option(&ROUNDINGS, rounding));
        let tininess = self.tininess.map(|tininess| option(&TININESS, tininess));
        let exactness = self.exactness.map(|exactness| match exactness {
            Exactness::Exact => "-exact",
            Exactness::NotExact => "-notexact",
        });
        let precision = self.precision.map(|precision| precision.option);
        ["-level", level]
            .into_iter()
            .chain(
                [rounding, tininess, exactness, precision]
                    .into_iter()
                    .flatten(),
            )
            .chain([function])
            .collect()
    }
}

/// One run of a generator: the generator, its options, and the behavior and
/// the exactness that the options give.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    /// The generator.
    pub generator: Generator,
    /// The options of the generator.
    pub options: Options,
    /// The behavior of the run: [`Env::IEEE`] with the NaN rule of the
    /// generator, and the rounding direction, the tininess rule, and the
    /// precision limit of the options.
    pub env: Env,
    /// Whether a rounding to an integer raises inexact.
    pub exactness: Exactness,
}

impl Run {
    /// Makes a run of `generator` with `options`. An option at `None` takes
    /// the default of `testfloat_gen`: round to nearest even, tininess after
    /// rounding, the full precision, and `-notexact`.
    #[must_use]
    pub fn new(generator: Generator, options: Options) -> Self {
        let env = Env::IEEE
            .with_nan(generator.rule)
            .with_rounding(options.rounding.unwrap_or(Rounding::TiesToEven))
            .with_tininess(options.tininess.unwrap_or(Tininess::AfterRounding))
            .with_precision(options.precision.and_then(|precision| precision.limit));
        Self {
            generator,
            options,
            env,
            exactness: options.exactness.unwrap_or(Exactness::NotExact),
        }
    }
}

/// Returns the `testfloat_gen` option of `value` in `table`.
///
/// # Panics
///
/// Panics when `table` has no entry for `value`.
fn option<T: Copy + PartialEq + Debug>(table: &[(T, &'static str)], value: T) -> &'static str {
    let Some(&(_, option)) = table.iter().find(|&&(entry, _)| entry == value) else {
        panic!("testfloat_gen has no option for {value:?}");
    };
    option
}

/// Returns the TestFloat flag bits of the IEEE flags: inexact 1, underflow 2,
/// overflow 4, infinite 8, and invalid 16.
#[must_use]
pub fn flag_bits(flags: Flags) -> u128 {
    [
        (Flags::INEXACT, 1),
        (Flags::UNDERFLOW, 2),
        (Flags::OVERFLOW, 4),
        (Flags::DIVIDE_BY_ZERO, 8),
        (Flags::INVALID, 16),
    ]
    .into_iter()
    .filter(|&(flag, _)| flags.contains(flag))
    .map(|(_, bit)| bit)
    .sum()
}

/// Splits a test case into its `N` hexadecimal fields: the operands, the
/// result, and the flags.
///
/// # Panics
///
/// Panics when the line does not have `N` hexadecimal fields.
#[must_use]
pub fn fields<const N: usize>(line: &str) -> [u128; N] {
    let values: Vec<u128> = line
        .split(' ')
        .map(|field| u128::from_str_radix(field, 16).expect("a field is hexadecimal"))
        .collect();
    values
        .try_into()
        .unwrap_or_else(|_| panic!("a test case has {N} fields: {line}"))
}

/// Runs `generator` on `function` at `level` with `options`, and calls
/// `check` on each test case. With a `limit`, it stops after that many
/// cases. Returns the case count.
///
/// # Panics
///
/// Panics when the generator does not run or fails, and for a rounding
/// direction that is not in [`ROUNDINGS`].
pub fn run(
    generator: Generator,
    level: Level,
    function: &str,
    options: &Options,
    limit: Option<usize>,
    mut check: impl FnMut(&str),
) -> usize {
    let mut child = Command::new(generator.path)
        .args(options.arguments(level, function))
        .stdout(Stdio::piped())
        .spawn()
        .expect("testfloat_gen can run");
    let stdout = child.stdout.take().expect("the output is piped");
    let mut count = 0;
    for line in BufReader::new(stdout).lines() {
        check(&line.expect("the output is text"));
        count += 1;
        if limit.is_some_and(|limit| count >= limit) {
            // The generator has more cases; stop it.
            child.kill().expect("testfloat_gen can be stopped");
            child.wait().expect("testfloat_gen exits");
            return count;
        }
    }
    assert!(
        child.wait().expect("testfloat_gen exits").success(),
        "testfloat_gen failed"
    );
    count
}

/// Returns an expected x87 result with the quiet bit set when it is a
/// signaling NaN.
///
/// SoftFloat 3e's ARM-VFPv2 `softfloat_propagateNaNExtF80UI` never quiets
/// the NaN it returns: line 79 of `s_propagateNaNExtF80UI.c`,
/// `uiZ.v0 | UINT64_C( 0xC000000000000000 );`, discards its result. Every
/// other specialization sets the quiet bit, and IEEE 754-2019 section 6.2
/// requires a quiet NaN. No ARM processor has the x87 format, so no hardware
/// shows the combination. The flags stay as TestFloat gives them. Every
/// other value passes through unchanged.
#[must_use]
pub fn quiet_extended_nan(bits: u128) -> u128 {
    let exponent = (bits >> 64) & 0x7FFF;
    let integer = (bits >> 63) & 1 == 1;
    let quiet = (bits >> 62) & 1 == 1;
    let payload = bits & ((1 << 62) - 1);
    if exponent == 0x7FFF && integer && !quiet && payload != 0 {
        bits | (1 << 62)
    } else {
        bits
    }
}
