//! Runs the Berkeley TestFloat generator, `testfloat_gen`, and reads its test
//! cases.
//!
//! `build.rs` builds one generator for each SoftFloat NaN specialization.
//! Each line of the output holds the operands, the expected result, and the
//! expected flags, all in hexadecimal.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use floaty::env::{NanPropagation, NanRule, Tininess};
use floaty::{Flags, Rounding};

/// The generator built against SoftFloat's ARM-VFPv2 specialization, which
/// follows [`ARM_RULE`].
pub const ARM: &str = env!("FLOATY_TESTFLOAT_GEN_ARM");

/// The generator built against SoftFloat's ARM-VFPv2-defaultNaN
/// specialization, which follows [`DEFAULT_NAN_RULE`].
pub const ARM_DEFAULT_NAN: &str = env!("FLOATY_TESTFLOAT_GEN_ARM_DEFAULT_NAN");

/// The NaN rule of the ARM-VFPv2 specialization: a signaling NaN first and a
/// positive default NaN. `Env::IEEE` uses this rule.
pub const ARM_RULE: NanRule = NanRule::new(NanPropagation::SignalingFirst);

/// The NaN rule of the default-NaN specialization.
pub const DEFAULT_NAN_RULE: NanRule = NanRule::new(NanPropagation::DefaultNan);

/// The generator built against SoftFloat's 8086 specialization, which
/// follows [`X87_RULE`].
pub const X87: &str = env!("FLOATY_TESTFLOAT_GEN_X87");

/// The NaN rule of the 8086 specialization: the larger significand and a
/// negative default NaN.
pub const X87_RULE: NanRule =
    NanRule::new(NanPropagation::LargerSignificand).with_default_negative(true);

/// The generator built against SoftFloat's 8086-SSE specialization, which
/// follows [`SSE_RULE`] for every format but x87 extended precision. Its x87
/// code is the 8086 code.
pub const SSE: &str = env!("FLOATY_TESTFLOAT_GEN_SSE");

/// The NaN rule of the 8086-SSE specialization: the first NaN operand and a
/// negative default NaN. SoftFloat signals invalid for `0 * inf + NaN`; the
/// processor does not, and [`crate::x86::sse_env`] follows the processor.
pub const SSE_RULE: NanRule =
    NanRule::new(NanPropagation::FirstOperand).with_default_negative(true);

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

/// Runs `generator` with `arguments` and calls `check` on each test case.
/// With a `limit`, it stops after that many cases. Returns the case count.
///
/// # Panics
///
/// Panics when the generator does not run or fails.
pub fn run(
    generator: &str,
    arguments: &[&str],
    limit: Option<usize>,
    mut check: impl FnMut(&str),
) -> usize {
    let mut child = Command::new(generator)
        .args(arguments)
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
