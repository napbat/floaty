//! Compares the conversions between binary16, binary32, binary64, x87
//! extended precision, and binary128 with Berkeley TestFloat.
//!
//! `testfloat_gen` writes each test case with the result and the flags that
//! Berkeley SoftFloat computes. Two SoftFloat builds give the expected
//! results: the ARM NaN specialization, which follows the NaN rule of the
//! default mode, and the ARM default-NaN specialization, which follows the
//! `DefaultNan` rule.

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};

use floaty::env::{NanPropagation, NanRule, Tininess};
use floaty::{Env, F16, F32, F64, F80, F128, Flags, Rounding};

/// The generators, and the NaN rule that each one follows.
const GENERATORS: [(&str, NanRule); 2] = [
    (env!("FLOATY_TESTFLOAT_GEN_ARM"), NanRule::ARM),
    (
        env!("FLOATY_TESTFLOAT_GEN_ARM_DEFAULT_NAN"),
        NanRule {
            propagation: NanPropagation::DefaultNan,
            default_negative: false,
        },
    ),
];

/// The rounding directions and their `testfloat_gen` options.
const ROUNDINGS: [(Rounding, &str); 6] = [
    (Rounding::NearestEven, "-rnear_even"),
    (Rounding::NearestAway, "-rnear_maxMag"),
    (Rounding::TowardZero, "-rminMag"),
    (Rounding::TowardNegative, "-rmin"),
    (Rounding::TowardPositive, "-rmax"),
    (Rounding::ToOdd, "-rodd"),
];

/// The tininess rules and their `testfloat_gen` options.
const TININESS: [(Tininess, &str); 2] = [
    (Tininess::BeforeRounding, "-tininessbefore"),
    (Tininess::AfterRounding, "-tininessafter"),
];

/// Returns the TestFloat flag bits of the IEEE flags: inexact 1, underflow 2,
/// overflow 4, infinite 8, and invalid 16.
fn testfloat_flags(flags: Flags) -> u8 {
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

/// Runs `testfloat_gen` for one function and returns its lines.
fn generate(generator: &str, function: &str, rounding: &str, tininess: &str) -> Vec<String> {
    let mut child = Command::new(generator)
        .args(["-level", "2", rounding, tininess, function])
        .stdout(Stdio::piped())
        .spawn()
        .expect("testfloat_gen can run");
    let stdout = child.stdout.take().expect("the output is piped");
    let lines: Vec<String> = BufReader::new(stdout)
        .lines()
        .map(|line| line.expect("the output is text"))
        .collect();
    assert!(
        child.wait().expect("testfloat_gen exits").success(),
        "testfloat_gen failed"
    );
    lines
}

/// Converts one encoding and returns the result bits and the flags.
macro_rules! convert {
    ($source:ty => $destination:ty, $bits:expr, $env:expr) => {{
        let source = <$source>::from_bits($bits.try_into().expect("the operand fits the storage"));
        let (result, flags): ($destination, Flags) = source.convert_with($env);
        (u128::from(result.to_bits()), flags)
    }};
}

/// Checks one generator's cases of one conversion. Returns the case count.
macro_rules! check_cases {
    ($function:literal, $source:ty => $destination:ty, $generator:expr, $env:expr, $options:expr) => {{
        let (rounding_option, tininess_option) = $options;
        let mut count = 0_usize;
        for line in generate($generator, $function, rounding_option, tininess_option) {
            let fields: Vec<&str> = line.split(' ').collect();
            let [operand, result, flags] = fields[..] else {
                panic!("malformed test case: {line}");
            };
            let operand = u128::from_str_radix(operand, 16).expect("hexadecimal operand");
            let expected = u128::from_str_radix(result, 16).expect("hexadecimal result");
            let expected_flags = u8::from_str_radix(flags, 16).expect("hexadecimal flags");
            let (ours, ours_flags) = convert!($source => $destination, operand, $env);
            let context = format!("{} {:?} {line}", $function, $env);
            assert_eq!(ours, expected, "{context}: result");
            assert_eq!(testfloat_flags(ours_flags), expected_flags, "{context}: flags {ours_flags:?}");
            count += 1;
        }
        count
    }};
}

/// Defines one test that checks one conversion against both generators, in
/// every rounding direction and with both tininess rules.
macro_rules! conversion_test {
    ($name:ident, $function:literal, $source:ty => $destination:ty) => {
        #[test]
        fn $name() {
            for (generator, nan) in GENERATORS {
                let mut count = 0_usize;
                for (rounding, rounding_option) in ROUNDINGS {
                    for (tininess, tininess_option) in TININESS {
                        let env = Env::IEEE.with_rounding(rounding).with_tininess(tininess).with_nan(nan);
                        let options = (rounding_option, tininess_option);
                        count += check_cases!($function, $source => $destination, generator, env, options);
                    }
                }
                assert!(count > 0, "{generator} gave test cases");
            }
        }
    };
}

conversion_test!(f16_to_f32, "f16_to_f32", F16 => F32);
conversion_test!(f16_to_f64, "f16_to_f64", F16 => F64);
conversion_test!(f16_to_ext_f80, "f16_to_extF80", F16 => F80);
conversion_test!(f16_to_f128, "f16_to_f128", F16 => F128);
conversion_test!(f32_to_f16, "f32_to_f16", F32 => F16);
conversion_test!(f32_to_f64, "f32_to_f64", F32 => F64);
conversion_test!(f32_to_ext_f80, "f32_to_extF80", F32 => F80);
conversion_test!(f32_to_f128, "f32_to_f128", F32 => F128);
conversion_test!(f64_to_f16, "f64_to_f16", F64 => F16);
conversion_test!(f64_to_f32, "f64_to_f32", F64 => F32);
conversion_test!(f64_to_ext_f80, "f64_to_extF80", F64 => F80);
conversion_test!(f64_to_f128, "f64_to_f128", F64 => F128);
conversion_test!(ext_f80_to_f16, "extF80_to_f16", F80 => F16);
conversion_test!(ext_f80_to_f32, "extF80_to_f32", F80 => F32);
conversion_test!(ext_f80_to_f64, "extF80_to_f64", F80 => F64);
conversion_test!(ext_f80_to_f128, "extF80_to_f128", F80 => F128);
conversion_test!(f128_to_f16, "f128_to_f16", F128 => F16);
conversion_test!(f128_to_f32, "f128_to_f32", F128 => F32);
conversion_test!(f128_to_f64, "f128_to_f64", F128 => F64);
conversion_test!(f128_to_ext_f80, "f128_to_extF80", F128 => F80);
