//! Compares add, subtract, multiply, divide, square root, and fused
//! multiply-add with Berkeley TestFloat, for binary16, binary32, binary64,
//! x87 extended precision, and binary128.
//!
//! The ARM generator checks each rounding direction of TestFloat with both
//! tininess rules, and x87 precision control at 32, 64, and 80 bits. The
//! default-NaN, 8086, and 8086-SSE generators check the `DefaultNan`,
//! `LargerSignificand`, and `FirstOperand` NaN rules, which do not depend on
//! the direction. Two- and three-operand functions use TestFloat level 1,
//! and square root uses level 2.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::env::{NanRule, Tininess};
use floaty::{Env, F16, F32, F64, F80, F128, Rounding, mode};
use floaty_verify::testfloat::{
    self, ARM, ARM_DEFAULT_NAN, DEFAULT_NAN_RULE, ROUNDINGS, SSE, SSE_RULE, TININESS, X87,
    X87_RULE, fields, flag_bits, quiet_extended_nan,
};

/// The fused multiply-add cases for each behavior in the normal test run.
/// TestFloat level 1 has 6,133,248 cases for each run; the ignored test runs
/// them all.
const MUL_ADD_LIMIT: usize = 1_000_000;

/// One run of the generator: the generator, the behavior, and its options.
struct Run {
    generator: &'static str,
    env: Env,
    options: Vec<&'static str>,
}

/// Returns the runs for a function. `precision_control` adds the x87
/// precision limits of 32 and 64 bits and marks the x87 format. With
/// `every_direction`, the ARM generator runs each rounding direction and
/// tininess rule; otherwise only the default behavior runs.
///
/// The other generators check the other NaN rules at the default behavior,
/// because a NaN rule does not depend on the direction. The 8086-SSE
/// generator runs for every format but x87 extended precision, whose code in
/// that specialization is the 8086 code.
fn runs(precision_control: bool, every_direction: bool) -> Vec<Run> {
    let precisions: &[(Option<u32>, &'static str)] = if precision_control {
        &[
            (None, "-precision80"),
            (Some(24), "-precision32"),
            (Some(53), "-precision64"),
        ]
    } else {
        &[(None, "-precision80")]
    };
    let mut rules = vec![(ARM_DEFAULT_NAN, DEFAULT_NAN_RULE), (X87, X87_RULE)];
    if !precision_control {
        rules.push((SSE, SSE_RULE));
    }
    let mut runs = Vec::new();
    for &(precision, precision_option) in precisions {
        let base = Env::IEEE.with_precision(precision.and_then(NonZeroU32::new));
        if every_direction {
            for (rounding, rounding_option) in ROUNDINGS {
                for (tininess, tininess_option) in TININESS {
                    runs.push(Run {
                        generator: ARM,
                        env: base.with_rounding(rounding).with_tininess(tininess),
                        options: vec![rounding_option, tininess_option, precision_option],
                    });
                }
            }
        } else {
            runs.push(Run {
                generator: ARM,
                env: base,
                options: vec!["-rnear_even", "-tininessafter", precision_option],
            });
        }
        for &(generator, rule) in &rules {
            runs.push(Run {
                generator,
                env: base.with_nan(rule).with_tininess(Tininess::AfterRounding),
                options: vec!["-rnear_even", "-tininessafter", precision_option],
            });
        }
    }
    runs
}

/// Checks one function against every run.
fn check(
    function: &str,
    level: &str,
    runs: &[Run],
    limit: Option<usize>,
    compare: impl Fn(&str, Env),
) {
    assert!(
        runs.iter().any(|run| has_static_mode(run.env)),
        "{function} has a run that checks a static mode"
    );
    for run in runs {
        let mut arguments = vec!["-level", level];
        arguments.extend(&run.options);
        arguments.push(function);
        let count = testfloat::run(run.generator, &arguments, limit, |line| {
            compare(line, run.env);
        });
        assert!(count > 0, "{function} {:?} gave test cases", run.options);
    }
}

/// Returns `true` when a run has the default mode with one rounding
/// direction. The static mode `Rounded<Ieee, R>` then gives the same results,
/// which the tests check against TestFloat too.
fn has_static_mode(env: Env) -> bool {
    env == Env::IEEE.with_rounding(env.rounding)
}

/// Returns `true` when two NaN rules give the same NaN for one or two
/// operands.
fn same_rule(run: NanRule, mode: NanRule) -> bool {
    (run.propagation, run.default_negative) == (mode.propagation, mode.default_negative)
}

/// Returns `true` when two NaN rules give the same NaN for every operation,
/// the fused multiply-add included.
fn same_fused_rule(run: NanRule, mode: NanRule) -> bool {
    run == mode
}

/// Returns the bits of `$body` in the mode whose NaN rule `$same` matches the
/// rule of a run that rounds to nearest even at the full precision, or
/// `None` when no mode matches.
///
/// The entry points of a mode that return no flags, the operators, `sqrt`,
/// and `mul_add`, take a host path where the build has one. So a run that
/// matches a mode checks the host paths against TestFloat.
macro_rules! in_mode_of_run {
    ($alias:ty, $env:expr, $same:expr, [$($value:ident),+] => $body:expr) => {{
        let env: Env = $env;
        if env.rounding != Rounding::TiesToEven || env.precision.is_some() {
            None
        } else if $same(env.nan, Env::IEEE.nan) {
            Some(u128::from($body.to_bits()))
        } else if $same(env.nan, Env::X86_SSE.nan) {
            $(let $value = $value.with_mode::<mode::X86Sse>();)+
            Some(u128::from($body.to_bits()))
        } else if $same(env.nan, Env::X87.nan) && <$alias>::PRECISION <= 64 {
            // The x87 mode limits the precision to 64 bits.
            $(let $value = $value.with_mode::<mode::X87>();)+
            Some(u128::from($body.to_bits()))
        } else {
            None
        }
    }};
}

/// Compares one two-operand case.
///
/// A run that rounds to nearest even with the NaN rule of a mode also checks
/// the operator of a type with that mode.
macro_rules! two_operands {
    ($alias:ty, $method:ident, $operator:tt, $function:expr, $expected:expr) => {
        |line: &str, env: Env| {
            let [a, b, result, flags] = fields::<4>(line);
            let result = $expected(result);
            let a = <$alias>::from_bits(a.try_into().expect("the operand fits"));
            let b = <$alias>::from_bits(b.try_into().expect("the operand fits"));
            let (ours, ours_flags) = a.$method(b, env);
            let context = format!("{} {env:?} {line}", $function);
            assert_eq!(u128::from(ours.to_bits()), result, "{context}: result");
            if let Some(bits) = in_mode_of_run!($alias, env, same_rule, [a, b] => (a $operator b)) {
                assert_eq!(bits, result, "{context}: operator");
            }
            assert_eq!(
                flag_bits(ours_flags),
                flags,
                "{context}: flags {ours_flags:?}"
            );
            if has_static_mode(env) {
                let (fixed, fixed_flags) =
                    floaty_verify::with_rounding_mode!(env.rounding, mode::Ieee, Mode => {
                        let (a, b) = (a.with_mode::<Mode>(), b.with_mode::<Mode>());
                        let (value, flags) = a.$method(b, Mode::default());
                        (u128::from(value.to_bits()), flags)
                    });
                assert_eq!(
                    (fixed, flag_bits(fixed_flags)),
                    (result, flags),
                    "{context}: static mode"
                );
            }
        }
    };
}

/// Defines the tests of the two-operand functions and square root of one
/// format.
macro_rules! format_tests {
    ($module:ident, $alias:ty, $prefix:literal, $precision_control:expr, $expected:expr) => {
        mod $module {
            use super::*;

            #[test]
            fn add() {
                let function = concat!($prefix, "_add");
                check(
                    function,
                    "1",
                    &runs($precision_control, true),
                    None,
                    two_operands!($alias, add_with, +, function, $expected),
                );
            }

            #[test]
            fn sub() {
                let function = concat!($prefix, "_sub");
                check(
                    function,
                    "1",
                    &runs($precision_control, true),
                    None,
                    two_operands!($alias, sub_with, -, function, $expected),
                );
            }

            #[test]
            fn mul() {
                let function = concat!($prefix, "_mul");
                check(
                    function,
                    "1",
                    &runs($precision_control, true),
                    None,
                    two_operands!($alias, mul_with, *, function, $expected),
                );
            }

            #[test]
            fn div() {
                let function = concat!($prefix, "_div");
                check(
                    function,
                    "1",
                    &runs($precision_control, true),
                    None,
                    two_operands!($alias, div_with, /, function, $expected),
                );
            }

            #[test]
            fn sqrt() {
                let function = concat!($prefix, "_sqrt");
                check(
                    function,
                    "2",
                    &runs($precision_control, true),
                    None,
                    |line: &str, env: Env| {
                        let [a, result, flags] = fields::<3>(line);
                        let result = $expected(result);
                        let value = <$alias>::from_bits(a.try_into().expect("the operand fits"));
                        let (ours, ours_flags) = value.sqrt_with(env);
                        let context = format!("{function} {env:?} {line}");
                        assert_eq!(u128::from(ours.to_bits()), result, "{context}: result");
                        if let Some(bits) =
                            in_mode_of_run!($alias, env, same_rule, [value] => value.sqrt())
                        {
                            assert_eq!(bits, result, "{context}: sqrt of the mode");
                        }
                        assert_eq!(
                            flag_bits(ours_flags),
                            flags,
                            "{context}: flags {ours_flags:?}"
                        );
                        if has_static_mode(env) {
                            let (fixed, fixed_flags) = floaty_verify::with_rounding_mode!(
                                env.rounding,
                                mode::Ieee,
                                Mode => {
                                    let (root, flags) =
                                        value.with_mode::<Mode>().sqrt_with(Mode::default());
                                    (u128::from(root.to_bits()), flags)
                                }
                            );
                            assert_eq!(
                                (fixed, flag_bits(fixed_flags)),
                                (result, flags),
                                "{context}: static mode"
                            );
                        }
                    },
                );
            }
        }
    };
}

format_tests!(binary16, F16, "f16", false, core::convert::identity);
format_tests!(binary32, F32, "f32", false, core::convert::identity);
format_tests!(binary64, F64, "f64", false, core::convert::identity);
format_tests!(extended, F80, "extF80", true, quiet_extended_nan);
format_tests!(binary128, F128, "f128", false, core::convert::identity);

/// Compares one fused multiply-add case.
macro_rules! mul_add {
    ($alias:ty, $function:expr) => {
        |line: &str, env: Env| {
            let [a, b, c, result, flags] = fields::<5>(line);
            let [a, b, c] = [a, b, c]
                .map(|bits| <$alias>::from_bits(bits.try_into().expect("the operand fits")));
            let (ours, ours_flags) = a.mul_add_with(b, c, env);
            let context = format!("{} {env:?} {line}", $function);
            assert_eq!(u128::from(ours.to_bits()), result, "{context}: result");
            if let Some(bits) =
                in_mode_of_run!($alias, env, same_fused_rule, [a, b, c] => a.mul_add(b, c))
            {
                assert_eq!(bits, result, "{context}: mul_add of the mode");
            }
            assert_eq!(
                flag_bits(ours_flags),
                flags,
                "{context}: flags {ours_flags:?}"
            );
            if has_static_mode(env) {
                let (fixed, fixed_flags) =
                    floaty_verify::with_rounding_mode!(env.rounding, mode::Ieee, Mode => {
                        let [a, b, c] = [a, b, c].map(|value| value.with_mode::<Mode>());
                        let (value, flags) = a.mul_add_with(b, c, Mode::default());
                        (u128::from(value.to_bits()), flags)
                    });
                assert_eq!(
                    (fixed, flag_bits(fixed_flags)),
                    (result, flags),
                    "{context}: static mode"
                );
            }
        }
    };
}

#[test]
fn mul_add_prefix() {
    let runs = runs(false, false);
    check(
        "f16_mulAdd",
        "1",
        &runs,
        Some(MUL_ADD_LIMIT),
        mul_add!(F16, "f16_mulAdd"),
    );
    check(
        "f32_mulAdd",
        "1",
        &runs,
        Some(MUL_ADD_LIMIT),
        mul_add!(F32, "f32_mulAdd"),
    );
    check(
        "f64_mulAdd",
        "1",
        &runs,
        Some(MUL_ADD_LIMIT),
        mul_add!(F64, "f64_mulAdd"),
    );
    check(
        "f128_mulAdd",
        "1",
        &runs,
        Some(MUL_ADD_LIMIT),
        mul_add!(F128, "f128_mulAdd"),
    );
}

/// Every level 1 fused multiply-add case in every direction and tininess
/// rule, about 318 million cases. Run time: about five minutes.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn mul_add_level_1() {
    let runs = runs(false, true);
    check("f16_mulAdd", "1", &runs, None, mul_add!(F16, "f16_mulAdd"));
    check("f32_mulAdd", "1", &runs, None, mul_add!(F32, "f32_mulAdd"));
    check("f64_mulAdd", "1", &runs, None, mul_add!(F64, "f64_mulAdd"));
    check(
        "f128_mulAdd",
        "1",
        &runs,
        None,
        mul_add!(F128, "f128_mulAdd"),
    );
}

/// Checks that a direction override reaches every operation.
#[test]
fn a_rounding_override_reaches_every_operation() {
    let one = F32::from_bits(0x3F80_0000);
    let tiny = F32::from_bits(0x3380_0000);
    assert_eq!(
        one.add_with(tiny, Rounding::TowardPositive).0.to_bits(),
        0x3F80_0001
    );
    assert_eq!(
        one.add_with(tiny, Rounding::TiesToEven).0.to_bits(),
        0x3F80_0000
    );
}
