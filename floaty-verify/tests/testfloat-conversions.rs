//! Compares the conversions between binary16, binary32, binary64, x87
//! extended precision, and binary128 with Berkeley TestFloat.
//!
//! `testfloat_gen` writes each test case with the result and the flags that
//! Berkeley SoftFloat computes. Four SoftFloat builds give the expected
//! results, one for each NaN rule: the ARM specialization, which follows the
//! NaN rule of the default mode, the ARM default-NaN specialization, the 8086
//! specialization of x87, and the 8086-SSE specialization. The ARM, 8086, and
//! 8086-SSE builds share their NaN conversion code, so the 8086 and 8086-SSE
//! runs check that the propagation rule does not change a conversion. A run
//! with the default mode in one rounding direction also converts with the
//! static mode `Rounded<Ieee, R>` as the behavior.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::env::NanRule;
use floaty::{Env, F16, F32, F64, F80, F128, Flags};
use floaty_verify::testfloat::{
    self, ARM, ARM_DEFAULT_NAN, ARM_RULE, DEFAULT_NAN_RULE, ROUNDINGS, SSE, SSE_RULE, TININESS,
    X87, X87_RULE, fields, flag_bits,
};

/// The generators, and the NaN rule that each one follows.
const GENERATORS: [(&str, NanRule); 4] = [
    (ARM, ARM_RULE),
    (ARM_DEFAULT_NAN, DEFAULT_NAN_RULE),
    (X87, X87_RULE),
    (SSE, SSE_RULE),
];

/// Converts one encoding and returns the result bits and the flags.
macro_rules! convert {
    ($source:ty => $destination:ty, $bits:expr, $env:expr) => {{
        let source = <$source>::from_bits($bits.try_into().expect("the operand fits the storage"));
        let (result, flags): ($destination, Flags) = source.convert_with($env);
        (u128::from(result.to_bits()), flags)
    }};
}

/// Converts one encoding with `convert`, which returns no flags and takes a
/// host path where the build has one, and returns the result bits.
macro_rules! convert_default {
    ($source:ty => $destination:ty, $bits:expr) => {{
        let source = <$source>::from_bits($bits.try_into().expect("the operand fits the storage"));
        let result: $destination = source.convert();
        u128::from(result.to_bits())
    }};
}

/// Defines one test that checks one conversion against every generator, in
/// every rounding direction and with both tininess rules. The runs of the
/// default mode also check `convert`.
macro_rules! conversion_test {
    ($name:ident, $function:literal, $source:ty => $destination:ty) => {
        #[test]
        fn $name() {
            let fixed_count = core::cell::Cell::new(0_usize);
            let default_count = core::cell::Cell::new(0_usize);
            for (generator, nan) in GENERATORS {
                let mut count = 0_usize;
                for (rounding, rounding_option) in ROUNDINGS {
                    for (tininess, tininess_option) in TININESS {
                        let env = Env::IEEE.with_rounding(rounding).with_tininess(tininess).with_nan(nan);
                        let arguments = ["-level", "2", rounding_option, tininess_option, $function];
                        count += testfloat::run(generator, &arguments, None, |line| {
                            let [operand, result, flags] = fields::<3>(line);
                            let (ours, ours_flags) = convert!($source => $destination, operand, env);
                            let context = format!("{} {env:?} {line}", $function);
                            assert_eq!(ours, result, "{context}: result");
                            assert_eq!(flag_bits(ours_flags), flags, "{context}: flags {ours_flags:?}");
                            if env == Env::IEEE.with_rounding(rounding) {
                                let (fixed, fixed_flags) = floaty_verify::with_rounding_mode!(
                                    rounding,
                                    floaty::mode::Ieee,
                                    Mode => convert!($source => $destination, operand, Mode::default())
                                );
                                assert_eq!(
                                    (fixed, flag_bits(fixed_flags)),
                                    (result, flags),
                                    "{context}: static mode"
                                );
                                fixed_count.set(fixed_count.get() + 1);
                                if env == Env::IEEE {
                                    let default = convert_default!($source => $destination, operand);
                                    assert_eq!(default, result, "{context}: convert");
                                    default_count.set(default_count.get() + 1);
                                }
                            }
                        });
                    }
                }
                assert!(count > 0, "{generator} gave test cases");
            }
            assert!(fixed_count.get() > 0, "the static modes converted test cases");
            assert!(default_count.get() > 0, "convert converted test cases");
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
