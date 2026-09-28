//! Compares the conversions between binary16, binary32, binary64, x87
//! extended precision, and binary128 with Berkeley TestFloat.
//!
//! `testfloat_gen` writes each test case with the result and the flags that
//! Berkeley SoftFloat computes. Two SoftFloat builds give the expected
//! results: the ARM NaN specialization, which follows the NaN rule of the
//! default mode, and the ARM default-NaN specialization, which follows the
//! `DefaultNan` rule.

use floaty::env::NanRule;
use floaty::{Env, F16, F32, F64, F80, F128, Flags};
use floaty_verify::testfloat::{
    self, ARM, ARM_DEFAULT_NAN, ARM_RULE, DEFAULT_NAN_RULE, ROUNDINGS, TININESS, fields, flag_bits,
};

/// The generators, and the NaN rule that each one follows.
const GENERATORS: [(&str, NanRule); 2] = [(ARM, ARM_RULE), (ARM_DEFAULT_NAN, DEFAULT_NAN_RULE)];

/// Converts one encoding and returns the result bits and the flags.
macro_rules! convert {
    ($source:ty => $destination:ty, $bits:expr, $env:expr) => {{
        let source = <$source>::from_bits($bits.try_into().expect("the operand fits the storage"));
        let (result, flags): ($destination, Flags) = source.convert_with($env);
        (u128::from(result.to_bits()), flags)
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
                        let arguments = ["-level", "2", rounding_option, tininess_option, $function];
                        count += testfloat::run(generator, &arguments, None, |line| {
                            let [operand, result, flags] = fields::<3>(line);
                            let (ours, ours_flags) = convert!($source => $destination, operand, env);
                            let context = format!("{} {env:?} {line}", $function);
                            assert_eq!(ours, result, "{context}: result");
                            assert_eq!(flag_bits(ours_flags), flags, "{context}: flags {ours_flags:?}");
                        });
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
