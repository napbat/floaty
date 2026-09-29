//! Unit tests of the value type.

extern crate std;

use std::format;

use super::{
    BF16, Class, Decoded, F8E4M3, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F16, F32, F64, F80, F128, F256,
    F512, TF32,
};
use crate::env::{Env, Flags, Mode, NanPropagation, NanRule, Rounding, mode};
use crate::exact::Exact;

#[test]
fn from_bits_ignores_bits_above_the_width() {
    assert_eq!(TF32::from_bits(u32::MAX).to_bits(), 0x7_FFFF);
    assert_eq!(F32::from_bits(u32::MAX).to_bits(), u32::MAX);
}

#[test]
fn debug_writes_every_digit_of_the_width() {
    assert_eq!(
        format!("{:?}", F32::from_bits(0x3F80_0000)),
        "Float(0x3f800000)"
    );
    assert_eq!(format!("{:?}", TF32::from_bits(0x1)), "Float(0x00001)");
    let text = format!("{:?}", F512::from_bits([1, 0, 0, 0, 0, 0, 0, 1 << 63]));
    assert_eq!(text.len(), "Float(0x)".len() + 128);
    assert!(text.starts_with("Float(0x8000") && text.ends_with("0001)"));
}

#[test]
fn predicates_follow_the_class() {
    let quiet = F32::from_bits(0x7FC0_0000);
    assert!(quiet.is_nan() && !quiet.is_signaling_nan() && !quiet.is_finite());
    let signaling = F32::from_bits(0xFF80_0001);
    assert_eq!(signaling.classify(), Class::SignalingNan);
    assert!(signaling.is_signaling_nan() && signaling.is_sign_negative());
    let infinity = F32::from_bits(0x7F80_0000);
    assert!(infinity.is_infinite() && infinity.is_sign_positive());
    assert!(F32::from_bits(0x8000_0000).is_zero());
    assert!(F32::from_bits(0x0000_0001).is_subnormal());
    assert!(F32::from_bits(0x0080_0000).is_normal());
}

#[test]
fn decode_resizes_to_the_requested_limbs() {
    let value = F80::from_bits(0xBFFF_C000_0000_0000_0001);
    let expected = Decoded::Finite {
        negative: true,
        exponent: -63,
        significand: [0xC000_0000_0000_0001, 0, 0],
    };
    assert_eq!(value.decode::<3>(), expected);
    let nan = F32::from_bits(0x7FA0_0001).decode::<1>();
    assert_eq!(
        nan,
        Decoded::Nan {
            negative: false,
            signaling: true,
            payload: [0x20_0001]
        }
    );
}

#[test]
fn a_subnormal_input_reports_denormal_input_and_honors_daz() {
    let tiny = F32::from_bits(0x8000_0001);
    let (wide, flags): (F64, _) = tiny.convert_with(Env::IEEE);
    assert_eq!(wide.to_bits(), 0xB6A0_0000_0000_0000);
    assert_eq!(flags, Flags::DENORMAL_INPUT);
    let (zero, flags): (F64, _) = tiny.convert_with(Env::IEEE.with_denormals_are_zero(true));
    assert_eq!(
        (zero.to_bits(), flags),
        (0x8000_0000_0000_0000, Flags::DENORMAL_INPUT)
    );
}

#[test]
fn nan_inputs_follow_the_nan_rule() {
    let signaling = F64::from_bits(0xFFF4_0000_0000_0001);
    let (kept, flags): (F32, _) = signaling.convert_with(Env::IEEE);
    assert_eq!((kept.to_bits(), flags), (0xFFE0_0000, Flags::INVALID));
    let default =
        Env::IEEE.with_nan(NanRule::new(NanPropagation::DefaultNan).with_default_negative(true));
    let (replaced, flags): (F32, _) = signaling.convert_with(default);
    assert_eq!((replaced.to_bits(), flags), (0xFFC0_0000, Flags::INVALID));
    let quiet = F64::from_bits(0x7FF8_0000_0000_0000);
    let (fp8, flags): (F8E4M3, _) = quiet.convert_with(Env::IEEE);
    assert_eq!((fp8.to_bits(), flags), (0x7F, Flags::NONE));
}

#[test]
fn an_unsupported_x87_input_gives_the_default_nan() {
    let unnormal = F80::from_bits(0x3FFF_0000_0000_0000_0001);
    let (nan, flags): (F32, _) = unnormal.convert_with(Env::IEEE);
    assert_eq!((nan.to_bits(), flags), (0x7FC0_0000, Flags::INVALID));
}

#[test]
fn an_infinity_without_a_destination_infinity_is_invalid() {
    let infinity = F32::from_bits(0xFF80_0000);
    let (nan, flags): (F8E4M3, _) = infinity.convert_with(Env::IEEE);
    assert_eq!((nan.to_bits(), flags), (0xFF, Flags::INVALID));
    let (largest, flags): (F8E4M3, _) = infinity.convert_with(Env::IEEE.with_saturate(true));
    assert_eq!((largest.to_bits(), flags), (0xFE, Flags::INVALID));
    let (nan, _): (F8E4M3Fnuz, _) = infinity.convert_with(Env::IEEE);
    assert_eq!(nan.to_bits(), 0x80);
}

#[test]
fn a_rounding_override_keeps_the_other_fields() {
    let tiny = Exact {
        negative: false,
        exponent: -151,
        significand: [1],
        sticky: false,
    };
    let flush = Env::IEEE.with_flush_to_zero(true);
    let (value, _) = F32::round(tiny, flush.with_rounding(Rounding::TowardPositive));
    assert_eq!(value.to_bits(), 0, "flush-to-zero stays on");
    let (value, _) = F32::round(tiny, Rounding::TowardPositive);
    assert_eq!(value.to_bits(), 1, "the default mode does not flush");
    assert_eq!(F32::from_bits(1).with_mode::<mode::Ieee>().to_bits(), 1);
}

#[test]
fn aliases_have_the_published_parameters() {
    assert_eq!((F16::PRECISION, F16::EMAX, F16::EMIN), (11, 15, -14));
    assert_eq!((F32::PRECISION, F32::EMAX, F32::EMIN), (24, 127, -126));
    assert_eq!((F64::PRECISION, F64::EMAX, F64::EMIN), (53, 1023, -1022));
    assert_eq!(
        (F128::PRECISION, F128::EMAX, F128::EMIN),
        (113, 16383, -16382)
    );
    assert_eq!(
        (F256::PRECISION, F256::EMAX, F256::EMIN),
        (237, 262_143, -262_142)
    );
    assert_eq!(
        (F512::PRECISION, F512::EMAX, F512::EMIN),
        (489, 4_194_303, -4_194_302)
    );
    assert_eq!((BF16::PRECISION, BF16::EMAX, BF16::EMIN), (8, 127, -126));
    assert_eq!((TF32::PRECISION, TF32::EMAX, TF32::EMIN), (11, 127, -126));
    assert_eq!((F8E4M3::PRECISION, F8E4M3::EMAX, F8E4M3::EMIN), (4, 8, -6));
    assert_eq!(
        (F8E5M2::PRECISION, F8E5M2::EMAX, F8E5M2::EMIN),
        (3, 15, -14)
    );
    assert_eq!(
        (F8E4M3Fnuz::PRECISION, F8E4M3Fnuz::EMAX, F8E4M3Fnuz::EMIN),
        (4, 7, -7)
    );
    assert_eq!(
        (F8E5M2Fnuz::PRECISION, F8E5M2Fnuz::EMAX, F8E5M2Fnuz::EMIN),
        (3, 15, -15)
    );
    assert_eq!((F80::PRECISION, F80::EMAX, F80::EMIN), (64, 16383, -16382));
}

#[test]
fn sign_operations_change_only_the_sign_bit() {
    let nan = F32::from_bits(0x7FC0_0001);
    assert_eq!((-nan).to_bits(), 0xFFC0_0001);
    assert_eq!((-nan).abs().to_bits(), 0x7FC0_0001);
    assert_eq!(
        F32::from_bits(0x3F80_0000)
            .copy_sign(F32::from_bits(0x8000_0000))
            .to_bits(),
        0xBF80_0000
    );
    // FNUZ has one zero and one NaN.
    assert_eq!((-F8E4M3Fnuz::from_bits(0)).to_bits(), 0);
    assert_eq!(F8E4M3Fnuz::from_bits(0x80).abs().to_bits(), 0x80);
    assert_eq!((-F8E4M3Fnuz::from_bits(0x01)).to_bits(), 0x81);
    let unnormal = F80::from_bits(0x3FFF_0000_0000_0000_0001);
    assert_eq!((-unnormal).to_bits(), 0xBFFF_0000_0000_0000_0001);
}

#[test]
fn integers_convert_at_every_width() {
    use crate::integer::{Int, ToInt, UInt};

    assert_eq!(F32::from_int(i128::MIN).to_bits(), 0xFF00_0000);
    assert_eq!(F32::from_int(0_u8).to_bits(), 0);
    let (rounded, flags) = F32::from_int_with(u32::MAX, Rounding::TowardZero);
    assert_eq!((rounded.to_bits(), flags), (0x4F7F_FFFF, Flags::INEXACT));
    let largest = UInt::<512>::from_bits([u64::MAX; 8]);
    let (wide, flags) = F512::from_int_with(largest, Env::IEEE);
    assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
    assert_eq!(
        wide.to_int::<UInt<512>>(),
        ToInt::OutOfRange { negative: false }
    );
    let smallest = Int::<512>::from_bits([0, 0, 0, 0, 0, 0, 0, 1 << 63]);
    let exact = F512::from_int(smallest);
    assert_eq!(exact.to_int::<Int<512>>(), ToInt::Value(smallest));
}

#[test]
fn a_mode_value_and_its_env_give_the_same_results() {
    use crate::mode::direction::TowardZero;
    use crate::mode::{Rounded, X86Sse};

    type Truncating = Rounded<X86Sse, TowardZero>;

    let (a, b) = (
        F64::from_bits(0x3FB9_9999_9999_999A),
        F64::from_bits(0x3FC9_9999_9999_999A),
    );
    // A mode overrides the whole behavior, as its Env does.
    assert_eq!(a.add_with(b, X86Sse), a.add_with(b, Env::X86_SSE));
    let truncated = a.add_with(b, Truncating::default());
    assert_eq!(truncated, a.add_with(b, <Truncating as Mode>::ENV));
    assert_eq!(
        truncated,
        a.add_with(b, Env::X86_SSE.with_rounding(Rounding::TowardZero))
    );
    // A rounding override keeps the other fields of the mode of the type.
    let sse = a.with_mode::<X86Sse>();
    let (value, flags) = sse.add_with(b.with_mode::<X86Sse>(), Rounding::TowardZero);
    assert_eq!(
        (value.to_bits(), flags),
        (truncated.0.to_bits(), truncated.1)
    );
    // The override keeps the NaN rule of the mode: x86 gives a negative
    // default NaN, and the default mode a positive one.
    let infinity = F64::from_bits(0x7FF0_0000_0000_0000).with_mode::<X86Sse>();
    let (nan, flags) = infinity.sub_with(infinity, Rounding::TowardZero);
    assert_eq!(
        (nan.to_bits(), flags),
        (0xFFF8_0000_0000_0000, Flags::INVALID)
    );
    // The operator of a type with a mode uses the mode.
    let (x, y) = (a.with_mode::<Truncating>(), b.with_mode::<Truncating>());
    assert_eq!((x + y).to_bits(), truncated.0.to_bits());
}
