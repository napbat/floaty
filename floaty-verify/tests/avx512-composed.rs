//! Compares the AVX-512 scalar instructions that an emulator builds from
//! floaty operations with the host processor, in every MXCSR setting:
//! `VRNDSCALE`, `VREDUCE`, `VGETEXP`, `VGETMANT`, `VSCALEF`, `VRANGE`,
//! `VFIXUPIMM`, and `VFPCLASS` on binary32 and binary64, `VADDSS` with each
//! embedded rounding, the conversions to unsigned integers, and the
//! AVX-512 BF16 conversions and dot product. Each test holds the recipe of
//! its instructions. A recipe follows the operation that the Intel SDM
//! Volume 2, revision 325383-084US, gives for the instruction, or states
//! what the hardware shows.
//!
//! The test runs only on x86-64 hosts with AVX-512F and AVX-512DQ, and the
//! BF16 test only with AVX-512 BF16. On another processor, each test prints
//! that it checks nothing, as the checks of the x86-64-v4 host paths do.

#![cfg(target_arch = "x86_64")]

use floaty::{BF16, Env, F32, Flags, Rounding};
use floaty_verify::by_immediate;
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::avx512::{double, single};
use floaty_verify::x86::{
    self, SseSetting, composed, mxcsr_flags, sample_pairs as pairs, samples as operands, stored,
};

/// Returns `true` when the processor has the AVX-512 features of the tests,
/// and otherwise prints that the test checks nothing.
fn avx512(test: &str) -> bool {
    let present = std::arch::is_x86_feature_detected!("avx512f")
        && std::arch::is_x86_feature_detected!("avx512dq");
    if !present {
        println!("{test}: the processor lacks AVX-512F or AVX-512DQ; nothing checked");
    }
    present
}

/// The immediates of `VRNDSCALE` and `VREDUCE`: the scales 0, 1, 3, 8, and
/// 15 in bits 7 to 4, with each direction, the direction of MXCSR, and both
/// with the precision exception suppressed.
macro_rules! scale_immediates {
    ($module:ident::$function:ident, $immediate:expr, $arguments:tt) => {
        by_immediate!(
            $module::$function,
            $immediate,
            $arguments,
            [
                0x00, 0x01, 0x02, 0x03, 0x04, 0x08, 0x0C, 0x10, 0x11, 0x12, 0x13, 0x14, 0x18, 0x1C,
                0x30, 0x31, 0x32, 0x33, 0x34, 0x38, 0x3C, 0x80, 0x81, 0x82, 0x83, 0x84, 0x88, 0x8C,
                0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF8, 0xFC
            ]
        )
    };
}

/// The immediates of [`scale_immediates`].
const SCALE_IMMEDIATES: [u8; 35] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x08, 0x0C, 0x10, 0x11, 0x12, 0x13, 0x14, 0x18, 0x1C, 0x30, 0x31,
    0x32, 0x33, 0x34, 0x38, 0x3C, 0x80, 0x81, 0x82, 0x83, 0x84, 0x88, 0x8C, 0xF0, 0xF1, 0xF2, 0xF3,
    0xF4, 0xF8, 0xFC,
];

/// Calls `$module::$function::<N>$arguments` for `N` from 0 to 15.
macro_rules! nibble_immediates {
    ($module:ident::$function:ident, $immediate:expr, $arguments:tt) => {
        by_immediate!(
            $module::$function,
            $immediate,
            $arguments,
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        )
    };
}

/// The immediates of `VFIXUPIMM` and `VFPCLASS`: none, each single bit, and
/// all.
macro_rules! bit_immediates {
    ($module:ident::$function:ident, $immediate:expr, $arguments:tt) => {
        by_immediate!(
            $module::$function,
            $immediate,
            $arguments,
            [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF]
        )
    };
}

/// The immediates of [`bit_immediates`].
const BIT_IMMEDIATES: [u8; 10] = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0xFF];
/// An instruction of [`composed`] on three `xmm` operands under MXCSR.
type Instruction = fn(u128, u128, u128, u32) -> (u128, u32);

/// The low 32 or 64 bits of a 128-bit result.
fn low(result: u128, width: u32) -> u64 {
    u64::try_from(result & ((1 << width) - 1)).expect("the mask keeps at most 64 bits")
}

/// Checks one instruction on one operand: the low lane and the flags.
fn check(
    name: &str,
    input: &str,
    setting: SseSetting,
    ours: (u64, Flags),
    theirs: (u64, u32),
    nan: bool,
) {
    let context = format!("{name} {input} {setting:?} {:?}", ours.1);
    assert_eq!(ours.0, theirs.0, "{context}: result");
    assert_eq!(
        mxcsr_flags(ours.1, setting.daz, nan),
        theirs.1,
        "{context}: flags"
    );
}

#[test]
fn vrndscale_and_vreduce_round_to_a_scaled_integer() {
    if !avx512("vrndscale_and_vreduce_round_to_a_scaled_integer") {
        return;
    }
    let single_operands = operands(Layout::BINARY32, 3_000, 0x5CA1);
    let double_operands = operands(Layout::BINARY64, 3_000, 0x5CA2);
    for setting in x86::sse_settings() {
        let control = setting.control;
        for immediate in SCALE_IMMEDIATES {
            for &a in &single_operands {
                let x = single::value(a & 0xFFFF_FFFF);
                let input = format!("{immediate:#04x} {:#010x}", x.to_bits());
                let wide = u128::from(x.to_bits());
                let (ours, flags) = single::rndscale(x, immediate, &setting);
                let theirs =
                    scale_immediates!(composed::vrndscaless, immediate, (0, wide, 0, control));
                check(
                    "vrndscaless",
                    &input,
                    setting,
                    (u64::from(ours), flags),
                    (low(theirs.0, 32), theirs.1),
                    x.is_nan(),
                );
                let (ours, flags) = single::reduce(x, immediate, &setting);
                let theirs =
                    scale_immediates!(composed::vreducess, immediate, (0, wide, 0, control));
                check(
                    "vreducess",
                    &input,
                    setting,
                    (u64::from(ours), flags),
                    (low(theirs.0, 32), theirs.1),
                    x.is_nan(),
                );
            }
            for &a in &double_operands {
                let x = double::value(a);
                let input = format!("{immediate:#04x} {a:#018x}");
                let (ours, flags) = double::rndscale(x, immediate, &setting);
                let theirs = scale_immediates!(
                    composed::vrndscalesd,
                    immediate,
                    (0, u128::from(a), 0, control)
                );
                check(
                    "vrndscalesd",
                    &input,
                    setting,
                    (ours, flags),
                    (low(theirs.0, 64), theirs.1),
                    x.is_nan(),
                );
                let (ours, flags) = double::reduce(x, immediate, &setting);
                let theirs = scale_immediates!(
                    composed::vreducesd,
                    immediate,
                    (0, u128::from(a), 0, control)
                );
                check(
                    "vreducesd",
                    &input,
                    setting,
                    (ours, flags),
                    (low(theirs.0, 64), theirs.1),
                    x.is_nan(),
                );
            }
        }
    }
}

#[test]
fn vgetexp_and_vgetmant_split_a_value() {
    if !avx512("vgetexp_and_vgetmant_split_a_value") {
        return;
    }
    let single_operands = operands(Layout::BINARY32, 20_000, 0x6E7A);
    let double_operands = operands(Layout::BINARY64, 20_000, 0x6E7B);
    for setting in x86::sse_settings() {
        let (control, daz) = (setting.control, setting.daz);
        for &a in &single_operands {
            let x = single::value(a & 0xFFFF_FFFF);
            let wide = u128::from(x.to_bits());
            let input = format!("{:#010x}", x.to_bits());
            let (ours, flags) = single::getexp(x, daz);
            let theirs = composed::vgetexpss(0, wide, 0, control);
            check(
                "vgetexpss",
                &input,
                setting,
                (u64::from(ours), flags),
                (low(theirs.0, 32), theirs.1),
                x.is_nan(),
            );
            for immediate in 0..16_u8 {
                let (ours, flags) = single::getmant(x, immediate, daz);
                let theirs =
                    nibble_immediates!(composed::vgetmantss, immediate, (0, wide, 0, control));
                let input = format!("{immediate} {input}");
                check(
                    "vgetmantss",
                    &input,
                    setting,
                    (u64::from(ours), flags),
                    (low(theirs.0, 32), theirs.1),
                    x.is_nan(),
                );
            }
        }
        for &a in &double_operands {
            let x = double::value(a);
            let input = format!("{a:#018x}");
            let (ours, flags) = double::getexp(x, daz);
            let theirs = composed::vgetexpsd(0, u128::from(a), 0, control);
            check(
                "vgetexpsd",
                &input,
                setting,
                (ours, flags),
                (low(theirs.0, 64), theirs.1),
                x.is_nan(),
            );
            for immediate in 0..16_u8 {
                let (ours, flags) = double::getmant(x, immediate, daz);
                let theirs = nibble_immediates!(
                    composed::vgetmantsd,
                    immediate,
                    (0, u128::from(a), 0, control)
                );
                let input = format!("{immediate} {input}");
                check(
                    "vgetmantsd",
                    &input,
                    setting,
                    (ours, flags),
                    (low(theirs.0, 64), theirs.1),
                    x.is_nan(),
                );
            }
        }
    }
}

#[test]
fn vscalef_and_vrange_combine_two_values() {
    if !avx512("vscalef_and_vrange_combine_two_values") {
        return;
    }
    let single_operands = operands(Layout::BINARY32, 2_000, 0x5CA7);
    let double_operands = operands(Layout::BINARY64, 2_000, 0x5CA8);
    let single_pairs = pairs(&single_operands, 100, 20_000, 0x5CA9);
    let double_pairs = pairs(&double_operands, 100, 20_000, 0x5CAA);
    for setting in x86::sse_settings() {
        let (control, daz) = (setting.control, setting.daz);
        for &(a, b) in &single_pairs {
            let (x, y) = (
                single::value(a & 0xFFFF_FFFF),
                single::value(b & 0xFFFF_FFFF),
            );
            let (wide_x, wide_y) = (u128::from(x.to_bits()), u128::from(y.to_bits()));
            let input = format!("{:#010x} {:#010x}", x.to_bits(), y.to_bits());
            let nan = x.is_nan() || y.is_nan();
            let (ours, flags) = single::scalef(x, y, &setting);
            let theirs = composed::vscalefss(wide_x, wide_y, 0, control);
            check(
                "vscalefss",
                &input,
                setting,
                (u64::from(ours), flags),
                (low(theirs.0, 32), theirs.1),
                nan,
            );
            for immediate in 0..16_u8 {
                let (ours, flags) = single::range(x, y, immediate, daz);
                let theirs =
                    nibble_immediates!(composed::vrangess, immediate, (wide_x, wide_y, 0, control));
                let input = format!("{immediate} {input}");
                check(
                    "vrangess",
                    &input,
                    setting,
                    (u64::from(ours), flags),
                    (low(theirs.0, 32), theirs.1),
                    nan,
                );
            }
        }
        for &(a, b) in &double_pairs {
            let (x, y) = (double::value(a), double::value(b));
            let input = format!("{a:#018x} {b:#018x}");
            let nan = x.is_nan() || y.is_nan();
            let (ours, flags) = double::scalef(x, y, &setting);
            let theirs = composed::vscalefsd(u128::from(a), u128::from(b), 0, control);
            check(
                "vscalefsd",
                &input,
                setting,
                (ours, flags),
                (low(theirs.0, 64), theirs.1),
                nan,
            );
            for immediate in 0..16_u8 {
                let (ours, flags) = double::range(x, y, immediate, daz);
                let theirs = nibble_immediates!(
                    composed::vrangesd,
                    immediate,
                    (u128::from(a), u128::from(b), 0, control)
                );
                let input = format!("{immediate} {input}");
                check(
                    "vrangesd",
                    &input,
                    setting,
                    (ours, flags),
                    (low(theirs.0, 64), theirs.1),
                    nan,
                );
            }
        }
    }
}

#[test]
fn vfixupimm_and_vfpclass_read_the_class_of_a_value() {
    if !avx512("vfixupimm_and_vfpclass_read_the_class_of_a_value") {
        return;
    }
    let single_operands = operands(Layout::BINARY32, 3_000, 0xF1C5);
    let double_operands = operands(Layout::BINARY64, 3_000, 0xF1C6);
    let mut random = SplitMix64::new(0xF1C7);
    for setting in x86::sse_settings() {
        let (control, daz) = (setting.control, setting.daz);
        for immediate in BIT_IMMEDIATES {
            for &a in &single_operands {
                let x = single::value(a & 0xFFFF_FFFF);
                let (destination, table) = (
                    u32::try_from(random.next_u64() >> 32).expect("32 bits"),
                    u32::try_from(random.next_u64() >> 32).expect("32 bits"),
                );
                let input = format!("{immediate:#04x} {:#010x} table {table:#010x}", x.to_bits());
                let (ours, flags) = single::fixupimm(destination, x, table, immediate, daz);
                let theirs = bit_immediates!(
                    composed::vfixupimmss,
                    immediate,
                    (
                        u128::from(destination),
                        u128::from(x.to_bits()),
                        u128::from(table),
                        control
                    )
                );
                assert_eq!(
                    u64::from(ours),
                    low(theirs.0, 32),
                    "vfixupimmss {input} {setting:?}: result"
                );
                assert_eq!(
                    mxcsr_flags(flags, daz, false),
                    theirs.1,
                    "vfixupimmss {input} {setting:?}: flags"
                );
                let ours = single::fpclass(x, immediate, daz);
                let theirs = bit_immediates!(
                    composed::vfpclassss,
                    immediate,
                    (u128::from(x.to_bits()), control)
                );
                assert_eq!(ours, theirs, "vfpclassss {input} {setting:?}");
            }
            for &a in &double_operands {
                let x = double::value(a);
                let (destination, table) = (
                    random.next_u64(),
                    u32::try_from(random.next_u64() >> 32).expect("32 bits"),
                );
                let input = format!("{immediate:#04x} {a:#018x} table {table:#010x}");
                let (ours, flags) = double::fixupimm(destination, x, table, immediate, daz);
                let theirs = bit_immediates!(
                    composed::vfixupimmsd,
                    immediate,
                    (
                        u128::from(destination),
                        u128::from(a),
                        u128::from(table),
                        control
                    )
                );
                assert_eq!(
                    ours,
                    low(theirs.0, 64),
                    "vfixupimmsd {input} {setting:?}: result"
                );
                assert_eq!(
                    mxcsr_flags(flags, daz, false),
                    theirs.1,
                    "vfixupimmsd {input} {setting:?}: flags"
                );
                let ours = double::fpclass(x, immediate, daz);
                let theirs =
                    bit_immediates!(composed::vfpclasssd, immediate, (u128::from(a), control));
                assert_eq!(ours, theirs, "vfpclasssd {input} {setting:?}");
            }
        }
    }
}

/// `VADDSS` with an embedded rounding (Volume 2A, section 2.7.5) rounds in
/// the direction of the encoding, ignores MXCSR.RC, and suppresses every
/// flag. DAZ and FTZ still apply.
#[test]
fn embedded_rounding_overrides_mxcsr_and_suppresses_flags() {
    if !avx512("embedded_rounding_overrides_mxcsr_and_suppresses_flags") {
        return;
    }
    let single_operands = operands(Layout::BINARY32, 2_000, 0xE4B0);
    let single_pairs = pairs(&single_operands, 100, 20_000, 0xE4B1);
    let instructions: [(Rounding, Instruction); 4] = [
        (Rounding::TiesToEven, composed::vaddss_nearest),
        (Rounding::TowardNegative, composed::vaddss_down),
        (Rounding::TowardPositive, composed::vaddss_up),
        (Rounding::TowardZero, composed::vaddss_zero),
    ];
    for setting in x86::sse_settings() {
        for &(a, b) in &single_pairs {
            let (x, y) = (
                single::value(a & 0xFFFF_FFFF),
                single::value(b & 0xFFFF_FFFF),
            );
            for (rounding, instruction) in instructions {
                let (ours, _) = x.add_with(y, setting.env().with_rounding(rounding));
                let (theirs, flags) = instruction(
                    u128::from(x.to_bits()),
                    u128::from(y.to_bits()),
                    0,
                    setting.control,
                );
                let context = format!("vaddss {rounding:?} {a:#010x} {b:#010x} {setting:?}");
                assert_eq!(
                    u64::from(ours.to_bits()),
                    low(theirs, 32),
                    "{context}: result"
                );
                assert_eq!(flags, 0, "{context}: flags");
            }
        }
    }
}

/// `VCVTSS2USI`, `VCVTTSS2USI`, `VCVTSD2USI`, and `VCVTTSD2USI` (Volume 2B,
/// pages 5-120 to 5-140) give the unsigned integer indefinite, all ones,
/// for a value out of range or a NaN, and never signal DE, as the signed
/// conversions.
#[test]
fn unsigned_conversions_give_all_ones_out_of_range() {
    if !avx512("unsigned_conversions_give_all_ones_out_of_range") {
        return;
    }
    let single_operands = operands(Layout::BINARY32, 40_000, 0x0C57);
    let double_operands = operands(Layout::BINARY64, 40_000, 0x0C58);
    for setting in x86::sse_settings() {
        let control = setting.control;
        for &a in &single_operands {
            let x = single::value(a & 0xFFFF_FFFF);
            let wide = u128::from(x.to_bits());
            for (name, rounding, (theirs, theirs_flags)) in [
                (
                    "vcvtss2usi",
                    setting.rounding,
                    composed::vcvtss2usi32(wide, control),
                ),
                (
                    "vcvttss2usi",
                    Rounding::TowardZero,
                    composed::vcvttss2usi32(wide, control),
                ),
            ] {
                let (ours, flags) = x.to_int_with::<u32>(setting.env().with_rounding(rounding));
                let context = format!(
                    "{name} {:#010x} {setting:?} {ours:?} {flags:?}",
                    x.to_bits()
                );
                assert_eq!(stored(ours, u32::MAX), theirs, "{context}: result");
                let flags =
                    mxcsr_flags(flags.difference(Flags::DENORMAL_INPUT), setting.daz, false);
                assert_eq!(flags, theirs_flags, "{context}: flags");
            }
        }
        for &a in &double_operands {
            let x = double::value(a);
            for (name, rounding, (theirs, theirs_flags)) in [
                (
                    "vcvtsd2usi",
                    setting.rounding,
                    composed::vcvtsd2usi64(u128::from(a), control),
                ),
                (
                    "vcvttsd2usi",
                    Rounding::TowardZero,
                    composed::vcvttsd2usi64(u128::from(a), control),
                ),
            ] {
                let (ours, flags) = x.to_int_with::<u64>(setting.env().with_rounding(rounding));
                let context = format!("{name} {a:#018x} {setting:?} {ours:?} {flags:?}");
                assert_eq!(stored(ours, u64::MAX), theirs, "{context}: result");
                let flags =
                    mxcsr_flags(flags.difference(Flags::DENORMAL_INPUT), setting.daz, false);
                assert_eq!(flags, theirs_flags, "{context}: flags");
            }
        }
    }
}

/// The behavior of the AVX-512 BF16 instructions (Volume 2B, pages 5-112
/// and 5-226): round to nearest even, a subnormal operand reads as zero, a
/// tiny result flushes to zero, and MXCSR is neither read nor written. The
/// NaN rules are those of the SSE unit.
fn bf16_behavior() -> Env {
    Env::X86_SSE
        .with_rounding(Rounding::TiesToEven)
        .with_denormals_are_zero(true)
        .with_flush_to_zero(true)
}

/// Returns the binary32 value of a bfloat16 encoding, `make_fp32`.
fn widen(bits: u16) -> F32 {
    F32::from_bits(u32::from(bits) << 16)
}

/// Returns eight random bfloat16 encodings from `operands`.
fn bfloat_lanes(operands: &[u64], random: &mut SplitMix64) -> [u16; 8] {
    let length = u64::try_from(operands.len()).expect("a length fits a u64");
    core::array::from_fn(|_| {
        let index = usize::try_from(random.below(length)).expect("an index fits");
        u16::try_from(operands[index]).expect("bfloat16 has 16 bits")
    })
}

/// Returns four random binary32 encodings from `operands`.
fn single_lanes(operands: &[u64], random: &mut SplitMix64) -> [u32; 4] {
    let length = u64::try_from(operands.len()).expect("a length fits a u64");
    core::array::from_fn(|_| {
        let index = usize::try_from(random.below(length)).expect("an index fits");
        u32::try_from(operands[index]).expect("binary32 has 32 bits")
    })
}

/// Returns the 128-bit value of lanes of `width` bits, the lowest first.
fn pack<const N: usize>(lanes: [u32; N], width: u32) -> u128 {
    lanes.iter().enumerate().fold(0, |value, (lane, &bits)| {
        value | (u128::from(bits) << (width * u32::try_from(lane).expect("a lane fits")))
    })
}

/// Returns the 128-bit value of eight bfloat16 lanes, the lowest first.
fn pack16(lanes: [u16; 8]) -> u128 {
    pack(lanes.map(u32::from), 16)
}

/// Returns `true` when the processor has AVX-512 BF16, and otherwise prints
/// that the test checks nothing.
fn avx512_bf16(test: &str) -> bool {
    let present = std::arch::is_x86_feature_detected!("avx512bf16");
    if !present {
        println!("{test}: the processor lacks AVX-512 BF16; nothing checked");
    }
    present
}

/// `VCVTNEPS2BF16` rounds each binary32 lane to bfloat16 in the low 64
/// bits, and `VCVTNE2PS2BF16` rounds the lanes of its second source to the
/// low four bfloat16 lanes and those of its first source to the high four.
/// A NaN keeps its high bits and becomes quiet.
#[test]
fn bfloat16_conversions_round_to_nearest_even() {
    if !avx512_bf16("bfloat16_conversions_round_to_nearest_even") {
        return;
    }
    let env = bf16_behavior();
    let singles = operands(Layout::BINARY32, 4_000, 0xBF16);
    let mut random = SplitMix64::new(0xBF17);
    let narrow = |bits: u32| -> u32 {
        let (bfloat, _): (BF16, _) = F32::from_bits(bits).convert_with(env);
        u32::from(bfloat.to_bits())
    };
    for setting in x86::sse_settings() {
        let control = setting.control;
        for _ in 0..3_000 {
            let (first, second) = (
                single_lanes(&singles, &mut random),
                single_lanes(&singles, &mut random),
            );
            let one = composed::vcvtneps2bf16(0, pack(first, 32), 0, control);
            let context = format!("vcvtneps2bf16 {first:#010x?} {setting:?}");
            assert_eq!(one.0, pack(first.map(narrow), 16), "{context}: result");
            assert_eq!(one.1, 0, "{context}: flags");
            let two = composed::vcvtne2ps2bf16(0, pack(first, 32), pack(second, 32), control);
            let lanes: [u32; 8] = core::array::from_fn(|lane| {
                if lane < 4 {
                    narrow(second[lane])
                } else {
                    narrow(first[lane - 4])
                }
            });
            let context = format!("vcvtne2ps2bf16 {first:#010x?} {second:#010x?} {setting:?}");
            assert_eq!(two.0, pack(lanes, 16), "{context}: result");
            assert_eq!(two.1, 0, "{context}: flags");
        }
    }
}

/// Lanes on which the steps of `VDPBF16PS` decide the result: two fused
/// multiply-adds, the high pair first. Each entry is the accumulator and
/// the two bfloat16 pairs, the low pair first.
const DOT_PRODUCT_CASES: [(u32, [u16; 2], [u16; 2]); 6] = [
    // 2^-126 + 2^-70 * 2^-70: a separate product flushes to zero.
    (0x0080_0000, [0, 0x1C80], [0, 0x1C80]),
    // -1.5 * 2^127 + 2^64 * 2^64: a separate product overflows.
    (0xFF40_0000, [0, 0x5F80], [0, 0x5F80]),
    // 2^24 - 2^12 * 2^12 + 1 * 1: the low pair first loses the 1.
    (0x4B80_0000, [0x3F80, 0x4580], [0x3F80, 0xC580]),
    // -2^-126 + 2^-70 * 2^-70: a tiny sum flushes to a signed zero.
    (0x8080_0000, [0, 0x1C80], [0, 0x1C80]),
    // A subnormal accumulator and a subnormal factor read as zero.
    (0x0000_0001, [0x0001, 0x3F80], [0x3F80, 0x0040]),
    // A signaling NaN in the low pair and a quiet NaN accumulator.
    (0x7FC0_1234, [0xFF81, 0x3F80], [0x3F80, 0x3F80]),
];

/// `VDPBF16PS` adds to each binary32 lane the products of its two bfloat16
/// pairs by two fused multiply-adds, the high pair first, which gives the
/// NaN priority of Table 5-1. Products of bfloat16 values are exact in
/// binary32 unless they overflow or underflow, so [`DOT_PRODUCT_CASES`]
/// holds the lanes that tell fused steps from separate ones.
#[test]
fn vdpbf16ps_adds_two_fused_products() {
    if !avx512_bf16("vdpbf16ps_adds_two_fused_products") {
        return;
    }
    let env = bf16_behavior();
    let singles = operands(Layout::BINARY32, 4_000, 0xBF18);
    let bfloats = operands(Layout::BFLOAT16, 2_000, 0xBF19);
    let mut random = SplitMix64::new(0xBF1A);
    let mut groups: Vec<([u32; 4], [u16; 8], [u16; 8])> = DOT_PRODUCT_CASES
        .chunks(4)
        .map(|chunk| {
            let lane = |index: usize| chunk.get(index).copied().unwrap_or_default();
            let lanes: [(u32, [u16; 2], [u16; 2]); 4] = core::array::from_fn(lane);
            (
                lanes.map(|(accumulator, _, _)| accumulator),
                core::array::from_fn(|index| lanes[index / 2].1[index % 2]),
                core::array::from_fn(|index| lanes[index / 2].2[index % 2]),
            )
        })
        .collect();
    groups.extend((0..3_000).map(|_| {
        (
            single_lanes(&singles, &mut random),
            bfloat_lanes(&bfloats, &mut random),
            bfloat_lanes(&bfloats, &mut random),
        )
    }));
    for setting in x86::sse_settings() {
        for (accumulators, left, right) in &groups {
            let sums: [u32; 4] = core::array::from_fn(|lane| {
                let accumulator = F32::from_bits(accumulators[lane]);
                let (high, _) = widen(left[2 * lane + 1]).mul_add_with(
                    widen(right[2 * lane + 1]),
                    accumulator,
                    env,
                );
                let (sum, _) =
                    widen(left[2 * lane]).mul_add_with(widen(right[2 * lane]), high, env);
                sum.to_bits()
            });
            let dot = composed::vdpbf16ps(
                pack(*accumulators, 32),
                pack16(*left),
                pack16(*right),
                setting.control,
            );
            let context =
                format!("vdpbf16ps {accumulators:#010x?} {left:#06x?} {right:#06x?} {setting:?}");
            assert_eq!(dot.0, pack(sums, 32), "{context}: result");
            assert_eq!(dot.1, 0, "{context}: flags");
        }
    }
}
