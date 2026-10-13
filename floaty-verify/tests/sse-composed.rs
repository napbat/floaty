//! Compares the SSE, SSE3, SSE4.1, AVX, and F16C instructions that an
//! emulator builds from several floaty operations with the host processor,
//! in every MXCSR setting: the minimum and maximum, the compares with each
//! of the 32 AVX predicates, the horizontal sums, the dot products, and the
//! conversions between binary16 and binary32. Each test holds the recipe of
//! its instructions: the floaty operations and the selection that give the
//! result and the flags of the instruction. The test runs only on x86-64
//! hosts.
//!
//! A recipe cites the Intel SDM Volume 2, revision 325383-084US, for the
//! operation of an instruction, or states what the hardware shows.

#![cfg(target_arch = "x86_64")]

use core::cmp::Ordering;

use floaty::{Env, F16, F32, F64, Flags};
use floaty_verify::by_immediate;
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, MXCSR_ROUNDINGS, composed, mxcsr_flags, sample_pairs as pairs, samples as operands,
};

/// Returns the 128-bit value of four 32-bit lanes, the lowest first.
fn pack32(lanes: [u32; 4]) -> u128 {
    lanes.iter().enumerate().fold(0, |value, (lane, &bits)| {
        value | (u128::from(bits) << (32 * lane))
    })
}

/// Returns the 128-bit value of two 64-bit lanes, the lowest first.
fn pack64(lanes: [u64; 2]) -> u128 {
    u128::from(lanes[0]) | (u128::from(lanes[1]) << 64)
}

/// Returns `x` as an instruction under DAZ reads it: a subnormal reads as a
/// zero with its sign.
fn read32(x: F32, daz: bool) -> F32 {
    if daz && x.is_subnormal() {
        F32::from_bits(x.to_bits() & 0x8000_0000)
    } else {
        x
    }
}

/// As [`read32`], for binary64.
fn read64(x: F64, daz: bool) -> F64 {
    if daz && x.is_subnormal() {
        F64::from_bits(x.to_bits() & 0x8000_0000_0000_0000)
    } else {
        x
    }
}

/// The MXCSR flags of one operation, as `mxcsr_flags` maps them, where
/// `nan` says that an operand is a NaN.
fn flags_of(flags: Flags, daz: bool, nan: bool) -> u32 {
    mxcsr_flags(flags, daz, nan)
}

/// `MINSS` and `MAXSS` (Volume 2B, pages 4-28 and 4-12): the result is the
/// first operand when it compares below (or above) the second, and the
/// second operand otherwise: for two zeros, for any NaN, and for equal
/// values. A NaN result is the second operand as it is, not quieted. The
/// compare is signaling: any NaN signals invalid. Under DAZ, a subnormal
/// operand reads as a zero, also as the result.
fn select<T: Copy>(
    order: (Option<Ordering>, Flags),
    wanted: Ordering,
    (first, second): (T, T),
) -> (T, Flags) {
    let (order, flags) = order;
    let result = if order == Some(wanted) { first } else { second };
    (result, flags)
}

#[test]
fn minss_maxss_minsd_and_maxsd_select_by_the_signaling_compare() {
    let single = operands(Layout::BINARY32, 2_000, 0x4D1A);
    let double = operands(Layout::BINARY64, 2_000, 0x4D1B);
    let single_pairs = pairs(&single, 120, 20_000, 0x4D1C);
    let double_pairs = pairs(&double, 120, 20_000, 0x4D1D);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for &(a, b) in &single_pairs {
            let (x, y) = (F32::from_bits(narrow32(a)), F32::from_bits(narrow32(b)));
            let operands = (read32(x, daz), read32(y, daz));
            let nan = x.is_nan() || y.is_nan();
            for (name, wanted, instruction) in [
                ("minss", Ordering::Less, composed::minss as Instruction),
                ("maxss", Ordering::Greater, composed::maxss),
            ] {
                let (result, flags) = select(x.compare_signaling_with(y, env), wanted, operands);
                let (expected, expected_flags) =
                    instruction(u128::from(a), u128::from(b), 0, control);
                let context = format!("{name} {a:#010x} {b:#010x} {env:?} {flags:?}");
                assert_eq!(u128::from(result.to_bits()), expected, "{context}: result");
                assert_eq!(
                    flags_of(flags, daz, nan),
                    expected_flags,
                    "{context}: flags"
                );
            }
        }
        for &(a, b) in &double_pairs {
            let (x, y) = (F64::from_bits(a), F64::from_bits(b));
            let operands = (read64(x, daz), read64(y, daz));
            let nan = x.is_nan() || y.is_nan();
            for (name, wanted, instruction) in [
                ("minsd", Ordering::Less, composed::minsd as Instruction),
                ("maxsd", Ordering::Greater, composed::maxsd),
            ] {
                let (result, flags) = select(x.compare_signaling_with(y, env), wanted, operands);
                let (expected, expected_flags) =
                    instruction(u128::from(a), u128::from(b), 0, control);
                let context = format!("{name} {a:#018x} {b:#018x} {env:?} {flags:?}");
                assert_eq!(u128::from(result.to_bits()), expected, "{context}: result");
                assert_eq!(
                    flags_of(flags, daz, nan),
                    expected_flags,
                    "{context}: flags"
                );
            }
        }
    }
}

/// An instruction of [`composed`] on three `xmm` operands under MXCSR.
type Instruction = fn(u128, u128, u128, u32) -> (u128, u32);

/// An instruction, and the bits and the MXCSR flags of each of its `N`
/// lanes, as a recipe gives them.
type Case<T, const N: usize> = (&'static str, Instruction, [(T, u32); N]);

/// Returns the low 32 bits of `bits`.
fn narrow32(bits: u64) -> u32 {
    u32::try_from(bits & 0xFFFF_FFFF).expect("the mask keeps 32 bits")
}

/// The 32 predicates of `VCMPSS` and `VCMPSD` (Volume 2A, Table 3-1 on page
/// 3-180): whether each holds when the first operand is less than, equal
/// to, greater than, or unordered with the second, and whether it signals
/// invalid for a quiet NaN. Predicates 16 to 31 are 0 to 15 with the other
/// signaling.
const PREDICATES: [([bool; 4], bool); 16] = [
    ([false, true, false, false], false),  // EQ_OQ
    ([true, false, false, false], true),   // LT_OS
    ([true, true, false, false], true),    // LE_OS
    ([false, false, false, true], false),  // UNORD_Q
    ([true, false, true, true], false),    // NEQ_UQ
    ([false, true, true, true], true),     // NLT_US
    ([false, false, true, true], true),    // NLE_US
    ([true, true, true, false], false),    // ORD_Q
    ([false, true, false, true], false),   // EQ_UQ
    ([true, false, false, true], true),    // NGE_US
    ([true, true, false, true], true),     // NGT_US
    ([false, false, false, false], false), // FALSE_OQ
    ([true, false, true, false], false),   // NEQ_OQ
    ([false, true, true, false], true),    // GE_OS
    ([false, false, true, false], true),   // GT_OS
    ([true, true, true, true], false),     // TRUE_UQ
];

/// Returns whether the predicate `immediate` holds for `order`, and whether
/// it signals invalid for a quiet NaN.
fn predicate(immediate: u8, order: Option<Ordering>) -> (bool, bool) {
    let (truth, signaling) = PREDICATES[usize::from(immediate & 15)];
    let holds = match order {
        Some(Ordering::Less) => truth[0],
        Some(Ordering::Equal) => truth[1],
        Some(Ordering::Greater) => truth[2],
        None => truth[3],
    };
    (holds, signaling != (immediate >= 16))
}

/// Calls `$function::<N>(args)` for the runtime value `$immediate` from 0
/// to 31.
macro_rules! seq_32 {
    ($module:ident::$function:ident, $immediate:expr, $arguments:tt) => {
        by_immediate!(
            $module::$function,
            $immediate,
            $arguments,
            [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31
            ]
        )
    };
}

/// `VCMPSS` and `VCMPSD` give all ones in the low lane when the predicate
/// holds and zero otherwise, and keep the upper lanes of the first operand.
/// A quiet compare signals invalid for a signaling NaN, and a signaling one
/// for any NaN.
#[test]
fn vcmpss_and_vcmpsd_give_every_predicate() {
    assert!(
        std::arch::is_x86_feature_detected!("avx"),
        "the compare test needs a host with AVX"
    );
    let single = operands(Layout::BINARY32, 1_000, 0xC3A0);
    let double = operands(Layout::BINARY64, 1_000, 0xC3A1);
    let single_pairs = pairs(&single, 60, 4_000, 0xC3A2);
    let double_pairs = pairs(&double, 60, 4_000, 0xC3A3);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for immediate in 0..32_u8 {
            for &(a, b) in &single_pairs {
                let (x, y) = (F32::from_bits(narrow32(a)), F32::from_bits(narrow32(b)));
                let (order, quiet_flags) = x.compare_quiet_with(y, env);
                let (holds, signaling) = predicate(immediate, order);
                let flags = if signaling {
                    x.compare_signaling_with(y, env).1
                } else {
                    quiet_flags
                };
                let (expected, expected_flags) = seq_32!(
                    composed::vcmpss,
                    immediate,
                    (u128::from(a), u128::from(b), 0, control)
                );
                let context = format!("vcmpss {immediate} {a:#010x} {b:#010x} {env:?} {flags:?}");
                let mask = if holds { 0xFFFF_FFFF } else { 0 };
                assert_eq!(expected, mask, "{context}: result");
                let nan = x.is_nan() || y.is_nan();
                assert_eq!(
                    flags_of(flags, daz, nan),
                    expected_flags,
                    "{context}: flags"
                );
            }
            for &(a, b) in &double_pairs {
                let (x, y) = (F64::from_bits(a), F64::from_bits(b));
                let (order, quiet_flags) = x.compare_quiet_with(y, env);
                let (holds, signaling) = predicate(immediate, order);
                let flags = if signaling {
                    x.compare_signaling_with(y, env).1
                } else {
                    quiet_flags
                };
                let (expected, expected_flags) = seq_32!(
                    composed::vcmpsd,
                    immediate,
                    (u128::from(a), u128::from(b), 0, control)
                );
                let context = format!("vcmpsd {immediate} {a:#018x} {b:#018x} {env:?} {flags:?}");
                let mask = if holds { u128::from(u64::MAX) } else { 0 };
                assert_eq!(expected, mask, "{context}: result");
                let nan = x.is_nan() || y.is_nan();
                assert_eq!(
                    flags_of(flags, daz, nan),
                    expected_flags,
                    "{context}: flags"
                );
            }
        }
    }
}

/// Returns four random binary32 lanes from `operands`.
fn vector32(operands: &[u64], random: &mut SplitMix64) -> [u32; 4] {
    let length = u64::try_from(operands.len()).expect("a length fits a u64");
    core::array::from_fn(|_| {
        narrow32(operands[usize::try_from(random.below(length)).expect("an index fits a usize")])
    })
}

/// Returns two random binary64 lanes from `operands`.
fn vector64(operands: &[u64], random: &mut SplitMix64) -> [u64; 2] {
    let length = u64::try_from(operands.len()).expect("a length fits a u64");
    core::array::from_fn(|_| {
        operands[usize::try_from(random.below(length)).expect("an index fits a usize")]
    })
}

/// Adds or subtracts two binary32 lanes, and returns the bits and the MXCSR
/// flags of the operation.
fn step32(x: u32, y: u32, subtract: bool, env: Env, daz: bool) -> (u32, u32) {
    let (x, y) = (F32::from_bits(x), F32::from_bits(y));
    let (result, flags) = if subtract {
        x.sub_with(y, env)
    } else {
        x.add_with(y, env)
    };
    (
        result.to_bits(),
        flags_of(flags, daz, x.is_nan() || y.is_nan()),
    )
}

/// As [`step32`], for binary64.
fn step64(x: u64, y: u64, subtract: bool, env: Env, daz: bool) -> (u64, u32) {
    let (x, y) = (F64::from_bits(x), F64::from_bits(y));
    let (result, flags) = if subtract {
        x.sub_with(y, env)
    } else {
        x.add_with(y, env)
    };
    (
        result.to_bits(),
        flags_of(flags, daz, x.is_nan() || y.is_nan()),
    )
}

/// `HADDPS`, `HSUBPS`, `ADDSUBPS`, `HADDPD`, and `ADDSUBPD` (Volume 2A,
/// pages 3-469, 3-474, and 3-41) round each sum or difference of a pair of
/// lanes, with the flags of every lane. `HADDPS` and `HSUBPS` pair the
/// adjacent lanes of the first operand, then those of the second.
#[test]
fn horizontal_sums_round_each_lane() {
    assert!(
        std::arch::is_x86_feature_detected!("sse3"),
        "the horizontal test needs a host with SSE3"
    );
    let single = operands(Layout::BINARY32, 2_000, 0x4ADD);
    let double = operands(Layout::BINARY64, 2_000, 0x4ADE);
    let mut random = SplitMix64::new(0x4ADF);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for _ in 0..4_000 {
            let (x, y) = (
                vector32(&single, &mut random),
                vector32(&single, &mut random),
            );
            let pair =
                |first: u32, second: u32, subtract| step32(first, second, subtract, env, daz);
            let cases: [Case<u32, 4>; 3] = [
                (
                    "haddps",
                    composed::haddps,
                    [
                        pair(x[0], x[1], false),
                        pair(x[2], x[3], false),
                        pair(y[0], y[1], false),
                        pair(y[2], y[3], false),
                    ],
                ),
                (
                    "hsubps",
                    composed::hsubps,
                    [
                        pair(x[0], x[1], true),
                        pair(x[2], x[3], true),
                        pair(y[0], y[1], true),
                        pair(y[2], y[3], true),
                    ],
                ),
                (
                    "addsubps",
                    composed::addsubps,
                    [
                        pair(x[0], y[0], true),
                        pair(x[1], y[1], false),
                        pair(x[2], y[2], true),
                        pair(x[3], y[3], false),
                    ],
                ),
            ];
            for (name, instruction, lanes) in cases {
                let (expected, expected_flags) = instruction(pack32(x), pack32(y), 0, control);
                let result = pack32(lanes.map(|(bits, _)| bits));
                let flags = lanes.iter().fold(0, |union, &(_, flags)| union | flags);
                let context = format!("{name} {x:#010x?} {y:#010x?} {env:?}");
                assert_eq!(result, expected, "{context}: result");
                assert_eq!(flags, expected_flags, "{context}: flags");
            }
            let (x, y) = (
                vector64(&double, &mut random),
                vector64(&double, &mut random),
            );
            let pair =
                |first: u64, second: u64, subtract| step64(first, second, subtract, env, daz);
            let cases: [Case<u64, 2>; 2] = [
                (
                    "haddpd",
                    composed::haddpd,
                    [pair(x[0], x[1], false), pair(y[0], y[1], false)],
                ),
                (
                    "addsubpd",
                    composed::addsubpd,
                    [pair(x[0], y[0], true), pair(x[1], y[1], false)],
                ),
            ];
            for (name, instruction, lanes) in cases {
                let (expected, expected_flags) = instruction(pack64(x), pack64(y), 0, control);
                let result = pack64(lanes.map(|(bits, _)| bits));
                let flags = lanes.iter().fold(0, |union, &(_, flags)| union | flags);
                let context = format!("{name} {x:#018x?} {y:#018x?} {env:?}");
                assert_eq!(result, expected, "{context}: result");
                assert_eq!(flags, expected_flags, "{context}: flags");
            }
        }
    }
}

/// The immediates of the dot-product test: each of the 16 product masks in
/// bits 7 to 4, with the result masks 0xF and 0x5 in bits 3 to 0.
const DOT_IMMEDIATES: [u8; 32] = [
    0x0F, 0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F, 0x8F, 0x9F, 0xAF, 0xBF, 0xCF, 0xDF, 0xEF, 0xFF,
    0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75, 0x85, 0x95, 0xA5, 0xB5, 0xC5, 0xD5, 0xE5, 0xF5,
];

/// Calls `$function::<N>(args)` for each value of [`DOT_IMMEDIATES`].
macro_rules! dot_immediate {
    ($module:ident::$function:ident, $immediate:expr, $arguments:tt) => {
        by_immediate!(
            $module::$function,
            $immediate,
            $arguments,
            [
                0x0F, 0x1F, 0x2F, 0x3F, 0x4F, 0x5F, 0x6F, 0x7F, 0x8F, 0x9F, 0xAF, 0xBF, 0xCF, 0xDF,
                0xEF, 0xFF, 0x05, 0x15, 0x25, 0x35, 0x45, 0x55, 0x65, 0x75, 0x85, 0x95, 0xA5, 0xB5,
                0xC5, 0xD5, 0xE5, 0xF5
            ]
        )
    };
}

/// Multiplies two binary32 lanes, or gives +0 without a flag when the lane
/// is masked off.
fn product32(x: u32, y: u32, enabled: bool, env: Env, daz: bool) -> (u32, u32) {
    if !enabled {
        return (0, 0);
    }
    let (x, y) = (F32::from_bits(x), F32::from_bits(y));
    let (result, flags) = x.mul_with(y, env);
    (
        result.to_bits(),
        flags_of(flags, daz, x.is_nan() || y.is_nan()),
    )
}

/// As [`product32`], for binary64.
fn product64(x: u64, y: u64, enabled: bool, env: Env, daz: bool) -> (u64, u32) {
    if !enabled {
        return (0, 0);
    }
    let (x, y) = (F64::from_bits(x), F64::from_bits(y));
    let (result, flags) = x.mul_with(y, env);
    (
        result.to_bits(),
        flags_of(flags, daz, x.is_nan() || y.is_nan()),
    )
}

/// `DPPS` (Volume 2A, page 3-367): bits 7 to 4 of the immediate select the
/// lanes whose products enter, and a lane that does not enter is +0. The
/// sum is `(p0 + p1) + (p2 + p3)`, each step rounded. Bits 3 to 0 select the
/// lanes that receive the sum; the others are +0. `DPPD` takes bits 5 and 4
/// and bits 1 and 0 of the immediate, and the sum `p0 + p1`.
#[test]
fn dpps_and_dppd_round_each_step() {
    assert!(
        std::arch::is_x86_feature_detected!("sse4.1"),
        "the dot-product test needs a host with SSE4.1"
    );
    let single = operands(Layout::BINARY32, 2_000, 0xD0D0);
    let double = operands(Layout::BINARY64, 2_000, 0xD0D1);
    let mut random = SplitMix64::new(0xD0D2);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for immediate in DOT_IMMEDIATES {
            for _ in 0..400 {
                let (x, y) = (
                    vector32(&single, &mut random),
                    vector32(&single, &mut random),
                );
                let products: [(u32, u32); 4] = core::array::from_fn(|lane| {
                    product32(x[lane], y[lane], immediate & (16 << lane) != 0, env, daz)
                });
                let low = step32(products[0].0, products[1].0, false, env, daz);
                let high = step32(products[2].0, products[3].0, false, env, daz);
                let sum = step32(low.0, high.0, false, env, daz);
                let lanes = core::array::from_fn(|lane| {
                    if immediate & (1 << lane) != 0 {
                        sum.0
                    } else {
                        0
                    }
                });
                let flags = products
                    .iter()
                    .fold(low.1 | high.1 | sum.1, |union, p| union | p.1);
                let (expected, expected_flags) = dot_immediate!(
                    composed::dpps,
                    immediate,
                    (pack32(x), pack32(y), 0, control)
                );
                let context = format!("dpps {immediate:#04x} {x:#010x?} {y:#010x?} {env:?}");
                assert_eq!(pack32(lanes), expected, "{context}: result");
                assert_eq!(flags, expected_flags, "{context}: flags");

                let (x, y) = (
                    vector64(&double, &mut random),
                    vector64(&double, &mut random),
                );
                let products: [(u64, u32); 2] = core::array::from_fn(|lane| {
                    product64(x[lane], y[lane], immediate & (16 << lane) != 0, env, daz)
                });
                let sum = step64(products[0].0, products[1].0, false, env, daz);
                let lanes = core::array::from_fn(|lane| {
                    if immediate & (1 << lane) != 0 {
                        sum.0
                    } else {
                        0
                    }
                });
                let flags = products.iter().fold(sum.1, |union, p| union | p.1);
                let (expected, expected_flags) = dot_immediate!(
                    composed::dppd,
                    immediate,
                    (pack64(x), pack64(y), 0, control)
                );
                let context = format!("dppd {immediate:#04x} {x:#018x?} {y:#018x?} {env:?}");
                assert_eq!(pack64(lanes), expected, "{context}: result");
                assert_eq!(flags, expected_flags, "{context}: flags");
            }
        }
    }
}

/// `VCVTPH2PS` (Volume 2B, page 5-37) widens each binary16 lane exactly. The
/// hardware shows that it reads a binary16 subnormal as it is under DAZ, and
/// reports no DE for it: binary32 holds the value as a normal number.
/// `VCVTPS2PH` (page 5-41) rounds each binary32 lane to binary16 in the
/// direction of bits 1 and 0 of the immediate, or of MXCSR when bit 2 is
/// set. The hardware shows that it keeps a binary16 subnormal result under
/// FTZ.
#[test]
fn f16c_conversions_round_each_lane() {
    assert!(
        std::arch::is_x86_feature_detected!("f16c"),
        "the conversion test needs a host with F16C"
    );
    let halves = operands(Layout::BINARY16, 4_000, 0xF16C);
    let singles = operands(Layout::BINARY32, 4_000, 0xF16D);
    let mut random = SplitMix64::new(0xF16E);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for chunk in halves.chunks(4) {
            let lanes: [u16; 4] = core::array::from_fn(|lane| {
                u16::try_from(chunk.get(lane).copied().unwrap_or(0)).expect("binary16 has 16 bits")
            });
            let source = lanes.iter().enumerate().fold(0, |value, (lane, &bits)| {
                value | (u128::from(bits) << (16 * lane))
            });
            let converted: [(u32, u32); 4] = lanes.map(|bits| {
                let half = F16::from_bits(bits);
                let (single, flags): (F32, _) =
                    half.convert_with(env.with_denormals_are_zero(false));
                let flags = flags.difference(Flags::DENORMAL_INPUT);
                (single.to_bits(), flags_of(flags, daz, half.is_nan()))
            });
            let (expected, expected_flags) = composed::vcvtph2ps(0, source, 0, control);
            let context = format!("vcvtph2ps {lanes:#06x?} {env:?}");
            assert_eq!(
                pack32(converted.map(|(bits, _)| bits)),
                expected,
                "{context}: result"
            );
            let flags = converted.iter().fold(0, |union, &(_, flags)| union | flags);
            assert_eq!(flags, expected_flags, "{context}: flags");
        }
        for immediate in 0..5_u8 {
            let rounding = if immediate == 4 {
                env.rounding
            } else {
                MXCSR_ROUNDINGS
                    .iter()
                    .find(|&&(_, field)| field >> 13 == u32::from(immediate))
                    .expect("bits 1 and 0 select a direction")
                    .0
            };
            for _ in 0..2_000 {
                let lanes = vector32(&singles, &mut random);
                let narrowed: [(u16, u32); 4] = lanes.map(|bits| {
                    let single = F32::from_bits(bits);
                    let behavior = env.with_rounding(rounding).with_flush_to_zero(false);
                    let (half, flags): (F16, _) = single.convert_with(behavior);
                    (half.to_bits(), flags_of(flags, daz, single.is_nan()))
                });
                let result = narrowed
                    .iter()
                    .enumerate()
                    .fold(0, |value, (lane, &(bits, _))| {
                        value | (u128::from(bits) << (16 * lane))
                    });
                let (expected, expected_flags) = match immediate {
                    0 => composed::vcvtps2ph::<0>(0, pack32(lanes), 0, control),
                    1 => composed::vcvtps2ph::<1>(0, pack32(lanes), 0, control),
                    2 => composed::vcvtps2ph::<2>(0, pack32(lanes), 0, control),
                    3 => composed::vcvtps2ph::<3>(0, pack32(lanes), 0, control),
                    _ => composed::vcvtps2ph::<4>(0, pack32(lanes), 0, control),
                };
                let context = format!("vcvtps2ph {immediate} {lanes:#010x?} {env:?} {rounding:?}");
                assert_eq!(result, expected, "{context}: result");
                let flags = narrowed.iter().fold(0, |union, &(_, flags)| union | flags);
                assert_eq!(flags, expected_flags, "{context}: flags");
            }
        }
    }
}
