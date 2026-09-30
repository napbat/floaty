//! Compares conversions and arithmetic with the SSE unit of the host
//! processor: rounding, flush-to-zero (FTZ), denormals-are-zero (DAZ), the
//! first-operand NaN rule, and the MXCSR flags. The test runs only on x86-64
//! hosts.
//!
//! The arithmetic runs twice for each MXCSR setting: with the behavior as an
//! `Env` at run time, and with the static mode of the setting, which
//! `with_sse_mode!` selects as an emulator would.

#![cfg(target_arch = "x86_64")]

use std::hint::black_box;

use floaty::{BF16, Env, F16, F32, F64, mode};
use floaty_verify::encodings::{IntegerBit, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, MXCSR_DAZ, MXCSR_EXCEPTION_MASKS, MXCSR_FTZ, MXCSR_MASKED, MXCSR_TOWARD_NEGATIVE,
    MXCSR_TOWARD_POSITIVE, MXCSR_TOWARD_ZERO, mxcsr_flags,
};

/// Returns the low-bit patterns of `dropped` discarded bits that decide a
/// rounding: exact, just below halfway, halfway, just above halfway, and all
/// ones.
fn edge_patterns(dropped: u32) -> [u64; 5] {
    let half = 1_u64 << (dropped - 1);
    [0, half - 1, half, half + 1, (half << 1) - 1]
}

/// Returns binary64 encodings near every binary32 boundary, and random ones.
fn binary64_to_binary32_inputs() -> Vec<u64> {
    let mut random = SplitMix64::new(0x5EE5);
    let mut inputs = vec![
        0,
        0x8000_0000_0000_0000,
        0x7FF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,
        0x7FF4_0000_0000_0001,
        0x0000_0000_0000_0001,
    ];
    // Every edge pattern at the exponents where binary32 changes from zero to
    // subnormal, from subnormal to normal, and from normal to overflow.
    for exponent in (-152..=-122).chain(125..=129) {
        // A binary32 result keeps 24 bits, and fewer below 2^-126.
        let dropped = (29 + (-126 - exponent).max(0)).min(52);
        let biased = u64::try_from(1023 + exponent).expect("a normal binary64 exponent");
        for pattern in edge_patterns(u32::try_from(dropped).expect("at most 52")) {
            for _ in 0..8 {
                let high = (random.next_u64() >> 12) & !((1 << dropped) - 1);
                let sign = random.next_u64() << 63;
                inputs.push(sign | (biased << 52) | high | pattern);
            }
        }
    }
    // Biased exponents from below the binary32 subnormals to above its range.
    for _ in 0..200_000 {
        let exponent = 1023 - 160 + random.next_u64() % 300;
        let fraction = random.next_u64() >> 12;
        let sign = random.next_u64() << 63;
        inputs.push(sign | (exponent << 52) | fraction);
    }
    inputs
}

#[test]
fn cvtsd2ss_matches_in_every_mode() {
    let inputs = binary64_to_binary32_inputs();
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for &input in &inputs {
            let (expected, expected_flags) = x86::cvtsd2ss(input, 0, control);
            let (ours, flags): (F32, _) = F64::from_bits(input).convert_with(env);
            let context = format!("{input:#018x} {env:?} {flags:?}");
            assert_eq!(
                u64::from(ours.to_bits()),
                expected & 0xFFFF_FFFF,
                "{context}: result"
            );
            assert_eq!(
                mxcsr_flags(flags, daz, false),
                expected_flags,
                "{context}: flags"
            );
        }
    }
}

#[test]
fn cvtss2sd_matches_with_and_without_daz() {
    let mut random = SplitMix64::new(0x55D);
    let special = [
        0x0000_0001,
        0x8000_0001,
        0x007F_FFFF,
        0x7F80_0001,
        0x7FC0_0001,
        0xFF80_0000,
    ];
    let samples = (0..200_000).map(|_| u32::try_from(random.next_u64() >> 32).expect("32 bits"));
    let inputs: Vec<u32> = special.into_iter().chain(samples).collect();
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for &input in &inputs {
            let (expected, expected_flags) = x86::cvtss2sd(u64::from(input), 0, control);
            let (ours, flags): (F64, _) = F32::from_bits(input).convert_with(env);
            let context = format!("{input:#010x} {env:?} {flags:?}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(
                mxcsr_flags(flags, daz, false),
                expected_flags,
                "{context}: flags"
            );
        }
    }
}

/// The binary32 special operands: zeros, infinities, NaNs of both kinds and
/// signs, subnormals, the normal boundaries, and ones.
const SINGLE_SPECIALS: [u32; 18] = [
    0x0000_0000,
    0x8000_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0001,
    0xFFC0_0002,
    0x7F80_0003,
    0xFFA0_0004,
    0x0000_0001,
    0x807F_FFFF,
    0x0080_0000,
    0x8080_0001,
    0x7F7F_FFFF,
    0xFF7F_FFFE,
    0x3F80_0000,
    0xBF80_0000,
    0x4000_0000,
    0x3F80_0001,
];

/// Returns binary32 operands: the specials, random values across the range,
/// and values near the underflow and overflow boundaries.
fn single_operands(random: &mut SplitMix64, count: usize) -> Vec<u32> {
    let mut operands = SINGLE_SPECIALS.to_vec();
    for index in 0..count {
        let fraction = u32::try_from(random.next_u64() >> 41).expect("23 bits");
        let exponent = match index % 3 {
            0 => u32::try_from(random.next_u64() % 256).expect("8 bits"),
            1 => u32::try_from(random.next_u64() % 40).expect("below 40"),
            _ => 215 + u32::try_from(random.next_u64() % 40).expect("below 40"),
        };
        let sign = u32::try_from(random.next_u64() & 1).expect("one bit") << 31;
        operands.push(sign | (exponent.min(255) << 23) | fraction);
    }
    operands
}

/// Returns binary64 operands in the same way as `single_operands`.
fn double_operands(random: &mut SplitMix64, count: usize) -> Vec<u64> {
    let mut operands: Vec<u64> = vec![
        0,
        1 << 63,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,
        0xFFF4_0000_0000_0002,
        0x0000_0000_0000_0001,
        0x0010_0000_0000_0000,
        0x7FEF_FFFF_FFFF_FFFF,
        0x3FF0_0000_0000_0000,
    ];
    for index in 0..count {
        let fraction = random.next_u64() >> 12;
        let exponent = match index % 3 {
            0 => random.next_u64() % 2048,
            1 => random.next_u64() % 80,
            _ => 1970 + random.next_u64() % 78,
        };
        let sign = (random.next_u64() & 1) << 63;
        operands.push(sign | (exponent << 52) | fraction);
    }
    operands
}

/// Compares one SSE arithmetic instruction with floaty in every setting.
///
/// Under the MXCSR value at reset, the operator of a type with the SSE mode
/// also gives the result. The operators of binary32 and binary64 take the
/// host fast path, so this checks that path.
macro_rules! arithmetic {
    ($pairs:expr, $alias:ty, $instruction:path, $method:ident, $operator:tt) => {
        for setting in x86::sse_settings() {
            let (control, env, daz) = (setting.control, setting.env(), setting.daz);
            for &(a, b) in &$pairs {
                let (expected, expected_flags) = $instruction(a, b, control);
                let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
                let nan_operand = x.is_nan() || y.is_nan();
                let (ours, flags) = x.$method(y, env);
                let context = format!(
                    "{} {a:#x} {b:#x} {env:?} {flags:?}",
                    stringify!($instruction)
                );
                assert_eq!(ours.to_bits(), expected, "{context}: result");
                if control == MXCSR_MASKED {
                    let (x, y) = (x.with_mode::<mode::X86Sse>(), y.with_mode::<mode::X86Sse>());
                    assert_eq!((x $operator y).to_bits(), expected, "{context}: operator");
                }
                assert_eq!(
                    mxcsr_flags(flags, daz, nan_operand),
                    expected_flags,
                    "{context}: flags"
                );
                let (fixed, fixed_flags) = floaty_verify::with_sse_mode!(
                    env.rounding,
                    env.flush_to_zero,
                    env.denormals_are_zero,
                    Mode => {
                        assert_eq!(<Mode as floaty::env::Mode>::ENV, env, "{context}: mode");
                        let (x, y) = (x.with_mode::<Mode>(), y.with_mode::<Mode>());
                        let (value, flags) = x.$method(y, Mode::default());
                        (value.to_bits(), flags)
                    }
                );
                assert_eq!(
                    (fixed, mxcsr_flags(fixed_flags, daz, nan_operand)),
                    (expected, expected_flags),
                    "{context}: static mode"
                );
            }
        }
    };
}

/// Returns operand pairs: every pair of specials, random pairs, and pairs
/// that cancel in a sum.
fn pairs<T: Copy + core::ops::BitXor<Output = T>>(
    operands: &[T],
    specials: usize,
    sign: T,
    flip: T,
) -> Vec<(T, T)> {
    let mut pairs = Vec::new();
    for &a in &operands[..specials] {
        for &b in &operands[..specials] {
            pairs.push((a, b));
        }
    }
    for pair in operands[specials..].chunks_exact(2) {
        pairs.push((pair[0], pair[1]));
        // The same magnitude with the other sign and a changed low bit.
        pairs.push((pair[0], pair[0] ^ sign ^ flip));
    }
    pairs
}

#[test]
fn binary32_arithmetic_matches_in_every_mode() {
    let mut random = SplitMix64::new(0x0055_0032);
    let operands = single_operands(&mut random, 40_000);
    let pairs = pairs(&operands, SINGLE_SPECIALS.len(), 1 << 31, 1);
    arithmetic!(pairs, F32, x86::addss, add_with, +);
    arithmetic!(pairs, F32, x86::subss, sub_with, -);
    arithmetic!(pairs, F32, x86::mulss, mul_with, *);
    arithmetic!(pairs, F32, x86::divss, div_with, /);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for &a in &operands {
            let (expected, expected_flags) = x86::sqrtss(a, 0, control);
            let value = F32::from_bits(a);
            let (ours, flags) = value.sqrt_with(env);
            let context = format!("sqrtss {a:#x} {env:?} {flags:?}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(
                mxcsr_flags(flags, daz, value.is_nan()),
                expected_flags,
                "{context}: flags"
            );
            let (fixed, fixed_flags) = floaty_verify::with_sse_mode!(
                env.rounding,
                env.flush_to_zero,
                env.denormals_are_zero,
                Mode => {
                    assert_eq!(<Mode as floaty::env::Mode>::ENV, env, "{context}: mode");
                    let (root, flags) = value.with_mode::<Mode>().sqrt_with(Mode::default());
                    (root.to_bits(), flags)
                }
            );
            assert_eq!(
                (fixed, mxcsr_flags(fixed_flags, daz, value.is_nan())),
                (expected, expected_flags),
                "{context}: static mode"
            );
        }
    }
}

#[test]
fn binary64_arithmetic_matches_in_every_mode() {
    let mut random = SplitMix64::new(0x0055_0064);
    let operands = double_operands(&mut random, 40_000);
    let pairs = pairs(&operands, 10, 1 << 63, 1);
    arithmetic!(pairs, F64, x86::addsd, add_with, +);
    arithmetic!(pairs, F64, x86::subsd, sub_with, -);
    arithmetic!(pairs, F64, x86::mulsd, mul_with, *);
    arithmetic!(pairs, F64, x86::divsd, div_with, /);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for &a in &operands {
            let (expected, expected_flags) = x86::sqrtsd(a, 0, control);
            let value = F64::from_bits(a);
            let (ours, flags) = value.sqrt_with(env);
            let context = format!("sqrtsd {a:#x} {env:?} {flags:?}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(
                mxcsr_flags(flags, daz, value.is_nan()),
                expected_flags,
                "{context}: flags"
            );
            let (fixed, fixed_flags) = floaty_verify::with_sse_mode!(
                env.rounding,
                env.flush_to_zero,
                env.denormals_are_zero,
                Mode => {
                    assert_eq!(<Mode as floaty::env::Mode>::ENV, env, "{context}: mode");
                    let (root, flags) = value.with_mode::<Mode>().sqrt_with(Mode::default());
                    (root.to_bits(), flags)
                }
            );
            assert_eq!(
                (fixed, mxcsr_flags(fixed_flags, daz, value.is_nan())),
                (expected, expected_flags),
                "{context}: static mode"
            );
        }
    }
}

/// `VFMADD213` computes `b * a + c` from its operands `a`, `b`, and `c`, and
/// takes the first NaN in the order of that expression: `b`, `a`, `c`. So
/// floaty computes `b * a + c` too. Mapping an instruction's operand order
/// belongs to a consumer.
#[test]
fn fused_multiply_add_matches_in_every_mode() {
    assert!(
        std::arch::is_x86_feature_detected!("fma"),
        "the FMA test needs a host with FMA3"
    );
    let mut random = SplitMix64::new(0x00F3_A000);
    let single = single_operands(&mut random, 30_000);
    let double = double_operands(&mut random, 30_000);
    let specials = SINGLE_SPECIALS.len();
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for &a in &single[..specials] {
            for &b in &single[..specials] {
                for &c in &single[..specials] {
                    check_single(a, b, c, control, env, daz);
                }
            }
        }
        for triple in single[specials..].chunks_exact(3) {
            check_single(triple[0], triple[1], triple[2], control, env, daz);
            // An addend that cancels the product.
            let product = F32::from_bits(triple[0])
                .mul_with(F32::from_bits(triple[1]), Env::IEEE)
                .0;
            check_single(
                triple[0],
                triple[1],
                product.to_bits() ^ (1 << 31),
                control,
                env,
                daz,
            );
        }
        for triple in double.chunks_exact(3) {
            let (expected, expected_flags) =
                x86::vfmadd213sd(triple[0], triple[1], triple[2], control);
            let [a, b, c] = [triple[0], triple[1], triple[2]].map(F64::from_bits);
            let nan_operand = a.is_nan() || b.is_nan() || c.is_nan();
            let (ours, flags) = b.mul_add_with(a, c, env);
            let context = format!("vfmadd213sd {triple:x?} {env:?} {flags:?}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(
                mxcsr_flags(flags, daz, nan_operand),
                expected_flags,
                "{context}: flags"
            );
        }
    }
}

fn check_single(first: u32, second: u32, third: u32, control: u32, env: Env, daz: bool) {
    let (expected, expected_flags) = x86::vfmadd213ss(first, second, third, control);
    let [multiplicand, multiplier, addend] = [first, second, third].map(F32::from_bits);
    let nan_operand = multiplicand.is_nan() || multiplier.is_nan() || addend.is_nan();
    let (ours, flags) = multiplier.mul_add_with(multiplicand, addend, env);
    let context = format!("vfmadd213ss {first:#x} {second:#x} {third:#x} {env:?} {flags:?}");
    assert_eq!(ours.to_bits(), expected, "{context}: result");
    assert_eq!(
        mxcsr_flags(flags, daz, nan_operand),
        expected_flags,
        "{context}: flags"
    );
}

/// Checks the four operators, `sqrt`, and `mul_add` of one type under one
/// MXCSR value against the engine results of the default mode.
macro_rules! operators_under {
    ($alias:ty, $control:expr, $pairs:expr) => {
        for &(a, b) in $pairs {
            let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
            let env = <$alias>::ENV;
            let expected = [
                x.add_with(y, env).0.to_bits(),
                x.sub_with(y, env).0.to_bits(),
                x.mul_with(y, env).0.to_bits(),
                x.div_with(y, env).0.to_bits(),
                x.sqrt_with(env).0.to_bits(),
                x.mul_add_with(y, x, env).0.to_bits(),
            ];
            let ours = x86::with_mxcsr($control, || {
                let (x, y) = (black_box(x), black_box(y));
                [
                    (x + y).to_bits(),
                    (x - y).to_bits(),
                    (x * y).to_bits(),
                    (x / y).to_bits(),
                    x.sqrt().to_bits(),
                    x.mul_add(y, x).to_bits(),
                ]
            });
            assert_eq!(ours, expected, "{a:#x} {b:#x} under MXCSR {:#x}", $control);
        }
    };
}

/// Checks the remainder of one type under one MXCSR value against the engine
/// result of the default mode. The binary16 path widens and rounds in F16C,
/// which reads MXCSR, before `FPREM1`.
macro_rules! remainders_under {
    ($alias:ty, $control:expr, $pairs:expr) => {
        for &(a, b) in $pairs {
            let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
            let expected = x.remainder_with(y, <$alias>::ENV).0.to_bits();
            let ours = x86::with_mxcsr($control, || black_box(x).remainder(black_box(y)).to_bits());
            assert_eq!(ours, expected, "{a:#x} {b:#x} under MXCSR {:#x}", $control);
        }
    };
}

/// Checks `round_to_integral`, `to_int`, `from_int`, and the conversions to
/// binary16, binary32, and binary64 of one type under one control value
/// against the engine results of the default mode.
macro_rules! conversions_under {
    ($alias:ty, $control:expr, $values:expr) => {
        for &bits in $values {
            let x = <$alias>::from_bits(bits);
            let integer =
                i64::from_ne_bytes(u64::from(bits).to_ne_bytes()) >> (u64::from(bits) % 64);
            let env = <$alias>::ENV;
            let expected = (
                u64::from(x.round_to_integral_with(env).0.to_bits()),
                x.to_int_with::<i64>(env).0,
                x.to_int_with::<u32>(env).0,
                u64::from(<$alias>::from_int_with(integer, env).0.to_bits()),
                x.convert_with::<F16>(F16::ENV).0.to_bits(),
                x.convert_with::<F32>(F32::ENV).0.to_bits(),
                x.convert_with::<F64>(F64::ENV).0.to_bits(),
            );
            let ours = x86::with_mxcsr($control, || {
                let x = black_box(x);
                (
                    u64::from(x.round_to_integral().to_bits()),
                    x.to_int::<i64>(),
                    x.to_int::<u32>(),
                    u64::from(<$alias>::from_int(black_box(integer)).to_bits()),
                    x.convert::<F16>().to_bits(),
                    x.convert::<F32>().to_bits(),
                    x.convert::<F64>().to_bits(),
                )
            });
            assert_eq!(ours, expected, "{bits:#x} {integer} under {:#x}", $control);
        }
    };
}

#[test]
fn operators_read_mxcsr_before_the_host_unit() {
    // FTZ, DAZ, and each directed rounding change the host results, and an
    // unmasked exception traps in the host unit. Under each, the operators
    // still give the engine results of the default mode.
    let directions = [
        MXCSR_MASKED | MXCSR_FTZ,
        MXCSR_MASKED | MXCSR_DAZ,
        MXCSR_MASKED | MXCSR_TOWARD_NEGATIVE,
        MXCSR_MASKED | MXCSR_TOWARD_POSITIVE,
        MXCSR_MASKED | MXCSR_TOWARD_ZERO,
    ];
    let unmasked = MXCSR_EXCEPTION_MASKS.map(|mask| MXCSR_MASKED & !mask);
    let controls: Vec<u32> = directions.into_iter().chain(unmasked).collect();
    let mut random = SplitMix64::new(0x00C5_0000);
    let single = single_operands(&mut random, 4_000);
    let single_pairs = pairs(&single, SINGLE_SPECIALS.len(), 1 << 31, 1);
    let double = double_operands(&mut random, 4_000);
    let double_pairs = pairs(&double, 10, 1 << 63, 1);
    // binary16 takes the F16C path in a build with F16C.
    let mut halves: Vec<u16> = boundary_encodings_u128(16, 5, IntegerBit::Implicit)
        .into_iter()
        .map(|bits| u16::try_from(bits).expect("a binary16 encoding has 16 bits"))
        .collect();
    halves.extend(
        (0..4_000)
            .map(|_| u16::try_from(random.next_u64() >> 48).expect("the shift keeps 16 bits")),
    );
    // bfloat16 takes the paths that widen by a shift.
    let mut bfloats: Vec<u16> = boundary_encodings_u128(16, 8, IntegerBit::Implicit)
        .into_iter()
        .map(|bits| u16::try_from(bits).expect("a bfloat16 encoding has 16 bits"))
        .collect();
    bfloats.extend(
        (0..4_000)
            .map(|_| u16::try_from(random.next_u64() >> 48).expect("the shift keeps 16 bits")),
    );
    let mut half_pairs: Vec<(u16, u16)> = halves
        .iter()
        .zip(halves.iter().rev())
        .map(|(&a, &b)| (a, b))
        .collect();
    // A signaling NaN raises invalid when it widens, and a subnormal
    // remainder raises underflow when it rounds to binary16.
    half_pairs.extend([(0x7D00, 0x3C00), (0x0401, 0x0400)]);
    let bfloat_pairs: Vec<(u16, u16)> = bfloats
        .iter()
        .zip(bfloats.iter().rev())
        .map(|(&a, &b)| (a, b))
        .collect();
    for control in controls {
        operators_under!(F16, control, &half_pairs);
        operators_under!(BF16, control, &bfloat_pairs);
        remainders_under!(F16, control, &half_pairs);
        remainders_under!(BF16, control, &bfloat_pairs);
        operators_under!(F32, control, &single_pairs);
        operators_under!(F64, control, &double_pairs);
        conversions_under!(F16, control, &halves);
        conversions_under!(BF16, control, &bfloats);
        conversions_under!(F32, control, &single);
        conversions_under!(F64, control, &double);
    }
}

/// Checks the comparison and the minimum and maximum operations of one type
/// under one MXCSR value against the engine results of the default mode.
macro_rules! comparisons_under {
    ($alias:ty, $control:expr, $pairs:expr) => {
        for &(a, b) in $pairs {
            let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
            let env = <$alias>::ENV;
            let expected = (
                x.compare_quiet_with(y, env).0,
                [
                    x.minimum_with(y, env).0.to_bits(),
                    x.maximum_with(y, env).0.to_bits(),
                    x.minimum_number_with(y, env).0.to_bits(),
                    x.maximum_number_with(y, env).0.to_bits(),
                    x.min_num_with(y, env).0.to_bits(),
                    x.max_num_with(y, env).0.to_bits(),
                ],
            );
            let ours = x86::with_mxcsr($control, || {
                let (x, y) = (black_box(x), black_box(y));
                (
                    x.partial_cmp(&y),
                    [
                        x.minimum(y).to_bits(),
                        x.maximum(y).to_bits(),
                        x.minimum_number(y).to_bits(),
                        x.maximum_number(y).to_bits(),
                        x.min_num(y).to_bits(),
                        x.max_num(y).to_bits(),
                    ],
                )
            });
            assert_eq!(ours, expected, "{a:#x} {b:#x} under MXCSR {:#x}", $control);
        }
    };
}

#[test]
fn comparisons_read_mxcsr_before_the_host_unit() {
    // DAZ changes the order of a subnormal operand, and an unmasked invalid
    // or denormal exception traps. Under each control, the comparison and the
    // minimum and maximum operations give the engine results of the default
    // mode.
    let directions = [
        MXCSR_MASKED,
        MXCSR_MASKED | MXCSR_FTZ,
        MXCSR_MASKED | MXCSR_DAZ,
        MXCSR_MASKED | MXCSR_TOWARD_NEGATIVE,
    ];
    let unmasked = MXCSR_EXCEPTION_MASKS.map(|mask| MXCSR_MASKED & !mask);
    let controls: Vec<u32> = directions.into_iter().chain(unmasked).collect();
    let mut random = SplitMix64::new(0x00C5_C0AA);
    let single = single_operands(&mut random, 2_000);
    let single_pairs = pairs(&single, SINGLE_SPECIALS.len(), 1 << 31, 1);
    let double = double_operands(&mut random, 2_000);
    let double_pairs = pairs(&double, 10, 1 << 63, 1);
    let sixteen = |exponent_bits: u32, random: &mut SplitMix64| -> Vec<(u16, u16)> {
        let mut encodings: Vec<u16> =
            boundary_encodings_u128(16, exponent_bits, IntegerBit::Implicit)
                .into_iter()
                .map(|bits| u16::try_from(bits).expect("a 16-bit encoding"))
                .collect();
        let specials = encodings.len();
        encodings
            .extend((0..2_000).map(|_| u16::try_from(random.next_u64() >> 48).expect("16 bits")));
        pairs(&encodings, specials, 1 << 15, 1)
    };
    let half_pairs = sixteen(5, &mut random);
    let bfloat_pairs = sixteen(8, &mut random);
    for control in controls {
        comparisons_under!(F16, control, &half_pairs);
        comparisons_under!(BF16, control, &bfloat_pairs);
        comparisons_under!(F32, control, &single_pairs);
        comparisons_under!(F64, control, &double_pairs);
    }
}
