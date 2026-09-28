//! Compares conversions with the SSE unit of the host processor: rounding,
//! flush-to-zero (FTZ), denormals-are-zero (DAZ), and the MXCSR flags. The
//! test runs only on x86-64 hosts.

#![cfg(target_arch = "x86_64")]

use core::arch::asm;

use floaty::env::{NanPropagation, NanRule};
use floaty::{Env, F32, F64, Flags, Rounding};
use floaty_verify::random::SplitMix64;

/// The MXCSR value with every exception masked and no flag set.
const MASKED: u32 = 0x1F80;
const FLAG_BITS: u32 = 0x3F;
const DAZ: u32 = 1 << 6;
const FTZ: u32 = 1 << 15;

/// The rounding directions and their MXCSR rounding-control field.
const ROUNDINGS: [(Rounding, u32); 4] = [
    (Rounding::NearestEven, 0),
    (Rounding::TowardNegative, 1 << 13),
    (Rounding::TowardPositive, 2 << 13),
    (Rounding::TowardZero, 3 << 13),
];

/// Runs `CVTSD2SS` under `control` and returns the result and the MXCSR
/// flags.
fn cvtsd2ss(input: u64, control: u32) -> (u32, u32) {
    let (mut saved, mut after) = (0_u32, 0_u32);
    let result: u32;
    // SAFETY: the code saves MXCSR, runs one conversion under `control`,
    // reads the flags, and restores MXCSR before it ends. It reads and writes
    // only the three local variables.
    unsafe {
        asm!(
            "stmxcsr [{saved}]",
            "ldmxcsr [{control}]",
            "movq {value}, {input}",
            "cvtsd2ss {value}, {value}",
            "movd {result:e}, {value}",
            "stmxcsr [{after}]",
            "ldmxcsr [{saved}]",
            saved = in(reg) &raw mut saved,
            control = in(reg) &raw const control,
            after = in(reg) &raw mut after,
            input = in(reg) input,
            value = out(xmm_reg) _,
            result = out(reg) result,
            options(nostack),
        );
    }
    (result, after & FLAG_BITS)
}

/// Runs `CVTSS2SD` under `control` and returns the result and the MXCSR
/// flags.
fn cvtss2sd(input: u32, control: u32) -> (u64, u32) {
    let (mut saved, mut after) = (0_u32, 0_u32);
    let result: u64;
    // SAFETY: as in `cvtsd2ss`.
    unsafe {
        asm!(
            "stmxcsr [{saved}]",
            "ldmxcsr [{control}]",
            "movd {value}, {input:e}",
            "cvtss2sd {value}, {value}",
            "movq {result}, {value}",
            "stmxcsr [{after}]",
            "ldmxcsr [{saved}]",
            saved = in(reg) &raw mut saved,
            control = in(reg) &raw const control,
            after = in(reg) &raw mut after,
            input = in(reg) input,
            value = out(xmm_reg) _,
            result = out(reg) result,
            options(nostack),
        );
    }
    (result, after & FLAG_BITS)
}

/// Returns the MXCSR flags that a floaty result reports. MXCSR sets DE only
/// when DAZ is clear.
fn mxcsr_flags(flags: Flags, daz: bool) -> u32 {
    let mut bits = 0;
    for (flag, bit) in [
        (Flags::INVALID, 0),
        (Flags::OVERFLOW, 3),
        (Flags::UNDERFLOW, 4),
        (Flags::INEXACT, 5),
    ] {
        if flags.contains(flag) {
            bits |= 1 << bit;
        }
    }
    if flags.contains(Flags::DENORMAL_INPUT) && !daz {
        bits |= 1 << 1;
    }
    bits
}

/// The environment that matches an MXCSR setting. Conversions never make a
/// default NaN, so the NaN rule only has to keep payloads, as SSE does.
fn env(rounding: Rounding, ftz: bool, daz: bool) -> Env {
    Env::IEEE
        .with_rounding(rounding)
        .with_flush_to_zero(ftz)
        .with_denormals_are_zero(daz)
        .with_nan(NanRule {
            propagation: NanPropagation::FirstOperand,
            default_negative: true,
        })
}

/// Returns the low-bit patterns of `dropped` discarded bits that decide a
/// rounding: exact, just below halfway, halfway, just above halfway, and all
/// ones.
fn edge_patterns(dropped: u32) -> [u64; 5] {
    let half = 1_u64 << (dropped - 1);
    [0, half - 1, half, half + 1, (half << 1) - 1]
}

/// Returns binary64 encodings near every binary32 boundary, and random ones.
fn binary64_inputs() -> Vec<u64> {
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
    let edges = (-152..=-122).chain(125..=129);
    for exponent in edges {
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
    let inputs = binary64_inputs();
    for (rounding, rc) in ROUNDINGS {
        for (ftz, daz) in [(false, false), (true, false), (false, true), (true, true)] {
            let control = MASKED | rc | if ftz { FTZ } else { 0 } | if daz { DAZ } else { 0 };
            let env = env(rounding, ftz, daz);
            for &input in &inputs {
                let (expected, expected_flags) = cvtsd2ss(input, control);
                let (ours, flags): (F32, _) = F64::from_bits(input).convert_with(env);
                let context = format!("{input:#018x} {rounding:?} ftz {ftz} daz {daz} {flags:?}");
                assert_eq!(ours.to_bits(), expected, "{context}: result");
                assert_eq!(mxcsr_flags(flags, daz), expected_flags, "{context}: flags");
            }
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
    for daz in [false, true] {
        let control = MASKED | if daz { DAZ } else { 0 };
        let env = env(Rounding::NearestEven, false, daz);
        for &input in &inputs {
            let (expected, expected_flags) = cvtss2sd(input, control);
            let (ours, flags): (F64, _) = F32::from_bits(input).convert_with(env);
            let context = format!("{input:#010x} daz {daz} {flags:?}");
            assert_eq!(ours.to_bits(), expected, "{context}: result");
            assert_eq!(mxcsr_flags(flags, daz), expected_flags, "{context}: flags");
        }
    }
}
