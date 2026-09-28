//! Confirms each row of the x86 SSE and x87 presets, `Env::X86_SSE` and
//! `Env::X87`, on the host processor. The test runs only on x86-64 hosts.
//!
//! The other SSE and x87 hardware tests compare floaty with the processor
//! under behaviors that `floaty_verify::x86` builds from the presets. This
//! file checks the initial values of the control registers. It also checks
//! the rows that only some operands tell apart: tininess, the NaN rules, the
//! default NaN, DE, precision control, and C1. The citations are to the Intel
//! SDM Volume 1, order number 253665-093US.

#![cfg(target_arch = "x86_64")]

use std::num::NonZeroU32;

use floaty::env::Tininess;
use floaty::{Env, F32, F80, Rounding};
use floaty_verify::x86::{
    self, MXCSR_DAZ, MXCSR_DE, MXCSR_FTZ, MXCSR_MASKED, MXCSR_ROUNDINGS, X87_C1, X87_DE, X87_PE,
    X87_PRECISIONS, X87_ROUNDINGS, X87_STATUS_FLAGS, mxcsr_flags, x87_arithmetic_status,
    x87_status,
};

/// The rounding-control field of MXCSR, bits 13 and 14.
const MXCSR_ROUNDING_FIELD: u32 = 3 << 13;

/// The rounding-control field of the x87 control word, bits 10 and 11.
const X87_ROUNDING_FIELD: u16 = 3 << 10;

/// The precision-control field of the x87 control word, bits 8 and 9.
const X87_PRECISION_FIELD: u16 = 3 << 8;

const ONE_32: u32 = 0x3F80_0000;
const ONE_80: u128 = 0x3FFF_8000_0000_0000_0000;

/// Returns the rounding direction, FTZ, and DAZ that an MXCSR value selects.
fn sse_fields(mxcsr: u32) -> (Rounding, bool, bool) {
    let rounding = MXCSR_ROUNDINGS
        .iter()
        .find(|&&(_, field)| field == mxcsr & MXCSR_ROUNDING_FIELD)
        .map(|&(rounding, _)| rounding)
        .expect("every value of the rounding field selects a direction");
    (rounding, mxcsr & MXCSR_FTZ != 0, mxcsr & MXCSR_DAZ != 0)
}

/// Returns the rounding direction and the precision that an x87 control word
/// selects.
fn x87_fields(word: u16) -> (Rounding, Option<NonZeroU32>) {
    let rounding = X87_ROUNDINGS
        .iter()
        .find(|&&(_, field)| field == word & X87_ROUNDING_FIELD)
        .map(|&(rounding, _)| rounding)
        .expect("every value of the rounding field selects a direction");
    let precision = X87_PRECISIONS
        .iter()
        .find(|&&(_, field)| field == word & X87_PRECISION_FIELD)
        .map(|&(bits, _)| NonZeroU32::new(bits))
        .expect("the reset value of the precision field selects a precision");
    (rounding, precision)
}

/// Compares `ADDSS` with floaty under the SSE preset.
fn check_addss(a: u32, b: u32) {
    let (x, y) = (F32::from_bits(a), F32::from_bits(b));
    let (result, flags) = x.add_with(y, Env::X86_SSE);
    let nan_operand = x.is_nan() || y.is_nan();
    assert_eq!(
        x86::addss(a, b, MXCSR_MASKED),
        (result.to_bits(), mxcsr_flags(flags, false, nan_operand)),
        "ADDSS {a:08x} {b:08x}"
    );
}

/// Compares `FADDP` with floaty under the x87 preset, with the control word
/// that `FNINIT` sets.
fn check_fadd(a: u128, b: u128) {
    let (x, y) = (F80::from_bits(a), F80::from_bits(b));
    let (result, flags) = x.add_with(y, Env::X87);
    let (bits, status) = x86::fadd(a, b, x86::initialized_control_word());
    assert_eq!(
        (bits, status & X87_STATUS_FLAGS),
        (
            result.to_bits(),
            x87_arithmetic_status(flags, x.is_nan() || y.is_nan())
        ),
        "FADD {a:020x} {b:020x}"
    );
}

#[test]
fn the_sse_preset_has_the_reset_value_of_mxcsr() {
    // MXCSR is 1F80H after power-up or reset: Table 11-2, page 11-20. The
    // operating system starts a process with that value, and a new thread
    // inherits the value of its parent. The harness restores MXCSR after each
    // instruction, so the test thread still has it.
    let initial = x86::mxcsr();
    assert_eq!(initial, 0x1F80, "the MXCSR of the process");
    let preset = Env::X86_SSE;
    assert_eq!(
        sse_fields(initial),
        (
            preset.rounding,
            preset.flush_to_zero,
            preset.denormals_are_zero
        )
    );
    assert_eq!(
        preset.precision, None,
        "the SSE unit has no precision control"
    );
}

#[test]
fn the_x87_preset_has_the_control_word_after_fninit() {
    // FNINIT sets the control word to 037FH: section 8.1.5, page 8-7.
    let word = x86::initialized_control_word();
    assert_eq!(word, 0x037F, "the control word after FNINIT");
    let preset = Env::X87;
    assert_eq!(x87_fields(word), (preset.rounding, preset.precision));
    assert!(
        !preset.flush_to_zero && !preset.denormals_are_zero,
        "the x87 unit has no FTZ or DAZ"
    );
}

#[test]
fn both_units_detect_tininess_after_rounding() {
    // Products just below the smallest normal value: (1 - k 2^-p) times
    // (1 + j 2^(1-p)) times the smallest normal value. With k = 2j the exact
    // product rounds up to the smallest normal value at the full precision,
    // so only the tininess rule decides UE. Section 4.9.1.5, page 4-23.
    let mut decided = 0;
    for k in 1..=256_u32 {
        for j in 0..256_u32 {
            let (a, b) = (ONE_32 - k, 0x0080_0000 + j);
            let (x, y) = (F32::from_bits(a), F32::from_bits(b));
            let (result, flags) = x.mul_with(y, Env::X86_SSE);
            let (_, before) = x.mul_with(y, Env::X86_SSE.with_tininess(Tininess::BeforeRounding));
            if before == flags {
                continue;
            }
            decided += 1;
            assert_eq!(
                x86::mulss(a, b, MXCSR_MASKED),
                (result.to_bits(), mxcsr_flags(flags, false, false)),
                "MULSS {a:08x} {b:08x}"
            );
        }
    }
    assert!(decided > 0, "some binary32 products tell the rules apart");

    let control = x86::initialized_control_word();
    let mut decided = 0;
    for k in 1..=256_u128 {
        for j in 0..256_u128 {
            // 1 - k 2^-64, and the smallest normal x87 value plus j units.
            let (a, b) = (
                0x3FFF_0000_0000_0000_0000 - k,
                0x0001_8000_0000_0000_0000 + j,
            );
            let (x, y) = (F80::from_bits(a), F80::from_bits(b));
            let (result, flags) = x.mul_with(y, Env::X87);
            let (_, before) = x.mul_with(y, Env::X87.with_tininess(Tininess::BeforeRounding));
            if before == flags {
                continue;
            }
            decided += 1;
            let (bits, status) = x86::fmul(a, b, control);
            assert_eq!(
                (bits, status & X87_STATUS_FLAGS),
                (result.to_bits(), x87_status(flags)),
                "FMUL {a:020x} {b:020x}"
            );
        }
    }
    assert!(decided > 0, "some x87 products tell the rules apart");
}

#[test]
fn the_sse_unit_selects_the_first_nan_and_a_negative_default_nan() {
    // Table 4-8, page 4-17: the first source operand, made quiet.
    let (signaling, quiet) = (0x7FA0_0001, 0x7FC0_0002);
    let (negative_signaling, negative_quiet) = (0xFF80_0003, 0xFFC0_0004);
    for (a, b) in [
        (signaling, quiet),
        (quiet, signaling),
        (signaling, negative_signaling),
        (negative_quiet, quiet),
        (signaling, ONE_32),
        (ONE_32, negative_quiet),
    ] {
        check_addss(a, b);
    }
    // One operand: `SQRTSS` of a signaling NaN gives it made quiet.
    let (root, flags) = F32::from_bits(signaling).sqrt_with(Env::X86_SSE);
    assert_eq!(
        x86::sqrtss(signaling, 0, MXCSR_MASKED),
        (root.to_bits(), mxcsr_flags(flags, false, true))
    );
    // inf - inf gives the QNaN floating-point indefinite, 0xFFC00000:
    // sections 4.8.3.5 and 4.8.3.7, pages 4-17 and 4-18, and Table 4-3, page
    // 4-6.
    check_addss(0x7F80_0000, 0xFF80_0000);
    assert_eq!(
        x86::addss(0x7F80_0000, 0xFF80_0000, MXCSR_MASKED).0,
        0xFFC0_0000
    );
    // 0 * inf + qNaN gives the NaN addend and no IE: section 4.9.2, page 4-24.
    // `VFMADD213SS` with a, b, c computes b * a + c.
    assert!(
        is_x86_feature_detected!("fma"),
        "the host processor has FMA3"
    );
    let (zero, infinity) = (F32::from_bits(0), F32::from_bits(0x7F80_0000));
    let (sum, flags) = infinity.mul_add_with(zero, F32::from_bits(quiet), Env::X86_SSE);
    assert_eq!(
        x86::vfmadd213ss(0, 0x7F80_0000, quiet, MXCSR_MASKED),
        (sum.to_bits(), mxcsr_flags(flags, false, true))
    );
    assert_eq!(sum.to_bits(), quiet);
}

#[test]
fn the_x87_unit_selects_the_larger_nan_and_a_negative_default_nan() {
    // Table 4-8, page 4-17: a QNaN before an SNaN, and the larger significand
    // between two NaNs of one kind. Between equal significands the processor
    // gives the positive NaN.
    let signaling = 0x7FFF_A000_0000_0000_0007;
    let quiet = 0x7FFF_C000_0000_0000_0001;
    let larger_quiet = 0xFFFF_C000_0000_0000_0009;
    let negative_twin = 0xFFFF_C000_0000_0000_0001;
    for (a, b) in [
        (signaling, quiet),
        (quiet, signaling),
        (quiet, larger_quiet),
        (larger_quiet, quiet),
        (negative_twin, quiet),
        (quiet, negative_twin),
        (signaling, ONE_80),
    ] {
        check_fadd(a, b);
    }
    // inf - inf gives the QNaN floating-point indefinite: sections 4.8.3.5 and
    // 4.8.3.7, pages 4-17 and 4-18, and Table 4-3, page 4-6.
    let infinity = 0x7FFF_8000_0000_0000_0000;
    let negative_infinity = 0xFFFF_8000_0000_0000_0000;
    check_fadd(infinity, negative_infinity);
    let (bits, _) = x86::fadd(infinity, negative_infinity, x86::initialized_control_word());
    assert_eq!(bits, 0xFFFF_C000_0000_0000_0000);
}

#[test]
fn a_subnormal_operand_sets_de_unless_a_nan_comes_first() {
    // DE for a subnormal operand, section 4.9.1.2, page 4-21. A NaN operand
    // comes before it, section 4.9.2, page 4-24.
    check_addss(1, ONE_32);
    assert_eq!(x86::addss(1, ONE_32, MXCSR_MASKED).1 & MXCSR_DE, MXCSR_DE);
    check_addss(1, 0x7FC0_0000);
    check_fadd(1, ONE_80);
    let (_, status) = x86::fadd(1, ONE_80, x86::initialized_control_word());
    assert_eq!(status & X87_DE, X87_DE);
    check_fadd(1, 0x7FFF_C000_0000_0000_0000);
}

#[test]
fn the_x87_unit_rounds_at_64_bits_and_reports_c1() {
    // After FNINIT the precision control is 64 bits: section 8.1.5.2, page
    // 8-7. 1 + 2^-63 is exact at 64 bits and not at 53. 1 + 3 * 2^-65 rounds
    // up to 1 + 2^-63, which sets PE and C1.
    let exact = 0x3FC0_8000_0000_0000_0000;
    let rounds_up = 0x3FBF_C000_0000_0000_0000;
    check_fadd(ONE_80, exact);
    check_fadd(ONE_80, rounds_up);
    let (bits, status) = x86::fadd(ONE_80, rounds_up, x86::initialized_control_word());
    assert_eq!(bits, 0x3FFF_8000_0000_0000_0001);
    assert_eq!(status & X87_STATUS_FLAGS, X87_C1 | X87_PE, "C1 and PE");
}
