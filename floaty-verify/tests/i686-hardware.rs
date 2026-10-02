//! Checks the host paths of 32-bit x86 under every value of MXCSR and of the
//! x87 control word that changes a result of the host units or unmasks an
//! exception. The test runs only on 32-bit x86, under `qemu-i386`.
//!
//! The fields are those of the Intel SDM Volume 1, revision 253665-093US:
//! MXCSR in section 10.2.3, Figure 10-3, and the x87 control word in section
//! 8.1.5, Figure 8-6.

#![cfg(target_arch = "x86")]

use core::arch::asm;

use floaty::{BF16, F16, F32, F64, F80};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::entry_points::{
    assert_arithmetic_under, assert_comparisons_under, assert_conversions_under,
    assert_remainder_under,
};
use floaty_verify::random::SplitMix64;

/// The default MXCSR value: every exception masked, round to nearest, and
/// neither FTZ nor DAZ.
const MXCSR_DEFAULT: u32 = 0x1F80;

/// The MXCSR values that change a result of the SSE unit, or unmask an
/// exception: each rounding direction, FTZ, DAZ, and each exception mask
/// cleared.
const MXCSR_SETTINGS: [u32; 11] = [
    MXCSR_DEFAULT | 1 << 13,
    MXCSR_DEFAULT | 2 << 13,
    MXCSR_DEFAULT | 3 << 13,
    MXCSR_DEFAULT | 1 << 15,
    MXCSR_DEFAULT | 1 << 6,
    MXCSR_DEFAULT & !(1 << 7),
    MXCSR_DEFAULT & !(1 << 8),
    MXCSR_DEFAULT & !(1 << 9),
    MXCSR_DEFAULT & !(1 << 10),
    MXCSR_DEFAULT & !(1 << 11),
    MXCSR_DEFAULT & !(1 << 12),
];

/// The default x87 control word of Linux: every exception masked, round to
/// nearest, and the 64-bit precision.
const X87_DEFAULT: u16 = 0x037F;

/// The x87 control words that change a result of the x87 unit, or unmask an
/// exception: the 24-bit and 53-bit precisions, each rounding direction, and
/// each exception mask cleared.
const X87_SETTINGS: [u16; 11] = [
    X87_DEFAULT & !(3 << 8),
    X87_DEFAULT & !(1 << 8),
    X87_DEFAULT | 1 << 10,
    X87_DEFAULT | 2 << 10,
    X87_DEFAULT | 3 << 10,
    X87_DEFAULT & !1,
    X87_DEFAULT & !(1 << 1),
    X87_DEFAULT & !(1 << 2),
    X87_DEFAULT & !(1 << 3),
    X87_DEFAULT & !(1 << 4),
    X87_DEFAULT & !(1 << 5),
];

/// Runs `body` with MXCSR set to `control`, and restores MXCSR afterward.
/// `body` must not depend on its own host floating-point arithmetic.
fn with_mxcsr<T>(control: u32, body: impl FnOnce() -> T) -> T {
    let mut saved = 0_u32;
    // SAFETY: STMXCSR stores MXCSR through `saved`, and LDMXCSR loads
    // `control`. The code changes no other state.
    unsafe {
        asm!(
            "stmxcsr [{saved}]",
            "ldmxcsr [{control}]",
            saved = in(reg) &raw mut saved,
            control = in(reg) &raw const control,
            options(nostack),
        );
    }
    let result = body();
    // SAFETY: LDMXCSR restores the saved value.
    unsafe {
        asm!("ldmxcsr [{saved}]", saved = in(reg) &raw const saved, options(nostack));
    }
    result
}

/// Runs `body` with the x87 control word set to `control`, and restores the
/// control word afterward. `body` must not depend on its own host
/// floating-point arithmetic.
fn with_x87_control<T>(control: u16, body: impl FnOnce() -> T) -> T {
    let mut saved = 0_u16;
    // SAFETY: FNSTCW stores the control word through `saved`, and FLDCW loads
    // `control`. The code changes no other state.
    unsafe {
        asm!(
            "fnstcw word ptr [{saved}]",
            "fldcw word ptr [{control}]",
            saved = in(reg) &raw mut saved,
            control = in(reg) &raw const control,
            options(nostack),
        );
    }
    let result = body();
    // SAFETY: FLDCW restores the saved value.
    unsafe {
        asm!("fldcw word ptr [{saved}]", saved = in(reg) &raw const saved, options(nostack));
    }
    result
}

/// Returns boundary and random encodings of a binary format, in pairs.
fn pairs(random: &mut SplitMix64, layout: Layout) -> Vec<(u128, u128)> {
    let mut encodings = boundary_encodings_u128(layout);
    encodings.extend((0..1_000).map(|_| random.next_u128() >> (128 - layout.width)));
    encodings
        .iter()
        .zip(encodings.iter().rev())
        .map(|(&a, &b)| (a, b))
        .collect()
}

/// Checks the arithmetic, the comparison, the conversion, and the remainder
/// entry points of one type on each pair, each run by `$with` under the
/// control value `$control`, with the first operand as the addend, and an
/// integer made from the bits of the first operand.
macro_rules! scalars_under {
    ($alias:ty, $bits:ty, $with:ident, $control:expr, $pairs:expr) => {{
        let setting = format!("{} {:#x}", stringify!($with), $control);
        for &(a, b) in $pairs {
            let convert = |bits: u128| <$bits>::try_from(bits).expect("the encoding fits");
            let (x, y) = (
                <$alias>::from_bits(convert(a)),
                <$alias>::from_bits(convert(b)),
            );
            assert_arithmetic_under([x, y, x], &setting, |body| $with($control, body));
            assert_comparisons_under([x, y], &setting, |body| $with($control, body));
            assert_remainder_under([x, y], &setting, |body| $with($control, body));
            let bits = u64::try_from(a & u128::from(u64::MAX)).expect("the mask keeps 64 bits");
            let integer = i64::from_ne_bytes(bits.to_ne_bytes()) >> (bits % 64);
            assert_conversions_under(x, integer, &setting, |body| $with($control, body));
        }
    }};
}

#[test]
fn entry_points_read_mxcsr_before_the_sse_unit() {
    // Each setting changes a result of the SSE unit or unmasks an exception.
    // Under each, and under the default, the entry points give the engine
    // results of the default mode.
    let mut random = SplitMix64::new(0x0686_5E00);
    let half = pairs(&mut random, Layout::BINARY16);
    let bfloat = pairs(&mut random, Layout::BFLOAT16);
    let single = pairs(&mut random, Layout::BINARY32);
    let double = pairs(&mut random, Layout::BINARY64);
    for control in core::iter::once(MXCSR_DEFAULT).chain(MXCSR_SETTINGS) {
        scalars_under!(F16, u16, with_mxcsr, control, &half);
        scalars_under!(BF16, u16, with_mxcsr, control, &bfloat);
        scalars_under!(F32, u32, with_mxcsr, control, &single);
        scalars_under!(F64, u64, with_mxcsr, control, &double);
    }
}

#[test]
fn entry_points_read_the_control_word_before_the_x87_unit() {
    // Each setting changes a result of the x87 unit or unmasks an exception.
    // The x87 unit computes x87 extended precision and the remainder of every
    // format.
    let mut random = SplitMix64::new(0x0686_8700);
    let extended = pairs(&mut random, Layout::X87_EXTENDED);
    let single = pairs(&mut random, Layout::BINARY32);
    let double = pairs(&mut random, Layout::BINARY64);
    for control in core::iter::once(X87_DEFAULT).chain(X87_SETTINGS) {
        scalars_under!(F80, u128, with_x87_control, control, &extended);
        scalars_under!(F32, u32, with_x87_control, control, &single);
        scalars_under!(F64, u64, with_x87_control, control, &double);
    }
}
