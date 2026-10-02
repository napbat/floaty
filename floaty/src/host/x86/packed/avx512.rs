//! The 512-bit forms of AVX-512F, in functions that enable their feature
//! for their own code, as `wide` does for the 256-bit forms. Each form runs
//! the instruction of its 256-bit form on sixteen binary32 lanes, so each
//! lane gives the bits of that form. Each function inlines into a caller
//! with the feature: a build that enables it, or a copy of the module
//! `dispatch` that runs only on a processor that has it.

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m256i, __m512, __m512i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m256i, __m512, __m512i};
use core::mem::transmute;

use super::Operation;
use crate::env::Rounding;
use crate::format::internal::MinMax;

/// Returns sixteen binary32 lanes as the value of an AVX-512 register.
#[inline]
fn singles(lanes: [f32; 16]) -> __m512 {
    // SAFETY: both types hold 64 bytes, and every bit pattern is a value
    // of each.
    unsafe { transmute::<[f32; 16], __m512>(lanes) }
}

/// Returns the sixteen binary32 lanes of an AVX-512 register value.
#[inline]
fn single_lanes(value: __m512) -> [f32; 16] {
    // SAFETY: as in `singles`.
    unsafe { transmute::<__m512, [f32; 16]>(value) }
}

/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn binary_f32x16(left: [f32; 16], right: [f32; 16], operation: Operation) -> [f32; 16] {
    let (mut a, b) = (singles(left), singles(right));
    arithmetic!(
        zmm_reg,
        operation,
        [
            "vaddps {a}, {a}, {b}",
            "vsubps {a}, {a}, {b}",
            "vmulps {a}, {a}, {b}",
            "vdivps {a}, {a}, {b}",
        ],
        a,
        b
    );
    single_lanes(a)
}

/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn mul_add_f32x16(left: [f32; 16], right: [f32; 16], addend: [f32; 16]) -> [f32; 16] {
    let (mut a, b, c) = (singles(left), singles(right), singles(addend));
    fused!(zmm_reg, "vfmadd213ps", a, b, c);
    single_lanes(a)
}

/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn min_max_f32x16(left: [f32; 16], right: [f32; 16], operation: MinMax) -> [f32; 16] {
    let (mut a, b) = (singles(left), singles(right));
    if operation.is_minimum() {
        packed!(zmm_reg, "vminps {a}, {a}, {b}", a, b);
    } else {
        packed!(zmm_reg, "vmaxps {a}, {a}, {b}", a, b);
    }
    single_lanes(a)
}

/// Returns sixteen binary32 lanes rounded to integral values by
/// `VRNDSCALEPS`. Bits 1:0 of its immediate hold the rounding control, as
/// those of `VROUNDPS` do, and bits 7:4 hold a scale of zero, so the
/// instruction rounds to an integral value.
///
/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn round_f32x16(value: [f32; 16], rounding: Rounding) -> Option<[f32; 16]> {
    let mut a = singles(value);
    round_packed!(
        zmm_reg,
        rounding,
        [
            "vrndscaleps {a}, {b}, 0",
            "vrndscaleps {a}, {b}, 1",
            "vrndscaleps {a}, {b}, 2",
            "vrndscaleps {a}, {b}, 3",
        ],
        a
    );
    Some(single_lanes(a))
}

/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn to_int_f32x16(value: [f32; 16]) -> [i32; 16] {
    let a = singles(value);
    let result: __m512i;
    // SAFETY: VCVTPS2DQ reads and writes AVX-512 registers. The caller
    // guarantees AVX-512F, and the conversion changes only the status flags
    // of MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vcvtps2dq {result}, {a}",
            a = in(zmm_reg) a,
            result = lateout(zmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: both types hold 64 bytes, and every bit pattern is a value of
    // each.
    unsafe { transmute::<__m512i, [i32; 16]>(result) }
}

/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn from_int_x16(value: [i32; 16]) -> [f32; 16] {
    // SAFETY: both types hold 64 bytes, and every bit pattern is a value of
    // each.
    let a = unsafe { transmute::<[i32; 16], __m512i>(value) };
    let result: __m512;
    // SAFETY: as in `to_int_f32x16`, with `VCVTDQ2PS`.
    unsafe {
        core::arch::asm!(
            "vcvtdq2ps {result}, {a}",
            a = in(zmm_reg) a,
            result = lateout(zmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    single_lanes(result)
}

/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn widen_halves_x16(value: [u16; 16]) -> [f32; 16] {
    // SAFETY: both types hold 32 bytes, and every bit pattern is a value of
    // each.
    let a = unsafe { transmute::<[u16; 16], __m256i>(value) };
    let result: __m512;
    // SAFETY: VCVTPH2PS reads an AVX register and writes an AVX-512
    // register. The caller guarantees AVX-512F, and the widening is exact,
    // so it changes no state.
    unsafe {
        core::arch::asm!(
            "vcvtph2ps {result}, {a}",
            a = in(ymm_reg) a,
            result = lateout(zmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    single_lanes(result)
}

/// # Safety
///
/// The processor must have AVX-512F.
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn narrow_halves_x16(value: [f32; 16]) -> [u16; 16] {
    let a = singles(value);
    let result: __m256i;
    // SAFETY: VCVTPS2PH reads an AVX-512 register and writes an AVX
    // register. The caller guarantees AVX-512F, the rounding control 0 in
    // the immediate rounds to nearest even, and the rounding changes only
    // the status flags of MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vcvtps2ph {result}, {a}, 0",
            a = in(zmm_reg) a,
            result = lateout(ymm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: as in `widen_halves_x16`.
    unsafe { transmute::<__m256i, [u16; 16]>(result) }
}
