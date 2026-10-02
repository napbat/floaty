//! The 256-bit forms, in functions that enable their features for their own
//! code. A caller compiled without AVX, such as a doctest, which does not
//! take `RUSTFLAGS`, then still compiles the 256-bit register class. Each
//! function inlines into a caller with the features.

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128, __m128i, __m256, __m256d, __m256i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128, __m128i, __m256, __m256d, __m256i};
use core::mem::transmute;

use super::super::super::packed::Masks;
use super::Operation;
use crate::env::Rounding;
use crate::format::internal::MinMax;

/// Returns eight binary32 lanes as the value of an AVX register.
#[inline]
fn singles(lanes: [f32; 8]) -> __m256 {
    // SAFETY: both types hold 32 bytes, and every bit pattern is a value
    // of each.
    unsafe { transmute::<[f32; 8], __m256>(lanes) }
}

/// Returns the eight binary32 lanes of an AVX register value.
#[inline]
fn single_lanes(value: __m256) -> [f32; 8] {
    // SAFETY: as in `singles`.
    unsafe { transmute::<__m256, [f32; 8]>(value) }
}

/// Returns four binary64 lanes as the value of an AVX register.
#[inline]
fn doubles(lanes: [f64; 4]) -> __m256d {
    // SAFETY: as in `singles`.
    unsafe { transmute::<[f64; 4], __m256d>(lanes) }
}

/// Returns the four binary64 lanes of an AVX register value.
#[inline]
fn double_lanes(value: __m256d) -> [f64; 4] {
    // SAFETY: as in `singles`.
    unsafe { transmute::<__m256d, [f64; 4]>(value) }
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn binary_f32x8(left: [f32; 8], right: [f32; 8], operation: Operation) -> [f32; 8] {
    let (mut a, b) = (singles(left), singles(right));
    arithmetic!(
        ymm_reg,
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
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn binary_f64x4(left: [f64; 4], right: [f64; 4], operation: Operation) -> [f64; 4] {
    let (mut a, b) = (doubles(left), doubles(right));
    arithmetic!(
        ymm_reg,
        operation,
        [
            "vaddpd {a}, {a}, {b}",
            "vsubpd {a}, {a}, {b}",
            "vmulpd {a}, {a}, {b}",
            "vdivpd {a}, {a}, {b}",
        ],
        a,
        b
    );
    double_lanes(a)
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn sqrt_f32x8(value: [f32; 8]) -> [f32; 8] {
    let mut a = singles(value);
    packed!(ymm_reg, "vsqrtps {a}, {b}", a, a);
    single_lanes(a)
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn sqrt_f64x4(value: [f64; 4]) -> [f64; 4] {
    let mut a = doubles(value);
    packed!(ymm_reg, "vsqrtpd {a}, {b}", a, a);
    double_lanes(a)
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn round_f32x8(value: [f32; 8], rounding: Rounding) -> Option<[f32; 8]> {
    let mut a = singles(value);
    round_packed!(
        ymm_reg,
        rounding,
        [
            "vroundps {a}, {b}, 0",
            "vroundps {a}, {b}, 1",
            "vroundps {a}, {b}, 2",
            "vroundps {a}, {b}, 3",
        ],
        a
    );
    Some(single_lanes(a))
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn round_f64x4(value: [f64; 4], rounding: Rounding) -> Option<[f64; 4]> {
    let mut a = doubles(value);
    round_packed!(
        ymm_reg,
        rounding,
        [
            "vroundpd {a}, {b}, 0",
            "vroundpd {a}, {b}, 1",
            "vroundpd {a}, {b}, 2",
            "vroundpd {a}, {b}, 3",
        ],
        a
    );
    Some(double_lanes(a))
}

/// # Safety
///
/// The processor must have AVX and FMA.
#[cfg(target_feature = "fma")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub unsafe fn mul_add_f32x8(left: [f32; 8], right: [f32; 8], addend: [f32; 8]) -> [f32; 8] {
    let (mut a, b, c) = (singles(left), singles(right), singles(addend));
    fused!(ymm_reg, "vfmadd213ps", a, b, c);
    single_lanes(a)
}

/// # Safety
///
/// The processor must have AVX and FMA.
#[cfg(target_feature = "fma")]
#[target_feature(enable = "avx,fma")]
#[inline]
pub unsafe fn mul_add_f64x4(left: [f64; 4], right: [f64; 4], addend: [f64; 4]) -> [f64; 4] {
    let (mut a, b, c) = (doubles(left), doubles(right), doubles(addend));
    fused!(ymm_reg, "vfmadd213pd", a, b, c);
    double_lanes(a)
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn widen_x4(value: [f32; 4]) -> [f64; 4] {
    let a = super::singles(value);
    let result: __m256d;
    // SAFETY: VCVTPS2PD reads an SSE register and writes an AVX register.
    // The caller guarantees AVX, and the conversion changes only the
    // status flags of MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vcvtps2pd {result}, {a}",
            a = in(xmm_reg) a,
            result = lateout(ymm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    double_lanes(result)
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn narrow_x4(value: [f64; 4]) -> [f32; 4] {
    let a = doubles(value);
    let result: __m128;
    // SAFETY: as in `widen_x4`, with `VCVTPD2PS`.
    unsafe {
        core::arch::asm!(
            "vcvtpd2ps {result}, {a}",
            a = in(ymm_reg) a,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    super::single_lanes(result)
}

/// # Safety
///
/// The processor must have AVX and F16C.
#[cfg(target_feature = "f16c")]
#[target_feature(enable = "avx,f16c")]
#[inline]
pub unsafe fn widen_halves_x8(value: [u16; 8]) -> [f32; 8] {
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value
    // of each.
    let a = unsafe { transmute::<[u16; 8], __m128i>(value) };
    let result: __m256;
    // SAFETY: VCVTPH2PS reads an SSE register and writes an AVX register.
    // The caller guarantees AVX and F16C, and the widening is exact, so it
    // changes no state.
    unsafe {
        core::arch::asm!(
            "vcvtph2ps {result}, {a}",
            a = in(xmm_reg) a,
            result = lateout(ymm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    single_lanes(result)
}

/// # Safety
///
/// The processor must have AVX and F16C.
#[cfg(target_feature = "f16c")]
#[target_feature(enable = "avx,f16c")]
#[inline]
pub unsafe fn narrow_halves_x8(value: [f32; 8]) -> [u16; 8] {
    let a = singles(value);
    let result: __m128i;
    // SAFETY: VCVTPS2PH reads an AVX register and writes an SSE register.
    // The caller guarantees AVX and F16C, and the rounding changes only the
    // status flags of MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vcvtps2ph {result}, {a}, 0",
            a = in(ymm_reg) a,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: as in `widen_halves_x8`.
    unsafe { transmute::<__m128i, [u16; 8]>(result) }
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn compare_f32x8(left: [f32; 8], right: [f32; 8]) -> Masks<u32, 8> {
    let (x, y) = (singles(left), singles(right));
    let (mut less, mut greater, mut unordered) = (x, y, x);
    packed!(ymm_reg, "vcmpltps {a}, {a}, {b}", less, y);
    packed!(ymm_reg, "vcmpltps {a}, {a}, {b}", greater, x);
    packed!(ymm_reg, "vcmpunordps {a}, {a}, {b}", unordered, y);
    let bits = |value: __m256| single_lanes(value).map(f32::to_bits);
    Masks {
        less: bits(less),
        greater: bits(greater),
        unordered: bits(unordered),
    }
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn compare_f64x4(left: [f64; 4], right: [f64; 4]) -> Masks<u64, 4> {
    let (x, y) = (doubles(left), doubles(right));
    let (mut less, mut greater, mut unordered) = (x, y, x);
    packed!(ymm_reg, "vcmpltpd {a}, {a}, {b}", less, y);
    packed!(ymm_reg, "vcmpltpd {a}, {a}, {b}", greater, x);
    packed!(ymm_reg, "vcmpunordpd {a}, {a}, {b}", unordered, y);
    let bits = |value: __m256d| double_lanes(value).map(f64::to_bits);
    Masks {
        less: bits(less),
        greater: bits(greater),
        unordered: bits(unordered),
    }
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn min_max_f32x8(left: [f32; 8], right: [f32; 8], operation: MinMax) -> [f32; 8] {
    let (mut a, b) = (singles(left), singles(right));
    if operation.is_minimum() {
        packed!(ymm_reg, "vminps {a}, {a}, {b}", a, b);
    } else {
        packed!(ymm_reg, "vmaxps {a}, {a}, {b}", a, b);
    }
    single_lanes(a)
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn min_max_f64x4(left: [f64; 4], right: [f64; 4], operation: MinMax) -> [f64; 4] {
    let (mut a, b) = (doubles(left), doubles(right));
    if operation.is_minimum() {
        packed!(ymm_reg, "vminpd {a}, {a}, {b}", a, b);
    } else {
        packed!(ymm_reg, "vmaxpd {a}, {a}, {b}", a, b);
    }
    double_lanes(a)
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn to_int_f32x8(value: [f32; 8]) -> [i32; 8] {
    let a = singles(value);
    let result: __m256i;
    // SAFETY: VCVTPS2DQ reads and writes AVX registers. The caller
    // guarantees AVX, and the conversion changes only the status flags of
    // MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vcvtps2dq {result}, {a}",
            a = in(ymm_reg) a,
            result = lateout(ymm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: both types hold 32 bytes, and every bit pattern is a value of
    // each.
    unsafe { transmute::<__m256i, [i32; 8]>(result) }
}

/// # Safety
///
/// The processor must have AVX.
#[target_feature(enable = "avx")]
#[inline]
pub unsafe fn to_int_f64x4(value: [f64; 4]) -> [i32; 4] {
    let a = doubles(value);
    let result: __m128i;
    // SAFETY: as in `to_int_f32x8`, with `VCVTPD2DQ`, which writes an SSE
    // register.
    unsafe {
        core::arch::asm!(
            "vcvtpd2dq {result}, {a}",
            a = in(ymm_reg) a,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value of
    // each.
    unsafe { transmute::<__m128i, [i32; 4]>(result) }
}
