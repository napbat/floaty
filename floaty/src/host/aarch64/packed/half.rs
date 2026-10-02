//! The binary16 forms of `FEAT_FP16`, in functions that enable the feature
//! for their own code. A caller compiled without the feature, such as a
//! doctest, which does not take `RUSTFLAGS`, then still assembles the
//! instructions. Each function inlines into a caller with the feature.
//!
//! Each instruction computes eight binary16 lanes in their own precision, so
//! each lane rounds once, as the scalar binary16 path does. The path
//! requires FPCR.FZ16 to be zero, so no subnormal lane flushes to zero.

use core::arch::aarch64::uint16x8_t;
use core::mem::transmute;

use super::super::super::Operation;
use super::super::super::packed::Masks;
use crate::env::Rounding;
use crate::format::internal::MinMax;

/// Returns eight binary16 lanes as the value of a vector register.
#[inline]
fn halves(lanes: [u16; 8]) -> uint16x8_t {
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value of
    // each.
    unsafe { transmute::<[u16; 8], uint16x8_t>(lanes) }
}

/// Returns the eight binary16 lanes of a vector register value.
#[inline]
fn half_lanes(value: uint16x8_t) -> [u16; 8] {
    // SAFETY: as in `halves`.
    unsafe { transmute::<uint16x8_t, [u16; 8]>(value) }
}

/// # Safety
///
/// The processor must have `FEAT_FP16`.
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn binary_f16x8(left: [u16; 8], right: [u16; 8], operation: Operation) -> [u16; 8] {
    let (mut a, b) = (halves(left), halves(right));
    arithmetic!(operation, ".8h", a, b);
    half_lanes(a)
}

/// # Safety
///
/// The processor must have `FEAT_FP16`.
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn sqrt_f16x8(value: [u16; 8]) -> [u16; 8] {
    let mut a = halves(value);
    vector!("fsqrt {a:v}.8h, {b:v}.8h", a, a);
    half_lanes(a)
}

/// # Safety
///
/// The processor must have `FEAT_FP16`.
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn round_f16x8(value: [u16; 8], rounding: Rounding) -> Option<[u16; 8]> {
    let mut a = halves(value);
    round_vector!(rounding, ".8h", a);
    Some(half_lanes(a))
}

/// # Safety
///
/// The processor must have `FEAT_FP16`.
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn mul_add_f16x8(left: [u16; 8], right: [u16; 8], addend: [u16; 8]) -> [u16; 8] {
    let (mut a, b, c) = (halves(addend), halves(left), halves(right));
    fused!(".8h", a, b, c);
    half_lanes(a)
}

/// # Safety
///
/// The processor must have `FEAT_FP16`.
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn compare_f16x8(left: [u16; 8], right: [u16; 8]) -> Masks<u16, 8> {
    let (x, y) = (halves(left), halves(right));
    let (less, greater, ordered) = compare_vector!(".8h", uint16x8_t, x, y);
    Masks {
        less: half_lanes(less),
        greater: half_lanes(greater),
        unordered: half_lanes(ordered).map(|mask| !mask),
    }
}

/// # Safety
///
/// The processor must have `FEAT_FP16`.
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn min_max_f16x8(left: [u16; 8], right: [u16; 8], operation: MinMax) -> [u16; 8] {
    let (mut a, b) = (halves(left), halves(right));
    if operation.is_minimum() {
        vector!("fmin {a:v}.8h, {a:v}.8h, {b:v}.8h", a, b);
    } else {
        vector!("fmax {a:v}.8h, {a:v}.8h, {b:v}.8h", a, b);
    }
    half_lanes(a)
}
