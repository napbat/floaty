//! The vector instructions of the AArch64 floating-point unit, for the paths
//! of `Lanes`: 128-bit registers of four binary32 or two binary64 lanes.
//!
//! Each function computes one chunk of lanes in registers. The block takes
//! and gives register values, so it touches no memory: the compiler loads and
//! stores the lanes, and keeps them in registers between operations. The base
//! architecture has no wider registers, so each function for a 256-bit chunk
//! returns `None`.

use core::arch::aarch64::{float32x2_t, float32x4_t, float64x2_t};
use core::mem::transmute;

use super::super::Operation;

/// `false`: AArch64 has no vector registers wider than 128 bits in the base
/// architecture.
pub const WIDE: bool = false;

/// Returns four binary32 lanes as the value of a vector register.
#[inline]
fn singles(lanes: [f32; 4]) -> float32x4_t {
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value of
    // each.
    unsafe { transmute::<[f32; 4], float32x4_t>(lanes) }
}

/// Returns the four binary32 lanes of a vector register value.
#[inline]
fn single_lanes(value: float32x4_t) -> [f32; 4] {
    // SAFETY: as in `singles`.
    unsafe { transmute::<float32x4_t, [f32; 4]>(value) }
}

/// Returns two binary64 lanes as the value of a vector register.
#[inline]
fn doubles(lanes: [f64; 2]) -> float64x2_t {
    // SAFETY: as in `singles`.
    unsafe { transmute::<[f64; 2], float64x2_t>(lanes) }
}

/// Returns the two binary64 lanes of a vector register value.
#[inline]
fn double_lanes(value: float64x2_t) -> [f64; 2] {
    // SAFETY: as in `singles`.
    unsafe { transmute::<float64x2_t, [f64; 2]>(value) }
}

/// Runs one vector instruction on the register values `$a` and `$b`, and
/// leaves the result in `$a`.
macro_rules! vector {
    ($instruction:expr, $a:ident, $b:ident) => {
        // SAFETY: the instruction reads and writes SIMD and floating-point
        // registers. Every AArch64 target has it, and it changes only the
        // status flags of FPSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                a = inout(vreg) $a,
                b = in(vreg) $b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Runs one instruction of `$operation` on `$a` and `$b` with the
/// arrangement `$arrangement`, and leaves the result in `$a`.
macro_rules! arithmetic {
    ($operation:expr, $arrangement:literal, $a:ident, $b:ident) => {
        match $operation {
            Operation::Add => vector!(
                concat!(
                    "fadd {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $a,
                $b
            ),
            Operation::Sub => vector!(
                concat!(
                    "fsub {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $a,
                $b
            ),
            Operation::Mul => vector!(
                concat!(
                    "fmul {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $a,
                $b
            ),
            Operation::Div => vector!(
                concat!(
                    "fdiv {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $a,
                $b
            ),
        }
    };
}

/// Returns `operation` of four pairs of binary32 lanes, by `FADD`, `FSUB`,
/// `FMUL`, or `FDIV` on `.4S`.
#[inline]
pub fn binary_f32x4(left: [f32; 4], right: [f32; 4], operation: Operation) -> [f32; 4] {
    let (mut a, b) = (singles(left), singles(right));
    arithmetic!(operation, ".4s", a, b);
    single_lanes(a)
}

/// Returns `operation` of two pairs of binary64 lanes, by `FADD`, `FSUB`,
/// `FMUL`, or `FDIV` on `.2D`.
#[inline]
pub fn binary_f64x2(left: [f64; 2], right: [f64; 2], operation: Operation) -> [f64; 2] {
    let (mut a, b) = (doubles(left), doubles(right));
    arithmetic!(operation, ".2d", a, b);
    double_lanes(a)
}

/// Returns the square roots of four binary32 lanes, by `FSQRT`.
#[inline]
pub fn sqrt_f32x4(value: [f32; 4]) -> [f32; 4] {
    let mut a = singles(value);
    vector!("fsqrt {a:v}.4s, {b:v}.4s", a, a);
    single_lanes(a)
}

/// Returns the square roots of two binary64 lanes, by `FSQRT`.
#[inline]
pub fn sqrt_f64x2(value: [f64; 2]) -> [f64; 2] {
    let mut a = doubles(value);
    vector!("fsqrt {a:v}.2d, {b:v}.2d", a, a);
    double_lanes(a)
}

/// Returns four binary32 lanes rounded to integral values to nearest even,
/// by `FRINTN`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack SSE4.1.
pub fn round_f32x4(value: [f32; 4]) -> Option<[f32; 4]> {
    let mut a = singles(value);
    vector!("frintn {a:v}.4s, {b:v}.4s", a, a);
    Some(single_lanes(a))
}

/// Returns two binary64 lanes rounded to integral values to nearest even, by
/// `FRINTN`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack SSE4.1.
pub fn round_f64x2(value: [f64; 2]) -> Option<[f64; 2]> {
    let mut a = doubles(value);
    vector!("frintn {a:v}.2d, {b:v}.2d", a, a);
    Some(double_lanes(a))
}

/// Runs a vector fused multiply-add: `FMLA` adds the product of `$b` and
/// `$c` to `$a`, rounded once.
macro_rules! fused {
    ($arrangement:literal, $a:ident, $b:ident, $c:ident) => {
        // SAFETY: as in `vector!`, with a third register.
        unsafe {
            core::arch::asm!(
                concat!("fmla {a:v}", $arrangement, ", {b:v}", $arrangement, ", {c:v}", $arrangement),
                a = inout(vreg) $a,
                b = in(vreg) $b,
                c = in(vreg) $c,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Returns `left * right + addend` of four triples of binary32 lanes, each
/// rounded once, by `FMLA`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack FMA.
pub fn mul_add_f32x4(left: [f32; 4], right: [f32; 4], addend: [f32; 4]) -> Option<[f32; 4]> {
    let (mut a, b, c) = (singles(addend), singles(left), singles(right));
    fused!(".4s", a, b, c);
    Some(single_lanes(a))
}

/// Returns `left * right + addend` of two triples of binary64 lanes, each
/// rounded once, by `FMLA`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack FMA.
pub fn mul_add_f64x2(left: [f64; 2], right: [f64; 2], addend: [f64; 2]) -> Option<[f64; 2]> {
    let (mut a, b, c) = (doubles(addend), doubles(left), doubles(right));
    fused!(".2d", a, b, c);
    Some(double_lanes(a))
}

/// Returns two binary32 lanes widened exactly to binary64, by `FCVTL`.
#[inline]
pub fn widen_x2(value: [f32; 2]) -> [f64; 2] {
    // SAFETY: both types hold 8 bytes, and every bit pattern is a value of
    // each.
    let a = unsafe { transmute::<[f32; 2], float32x2_t>(value) };
    let result: float64x2_t;
    // SAFETY: FCVTL reads and writes SIMD and floating-point registers. Every
    // AArch64 target has it, and the widening changes only the status flags
    // of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvtl {result:v}.2d, {a:v}.2s",
            a = in(vreg) a,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    double_lanes(result)
}

/// Returns two binary64 lanes rounded to binary32, by `FCVTN`.
#[inline]
pub fn narrow_x2(value: [f64; 2]) -> [f32; 2] {
    let a = doubles(value);
    let result: float32x2_t;
    // SAFETY: as in `widen_x2`, with `FCVTN`.
    unsafe {
        core::arch::asm!(
            "fcvtn {result:v}.2s, {a:v}.2d",
            a = in(vreg) a,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: both types hold 8 bytes, and every bit pattern is a value of
    // each.
    unsafe { transmute::<float32x2_t, [f32; 2]>(result) }
}

/// Defines a function for a 256-bit chunk, which returns `None`.
macro_rules! no_wide {
    ($name:ident, ($($type:ty),+) -> $result:ty) => {
        #[doc = "Returns `None`: the base architecture has no 256-bit registers."]
        #[inline]
        pub fn $name($(_: $type),+) -> Option<$result> {
            None
        }
    };
}

no_wide!(binary_f32x8, ([f32; 8], [f32; 8], Operation) -> [f32; 8]);
no_wide!(binary_f64x4, ([f64; 4], [f64; 4], Operation) -> [f64; 4]);
no_wide!(sqrt_f32x8, ([f32; 8]) -> [f32; 8]);
no_wide!(sqrt_f64x4, ([f64; 4]) -> [f64; 4]);
no_wide!(round_f32x8, ([f32; 8]) -> [f32; 8]);
no_wide!(round_f64x4, ([f64; 4]) -> [f64; 4]);
no_wide!(mul_add_f32x8, ([f32; 8], [f32; 8], [f32; 8]) -> [f32; 8]);
no_wide!(mul_add_f64x4, ([f64; 4], [f64; 4], [f64; 4]) -> [f64; 4]);
no_wide!(widen_x4, ([f32; 4]) -> [f64; 4]);
no_wide!(narrow_x4, ([f64; 4]) -> [f32; 4]);
