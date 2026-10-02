//! The vector instructions of the AArch64 floating-point unit, for the paths
//! of `Lanes`: 128-bit registers of four binary32 or two binary64 lanes, and
//! of eight binary16 lanes with `FEAT_FP16`.
//!
//! Each function computes one chunk of lanes in registers. The block takes
//! and gives register values, so it touches no memory: the compiler loads and
//! stores the lanes, and keeps them in registers between operations. The base
//! architecture has no wider registers, so each function for a 256-bit chunk
//! returns `None`. Each function for binary16 lanes returns `None` in a build
//! without `FEAT_FP16`.

use core::arch::aarch64::{
    float32x2_t, float32x4_t, float64x2_t, uint16x4_t, uint32x4_t, uint64x2_t, vcombine_f32,
    vget_high_f32, vget_low_f32, vuzp1q_f32, vuzp2q_f32,
};
use core::mem::transmute;

use super::super::Operation;
use super::super::packed::Masks;
use crate::env::Rounding;
use crate::format::internal::MinMax;

/// `false`: AArch64 has no vector registers wider than 128 bits in the base
/// architecture.
pub const WIDE: bool = false;

/// `false`: no form uses 512-bit registers on AArch64.
pub const EXTRA_WIDE: bool = false;

/// `true`: every AArch64 target widens binary16 lanes to binary32 and
/// rounds binary32 lanes to binary16.
pub const HALF: bool = super::HALF;

/// `false`: no packed path converts lanes to integers on AArch64.
pub const INTEGERS: bool = false;

/// `true` in a build with `FEAT_FP16`: the vector instructions compute eight
/// binary16 lanes in their own precision.
pub const NATIVE_HALF: bool = cfg!(target_feature = "fp16");

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
        // registers. Every AArch64 target has it on binary32 and binary64
        // lanes, and a function on binary16 lanes runs only with
        // `FEAT_FP16`. The instruction changes only the status flags of FPSR,
        // which floaty does not read.
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

/// Returns lanes 0 and 1 of `first` and then of `second`, and lanes 2 and 3
/// of `first` and then of `second`.
///
/// A move of lanes is no floating-point operation, so the intrinsics serve.
/// Each takes and gives whole registers. LLVM read lanes that Rust moved one
/// at a time from the stack, in loads that spanned two stores, and each load
/// waited for its stores to complete.
#[inline]
pub fn halves_f32x4(first: [f32; 4], second: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    let (a, b) = (singles(first), singles(second));
    // SAFETY: the instructions need NEON, which every AArch64 target has.
    let (low, high) = unsafe {
        (
            vcombine_f32(vget_low_f32(a), vget_low_f32(b)),
            vcombine_f32(vget_high_f32(a), vget_high_f32(b)),
        )
    };
    (single_lanes(low), single_lanes(high))
}

/// Returns lanes 0 and 2 of `first` and then of `second`, and lanes 1 and 3
/// of `first` and then of `second`, by `UZP1` and `UZP2`, as
/// `halves_f32x4` states.
#[inline]
pub fn evens_odds_f32x4(first: [f32; 4], second: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    let (a, b) = (singles(first), singles(second));
    // SAFETY: as in `halves_f32x4`.
    let (evens, odds) = unsafe { (vuzp1q_f32(a, b), vuzp2q_f32(a, b)) };
    (single_lanes(evens), single_lanes(odds))
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

/// Runs one packed rounding to an integral value on `$a`, from the
/// instruction of the direction `$rounding` on the arrangement
/// `$arrangement`: `FRINTN`, `FRINTA`, `FRINTM`, `FRINTP`, or `FRINTZ`.
/// Returns `None` from the function for another direction.
macro_rules! round_vector {
    ($rounding:expr, $arrangement:literal, $a:ident) => {
        match $rounding {
            Rounding::TiesToEven => {
                vector!(
                    concat!("frintn {a:v}", $arrangement, ", {b:v}", $arrangement),
                    $a,
                    $a
                )
            }
            Rounding::TiesToAway => {
                vector!(
                    concat!("frinta {a:v}", $arrangement, ", {b:v}", $arrangement),
                    $a,
                    $a
                )
            }
            Rounding::TowardNegative => {
                vector!(
                    concat!("frintm {a:v}", $arrangement, ", {b:v}", $arrangement),
                    $a,
                    $a
                )
            }
            Rounding::TowardPositive => {
                vector!(
                    concat!("frintp {a:v}", $arrangement, ", {b:v}", $arrangement),
                    $a,
                    $a
                )
            }
            Rounding::TowardZero => {
                vector!(
                    concat!("frintz {a:v}", $arrangement, ", {b:v}", $arrangement),
                    $a,
                    $a
                )
            }
            Rounding::TiesTowardZero | Rounding::AwayFromZero | Rounding::ToOdd => return None,
        }
    };
}

/// Returns four binary32 lanes rounded to integral values in the direction
/// `rounding`, or `None` for a direction that no instruction has.
#[inline]
pub fn round_f32x4(value: [f32; 4], rounding: Rounding) -> Option<[f32; 4]> {
    let mut a = singles(value);
    round_vector!(rounding, ".4s", a);
    Some(single_lanes(a))
}

/// Returns two binary64 lanes rounded to integral values in the direction
/// `rounding`, or `None` for a direction that no instruction has.
#[inline]
pub fn round_f64x2(value: [f64; 2], rounding: Rounding) -> Option<[f64; 2]> {
    let mut a = doubles(value);
    round_vector!(rounding, ".2d", a);
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

/// Returns four 32-bit integers converted to binary32 in the rounding
/// direction of FPCR, by `SCVTF`. The conversion is exact for an integer
/// below `2^24` in magnitude.
#[inline]
pub fn from_int_x4(value: [i32; 4]) -> [f32; 4] {
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value of
    // each.
    let a = unsafe { transmute::<[i32; 4], uint32x4_t>(value) };
    let result: float32x4_t;
    // SAFETY: SCVTF reads and writes SIMD and floating-point registers.
    // Every AArch64 target has it, and the conversion changes only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "scvtf {result:v}.4s, {a:v}.4s",
            a = in(vreg) a,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    single_lanes(result)
}

/// Returns four binary16 lanes widened exactly to binary32, by `FCVTL`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack F16C.
pub fn widen_halves_x4(value: [u16; 4]) -> Option<[f32; 4]> {
    // SAFETY: both types hold 8 bytes, and every bit pattern is a value of
    // each.
    let a = unsafe { transmute::<[u16; 4], uint16x4_t>(value) };
    let result: float32x4_t;
    // SAFETY: FCVTL reads and writes SIMD and floating-point registers. Every
    // AArch64 target has it. The path requires FPCR.AHP to be zero, so the
    // lanes are IEEE binary16, and the widening is exact.
    unsafe {
        core::arch::asm!(
            "fcvtl {result:v}.4s, {a:v}.4h",
            a = in(vreg) a,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(single_lanes(result))
}

/// Returns four binary32 lanes rounded to binary16 in the rounding direction
/// of FPCR, by `FCVTN`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack F16C.
pub fn narrow_halves_x4(value: [f32; 4]) -> Option<[u16; 4]> {
    let a = singles(value);
    let result: uint16x4_t;
    // SAFETY: as in `widen_halves_x4`, with `FCVTN`, which changes only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvtn {result:v}.4h, {a:v}.4s",
            a = in(vreg) a,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: as in `widen_halves_x4`.
    Some(unsafe { transmute::<uint16x4_t, [u16; 4]>(result) })
}

/// Returns the masks of a comparison of each pair of lanes, by `FCMGT` in
/// both orders and `FCMEQ` of each operand with itself, on the arrangement
/// `$arrangement`. A lane is unordered when an operand is not equal to
/// itself.
macro_rules! compare_vector {
    ($arrangement:literal, $mask:ty, $x:ident, $y:ident) => {{
        let (less, greater, ordered): ($mask, $mask, $mask);
        // SAFETY: as in `vector!`, with five instructions.
        unsafe {
            core::arch::asm!(
                concat!("fcmgt {greater:v}", $arrangement, ", {x:v}", $arrangement, ", {y:v}", $arrangement),
                concat!("fcmgt {less:v}", $arrangement, ", {y:v}", $arrangement, ", {x:v}", $arrangement),
                concat!("fcmeq {ordered:v}", $arrangement, ", {x:v}", $arrangement, ", {x:v}", $arrangement),
                concat!("fcmeq {other:v}", $arrangement, ", {y:v}", $arrangement, ", {y:v}", $arrangement),
                "and {ordered:v}.16b, {ordered:v}.16b, {other:v}.16b",
                x = in(vreg) $x,
                y = in(vreg) $y,
                less = out(vreg) less,
                greater = out(vreg) greater,
                ordered = out(vreg) ordered,
                other = out(vreg) _,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        (less, greater, ordered)
    }};
}

/// Returns the masks of a comparison of each pair of binary32 lanes.
#[inline]
pub fn compare_f32x4(left: [f32; 4], right: [f32; 4]) -> Masks<u32, 4> {
    let (x, y) = (singles(left), singles(right));
    let (less, greater, ordered) = compare_vector!(".4s", uint32x4_t, x, y);
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value of
    // each.
    let bits = |value: uint32x4_t| unsafe { transmute::<uint32x4_t, [u32; 4]>(value) };
    Masks {
        less: bits(less),
        greater: bits(greater),
        unordered: bits(ordered).map(|mask| !mask),
    }
}

/// Returns the masks of a comparison of each pair of binary64 lanes.
#[inline]
pub fn compare_f64x2(left: [f64; 2], right: [f64; 2]) -> Masks<u64, 2> {
    let (x, y) = (doubles(left), doubles(right));
    let (less, greater, ordered) = compare_vector!(".2d", uint64x2_t, x, y);
    // SAFETY: as in `compare_f32x4`.
    let bits = |value: uint64x2_t| unsafe { transmute::<uint64x2_t, [u64; 2]>(value) };
    Masks {
        less: bits(less),
        greater: bits(greater),
        unordered: bits(ordered).map(|mask| !mask),
    }
}

/// Returns the smaller or the larger of each pair of binary32 lanes, as
/// `operation` selects, by `FMIN` or `FMAX`.
#[inline]
pub fn min_max_f32x4(left: [f32; 4], right: [f32; 4], operation: MinMax) -> [f32; 4] {
    let (mut a, b) = (singles(left), singles(right));
    if operation.is_minimum() {
        vector!("fmin {a:v}.4s, {a:v}.4s, {b:v}.4s", a, b);
    } else {
        vector!("fmax {a:v}.4s, {a:v}.4s, {b:v}.4s", a, b);
    }
    single_lanes(a)
}

/// Returns the smaller or the larger of each pair of binary64 lanes, as
/// `operation` selects, by `FMIN` or `FMAX`.
#[inline]
pub fn min_max_f64x2(left: [f64; 2], right: [f64; 2], operation: MinMax) -> [f64; 2] {
    let (mut a, b) = (doubles(left), doubles(right));
    if operation.is_minimum() {
        vector!("fmin {a:v}.2d, {a:v}.2d, {b:v}.2d", a, b);
    } else {
        vector!("fmax {a:v}.2d, {a:v}.2d, {b:v}.2d", a, b);
    }
    double_lanes(a)
}

/// Defines a function for eight binary16 lanes, which runs its form in
/// `half` where the build has `FEAT_FP16`, and returns `None` otherwise.
macro_rules! native {
    ($(#[$doc:meta])* $name:ident, ($($argument:ident: $type:ty),+) -> Option<$result:ty>) => {
        $(#[$doc])*
        #[cfg(target_feature = "fp16")]
        #[inline]
        pub fn $name($($argument: $type),+) -> Option<$result> {
            // SAFETY: the build enables `FEAT_FP16`, so the processor that
            // runs it has the feature.
            unsafe { half::$name($($argument),+) }
        }

        #[doc = "Returns `None`: the build has no `FEAT_FP16`."]
        #[cfg(not(target_feature = "fp16"))]
        #[inline]
        pub fn $name($(_: $type),+) -> Option<$result> {
            None
        }
    };
    ($(#[$doc:meta])* $name:ident, ($($argument:ident: $type:ty),+) -> $result:ty) => {
        $(#[$doc])*
        #[cfg(target_feature = "fp16")]
        #[inline]
        #[allow(clippy::unnecessary_wraps)] // A build without the feature returns `None` from the same signature.
        pub fn $name($($argument: $type),+) -> Option<$result> {
            // SAFETY: the build enables `FEAT_FP16`, so the processor that
            // runs it has the feature.
            Some(unsafe { half::$name($($argument),+) })
        }

        #[doc = "Returns `None`: the build has no `FEAT_FP16`."]
        #[cfg(not(target_feature = "fp16"))]
        #[inline]
        pub fn $name($(_: $type),+) -> Option<$result> {
            None
        }
    };
}

native!(
    /// Returns `operation` of eight pairs of binary16 lanes, by `FADD`,
    /// `FSUB`, `FMUL`, or `FDIV` on `.8H`.
    binary_f16x8, (left: [u16; 8], right: [u16; 8], operation: Operation) -> [u16; 8]
);
native!(
    /// Returns the square roots of eight binary16 lanes, by `FSQRT` on
    /// `.8H`.
    sqrt_f16x8, (value: [u16; 8]) -> [u16; 8]
);
native!(
    /// Returns eight binary16 lanes rounded to integral values in the
    /// direction `rounding`, by `FRINTN`, `FRINTA`, `FRINTM`, `FRINTP`, or
    /// `FRINTZ` on `.8H`, or `None` for a direction that no instruction has.
    round_f16x8, (value: [u16; 8], rounding: Rounding) -> Option<[u16; 8]>
);
native!(
    /// Returns `left * right + addend` of eight triples of binary16 lanes,
    /// each rounded once, by `FMLA` on `.8H`.
    mul_add_f16x8, (left: [u16; 8], right: [u16; 8], addend: [u16; 8]) -> [u16; 8]
);
native!(
    /// Returns the masks of a comparison of each pair of eight binary16
    /// lanes, by `FCMGT` and `FCMEQ` on `.8H`.
    compare_f16x8, (left: [u16; 8], right: [u16; 8]) -> Masks<u16, 8>
);
native!(
    /// Returns the smaller or the larger of each pair of eight binary16
    /// lanes, as `operation` selects, by `FMIN` or `FMAX` on `.8H`.
    min_max_f16x8, (left: [u16; 8], right: [u16; 8], operation: MinMax) -> [u16; 8]
);

#[cfg(target_feature = "fp16")]
mod half;

/// Defines a function for a chunk of a 256-bit or a 512-bit register, which
/// returns `None`.
macro_rules! no_wide {
    ($name:ident, ($($type:ty),+) -> $result:ty) => {
        #[doc = "Returns `None`: the base architecture has no registers wider than 128 bits."]
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
no_wide!(round_f32x8, ([f32; 8], Rounding) -> [f32; 8]);
no_wide!(round_f64x4, ([f64; 4], Rounding) -> [f64; 4]);
no_wide!(mul_add_f32x8, ([f32; 8], [f32; 8], [f32; 8]) -> [f32; 8]);
no_wide!(mul_add_f64x4, ([f64; 4], [f64; 4], [f64; 4]) -> [f64; 4]);
no_wide!(widen_x4, ([f32; 4]) -> [f64; 4]);
no_wide!(narrow_x4, ([f64; 4]) -> [f32; 4]);
no_wide!(widen_halves_x8, ([u16; 8]) -> [f32; 8]);
no_wide!(narrow_halves_x8, ([f32; 8]) -> [u16; 8]);
no_wide!(compare_f32x8, ([f32; 8], [f32; 8]) -> Masks<u32, 8>);
no_wide!(compare_f64x4, ([f64; 4], [f64; 4]) -> Masks<u64, 4>);
no_wide!(min_max_f32x8, ([f32; 8], [f32; 8], MinMax) -> [f32; 8]);
no_wide!(min_max_f64x4, ([f64; 4], [f64; 4], MinMax) -> [f64; 4]);
no_wide!(to_int_f32x8, ([f32; 8]) -> [i32; 8]);
no_wide!(to_int_f64x4, ([f64; 4]) -> [i32; 4]);
no_wide!(from_int_x8, ([i32; 8]) -> [f32; 8]);
no_wide!(binary_f32x16, ([f32; 16], [f32; 16], Operation) -> [f32; 16]);
no_wide!(mul_add_f32x16, ([f32; 16], [f32; 16], [f32; 16]) -> [f32; 16]);
no_wide!(min_max_f32x16, ([f32; 16], [f32; 16], MinMax) -> [f32; 16]);
no_wide!(round_f32x16, ([f32; 16], Rounding) -> [f32; 16]);
no_wide!(to_int_f32x16, ([f32; 16]) -> [i32; 16]);
no_wide!(from_int_x16, ([i32; 16]) -> [f32; 16]);
no_wide!(widen_halves_x16, ([u16; 16]) -> [f32; 16]);
no_wide!(narrow_halves_x16, ([f32; 16]) -> [u16; 16]);

/// Returns the integer indefinite in every lane, which sends each lane to
/// the scalar conversion. AArch64 `FCVTNS` saturates a lane outside the
/// range of `i32` and gives zero for a NaN, so no lane shows that the scalar
/// conversion must decide it. `INTEGERS` keeps the packed path from calling
/// this function.
#[inline]
pub fn to_int_f32x4(_value: [f32; 4]) -> [i32; 4] {
    [i32::MIN; 4]
}

/// Returns the integer indefinite in every lane, as `to_int_f32x4` does.
#[inline]
pub fn to_int_f64x2(_value: [f64; 2]) -> [i32; 2] {
    [i32::MIN; 2]
}
