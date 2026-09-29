//! The packed instructions of the SSE unit, for the paths of `Lanes`: 128-bit
//! registers with SSE2, SSE4.1, and FMA, and 256-bit registers with AVX.
//!
//! Each function computes one chunk of lanes in registers. The block takes
//! and gives register values, so it touches no memory: the compiler loads and
//! stores the lanes, and keeps them in registers between operations. A legacy
//! SSE instruction with a 128-bit memory operand faults on an address that is
//! not a multiple of 16, and a block with a memory operand stops the compiler
//! from keeping values in registers across it. A function for a feature that
//! the build does not have returns `None`.

#[cfg(target_feature = "f16c")]
use core::arch::x86_64::__m128i;
use core::arch::x86_64::{__m128, __m128d};
use core::mem::transmute;

use super::super::Operation;
use super::super::packed::Masks;
use super::sse;
use crate::env::Rounding;
use crate::format::internal::MinMax;

/// `true` when the build has 256-bit registers, which hold eight binary32 or
/// four binary64 lanes.
pub const WIDE: bool = cfg!(target_feature = "avx");

/// `true` when the build has F16C, which widens binary16 lanes to binary32
/// and rounds binary32 lanes to binary16.
pub const HALF: bool = cfg!(target_feature = "f16c");

/// Returns four binary32 lanes as the value of an SSE register.
#[inline]
fn singles(lanes: [f32; 4]) -> __m128 {
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value of
    // each.
    unsafe { transmute::<[f32; 4], __m128>(lanes) }
}

/// Returns the four binary32 lanes of an SSE register value.
#[inline]
fn single_lanes(value: __m128) -> [f32; 4] {
    // SAFETY: as in `singles`.
    unsafe { transmute::<__m128, [f32; 4]>(value) }
}

/// Returns two binary64 lanes as the value of an SSE register.
#[inline]
fn doubles(lanes: [f64; 2]) -> __m128d {
    // SAFETY: as in `singles`.
    unsafe { transmute::<[f64; 2], __m128d>(lanes) }
}

/// Returns the two binary64 lanes of an SSE register value.
#[inline]
fn double_lanes(value: __m128d) -> [f64; 2] {
    // SAFETY: as in `singles`.
    unsafe { transmute::<__m128d, [f64; 2]>(value) }
}

/// Runs one packed instruction on the register values `$a` and `$b` of the
/// class `$class`, and leaves the result in `$a`.
macro_rules! packed {
    ($class:ident, $instruction:expr, $a:ident, $b:ident) => {
        // SAFETY: the instruction reads and writes registers of the class,
        // which the build enables, and `sse!` selects a VEX form only in a
        // build with AVX. The instruction changes only the status flags of
        // MXCSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                a = inout($class) $a,
                b = in($class) $b,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Runs the instruction of `$operation` from its four templates on `$a` and
/// `$b`, and leaves the result in `$a`.
macro_rules! arithmetic {
    ($class:ident, $operation:expr, [$add:expr, $sub:expr, $mul:expr, $div:expr $(,)?], $a:ident, $b:ident) => {
        match $operation {
            Operation::Add => packed!($class, $add, $a, $b),
            Operation::Sub => packed!($class, $sub, $a, $b),
            Operation::Mul => packed!($class, $mul, $a, $b),
            Operation::Div => packed!($class, $div, $a, $b),
        }
    };
}

/// Returns `operation` of four pairs of binary32 lanes, by `ADDPS`, `SUBPS`,
/// `MULPS`, or `DIVPS`.
#[inline]
pub fn binary_f32x4(left: [f32; 4], right: [f32; 4], operation: Operation) -> [f32; 4] {
    let (mut a, b) = (singles(left), singles(right));
    arithmetic!(
        xmm_reg,
        operation,
        [
            sse!("addps {a}, {b}", "vaddps {a}, {a}, {b}"),
            sse!("subps {a}, {b}", "vsubps {a}, {a}, {b}"),
            sse!("mulps {a}, {b}", "vmulps {a}, {a}, {b}"),
            sse!("divps {a}, {b}", "vdivps {a}, {a}, {b}"),
        ],
        a,
        b
    );
    single_lanes(a)
}

/// Returns `operation` of two pairs of binary64 lanes, by `ADDPD`, `SUBPD`,
/// `MULPD`, or `DIVPD`.
#[inline]
pub fn binary_f64x2(left: [f64; 2], right: [f64; 2], operation: Operation) -> [f64; 2] {
    let (mut a, b) = (doubles(left), doubles(right));
    arithmetic!(
        xmm_reg,
        operation,
        [
            sse!("addpd {a}, {b}", "vaddpd {a}, {a}, {b}"),
            sse!("subpd {a}, {b}", "vsubpd {a}, {a}, {b}"),
            sse!("mulpd {a}, {b}", "vmulpd {a}, {a}, {b}"),
            sse!("divpd {a}, {b}", "vdivpd {a}, {a}, {b}"),
        ],
        a,
        b
    );
    double_lanes(a)
}

/// Returns the square roots of four binary32 lanes, by `SQRTPS`.
#[inline]
pub fn sqrt_f32x4(value: [f32; 4]) -> [f32; 4] {
    let mut a = singles(value);
    packed!(xmm_reg, sse!("sqrtps {a}, {b}", "vsqrtps {a}, {b}"), a, a);
    single_lanes(a)
}

/// Returns the square roots of two binary64 lanes, by `SQRTPD`.
#[inline]
pub fn sqrt_f64x2(value: [f64; 2]) -> [f64; 2] {
    let mut a = doubles(value);
    packed!(xmm_reg, sse!("sqrtpd {a}, {b}", "vsqrtpd {a}, {b}"), a, a);
    double_lanes(a)
}

/// Runs one packed rounding to an integral value on `$a`, in registers of
/// the class `$class`, with the rounding control of `$rounding` in the
/// immediate of the instruction: 0 to nearest even, 1 toward negative
/// infinity, 2 toward positive infinity, and 3 toward zero. Returns `None`
/// from the function for another direction.
#[cfg(target_feature = "sse4.1")]
macro_rules! round_packed {
    ($class:ident, $rounding:expr, [$even:expr, $down:expr, $up:expr, $zero:expr $(,)?], $a:ident) => {
        match $rounding {
            Rounding::NearestEven => packed!($class, $even, $a, $a),
            Rounding::TowardNegative => packed!($class, $down, $a, $a),
            Rounding::TowardPositive => packed!($class, $up, $a, $a),
            Rounding::TowardZero => packed!($class, $zero, $a, $a),
            Rounding::NearestAway | Rounding::ToOdd => return None,
        }
    };
}

/// Returns four binary32 lanes rounded to integral values in the direction
/// `rounding`, by `ROUNDPS` with the direction in its immediate, or `None`
/// for a direction that the immediate does not have.
#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn round_f32x4(value: [f32; 4], rounding: Rounding) -> Option<[f32; 4]> {
    let mut a = singles(value);
    round_packed!(
        xmm_reg,
        rounding,
        [
            sse!("roundps {a}, {b}, 0", "vroundps {a}, {b}, 0"),
            sse!("roundps {a}, {b}, 1", "vroundps {a}, {b}, 1"),
            sse!("roundps {a}, {b}, 2", "vroundps {a}, {b}, 2"),
            sse!("roundps {a}, {b}, 3", "vroundps {a}, {b}, 3"),
        ],
        a
    );
    Some(single_lanes(a))
}

/// Returns two binary64 lanes rounded to integral values in the direction
/// `rounding`, by `ROUNDPD` with the direction in its immediate, or `None`
/// for a direction that the immediate does not have.
#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn round_f64x2(value: [f64; 2], rounding: Rounding) -> Option<[f64; 2]> {
    let mut a = doubles(value);
    round_packed!(
        xmm_reg,
        rounding,
        [
            sse!("roundpd {a}, {b}, 0", "vroundpd {a}, {b}, 0"),
            sse!("roundpd {a}, {b}, 1", "vroundpd {a}, {b}, 1"),
            sse!("roundpd {a}, {b}, 2", "vroundpd {a}, {b}, 2"),
            sse!("roundpd {a}, {b}, 3", "vroundpd {a}, {b}, 3"),
        ],
        a
    );
    Some(double_lanes(a))
}

/// Returns `None`: a build without SSE4.1 has no packed rounding.
#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn round_f32x4(_value: [f32; 4], _rounding: Rounding) -> Option<[f32; 4]> {
    None
}

/// Returns `None`: a build without SSE4.1 has no packed rounding.
#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn round_f64x2(_value: [f64; 2], _rounding: Rounding) -> Option<[f64; 2]> {
    None
}

/// Runs a packed fused multiply-add, `$a = $a * $b + $c`, in registers of the
/// class `$class`.
#[cfg(target_feature = "fma")]
macro_rules! fused {
    ($class:ident, $instruction:literal, $a:ident, $b:ident, $c:ident) => {
        // SAFETY: as in `packed!`, with a third register. The build enables
        // FMA.
        unsafe {
            core::arch::asm!(
                concat!($instruction, " {a}, {b}, {c}"),
                a = inout($class) $a,
                b = in($class) $b,
                c = in($class) $c,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Returns `left * right + addend` of four triples of binary32 lanes, each
/// rounded once, by `VFMADD213PS`.
#[cfg(target_feature = "fma")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without FMA returns `None` from the same signature.
pub fn mul_add_f32x4(left: [f32; 4], right: [f32; 4], addend: [f32; 4]) -> Option<[f32; 4]> {
    let (mut a, b, c) = (singles(left), singles(right), singles(addend));
    fused!(xmm_reg, "vfmadd213ps", a, b, c);
    Some(single_lanes(a))
}

/// Returns `left * right + addend` of two triples of binary64 lanes, each
/// rounded once, by `VFMADD213PD`.
#[cfg(target_feature = "fma")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without FMA returns `None` from the same signature.
pub fn mul_add_f64x2(left: [f64; 2], right: [f64; 2], addend: [f64; 2]) -> Option<[f64; 2]> {
    let (mut a, b, c) = (doubles(left), doubles(right), doubles(addend));
    fused!(xmm_reg, "vfmadd213pd", a, b, c);
    Some(double_lanes(a))
}

/// Returns `None`: a build without FMA has no packed fused multiply-add.
#[cfg(not(target_feature = "fma"))]
#[inline]
pub fn mul_add_f32x4(_left: [f32; 4], _right: [f32; 4], _addend: [f32; 4]) -> Option<[f32; 4]> {
    None
}

/// Returns `None`: a build without FMA has no packed fused multiply-add.
#[cfg(not(target_feature = "fma"))]
#[inline]
pub fn mul_add_f64x2(_left: [f64; 2], _right: [f64; 2], _addend: [f64; 2]) -> Option<[f64; 2]> {
    None
}

/// Returns two binary32 lanes widened exactly to binary64, by `CVTPS2PD`.
#[inline]
pub fn widen_x2(value: [f32; 2]) -> [f64; 2] {
    let a = singles([value[0], value[1], 0.0, 0.0]);
    let result: __m128d;
    // SAFETY: CVTPS2PD reads and writes SSE registers. SSE2 is part of every
    // x86-64 target, and `sse!` selects a VEX form only in a build with AVX.
    // The conversion changes only the status flags of MXCSR, which floaty
    // does not read.
    unsafe {
        core::arch::asm!(
            sse!("cvtps2pd {result}, {a}", "vcvtps2pd {result}, {a}"),
            a = in(xmm_reg) a,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    double_lanes(result)
}

/// Returns two binary64 lanes rounded to binary32, by `CVTPD2PS`.
#[inline]
pub fn narrow_x2(value: [f64; 2]) -> [f32; 2] {
    let a = doubles(value);
    let result: __m128;
    // SAFETY: as in `widen_x2`, with `CVTPD2PS`.
    unsafe {
        core::arch::asm!(
            sse!("cvtpd2ps {result}, {a}", "vcvtpd2ps {result}, {a}"),
            a = in(xmm_reg) a,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    let [low, high, _, _] = single_lanes(result);
    [low, high]
}

/// Returns four binary16 lanes widened exactly to binary32, by `VCVTPH2PS`.
#[cfg(target_feature = "f16c")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without F16C returns `None` from the same signature.
pub fn widen_halves_x4(value: [u16; 4]) -> Option<[f32; 4]> {
    let [a0, a1, a2, a3] = value;
    // SAFETY: both types hold 16 bytes, and every bit pattern is a value of
    // each.
    let a = unsafe { transmute::<[u16; 8], __m128i>([a0, a1, a2, a3, 0, 0, 0, 0]) };
    let result: __m128;
    // SAFETY: VCVTPH2PS reads and writes SSE registers. The build enables
    // F16C, and the widening is exact, so it changes no state.
    unsafe {
        core::arch::asm!(
            "vcvtph2ps {result}, {a}",
            a = in(xmm_reg) a,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(single_lanes(result))
}

/// Returns four binary32 lanes rounded to binary16 to nearest even, by
/// `VCVTPS2PH` with the rounding control 0 in its immediate.
#[cfg(target_feature = "f16c")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without F16C returns `None` from the same signature.
pub fn narrow_halves_x4(value: [f32; 4]) -> Option<[u16; 4]> {
    let a = singles(value);
    let result: __m128i;
    // SAFETY: VCVTPS2PH reads and writes SSE registers. The build enables
    // F16C, and the rounding changes only the status flags of MXCSR, which
    // floaty does not read.
    unsafe {
        core::arch::asm!(
            "vcvtps2ph {result}, {a}, 0",
            a = in(xmm_reg) a,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // SAFETY: as in `widen_halves_x4`.
    let [r0, r1, r2, r3, _, _, _, _] = unsafe { transmute::<__m128i, [u16; 8]>(result) };
    Some([r0, r1, r2, r3])
}

/// Returns `None`: a build without F16C has no packed binary16 path.
#[cfg(not(target_feature = "f16c"))]
#[inline]
pub fn widen_halves_x4(_value: [u16; 4]) -> Option<[f32; 4]> {
    None
}

/// Returns `None`: a build without F16C has no packed binary16 path.
#[cfg(not(target_feature = "f16c"))]
#[inline]
pub fn narrow_halves_x4(_value: [f32; 4]) -> Option<[u16; 4]> {
    None
}

/// Returns the masks of a comparison of each pair of binary32 lanes, by
/// `CMPLTPS` and `CMPUNORDPS`.
#[inline]
pub fn compare_f32x4(left: [f32; 4], right: [f32; 4]) -> Masks<u32, 4> {
    let (x, y) = (singles(left), singles(right));
    let (mut less, mut greater, mut unordered) = (x, y, x);
    packed!(
        xmm_reg,
        sse!("cmpltps {a}, {b}", "vcmpltps {a}, {a}, {b}"),
        less,
        y
    );
    packed!(
        xmm_reg,
        sse!("cmpltps {a}, {b}", "vcmpltps {a}, {a}, {b}"),
        greater,
        x
    );
    packed!(
        xmm_reg,
        sse!("cmpunordps {a}, {b}", "vcmpunordps {a}, {a}, {b}"),
        unordered,
        y
    );
    let bits = |value: __m128| single_lanes(value).map(f32::to_bits);
    Masks {
        less: bits(less),
        greater: bits(greater),
        unordered: bits(unordered),
    }
}

/// Returns the masks of a comparison of each pair of binary64 lanes, by
/// `CMPLTPD` and `CMPUNORDPD`.
#[inline]
pub fn compare_f64x2(left: [f64; 2], right: [f64; 2]) -> Masks<u64, 2> {
    let (x, y) = (doubles(left), doubles(right));
    let (mut less, mut greater, mut unordered) = (x, y, x);
    packed!(
        xmm_reg,
        sse!("cmpltpd {a}, {b}", "vcmpltpd {a}, {a}, {b}"),
        less,
        y
    );
    packed!(
        xmm_reg,
        sse!("cmpltpd {a}, {b}", "vcmpltpd {a}, {a}, {b}"),
        greater,
        x
    );
    packed!(
        xmm_reg,
        sse!("cmpunordpd {a}, {b}", "vcmpunordpd {a}, {a}, {b}"),
        unordered,
        y
    );
    let bits = |value: __m128d| double_lanes(value).map(f64::to_bits);
    Masks {
        less: bits(less),
        greater: bits(greater),
        unordered: bits(unordered),
    }
}

/// Returns the smaller or the larger of each pair of binary32 lanes, as
/// `operation` selects, by `MINPS` or `MAXPS`, which give the right lane
/// when a lane is a NaN or both are zeros.
#[inline]
pub fn min_max_f32x4(left: [f32; 4], right: [f32; 4], operation: MinMax) -> [f32; 4] {
    let (mut a, b) = (singles(left), singles(right));
    if operation.is_minimum() {
        packed!(
            xmm_reg,
            sse!("minps {a}, {b}", "vminps {a}, {a}, {b}"),
            a,
            b
        );
    } else {
        packed!(
            xmm_reg,
            sse!("maxps {a}, {b}", "vmaxps {a}, {a}, {b}"),
            a,
            b
        );
    }
    single_lanes(a)
}

/// Returns the smaller or the larger of each pair of binary64 lanes, as
/// `operation` selects, by `MINPD` or `MAXPD`, which give the right lane
/// when a lane is a NaN or both are zeros.
#[inline]
pub fn min_max_f64x2(left: [f64; 2], right: [f64; 2], operation: MinMax) -> [f64; 2] {
    let (mut a, b) = (doubles(left), doubles(right));
    if operation.is_minimum() {
        packed!(
            xmm_reg,
            sse!("minpd {a}, {b}", "vminpd {a}, {a}, {b}"),
            a,
            b
        );
    } else {
        packed!(
            xmm_reg,
            sse!("maxpd {a}, {b}", "vmaxpd {a}, {a}, {b}"),
            a,
            b
        );
    }
    double_lanes(a)
}

/// Defines a function for a 256-bit chunk, which runs its form in `wide`
/// where the build has the features, and returns `None` otherwise.
macro_rules! wide {
    ($(#[$doc:meta])* $name:ident, $features:meta, ($($argument:ident: $type:ty),+) -> Option<$result:ty>) => {
        $(#[$doc])*
        #[cfg($features)]
        #[inline]
        pub fn $name($($argument: $type),+) -> Option<$result> {
            // SAFETY: the build enables the features of the 256-bit form, so
            // the processor that runs it has them.
            unsafe { wide::$name($($argument),+) }
        }

        #[doc = "Returns `None`: the build has no 256-bit form."]
        #[cfg(not($features))]
        #[inline]
        pub fn $name($(_: $type),+) -> Option<$result> {
            None
        }
    };
    ($(#[$doc:meta])* $name:ident, $features:meta, ($($argument:ident: $type:ty),+) -> $result:ty) => {
        $(#[$doc])*
        #[cfg($features)]
        #[inline]
        #[allow(clippy::unnecessary_wraps)] // A build without the feature returns `None` from the same signature.
        pub fn $name($($argument: $type),+) -> Option<$result> {
            // SAFETY: the build enables the features of the 256-bit form, so
            // the processor that runs it has them.
            Some(unsafe { wide::$name($($argument),+) })
        }

        #[doc = "Returns `None`: the build has no 256-bit form."]
        #[cfg(not($features))]
        #[inline]
        pub fn $name($(_: $type),+) -> Option<$result> {
            None
        }
    };
}

wide!(
    /// Returns `operation` of eight pairs of binary32 lanes, by `VADDPS`,
    /// `VSUBPS`, `VMULPS`, or `VDIVPS`.
    binary_f32x8, target_feature = "avx", (left: [f32; 8], right: [f32; 8], operation: Operation) -> [f32; 8]
);
wide!(
    /// Returns `operation` of four pairs of binary64 lanes, by `VADDPD`,
    /// `VSUBPD`, `VMULPD`, or `VDIVPD`.
    binary_f64x4, target_feature = "avx", (left: [f64; 4], right: [f64; 4], operation: Operation) -> [f64; 4]
);
wide!(
    /// Returns the square roots of eight binary32 lanes, by `VSQRTPS`.
    sqrt_f32x8, target_feature = "avx", (value: [f32; 8]) -> [f32; 8]
);
wide!(
    /// Returns the square roots of four binary64 lanes, by `VSQRTPD`.
    sqrt_f64x4, target_feature = "avx", (value: [f64; 4]) -> [f64; 4]
);
wide!(
    /// Returns eight binary32 lanes rounded to integral values in the
    /// direction `rounding`, by `VROUNDPS` with the direction in its
    /// immediate, or `None` for a direction that the immediate does not have.
    round_f32x8, target_feature = "avx", (value: [f32; 8], rounding: Rounding) -> Option<[f32; 8]>
);
wide!(
    /// Returns four binary64 lanes rounded to integral values in the
    /// direction `rounding`, by `VROUNDPD` with the direction in its
    /// immediate, or `None` for a direction that the immediate does not have.
    round_f64x4, target_feature = "avx", (value: [f64; 4], rounding: Rounding) -> Option<[f64; 4]>
);
wide!(
    /// Returns `left * right + addend` of eight triples of binary32 lanes,
    /// each rounded once, by `VFMADD213PS`.
    mul_add_f32x8, all(target_feature = "avx", target_feature = "fma"), (left: [f32; 8], right: [f32; 8], addend: [f32; 8]) -> [f32; 8]
);
wide!(
    /// Returns `left * right + addend` of four triples of binary64 lanes,
    /// each rounded once, by `VFMADD213PD`.
    mul_add_f64x4, all(target_feature = "avx", target_feature = "fma"), (left: [f64; 4], right: [f64; 4], addend: [f64; 4]) -> [f64; 4]
);
wide!(
    /// Returns four binary32 lanes widened exactly to binary64, by
    /// `VCVTPS2PD`.
    widen_x4, target_feature = "avx", (value: [f32; 4]) -> [f64; 4]
);
wide!(
    /// Returns four binary64 lanes rounded to binary32, by `VCVTPD2PS`.
    narrow_x4, target_feature = "avx", (value: [f64; 4]) -> [f32; 4]
);
wide!(
    /// Returns eight binary16 lanes widened exactly to binary32, by
    /// `VCVTPH2PS`.
    widen_halves_x8, all(target_feature = "avx", target_feature = "f16c"), (value: [u16; 8]) -> [f32; 8]
);
wide!(
    /// Returns eight binary32 lanes rounded to binary16 to nearest even, by
    /// `VCVTPS2PH` with the rounding control 0 in its immediate.
    narrow_halves_x8, all(target_feature = "avx", target_feature = "f16c"), (value: [f32; 8]) -> [u16; 8]
);
wide!(
    /// Returns the masks of a comparison of each pair of eight binary32
    /// lanes, by `VCMPLTPS` and `VCMPUNORDPS`.
    compare_f32x8, target_feature = "avx", (left: [f32; 8], right: [f32; 8]) -> Masks<u32, 8>
);
wide!(
    /// Returns the masks of a comparison of each pair of four binary64 lanes,
    /// by `VCMPLTPD` and `VCMPUNORDPD`.
    compare_f64x4, target_feature = "avx", (left: [f64; 4], right: [f64; 4]) -> Masks<u64, 4>
);
wide!(
    /// Returns the smaller or the larger of each pair of eight binary32
    /// lanes, by `VMINPS` or `VMAXPS`.
    min_max_f32x8, target_feature = "avx", (left: [f32; 8], right: [f32; 8], operation: MinMax) -> [f32; 8]
);
wide!(
    /// Returns the smaller or the larger of each pair of four binary64
    /// lanes, by `VMINPD` or `VMAXPD`.
    min_max_f64x4, target_feature = "avx", (left: [f64; 4], right: [f64; 4], operation: MinMax) -> [f64; 4]
);

#[cfg(target_feature = "avx")]
mod wide;

#[cfg(test)]
mod tests;
