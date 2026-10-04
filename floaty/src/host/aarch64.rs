//! The floating-point environment of AArch64, FPCR, and the instructions of
//! the host paths.

use core::cmp::Ordering;

use super::{Operation, bits};
use crate::env::Rounding;

pub mod packed;

/// `true`: the build has the floating-point unit.
pub const UNIT: bool = true;
/// `true`: every AArch64 target has the fused multiply-add.
pub const FUSED: bool = true;
/// `true`: every AArch64 target widens binary16 to binary32 and rounds
/// binary32 to binary16 in `FCVT`.
pub const HALF: bool = true;
/// `true`: every AArch64 target rounds to an integral value in `FRINT`.
pub const ROUNDING: bool = true;
/// `false`: AArch64 has no x87 unit.
pub const X87: bool = false;
/// `false`: AArch64 has no x87 unit.
pub const X87_FULL_PRECISION: bool = false;
/// `true`: `FEAT_FP16` computes the binary16 fused multiply-add in its own
/// precision, and every other AArch64 target computes it through binary64.
pub const HALF_FUSED: bool = true;
/// `true`: `FCVT` rounds binary64 to binary16 once.
pub const DOUBLE_TO_HALF: bool = true;
/// `true`: `FCVTXN` rounds binary64 to binary32 with round to odd.
pub const ROUND_TO_ODD: bool = true;

/// Returns `true` when the floating-point unit rounds to nearest even without
/// flushing, and enables no exception trap.
///
/// Rust code runs with FPCR zero, but an emulator can load another value. The
/// path then goes back to the engine. The fields are those of FPCR in the Arm
/// Architecture Registers, DDI 0601: `RMode` rounds to nearest when it is zero,
/// FZ, FZ16, and FIZ flush, AH changes the flushing and the handling of
/// denormal numbers, and IOE, DZE, OFE, UFE, IXE, and IDE enable traps. AHP
/// selects another half-precision format. DN and NEP change only a NaN result
/// and the other elements of a vector register, which the path does not use.
#[inline]
pub fn default_environment() -> bool {
    // FIZ (0), AH (1), IOE to IXE (8 to 12), IDE (15), FZ16 (19), `RMode` (22
    // and 23), FZ (24), and AHP (26): each is zero by default.
    const RESULT_FIELDS: u64 =
        0b11 | (0x1F << 8) | (1 << 15) | (1 << 19) | (3 << 22) | (1 << 24) | (1 << 26);
    let fpcr: u64;
    // SAFETY: MRS reads FPCR into a register. It reads no memory and changes
    // no other state.
    unsafe {
        core::arch::asm!(
            "mrs {fpcr}, fpcr",
            fpcr = out(reg) fpcr,
            options(nomem, nostack, preserves_flags),
        );
    }
    fpcr & RESULT_FIELDS == 0
}

/// Runs one scalar instruction of the floating-point unit on `$left` and
/// `$right`, and leaves the result in `$left`.
macro_rules! scalar {
    ($instruction:literal, $left:ident, $right:ident) => {
        // SAFETY: the instruction reads and writes SIMD and floating-point
        // registers. Every AArch64 target has it, and it changes only the
        // status flags of FPSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                left = inout(vreg) $left,
                right = in(vreg) $right,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Returns `operation` of two binary32 values, by `FADD`, `FSUB`, `FMUL`, or
/// `FDIV` in the rounding direction of FPCR.
#[inline]
pub fn binary_f32(mut left: f32, right: f32, operation: Operation) -> f32 {
    match operation {
        Operation::Add => scalar!("fadd {left:s}, {left:s}, {right:s}", left, right),
        Operation::Sub => scalar!("fsub {left:s}, {left:s}, {right:s}", left, right),
        Operation::Mul => scalar!("fmul {left:s}, {left:s}, {right:s}", left, right),
        Operation::Div => scalar!("fdiv {left:s}, {left:s}, {right:s}", left, right),
    }
    left
}

/// Returns `operation` of two binary64 values, by `FADD`, `FSUB`, `FMUL`, or
/// `FDIV` in the rounding direction of FPCR.
#[inline]
pub fn binary_f64(mut left: f64, right: f64, operation: Operation) -> f64 {
    match operation {
        Operation::Add => scalar!("fadd {left:d}, {left:d}, {right:d}", left, right),
        Operation::Sub => scalar!("fsub {left:d}, {left:d}, {right:d}", left, right),
        Operation::Mul => scalar!("fmul {left:d}, {left:d}, {right:d}", left, right),
        Operation::Div => scalar!("fdiv {left:d}, {left:d}, {right:d}", left, right),
    }
    left
}

/// Returns a binary32 value widened exactly to binary64, by `FCVT`.
#[inline]
pub fn widen_single(value: f32) -> f64 {
    let result: f64;
    // SAFETY: FCVT reads and writes SIMD and floating-point registers. Every
    // AArch64 target has the instruction, and the widening changes only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvt {result:d}, {value:s}",
            value = in(vreg) value,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Returns the square root of a binary32 value, by `FSQRT`.
#[inline]
pub fn sqrt_f32(mut value: f32) -> f32 {
    // SAFETY: FSQRT reads and writes one SIMD and floating-point register.
    // Every AArch64 target has the instruction, and it changes only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fsqrt {value:s}, {value:s}",
            value = inout(vreg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    value
}

/// Returns the square root of a binary64 value, by `FSQRT`.
#[inline]
pub fn sqrt_f64(mut value: f64) -> f64 {
    // SAFETY: FSQRT reads and writes one SIMD and floating-point register.
    // Every AArch64 target has the instruction, and it changes only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fsqrt {value:d}, {value:d}",
            value = inout(vreg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    value
}

/// Returns `left * right + addend`, rounded once, by `FMADD`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack FMA.
pub fn mul_add_f32(left: f32, right: f32, addend: f32) -> Option<f32> {
    let result: f32;
    // SAFETY: FMADD computes `addend + left * right` in SIMD and
    // floating-point registers. Every AArch64 target has the instruction, and
    // it changes only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fmadd {result:s}, {left:s}, {right:s}, {addend:s}",
            result = lateout(vreg) result,
            left = in(vreg) left,
            right = in(vreg) right,
            addend = in(vreg) addend,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(result)
}

/// Returns the square root of a binary32 value for the steps of a block, by
/// an operation that LLVM sees: `FSQRT`.
#[inline]
pub fn block_sqrt(value: f32) -> f32 {
    use core::arch::aarch64::{vdup_n_f32, vget_lane_f32, vsqrt_f32};
    // SAFETY: the intrinsics need NEON, which every AArch64 target has. Both
    // lanes hold the value, so lane 0 holds its square root.
    unsafe { vget_lane_f32::<0>(vsqrt_f32(vdup_n_f32(value))) }
}

/// Returns `left * right + addend`, rounded once, for the steps of a block,
/// by an operation that LLVM sees: `FMADD`, or `FMLA` in a vectorized loop.
///
/// # Safety
///
/// None beyond the call: every AArch64 processor has the instruction. The
/// function is unsafe as on x86, where a processor can lack FMA.
#[inline]
pub unsafe fn block_mul_add(left: f32, right: f32, addend: f32) -> f32 {
    use core::arch::aarch64::{vdup_n_f32, vfmas_lane_f32};
    // SAFETY: the intrinsics need NEON, which every AArch64 target has.
    unsafe { vfmas_lane_f32::<0>(addend, left, vdup_n_f32(right)) }
}

/// Returns `left * right + addend`, rounded once, by `FMADD`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack FMA.
pub fn mul_add_f64(left: f64, right: f64, addend: f64) -> Option<f64> {
    let result: f64;
    // SAFETY: FMADD computes `addend + left * right` in SIMD and
    // floating-point registers. Every AArch64 target has the instruction, and
    // it changes only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fmadd {result:d}, {left:d}, {right:d}, {addend:d}",
            result = lateout(vreg) result,
            left = in(vreg) left,
            right = in(vreg) right,
            addend = in(vreg) addend,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(result)
}

/// Returns a binary16 value widened exactly to binary32, by `FCVT`.
#[inline]
pub fn widen_half(bits: u16) -> f32 {
    let value: f32;
    // SAFETY: FMOV moves the bits into the low half of a SIMD and
    // floating-point register, and FCVT widens them. Every AArch64 target has
    // both instructions, and the widening is exact, so it changes no state.
    unsafe {
        core::arch::asm!(
            "fmov {value:s}, {bits:w}",
            "fcvt {value:s}, {value:h}",
            bits = in(reg) u32::from(bits),
            value = out(vreg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    value
}

/// Returns a binary32 value rounded to binary16 in the rounding direction of
/// FPCR, which the path requires to be to nearest even, by `FCVT`.
#[inline]
pub fn narrow_half(value: f32) -> u16 {
    let bits: u32;
    // SAFETY: FCVT rounds in a SIMD and floating-point register and clears
    // its other bits, and FMOV moves the low 32 bits to a general register.
    // Every AArch64 target has both instructions, and the rounding changes
    // only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvt {value:h}, {value:s}",
            "fmov {bits:w}, {value:s}",
            value = inout(vreg) value => _,
            bits = out(reg) bits,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits")
}

/// `true` when `narrow_bfloat` rounds in integer instructions: in a build
/// without `FEAT_BF16`. `BFCVT` reads FPCR.
pub const NARROW_BFLOAT_IN_INTEGERS: bool = !cfg!(target_feature = "bf16");

/// Returns a binary32 value rounded to bfloat16 in the rounding direction of
/// FPCR, which the path requires to be to nearest even, by `BFCVT`. The Arm
/// Architecture Reference Manual states that `BFCVT` honors every control of
/// FPCR that applies to single-precision arithmetic.
#[cfg(target_feature = "bf16")]
#[inline]
pub fn narrow_bfloat(value: f32) -> u16 {
    // SAFETY: the build enables `FEAT_BF16`, so the processor that runs it
    // has the feature.
    unsafe { bfcvt(value) }
}

/// Rounds a binary32 value to bfloat16 by `BFCVT`.
///
/// The function enables `FEAT_BF16` for its own code. A caller compiled
/// without the feature, such as a doctest, which does not take `RUSTFLAGS`,
/// then still assembles the instruction.
///
/// # Safety
///
/// The processor must have `FEAT_BF16`.
#[cfg(target_feature = "bf16")]
#[target_feature(enable = "bf16")]
#[inline]
unsafe fn bfcvt(value: f32) -> u16 {
    let bits: u32;
    // SAFETY: BFCVT rounds in SIMD and floating-point registers and clears
    // the other bits of its destination, and FMOV moves the low 32 bits to a
    // general register. The caller guarantees `FEAT_BF16`, and the rounding
    // changes only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "bfcvt {half:h}, {value:s}",
            "fmov {bits:w}, {half:s}",
            value = in(vreg) value,
            half = out(vreg) _,
            bits = out(reg) bits,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits")
}

/// Returns a binary32 value rounded to bfloat16 to nearest even, by the
/// integer rounding of the host paths, in a build without `FEAT_BF16`.
#[cfg(not(target_feature = "bf16"))]
#[inline]
pub fn narrow_bfloat(value: f32) -> u16 {
    super::narrow::round_to_bfloat(value.to_bits())
}

/// Returns `left * right + addend` of binary16 encodings, rounded once, by
/// `FMADD` on half-precision registers.
#[cfg(target_feature = "fp16")]
#[inline]
pub fn mul_add_f16(left: u16, right: u16, addend: u16) -> u16 {
    // SAFETY: the build enables `FEAT_FP16`, so the processor that runs it
    // has the feature.
    unsafe { fmadd_f16(left, right, addend) }
}

/// Returns `left * right + addend` of binary16 encodings by `FMADD`.
///
/// The function enables `FEAT_FP16` for its own code, as `bfcvt` does for
/// `FEAT_BF16`.
///
/// # Safety
///
/// The processor must have `FEAT_FP16`.
#[cfg(target_feature = "fp16")]
#[target_feature(enable = "fp16")]
#[inline]
unsafe fn fmadd_f16(left: u16, right: u16, addend: u16) -> u16 {
    let bits: u32;
    // SAFETY: FMOV moves each encoding into a half-precision register, FMADD
    // computes `addend + left * right` there, and FMOV moves the result back.
    // The caller guarantees `FEAT_FP16`, and FMADD changes only the status
    // flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fmov {a:h}, {left:w}",
            "fmov {b:h}, {right:w}",
            "fmov {c:h}, {addend:w}",
            "fmadd {a:h}, {a:h}, {b:h}, {c:h}",
            "fmov {bits:w}, {a:h}",
            left = in(reg) u32::from(left),
            right = in(reg) u32::from(right),
            addend = in(reg) u32::from(addend),
            a = out(vreg) _,
            b = out(vreg) _,
            c = out(vreg) _,
            bits = lateout(reg) bits,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits")
}

/// Returns `left * right + addend` of binary16 encodings, rounded once, in a
/// build without `FEAT_FP16`. `FCVT` widens each operand to binary64 exactly,
/// `FMADD` computes in binary64, and `FCVT` rounds the result to binary16.
/// [`Host::Half`](super::Host::Half) states why the two roundings give the
/// result of one.
#[cfg(not(target_feature = "fp16"))]
#[inline]
pub fn mul_add_f16(left: u16, right: u16, addend: u16) -> u16 {
    let bits: u32;
    // SAFETY: FMOV moves each encoding into a SIMD and floating-point
    // register, FCVT widens it to binary64, FMADD computes `addend + left *
    // right`, FCVT rounds the result to binary16, and FMOV moves it back.
    // Every AArch64 target has these instructions, and they change only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fmov {a:s}, {left:w}",
            "fcvt {a:d}, {a:h}",
            "fmov {b:s}, {right:w}",
            "fcvt {b:d}, {b:h}",
            "fmov {c:s}, {addend:w}",
            "fcvt {c:d}, {c:h}",
            "fmadd {a:d}, {a:d}, {b:d}, {c:d}",
            "fcvt {a:h}, {a:d}",
            "fmov {bits:w}, {a:s}",
            left = in(reg) u32::from(left),
            right = in(reg) u32::from(right),
            addend = in(reg) u32::from(addend),
            a = out(vreg) _,
            b = out(vreg) _,
            c = out(vreg) _,
            bits = lateout(reg) bits,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits")
}

/// Returns a binary64 value rounded to binary32, by `FCVT` in the rounding
/// direction of FPCR, which the path requires to be to nearest even.
#[inline]
pub fn narrow_double(value: f64) -> f32 {
    let result: f32;
    // SAFETY: FCVT reads and writes SIMD and floating-point registers. Every
    // AArch64 target has the instruction, and the rounding changes only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvt {result:s}, {value:d}",
            value = in(vreg) value,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Returns a binary64 value rounded to binary32 with round to odd, by
/// `FCVTXN`, whatever the rounding direction of FPCR.
///
/// Round to odd keeps at least two bits more than a format of at most 22
/// bits of precision, so a rounding of the result to such a format, to
/// nearest even, gives the binary64 value rounded once to that format.
/// binary32 also has the exponent range of bfloat16.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, which has no such instruction.
pub fn narrow_double_to_odd(value: f64) -> Option<f32> {
    let result: f32;
    // SAFETY: FCVTXN reads and writes SIMD and floating-point registers.
    // Every AArch64 target has the instruction, and the rounding changes only
    // the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvtxn {result:s}, {value:d}",
            value = in(vreg) value,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(result)
}

/// Returns a binary64 value rounded once to binary16, by `FCVT` in the rounding
/// direction of FPCR.
#[inline]
pub fn narrow_double_to_half(value: f64) -> u16 {
    let bits: u32;
    // SAFETY: FCVT rounds in a SIMD and floating-point register and clears its
    // other bits, and FMOV moves the low 32 bits to a general register. Every
    // AArch64 target has both instructions, and the rounding changes only the
    // status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvt {value:h}, {value:d}",
            "fmov {bits:w}, {value:s}",
            value = inout(vreg) value => _,
            bits = out(reg) bits,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits")
}

/// Runs one rounding to an integral value on `$value`, from the instruction
/// of the direction `$rounding`: `FRINTN` to nearest even, `FRINTA` to
/// nearest with ties away from zero, `FRINTM` toward negative infinity,
/// `FRINTP` toward positive infinity, and `FRINTZ` toward zero. Returns
/// `None` from the function for another direction.
macro_rules! round_scalar {
    ($rounding:expr, $register:literal, $value:ident) => {
        match $rounding {
            Rounding::TiesToEven => round_scalar!(concat!("frintn ", $register), $value),
            Rounding::TiesToAway => round_scalar!(concat!("frinta ", $register), $value),
            Rounding::TowardNegative => round_scalar!(concat!("frintm ", $register), $value),
            Rounding::TowardPositive => round_scalar!(concat!("frintp ", $register), $value),
            Rounding::TowardZero => round_scalar!(concat!("frintz ", $register), $value),
            Rounding::TiesTowardZero | Rounding::AwayFromZero | Rounding::ToOdd => return None,
        }
    };
    ($instruction:expr, $value:ident) => {
        // SAFETY: the instruction reads and writes one SIMD and floating-point
        // register. Every AArch64 target has it, and it changes only the
        // status flags of FPSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                value = inout(vreg) $value,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Returns a binary32 value rounded to an integral value in the direction
/// `rounding`, or `None` for a direction that no instruction has.
#[inline]
pub fn round_f32(mut value: f32, rounding: Rounding) -> Option<f32> {
    round_scalar!(rounding, "{value:s}, {value:s}", value);
    Some(value)
}

/// Returns a binary64 value rounded to an integral value in the direction
/// `rounding`, or `None` for a direction that no instruction has.
#[inline]
pub fn round_f64(mut value: f64, rounding: Rounding) -> Option<f64> {
    round_scalar!(rounding, "{value:d}, {value:d}", value);
    Some(value)
}

/// Returns a binary32 value rounded to a 64-bit integer to nearest even by
/// `FCVTNS`, or `None` for a value at a bound of the saturating conversion,
/// which a NaN, an infinity, and a value out of range can give.
#[inline]
pub fn to_int_f32(value: f32) -> Option<i64> {
    let result: i64;
    // SAFETY: FCVTNS reads a SIMD and floating-point register and writes a
    // general register. Every AArch64 target has the instruction, and it
    // changes only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvtns {result}, {value:s}",
            value = in(vreg) value,
            result = lateout(reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // FCVTNS gives zero for a NaN. The NaN test uses integer instructions, as
    // the paths do.
    let nan = bits::nan_32(value.to_bits());
    (result != i64::MIN && result != i64::MAX && !nan).then_some(result)
}

/// Returns a binary64 value rounded to a 64-bit integer to nearest even by
/// `FCVTNS`, or `None` for a value at a bound of the saturating conversion,
/// which a NaN, an infinity, and a value out of range can give.
#[inline]
pub fn to_int_f64(value: f64) -> Option<i64> {
    let result: i64;
    // SAFETY: FCVTNS reads a SIMD and floating-point register and writes a
    // general register. Every AArch64 target has the instruction, and it
    // changes only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "fcvtns {result}, {value:d}",
            value = in(vreg) value,
            result = lateout(reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    // FCVTNS gives zero for a NaN. The NaN test uses integer instructions, as
    // the paths do.
    let nan = bits::nan_64(value.to_bits());
    (result != i64::MIN && result != i64::MAX && !nan).then_some(result)
}

/// Returns a 64-bit integer rounded to binary32, by `SCVTF` in the rounding
/// direction of FPCR.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of 32-bit x86, which converts only an `i32`.
pub fn from_int_f32(value: i64) -> Option<f32> {
    let result: f32;
    // SAFETY: SCVTF reads a general register and writes a SIMD and
    // floating-point register. Every AArch64 target has the instruction, and
    // it changes only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "scvtf {result:s}, {value}",
            value = in(reg) value,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(result)
}

/// Returns a 64-bit integer rounded to binary64, by `SCVTF` in the rounding
/// direction of FPCR.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of 32-bit x86, which converts only an `i32`.
pub fn from_int_f64(value: i64) -> Option<f64> {
    let result: f64;
    // SAFETY: SCVTF reads a general register and writes a SIMD and
    // floating-point register. Every AArch64 target has the instruction, and
    // it changes only the status flags of FPSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "scvtf {result:d}, {value}",
            value = in(reg) value,
            result = lateout(vreg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(result)
}

/// Returns `false`: AArch64 has no x87 unit, so no x87 extended path applies.
#[inline]
pub fn x87_environment() -> bool {
    false
}

/// Returns `false`: AArch64 has no x87 unit.
#[inline]
pub fn x87_full_precision() -> bool {
    false
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_binary(_left: &[u64; 2], _right: &[u64; 2], _operation: Operation) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_sqrt(_value: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_round(_value: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_from_int(_value: i64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_from_single(_bits: u32) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_from_double(_bits: u64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_remainder(_dividend: &[u64; 2], _divisor: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no x87 unit and no remainder instruction.
#[inline]
pub fn x87_remainder_single(_dividend: u32, _divisor: u32) -> Option<u32> {
    None
}

/// Returns `None`: AArch64 has no x87 unit and no remainder instruction.
#[inline]
pub fn x87_remainder_double(_dividend: u64, _divisor: u64) -> Option<u64> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_to_int(_value: &[u64; 2]) -> Option<i64> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_to_single(_value: &[u64; 2]) -> Option<u32> {
    None
}

/// Returns `None`: AArch64 has no x87 unit.
#[inline]
pub fn x87_to_double(_value: &[u64; 2]) -> Option<u64> {
    None
}

/// `false`: AArch64 has no binary128 unit.
pub const QUAD: bool = false;

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_binary(_left: &[u64; 2], _right: &[u64; 2], _operation: Operation) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_sqrt(_value: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_round(_value: &[u64; 2], _rounding: Rounding) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_to_int(_value: &[u64; 2]) -> Option<i64> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_from_int(_value: i64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_compare(_left: &[u64; 2], _right: &[u64; 2]) -> Option<Ordering> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_from_single(_bits: u32) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_from_double(_bits: u64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_to_single(_value: &[u64; 2]) -> Option<u32> {
    None
}

/// Returns `None`: AArch64 has no binary128 unit.
#[inline]
pub fn quad_to_double(_value: &[u64; 2]) -> Option<u64> {
    None
}

/// Runs one quiet comparison of `$left` with `$right`, `FCMP`, and returns
/// the order, or `None` for an unordered pair.
macro_rules! compare {
    ($instruction:literal, $left:ident, $right:ident) => {{
        let (unordered, greater, less): (u32, u32, u32);
        // SAFETY: FCMP reads two SIMD and floating-point registers and writes
        // the condition flags, which CSET copies into three general registers.
        // Every AArch64 target has them, and FCMP changes only the status
        // flags of FPSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                "cset {unordered:w}, vs",
                "cset {greater:w}, gt",
                "cset {less:w}, mi",
                left = in(vreg) $left,
                right = in(vreg) $right,
                unordered = out(reg) unordered,
                greater = out(reg) greater,
                less = out(reg) less,
                options(pure, nomem, nostack),
            );
        }
        if unordered != 0 {
            None
        } else if greater != 0 {
            Some(Ordering::Greater)
        } else if less != 0 {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Equal)
        }
    }};
}

/// Returns the order of two binary32 values by `FCMP`, which signals
/// invalid only for a signaling NaN, or `None` when they are unordered.
#[inline]
pub fn compare_f32(left: f32, right: f32) -> Option<Ordering> {
    compare!("fcmp {left:s}, {right:s}", left, right)
}

/// Returns the order of two binary64 values by `FCMP`, which signals
/// invalid only for a signaling NaN, or `None` when they are unordered.
#[inline]
pub fn compare_f64(left: f64, right: f64) -> Option<Ordering> {
    compare!("fcmp {left:d}, {right:d}", left, right)
}

/// Returns the smaller of two binary32 values by `FMIN`.
#[inline]
pub fn min_f32(mut left: f32, right: f32) -> f32 {
    scalar!("fmin {left:s}, {left:s}, {right:s}", left, right);
    left
}

/// Returns the larger of two binary32 values by `FMAX`.
#[inline]
pub fn max_f32(mut left: f32, right: f32) -> f32 {
    scalar!("fmax {left:s}, {left:s}, {right:s}", left, right);
    left
}

/// Returns the smaller of two binary64 values by `FMIN`.
#[inline]
pub fn min_f64(mut left: f64, right: f64) -> f64 {
    scalar!("fmin {left:d}, {left:d}, {right:d}", left, right);
    left
}

/// Returns the larger of two binary64 values by `FMAX`.
#[inline]
pub fn max_f64(mut left: f64, right: f64) -> f64 {
    scalar!("fmax {left:d}, {left:d}, {right:d}", left, right);
    left
}
