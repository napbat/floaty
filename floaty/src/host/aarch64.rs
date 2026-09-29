//! The floating-point environment of AArch64, FPCR, and the instructions of
//! the host paths.

use super::Operation;
use crate::env::Rounding;

pub mod packed;

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
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack F16C.
pub fn widen_half(bits: u16) -> Option<f32> {
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
    Some(value)
}

/// Returns a binary32 value rounded to binary16 in the rounding direction of
/// FPCR, which the path requires to be to nearest even, by `FCVT`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack F16C.
pub fn narrow_half(value: f32) -> Option<u16> {
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
    Some(u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits"))
}

/// Returns a binary32 value rounded to bfloat16 in the rounding direction of
/// FPCR, which the path requires to be to nearest even, by `BFCVT`. The Arm
/// Architecture Reference Manual states that `BFCVT` honors every control of
/// FPCR that applies to single-precision arithmetic.
#[cfg(target_feature = "bf16")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without `FEAT_BF16` returns `None` from the same signature.
pub fn narrow_bfloat(value: f32) -> Option<u16> {
    // SAFETY: the build enables `FEAT_BF16`, so the processor that runs it
    // has the feature.
    Some(unsafe { bfcvt(value) })
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

/// Returns `None`: a build without `FEAT_BF16` has no bfloat16 path.
#[cfg(not(target_feature = "bf16"))]
#[inline]
pub fn narrow_bfloat(_value: f32) -> Option<u16> {
    None
}

/// Returns `left * right + addend` of binary16 encodings, rounded once, by
/// `FMADD` on half-precision registers.
#[cfg(target_feature = "fp16")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without `FEAT_FP16` returns `None` from the same signature.
pub fn mul_add_f16(left: u16, right: u16, addend: u16) -> Option<u16> {
    // SAFETY: the build enables `FEAT_FP16`, so the processor that runs it
    // has the feature.
    Some(unsafe { fmadd_f16(left, right, addend) })
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

/// Returns `None`: a build without `FEAT_FP16` has no binary16 fused
/// multiply-add path.
#[cfg(not(target_feature = "fp16"))]
#[inline]
pub fn mul_add_f16(_left: u16, _right: u16, _addend: u16) -> Option<u16> {
    None
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

/// Returns a binary64 value rounded once to binary16, by `FCVT` in the rounding
/// direction of FPCR.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, which has no such instruction.
pub fn narrow_double_to_half(value: f64) -> Option<u16> {
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
    Some(u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits"))
}

/// Runs one rounding to an integral value on `$value`, from the instruction
/// of the direction `$rounding`: `FRINTN` to nearest even, `FRINTA` to
/// nearest with ties away from zero, `FRINTM` toward negative infinity,
/// `FRINTP` toward positive infinity, and `FRINTZ` toward zero. Returns
/// `None` from the function for another direction.
macro_rules! round_scalar {
    ($rounding:expr, $register:literal, $value:ident) => {
        match $rounding {
            Rounding::NearestEven => round_scalar!(concat!("frintn ", $register), $value),
            Rounding::NearestAway => round_scalar!(concat!("frinta ", $register), $value),
            Rounding::TowardNegative => round_scalar!(concat!("frintm ", $register), $value),
            Rounding::TowardPositive => round_scalar!(concat!("frintp ", $register), $value),
            Rounding::TowardZero => round_scalar!(concat!("frintz ", $register), $value),
            Rounding::ToOdd => return None,
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
    let nan = value.to_bits() & 0x7FFF_FFFF > 0x7F80_0000;
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
    let nan = value.to_bits() & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000;
    (result != i64::MIN && result != i64::MAX && !nan).then_some(result)
}

/// Returns a 64-bit integer rounded to binary32, by `SCVTF` in the rounding
/// direction of FPCR.
#[inline]
pub fn from_int_f32(value: i64) -> f32 {
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
    result
}

/// Returns a 64-bit integer rounded to binary64, by `SCVTF` in the rounding
/// direction of FPCR.
#[inline]
pub fn from_int_f64(value: i64) -> f64 {
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
    result
}

/// Returns `false`: AArch64 has no x87 unit, so no x87 extended path applies.
#[inline]
pub fn x87_environment() -> bool {
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
