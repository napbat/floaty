//! The floating-point environment of x86-64: MXCSR of the SSE unit.

/// Returns `true` when the SSE unit rounds to nearest even without FTZ or DAZ,
/// and masks every exception.
///
/// Rust code runs with that MXCSR value, but an emulator can load another
/// value, and a library built with `-ffast-math` can set FTZ and DAZ when it
/// loads. An unmasked exception traps, and the engine never traps. The path
/// then goes back to the engine. The fields are in the Intel SDM Volume 1,
/// revision 253665-093US, section 10.2.3, Figure 10-3.
#[inline]
pub fn default_environment() -> bool {
    // The rounding control, FTZ, and DAZ are zero in the default
    // environment. The exception masks IM, DM, ZM, OM, UM, and PM are one.
    const RESULT_FIELDS: u32 = (3 << 13) | (1 << 15) | (1 << 6);
    const MASKS: u32 = 0x3F << 7;
    let mxcsr: u32;
    // SAFETY: the block stores MXCSR in eight bytes that it takes below the
    // stack pointer, loads the value into a register, and restores the stack
    // pointer. It changes no other state. Its own stack slot keeps the
    // four-byte store apart from the stack frame of the caller.
    unsafe {
        core::arch::asm!(
            "sub rsp, 8",
            "stmxcsr [rsp]",
            "mov {mxcsr:e}, dword ptr [rsp]",
            "add rsp, 8",
            mxcsr = out(reg) mxcsr,
            options(preserves_flags),
        );
    }
    mxcsr & (RESULT_FIELDS | MASKS) == MASKS
}

/// Returns the square root of a binary32 value, by `SQRTSS`.
#[inline]
pub fn sqrt_f32(mut value: f32) -> f32 {
    // SAFETY: SQRTSS reads and writes one SSE register. SSE2 is part of every
    // x86-64 target, and the instruction changes only the status flags of
    // MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "sqrtss {value}, {value}",
            value = inout(xmm_reg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    value
}

/// Returns the square root of a binary64 value, by `SQRTSD`.
#[inline]
pub fn sqrt_f64(mut value: f64) -> f64 {
    // SAFETY: SQRTSD reads and writes one SSE register. SSE2 is part of every
    // x86-64 target, and the instruction changes only the status flags of
    // MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "sqrtsd {value}, {value}",
            value = inout(xmm_reg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    value
}

/// Returns `left * right + addend`, rounded once, by `VFMADD213SS`.
#[cfg(target_feature = "fma")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without FMA returns `None` from the same signature.
pub fn mul_add_f32(mut left: f32, right: f32, addend: f32) -> Option<f32> {
    // SAFETY: VFMADD213SS computes `right * left + addend` in SSE registers.
    // The build enables FMA, and the instruction changes only the status
    // flags of MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vfmadd213ss {left}, {right}, {addend}",
            left = inout(xmm_reg) left,
            right = in(xmm_reg) right,
            addend = in(xmm_reg) addend,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(left)
}

/// Returns `left * right + addend`, rounded once, by `VFMADD213SD`.
#[cfg(target_feature = "fma")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without FMA returns `None` from the same signature.
pub fn mul_add_f64(mut left: f64, right: f64, addend: f64) -> Option<f64> {
    // SAFETY: VFMADD213SD computes `right * left + addend` in SSE registers.
    // The build enables FMA, and the instruction changes only the status
    // flags of MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vfmadd213sd {left}, {right}, {addend}",
            left = inout(xmm_reg) left,
            right = in(xmm_reg) right,
            addend = in(xmm_reg) addend,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(left)
}

/// Returns `None`: a build without FMA has no fused multiply-add path.
#[cfg(not(target_feature = "fma"))]
#[inline]
pub fn mul_add_f32(_left: f32, _right: f32, _addend: f32) -> Option<f32> {
    None
}

/// Returns `None`: a build without FMA has no fused multiply-add path.
#[cfg(not(target_feature = "fma"))]
#[inline]
pub fn mul_add_f64(_left: f64, _right: f64, _addend: f64) -> Option<f64> {
    None
}

/// Returns a binary16 value widened exactly to binary32, by `VCVTPH2PS`.
#[cfg(target_feature = "f16c")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without F16C returns `None` from the same signature.
pub fn widen_half(bits: u16) -> Option<f32> {
    let value: f32;
    // SAFETY: VMOVD and VCVTPH2PS read a general register and write one SSE
    // register. The build enables F16C, and the widening is exact, so it
    // changes no state.
    unsafe {
        core::arch::asm!(
            "vmovd {value}, {bits:e}",
            "vcvtph2ps {value}, {value}",
            bits = in(reg) u32::from(bits),
            value = out(xmm_reg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(value)
}

/// Returns a binary32 value rounded to binary16 to nearest even, by
/// `VCVTPS2PH` with the rounding control 0 in its immediate.
#[cfg(target_feature = "f16c")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without F16C returns `None` from the same signature.
pub fn narrow_half(value: f32) -> Option<u16> {
    let bits: u32;
    // SAFETY: VCVTPS2PH and VMOVD read and write one SSE register and write a
    // general register. The build enables F16C, and the rounding changes only
    // the status flags of MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            "vcvtps2ph {value}, {value}, 0",
            "vmovd {bits:e}, {value}",
            value = inout(xmm_reg) value => _,
            bits = out(reg) bits,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits"))
}

/// Returns `None`: a build without F16C has no binary16 path.
#[cfg(not(target_feature = "f16c"))]
#[inline]
pub fn widen_half(_bits: u16) -> Option<f32> {
    None
}

/// Returns `None`: a build without F16C has no binary16 path.
#[cfg(not(target_feature = "f16c"))]
#[inline]
pub fn narrow_half(_value: f32) -> Option<u16> {
    None
}
