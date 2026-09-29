//! The floating-point environment of AArch64: FPCR.

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
