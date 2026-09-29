//! The floating-point environment of x86-64, MXCSR of the SSE unit and the
//! control word of the x87 unit, and the instructions of the host paths.

use super::Operation;

pub mod packed;

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

/// Selects the template of an SSE instruction: the VEX form `$vex` in a build
/// with AVX, and the legacy form `$legacy` otherwise.
///
/// A legacy SSE instruction keeps the upper half of its 256-bit register.
/// After a 256-bit instruction, the legacy instruction waits on that half, or
/// the unit saves the upper halves of all registers. A VEX form clears the
/// upper half and does not wait on it.
#[cfg(target_feature = "avx")]
macro_rules! sse {
    ($legacy:literal, $vex:literal) => {
        $vex
    };
}

/// Selects the legacy form `$legacy` of an SSE instruction in a build
/// without AVX, which has no VEX forms.
#[cfg(not(target_feature = "avx"))]
macro_rules! sse {
    ($legacy:literal, $vex:literal) => {
        $legacy
    };
}

pub(super) use sse;

/// Runs one scalar SSE instruction on `$left` and `$right`, and leaves the
/// result in `$left`.
macro_rules! sse_scalar {
    ($instruction:expr, $left:ident, $right:ident) => {
        // SAFETY: the instruction reads and writes SSE registers. SSE2 is part
        // of every x86-64 target, and `sse!` selects a VEX form only in a
        // build with AVX. The instruction changes only the status flags of
        // MXCSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                left = inout(xmm_reg) $left,
                right = in(xmm_reg) $right,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Returns `operation` of two binary32 values, by `ADDSS`, `SUBSS`, `MULSS`,
/// or `DIVSS` in the rounding direction of MXCSR.
#[inline]
pub fn binary_f32(mut left: f32, right: f32, operation: Operation) -> f32 {
    match operation {
        Operation::Add => sse_scalar!(
            sse!("addss {left}, {right}", "vaddss {left}, {left}, {right}"),
            left,
            right
        ),
        Operation::Sub => sse_scalar!(
            sse!("subss {left}, {right}", "vsubss {left}, {left}, {right}"),
            left,
            right
        ),
        Operation::Mul => sse_scalar!(
            sse!("mulss {left}, {right}", "vmulss {left}, {left}, {right}"),
            left,
            right
        ),
        Operation::Div => sse_scalar!(
            sse!("divss {left}, {right}", "vdivss {left}, {left}, {right}"),
            left,
            right
        ),
    }
    left
}

/// Returns `operation` of two binary64 values, by `ADDSD`, `SUBSD`, `MULSD`,
/// or `DIVSD` in the rounding direction of MXCSR.
#[inline]
pub fn binary_f64(mut left: f64, right: f64, operation: Operation) -> f64 {
    match operation {
        Operation::Add => sse_scalar!(
            sse!("addsd {left}, {right}", "vaddsd {left}, {left}, {right}"),
            left,
            right
        ),
        Operation::Sub => sse_scalar!(
            sse!("subsd {left}, {right}", "vsubsd {left}, {left}, {right}"),
            left,
            right
        ),
        Operation::Mul => sse_scalar!(
            sse!("mulsd {left}, {right}", "vmulsd {left}, {left}, {right}"),
            left,
            right
        ),
        Operation::Div => sse_scalar!(
            sse!("divsd {left}, {right}", "vdivsd {left}, {left}, {right}"),
            left,
            right
        ),
    }
    left
}

/// Returns a binary32 value widened exactly to binary64, by `CVTSS2SD`.
#[inline]
pub fn widen_single(value: f32) -> f64 {
    let result: f64;
    // SAFETY: CVTSS2SD reads and writes SSE registers. SSE2 is part of every
    // x86-64 target, and `sse!` selects a VEX form only in a build with AVX.
    // The conversion changes only the status flags of MXCSR, which floaty
    // does not read.
    unsafe {
        core::arch::asm!(
            sse!("cvtss2sd {result}, {value}", "vcvtss2sd {result}, {value}, {value}"),
            value = in(xmm_reg) value,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Returns the square root of a binary32 value, by `SQRTSS`.
#[inline]
pub fn sqrt_f32(mut value: f32) -> f32 {
    // SAFETY: SQRTSS reads and writes one SSE register. SSE2 is part of every
    // x86-64 target, and `sse!` selects a VEX form only in a build with AVX.
    // The instruction changes only the status flags of MXCSR, which floaty
    // does not read.
    unsafe {
        core::arch::asm!(
            sse!("sqrtss {value}, {value}", "vsqrtss {value}, {value}, {value}"),
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
    // x86-64 target, and `sse!` selects a VEX form only in a build with AVX.
    // The instruction changes only the status flags of MXCSR, which floaty
    // does not read.
    unsafe {
        core::arch::asm!(
            sse!("sqrtsd {value}, {value}", "vsqrtsd {value}, {value}, {value}"),
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

/// Returns `None`: the bfloat16 conversions of x86-64 flush subnormal
/// values, so no bfloat16 path exists.
#[inline]
pub fn narrow_bfloat(_value: f32) -> Option<u16> {
    None
}

/// Returns `None`: x86-64 has no binary16 fused multiply-add below
/// AVX512-FP16, which no host of the gates has.
#[inline]
pub fn mul_add_f16(_left: u16, _right: u16, _addend: u16) -> Option<u16> {
    None
}

/// Returns a binary64 value rounded to binary32, by `CVTSD2SS` in the rounding
/// direction of MXCSR, which the path requires to be to nearest even.
#[inline]
pub fn narrow_double(value: f64) -> f32 {
    let result: f32;
    // SAFETY: CVTSD2SS reads and writes SSE registers. SSE2 is part of every
    // x86-64 target, and `sse!` selects a VEX form only in a build with AVX.
    // The rounding changes only the status flags of MXCSR, which floaty does
    // not read.
    unsafe {
        core::arch::asm!(
            sse!("cvtsd2ss {result}, {value}", "vcvtsd2ss {result}, {value}, {value}"),
            value = in(xmm_reg) value,
            result = lateout(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Returns `None`: x86-64 has no instruction that rounds binary64 to binary16
/// once.
#[inline]
pub fn narrow_double_to_half(_value: f64) -> Option<u16> {
    None
}

/// Returns a binary32 value rounded to an integral value to nearest even, by
/// `ROUNDSS` with the rounding control 0 in its immediate.
#[cfg(target_feature = "sse4.1")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without SSE4.1 returns `None` from the same signature.
pub fn round_f32(mut value: f32) -> Option<f32> {
    // SAFETY: ROUNDSS reads and writes one SSE register. The build enables
    // SSE4.1, and `sse!` selects a VEX form only in a build with AVX. The
    // instruction changes only the status flags of MXCSR, which floaty does
    // not read.
    unsafe {
        core::arch::asm!(
            sse!("roundss {value}, {value}, 0", "vroundss {value}, {value}, {value}, 0"),
            value = inout(xmm_reg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(value)
}

/// Returns a binary64 value rounded to an integral value to nearest even, by
/// `ROUNDSD` with the rounding control 0 in its immediate.
#[cfg(target_feature = "sse4.1")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // A build without SSE4.1 returns `None` from the same signature.
pub fn round_f64(mut value: f64) -> Option<f64> {
    // SAFETY: ROUNDSD reads and writes one SSE register. The build enables
    // SSE4.1, and `sse!` selects a VEX form only in a build with AVX. The
    // instruction changes only the status flags of MXCSR, which floaty does
    // not read.
    unsafe {
        core::arch::asm!(
            sse!("roundsd {value}, {value}, 0", "vroundsd {value}, {value}, {value}, 0"),
            value = inout(xmm_reg) value,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    Some(value)
}

/// Returns `None`: a build without SSE4.1 has no path that rounds to an
/// integral value.
#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn round_f32(_value: f32) -> Option<f32> {
    None
}

/// Returns `None`: a build without SSE4.1 has no path that rounds to an
/// integral value.
#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn round_f64(_value: f64) -> Option<f64> {
    None
}

/// Returns a binary32 value rounded to a 64-bit integer by `CVTSS2SI` in the
/// rounding direction of MXCSR, or `None` for the integer indefinite, which a
/// NaN, an infinity, a value out of range, and `-2^63` give.
#[inline]
pub fn to_int_f32(value: f32) -> Option<i64> {
    let result: i64;
    // SAFETY: CVTSS2SI reads an SSE register and writes a general register.
    // SSE2 is part of every x86-64 target, and `sse!` selects a VEX form only
    // in a build with AVX. The conversion changes only the status flags of
    // MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            sse!("cvtss2si {result}, {value}", "vcvtss2si {result}, {value}"),
            value = in(xmm_reg) value,
            result = lateout(reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (result != i64::MIN).then_some(result)
}

/// Returns a binary64 value rounded to a 64-bit integer by `CVTSD2SI` in the
/// rounding direction of MXCSR, or `None` for the integer indefinite, which a
/// NaN, an infinity, a value out of range, and `-2^63` give.
#[inline]
pub fn to_int_f64(value: f64) -> Option<i64> {
    let result: i64;
    // SAFETY: CVTSD2SI reads an SSE register and writes a general register.
    // SSE2 is part of every x86-64 target, and `sse!` selects a VEX form only
    // in a build with AVX. The conversion changes only the status flags of
    // MXCSR, which floaty does not read.
    unsafe {
        core::arch::asm!(
            sse!("cvtsd2si {result}, {value}", "vcvtsd2si {result}, {value}"),
            value = in(xmm_reg) value,
            result = lateout(reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    (result != i64::MIN).then_some(result)
}

/// Returns a 64-bit integer rounded to binary32, by `CVTSI2SS` in the rounding
/// direction of MXCSR.
#[inline]
pub fn from_int_f32(value: i64) -> f32 {
    let result: f32;
    // SAFETY: XORPS clears an SSE register, so CVTSI2SS does not wait on its
    // old value. CVTSI2SS writes the converted value into the low lane. SSE2 is
    // part of every x86-64 target, and `sse!` selects a VEX form only in a
    // build with AVX. The conversion changes only the status flags of MXCSR,
    // which floaty does not read.
    unsafe {
        core::arch::asm!(
            sse!("xorps {result}, {result}", "vxorps {result}, {result}, {result}"),
            sse!("cvtsi2ss {result}, {value}", "vcvtsi2ss {result}, {result}, {value}"),
            value = in(reg) value,
            result = out(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Returns a 64-bit integer rounded to binary64, by `CVTSI2SD` in the rounding
/// direction of MXCSR.
#[inline]
pub fn from_int_f64(value: i64) -> f64 {
    let result: f64;
    // SAFETY: XORPD clears an SSE register, so CVTSI2SD does not wait on its
    // old value. CVTSI2SD writes the converted value into the low lane. SSE2 is
    // part of every x86-64 target, and `sse!` selects a VEX form only in a
    // build with AVX. The conversion changes only the status flags of MXCSR,
    // which floaty does not read.
    unsafe {
        core::arch::asm!(
            sse!("xorpd {result}, {result}", "vxorpd {result}, {result}, {result}"),
            sse!("cvtsi2sd {result}, {value}", "vcvtsi2sd {result}, {result}, {value}"),
            value = in(reg) value,
            result = out(xmm_reg) result,
            options(pure, nomem, nostack, preserves_flags),
        );
    }
    result
}

/// Returns `true` when the x87 unit rounds to nearest even at the 64-bit
/// precision, and masks every exception.
///
/// Linux starts a process with that control word, 037FH, but other systems
/// and libraries can select a 53-bit precision. An unmasked exception traps,
/// and the engine never traps. The fields are in the Intel SDM Volume 1,
/// revision 253665-093US, section 8.1.5, Figure 8-6.
#[inline]
pub fn x87_environment() -> bool {
    // The exception masks IM, DM, ZM, OM, UM, and PM are one, the precision
    // control is 11B for 64 bits, and the rounding control is zero.
    const MASKS: u16 = 0x3F;
    const PRECISION: u16 = 3 << 8;
    const ROUNDING: u16 = 3 << 10;
    let control: u16;
    // SAFETY: the block stores the x87 control word in eight bytes that it
    // takes below the stack pointer, loads the value into a register, and
    // restores the stack pointer. It changes no other state.
    unsafe {
        core::arch::asm!(
            "sub rsp, 8",
            "fnstcw word ptr [rsp]",
            "movzx {control:e}, word ptr [rsp]",
            "add rsp, 8",
            control = out(reg) control,
            options(preserves_flags),
        );
    }
    control & (MASKS | PRECISION | ROUNDING) == MASKS | PRECISION
}

/// Returns an 80-bit result of the x87 unit, or `None` for a NaN. The unit
/// gives only canonical encodings, so a NaN is an encoding with the largest
/// exponent field and a significand other than the integer bit alone.
#[inline]
fn extended_result(bits: [u64; 2]) -> Option<[u64; 2]> {
    let nan = bits[1] & 0x7FFF == 0x7FFF && bits[0] != 1 << 63;
    (!nan).then_some(bits)
}

/// Runs x87 instructions that leave one value on the x87 stack, and stores
/// that value as an 80-bit encoding. Returns the encoding, or `None` for a
/// NaN. Each named operand is a pointer that the instructions read.
///
/// The block reads the stored encoding back as 8 and 2 bytes, the parts that
/// `FSTP` writes. An 8-byte load of the last 2 bytes waited on the store and
/// tripled the time of an operation.
macro_rules! x87_extended {
    ($($instruction:literal),+; $($name:ident = $pointer:expr),+) => {{
        let mut scratch = [0_u64; 2];
        let (low, high): (u64, u64);
        // SAFETY: the instructions read their operands through the pointers.
        // `FSTP` writes 10 bytes to `scratch`, which holds 16, and the block
        // reads them back. Every x87 register is a clobber, so the x87 stack
        // is empty on entry, and the `FSTP` leaves it empty on exit. The x87
        // unit changes only its status word, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $($instruction,)+
                "fstp tbyte ptr [{scratch}]",
                "mov {low}, qword ptr [{scratch}]",
                "movzx {high:e}, word ptr [{scratch} + 8]",
                $($name = in(reg) $pointer,)+
                scratch = in(reg) scratch.as_mut_ptr(),
                // `low` needs its own register, because the next load reads
                // `scratch`.
                low = out(reg) low,
                high = lateout(reg) high,
                out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                options(nostack, preserves_flags),
            );
        }
        extended_result([low, high])
    }};
}

/// Returns the result of `operation` on two 80-bit encodings from the x87
/// unit, by `FADDP`, `FSUBP`, `FMULP`, or `FDIVP`, or `None` for a NaN.
#[inline]
pub fn x87_binary(left: &[u64; 2], right: &[u64; 2], operation: Operation) -> Option<[u64; 2]> {
    let (left, right) = (left.as_ptr(), right.as_ptr());
    match operation {
        Operation::Add => x87_extended!(
            "fld tbyte ptr [{left}]", "fld tbyte ptr [{right}]", "faddp st(1), st";
            left = left, right = right
        ),
        Operation::Sub => x87_extended!(
            "fld tbyte ptr [{left}]", "fld tbyte ptr [{right}]", "fsubp st(1), st";
            left = left, right = right
        ),
        Operation::Mul => x87_extended!(
            "fld tbyte ptr [{left}]", "fld tbyte ptr [{right}]", "fmulp st(1), st";
            left = left, right = right
        ),
        Operation::Div => x87_extended!(
            "fld tbyte ptr [{left}]", "fld tbyte ptr [{right}]", "fdivp st(1), st";
            left = left, right = right
        ),
    }
}

/// Returns the square root of an 80-bit encoding from the x87 unit, by
/// `FSQRT`, or `None` for a NaN.
#[inline]
pub fn x87_sqrt(value: &[u64; 2]) -> Option<[u64; 2]> {
    x87_extended!("fld tbyte ptr [{value}]", "fsqrt"; value = value.as_ptr())
}

/// Returns an 80-bit encoding rounded to an integral value in the rounding
/// direction of the control word, by `FRNDINT`, or `None` for a NaN.
#[inline]
pub fn x87_round(value: &[u64; 2]) -> Option<[u64; 2]> {
    x87_extended!("fld tbyte ptr [{value}]", "frndint"; value = value.as_ptr())
}

/// Returns a 64-bit integer converted exactly to an 80-bit encoding, by
/// `FILD`.
#[inline]
pub fn x87_from_int(value: i64) -> Option<[u64; 2]> {
    x87_extended!("fild qword ptr [{value}]"; value = &raw const value)
}

/// Returns a binary32 encoding widened exactly to an 80-bit encoding, by
/// `FLD`, or `None` for a NaN.
#[inline]
pub fn x87_from_single(bits: u32) -> Option<[u64; 2]> {
    x87_extended!("fld dword ptr [{value}]"; value = &raw const bits)
}

/// Returns a binary64 encoding widened exactly to an 80-bit encoding, by
/// `FLD`, or `None` for a NaN.
#[inline]
pub fn x87_from_double(bits: u64) -> Option<[u64; 2]> {
    x87_extended!("fld qword ptr [{value}]"; value = &raw const bits)
}

/// Runs x87 instructions that load an 80-bit encoding through `value` and
/// store and pop one result of type `$result` through `result`.
macro_rules! x87_store {
    ($result:ty, $($instruction:literal),+; $value:expr) => {{
        let mut result: $result = 0;
        // SAFETY: the instructions read 10 bytes through `value` and write one
        // result of the store width to `result`. Every x87 register is a
        // clobber, so the x87 stack is empty on entry, and the store pops the
        // one value, so it is empty on exit. The x87 unit changes only its
        // status word, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $($instruction,)+
                value = in(reg) $value,
                result = in(reg) &raw mut result,
                out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                options(nostack, preserves_flags),
            );
        }
        result
    }};
}

/// Returns an 80-bit encoding rounded to a 64-bit integer in the rounding
/// direction of the control word, by `FISTP`, or `None` for the integer
/// indefinite, which a NaN, an infinity, a value out of range, and `-2^63`
/// give.
#[inline]
pub fn x87_to_int(value: &[u64; 2]) -> Option<i64> {
    let result = x87_store!(
        i64, "fld tbyte ptr [{value}]", "fistp qword ptr [{result}]"; value.as_ptr()
    );
    (result != i64::MIN).then_some(result)
}

/// Returns an 80-bit encoding rounded to binary32 in the rounding direction
/// of the control word, by `FSTP`, or `None` for a NaN.
#[inline]
pub fn x87_to_single(value: &[u64; 2]) -> Option<u32> {
    let result = x87_store!(
        u32, "fld tbyte ptr [{value}]", "fstp dword ptr [{result}]"; value.as_ptr()
    );
    // The NaN test uses integer instructions, as the paths do.
    let nan = result & 0x7FFF_FFFF > 0x7F80_0000;
    (!nan).then_some(result)
}

/// Returns an 80-bit encoding rounded to binary64 in the rounding direction
/// of the control word, by `FSTP`, or `None` for a NaN.
#[inline]
pub fn x87_to_double(value: &[u64; 2]) -> Option<u64> {
    let result = x87_store!(
        u64, "fld tbyte ptr [{value}]", "fstp qword ptr [{result}]"; value.as_ptr()
    );
    // The NaN test uses integer instructions, as the paths do.
    let nan = result & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000;
    (!nan).then_some(result)
}
