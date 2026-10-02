//! The floating-point environment of x86 and x86-64, MXCSR of the SSE unit,
//! and the scalar SSE instructions of the host paths. `x87` holds the x87
//! unit. The module builds for 32-bit x86 with SSE2 too, where no general
//! register holds 64 bits: the conversions between floats and 64-bit
//! integers there take only an integer in the range of `i32`.

use core::cmp::Ordering;

use super::Operation;
use crate::env::Rounding;

/// Names the stack pointer of the target: `rsp` on x86-64, and `esp` on
/// 32-bit x86.
#[cfg(target_arch = "x86_64")]
macro_rules! stack_pointer {
    () => {
        "rsp"
    };
}

/// Names the stack pointer of the target, as on x86-64.
#[cfg(target_arch = "x86")]
macro_rules! stack_pointer {
    () => {
        "esp"
    };
}

pub mod packed;
mod x87;

pub use self::x87::{
    x87_binary, x87_environment, x87_from_double, x87_from_int, x87_from_single, x87_remainder,
    x87_remainder_double, x87_remainder_single, x87_round, x87_sqrt, x87_to_double, x87_to_int,
    x87_to_single,
};

/// `true`: the build has the SSE unit.
pub const UNIT: bool = true;
/// `true` when the build has FMA, the fused multiply-add of binary32 and
/// binary64.
pub const FUSED: bool = cfg!(target_feature = "fma");
/// `true` when the build has F16C, which widens binary16 to binary32 and
/// rounds binary32 to binary16.
pub const HALF: bool = cfg!(target_feature = "f16c");
/// `true` when the build has SSE4.1, which rounds to an integral value.
pub const ROUNDING: bool = cfg!(target_feature = "sse4.1");
/// `true`: every x86 processor has the x87 unit.
pub const X87: bool = true;
/// `true`: every x86 target with SSE2 computes the binary16 fused multiply-add
/// through binary64, with integer instructions for the last rounding.
pub const HALF_FUSED: bool = true;
/// `true`: integer instructions round binary64 to binary16 once.
pub const DOUBLE_TO_HALF: bool = true;
/// `false`: x86 has no instruction that rounds to odd.
pub const ROUND_TO_ODD: bool = false;

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
            concat!("sub ", stack_pointer!(), ", 8"),
            concat!("stmxcsr [", stack_pointer!(), "]"),
            concat!("mov {mxcsr:e}, dword ptr [", stack_pointer!(), "]"),
            concat!("add ", stack_pointer!(), ", 8"),
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
        // SAFETY: the instruction reads and writes SSE registers. The build enables SSE2, and `sse!` selects a VEX form only in a
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
    // SAFETY: CVTSS2SD reads and writes SSE registers. The build enables SSE2, and `sse!` selects a VEX form only in a build with AVX.
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
    // SAFETY: SQRTSS reads and writes one SSE register. The build enables SSE2, and `sse!` selects a VEX form only in a build with AVX.
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
    // SAFETY: SQRTSD reads and writes one SSE register. The build enables SSE2, and `sse!` selects a VEX form only in a build with AVX.
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
pub fn widen_half(bits: u16) -> f32 {
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
    value
}

/// Returns a binary32 value rounded to binary16 to nearest even, by
/// `VCVTPS2PH` with the rounding control 0 in its immediate.
#[cfg(target_feature = "f16c")]
#[inline]
pub fn narrow_half(value: f32) -> u16 {
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
    u16::try_from(bits & 0xFFFF).expect("the mask keeps 16 bits")
}

/// Returns a binary16 value widened exactly to binary32, by the integer
/// widening of the host paths, in a build without F16C.
#[cfg(not(target_feature = "f16c"))]
#[inline]
pub fn widen_half(bits: u16) -> f32 {
    f32::from_bits(super::narrow::widen_half_bits(bits))
}

/// Returns a binary32 value rounded to binary16 to nearest even, by the
/// integer rounding of the host paths, in a build without F16C.
#[cfg(not(target_feature = "f16c"))]
#[inline]
pub fn narrow_half(value: f32) -> u16 {
    super::narrow::round_to_half(value.to_bits())
}

/// Returns a binary32 value rounded to bfloat16 to nearest even, by the
/// integer rounding of the host paths. The bfloat16 conversions of x86
/// read a subnormal input as zero.
#[inline]
pub fn narrow_bfloat(value: f32) -> u16 {
    super::narrow::round_to_bfloat(value.to_bits())
}

/// Returns `left * right + addend` of binary16 encodings, rounded once. The
/// operands widen to binary64 exactly, `MULSD` gives the exact product,
/// `ADDSD` rounds the sum to binary64, and integer instructions round that
/// sum to binary16. [`Host::Half`](super::Host::Half) states why the two
/// roundings give the result of one.
#[inline]
pub fn mul_add_f16(left: u16, right: u16, addend: u16) -> u16 {
    let [a, b, c] = [left, right, addend].map(|bits| widen_single(widen_half(bits)));
    let product = binary_f64(a, b, Operation::Mul);
    let sum = binary_f64(product, c, Operation::Add);
    super::narrow::round_double_to_half(sum.to_bits())
}

/// Returns a binary64 value rounded to binary32, by `CVTSD2SS` in the rounding
/// direction of MXCSR, which the path requires to be to nearest even.
#[inline]
pub fn narrow_double(value: f64) -> f32 {
    let result: f32;
    // SAFETY: CVTSD2SS reads and writes SSE registers. The build enables SSE2, and `sse!` selects a VEX form only in a build with AVX.
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

/// Returns `None`: x86 has no instruction that rounds to odd.
#[inline]
pub fn narrow_double_to_odd(_value: f64) -> Option<f32> {
    None
}

/// Returns a binary64 value rounded once to binary16, to nearest even, by the
/// integer rounding of the host paths. x86 has no instruction for it below
/// AVX512-FP16.
#[inline]
pub fn narrow_double_to_half(value: f64) -> u16 {
    super::narrow::round_double_to_half(value.to_bits())
}

/// Runs one rounding to an integral value on `$value`, with the rounding
/// control of `$rounding` in the immediate of the instruction: 0 to nearest
/// even, 1 toward negative infinity, 2 toward positive infinity, and 3
/// toward zero. Returns `None` from the function for another direction.
#[cfg(target_feature = "sse4.1")]
macro_rules! round_scalar {
    ($rounding:expr, [$even:expr, $down:expr, $up:expr, $zero:expr $(,)?], $value:ident) => {
        match $rounding {
            Rounding::TiesToEven => round_scalar!($even, $value),
            Rounding::TowardNegative => round_scalar!($down, $value),
            Rounding::TowardPositive => round_scalar!($up, $value),
            Rounding::TowardZero => round_scalar!($zero, $value),
            Rounding::TiesToAway
            | Rounding::TiesTowardZero
            | Rounding::AwayFromZero
            | Rounding::ToOdd => return None,
        }
    };
    ($instruction:expr, $value:ident) => {
        // SAFETY: ROUNDSS and ROUNDSD read and write one SSE register. The
        // build enables SSE4.1, and `sse!` selects a VEX form only in a build
        // with AVX. The instruction changes only the status flags of MXCSR,
        // which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                value = inout(xmm_reg) $value,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
    };
}

/// Returns a binary32 value rounded to an integral value in the direction
/// `rounding`, by `ROUNDSS` with the direction in its immediate, or `None`
/// for a direction that the immediate does not have.
#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn round_f32(mut value: f32, rounding: Rounding) -> Option<f32> {
    round_scalar!(
        rounding,
        [
            sse!(
                "roundss {value}, {value}, 0",
                "vroundss {value}, {value}, {value}, 0"
            ),
            sse!(
                "roundss {value}, {value}, 1",
                "vroundss {value}, {value}, {value}, 1"
            ),
            sse!(
                "roundss {value}, {value}, 2",
                "vroundss {value}, {value}, {value}, 2"
            ),
            sse!(
                "roundss {value}, {value}, 3",
                "vroundss {value}, {value}, {value}, 3"
            ),
        ],
        value
    );
    Some(value)
}

/// Returns a binary64 value rounded to an integral value in the direction
/// `rounding`, by `ROUNDSD` with the direction in its immediate, or `None`
/// for a direction that the immediate does not have.
#[cfg(target_feature = "sse4.1")]
#[inline]
pub fn round_f64(mut value: f64, rounding: Rounding) -> Option<f64> {
    round_scalar!(
        rounding,
        [
            sse!(
                "roundsd {value}, {value}, 0",
                "vroundsd {value}, {value}, {value}, 0"
            ),
            sse!(
                "roundsd {value}, {value}, 1",
                "vroundsd {value}, {value}, {value}, 1"
            ),
            sse!(
                "roundsd {value}, {value}, 2",
                "vroundsd {value}, {value}, {value}, 2"
            ),
            sse!(
                "roundsd {value}, {value}, 3",
                "vroundsd {value}, {value}, {value}, 3"
            ),
        ],
        value
    );
    Some(value)
}

/// Returns `None`: a build without SSE4.1 has no path that rounds to an
/// integral value.
#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn round_f32(_value: f32, _rounding: Rounding) -> Option<f32> {
    None
}

/// Returns `None`: a build without SSE4.1 has no path that rounds to an
/// integral value.
#[cfg(not(target_feature = "sse4.1"))]
#[inline]
pub fn round_f64(_value: f64, _rounding: Rounding) -> Option<f64> {
    None
}

/// Runs the conversion of `$value` to an integer of the type `$type` by
/// `$instruction`, in the rounding direction of MXCSR, and returns the
/// integer. The integer indefinite, the smallest value of the type, stands
/// for a NaN, an infinity, and a value out of range.
macro_rules! to_int {
    ($instruction:expr, $value:ident, $type:ty) => {{
        let result: $type;
        // SAFETY: the conversion reads an SSE register and writes a general
        // register. The build enables SSE2, and `sse!` selects a VEX form
        // only in a build with AVX. The conversion changes only the status
        // flags of MXCSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                value = in(xmm_reg) $value,
                result = lateout(reg) result,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        result
    }};
}

/// Returns a binary32 value rounded to a 64-bit integer by `CVTSS2SI` in the
/// rounding direction of MXCSR, or `None` for the integer indefinite, which a
/// NaN, an infinity, a value out of range, and `-2^63` give.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn to_int_f32(value: f32) -> Option<i64> {
    let result = to_int!(
        sse!("cvtss2si {result}, {value}", "vcvtss2si {result}, {value}"),
        value,
        i64
    );
    (result != i64::MIN).then_some(result)
}

/// Returns a binary64 value rounded to a 64-bit integer by `CVTSD2SI` in the
/// rounding direction of MXCSR, or `None` for the integer indefinite, which a
/// NaN, an infinity, a value out of range, and `-2^63` give.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn to_int_f64(value: f64) -> Option<i64> {
    let result = to_int!(
        sse!("cvtsd2si {result}, {value}", "vcvtsd2si {result}, {value}"),
        value,
        i64
    );
    (result != i64::MIN).then_some(result)
}

/// Returns a binary32 value rounded to a 32-bit integer by `CVTSS2SI` in the
/// rounding direction of MXCSR, or `None` for the integer indefinite, which a
/// NaN, an infinity, a value out of the range of `i32`, and `-2^31` give. The
/// engine converts those values to a 64-bit integer.
#[cfg(target_arch = "x86")]
#[inline]
pub fn to_int_f32(value: f32) -> Option<i64> {
    let result = to_int!(
        sse!("cvtss2si {result}, {value}", "vcvtss2si {result}, {value}"),
        value,
        i32
    );
    (result != i32::MIN).then_some(i64::from(result))
}

/// Returns a binary64 value rounded to a 32-bit integer by `CVTSD2SI`, as
/// [`to_int_f32`] states.
#[cfg(target_arch = "x86")]
#[inline]
pub fn to_int_f64(value: f64) -> Option<i64> {
    let result = to_int!(
        sse!("cvtsd2si {result}, {value}", "vcvtsd2si {result}, {value}"),
        value,
        i32
    );
    (result != i32::MIN).then_some(i64::from(result))
}

/// Runs the conversion of the integer `$value` to a binary format by
/// `$instruction`, after `$clear` clears the destination, in the rounding
/// direction of MXCSR, and returns the result of the type `$type`.
macro_rules! from_int {
    ($clear:expr, $instruction:expr, $value:expr, $type:ty) => {{
        let result: $type;
        // SAFETY: the clearing instruction clears an SSE register, so the
        // conversion does not wait on its old value. The conversion writes
        // the converted value into the low lane. The build enables SSE2, and
        // `sse!` selects a VEX form only in a build with AVX. The conversion
        // changes only the status flags of MXCSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $clear,
                $instruction,
                value = in(reg) $value,
                result = out(xmm_reg) result,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        result
    }};
}

/// Returns a 64-bit integer rounded to binary32, by `CVTSI2SS` in the rounding
/// direction of MXCSR.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of 32-bit x86, which converts only an `i32`.
pub fn from_int_f32(value: i64) -> Option<f32> {
    Some(from_int!(
        sse!(
            "xorps {result}, {result}",
            "vxorps {result}, {result}, {result}"
        ),
        sse!(
            "cvtsi2ss {result}, {value}",
            "vcvtsi2ss {result}, {result}, {value}"
        ),
        value,
        f32
    ))
}

/// Returns a 64-bit integer rounded to binary64, by `CVTSI2SD` in the rounding
/// direction of MXCSR.
#[cfg(target_arch = "x86_64")]
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of 32-bit x86, which converts only an `i32`.
pub fn from_int_f64(value: i64) -> Option<f64> {
    Some(from_int!(
        sse!(
            "xorpd {result}, {result}",
            "vxorpd {result}, {result}, {result}"
        ),
        sse!(
            "cvtsi2sd {result}, {value}",
            "vcvtsi2sd {result}, {result}, {value}"
        ),
        value,
        f64
    ))
}

/// Returns an integer in the range of `i32` rounded to binary32, by
/// `CVTSI2SS` in the rounding direction of MXCSR, or `None` for another
/// integer, which no general register of 32-bit x86 holds.
#[cfg(target_arch = "x86")]
#[inline]
pub fn from_int_f32(value: i64) -> Option<f32> {
    let value = i32::try_from(value).ok()?;
    Some(from_int!(
        sse!(
            "xorps {result}, {result}",
            "vxorps {result}, {result}, {result}"
        ),
        sse!(
            "cvtsi2ss {result}, {value}",
            "vcvtsi2ss {result}, {result}, {value}"
        ),
        value,
        f32
    ))
}

/// Returns an integer in the range of `i32` converted exactly to binary64,
/// by `CVTSI2SD`, or `None` for another integer, as [`from_int_f32`] states.
#[cfg(target_arch = "x86")]
#[inline]
pub fn from_int_f64(value: i64) -> Option<f64> {
    let value = i32::try_from(value).ok()?;
    Some(from_int!(
        sse!(
            "xorpd {result}, {result}",
            "vxorpd {result}, {result}, {result}"
        ),
        sse!(
            "cvtsi2sd {result}, {value}",
            "vcvtsi2sd {result}, {result}, {value}"
        ),
        value,
        f64
    ))
}

/// Runs one unordered comparison of `$left` with `$right`, `UCOMISS` or
/// `UCOMISD`, and returns the order, or `None` for an unordered pair.
macro_rules! compare {
    ($instruction:expr, $left:ident, $right:ident) => {{
        let (unordered, greater, less): (u8, u8, u8);
        // SAFETY: the comparison reads two SSE registers and writes the
        // arithmetic flags, which SETP, SETA, and SETB copy into three byte
        // registers. The build enables SSE2, and `sse!` selects a
        // VEX form only in a build with AVX. The comparison changes only the
        // status flags of MXCSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                $instruction,
                "setp {unordered}",
                "seta {greater}",
                "setb {less}",
                left = in(xmm_reg) $left,
                right = in(xmm_reg) $right,
                unordered = out(reg_byte) unordered,
                greater = out(reg_byte) greater,
                less = out(reg_byte) less,
                options(pure, nomem, nostack),
            );
        }
        // An unordered pair also sets the flags that SETB reads.
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

/// Returns the order of two binary32 values by `UCOMISS`, which signals
/// invalid only for a signaling NaN, or `None` when they are unordered.
#[inline]
pub fn compare_f32(left: f32, right: f32) -> Option<Ordering> {
    compare!(
        sse!("ucomiss {left}, {right}", "vucomiss {left}, {right}"),
        left,
        right
    )
}

/// Returns the order of two binary64 values by `UCOMISD`, which signals
/// invalid only for a signaling NaN, or `None` when they are unordered.
#[inline]
pub fn compare_f64(left: f64, right: f64) -> Option<Ordering> {
    compare!(
        sse!("ucomisd {left}, {right}", "vucomisd {left}, {right}"),
        left,
        right
    )
}

/// Returns the smaller of two binary32 values by `MINSS`, which gives `right`
/// when an operand is a NaN or both are zeros.
#[inline]
pub fn min_f32(mut left: f32, right: f32) -> f32 {
    sse_scalar!(
        sse!("minss {left}, {right}", "vminss {left}, {left}, {right}"),
        left,
        right
    );
    left
}

/// Returns the larger of two binary32 values by `MAXSS`, which gives `right`
/// when an operand is a NaN or both are zeros.
#[inline]
pub fn max_f32(mut left: f32, right: f32) -> f32 {
    sse_scalar!(
        sse!("maxss {left}, {right}", "vmaxss {left}, {left}, {right}"),
        left,
        right
    );
    left
}

/// Returns the smaller of two binary64 values by `MINSD`, which gives `right`
/// when an operand is a NaN or both are zeros.
#[inline]
pub fn min_f64(mut left: f64, right: f64) -> f64 {
    sse_scalar!(
        sse!("minsd {left}, {right}", "vminsd {left}, {left}, {right}"),
        left,
        right
    );
    left
}

/// Returns the larger of two binary64 values by `MAXSD`, which gives `right`
/// when an operand is a NaN or both are zeros.
#[inline]
pub fn max_f64(mut left: f64, right: f64) -> f64 {
    sse_scalar!(
        sse!("maxsd {left}, {right}", "vmaxsd {left}, {left}, {right}"),
        left,
        right
    );
    left
}

/// `false`: x86 has no binary128 unit.
pub const QUAD: bool = false;

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_binary(_left: &[u64; 2], _right: &[u64; 2], _operation: Operation) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_sqrt(_value: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_round(_value: &[u64; 2], _rounding: Rounding) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_to_int(_value: &[u64; 2]) -> Option<i64> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_from_int(_value: i64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_compare(_left: &[u64; 2], _right: &[u64; 2]) -> Option<Ordering> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_from_single(_bits: u32) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_from_double(_bits: u64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_to_single(_value: &[u64; 2]) -> Option<u32> {
    None
}

/// Returns `None`: x86 has no binary128 unit.
#[inline]
pub fn quad_to_double(_value: &[u64; 2]) -> Option<u64> {
    None
}
