//! The floating-point environment of s390x, the FPC register, and the binary
//! floating-point instructions of the host paths.
//!
//! The instructions are those of the base z/Architecture: z/Architecture
//! Principles of Operation, SA22-7832-13, chapter 19, "Binary-Floating-Point
//! Instructions". binary16 and bfloat16 widen and round in integer
//! instructions, as on x86-64 without F16C. Many of the instructions set the
//! condition code, so no block preserves the flags.

use core::cmp::Ordering;

use super::Operation;
use crate::env::Rounding;

pub mod packed;

/// `true`: the build has the floating-point unit.
pub const UNIT: bool = true;
/// `true`: `MAEBR` and `MADBR` compute the fused multiply-add.
pub const FUSED: bool = true;
/// `false`: s390x has no binary16 conversion instruction, so integer
/// instructions widen and round binary16.
pub const HALF: bool = false;
/// `true`: `FIEBR` and `FIDBR` round to an integral value in a direction from
/// their encoding.
pub const ROUNDING: bool = true;
/// `false`: s390x has no x87 unit.
pub const X87: bool = false;
/// `true`: `MADBR` computes the binary16 fused multiply-add in binary64, and
/// integer instructions round the binary64 sum to binary16.
pub const HALF_FUSED: bool = true;
/// `true`: integer instructions round binary64 to binary16 once.
pub const DOUBLE_TO_HALF: bool = true;
/// `false`: the base z/Architecture has no instruction that rounds to odd.
/// The floating-point-extension facility adds one, and the default target
/// does not have it.
pub const ROUND_TO_ODD: bool = false;

/// Returns `true` when the floating-point unit rounds binary results to
/// nearest even and enables no exception trap.
///
/// Rust code runs with the FPC register zero, but an emulator or a caller can
/// load another value. The path then goes back to the engine. The fields are
/// those of the FPC register in the z/Architecture Principles of Operation,
/// SA22-7832-13, chapter 9: the IEEE masks of invalid operation, division by
/// zero, overflow, underflow, and inexact are bits 0 to 4, and the
/// binary-floating-point rounding mode is bits 29 to 31, where zero rounds to
/// nearest even. The binary unit has no flush to zero. The flags and the data
/// exception code record exceptions, and the decimal fields change no binary
/// result.
#[inline]
pub fn default_environment() -> bool {
    // The IEEE masks, bits 0 to 4 counted from the most significant bit, and
    // the binary rounding mode, bits 29 to 31.
    const RESULT_FIELDS: u32 = (0x1F << 27) | 0b111;
    let fpc: u32;
    // SAFETY: EFPC copies the FPC register into a general register. It reads
    // no memory and changes no other state.
    unsafe {
        core::arch::asm!(
            "efpc {fpc}",
            fpc = out(reg) fpc,
            options(nomem, nostack, preserves_flags),
        );
    }
    fpc & RESULT_FIELDS == 0
}

/// Runs one instruction of the floating-point unit on `$left` and `$right`,
/// and leaves the result in `$left`.
macro_rules! scalar {
    ($instruction:literal, $left:ident, $right:ident) => {
        // SAFETY: the instruction reads and writes floating-point registers
        // and can set the condition code. Every z/Architecture processor has
        // it. It sets the IEEE flags of the FPC, which floaty does not read,
        // and the path requires the IEEE masks to be zero, so it does not
        // trap.
        unsafe {
            core::arch::asm!(
                $instruction,
                left = inout(freg) $left,
                right = in(freg) $right,
                options(pure, nomem, nostack),
            );
        }
    };
}

/// Returns `operation` of two binary32 values, by `AEBR`, `SEBR`, `MEEBR`, or
/// `DEBR` in the rounding mode of the FPC register.
#[inline]
pub fn binary_f32(mut left: f32, right: f32, operation: Operation) -> f32 {
    match operation {
        Operation::Add => scalar!("aebr {left}, {right}", left, right),
        Operation::Sub => scalar!("sebr {left}, {right}", left, right),
        Operation::Mul => scalar!("meebr {left}, {right}", left, right),
        Operation::Div => scalar!("debr {left}, {right}", left, right),
    }
    left
}

/// Returns `operation` of two binary64 values, by `ADBR`, `SDBR`, `MDBR`, or
/// `DDBR` in the rounding mode of the FPC register.
#[inline]
pub fn binary_f64(mut left: f64, right: f64, operation: Operation) -> f64 {
    match operation {
        Operation::Add => scalar!("adbr {left}, {right}", left, right),
        Operation::Sub => scalar!("sdbr {left}, {right}", left, right),
        Operation::Mul => scalar!("mdbr {left}, {right}", left, right),
        Operation::Div => scalar!("ddbr {left}, {right}", left, right),
    }
    left
}

/// Runs one instruction of the floating-point unit on `$value`, and returns
/// its result of the type `$type`.
macro_rules! unary {
    ($instruction:expr, $value:ident, $type:ty) => {{
        let result: $type;
        // SAFETY: as in `scalar!`.
        unsafe {
            core::arch::asm!(
                $instruction,
                value = in(freg) $value,
                result = lateout(freg) result,
                options(pure, nomem, nostack),
            );
        }
        result
    }};
}

/// Returns a binary32 value widened exactly to binary64, by `LDEBR`.
#[inline]
pub fn widen_single(value: f32) -> f64 {
    unary!("ldebr {result}, {value}", value, f64)
}

/// Returns the square root of a binary32 value, by `SQEBR`.
#[inline]
pub fn sqrt_f32(value: f32) -> f32 {
    unary!("sqebr {result}, {value}", value, f32)
}

/// Returns the square root of a binary64 value, by `SQDBR`.
#[inline]
pub fn sqrt_f64(value: f64) -> f64 {
    unary!("sqdbr {result}, {value}", value, f64)
}

/// Runs a fused multiply-add: `$instruction` adds the product of `$left` and
/// `$right` to `$addend`, rounded once, and leaves the result in `$addend`.
macro_rules! fused {
    ($instruction:literal, $left:ident, $right:ident, $addend:ident) => {
        // SAFETY: as in `scalar!`, with a third register.
        unsafe {
            core::arch::asm!(
                $instruction,
                addend = inout(freg) $addend,
                left = in(freg) $left,
                right = in(freg) $right,
                options(pure, nomem, nostack),
            );
        }
    };
}

/// Returns `left * right + addend` of binary32 values, rounded once, by
/// `MAEBR`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack FMA.
pub fn mul_add_f32(left: f32, right: f32, mut addend: f32) -> Option<f32> {
    fused!("maebr {addend}, {left}, {right}", left, right, addend);
    Some(addend)
}

/// Returns `left * right + addend` of binary64 values, rounded once, by
/// `MADBR`.
#[inline]
fn madbr(left: f64, right: f64, mut addend: f64) -> f64 {
    fused!("madbr {addend}, {left}, {right}", left, right, addend);
    addend
}

/// Returns `left * right + addend` of binary64 values, rounded once, by
/// `MADBR`.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of x86-64, where a build can lack FMA.
pub fn mul_add_f64(left: f64, right: f64, addend: f64) -> Option<f64> {
    Some(madbr(left, right, addend))
}

/// Returns a binary16 value widened exactly to binary32, by the integer
/// widening of the host paths.
#[inline]
pub fn widen_half(bits: u16) -> f32 {
    f32::from_bits(super::narrow::widen_half_bits(bits))
}

/// Returns a binary32 value rounded to binary16 to nearest even, by the
/// integer rounding of the host paths.
#[inline]
pub fn narrow_half(value: f32) -> u16 {
    super::narrow::round_to_half(value.to_bits())
}

/// Returns a binary32 value rounded to bfloat16 to nearest even, by the
/// integer rounding of the host paths.
#[inline]
pub fn narrow_bfloat(value: f32) -> u16 {
    super::narrow::round_to_bfloat(value.to_bits())
}

/// Returns `left * right + addend` of binary16 encodings, rounded once. The
/// operands widen to binary64 exactly, `MADBR` rounds the sum of the exact
/// product to binary64, and integer instructions round that sum to binary16.
/// [`Host::Half`](super::Host::Half) states why the two roundings give the
/// result of one.
#[inline]
pub fn mul_add_f16(left: u16, right: u16, addend: u16) -> u16 {
    let [a, b, c] = [left, right, addend].map(|bits| widen_single(widen_half(bits)));
    super::narrow::round_double_to_half(madbr(a, b, c).to_bits())
}

/// Returns a binary64 value rounded to binary32, by `LEDBR` in the rounding
/// mode of the FPC register, which the path requires to be to nearest even.
#[inline]
pub fn narrow_double(value: f64) -> f32 {
    unary!("ledbr {result}, {value}", value, f32)
}

/// Returns `None`: the base z/Architecture has no instruction that rounds to
/// odd.
#[inline]
pub fn narrow_double_to_odd(_value: f64) -> Option<f32> {
    None
}

/// Returns a binary64 value rounded once to binary16, to nearest even, by the
/// integer rounding of the host paths.
#[inline]
pub fn narrow_double_to_half(value: f64) -> u16 {
    super::narrow::round_double_to_half(value.to_bits())
}

/// Runs the rounding to an integral value of `$value` by `$instruction`, with
/// the rounding method of `$rounding` in its M3 field: 4 to nearest even, 1
/// to nearest with ties away from zero, 5 toward zero, 6 toward positive
/// infinity, and 7 toward negative infinity. Returns `None` from the function
/// for another direction.
macro_rules! round_scalar {
    ($instruction:literal, $rounding:expr, $value:ident, $type:ty) => {
        match $rounding {
            Rounding::TiesToEven => {
                unary!(
                    concat!($instruction, " {result}, 4, {value}"),
                    $value,
                    $type
                )
            }
            Rounding::TiesToAway => {
                unary!(
                    concat!($instruction, " {result}, 1, {value}"),
                    $value,
                    $type
                )
            }
            Rounding::TowardZero => {
                unary!(
                    concat!($instruction, " {result}, 5, {value}"),
                    $value,
                    $type
                )
            }
            Rounding::TowardPositive => {
                unary!(
                    concat!($instruction, " {result}, 6, {value}"),
                    $value,
                    $type
                )
            }
            Rounding::TowardNegative => {
                unary!(
                    concat!($instruction, " {result}, 7, {value}"),
                    $value,
                    $type
                )
            }
            Rounding::TiesTowardZero | Rounding::AwayFromZero | Rounding::ToOdd => return None,
        }
    };
}

/// Returns a binary32 value rounded to an integral value in the direction
/// `rounding` by `FIEBR`, or `None` for a direction that the instruction does
/// not encode.
#[inline]
pub fn round_f32(value: f32, rounding: Rounding) -> Option<f32> {
    Some(round_scalar!("fiebr", rounding, value, f32))
}

/// Returns a binary64 value rounded to an integral value in the direction
/// `rounding` by `FIDBR`, or `None` for a direction that the instruction does
/// not encode.
#[inline]
pub fn round_f64(value: f64, rounding: Rounding) -> Option<f64> {
    Some(round_scalar!("fidbr", rounding, value, f64))
}

/// Runs the conversion of `$value` to a 64-bit integer to nearest even by
/// `$instruction` with the rounding method 4, and returns the integer.
macro_rules! to_int {
    ($instruction:literal, $value:ident) => {{
        let result: i64;
        // SAFETY: the instruction reads a floating-point register, writes a
        // general register, and sets the condition code. Every z/Architecture
        // processor has it. It sets the IEEE flags of the FPC, which floaty
        // does not read, and the path requires the IEEE masks to be zero, so
        // it does not trap.
        unsafe {
            core::arch::asm!(
                concat!($instruction, " {result}, 4, {value}"),
                value = in(freg) $value,
                result = lateout(reg) result,
                options(pure, nomem, nostack),
            );
        }
        result
    }};
}

/// Returns a binary32 value rounded to a 64-bit integer to nearest even by
/// `CGEBR`, or `None` for a value at a bound of the conversion: a value out
/// of range gives the bound of its sign, and a NaN gives `i64::MIN`.
#[inline]
pub fn to_int_f32(value: f32) -> Option<i64> {
    let result = to_int!("cgebr", value);
    (result != i64::MIN && result != i64::MAX).then_some(result)
}

/// Returns a binary64 value rounded to a 64-bit integer to nearest even by
/// `CGDBR`, or `None` for a value at a bound of the conversion, as
/// [`to_int_f32`] states.
#[inline]
pub fn to_int_f64(value: f64) -> Option<i64> {
    let result = to_int!("cgdbr", value);
    (result != i64::MIN && result != i64::MAX).then_some(result)
}

/// Runs the conversion of the 64-bit integer `$value` to a binary format by
/// `$instruction` in the rounding mode of the FPC register, and returns the
/// result of the type `$type`.
macro_rules! from_int {
    ($instruction:literal, $value:ident, $type:ty) => {{
        let result: $type;
        // SAFETY: the instruction reads a general register and writes a
        // floating-point register. Every z/Architecture processor has it. It
        // sets the IEEE flags of the FPC, which floaty does not read, and the
        // path requires the IEEE masks to be zero, so it does not trap.
        unsafe {
            core::arch::asm!(
                concat!($instruction, " {result}, {value}"),
                value = in(reg) $value,
                result = lateout(freg) result,
                options(pure, nomem, nostack),
            );
        }
        result
    }};
}

/// Returns a 64-bit integer rounded to binary32, by `CEGBR` in the rounding
/// mode of the FPC register.
#[inline]
pub fn from_int_f32(value: i64) -> f32 {
    from_int!("cegbr", value, f32)
}

/// Returns a 64-bit integer rounded to binary64, by `CDGBR` in the rounding
/// mode of the FPC register.
#[inline]
pub fn from_int_f64(value: i64) -> f64 {
    from_int!("cdgbr", value, f64)
}

/// Returns `false`: s390x has no x87 unit, so no x87 extended path applies.
#[inline]
pub fn x87_environment() -> bool {
    false
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_binary(_left: &[u64; 2], _right: &[u64; 2], _operation: Operation) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_sqrt(_value: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_round(_value: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_from_int(_value: i64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_from_single(_bits: u32) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_from_double(_bits: u64) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_remainder(_dividend: &[u64; 2], _divisor: &[u64; 2]) -> Option<[u64; 2]> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_remainder_single(_dividend: u32, _divisor: u32) -> Option<u32> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_remainder_double(_dividend: u64, _divisor: u64) -> Option<u64> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_to_int(_value: &[u64; 2]) -> Option<i64> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_to_single(_value: &[u64; 2]) -> Option<u32> {
    None
}

/// Returns `None`: s390x has no x87 unit.
#[inline]
pub fn x87_to_double(_value: &[u64; 2]) -> Option<u64> {
    None
}

/// Runs one quiet comparison of `$left` with `$right` by `$instruction`, and
/// returns the order, or `None` for an unordered pair. The condition code is
/// 0 for equal operands, 1 when the first is low, 2 when it is high, and 3
/// for unordered operands.
macro_rules! compare {
    ($instruction:literal, $left:ident, $right:ident) => {{
        let code: u32;
        // SAFETY: the comparison reads two floating-point registers and sets
        // the condition code, which IPM copies into a general register. Every
        // z/Architecture processor has them. The comparison signals invalid
        // only for a signaling NaN, sets the IEEE flags of the FPC, which
        // floaty does not read, and the path requires the IEEE masks to be
        // zero, so it does not trap.
        unsafe {
            core::arch::asm!(
                concat!($instruction, " {left}, {right}"),
                "ipm {code}",
                "srl {code}, 28",
                left = in(freg) $left,
                right = in(freg) $right,
                code = out(reg) code,
                options(pure, nomem, nostack),
            );
        }
        match code {
            0 => Some(Ordering::Equal),
            1 => Some(Ordering::Less),
            2 => Some(Ordering::Greater),
            _ => None,
        }
    }};
}

/// Returns the order of two binary32 values by `CEBR`, which signals invalid
/// only for a signaling NaN, or `None` when they are unordered.
#[inline]
pub fn compare_f32(left: f32, right: f32) -> Option<Ordering> {
    compare!("cebr", left, right)
}

/// Returns the order of two binary64 values by `CDBR`, which signals invalid
/// only for a signaling NaN, or `None` when they are unordered.
#[inline]
pub fn compare_f64(left: f64, right: f64) -> Option<Ordering> {
    compare!("cdbr", left, right)
}

/// Returns the smaller of two binary32 values, or `left` when they are equal,
/// by `CEBR`. The base z/Architecture has no minimum instruction, and the
/// paths never pass a NaN or two zeros.
#[inline]
pub fn min_f32(left: f32, right: f32) -> f32 {
    if compare_f32(right, left) == Some(Ordering::Less) {
        right
    } else {
        left
    }
}

/// Returns the larger of two binary32 values, or `left` when they are equal,
/// by `CEBR`, as [`min_f32`] states.
#[inline]
pub fn max_f32(left: f32, right: f32) -> f32 {
    if compare_f32(right, left) == Some(Ordering::Greater) {
        right
    } else {
        left
    }
}

/// Returns the smaller of two binary64 values, or `left` when they are equal,
/// by `CDBR`, as [`min_f32`] states.
#[inline]
pub fn min_f64(left: f64, right: f64) -> f64 {
    if compare_f64(right, left) == Some(Ordering::Less) {
        right
    } else {
        left
    }
}

/// Returns the larger of two binary64 values, or `left` when they are equal,
/// by `CDBR`, as [`min_f32`] states.
#[inline]
pub fn max_f64(left: f64, right: f64) -> f64 {
    if compare_f64(right, left) == Some(Ordering::Greater) {
        right
    } else {
        left
    }
}
