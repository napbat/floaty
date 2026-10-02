//! The binary128 instructions of s390x, the extended binary format of
//! z/Architecture, which is IEEE 754 binary128.
//!
//! A binary128 value occupies a pair of floating-point registers, the high
//! doubleword in the lower register: f0 with f2, or f1 with f3. Rust names no
//! register pair, so each block loads the limbs into named registers with
//! `LDGR`, and stores the result with `LGDR`. A floaty encoding holds the low
//! limb first. The instructions are those of the base z/Architecture:
//! Principles of Operation, SA22-7832-13, chapter 19.

use core::cmp::Ordering;

use super::super::Operation;
use crate::env::Rounding;

/// Returns a binary128 result, or `None` for a NaN: the largest exponent
/// field with a significand other than zero.
#[inline]
fn quad_result(bits: [u64; 2]) -> Option<[u64; 2]> {
    let exponent = (bits[1] >> 48) & 0x7FFF;
    let significand = (bits[1] & 0xFFFF_FFFF_FFFF) | bits[0];
    (exponent != 0x7FFF || significand == 0).then_some(bits)
}

/// Runs `$instruction` on the binary128 operand `$value` in f0 and f2, and
/// on a second operand `$other` in f1 and f3, and returns the binary128 value
/// that it leaves in f0 and f2, or `None` for a NaN.
macro_rules! quad {
    ($instruction:literal, $value:expr) => {
        quad!($instruction, $value, &[0, 0])
    };
    ($instruction:literal, $value:expr, $other:expr) => {{
        let (value, other): (&[u64; 2], &[u64; 2]) = ($value, $other);
        let (high, low): (u64, u64);
        // SAFETY: LDGR loads the limbs into the named floating-point
        // registers, the instruction computes in register pairs, and LGDR
        // copies the result into general registers. Every z/Architecture
        // processor has them. The instruction can set the condition code and
        // sets the IEEE flags of the FPC, which floaty does not read, and the
        // path requires the IEEE masks to be zero, so it does not trap. The
        // block names every register that it writes.
        unsafe {
            core::arch::asm!(
                "ldgr %f0, {value_high}",
                "ldgr %f2, {value_low}",
                "ldgr %f1, {other_high}",
                "ldgr %f3, {other_low}",
                $instruction,
                "lgdr {high}, %f0",
                "lgdr {low}, %f2",
                value_high = in(reg) value[1],
                value_low = in(reg) value[0],
                other_high = in(reg) other[1],
                other_low = in(reg) other[0],
                high = lateout(reg) high,
                low = lateout(reg) low,
                out("f0") _,
                out("f1") _,
                out("f2") _,
                out("f3") _,
                options(pure, nomem, nostack),
            );
        }
        quad_result([low, high])
    }};
}

/// Returns `operation` of two binary128 values, by `AXBR`, `SXBR`, `MXBR`, or
/// `DXBR` in the rounding mode of the FPC register, or `None` for a NaN.
#[inline]
pub fn quad_binary(left: &[u64; 2], right: &[u64; 2], operation: Operation) -> Option<[u64; 2]> {
    match operation {
        Operation::Add => quad!("axbr %f0, %f1", left, right),
        Operation::Sub => quad!("sxbr %f0, %f1", left, right),
        Operation::Mul => quad!("mxbr %f0, %f1", left, right),
        Operation::Div => quad!("dxbr %f0, %f1", left, right),
    }
}

/// Returns the square root of a binary128 value, by `SQXBR`, or `None` for
/// a NaN.
#[inline]
pub fn quad_sqrt(value: &[u64; 2]) -> Option<[u64; 2]> {
    quad!("sqxbr %f0, %f0", value)
}

/// Returns a binary128 value rounded to an integral value in the direction
/// `rounding` by `FIXBR`, with the rounding method in its M3 field as
/// `round_f32` states, or `None` for a direction that the instruction does
/// not encode or a NaN.
#[inline]
pub fn quad_round(value: &[u64; 2], rounding: Rounding) -> Option<[u64; 2]> {
    match rounding {
        Rounding::TiesToEven => quad!("fixbr %f0, 4, %f0", value),
        Rounding::TiesToAway => quad!("fixbr %f0, 1, %f0", value),
        Rounding::TowardZero => quad!("fixbr %f0, 5, %f0", value),
        Rounding::TowardPositive => quad!("fixbr %f0, 6, %f0", value),
        Rounding::TowardNegative => quad!("fixbr %f0, 7, %f0", value),
        Rounding::TiesTowardZero | Rounding::AwayFromZero | Rounding::ToOdd => None,
    }
}

/// Returns a binary128 value rounded to a 64-bit integer to nearest even by
/// `CGXBR`, or `None` for a value at a bound of the conversion, as
/// `to_int_f32` states.
#[inline]
pub fn quad_to_int(value: &[u64; 2]) -> Option<i64> {
    let result: i64;
    // SAFETY: as in `quad!`. CGXBR writes a general register.
    unsafe {
        core::arch::asm!(
            "ldgr %f0, {high}",
            "ldgr %f2, {low}",
            "cgxbr {result}, 4, %f0",
            high = in(reg) value[1],
            low = in(reg) value[0],
            result = lateout(reg) result,
            out("f0") _,
            out("f2") _,
            options(pure, nomem, nostack),
        );
    }
    (result != i64::MIN && result != i64::MAX).then_some(result)
}

/// Returns a 64-bit integer converted to binary128 by `CXGBR`, which holds
/// it exactly.
#[inline]
#[allow(clippy::unnecessary_wraps)] // The signature is that of the other architectures, which return `None`.
pub fn quad_from_int(value: i64) -> Option<[u64; 2]> {
    let (high, low): (u64, u64);
    // SAFETY: as in `quad!`. CXGBR reads a general register.
    unsafe {
        core::arch::asm!(
            "cxgbr %f0, {value}",
            "lgdr {high}, %f0",
            "lgdr {low}, %f2",
            value = in(reg) value,
            high = lateout(reg) high,
            low = lateout(reg) low,
            out("f0") _,
            out("f2") _,
            options(pure, nomem, nostack),
        );
    }
    Some([low, high])
}

/// Returns the order of two binary128 values by `CXBR`, which signals invalid
/// only for a signaling NaN, or `None` when they are unordered.
#[inline]
pub fn quad_compare(left: &[u64; 2], right: &[u64; 2]) -> Option<Ordering> {
    let code: u32;
    // SAFETY: as in `quad!`. CXBR sets the condition code, which IPM copies
    // into a general register.
    unsafe {
        core::arch::asm!(
            "ldgr %f0, {left_high}",
            "ldgr %f2, {left_low}",
            "ldgr %f1, {right_high}",
            "ldgr %f3, {right_low}",
            "cxbr %f0, %f1",
            "ipm {code}",
            "srl {code}, 28",
            left_high = in(reg) left[1],
            left_low = in(reg) left[0],
            right_high = in(reg) right[1],
            right_low = in(reg) right[0],
            code = lateout(reg) code,
            out("f0") _,
            out("f1") _,
            out("f2") _,
            out("f3") _,
            options(pure, nomem, nostack),
        );
    }
    match code {
        0 => Some(Ordering::Equal),
        1 => Some(Ordering::Less),
        2 => Some(Ordering::Greater),
        _ => None,
    }
}

/// Returns a binary32 encoding widened exactly to binary128 by `LXEBR`, or
/// `None` for a NaN.
#[inline]
pub fn quad_from_single(bits: u32) -> Option<[u64; 2]> {
    // A binary32 value occupies the high half of its register.
    quad!("lxebr %f0, %f1", &[0, 0], &[0, u64::from(bits) << 32])
}

/// Returns a binary64 encoding widened exactly to binary128 by `LXDBR`, or
/// `None` for a NaN.
#[inline]
pub fn quad_from_double(bits: u64) -> Option<[u64; 2]> {
    quad!("lxdbr %f0, %f1", &[0, 0], &[0, bits])
}

/// Runs the rounding of the binary128 value `$value` in f1 and f3 to a
/// shorter format by `$instruction`, and returns the 64 bits of f0.
macro_rules! narrow {
    ($instruction:literal, $value:expr) => {{
        let value: &[u64; 2] = $value;
        let result: u64;
        // SAFETY: as in `quad!`. The instruction rounds in the rounding mode
        // of the FPC register, which the path requires to be to nearest even.
        unsafe {
            core::arch::asm!(
                "ldgr %f1, {high}",
                "ldgr %f3, {low}",
                $instruction,
                "lgdr {result}, %f0",
                high = in(reg) value[1],
                low = in(reg) value[0],
                result = lateout(reg) result,
                out("f0") _,
                out("f1") _,
                out("f2") _,
                out("f3") _,
                options(pure, nomem, nostack),
            );
        }
        result
    }};
}

/// Returns a binary128 value rounded to binary32 by `LEXBR`, or `None` for a
/// NaN.
#[inline]
pub fn quad_to_single(value: &[u64; 2]) -> Option<u32> {
    let high = narrow!("lexbr %f0, %f1", value);
    let bits = u32::try_from(high >> 32).expect("a shift by 32 leaves 32 bits");
    (!super::super::bits::nan_32(bits)).then_some(bits)
}

/// Returns a binary128 value rounded to binary64 by `LDXBR`, or `None` for a
/// NaN.
#[inline]
pub fn quad_to_double(value: &[u64; 2]) -> Option<u64> {
    let bits = narrow!("ldxbr %f0, %f1", value);
    (!super::super::bits::nan_64(bits)).then_some(bits)
}
