//! The vector instructions of the AArch64 floating-point unit, for the paths
//! of `Lanes`: 128-bit registers of four binary32 or two binary64 lanes.
//!
//! Each function computes one chunk of `C` lanes into `out`, and returns
//! `None` for a chunk width that has no instruction. The block loads the
//! operands through pointers, computes in registers, and stores the result
//! through `out`.

use super::super::Operation;

/// `false`: AArch64 has no vector registers wider than 128 bits in the base
/// architecture.
pub const WIDE: bool = false;

/// Runs a vector operation on two chunks, and stores the result chunk
/// through `$out`. `$operation` combines the registers `a` and `b` into `a`.
macro_rules! two_chunks {
    ($operation:expr, $left:expr, $right:expr, $out:expr) => {{
        // SAFETY: the block loads each 128-bit operand through its pointer,
        // which points to a chunk of 16 bytes, and stores the result through a
        // pointer to a chunk of 16 bytes. Every AArch64 target has the
        // instructions, and they change only the status flags of FPSR, which
        // floaty does not read.
        unsafe {
            core::arch::asm!(
                "ldr {a:q}, [{left}]",
                "ldr {b:q}, [{right}]",
                $operation,
                "str {a:q}, [{result}]",
                left = in(reg) $left.as_ptr(),
                right = in(reg) $right.as_ptr(),
                result = in(reg) $out.as_mut_ptr(),
                a = out(vreg) _,
                b = out(vreg) _,
                options(nostack, preserves_flags),
            );
        }
        Some(())
    }};
}

/// Runs a vector operation on one chunk, and stores the result chunk through
/// `$out`. `$load` loads the chunk into `a`, `$operation` computes in `a`,
/// and `$store` stores `a`.
macro_rules! one_chunk {
    ($load:literal, $operation:literal, $store:literal, $value:expr, $out:expr) => {{
        // SAFETY: as in `two_chunks`, with one operand. The load and the
        // store move the widths of the chunks.
        unsafe {
            core::arch::asm!(
                $load,
                $operation,
                $store,
                value = in(reg) $value.as_ptr(),
                result = in(reg) $out.as_mut_ptr(),
                a = out(vreg) _,
                options(nostack, preserves_flags),
            );
        }
        Some(())
    }};
}

/// Runs a vector fused multiply-add on three chunks, and stores the result
/// chunk through `$out`. `FMLA` adds the product of `a` and `b` to `c`,
/// rounded once.
macro_rules! fused {
    ($arrangement:literal, $left:expr, $right:expr, $addend:expr, $out:expr) => {{
        // SAFETY: as in `two_chunks`, with a third operand that the block
        // loads through `addend`.
        unsafe {
            core::arch::asm!(
                "ldr {a:q}, [{left}]",
                "ldr {b:q}, [{right}]",
                "ldr {c:q}, [{addend}]",
                concat!("fmla {c:v}", $arrangement, ", {a:v}", $arrangement, ", {b:v}", $arrangement),
                "str {c:q}, [{result}]",
                left = in(reg) $left.as_ptr(),
                right = in(reg) $right.as_ptr(),
                addend = in(reg) $addend.as_ptr(),
                result = in(reg) $out.as_mut_ptr(),
                a = out(vreg) _,
                b = out(vreg) _,
                c = out(vreg) _,
                options(nostack, preserves_flags),
            );
        }
        Some(())
    }};
}

/// Selects the instruction of an operation, and stores the result chunk
/// through `$out`.
macro_rules! arithmetic {
    ($operation:expr, $arrangement:literal, $left:expr, $right:expr, $out:expr) => {
        match $operation {
            Operation::Add => two_chunks!(
                concat!(
                    "fadd {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $left,
                $right,
                $out
            ),
            Operation::Sub => two_chunks!(
                concat!(
                    "fsub {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $left,
                $right,
                $out
            ),
            Operation::Mul => two_chunks!(
                concat!(
                    "fmul {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $left,
                $right,
                $out
            ),
            Operation::Div => two_chunks!(
                concat!(
                    "fdiv {a:v}",
                    $arrangement,
                    ", {a:v}",
                    $arrangement,
                    ", {b:v}",
                    $arrangement
                ),
                $left,
                $right,
                $out
            ),
        }
    };
}

/// Computes `operation` of two chunks of four binary32 lanes into `out`, by
/// `FADD`, `FSUB`, `FMUL`, or `FDIV` on `.4S`.
#[inline]
pub fn binary_f32<const C: usize>(
    left: &[f32; C],
    right: &[f32; C],
    out: &mut [f32; C],
    operation: Operation,
) -> Option<()> {
    match C {
        4 => arithmetic!(operation, ".4s", left, right, out),
        _ => None,
    }
}

/// Computes `operation` of two chunks of two binary64 lanes into `out`, by
/// `FADD`, `FSUB`, `FMUL`, or `FDIV` on `.2D`.
#[inline]
pub fn binary_f64<const C: usize>(
    left: &[f64; C],
    right: &[f64; C],
    out: &mut [f64; C],
    operation: Operation,
) -> Option<()> {
    match C {
        2 => arithmetic!(operation, ".2d", left, right, out),
        _ => None,
    }
}

/// Computes the square roots of a chunk of four binary32 lanes into `out`,
/// by `FSQRT`.
#[inline]
pub fn sqrt_f32<const C: usize>(value: &[f32; C], out: &mut [f32; C]) -> Option<()> {
    match C {
        4 => one_chunk!(
            "ldr {a:q}, [{value}]",
            "fsqrt {a:v}.4s, {a:v}.4s",
            "str {a:q}, [{result}]",
            value,
            out
        ),
        _ => None,
    }
}

/// Computes the square roots of a chunk of two binary64 lanes into `out`,
/// by `FSQRT`.
#[inline]
pub fn sqrt_f64<const C: usize>(value: &[f64; C], out: &mut [f64; C]) -> Option<()> {
    match C {
        2 => one_chunk!(
            "ldr {a:q}, [{value}]",
            "fsqrt {a:v}.2d, {a:v}.2d",
            "str {a:q}, [{result}]",
            value,
            out
        ),
        _ => None,
    }
}

/// Computes the lanes of a chunk of four binary32 lanes rounded to integral
/// values to nearest even into `out`, by `FRINTN`.
#[inline]
pub fn round_f32<const C: usize>(value: &[f32; C], out: &mut [f32; C]) -> Option<()> {
    match C {
        4 => one_chunk!(
            "ldr {a:q}, [{value}]",
            "frintn {a:v}.4s, {a:v}.4s",
            "str {a:q}, [{result}]",
            value,
            out
        ),
        _ => None,
    }
}

/// Computes the lanes of a chunk of two binary64 lanes rounded to integral
/// values to nearest even into `out`, by `FRINTN`.
#[inline]
pub fn round_f64<const C: usize>(value: &[f64; C], out: &mut [f64; C]) -> Option<()> {
    match C {
        2 => one_chunk!(
            "ldr {a:q}, [{value}]",
            "frintn {a:v}.2d, {a:v}.2d",
            "str {a:q}, [{result}]",
            value,
            out
        ),
        _ => None,
    }
}

/// Computes `left * right + addend` of chunks of four binary32 lanes into
/// `out`, each lane rounded once, by `FMLA`.
#[inline]
pub fn mul_add_f32<const C: usize>(
    left: &[f32; C],
    right: &[f32; C],
    addend: &[f32; C],
    out: &mut [f32; C],
) -> Option<()> {
    match C {
        4 => fused!(".4s", left, right, addend, out),
        _ => None,
    }
}

/// Computes `left * right + addend` of chunks of two binary64 lanes into
/// `out`, each lane rounded once, by `FMLA`.
#[inline]
pub fn mul_add_f64<const C: usize>(
    left: &[f64; C],
    right: &[f64; C],
    addend: &[f64; C],
    out: &mut [f64; C],
) -> Option<()> {
    match C {
        2 => fused!(".2d", left, right, addend, out),
        _ => None,
    }
}

/// Computes a chunk of two binary32 lanes widened exactly to binary64 into
/// `out`, by `FCVTL`.
#[inline]
pub fn widen<const C: usize>(value: &[f32; C], out: &mut [f64; C]) -> Option<()> {
    match C {
        2 => one_chunk!(
            "ldr {a:d}, [{value}]",
            "fcvtl {a:v}.2d, {a:v}.2s",
            "str {a:q}, [{result}]",
            value,
            out
        ),
        _ => None,
    }
}

/// Computes a chunk of two binary64 lanes rounded to binary32 into `out`, by
/// `FCVTN`.
#[inline]
pub fn narrow<const C: usize>(value: &[f64; C], out: &mut [f32; C]) -> Option<()> {
    match C {
        2 => one_chunk!(
            "ldr {a:q}, [{value}]",
            "fcvtn {a:v}.2s, {a:v}.2d",
            "str {a:d}, [{result}]",
            value,
            out
        ),
        _ => None,
    }
}
