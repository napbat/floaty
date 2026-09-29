//! The packed instructions of the SSE unit, for the paths of `Lanes`: 128-bit
//! registers with SSE2, SSE4.1, and FMA, and 256-bit registers with AVX.
//!
//! Each function computes one chunk of `C` lanes into `out`, and returns
//! `None` for a chunk width that the build has no instruction for. The block
//! loads the operands through pointers with an unaligned move, computes in
//! registers, and stores the result through `out`. A legacy SSE instruction
//! with a 128-bit memory operand faults on an address that is not a multiple
//! of 16, and an array of lanes has no such alignment. The last arm of each
//! `match` uses the parameters, so a build with no instruction for any width
//! has no unused parameter.

use super::super::Operation;

/// `true` when the build has 256-bit registers, which hold eight binary32 or
/// four binary64 lanes.
pub const WIDE: bool = cfg!(target_feature = "avx");

/// Runs a packed operation on two chunks in registers of the class `$class`,
/// and stores the result chunk through `$out`. `$operation` is the
/// instruction that combines the registers `a` and `b` into `a`.
macro_rules! two_chunks {
    ($class:ident, $move:literal, $operation:expr, $left:expr, $right:expr, $out:expr) => {{
        // SAFETY: the block loads each operand through its pointer, which
        // points to a chunk of the width of the register, and stores the
        // result through a pointer to a chunk of that width. The build enables
        // the instructions of the register class, and they change only the
        // status flags of MXCSR, which floaty does not read.
        unsafe {
            core::arch::asm!(
                concat!($move, " {a}, [{left}]"),
                concat!($move, " {b}, [{right}]"),
                $operation,
                concat!($move, " [{result}], {a}"),
                left = in(reg) $left.as_ptr(),
                right = in(reg) $right.as_ptr(),
                result = in(reg) $out.as_mut_ptr(),
                a = out($class) _,
                b = out($class) _,
                options(nostack, preserves_flags),
            );
        }
        Some(())
    }};
}

/// Runs a packed operation on one chunk, and stores the result chunk through
/// `$out`. `$load` loads the chunk through `value` into `a`, `$operation`
/// computes in `a`, and `$store` stores `a` through `result`.
macro_rules! one_chunk {
    ($class:ident, $load:literal, $operation:literal, $store:literal, $value:expr, $out:expr) => {{
        // SAFETY: as in `two_chunks`, with one operand. The load and the
        // store move the widths of the chunks.
        unsafe {
            core::arch::asm!(
                $load,
                $operation,
                $store,
                value = in(reg) $value.as_ptr(),
                result = in(reg) $out.as_mut_ptr(),
                a = out($class) _,
                options(nostack, preserves_flags),
            );
        }
        Some(())
    }};
}

/// Runs a packed fused multiply-add on three chunks, `a = a * b + addend`,
/// and stores the result chunk through `$out`.
#[cfg(target_feature = "fma")]
macro_rules! fused {
    ($class:ident, $move:literal, $operation:literal, $left:expr, $right:expr, $addend:expr, $out:expr) => {{
        // SAFETY: as in `two_chunks`, with a third operand that the
        // instruction reads through `addend`. The build enables FMA, whose
        // VEX encoding needs no alignment of a memory operand.
        unsafe {
            core::arch::asm!(
                concat!($move, " {a}, [{left}]"),
                concat!($move, " {b}, [{right}]"),
                concat!($operation, " {a}, {b}, [{addend}]"),
                concat!($move, " [{result}], {a}"),
                left = in(reg) $left.as_ptr(),
                right = in(reg) $right.as_ptr(),
                addend = in(reg) $addend.as_ptr(),
                result = in(reg) $out.as_mut_ptr(),
                a = out($class) _,
                b = out($class) _,
                options(nostack, preserves_flags),
            );
        }
        Some(())
    }};
}

/// Selects the instruction of an operation from its four names, and stores
/// the result chunk through `$out`.
macro_rules! arithmetic {
    ($class:ident, $move:literal, $operation:expr, [$add:literal, $sub:literal, $mul:literal, $div:literal], $form:literal, $left:expr, $right:expr, $out:expr) => {
        match $operation {
            Operation::Add => two_chunks!($class, $move, concat!($add, $form), $left, $right, $out),
            Operation::Sub => two_chunks!($class, $move, concat!($sub, $form), $left, $right, $out),
            Operation::Mul => two_chunks!($class, $move, concat!($mul, $form), $left, $right, $out),
            Operation::Div => two_chunks!($class, $move, concat!($div, $form), $left, $right, $out),
        }
    };
}

/// Computes `operation` of two chunks of binary32 lanes into `out`, by
/// `ADDPS`, `SUBPS`, `MULPS`, or `DIVPS`, or their 256-bit AVX forms.
#[inline]
pub fn binary_f32<const C: usize>(
    left: &[f32; C],
    right: &[f32; C],
    out: &mut [f32; C],
    operation: Operation,
) -> Option<()> {
    match C {
        4 => arithmetic!(
            xmm_reg,
            "movups",
            operation,
            ["addps", "subps", "mulps", "divps"],
            " {a}, {b}",
            left,
            right,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        8 => unsafe { wide::binary_f32(left, right, out, operation) },
        _ => None,
    }
}

/// Computes `operation` of two chunks of binary64 lanes into `out`, by
/// `ADDPD`, `SUBPD`, `MULPD`, or `DIVPD`, or their 256-bit AVX forms.
#[inline]
pub fn binary_f64<const C: usize>(
    left: &[f64; C],
    right: &[f64; C],
    out: &mut [f64; C],
    operation: Operation,
) -> Option<()> {
    match C {
        2 => arithmetic!(
            xmm_reg,
            "movupd",
            operation,
            ["addpd", "subpd", "mulpd", "divpd"],
            " {a}, {b}",
            left,
            right,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        4 => unsafe { wide::binary_f64(left, right, out, operation) },
        _ => None,
    }
}

/// Computes the square roots of a chunk of binary32 lanes into `out`, by
/// `SQRTPS` or its 256-bit AVX form.
#[inline]
pub fn sqrt_f32<const C: usize>(value: &[f32; C], out: &mut [f32; C]) -> Option<()> {
    match C {
        4 => one_chunk!(
            xmm_reg,
            "movups {a}, [{value}]",
            "sqrtps {a}, {a}",
            "movups [{result}], {a}",
            value,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        8 => unsafe { wide::sqrt_f32(value, out) },
        _ => None,
    }
}

/// Computes the square roots of a chunk of binary64 lanes into `out`, by
/// `SQRTPD` or its 256-bit AVX form.
#[inline]
pub fn sqrt_f64<const C: usize>(value: &[f64; C], out: &mut [f64; C]) -> Option<()> {
    match C {
        2 => one_chunk!(
            xmm_reg,
            "movupd {a}, [{value}]",
            "sqrtpd {a}, {a}",
            "movupd [{result}], {a}",
            value,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        4 => unsafe { wide::sqrt_f64(value, out) },
        _ => None,
    }
}

/// Computes the lanes of a binary32 chunk rounded to integral values to
/// nearest even into `out`, by `ROUNDPS` with the rounding control 0 in its
/// immediate, or its 256-bit AVX form.
#[inline]
pub fn round_f32<const C: usize>(value: &[f32; C], out: &mut [f32; C]) -> Option<()> {
    match C {
        #[cfg(target_feature = "sse4.1")]
        4 => one_chunk!(
            xmm_reg,
            "movups {a}, [{value}]",
            "roundps {a}, {a}, 0",
            "movups [{result}], {a}",
            value,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        8 => unsafe { wide::round_f32(value, out) },
        _ => {
            let _ = (value, out);
            None
        }
    }
}

/// Computes the lanes of a binary64 chunk rounded to integral values to
/// nearest even into `out`, by `ROUNDPD` with the rounding control 0 in its
/// immediate, or its 256-bit AVX form.
#[inline]
pub fn round_f64<const C: usize>(value: &[f64; C], out: &mut [f64; C]) -> Option<()> {
    match C {
        #[cfg(target_feature = "sse4.1")]
        2 => one_chunk!(
            xmm_reg,
            "movupd {a}, [{value}]",
            "roundpd {a}, {a}, 0",
            "movupd [{result}], {a}",
            value,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        4 => unsafe { wide::round_f64(value, out) },
        _ => {
            let _ = (value, out);
            None
        }
    }
}

/// Computes `left * right + addend` of chunks of binary32 lanes into `out`,
/// each lane rounded once, by `VFMADD213PS`.
#[inline]
pub fn mul_add_f32<const C: usize>(
    left: &[f32; C],
    right: &[f32; C],
    addend: &[f32; C],
    out: &mut [f32; C],
) -> Option<()> {
    match C {
        #[cfg(target_feature = "fma")]
        4 => fused!(xmm_reg, "vmovups", "vfmadd213ps", left, right, addend, out),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "fma")]
        8 => unsafe { wide::mul_add_f32(left, right, addend, out) },
        _ => {
            let _ = (left, right, addend, out);
            None
        }
    }
}

/// Computes `left * right + addend` of chunks of binary64 lanes into `out`,
/// each lane rounded once, by `VFMADD213PD`.
#[inline]
pub fn mul_add_f64<const C: usize>(
    left: &[f64; C],
    right: &[f64; C],
    addend: &[f64; C],
    out: &mut [f64; C],
) -> Option<()> {
    match C {
        #[cfg(target_feature = "fma")]
        2 => fused!(xmm_reg, "vmovupd", "vfmadd213pd", left, right, addend, out),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "fma")]
        4 => unsafe { wide::mul_add_f64(left, right, addend, out) },
        _ => {
            let _ = (left, right, addend, out);
            None
        }
    }
}

/// Computes a chunk of binary32 lanes widened exactly to binary64 into
/// `out`, by `CVTPS2PD` for two lanes or its AVX form for four.
#[inline]
pub fn widen<const C: usize>(value: &[f32; C], out: &mut [f64; C]) -> Option<()> {
    match C {
        2 => one_chunk!(
            xmm_reg,
            "movsd {a}, qword ptr [{value}]",
            "cvtps2pd {a}, {a}",
            "movupd [{result}], {a}",
            value,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        4 => unsafe { wide::widen(value, out) },
        _ => None,
    }
}

/// Computes a chunk of binary64 lanes rounded to binary32 into `out`, by
/// `CVTPD2PS` for two lanes or its AVX form for four.
#[inline]
pub fn narrow<const C: usize>(value: &[f64; C], out: &mut [f32; C]) -> Option<()> {
    match C {
        2 => one_chunk!(
            xmm_reg,
            "movupd {a}, [{value}]",
            "cvtpd2ps {a}, {a}",
            "movsd qword ptr [{result}], {a}",
            value,
            out
        ),
        // SAFETY: the build enables the features of the 256-bit form, so
        // the processor that runs it has them.
        #[cfg(target_feature = "avx")]
        4 => unsafe { wide::narrow(value, out) },
        _ => None,
    }
}

/// The 256-bit forms, in functions that enable their features for their own
/// code. A caller compiled without AVX, such as a doctest, which does not
/// take `RUSTFLAGS`, then still compiles the 256-bit register class. Each
/// function inlines into a caller with the features.
#[cfg(target_feature = "avx")]
mod wide {
    use super::Operation;

    /// Computes `operation` of two chunks of binary32 lanes into `out`, by
    /// `VADDPS`, `VSUBPS`, `VMULPS`, or `VDIVPS`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn binary_f32<const C: usize>(
        left: &[f32; C],
        right: &[f32; C],
        out: &mut [f32; C],
        operation: Operation,
    ) -> Option<()> {
        arithmetic!(
            ymm_reg,
            "vmovups",
            operation,
            ["vaddps", "vsubps", "vmulps", "vdivps"],
            " {a}, {a}, {b}",
            left,
            right,
            out
        )
    }

    /// Computes `operation` of two chunks of binary64 lanes into `out`, by
    /// `VADDPD`, `VSUBPD`, `VMULPD`, or `VDIVPD`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn binary_f64<const C: usize>(
        left: &[f64; C],
        right: &[f64; C],
        out: &mut [f64; C],
        operation: Operation,
    ) -> Option<()> {
        arithmetic!(
            ymm_reg,
            "vmovupd",
            operation,
            ["vaddpd", "vsubpd", "vmulpd", "vdivpd"],
            " {a}, {a}, {b}",
            left,
            right,
            out
        )
    }

    /// Computes the square roots of a chunk of binary32 lanes into `out`, by
    /// `VSQRTPS`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn sqrt_f32<const C: usize>(value: &[f32; C], out: &mut [f32; C]) -> Option<()> {
        one_chunk!(
            ymm_reg,
            "vmovups {a}, [{value}]",
            "vsqrtps {a}, {a}",
            "vmovups [{result}], {a}",
            value,
            out
        )
    }

    /// Computes the square roots of a chunk of binary64 lanes into `out`, by
    /// `VSQRTPD`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn sqrt_f64<const C: usize>(value: &[f64; C], out: &mut [f64; C]) -> Option<()> {
        one_chunk!(
            ymm_reg,
            "vmovupd {a}, [{value}]",
            "vsqrtpd {a}, {a}",
            "vmovupd [{result}], {a}",
            value,
            out
        )
    }

    /// Computes the lanes of a binary32 chunk rounded to integral values to
    /// nearest even into `out`, by `VROUNDPS` with the immediate 0.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn round_f32<const C: usize>(value: &[f32; C], out: &mut [f32; C]) -> Option<()> {
        one_chunk!(
            ymm_reg,
            "vmovups {a}, [{value}]",
            "vroundps {a}, {a}, 0",
            "vmovups [{result}], {a}",
            value,
            out
        )
    }

    /// Computes the lanes of a binary64 chunk rounded to integral values to
    /// nearest even into `out`, by `VROUNDPD` with the immediate 0.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn round_f64<const C: usize>(value: &[f64; C], out: &mut [f64; C]) -> Option<()> {
        one_chunk!(
            ymm_reg,
            "vmovupd {a}, [{value}]",
            "vroundpd {a}, {a}, 0",
            "vmovupd [{result}], {a}",
            value,
            out
        )
    }

    /// Computes `left * right + addend` of chunks of binary32 lanes into
    /// `out`, each lane rounded once, by `VFMADD213PS`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX and FMA.
    #[cfg(target_feature = "fma")]
    #[target_feature(enable = "avx,fma")]
    #[inline]
    pub unsafe fn mul_add_f32<const C: usize>(
        left: &[f32; C],
        right: &[f32; C],
        addend: &[f32; C],
        out: &mut [f32; C],
    ) -> Option<()> {
        fused!(ymm_reg, "vmovups", "vfmadd213ps", left, right, addend, out)
    }

    /// Computes `left * right + addend` of chunks of binary64 lanes into
    /// `out`, each lane rounded once, by `VFMADD213PD`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX and FMA.
    #[cfg(target_feature = "fma")]
    #[target_feature(enable = "avx,fma")]
    #[inline]
    pub unsafe fn mul_add_f64<const C: usize>(
        left: &[f64; C],
        right: &[f64; C],
        addend: &[f64; C],
        out: &mut [f64; C],
    ) -> Option<()> {
        fused!(ymm_reg, "vmovupd", "vfmadd213pd", left, right, addend, out)
    }

    /// Computes a chunk of binary32 lanes widened exactly to binary64 into
    /// `out`, by `VCVTPS2PD`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn widen<const C: usize>(value: &[f32; C], out: &mut [f64; C]) -> Option<()> {
        one_chunk!(
            ymm_reg,
            "vmovups {a:x}, [{value}]",
            "vcvtps2pd {a}, {a:x}",
            "vmovupd [{result}], {a}",
            value,
            out
        )
    }

    /// Computes a chunk of binary64 lanes rounded to binary32 into `out`, by
    /// `VCVTPD2PS`.
    ///
    /// # Safety
    ///
    /// The processor must have AVX.
    #[target_feature(enable = "avx")]
    #[inline]
    pub unsafe fn narrow<const C: usize>(value: &[f64; C], out: &mut [f32; C]) -> Option<()> {
        one_chunk!(
            ymm_reg,
            "vmovupd {a}, [{value}]",
            "vcvtpd2ps {a:x}, {a}",
            "vmovups [{result}], {a:x}",
            value,
            out
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::{binary_f32, binary_f64, narrow_double, sqrt_f32, sqrt_f64, widen_single};
    use super::Operation;

    /// Storage aligned to 32 bytes, so that the chunk at an offset of one lane
    /// has an address that is not a multiple of 16.
    #[repr(align(32))]
    struct Aligned<T>(T);

    /// Returns the chunk of `C` lanes at lane 1 of `lanes`.
    fn unaligned<T, const C: usize>(lanes: &[T]) -> &[T; C] {
        lanes[1..=C]
            .try_into()
            .expect("the storage holds a chunk after lane 0")
    }

    /// Returns the mutable chunk of `C` lanes at lane 1 of `lanes`.
    fn unaligned_mut<T, const C: usize>(lanes: &mut [T]) -> &mut [T; C] {
        (&mut lanes[1..=C])
            .try_into()
            .expect("the storage holds a chunk after lane 0")
    }

    #[test]
    fn every_chunk_reads_and_writes_through_an_unaligned_address() {
        let singles = Aligned(
            [
                0x3FC0_0000_u32,
                0x4010_0000,
                0x4080_0000,
                0x4110_0000,
                0x4180_0000,
            ]
            .map(f32::from_bits),
        );
        let doubles = Aligned(
            [
                0x4000_0000_0000_0000_u64,
                0x3FD0_0000_0000_0000,
                0x4022_0000_0000_0000,
            ]
            .map(f64::from_bits),
        );
        let (mut single_out, mut double_out) = (Aligned([0.0_f32; 5]), Aligned([0.0_f64; 3]));
        let four: &[f32; 4] = unaligned(&singles.0);
        let two: &[f64; 2] = unaligned(&doubles.0);
        let pair: &[f32; 2] = unaligned(&singles.0);

        let out: &mut [f32; 4] = unaligned_mut(&mut single_out.0);
        super::binary_f32(four, four, out, Operation::Mul).expect("SSE2 has MULPS");
        let each = four.map(|lane| binary_f32(lane, lane, Operation::Mul));
        assert_eq!(out.map(f32::to_bits), each.map(f32::to_bits), "MULPS");
        super::sqrt_f32(four, out).expect("SSE2 has SQRTPS");
        assert_eq!(
            out.map(f32::to_bits),
            four.map(sqrt_f32).map(f32::to_bits),
            "SQRTPS"
        );
        if super::round_f32(four, out).is_some() {
            let each = *out;
            super::round_f32(&each, out).expect("the build has ROUNDPS");
            assert_eq!(out.map(f32::to_bits), each.map(f32::to_bits), "ROUNDPS");
        }

        let out: &mut [f64; 2] = unaligned_mut(&mut double_out.0);
        super::binary_f64(two, two, out, Operation::Div).expect("SSE2 has DIVPD");
        let each = two.map(|lane| binary_f64(lane, lane, Operation::Div));
        assert_eq!(out.map(f64::to_bits), each.map(f64::to_bits), "DIVPD");
        super::sqrt_f64(two, out).expect("SSE2 has SQRTPD");
        assert_eq!(
            out.map(f64::to_bits),
            two.map(sqrt_f64).map(f64::to_bits),
            "SQRTPD"
        );
        if super::round_f64(two, out).is_some() {
            let each = *out;
            super::round_f64(&each, out).expect("the build has ROUNDPD");
            assert_eq!(out.map(f64::to_bits), each.map(f64::to_bits), "ROUNDPD");
        }
        super::widen(pair, out).expect("SSE2 has CVTPS2PD");
        assert_eq!(
            out.map(f64::to_bits),
            pair.map(widen_single).map(f64::to_bits),
            "CVTPS2PD"
        );

        let out: &mut [f32; 2] = unaligned_mut(&mut single_out.0);
        super::narrow(two, out).expect("SSE2 has CVTPD2PS");
        assert_eq!(
            out.map(f32::to_bits),
            two.map(narrow_double).map(f32::to_bits),
            "CVTPD2PS"
        );
    }
}
