//! Runs SSE, SSE2, SSE4.1, and FMA instructions under an MXCSR value.

use core::arch::asm;
use core::cmp::Ordering;

use super::{CompareFlags, MXCSR_FLAGS};

/// Defines a function that runs a two-operand SSE instruction on 32-bit or
/// 64-bit values under an MXCSR value, and returns the result and the flags.
macro_rules! sse_binary {
    ($(#[$doc:meta])* $name:ident, $bits:ty, $load_a:literal, $load_b:literal, $operation:literal, $store:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: $bits, b: $bits, control: u32) -> ($bits, u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let result: $bits;
            // SAFETY: the code saves MXCSR, runs one instruction under
            // `control`, reads the flags, and restores MXCSR. It reads and
            // writes only the local variables.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    $load_a,
                    $load_b,
                    $operation,
                    $store,
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    a = in(reg) a,
                    b = in(reg) b,
                    x = out(xmm_reg) _,
                    y = out(xmm_reg) _,
                    result = out(reg) result,
                    options(nostack),
                );
            }
            (result, after & MXCSR_FLAGS)
        }
    };
}

sse_binary!(
    /// `ADDSS`.
    addss, u32, "movd {x}, {a:e}", "movd {y}, {b:e}", "addss {x}, {y}", "movd {result:e}, {x}"
);
sse_binary!(
    /// `SUBSS`.
    subss, u32, "movd {x}, {a:e}", "movd {y}, {b:e}", "subss {x}, {y}", "movd {result:e}, {x}"
);
sse_binary!(
    /// `MULSS`.
    mulss, u32, "movd {x}, {a:e}", "movd {y}, {b:e}", "mulss {x}, {y}", "movd {result:e}, {x}"
);
sse_binary!(
    /// `DIVSS`.
    divss, u32, "movd {x}, {a:e}", "movd {y}, {b:e}", "divss {x}, {y}", "movd {result:e}, {x}"
);
sse_binary!(
    /// `SQRTSS` of `a`; `b` is unused.
    sqrtss, u32, "movd {x}, {a:e}", "movd {y}, {b:e}", "sqrtss {x}, {x}", "movd {result:e}, {x}"
);
sse_binary!(
    /// `CVTSD2SS` of `a`; `b` is unused. The result is in the low 32 bits.
    cvtsd2ss, u64, "movq {x}, {a}", "movq {y}, {b}", "cvtsd2ss {x}, {x}", "movq {result}, {x}"
);
sse_binary!(
    /// `CVTSS2SD` of the low 32 bits of `a`; `b` is unused.
    cvtss2sd, u64, "movq {x}, {a}", "movq {y}, {b}", "cvtss2sd {x}, {x}", "movq {result}, {x}"
);
sse_binary!(
    /// `ADDSD`.
    addsd, u64, "movq {x}, {a}", "movq {y}, {b}", "addsd {x}, {y}", "movq {result}, {x}"
);
sse_binary!(
    /// `SUBSD`.
    subsd, u64, "movq {x}, {a}", "movq {y}, {b}", "subsd {x}, {y}", "movq {result}, {x}"
);
sse_binary!(
    /// `MULSD`.
    mulsd, u64, "movq {x}, {a}", "movq {y}, {b}", "mulsd {x}, {y}", "movq {result}, {x}"
);
sse_binary!(
    /// `DIVSD`.
    divsd, u64, "movq {x}, {a}", "movq {y}, {b}", "divsd {x}, {y}", "movq {result}, {x}"
);
sse_binary!(
    /// `SQRTSD` of `a`; `b` is unused.
    sqrtsd, u64, "movq {x}, {a}", "movq {y}, {b}", "sqrtsd {x}, {x}", "movq {result}, {x}"
);

/// Defines a function that runs a three-operand FMA instruction.
macro_rules! fma {
    ($(#[$doc:meta])* $name:ident, $bits:ty, $load:literal, $operation:literal, $store:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: $bits, b: $bits, c: $bits, control: u32) -> ($bits, u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let result: $bits;
            let operands = [a, b, c];
            // SAFETY: as in the two-operand functions. The code also reads the
            // three operands from `operands`.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    $load,
                    $operation,
                    $store,
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    operands = in(reg) operands.as_ptr(),
                    x = out(xmm_reg) _,
                    y = out(xmm_reg) _,
                    z = out(xmm_reg) _,
                    result = out(reg) result,
                    options(nostack),
                );
            }
            (result, after & MXCSR_FLAGS)
        }
    };
}

fma!(
    /// `VFMADD213SS x, y, z` with `x = a`, `y = b`, and `z = c`: `b * a + c`.
    vfmadd213ss,
    u32,
    "vmovss {x}, dword ptr [{operands}]\n vmovss {y}, dword ptr [{operands} + 4]\n vmovss {z}, dword ptr [{operands} + 8]",
    "vfmadd213ss {x}, {y}, {z}",
    "vmovd {result:e}, {x}"
);
fma!(
    /// `VFMADD213SD x, y, z` with `x = a`, `y = b`, and `z = c`: `b * a + c`.
    vfmadd213sd,
    u64,
    "vmovsd {x}, qword ptr [{operands}]\n vmovsd {y}, qword ptr [{operands} + 8]\n vmovsd {z}, qword ptr [{operands} + 16]",
    "vfmadd213sd {x}, {y}, {z}",
    "vmovq {result}, {x}"
);

/// Defines a function that runs a compare instruction on 32-bit or 64-bit
/// values under an MXCSR value. Returns the order of `a` to `b` that the
/// instruction reports in ZF, PF, and CF, and the flags.
macro_rules! sse_compare {
    ($(#[$doc:meta])* $name:ident, $bits:ty, $load:literal, $compare:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: $bits, b: $bits, control: u32) -> (Option<Ordering>, u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let (zero, parity, carry): (u8, u8, u8);
            // SAFETY: as in the two-operand functions. The code also copies
            // ZF, PF, and CF to three byte registers directly after the
            // compare, before an instruction can change them.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    $load,
                    $compare,
                    "setz {zero}",
                    "setp {parity}",
                    "setc {carry}",
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    a = in(reg) a,
                    b = in(reg) b,
                    x = out(xmm_reg) _,
                    y = out(xmm_reg) _,
                    zero = out(reg_byte) zero,
                    parity = out(reg_byte) parity,
                    carry = out(reg_byte) carry,
                    options(nostack),
                );
            }
            let flags = CompareFlags {
                zero: zero == 1,
                parity: parity == 1,
                carry: carry == 1,
            };
            (flags.order(), after & MXCSR_FLAGS)
        }
    };
}

sse_compare!(
    /// `UCOMISS a, b`.
    ucomiss, u32, "movd {x}, {a:e}\n movd {y}, {b:e}", "ucomiss {x}, {y}"
);
sse_compare!(
    /// `COMISS a, b`.
    comiss, u32, "movd {x}, {a:e}\n movd {y}, {b:e}", "comiss {x}, {y}"
);
sse_compare!(
    /// `UCOMISD a, b`.
    ucomisd, u64, "movq {x}, {a}\n movq {y}, {b}", "ucomisd {x}, {y}"
);
sse_compare!(
    /// `COMISD a, b`.
    comisd, u64, "movq {x}, {a}\n movq {y}, {b}", "comisd {x}, {y}"
);

/// Defines a function that runs a conversion between a float and an integer
/// register under an MXCSR value. Returns the result and the flags.
macro_rules! sse_convert {
    ($(#[$doc:meta])* $name:ident, $source:ty, $result:ty, $operation:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: $source, control: u32) -> ($result, u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let result: $result;
            // SAFETY: as in the two-operand functions.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    $operation,
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    a = in(reg) a,
                    x = out(xmm_reg) _,
                    result = out(reg) result,
                    options(nostack),
                );
            }
            (result, after & MXCSR_FLAGS)
        }
    };
}

sse_convert!(
    /// `CVTSS2SI r32`: binary32 to a 32-bit integer, rounded as MXCSR selects.
    cvtss2si_r32, u32, i32, "movd {x}, {a:e}\n cvtss2si {result:e}, {x}"
);
sse_convert!(
    /// `CVTSS2SI r64`: binary32 to a 64-bit integer, rounded as MXCSR selects.
    cvtss2si_r64, u32, i64, "movd {x}, {a:e}\n cvtss2si {result}, {x}"
);
sse_convert!(
    /// `CVTTSS2SI r32`: binary32 to a 32-bit integer, truncated.
    cvttss2si_r32, u32, i32, "movd {x}, {a:e}\n cvttss2si {result:e}, {x}"
);
sse_convert!(
    /// `CVTTSS2SI r64`: binary32 to a 64-bit integer, truncated.
    cvttss2si_r64, u32, i64, "movd {x}, {a:e}\n cvttss2si {result}, {x}"
);
sse_convert!(
    /// `CVTSD2SI r32`: binary64 to a 32-bit integer, rounded as MXCSR selects.
    cvtsd2si_r32, u64, i32, "movq {x}, {a}\n cvtsd2si {result:e}, {x}"
);
sse_convert!(
    /// `CVTSD2SI r64`: binary64 to a 64-bit integer, rounded as MXCSR selects.
    cvtsd2si_r64, u64, i64, "movq {x}, {a}\n cvtsd2si {result}, {x}"
);
sse_convert!(
    /// `CVTTSD2SI r32`: binary64 to a 32-bit integer, truncated.
    cvttsd2si_r32, u64, i32, "movq {x}, {a}\n cvttsd2si {result:e}, {x}"
);
sse_convert!(
    /// `CVTTSD2SI r64`: binary64 to a 64-bit integer, truncated.
    cvttsd2si_r64, u64, i64, "movq {x}, {a}\n cvttsd2si {result}, {x}"
);
sse_convert!(
    /// `CVTSI2SS` from a 32-bit integer to binary32.
    cvtsi2ss_r32, i32, u32, "cvtsi2ss {x}, {a:e}\n movd {result:e}, {x}"
);
sse_convert!(
    /// `CVTSI2SS` from a 64-bit integer to binary32.
    cvtsi2ss_r64, i64, u32, "cvtsi2ss {x}, {a}\n movd {result:e}, {x}"
);
sse_convert!(
    /// `CVTSI2SD` from a 32-bit integer to binary64.
    cvtsi2sd_r32, i32, u64, "cvtsi2sd {x}, {a:e}\n movq {result}, {x}"
);
sse_convert!(
    /// `CVTSI2SD` from a 64-bit integer to binary64.
    cvtsi2sd_r64, i64, u64, "cvtsi2sd {x}, {a}\n movq {result}, {x}"
);

/// Defines a function that runs an SSE4.1 round instruction with the
/// immediate `IMMEDIATE` under an MXCSR value. Returns the result and the
/// flags.
///
/// Bits 1 and 0 of the immediate select the rounding direction, in the order
/// of the MXCSR rounding-control field. Bit 2 selects the MXCSR direction
/// instead. Bit 3 suppresses the precision exception.
macro_rules! sse_round {
    ($(#[$doc:meta])* $name:ident, $bits:ty, $operation:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name<const IMMEDIATE: u8>(a: $bits, control: u32) -> ($bits, u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let result: $bits;
            // SAFETY: as in the two-operand functions.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    $operation,
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    a = in(reg) a,
                    x = out(xmm_reg) _,
                    result = out(reg) result,
                    immediate = const IMMEDIATE,
                    options(nostack),
                );
            }
            (result, after & MXCSR_FLAGS)
        }
    };
}

sse_round!(
    /// `ROUNDSS` with the immediate `IMMEDIATE`.
    roundss, u32, "movd {x}, {a:e}\n roundss {x}, {x}, {immediate}\n movd {result:e}, {x}"
);
sse_round!(
    /// `ROUNDSD` with the immediate `IMMEDIATE`.
    roundsd, u64, "movq {x}, {a}\n roundsd {x}, {x}, {immediate}\n movq {result}, {x}"
);
