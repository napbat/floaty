//! Runs SSE, SSE2, SSE4.1, and FMA instructions under an MXCSR value.

use core::arch::asm;
use core::cmp::Ordering;

use super::{CompareFlags, MXCSR_FLAGS};

/// Reads MXCSR.
#[must_use]
pub fn mxcsr() -> u32 {
    let mut value = 0_u32;
    // SAFETY: `STMXCSR` writes four bytes to `value` and changes no other
    // state.
    unsafe {
        asm!("stmxcsr [{value}]", value = in(reg) &raw mut value, options(nostack));
    }
    value
}

/// Runs `body` with MXCSR set to `control`, and restores MXCSR afterward.
///
/// Rust assumes the default floating-point environment, so `body` must not
/// depend on its own host floating-point arithmetic. The tests use it to show
/// that floaty reads MXCSR before its host fast path.
pub fn with_mxcsr<T>(control: u32, body: impl FnOnce() -> T) -> T {
    let mut saved = 0_u32;
    // SAFETY: the code saves MXCSR to `saved` and loads `control`. It reads
    // and writes only those variables.
    unsafe {
        asm!(
            "stmxcsr [{saved}]",
            "ldmxcsr [{control}]",
            saved = in(reg) &raw mut saved,
            control = in(reg) &raw const control,
            options(nostack),
        );
    }
    let result = body();
    // SAFETY: the code loads the saved MXCSR value.
    unsafe {
        asm!("ldmxcsr [{saved}]", saved = in(reg) &raw const saved, options(nostack));
    }
    result
}

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

/// Defines a function that runs a packed SSE instruction on two 128-bit
/// chunks under an MXCSR value, and returns the result chunk and the flags.
/// A one-operand instruction reads `x` and ignores `b`.
macro_rules! sse_packed {
    ($(#[$doc:meta])* $name:ident, $lane:ty, $lanes:literal, $operation:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: [$lane; $lanes], b: [$lane; $lanes], control: u32) -> ([$lane; $lanes], u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let mut result = [0; $lanes];
            // SAFETY: the code saves MXCSR, loads both chunks from their
            // arrays, runs one instruction under `control`, stores the result
            // chunk into `result`, reads the flags, and restores MXCSR. Each
            // array holds 16 bytes.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    "movups {x}, [{a}]",
                    "movups {y}, [{b}]",
                    $operation,
                    "movups [{result}], {x}",
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    a = in(reg) a.as_ptr(),
                    b = in(reg) b.as_ptr(),
                    result = in(reg) result.as_mut_ptr(),
                    x = out(xmm_reg) _,
                    y = out(xmm_reg) _,
                    options(nostack),
                );
            }
            (result, after & MXCSR_FLAGS)
        }
    };
}

sse_packed!(
    /// `ADDPS`.
    addps, u32, 4, "addps {x}, {y}"
);
sse_packed!(
    /// `SUBPS`.
    subps, u32, 4, "subps {x}, {y}"
);
sse_packed!(
    /// `MULPS`.
    mulps, u32, 4, "mulps {x}, {y}"
);
sse_packed!(
    /// `DIVPS`.
    divps, u32, 4, "divps {x}, {y}"
);
sse_packed!(
    /// `SQRTPS` of `a`; `b` is unused.
    sqrtps, u32, 4, "sqrtps {x}, {x}"
);
sse_packed!(
    /// `ROUNDPS` of `a` in the rounding direction of MXCSR, immediate 4; `b`
    /// is unused.
    roundps, u32, 4, "roundps {x}, {x}, 4"
);
sse_packed!(
    /// `ADDPD`.
    addpd, u64, 2, "addpd {x}, {y}"
);
sse_packed!(
    /// `SUBPD`.
    subpd, u64, 2, "subpd {x}, {y}"
);
sse_packed!(
    /// `MULPD`.
    mulpd, u64, 2, "mulpd {x}, {y}"
);
sse_packed!(
    /// `DIVPD`.
    divpd, u64, 2, "divpd {x}, {y}"
);
sse_packed!(
    /// `SQRTPD` of `a`; `b` is unused.
    sqrtpd, u64, 2, "sqrtpd {x}, {x}"
);
sse_packed!(
    /// `ROUNDPD` of `a` in the rounding direction of MXCSR, immediate 4; `b`
    /// is unused.
    roundpd, u64, 2, "roundpd {x}, {x}, 4"
);
sse_packed!(
    /// `CVTPS2PD` of the two low binary32 lanes of `a` into two binary64
    /// lanes, returned as four 32-bit halves, low half first; `b` is unused.
    cvtps2pd, u32, 4, "cvtps2pd {x}, {x}"
);
sse_packed!(
    /// `CVTPD2PS` of the two binary64 lanes of `a`, given as four 32-bit
    /// halves, into the two low binary32 lanes of the result; `b` is unused.
    cvtpd2ps, u32, 4, "cvtpd2ps {x}, {x}"
);

/// Defines a function that runs a packed three-operand FMA instruction on
/// 128-bit chunks under an MXCSR value, and returns the result chunk and the
/// flags.
macro_rules! fma_packed {
    ($(#[$doc:meta])* $name:ident, $lane:ty, $lanes:literal, $operation:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(
            a: [$lane; $lanes],
            b: [$lane; $lanes],
            c: [$lane; $lanes],
            control: u32,
        ) -> ([$lane; $lanes], u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let mut result = [0; $lanes];
            // SAFETY: as in `sse_packed!`, with a third chunk. The processor
            // of the test host has FMA.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    "vmovups {x}, [{a}]",
                    "vmovups {y}, [{b}]",
                    "vmovups {z}, [{c}]",
                    $operation,
                    "vmovups [{result}], {x}",
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    a = in(reg) a.as_ptr(),
                    b = in(reg) b.as_ptr(),
                    c = in(reg) c.as_ptr(),
                    result = in(reg) result.as_mut_ptr(),
                    x = out(xmm_reg) _,
                    y = out(xmm_reg) _,
                    z = out(xmm_reg) _,
                    options(nostack),
                );
            }
            (result, after & MXCSR_FLAGS)
        }
    };
}

fma_packed!(
    /// `VFMADD213PS`: `a * b + c` in each lane.
    vfmadd213ps, u32, 4, "vfmadd213ps {x}, {y}, {z}"
);
fma_packed!(
    /// `VFMADD213PD`: `a * b + c` in each lane.
    vfmadd213pd, u64, 2, "vfmadd213pd {x}, {y}, {z}"
);
