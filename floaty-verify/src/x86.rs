//! Runs SSE and x87 instructions on the host processor, as an oracle.
//!
//! Each function saves the MXCSR register or the x87 control word, runs one
//! instruction under the requested control value, reads the flags, and
//! restores the saved value. The x87 functions leave the x87 stack empty, as
//! the ABI requires.

use core::arch::asm;

use floaty::env::{InvalidProduct, NanPropagation, NanRule, Tininess};
use floaty::{Env, Flags, Rounding};

/// The MXCSR value with every exception masked and no flag set.
pub const MXCSR_MASKED: u32 = 0x1F80;
/// The MXCSR flag bits: IE, DE, ZE, OE, UE, and PE.
pub const MXCSR_FLAGS: u32 = 0x3F;
/// The MXCSR denormals-are-zero bit.
pub const MXCSR_DAZ: u32 = 1 << 6;
/// The MXCSR flush-to-zero bit.
pub const MXCSR_FTZ: u32 = 1 << 15;

/// The rounding directions and their MXCSR rounding-control field.
pub const MXCSR_ROUNDINGS: [(Rounding, u32); 4] = [
    (Rounding::NearestEven, 0),
    (Rounding::TowardNegative, 1 << 13),
    (Rounding::TowardPositive, 2 << 13),
    (Rounding::TowardZero, 3 << 13),
];

/// The rounding directions and their x87 rounding-control field.
pub const X87_ROUNDINGS: [(Rounding, u16); 4] = [
    (Rounding::NearestEven, 0),
    (Rounding::TowardNegative, 1 << 10),
    (Rounding::TowardPositive, 2 << 10),
    (Rounding::TowardZero, 3 << 10),
];

/// The x87 precision-control field: 24, 53, and 64 bits.
pub const X87_PRECISIONS: [(u32, u16); 3] = [(24, 0), (53, 2 << 8), (64, 3 << 8)];

/// The x87 control word with every exception masked, before the rounding and
/// precision fields.
pub const X87_MASKED: u16 = 0x007F;

/// Returns the MXCSR flags that floaty flags report.
///
/// MXCSR sets DE only when DAZ is clear, and only when no higher-priority
/// condition occurs: a NaN operand, an invalid operation, or a division by
/// zero, as the Intel SDM Volume 1, section 4.9.2, orders them.
/// `nan_operand` says that an operand is a NaN.
#[must_use]
pub fn mxcsr_flags(flags: Flags, daz: bool, nan_operand: bool) -> u32 {
    let mut bits = 0;
    for (flag, bit) in [
        (Flags::INVALID, 0),
        (Flags::DIVIDE_BY_ZERO, 2),
        (Flags::OVERFLOW, 3),
        (Flags::UNDERFLOW, 4),
        (Flags::INEXACT, 5),
    ] {
        if flags.contains(flag) {
            bits |= 1 << bit;
        }
    }
    let higher =
        nan_operand || flags.contains(Flags::INVALID) || flags.contains(Flags::DIVIDE_BY_ZERO);
    if flags.contains(Flags::DENORMAL_INPUT) && !daz && !higher {
        bits |= 1 << 1;
    }
    bits
}

/// Returns the behavior of the SSE unit under an MXCSR setting: the first NaN
/// operand, a negative default NaN, and tininess after rounding.
#[must_use]
pub fn sse_env(rounding: Rounding, ftz: bool, daz: bool) -> Env {
    Env::IEEE
        .with_rounding(rounding)
        .with_flush_to_zero(ftz)
        .with_denormals_are_zero(daz)
        .with_tininess(Tininess::AfterRounding)
        .with_nan(
            NanRule::new(NanPropagation::FirstOperand)
                .with_default_negative(true)
                .with_invalid_product(InvalidProduct::YieldsToNan),
        )
}

/// Returns the behavior of the x87 unit: the larger-significand NaN rule, a
/// negative default NaN, and tininess after rounding.
#[must_use]
pub fn x87_env(rounding: Rounding) -> Env {
    Env::IEEE
        .with_rounding(rounding)
        .with_tininess(Tininess::AfterRounding)
        .with_nan(NanRule::new(NanPropagation::LargerSignificand).with_default_negative(true))
}

/// Returns the x87 status bits that floaty flags report: IE, DE, ZE, OE, UE,
/// PE, and C1 for a rounding that grew the magnitude.
#[must_use]
pub fn x87_status(flags: Flags) -> u16 {
    let mut bits = 0;
    for (flag, bit) in [
        (Flags::INVALID, 0),
        (Flags::DENORMAL_INPUT, 1),
        (Flags::DIVIDE_BY_ZERO, 2),
        (Flags::OVERFLOW, 3),
        (Flags::UNDERFLOW, 4),
        (Flags::INEXACT, 5),
        (Flags::ROUNDED_UP, 9),
    ] {
        if flags.contains(flag) {
            bits |= 1 << bit;
        }
    }
    bits
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

/// Runs `FXAM` on an 80-bit encoding. Returns the status word.
#[must_use]
pub fn fxam(bits: u128) -> u16 {
    let bytes = bits.to_le_bytes();
    let status: u16;
    // SAFETY: the code reads 10 bytes from `bytes`, which holds 16. It pushes
    // one value onto the x87 stack and pops it, so the stack is empty on exit.
    // `FLD` of an 80-bit operand raises no exception.
    unsafe {
        asm!(
            "fld tbyte ptr [{source}]",
            "fxam",
            "fnstsw ax",
            "fstp st(0)",
            source = in(reg) bytes.as_ptr(),
            out("ax") status,
            out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
            options(nostack, readonly),
        );
    }
    status
}

/// Reads the x87 control word.
#[must_use]
pub fn control_word() -> u16 {
    let mut word = 0_u16;
    // SAFETY: `FNSTCW` writes two bytes to `word` and changes no other state.
    unsafe {
        asm!("fnstcw word ptr [{word}]", word = in(reg) &raw mut word, options(nostack));
    }
    word
}

/// Defines a function that loads x87 operands, runs an operation under a
/// control word, and stores the result. Returns the result bits and the status
/// word that the operation leaves.
///
/// The status word is read before the store: an exact `FSTP` clears C1, the
/// round-up bit that the operation sets.
macro_rules! x87 {
    ($(#[$doc:meta])* $name:ident, ($($operand:ident),+), $load:literal, $operation:literal, $store:literal, $result:ty) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name($($operand: u128,)+ control: u16) -> ($result, u16) {
            let operands = [$($operand.to_le_bytes()),+];
            let mut output = [0_u8; 16];
            let mut saved = 0_u16;
            let status: u16;
            // SAFETY: the code saves the control word, loads `control`,
            // clears the flags, loads each operand from its 16-byte slot, runs
            // the operation, reads the status word, stores and pops the
            // result, clears the flags again, and restores the control word.
            // The x87 stack is empty on exit.
            unsafe {
                asm!(
                    "fnstcw word ptr [{saved}]",
                    "fldcw word ptr [{control}]",
                    "fnclex",
                    $load,
                    $operation,
                    "fnstsw ax",
                    $store,
                    "fnclex",
                    "fldcw word ptr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    operands = in(reg) operands.as_ptr(),
                    target = in(reg) output.as_mut_ptr(),
                    out("ax") status,
                    out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                    out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                    options(nostack),
                );
            }
            let result = <$result>::from_le_bytes(output[..core::mem::size_of::<$result>()].try_into().expect("the output holds the result"));
            (result, status)
        }
    };
}

x87!(
    /// `FLD` of an 80-bit value and `FSTP` to 64 bits.
    store_double, (a), "fld tbyte ptr [{operands}]", "fstp qword ptr [{target}]", "", u64
);
x87!(
    /// `FLD` of an 80-bit value and `FSTP` to 32 bits.
    store_single, (a), "fld tbyte ptr [{operands}]", "fstp dword ptr [{target}]", "", u32
);
x87!(
    /// `a + b` with `FADDP`.
    fadd, (a, b), "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]", "faddp st(1), st", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// `a - b` with `FSUBP`.
    fsub, (a, b), "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]", "fsubp st(1), st", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// `a * b` with `FMULP`.
    fmul, (a, b), "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]", "fmulp st(1), st", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// `a / b` with `FDIVP`.
    fdiv, (a, b), "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]", "fdivp st(1), st", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// The square root of `a` with `FSQRT`.
    fsqrt, (a), "fld tbyte ptr [{operands}]", "fsqrt", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// `a * 1.0` with `FMULP`, which stores a pseudo-denormal as a normal
    /// encoding.
    times_one, (a), "fld tbyte ptr [{operands}]\n fld1", "fmulp st(1), st", "fstp tbyte ptr [{target}]", u128
);
