//! Runs x87 instructions under an x87 control word.

use core::arch::asm;
use core::cmp::Ordering;

use super::CompareFlags;

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

/// Runs `body` with the x87 control word set to `control`, and restores the
/// control word afterward.
///
/// `FNCLEX` clears the exception flags before each load, so a flag that an
/// earlier instruction set cannot trap under a control word that unmasks it.
/// The tests use the function to show that floaty reads the control word
/// before its host paths on the x87 unit.
pub fn with_control_word<T>(control: u16, body: impl FnOnce() -> T) -> T {
    let saved = control_word();
    // SAFETY: `FNCLEX` clears the exception flags, and `FLDCW` loads the
    // control word from `control`. Neither uses the x87 stack.
    unsafe {
        asm!(
            "fnclex",
            "fldcw word ptr [{control}]",
            control = in(reg) &raw const control,
            options(nostack),
        );
    }
    let result = body();
    // SAFETY: as above, with the saved control word.
    unsafe {
        asm!(
            "fnclex",
            "fldcw word ptr [{saved}]",
            saved = in(reg) &raw const saved,
            options(nostack),
        );
    }
    result
}

/// Returns a runner that runs a body with the x87 control word set to
/// `control`, by [`with_control_word`], for the `assert_*_under` functions of
/// [`crate::entry_points`].
pub fn under_control_word<T>(control: u16) -> impl FnOnce(&dyn Fn() -> T) -> T {
    move |body| with_control_word(control, body)
}

/// Runs `FNINIT` and returns the control word that it sets. `FNINIT` also
/// clears the status word and empties the x87 stack. The previous control
/// word is restored afterward.
#[must_use]
pub fn initialized_control_word() -> u16 {
    let (mut saved, mut word) = (0_u16, 0_u16);
    // SAFETY: `FNINIT` resets the control, status, and tag words and empties
    // the x87 stack. The x87 stack is empty outside the harness functions, so
    // no value is lost, and the code restores the control word.
    unsafe {
        asm!(
            "fnstcw word ptr [{saved}]",
            "fninit",
            "fnstcw word ptr [{word}]",
            "fldcw word ptr [{saved}]",
            saved = in(reg) &raw mut saved,
            word = in(reg) &raw mut word,
            out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
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
x87!(
    /// `FRNDINT`.
    frndint, (a), "fld tbyte ptr [{operands}]", "frndint", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// `FCHS`.
    fchs, (a), "fld tbyte ptr [{operands}]", "fchs", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// `FABS`.
    fabs, (a), "fld tbyte ptr [{operands}]", "fabs", "fstp tbyte ptr [{target}]", u128
);
x87!(
    /// `FISTP m16int`, rounded as the control word selects.
    fistp_m16, (a), "fld tbyte ptr [{operands}]", "fistp word ptr [{target}]", "", i16
);
x87!(
    /// `FISTP m32int`, rounded as the control word selects.
    fistp_m32, (a), "fld tbyte ptr [{operands}]", "fistp dword ptr [{target}]", "", i32
);
x87!(
    /// `FISTP m64int`, rounded as the control word selects.
    fistp_m64, (a), "fld tbyte ptr [{operands}]", "fistp qword ptr [{target}]", "", i64
);
x87!(
    /// `FISTTP m16int`, truncated.
    fisttp_m16, (a), "fld tbyte ptr [{operands}]", "fisttp word ptr [{target}]", "", i16
);
x87!(
    /// `FISTTP m32int`, truncated.
    fisttp_m32, (a), "fld tbyte ptr [{operands}]", "fisttp dword ptr [{target}]", "", i32
);
x87!(
    /// `FISTTP m64int`, truncated.
    fisttp_m64, (a), "fld tbyte ptr [{operands}]", "fisttp qword ptr [{target}]", "", i64
);

/// Defines a function that compares `a` in ST(0) with `b` in ST(1) under a
/// control word, with an instruction that sets ZF, PF, and CF and pops ST(0).
/// Returns the order of `a` to `b` and the status word.
macro_rules! x87_compare_eflags {
    ($(#[$doc:meta])* $name:ident, $compare:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: u128, b: u128, control: u16) -> (Option<Ordering>, u16) {
            let operands = [a.to_le_bytes(), b.to_le_bytes()];
            let mut saved = 0_u16;
            let status: u16;
            let (zero, parity, carry): (u8, u8, u8);
            // SAFETY: as in `x87!`. The code copies ZF, PF, and CF to three
            // byte registers directly after the compare. The compare pops one
            // register and `FSTP` pops the other, so the stack is empty on
            // exit.
            unsafe {
                asm!(
                    "fnstcw word ptr [{saved}]",
                    "fldcw word ptr [{control}]",
                    "fnclex",
                    "fld tbyte ptr [{operands} + 16]",
                    "fld tbyte ptr [{operands}]",
                    $compare,
                    "setz {zero}",
                    "setp {parity}",
                    "setc {carry}",
                    "fnstsw ax",
                    "fstp st(0)",
                    "fnclex",
                    "fldcw word ptr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    operands = in(reg) operands.as_ptr(),
                    zero = out(reg_byte) zero,
                    parity = out(reg_byte) parity,
                    carry = out(reg_byte) carry,
                    out("ax") status,
                    out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                    out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                    options(nostack),
                );
            }
            let flags = CompareFlags {
                zero: zero == 1,
                parity: parity == 1,
                carry: carry == 1,
            };
            (flags.order(), status)
        }
    };
}

x87_compare_eflags!(
    /// `FUCOMIP`: the quiet compare that sets EFLAGS.
    fucomip, "fucomip st, st(1)"
);
x87_compare_eflags!(
    /// `FCOMIP`: the signaling compare that sets EFLAGS.
    fcomip, "fcomip st, st(1)"
);

/// Defines a function that compares `a` in ST(0) with `b` in ST(1) under a
/// control word, with an instruction that sets C3, C2, and C0 and pops both
/// registers. Returns the order of `a` to `b` and the status word.
macro_rules! x87_compare_status {
    ($(#[$doc:meta])* $name:ident, $compare:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: u128, b: u128, control: u16) -> (Option<Ordering>, u16) {
            let operands = [a.to_le_bytes(), b.to_le_bytes()];
            let mut saved = 0_u16;
            let status: u16;
            // SAFETY: as in `x87!`. The compare pops both registers, so the
            // stack is empty on exit.
            unsafe {
                asm!(
                    "fnstcw word ptr [{saved}]",
                    "fldcw word ptr [{control}]",
                    "fnclex",
                    "fld tbyte ptr [{operands} + 16]",
                    "fld tbyte ptr [{operands}]",
                    $compare,
                    "fnstsw ax",
                    "fnclex",
                    "fldcw word ptr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    operands = in(reg) operands.as_ptr(),
                    out("ax") status,
                    out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                    out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                    options(nostack),
                );
            }
            (CompareFlags::from_status(status).order(), status)
        }
    };
}

x87_compare_status!(
    /// `FUCOMPP`: the quiet compare that sets C3, C2, and C0.
    fucompp, "fucompp"
);
x87_compare_status!(
    /// `FCOMPP`: the signaling compare that sets C3, C2, and C0.
    fcompp, "fcompp"
);

/// Defines a function that loads an integer with `FILD` under a control word
/// and stores it with `FSTP` to 80 bits. Returns the result bits and the
/// status word after the load.
macro_rules! fild {
    ($(#[$doc:meta])* $name:ident, $integer:ty, $load:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(value: $integer, control: u16) -> (u128, u16) {
            let source = value.to_le_bytes();
            let mut output = [0_u8; 16];
            let mut saved = 0_u16;
            let status: u16;
            // SAFETY: as in `x87!`. The code reads the integer from `source`
            // and writes 10 bytes to `output`, which holds 16.
            unsafe {
                asm!(
                    "fnstcw word ptr [{saved}]",
                    "fldcw word ptr [{control}]",
                    "fnclex",
                    $load,
                    "fnstsw ax",
                    "fstp tbyte ptr [{target}]",
                    "fnclex",
                    "fldcw word ptr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    source = in(reg) source.as_ptr(),
                    target = in(reg) output.as_mut_ptr(),
                    out("ax") status,
                    out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                    out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                    options(nostack),
                );
            }
            (u128::from_le_bytes(output), status)
        }
    };
}

fild!(
    /// `FILD m16int`.
    fild_m16, i16, "fild word ptr [{source}]"
);
fild!(
    /// `FILD m32int`.
    fild_m32, i32, "fild dword ptr [{source}]"
);
fild!(
    /// `FILD m64int`.
    fild_m64, i64, "fild qword ptr [{source}]"
);

/// The largest number of steps that [`fprem1`] and [`fprem`] run. A step
/// that leaves the reduction incomplete lowers the exponent by at least 32
/// (Intel SDM Volume 2A, `FPREM1` and `FPREM`). The exponents of two x87
/// values differ by less than 2^15 + 64, so about 1,030 steps reduce every
/// pair.
const PARTIAL_REMAINDER_STEPS: u64 = 2048;

/// Defines a function that runs a partial remainder instruction until C2
/// reports a complete reduction.
macro_rules! partial_remainder {
    ($(#[$doc:meta])* $name:ident, $instruction:literal) => {
        $(#[$doc])*
        ///
        /// The dividend `a` is in ST(0) and the divisor `b` in ST(1), under
        /// a control word. Returns the remainder bits and the status word of
        /// the last step. The exception flags collect over the steps. C2 is
        /// set in the returned status only when the reduction is still
        /// incomplete after `PARTIAL_REMAINDER_STEPS` steps. C3, C1, and C0
        /// hold the low three bits of the quotient.
        #[must_use]
        pub fn $name(a: u128, b: u128, control: u16) -> (u128, u16) {
            let operands = [a.to_le_bytes(), b.to_le_bytes()];
            let mut output = [0_u8; 16];
            let mut saved = 0_u16;
            let status: u16;
            // SAFETY: as in `x87!`. The loop repeats the instruction while
            // C2, bit 2 of AH, is set, and `LOOPNZ` stops the loop after
            // `PARTIAL_REMAINDER_STEPS` steps. Two `FSTP` instructions pop
            // the remainder and the divisor, so the stack is empty on exit.
            unsafe {
                asm!(
                    "fnstcw word ptr [{saved}]",
                    "fldcw word ptr [{control}]",
                    "fnclex",
                    "fld tbyte ptr [{operands} + 16]",
                    "fld tbyte ptr [{operands}]",
                    "2:",
                    $instruction,
                    "fnstsw ax",
                    "test ah, 4",
                    "loopnz 2b",
                    "fstp tbyte ptr [{target}]",
                    "fstp st(0)",
                    "fnclex",
                    "fldcw word ptr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    operands = in(reg) operands.as_ptr(),
                    target = in(reg) output.as_mut_ptr(),
                    out("ax") status,
                    inout("rcx") PARTIAL_REMAINDER_STEPS => _,
                    out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                    out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                    options(nostack),
                );
            }
            (u128::from_le_bytes(output), status)
        }
    };
}

partial_remainder!(
    /// Runs `FPREM1`, the IEEE 754 remainder.
    fprem1, "fprem1"
);
partial_remainder!(
    /// Runs `FPREM`, the remainder of the quotient truncated toward zero.
    fprem, "fprem"
);

/// Runs `FSCALE` on `a` in ST(0) with `scale` in ST(1), under a control word.
/// Returns the result bits and the status word.
///
/// `FILD` loads the scale, so ST(1) holds an integer, which `FSCALE` uses as
/// it is.
#[must_use]
pub fn fscale(a: u128, scale: i32, control: u16) -> (u128, u16) {
    let value = a.to_le_bytes();
    let mut output = [0_u8; 16];
    let mut saved = 0_u16;
    let status: u16;
    // SAFETY: as in `x87!`. The code reads the scale from `scale` and the
    // value from `value`. Two `FSTP` instructions pop the result and the
    // scale, so the stack is empty on exit.
    unsafe {
        asm!(
            "fnstcw word ptr [{saved}]",
            "fldcw word ptr [{control}]",
            "fnclex",
            "fild dword ptr [{scale}]",
            "fld tbyte ptr [{value}]",
            "fscale",
            "fnstsw ax",
            "fstp tbyte ptr [{target}]",
            "fstp st(0)",
            "fnclex",
            "fldcw word ptr [{saved}]",
            saved = in(reg) &raw mut saved,
            control = in(reg) &raw const control,
            scale = in(reg) &raw const scale,
            value = in(reg) value.as_ptr(),
            target = in(reg) output.as_mut_ptr(),
            out("ax") status,
            out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
    }
    (u128::from_le_bytes(output), status)
}
