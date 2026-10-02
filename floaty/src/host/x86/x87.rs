//! The x87 unit of x86: its control word and the instructions of the x87
//! extended paths.

use super::super::Operation;
use crate::host::bits;

/// The exception masks IM, DM, ZM, OM, UM, and PM of the x87 control word.
/// A clear mask traps, and the engine never traps.
const MASKS: u16 = 0x3F;

/// The precision control field of the x87 control word, which holds 11B for
/// the 64-bit precision.
const PRECISION: u16 = 3 << 8;

/// The rounding control field of the x87 control word, which holds zero to
/// round to nearest even.
const ROUNDING: u16 = 3 << 10;

/// Returns the x87 control word. The fields are in the Intel SDM Volume 1,
/// revision 253665-093US, section 8.1.5, Figure 8-6.
#[inline]
fn control_word() -> u16 {
    let control: u16;
    // SAFETY: the block stores the x87 control word in eight bytes that it
    // takes below the stack pointer, loads the value into a register, and
    // restores the stack pointer. `SUB` and `ADD` change the arithmetic
    // flags, so the block does not preserve them. It changes no other state.
    unsafe {
        core::arch::asm!(
            concat!("sub ", stack_pointer!(), ", 8"),
            concat!("fnstcw word ptr [", stack_pointer!(), "]"),
            concat!("movzx {control:e}, word ptr [", stack_pointer!(), "]"),
            concat!("add ", stack_pointer!(), ", 8"),
            control = out(reg) control,
        );
    }
    control
}

/// Returns `true` when the x87 unit rounds to nearest even and masks every
/// exception, at any precision.
///
/// Linux starts a thread with the control word 037FH, at the 64-bit
/// precision. Windows starts one with 027FH, at the 53-bit precision: the
/// x64 calling convention states it, and 32-bit Windows does the same. The
/// precision control applies only to `FADD`, `FSUB`, `FMUL`, `FDIV`, their
/// other forms, and `FSQRT`, Intel SDM Volume 1, section 8.1.5.2. So only
/// the arithmetic and the square root of x87 extended values need the 64-bit
/// precision, which `x87_full_precision` checks. The loads, the stores,
/// `FRNDINT`, `FISTP`, and `FPREM1` give the same bits at every precision.
#[inline]
pub fn x87_environment() -> bool {
    control_word() & (MASKS | ROUNDING) == MASKS
}

/// Returns `true` when the x87 unit rounds to nearest even at the 64-bit
/// precision, and masks every exception, as the arithmetic and the square
/// root of x87 extended values need.
#[inline]
pub fn x87_full_precision() -> bool {
    control_word() & (MASKS | PRECISION | ROUNDING) == MASKS | PRECISION
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
#[cfg(target_arch = "x86_64")]
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

/// Runs x87 instructions that leave one value on the x87 stack, and stores
/// that value as an 80-bit encoding, as on x86-64. 32-bit x86 has no 64-bit
/// general register, so the code reads the stored encoding after the block.
#[cfg(target_arch = "x86")]
macro_rules! x87_extended {
    ($($instruction:literal),+; $($name:ident = $pointer:expr),+) => {{
        let mut scratch = [0_u64; 2];
        // SAFETY: as in the x86-64 form, without the loads of the result.
        unsafe {
            core::arch::asm!(
                $($instruction,)+
                "fstp tbyte ptr [{scratch}]",
                $($name = in(reg) $pointer,)+
                scratch = in(reg) scratch.as_mut_ptr(),
                out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                options(nostack, preserves_flags),
            );
        }
        extended_result(scratch)
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
    let nan = bits::nan_32(result);
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
    let nan = bits::nan_64(result);
    (!nan).then_some(result)
}

/// Computes the IEEE remainder of the value at `dividend` and the value at
/// `divisor`, both of the width `$width`, by `FPREM1`, and stores it through
/// `result`, of type `$result` and the same width.
///
/// Each `FPREM1` reduces the exponent difference of its operands by up to 63,
/// and sets C2, bit 2 of the high byte of the status word, while the
/// remainder is partial. The quotient rounds to nearest even whatever the
/// control word holds, and the remainder is exact, Intel SDM Volume 2:
/// `FPREM1`. A remainder of binary32 or binary64 operands is a value of
/// their format, so the store is exact.
macro_rules! x87_remainder {
    ($result:ty, $width:literal; $dividend:expr, $divisor:expr) => {{
        let mut result: $result = 0;
        // SAFETY: the instructions read the operands through the pointers and
        // write one result of the width to `result`. FNSTSW writes AX, which
        // is a clobber, and TEST writes the arithmetic flags. Every x87
        // register is a clobber, so the x87 stack is empty on entry, and
        // `FSTP ST(1)` and the store pop both values, so it is empty on exit.
        // The x87 unit changes only its status word, which floaty does not
        // read.
        unsafe {
            core::arch::asm!(
                concat!("fld ", $width, " ptr [{divisor}]"),
                concat!("fld ", $width, " ptr [{dividend}]"),
                "2:",
                "fprem1",
                "fnstsw ax",
                "test ah, 4",
                "jnz 2b",
                "fstp st(1)",
                concat!("fstp ", $width, " ptr [{result}]"),
                dividend = in(reg) $dividend,
                divisor = in(reg) $divisor,
                result = in(reg) &raw mut result,
                out("ax") _,
                out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                options(nostack),
            );
        }
        result
    }};
}

/// Returns the IEEE remainder of two binary32 encodings from the x87 unit, by
/// `FPREM1`, or `None` for a NaN.
#[inline]
pub fn x87_remainder_single(dividend: u32, divisor: u32) -> Option<u32> {
    let result = x87_remainder!(u32, "dword"; &raw const dividend, &raw const divisor);
    // The NaN test uses integer instructions, as the paths do.
    let nan = bits::nan_32(result);
    (!nan).then_some(result)
}

/// Returns the IEEE remainder of two binary64 encodings from the x87 unit, by
/// `FPREM1`, or `None` for a NaN.
#[inline]
pub fn x87_remainder_double(dividend: u64, divisor: u64) -> Option<u64> {
    let result = x87_remainder!(u64, "qword"; &raw const dividend, &raw const divisor);
    // The NaN test uses integer instructions, as the paths do.
    let nan = bits::nan_64(result);
    (!nan).then_some(result)
}

/// Returns the IEEE remainder of two 80-bit encodings from the x87 unit, by
/// `FPREM1`, or `None` for a NaN. The block reads the stored encoding back
/// as `x87_extended!` does.
#[cfg(target_arch = "x86_64")]
#[inline]
pub fn x87_remainder(dividend: &[u64; 2], divisor: &[u64; 2]) -> Option<[u64; 2]> {
    let mut scratch = [0_u64; 2];
    let (low, high): (u64, u64);
    // SAFETY: as in `x87_remainder!`, with the 80-bit loads and store of
    // `x87_extended!`. `FSTP` writes 10 bytes to `scratch`, which holds 16,
    // and the block reads them back.
    unsafe {
        core::arch::asm!(
            "fld tbyte ptr [{divisor}]",
            "fld tbyte ptr [{dividend}]",
            "2:",
            "fprem1",
            "fnstsw ax",
            "test ah, 4",
            "jnz 2b",
            "fstp st(1)",
            "fstp tbyte ptr [{scratch}]",
            "mov {low}, qword ptr [{scratch}]",
            "movzx {high:e}, word ptr [{scratch} + 8]",
            dividend = in(reg) dividend.as_ptr(),
            divisor = in(reg) divisor.as_ptr(),
            scratch = in(reg) scratch.as_mut_ptr(),
            low = out(reg) low,
            high = lateout(reg) high,
            out("ax") _,
            out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
    }
    extended_result([low, high])
}

/// Returns the IEEE remainder of two 80-bit encodings from the x87 unit, by
/// `FPREM1`, or `None` for a NaN. The code reads the stored encoding after
/// the block, as the 32-bit form of `x87_extended!` does.
#[cfg(target_arch = "x86")]
#[inline]
pub fn x87_remainder(dividend: &[u64; 2], divisor: &[u64; 2]) -> Option<[u64; 2]> {
    let mut scratch = [0_u64; 2];
    // SAFETY: as in the x86-64 form, without the loads of the result.
    unsafe {
        core::arch::asm!(
            "fld tbyte ptr [{divisor}]",
            "fld tbyte ptr [{dividend}]",
            "2:",
            "fprem1",
            "fnstsw ax",
            "test ah, 4",
            "jnz 2b",
            "fstp st(1)",
            "fstp tbyte ptr [{scratch}]",
            dividend = in(reg) dividend.as_ptr(),
            divisor = in(reg) divisor.as_ptr(),
            scratch = in(reg) scratch.as_mut_ptr(),
            out("ax") _,
            out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
            out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
            options(nostack),
        );
    }
    extended_result(scratch)
}

#[cfg(test)]
mod tests {
    extern crate std;

    use super::{control_word, x87_environment, x87_full_precision};
    use crate::env::Env;
    use crate::float::Float;
    use crate::format::internal::LimbConversion;
    use crate::format::{Binary, Standard, X87};
    use crate::host::{self, Host, Kind, Operation};
    use crate::limbs::Limbs;

    /// x87 extended precision.
    type Extended = Binary<15, X87>;

    /// The control word of a new Linux thread: round to nearest at the
    /// 64-bit precision, with every exception masked.
    const LINUX: u16 = 0x037F;

    /// The control word of a new Windows thread, as the x64 calling
    /// convention states: the same at the 53-bit precision.
    const WINDOWS: u16 = 0x027F;

    /// Loads an x87 control word, and loads the saved word back when it
    /// drops, also after a failed assertion.
    struct Loaded(u16);

    impl Loaded {
        fn new(control: u16) -> Self {
            let saved = control_word();
            load(control);
            Self(saved)
        }
    }

    impl Drop for Loaded {
        fn drop(&mut self) {
            load(self.0);
        }
    }

    /// Loads the x87 control word `control`.
    fn load(control: u16) {
        // SAFETY: `FNCLEX` clears the exception flags, so no flag traps under
        // the new word, and `FLDCW` loads the control word from `control`.
        // Neither uses the x87 stack.
        unsafe {
            core::arch::asm!(
                "fnclex",
                "fldcw word ptr [{control}]",
                control = in(reg) &raw const control,
                options(nostack),
            );
        }
    }

    /// Returns the encoding of `integer` in the format `S`.
    fn value<S: Standard<W>, const W: usize>(integer: i64) -> S::Bits {
        Float::<S, W>::from_int(integer).to_bits()
    }

    /// Asserts that the host path gives the remainder of 3 and 2 in the
    /// format `S`, which is -1.
    fn assert_remainder<S: Standard<W>, const W: usize>(control: u16) {
        let (three, two) = (value::<S, W>(3), value::<S, W>(2));
        assert_eq!(
            host::remainder::<S, W>(three, two, &Env::IEEE),
            Some(value::<S, W>(-1)),
            "{:?} remainder under {control:#06x}",
            S::HOST
        );
    }

    #[test]
    fn a_new_thread_has_the_x87_precision_of_the_claims() {
        let (environment, full) = std::thread::spawn(|| (x87_environment(), x87_full_precision()))
            .join()
            .expect("the thread reads the control word");
        assert!(
            environment,
            "a new thread rounds to nearest and masks every exception"
        );
        assert_eq!(
            full,
            super::super::X87_FULL_PRECISION,
            "the precision of a new thread"
        );
        for kind in [Kind::Arithmetic, Kind::SquareRoot] {
            assert_eq!(host::available(Host::Extended, kind), full, "{kind:?}");
        }
    }

    #[test]
    fn only_the_extended_arithmetic_needs_the_64_bit_precision() {
        let env = Env::IEEE;
        let extended = value::<Extended, 80>;
        let single_three = u64::from(3.0_f32.to_bits());
        let extended_three = extended(3).to_limbs();
        let extended_three = [extended_three.limb(0), extended_three.limb(1)];
        for (control, full) in [(LINUX, true), (WINDOWS, false)] {
            let _loaded = Loaded::new(control);
            assert!(x87_environment(), "{control:#06x}");
            assert_eq!(x87_full_precision(), full, "{control:#06x}");
            assert_remainder::<Binary<5>, 16>(control);
            assert_remainder::<Binary<8>, 16>(control);
            assert_remainder::<Binary<8>, 32>(control);
            assert_remainder::<Binary<11>, 64>(control);
            assert_remainder::<Extended, 80>(control);
            let three = extended(3);
            assert_eq!(
                host::round_to_integral::<Extended, 80>(three, &env),
                Some(three)
            );
            assert_eq!(host::to_int::<Extended, 80>(three, &env), Some(3));
            assert_eq!(host::from_int::<Extended, 80>(3, &env), Some(three));
            assert_eq!(
                host::convert(Host::Single, Host::Extended, [single_three, 0], &env),
                Some(extended_three)
            );
            assert_eq!(
                host::convert(Host::Extended, Host::Single, extended_three, &env),
                Some([single_three, 0])
            );
            // `FADD` and `FSQRT` round to the precision control.
            let sum = host::binary::<Extended, 80>(three, extended(2), Operation::Add, &env);
            assert_eq!(sum, full.then(|| extended(5)), "{control:#06x}");
            let root = host::sqrt::<Extended, 80>(extended(9), &env);
            assert_eq!(root, full.then_some(three), "{control:#06x}");
        }
    }
}
