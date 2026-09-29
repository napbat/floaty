//! The x87 unit of x86-64: its control word and the instructions of the x87
//! extended paths.

use super::super::Operation;

/// Returns `true` when the x87 unit rounds to nearest even at the 64-bit
/// precision, and masks every exception.
///
/// Linux starts a process with that control word, 037FH, but other systems
/// and libraries can select a 53-bit precision. An unmasked exception traps,
/// and the engine never traps. The fields are in the Intel SDM Volume 1,
/// revision 253665-093US, section 8.1.5, Figure 8-6.
#[inline]
pub fn x87_environment() -> bool {
    // The exception masks IM, DM, ZM, OM, UM, and PM are one, the precision
    // control is 11B for 64 bits, and the rounding control is zero.
    const MASKS: u16 = 0x3F;
    const PRECISION: u16 = 3 << 8;
    const ROUNDING: u16 = 3 << 10;
    let control: u16;
    // SAFETY: the block stores the x87 control word in eight bytes that it
    // takes below the stack pointer, loads the value into a register, and
    // restores the stack pointer. It changes no other state.
    unsafe {
        core::arch::asm!(
            "sub rsp, 8",
            "fnstcw word ptr [rsp]",
            "movzx {control:e}, word ptr [rsp]",
            "add rsp, 8",
            control = out(reg) control,
            options(preserves_flags),
        );
    }
    control & (MASKS | PRECISION | ROUNDING) == MASKS | PRECISION
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
    let nan = result & 0x7FFF_FFFF > 0x7F80_0000;
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
    let nan = result & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000;
    (!nan).then_some(result)
}
