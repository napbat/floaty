//! Runs SSE and x87 instructions on the host processor, as an oracle.
//!
//! Each function saves the MXCSR register or the x87 control word, runs one
//! instruction under the requested control value, reads the flags, and
//! restores the saved value. The x87 functions leave the x87 stack empty, as
//! the ABI requires.
//!
//! This module holds the control values and the mappings between floaty flags
//! and processor flags. The submodule `sse` runs the SSE instructions, and the
//! submodule `x87` runs the x87 instructions. This module re-exports both.

mod sse;
mod x87;

use core::cmp::Ordering;

use floaty::{Env, Flags, Rounding};

pub use sse::*;
pub use x87::*;

/// The MXCSR value with every exception masked and no flag set.
pub const MXCSR_MASKED: u32 = 0x1F80;
/// The MXCSR flag bits: IE, DE, ZE, OE, UE, and PE.
pub const MXCSR_FLAGS: u32 = 0x3F;
/// The MXCSR denormals-are-zero bit.
pub const MXCSR_DAZ: u32 = 1 << 6;
/// The DE flag of MXCSR.
pub const MXCSR_DE: u32 = 1 << 1;
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

/// Returns the behavior of the SSE unit under an MXCSR setting: the
/// [`Env::X86_SSE`] preset with the rounding control, FTZ, and DAZ of the
/// setting. Every SSE hardware test compares floaty under this behavior, so
/// the tests confirm the preset.
#[must_use]
pub fn sse_env(rounding: Rounding, ftz: bool, daz: bool) -> Env {
    Env::X86_SSE
        .with_rounding(rounding)
        .with_flush_to_zero(ftz)
        .with_denormals_are_zero(daz)
}

/// Returns the behavior of the x87 unit under a rounding control: the
/// [`Env::X87`] preset with that rounding control. Every x87 hardware test
/// compares floaty under this behavior, so the tests confirm the preset.
#[must_use]
pub fn x87_env(rounding: Rounding) -> Env {
    Env::X87.with_rounding(rounding)
}

/// The x87 status bits that floaty flags report: IE, DE, ZE, OE, UE, PE, and
/// C1.
pub const X87_STATUS_FLAGS: u16 = 0b10_0011_1111;

/// The DE bit of the x87 status word.
pub const X87_DE: u16 = 1 << 1;

/// The PE bit of the x87 status word.
pub const X87_PE: u16 = 1 << 5;

/// The C1 bit of the x87 status word.
pub const X87_C1: u16 = 1 << 9;

/// Returns the x87 status bits of an arithmetic instruction for floaty flags.
///
/// DE is `DENORMAL_INPUT` unless a NaN operand, an invalid operation, or a
/// division by zero comes first. The Intel SDM Volume 1, section 4.9.2 on page
/// 4-24, gives that order. An unsupported operand is an invalid operation.
/// `nan_operand` says that an operand is a NaN.
#[must_use]
pub fn x87_arithmetic_status(flags: Flags, nan_operand: bool) -> u16 {
    let higher =
        nan_operand || flags.contains(Flags::INVALID) || flags.contains(Flags::DIVIDE_BY_ZERO);
    let bits = x87_status(flags);
    if higher { bits & !X87_DE } else { bits }
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

/// The three flags in which a floating-point compare reports the order: ZF,
/// PF, and CF in EFLAGS, or C3, C2, and C0 in the x87 status word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CompareFlags {
    zero: bool,
    parity: bool,
    carry: bool,
}

impl CompareFlags {
    /// Reads C3, C2, and C0 from an x87 status word, in the EFLAGS positions
    /// that `FSTSW AX` and `SAHF` copy them to: C3 to ZF, C2 to PF, and C0 to
    /// CF (Intel SDM Volume 1, Figure 8-5 on page 8-6).
    fn from_status(status: u16) -> Self {
        let bit = |index: u16| (status >> index) & 1 == 1;
        Self {
            zero: bit(14),
            parity: bit(10),
            carry: bit(8),
        }
    }

    /// Returns the order of the first operand to the second, or `None` when
    /// the operands are unordered.
    ///
    /// The Intel SDM Volume 1, Tables 8-6 and 8-7 on page 8-19, give the
    /// encodings for the x87 compares. The SSE compares `UCOMISS`, `COMISS`,
    /// `UCOMISD`, and `COMISD` use the same encodings in EFLAGS.
    ///
    /// # Panics
    ///
    /// Panics for a combination that no compare instruction sets.
    fn order(self) -> Option<Ordering> {
        match (self.zero, self.parity, self.carry) {
            (false, false, false) => Some(Ordering::Greater),
            (false, false, true) => Some(Ordering::Less),
            (true, false, false) => Some(Ordering::Equal),
            (true, true, true) => None,
            _ => panic!("a compare reports one of four flag combinations, not {self:?}"),
        }
    }
}
