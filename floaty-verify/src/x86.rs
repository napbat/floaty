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
use core::num::NonZeroU32;

use floaty::{Env, Flags, Rounding, ToInt};

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

/// The MXCSR rounding-control field that rounds toward negative infinity.
pub const MXCSR_TOWARD_NEGATIVE: u32 = 1 << 13;
/// The MXCSR rounding-control field that rounds toward positive infinity.
pub const MXCSR_TOWARD_POSITIVE: u32 = 2 << 13;
/// The MXCSR rounding-control field that rounds toward zero.
pub const MXCSR_TOWARD_ZERO: u32 = 3 << 13;

/// The rounding directions and their MXCSR rounding-control field.
pub const MXCSR_ROUNDINGS: [(Rounding, u32); 4] = [
    (Rounding::TiesToEven, 0),
    (Rounding::TowardNegative, MXCSR_TOWARD_NEGATIVE),
    (Rounding::TowardPositive, MXCSR_TOWARD_POSITIVE),
    (Rounding::TowardZero, MXCSR_TOWARD_ZERO),
];

/// The MXCSR exception masks IM, DM, ZM, OM, UM, and PM, bits 7 to 12. An
/// exception whose mask is clear traps.
pub const MXCSR_EXCEPTION_MASKS: [u32; 6] = [1 << 7, 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12];

/// The MXCSR values under which the scalar operators and conversions must
/// give the engine results: FTZ, DAZ, each directed rounding, and each
/// unmasked exception. FTZ, DAZ, and each directed rounding change the host
/// results, and an unmasked exception traps.
pub const MXCSR_OPERATOR_CONTROLS: [u32; 11] = [
    MXCSR_MASKED | MXCSR_FTZ,
    MXCSR_MASKED | MXCSR_DAZ,
    MXCSR_MASKED | MXCSR_TOWARD_NEGATIVE,
    MXCSR_MASKED | MXCSR_TOWARD_POSITIVE,
    MXCSR_MASKED | MXCSR_TOWARD_ZERO,
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[0],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[1],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[2],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[3],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[4],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[5],
];

/// The MXCSR values under which the scalar comparisons and the minimum and
/// maximum operations must give the engine results: the default, FTZ, DAZ,
/// rounding toward negative infinity, and each unmasked exception.
pub const MXCSR_COMPARISON_CONTROLS: [u32; 10] = [
    MXCSR_MASKED,
    MXCSR_MASKED | MXCSR_FTZ,
    MXCSR_MASKED | MXCSR_DAZ,
    MXCSR_MASKED | MXCSR_TOWARD_NEGATIVE,
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[0],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[1],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[2],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[3],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[4],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[5],
];

/// The MXCSR values under which the paths of `Lanes` must give the engine
/// results: the default, FTZ, DAZ, each directed rounding, and each unmasked
/// exception.
pub const MXCSR_LANE_CONTROLS: [u32; 12] = [
    MXCSR_MASKED,
    MXCSR_MASKED | MXCSR_FTZ,
    MXCSR_MASKED | MXCSR_DAZ,
    MXCSR_MASKED | MXCSR_TOWARD_NEGATIVE,
    MXCSR_MASKED | MXCSR_TOWARD_POSITIVE,
    MXCSR_MASKED | MXCSR_TOWARD_ZERO,
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[0],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[1],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[2],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[3],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[4],
    MXCSR_MASKED & !MXCSR_EXCEPTION_MASKS[5],
];

/// Bit 2 of a `ROUNDSS`, `ROUNDSD`, `ROUNDPS`, or `ROUNDPD` immediate: round
/// in the MXCSR direction.
pub const ROUND_USE_MXCSR: u8 = 1 << 2;

/// Bit 3 of a `ROUNDSS`, `ROUNDSD`, `ROUNDPS`, or `ROUNDPD` immediate:
/// suppress the precision exception.
pub const ROUND_SUPPRESS_PRECISION: u8 = 1 << 3;

/// An MXCSR setting of the SSE hardware tests.
#[derive(Clone, Copy, Debug)]
pub struct SseSetting {
    /// The MXCSR value, with every exception masked.
    pub control: u32,
    /// The direction of the MXCSR rounding-control field.
    pub rounding: Rounding,
    /// The flush-to-zero bit.
    pub ftz: bool,
    /// The denormals-are-zero bit.
    pub daz: bool,
}

impl SseSetting {
    /// Returns the behavior of the SSE unit in this setting: [`sse_env`] of
    /// its fields.
    #[must_use]
    pub fn env(self) -> Env {
        sse_env(self.rounding, self.ftz, self.daz)
    }
}

/// Returns every MXCSR setting of the SSE hardware tests: each rounding
/// direction with FTZ and DAZ on and off.
#[must_use]
pub fn sse_settings() -> Vec<SseSetting> {
    let mut settings = Vec::new();
    for (rounding, field) in MXCSR_ROUNDINGS {
        for (ftz, daz) in [(false, false), (true, false), (false, true), (true, true)] {
            let control = MXCSR_MASKED
                | field
                | if ftz { MXCSR_FTZ } else { 0 }
                | if daz { MXCSR_DAZ } else { 0 };
            settings.push(SseSetting {
                control,
                rounding,
                ftz,
                daz,
            });
        }
    }
    settings
}

/// The x87 rounding-control field that rounds toward negative infinity.
pub const X87_TOWARD_NEGATIVE: u16 = 1 << 10;
/// The x87 rounding-control field that rounds toward positive infinity.
pub const X87_TOWARD_POSITIVE: u16 = 2 << 10;
/// The x87 rounding-control field that rounds toward zero.
pub const X87_TOWARD_ZERO: u16 = 3 << 10;

/// The rounding directions and their x87 rounding-control field.
pub const X87_ROUNDINGS: [(Rounding, u16); 4] = [
    (Rounding::TiesToEven, 0),
    (Rounding::TowardNegative, X87_TOWARD_NEGATIVE),
    (Rounding::TowardPositive, X87_TOWARD_POSITIVE),
    (Rounding::TowardZero, X87_TOWARD_ZERO),
];

/// The x87 precision-control field for 53 bits.
pub const X87_DOUBLE_PRECISION: u16 = 2 << 8;
/// The x87 precision-control field for 64 bits, the full precision.
pub const X87_FULL_PRECISION: u16 = 3 << 8;

/// The x87 precision-control field: 24, 53, and 64 bits.
pub const X87_PRECISIONS: [(u32, u16); 3] = [
    (24, 0),
    (53, X87_DOUBLE_PRECISION),
    (64, X87_FULL_PRECISION),
];

/// The x87 control word with every exception masked, before the rounding and
/// precision fields.
pub const X87_MASKED: u16 = 0x007F;

/// The x87 exception masks IM, DM, ZM, OM, UM, and PM, bits 0 to 5. An
/// exception whose mask is clear traps.
pub const X87_EXCEPTION_MASKS: [u16; 6] = [1, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5];

/// The x87 control words under which the x87 paths must give the engine
/// results: each rounding direction at full precision, each precision at
/// round to nearest, and each unmasked exception at full precision. Each
/// directed rounding and each precision below 64 bits changes the x87
/// results, and an unmasked exception traps.
pub const X87_HOST_PATH_CONTROLS: [u16; 13] = [
    X87_MASKED | X87_FULL_PRECISION,
    X87_MASKED | X87_FULL_PRECISION | X87_TOWARD_NEGATIVE,
    X87_MASKED | X87_FULL_PRECISION | X87_TOWARD_POSITIVE,
    X87_MASKED | X87_FULL_PRECISION | X87_TOWARD_ZERO,
    X87_MASKED,
    X87_MASKED | X87_DOUBLE_PRECISION,
    X87_MASKED | X87_FULL_PRECISION,
    (X87_MASKED | X87_FULL_PRECISION) & !X87_EXCEPTION_MASKS[0],
    (X87_MASKED | X87_FULL_PRECISION) & !X87_EXCEPTION_MASKS[1],
    (X87_MASKED | X87_FULL_PRECISION) & !X87_EXCEPTION_MASKS[2],
    (X87_MASKED | X87_FULL_PRECISION) & !X87_EXCEPTION_MASKS[3],
    (X87_MASKED | X87_FULL_PRECISION) & !X87_EXCEPTION_MASKS[4],
    (X87_MASKED | X87_FULL_PRECISION) & !X87_EXCEPTION_MASKS[5],
];

/// An x87 control setting of the x87 hardware tests.
#[derive(Clone, Copy, Debug)]
pub struct X87Setting {
    /// The control word, with every exception masked.
    pub control: u16,
    /// The direction of the rounding-control field.
    pub rounding: Rounding,
    /// The limit of the precision-control field.
    pub precision: Option<NonZeroU32>,
}

impl X87Setting {
    /// Returns the behavior of the x87 unit in this setting: [`x87_env`] with
    /// the limit of the precision-control field.
    #[must_use]
    pub fn env(self) -> Env {
        x87_env(self.rounding).with_precision(self.precision)
    }

    /// Returns the behavior of an instruction that precision control does
    /// not affect. Precision control affects only the add, subtract,
    /// multiply, divide, and square root instructions (Intel SDM Volume 1,
    /// section 8.1.5.2 on page 8-8).
    #[must_use]
    pub fn full_precision(self) -> Env {
        x87_env(self.rounding)
    }
}

/// Returns every x87 control setting of the x87 hardware tests: each
/// rounding direction at each precision.
#[must_use]
pub fn x87_settings() -> Vec<X87Setting> {
    let mut settings = Vec::new();
    for (rounding, rounding_field) in X87_ROUNDINGS {
        for (precision, precision_field) in X87_PRECISIONS {
            settings.push(X87Setting {
                control: X87_MASKED | rounding_field | precision_field,
                rounding,
                precision: NonZeroU32::new(precision),
            });
        }
    }
    settings
}

/// Returns the integer that an x86 conversion to an integer stores: the
/// value, or the integer indefinite for a value out of range or a NaN. The
/// indefinite is the smallest integer, which has only the sign bit set. The
/// Intel SDM Volume 1 gives the rule for SSE in Table 11-1 on page 11-15,
/// and for `FISTP` and `FISTTP` in Table 8-10 on page 8-27.
#[must_use]
pub fn stored<I: Copy>(result: ToInt<I>, indefinite: I) -> I {
    match result {
        ToInt::Value(value) => value,
        ToInt::OutOfRange { .. } | ToInt::Nan => indefinite,
    }
}

/// Returns `true` when a condition of higher priority than a denormal operand
/// occurs, so that the processor does not report DE: a NaN operand, an
/// invalid operation, or a division by zero. The Intel SDM Volume 1, section
/// 4.9.2 on page 4-24, gives that order. `nan_operand` says that an operand
/// is a NaN.
fn denormal_yields(flags: Flags, nan_operand: bool) -> bool {
    nan_operand || flags.contains(Flags::INVALID) || flags.contains(Flags::DIVIDE_BY_ZERO)
}

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
    if flags.contains(Flags::DENORMAL_INPUT) && !daz && !denormal_yields(flags, nan_operand) {
        bits |= MXCSR_DE;
    }
    bits
}

/// Returns the MXCSR flags of `ROUNDSS`, `ROUNDSD`, `ROUNDPS`, or `ROUNDPD`
/// with the immediate `immediate`.
///
/// The round instructions signal only IE and PE (Intel SDM Volume 1, section
/// 12.8.4 and Table 12-1 on page 12-10), so the consumer drops
/// `DENORMAL_INPUT`. The hardware shows that DAZ still reads a subnormal
/// operand as zero. With [`ROUND_SUPPRESS_PRECISION`], the consumer also
/// drops `INEXACT`, as for IEEE 754 `roundToIntegral` without `Exact`.
#[must_use]
pub fn mxcsr_round_flags(flags: Flags, immediate: u8, daz: bool) -> u32 {
    let mut ignored = Flags::DENORMAL_INPUT;
    if immediate & ROUND_SUPPRESS_PRECISION != 0 {
        ignored |= Flags::INEXACT;
    }
    mxcsr_flags(flags.difference(ignored), daz, false)
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
    let bits = x87_status(flags);
    if denormal_yields(flags, nan_operand) {
        bits & !X87_DE
    } else {
        bits
    }
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
