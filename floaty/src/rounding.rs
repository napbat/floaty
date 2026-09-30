//! The rounding rules that the binary and the decimal rounding routines
//! share: the step of each direction, the underflow flags, the integer that
//! a value rounds to, the direction of an overflow, and the quotient step of
//! the remainders. Each radix has its own routine: `exact` for the binary
//! formats and `decimal::round` for the decimal formats.

use core::cmp::Ordering;

use crate::env::{Env, Flags, Rounding};
use crate::format::internal::Quotient;
use crate::limbs::Limbs;

/// The dropped part of a value, against half a unit of the last kept digit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dropped {
    /// Nothing: the kept digits are exact.
    Nothing,
    /// Above zero and below half a unit.
    BelowHalf,
    /// Exactly half a unit.
    Half,
    /// Above half a unit.
    AboveHalf,
}

impl Dropped {
    /// Returns the dropped part from the order of the highest dropped digits
    /// against half a unit, and `true` when a dropped digit below them, or a
    /// dropped part below half, is nonzero.
    #[inline]
    pub fn new(first: Ordering, rest: bool) -> Self {
        match (first, rest) {
            (Ordering::Less, false) => Self::Nothing,
            (Ordering::Less, true) => Self::BelowHalf,
            (Ordering::Equal, false) => Self::Half,
            (Ordering::Equal, true) | (Ordering::Greater, _) => Self::AboveHalf,
        }
    }

    /// Returns `true` when the dropped part is not zero.
    #[inline]
    pub fn is_inexact(self) -> bool {
        self != Self::Nothing
    }
}

/// Returns `true` when a rounding in the direction `rounding` adds one unit
/// to the kept digits. The binary and the decimal rounding routines share
/// this choice.
///
/// `last_digit` is the last kept digit in `radix`. Round to odd adds one when
/// that digit is 0, or 5 in radix 10. In radix 10, this rule is IBM's round
/// to prepare for shorter precision.
#[inline]
pub fn rounds_up(
    rounding: Rounding,
    negative: bool,
    dropped: Dropped,
    last_digit: u64,
    radix: u32,
) -> bool {
    let inexact = dropped.is_inexact();
    match rounding {
        Rounding::TiesToEven => {
            dropped == Dropped::AboveHalf || (dropped == Dropped::Half && last_digit % 2 == 1)
        }
        Rounding::TiesToAway => matches!(dropped, Dropped::Half | Dropped::AboveHalf),
        Rounding::TiesTowardZero => dropped == Dropped::AboveHalf,
        Rounding::TowardPositive => inexact && !negative,
        Rounding::TowardNegative => inexact && negative,
        Rounding::TowardZero => false,
        Rounding::AwayFromZero => inexact,
        Rounding::ToOdd => inexact && (last_digit == 0 || (radix == 10 && last_digit == 5)),
    }
}

impl Quotient {
    /// Returns `true` when the quotient `n` of a remainder `x - n * y` rounds
    /// to one above the truncated quotient. `rest` is the remainder of the
    /// truncated quotient, and `odd` is its lowest bit. The nearest quotient
    /// grows past half of the divisor, and at a tie when it is odd. The
    /// truncated quotient never grows.
    #[inline]
    pub fn rounds_up<L: Limbs>(self, rest: &L, divisor: &L, odd: bool) -> bool {
        match self {
            Self::Nearest => match rest.shl(1).compare(divisor) {
                Ordering::Greater => true,
                Ordering::Equal => odd,
                Ordering::Less => false,
            },
            Self::Truncated => false,
        }
    }
}

/// The flags of a rounded result that can be tiny or inexact.
pub enum Underflow {
    /// Flush-to-zero replaces the tiny result with a zero of its sign, with
    /// these flags.
    Flushed(Flags),
    /// The result stays, with these flags.
    Kept(Flags),
}

/// Returns the flags of a rounded result from whether it is tiny and
/// inexact: `TINY`, `INEXACT`, and `UNDERFLOW` for a result that is both.
/// With flush-to-zero, a tiny result becomes a zero that is tiny, underflows,
/// and is inexact.
#[inline]
pub fn underflow(tiny: bool, inexact: bool, env: &Env) -> Underflow {
    let mut flags = Flags::NONE;
    if tiny {
        flags |= Flags::TINY;
        if env.flush_to_zero {
            return Underflow::Flushed(flags | Flags::UNDERFLOW | Flags::INEXACT);
        }
    }
    if inexact {
        flags |= Flags::INEXACT;
        if tiny {
            flags |= Flags::UNDERFLOW;
        }
    }
    Underflow::Kept(flags)
}

/// The integer that a value rounds to.
#[derive(Clone, Copy, Debug)]
pub struct Integral<L> {
    /// The magnitude of the integer.
    pub magnitude: L,
    /// `true` when the integer differs from the value.
    pub inexact: bool,
    /// `true` when the magnitude of the integer is above the magnitude of the
    /// value.
    pub rounded_up: bool,
}

impl<L> Integral<L> {
    /// Returns the flags of the rounding: `INEXACT` when the integer differs
    /// from the value, and `ROUNDED_UP` when its magnitude is above it.
    #[inline]
    pub fn flags(&self) -> Flags {
        let mut flags = Flags::NONE;
        if self.inexact {
            flags |= Flags::INEXACT;
        }
        if self.rounded_up {
            flags |= Flags::ROUNDED_UP;
        }
        flags
    }
}

/// Returns `true` when an overflow of the sign `negative` gives an infinity
/// in the direction `rounding`, and `false` when it gives the largest finite
/// value.
pub fn overflows_to_infinity(rounding: Rounding, negative: bool) -> bool {
    match rounding {
        Rounding::TiesToEven
        | Rounding::TiesToAway
        | Rounding::TiesTowardZero
        | Rounding::AwayFromZero => true,
        Rounding::TowardPositive => !negative,
        Rounding::TowardNegative => negative,
        Rounding::TowardZero | Rounding::ToOdd => false,
    }
}
