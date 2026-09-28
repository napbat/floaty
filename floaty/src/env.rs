//! Behavior: `Env`, `Rounding`, `Tininess`, `NanRule`, `Flags`, overrides,
//! and the sealed modes.

use core::fmt::{self, Debug, Formatter};
use core::num::NonZeroU32;
use core::ops::{BitOr, BitOrAssign};

use crate::sealed::Sealed;

/// A rounding direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Rounding {
    /// Round to the nearest value. A tie goes to the even significand.
    NearestEven,
    /// Round to the nearest value. A tie goes away from zero.
    NearestAway,
    /// Round toward positive infinity.
    TowardPositive,
    /// Round toward negative infinity.
    TowardNegative,
    /// Round toward zero.
    TowardZero,
    /// Round toward zero, then set the lowest significand bit when the result
    /// is inexact. The result never overflows to an infinity.
    ToOdd,
}

/// When a result is tiny, for the underflow flag and for flush-to-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tininess {
    /// A nonzero result is tiny when its exact value is below the smallest
    /// normal value. ARM uses this rule.
    BeforeRounding,
    /// A nonzero result is tiny when its value, rounded to the precision with
    /// an unbounded exponent range, is below the smallest normal value. x86
    /// and RISC-V use this rule.
    AfterRounding,
}

/// Which input NaN an operation returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum NanPropagation {
    /// A signaling NaN before a quiet NaN, and the earlier operand among NaNs
    /// of one kind. ARM uses this rule for two operands when its default-NaN
    /// mode is off.
    SignalingFirst,
    /// The first NaN operand. x86 SSE uses this rule.
    FirstOperand,
    /// A quiet NaN before a signaling NaN, and the larger significand among
    /// NaNs of one kind. Between equal significands the positive NaN wins.
    /// The x87 unit uses this rule.
    LargerSignificand,
    /// Always the default NaN. The rule drops the operand payloads. RISC-V
    /// uses this rule, and ARM uses it when its default-NaN mode is on.
    DefaultNan,
}

/// What a fused multiply-add does when its product is the invalid `0 * inf`
/// and its addend is a NaN.
///
/// IEEE 754-2019, section 7.2 (c), lets the implementation decide whether a
/// quiet NaN addend signals invalid here. A signaling NaN addend always
/// signals invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum InvalidProduct {
    /// The invalid product signals invalid and gives the default NaN. The
    /// propagation rule then selects the result from that NaN, as the first
    /// operand, and the addend. SoftFloat uses this rule. With
    /// [`NanPropagation::SignalingFirst`] it gives the result of the
    /// `FPMulAdd` pseudocode in the Arm Architecture Reference Manual.
    Signals,
    /// The NaN addend takes precedence over the invalid product. The
    /// propagation rule selects the result from the addend alone, and only a
    /// signaling addend signals invalid. x86 uses this rule: a NaN operand
    /// comes before an invalid operation in the Intel SDM Volume 1, section
    /// 4.9.2.
    YieldsToNan,
}

/// The NaN that an operation returns.
///
/// An operation with a NaN input returns a NaN that
/// [`propagation`](Self::propagation) selects, made quiet. An invalid
/// operation returns the default NaN: a quiet NaN with a zero payload and the
/// sign that [`default_negative`](Self::default_negative) sets.
///
/// Build a `NanRule` with [`NanRule::new`] and the builder methods. The struct
/// is `#[non_exhaustive]`, so a new field does not break callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct NanRule {
    /// Which input NaN an operation returns.
    pub propagation: NanPropagation,
    /// The sign of the default NaN.
    pub default_negative: bool,
    /// What a fused multiply-add does for `0 * inf + NaN`.
    pub invalid_product: InvalidProduct,
}

impl NanRule {
    /// Returns a rule with this propagation, a positive default NaN, and
    /// [`InvalidProduct::Signals`].
    #[must_use]
    pub const fn new(propagation: NanPropagation) -> Self {
        Self {
            propagation,
            default_negative: false,
            invalid_product: InvalidProduct::Signals,
        }
    }

    /// Returns a copy with another sign of the default NaN.
    #[must_use]
    pub const fn with_default_negative(self, default_negative: bool) -> Self {
        Self {
            default_negative,
            ..self
        }
    }

    /// Returns a copy with another rule for `0 * inf + NaN`.
    #[must_use]
    pub const fn with_invalid_product(self, invalid_product: InvalidProduct) -> Self {
        Self {
            invalid_product,
            ..self
        }
    }
}

/// The behavior of an operation: everything that changes the bits of its
/// result.
///
/// Build an `Env` from [`Env::IEEE`] with the builder methods. The struct is
/// `#[non_exhaustive]`, so a new field does not break callers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct Env {
    /// The rounding direction.
    pub rounding: Rounding,
    /// FTZ: a tiny result becomes a zero with the same sign.
    pub flush_to_zero: bool,
    /// DAZ: a subnormal input reads as a zero with the same sign.
    pub denormals_are_zero: bool,
    /// When a result is tiny.
    pub tininess: Tininess,
    /// Which NaN an operation returns.
    pub nan: NanRule,
    /// Rounds the significand to this many bits and keeps the exponent range
    /// of the format, as x87 precision control does. `None` uses the precision
    /// of the format. A value above the format precision has no effect.
    pub precision: Option<NonZeroU32>,
    /// An overflow in an encoding without infinity gives the largest finite
    /// value instead of a NaN.
    pub saturate: bool,
}

impl Env {
    /// The IEEE 754 default behavior: round to nearest even, no flushing,
    /// tininess after rounding, and the
    /// [`SignalingFirst`](NanPropagation::SignalingFirst) NaN rule with a
    /// positive default NaN.
    pub const IEEE: Self = Self {
        rounding: Rounding::NearestEven,
        flush_to_zero: false,
        denormals_are_zero: false,
        tininess: Tininess::AfterRounding,
        nan: NanRule::new(NanPropagation::SignalingFirst),
        precision: None,
        saturate: false,
    };

    /// Returns a copy with another rounding direction.
    #[must_use]
    pub const fn with_rounding(self, rounding: Rounding) -> Self {
        Self { rounding, ..self }
    }

    /// Returns a copy with flush-to-zero set or clear.
    #[must_use]
    pub const fn with_flush_to_zero(self, flush_to_zero: bool) -> Self {
        Self {
            flush_to_zero,
            ..self
        }
    }

    /// Returns a copy with denormals-are-zero set or clear.
    #[must_use]
    pub const fn with_denormals_are_zero(self, denormals_are_zero: bool) -> Self {
        Self {
            denormals_are_zero,
            ..self
        }
    }

    /// Returns a copy with another tininess rule.
    #[must_use]
    pub const fn with_tininess(self, tininess: Tininess) -> Self {
        Self { tininess, ..self }
    }

    /// Returns a copy with another NaN rule.
    #[must_use]
    pub const fn with_nan(self, nan: NanRule) -> Self {
        Self { nan, ..self }
    }

    /// Returns a copy with another precision limit.
    #[must_use]
    pub const fn with_precision(self, precision: Option<NonZeroU32>) -> Self {
        Self { precision, ..self }
    }

    /// Returns a copy with saturation set or clear.
    #[must_use]
    pub const fn with_saturate(self, saturate: bool) -> Self {
        Self { saturate, ..self }
    }
}

/// The exception flags and the extra status that an operation reports.
///
/// Combine flags with `|`. The five IEEE 754 flags follow the default,
/// non-trapping behavior.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Flags(u8);

impl Flags {
    /// No flags.
    pub const NONE: Self = Self(0);
    /// IEEE invalid operation.
    pub const INVALID: Self = Self(1);
    /// IEEE division by zero.
    pub const DIVIDE_BY_ZERO: Self = Self(1 << 1);
    /// IEEE overflow.
    pub const OVERFLOW: Self = Self(1 << 2);
    /// IEEE underflow: the result is tiny and inexact.
    pub const UNDERFLOW: Self = Self(1 << 3);
    /// IEEE inexact.
    pub const INEXACT: Self = Self(1 << 4);
    /// The result is tiny, even when exact. A consumer needs this flag to
    /// emulate an unmasked underflow exception.
    pub const TINY: Self = Self(1 << 5);
    /// The magnitude of the result is larger than the magnitude of the exact
    /// value. The x87 unit reports this in status bit C1.
    pub const ROUNDED_UP: Self = Self(1 << 6);
    /// An input was subnormal, before denormals-are-zero. x86 reports this as
    /// DE, and ARM reports a flushed input as IDC.
    pub const DENORMAL_INPUT: Self = Self(1 << 7);

    const NAMES: [(Self, &'static str); 8] = [
        (Self::INVALID, "INVALID"),
        (Self::DIVIDE_BY_ZERO, "DIVIDE_BY_ZERO"),
        (Self::OVERFLOW, "OVERFLOW"),
        (Self::UNDERFLOW, "UNDERFLOW"),
        (Self::INEXACT, "INEXACT"),
        (Self::TINY, "TINY"),
        (Self::ROUNDED_UP, "ROUNDED_UP"),
        (Self::DENORMAL_INPUT, "DENORMAL_INPUT"),
    ];

    /// Returns `true` when every flag of `other` is set in `self`.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns `true` when no flag is set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the flags of both values.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the flags of `self` that are not in `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl BitOr for Flags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for Flags {
    fn bitor_assign(&mut self, other: Self) {
        *self = self.union(other);
    }
}

impl Debug for Flags {
    /// Writes the set flags, for example `Flags(INEXACT | UNDERFLOW)`.
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("Flags(")?;
        let mut first = true;
        for (flag, name) in Self::NAMES {
            if self.contains(flag) {
                if !first {
                    formatter.write_str(" | ")?;
                }
                formatter.write_str(name)?;
                first = false;
            }
        }
        if first {
            formatter.write_str("NONE")?;
        }
        formatter.write_str(")")
    }
}

/// A change to the behavior of one operation.
///
/// A [`Rounding`] replaces the rounding direction and keeps every other field
/// of the type's `Env`. An [`Env`] replaces the whole behavior. The trait is
/// sealed.
pub trait Override: Sealed {
    /// Applies the change to the behavior of the type.
    #[doc(hidden)]
    fn apply(self, base: Env) -> Env;
}

impl Sealed for Rounding {}
impl Override for Rounding {
    fn apply(self, base: Env) -> Env {
        base.with_rounding(self)
    }
}

impl Sealed for Env {}
impl Override for Env {
    fn apply(self, _base: Env) -> Env {
        self
    }
}

/// A named, constant behavior that a [`Float`](crate::Float) type uses by
/// default. The trait is sealed. The modes are in [`mode`].
pub trait Mode: Sealed + 'static {
    /// The behavior.
    const ENV: Env;
}

/// The modes: named behaviors for [`Float`](crate::Float) types.
///
/// The modes live in this module because the encodings already use the names
/// [`Ieee`](crate::Ieee) and [`X87`](crate::X87) at the crate root.
pub mod mode {
    use super::{Env, Mode};
    use crate::sealed::Sealed;

    /// The IEEE 754 default behavior, [`Env::IEEE`]. It is the default mode of
    /// every [`Float`](crate::Float) type.
    pub enum Ieee {}

    impl Sealed for Ieee {}
    impl Mode for Ieee {
        const ENV: Env = Env::IEEE;
    }
}
