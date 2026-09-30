//! Behavior: `Env` and its presets, `Rounding`, `Tininess`, `NanRule`, `Flags`,
//! overrides, and the sealed modes.

use core::fmt::{self, Debug, Formatter};
use core::num::NonZeroU32;
use core::ops::{BitOr, BitOrAssign};

use crate::sealed::Sealed;

/// A rounding direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Rounding {
    /// Round to the nearest value. A tie goes to the even significand.
    TiesToEven,
    /// Round to the nearest value. A tie goes away from zero.
    TiesToAway,
    /// Round to the nearest value. A tie goes toward zero. The decimal unit
    /// of IBM POWER has this direction, and decNumber calls it `half_down`.
    TiesTowardZero,
    /// Round toward positive infinity.
    TowardPositive,
    /// Round toward negative infinity.
    TowardNegative,
    /// Round toward zero.
    TowardZero,
    /// Round away from zero. The decimal unit of IBM POWER has this
    /// direction, and decNumber calls it `up`.
    AwayFromZero,
    /// Round toward zero, then add one unit in the last place when the result
    /// is inexact and its last digit is 0, or 5 in a decimal format. For a
    /// binary format, the rule sets the lowest significand bit. For a decimal
    /// format, the rule is IBM's round to prepare for shorter precision. The
    /// result never overflows to an infinity.
    ToOdd,
}

/// When a binary result is tiny, for the underflow flag and for
/// flush-to-zero. A decimal result is always tiny before rounding, as IEEE 754
/// requires.
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
    /// The invalid product signals invalid, and the NaN addend takes
    /// precedence over its default NaN. The propagation rule selects the
    /// result from the addend alone. QEMU's PowerPC target gives this result
    /// for `fmadd` and `fmsub`.
    SignalsAndYieldsToNan,
}

/// The order in which a fused multiply-add offers its NaN operands to the
/// propagation rule.
///
/// IEEE 754-2019 section 6.2.3 does not say which input NaN gives the
/// payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum FusedNanOrder {
    /// The rule selects a NaN from the two factors, which it makes quiet. It
    /// then selects from that NaN and the addend. SoftFloat uses this order.
    /// With [`NanPropagation::FirstOperand`] it gives the first NaN of the
    /// three operands, as x86 does.
    ProductFirst,
    /// The rule selects from the addend, the first factor, and the second
    /// factor, in this order, with no quieting between them. With
    /// [`NanPropagation::SignalingFirst`] it gives the first signaling NaN of
    /// that order, or else its first NaN. The `FPMulAdd` pseudocode of the Arm
    /// Architecture Reference Manual passes its operands to `FPProcessNaNs3`
    /// in this order.
    AddendFirst,
    /// The rule selects from the first factor, the addend, and the second
    /// factor, in this order, with no quieting between them. With
    /// [`NanPropagation::FirstOperand`] it gives the first NaN of that order,
    /// as QEMU's PowerPC target gives for `fmadd` and `fmsub`.
    AddendSecond,
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
    /// The order in which a fused multiply-add offers its NaN operands.
    pub fused_order: FusedNanOrder,
}

impl NanRule {
    /// Returns a rule with this propagation, a positive default NaN,
    /// [`InvalidProduct::Signals`], and [`FusedNanOrder::ProductFirst`].
    #[must_use]
    #[inline]
    pub const fn new(propagation: NanPropagation) -> Self {
        Self {
            propagation,
            default_negative: false,
            invalid_product: InvalidProduct::Signals,
            fused_order: FusedNanOrder::ProductFirst,
        }
    }

    /// Returns a copy with another propagation rule.
    #[must_use]
    #[inline]
    pub const fn with_propagation(self, propagation: NanPropagation) -> Self {
        Self {
            propagation,
            ..self
        }
    }

    /// Returns a copy with another sign of the default NaN.
    #[must_use]
    #[inline]
    pub const fn with_default_negative(self, default_negative: bool) -> Self {
        Self {
            default_negative,
            ..self
        }
    }

    /// Returns a copy with another rule for `0 * inf + NaN`.
    #[must_use]
    #[inline]
    pub const fn with_invalid_product(self, invalid_product: InvalidProduct) -> Self {
        Self {
            invalid_product,
            ..self
        }
    }

    /// Returns a copy with another order of the NaN operands of a fused
    /// multiply-add.
    #[must_use]
    #[inline]
    pub const fn with_fused_order(self, fused_order: FusedNanOrder) -> Self {
        Self {
            fused_order,
            ..self
        }
    }
}

/// How [`total_cmp`](crate::Float::total_cmp) orders two encodings of one
/// datum: a non-canonical encoding and the canonical encoding of its value.
///
/// Only an x87 pseudo-denormal and the non-canonical decimal encodings have
/// such a twin. The other encodings order the same way under both rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TotalOrder {
    /// Two encodings of one datum are equal. The note in IEEE 754-2019
    /// section 5.10 states that `totalOrder` does not distinguish them.
    /// decNumber and the Intel decimal library follow this rule.
    Datum,
    /// Two encodings of one datum order by their bits: by the sign bit, and
    /// then by the other bits, reversed for a negative sign. Two different
    /// encodings are then never equal.
    Encoding,
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
    /// FTZ: a tiny result becomes a zero with the same sign, and reports
    /// underflow, inexact, and tiny, as the SSE unit does.
    pub flush_to_zero: bool,
    /// DAZ: a subnormal input reads as a zero with the same sign.
    pub denormals_are_zero: bool,
    /// When a binary result is tiny. A decimal result is tiny before
    /// rounding, as IEEE 754 requires.
    pub tininess: Tininess,
    /// Which NaN an operation returns.
    pub nan: NanRule,
    /// Rounds the significand to this many digits of the radix and keeps the
    /// exponent range of the format, as x87 precision control does. A tiny
    /// result then rounds at the quantum `RADIX^(EMIN - precision + 1)`.
    /// `None` uses the precision of the format. A value above the format
    /// precision has no effect.
    pub precision: Option<NonZeroU32>,
    /// A finite result that overflows gives the largest finite value of its
    /// sign, in every rounding direction, instead of an infinity or a NaN.
    /// OCP FP8 conversions call this saturation. An infinite operand and an
    /// exact infinite result, such as `1 / 0`, stay infinite in a format that
    /// has an infinity. The field applies to the binary formats.
    pub saturate: bool,
    /// How `total_cmp` orders two encodings of one datum.
    pub total_order: TotalOrder,
}

impl Env {
    /// The IEEE 754 default behavior: round to nearest even, no flushing,
    /// tininess after rounding, and the
    /// [`SignalingFirst`](NanPropagation::SignalingFirst) NaN rule with a
    /// positive default NaN.
    pub const IEEE: Self = Self {
        rounding: Rounding::TiesToEven,
        flush_to_zero: false,
        denormals_are_zero: false,
        tininess: Tininess::AfterRounding,
        nan: NanRule::new(NanPropagation::SignalingFirst),
        precision: None,
        saturate: false,
        total_order: TotalOrder::Datum,
    };

    /// The x86 SSE unit at reset, for SSE, SSE2, and AVX scalar and packed
    /// arithmetic.
    ///
    /// The fields cite the Intel SDM Volume 1, order number 253665-093US. The
    /// `floaty-verify` crate confirms each field on the processor. A consumer
    /// overrides the fields that MXCSR changes: the rounding control, FTZ,
    /// and DAZ.
    ///
    /// The DE flag of MXCSR is [`Flags::DENORMAL_INPUT`] with two
    /// exceptions. A NaN operand, an invalid operation, or a division by zero
    /// comes first, by the priority of section 4.9.2, page 4-24. DAZ also
    /// clears DE, by section 10.2.3.4, page 10-5. That mapping is the
    /// consumer's.
    pub const X86_SSE: Self = Self {
        // MXCSR is 1F80H after power-up or reset: round to nearest, FTZ and
        // DAZ clear. Table 11-2, page 11-20.
        rounding: Rounding::TiesToEven,
        flush_to_zero: false,
        denormals_are_zero: false,
        // Tininess is detected on the result of rounding with an unbounded
        // exponent. Section 4.9.1.5, page 4-23.
        tininess: Tininess::AfterRounding,
        // Two NaN operands give the first source operand, made quiet. Table
        // 4-8, page 4-17. An invalid operation gives the QNaN floating-point
        // indefinite, sections 4.8.3.5 and 4.8.3.7, pages 4-17 and 4-18. It
        // is negative, Table 4-3, page 4-6. A NaN operand comes before the
        // invalid operation of 0 * inf. Section 4.9.2, page 4-24.
        nan: NanRule::new(NanPropagation::FirstOperand)
            .with_default_negative(true)
            .with_invalid_product(InvalidProduct::YieldsToNan),
        // MXCSR has no precision-control field. Section 10.2.3, Figure 10-3,
        // page 10-4.
        precision: None,
        // The SSE formats of Volume 1 have an infinity. Table 4-3, page 4-6.
        saturate: false,
        // The unit has no total-order instruction, and every SSE encoding is
        // canonical. The field keeps the rule of IEEE 754-2019 section 5.10.
        total_order: TotalOrder::Datum,
    };

    /// The x87 unit after `FNINIT`, for arithmetic on x87 extended-precision
    /// values.
    ///
    /// The fields cite the Intel SDM Volume 1, order number 253665-093US. The
    /// `floaty-verify` crate confirms each field that the processor can show.
    /// A consumer overrides the fields that the control word changes: the
    /// rounding control and the precision control. The DE flag of the status
    /// word follows [`Flags::DENORMAL_INPUT`] by the priority of section
    /// 4.9.2, and C1 is [`Flags::ROUNDED_UP`].
    ///
    /// The precision limit of 64 bits is the full precision of the x87
    /// format, so it changes no x87 result. It also limits a wider format:
    /// binary128 arithmetic under this behavior rounds to 64 bits. The preset
    /// is for x87 values.
    pub const X87: Self = Self {
        // FNINIT sets the control word to 037FH: round to nearest and a
        // 64-bit precision. Section 8.1.5, page 8-7.
        rounding: Rounding::TiesToEven,
        // The control word has no FTZ or DAZ field. Section 8.1.5, Figure 8-6,
        // page 8-7.
        flush_to_zero: false,
        denormals_are_zero: false,
        // Tininess is detected on the result of rounding with an unbounded
        // exponent, at the precision control. Section 4.9.1.5, page 4-23.
        tininess: Tininess::AfterRounding,
        // A QNaN before an SNaN, and the larger significand between two NaNs
        // of one kind. Table 4-8, page 4-17. The processor gives the positive
        // NaN between equal significands. An invalid operation gives the
        // negative QNaN floating-point indefinite, sections 4.8.3.5 and
        // 4.8.3.7 and Table 4-3, page 4-6. The x87 unit has no fused
        // multiply-add, so no processor confirms `invalid_product`. The value
        // follows the priority of section 4.9.2, page 4-24: a NaN operand
        // comes before an invalid operation.
        nan: NanRule::new(NanPropagation::LargerSignificand)
            .with_default_negative(true)
            .with_invalid_product(InvalidProduct::YieldsToNan),
        // The precision control is 64 bits after FNINIT, the full precision of
        // the x87 format. Section 8.1.5.2, page 8-7.
        precision: NonZeroU32::new(64),
        // The x87 format has an infinity. Table 4-3, page 4-6.
        saturate: false,
        // The unit has no total-order instruction. The field keeps the rule
        // of IEEE 754-2019 section 5.10.
        total_order: TotalOrder::Datum,
    };

    /// Returns a copy with another rounding direction.
    #[must_use]
    #[inline]
    pub const fn with_rounding(self, rounding: Rounding) -> Self {
        Self { rounding, ..self }
    }

    /// Returns a copy with flush-to-zero set or clear.
    #[must_use]
    #[inline]
    pub const fn with_flush_to_zero(self, flush_to_zero: bool) -> Self {
        Self {
            flush_to_zero,
            ..self
        }
    }

    /// Returns a copy with denormals-are-zero set or clear.
    #[must_use]
    #[inline]
    pub const fn with_denormals_are_zero(self, denormals_are_zero: bool) -> Self {
        Self {
            denormals_are_zero,
            ..self
        }
    }

    /// Returns a copy with another rule for two encodings of one datum in
    /// `total_cmp`.
    #[must_use]
    #[inline]
    pub const fn with_total_order(self, total_order: TotalOrder) -> Self {
        Self {
            total_order,
            ..self
        }
    }

    /// Returns a copy with another tininess rule.
    #[must_use]
    #[inline]
    pub const fn with_tininess(self, tininess: Tininess) -> Self {
        Self { tininess, ..self }
    }

    /// Returns a copy with another NaN rule.
    #[must_use]
    #[inline]
    pub const fn with_nan(self, nan: NanRule) -> Self {
        Self { nan, ..self }
    }

    /// Returns a copy with another precision limit.
    #[must_use]
    #[inline]
    pub const fn with_precision(self, precision: Option<NonZeroU32>) -> Self {
        Self { precision, ..self }
    }

    /// Returns a copy with saturation set or clear.
    #[must_use]
    #[inline]
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
    /// The result is tiny, even when exact: nonzero and below the smallest
    /// normal magnitude, by the tininess rule of the behavior. A rounded
    /// result and a remainder report it. The operations that return an
    /// operand or its neighbor, such as the minimum and maximum operations
    /// and `next_up`, do not. A consumer needs this flag to emulate an
    /// unmasked underflow exception.
    pub const TINY: Self = Self(1 << 5);
    /// The magnitude of the result is larger than the magnitude of the exact
    /// value. The x87 unit reports this in status bit C1.
    pub const ROUNDED_UP: Self = Self(1 << 6);
    /// An input was subnormal, before denormals-are-zero. Every operation that
    /// takes a behavior reports it. x86 reports this as DE, and ARM reports a
    /// flushed input as IDC. Each instruction has its own rules for DE, which
    /// a consumer maps from this flag.
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
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns `true` when no flag is set.
    #[must_use]
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the flags of both values.
    #[must_use]
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Returns the flags of `self` that are not in `other`.
    #[must_use]
    #[inline]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl BitOr for Flags {
    type Output = Self;

    #[inline]
    fn bitor(self, other: Self) -> Self {
        self.union(other)
    }
}

impl BitOrAssign for Flags {
    #[inline]
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

/// The behavior that an operation runs under.
///
/// An [`Env`] is a behavior that the program chooses at run time: the engine
/// reads its fields. A mode, such as [`mode::X86Sse`], is a behavior fixed at
/// compile time: its fields are constants. So each mode gets its own compiled
/// copy of an operation, without the branches that the constants decide. The
/// trait is sealed.
pub trait Behavior: Sealed + Copy {
    /// Returns the fields of the behavior.
    #[doc(hidden)]
    fn env(self) -> Env;
}

impl Sealed for Env {}
impl Behavior for Env {
    #[inline]
    fn env(self) -> Env {
        self
    }
}

impl<M: Mode> Behavior for M {
    #[inline]
    fn env(self) -> Env {
        M::ENV
    }
}

/// A change to the behavior of one operation.
///
/// A [`Rounding`] replaces the rounding direction and keeps every other field
/// of the type's mode. A [`TotalOrder`] replaces the total order in the same
/// way. A [`Behavior`], an [`Env`] or a mode, replaces the whole behavior. A
/// mode keeps the operation fixed at compile time; the other changes give an
/// `Env`. The trait is sealed.
pub trait Override: Sealed {
    /// The behavior that the change gives.
    #[doc(hidden)]
    type Behavior: Behavior;

    /// Applies the change to the mode `M` of the type.
    #[doc(hidden)]
    fn apply<M: Mode>(self) -> Self::Behavior;
}

impl<B: Behavior> Override for B {
    type Behavior = B;

    #[inline]
    fn apply<M: Mode>(self) -> B {
        self
    }
}

impl Sealed for Rounding {}
impl Override for Rounding {
    type Behavior = Env;

    #[inline]
    fn apply<M: Mode>(self) -> Env {
        M::ENV.with_rounding(self)
    }
}

impl Sealed for TotalOrder {}
impl Override for TotalOrder {
    type Behavior = Env;

    #[inline]
    fn apply<M: Mode>(self) -> Env {
        M::ENV.with_total_order(self)
    }
}

/// A named, constant behavior. A [`Float`](crate::Float) type uses its mode
/// when a call does not override the behavior. A mode is a type of size zero,
/// and its value `M::default()` is a [`Behavior`] that a call can pass. The
/// trait is sealed. The modes are in [`mode`].
pub trait Mode: Sealed + Copy + Default + 'static {
    /// The behavior.
    const ENV: Env;
}

pub mod mode;
