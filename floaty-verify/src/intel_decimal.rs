//! Safe wrappers over the Intel Decimal Floating-Point Math Library.
//!
//! The build script builds release 2.0 Update 2 of the library from its
//! pinned archive. Each function takes its rounding direction and a pointer
//! to its status flags as arguments, so no state is global. The library works
//! on BID encodings. Every function here takes and returns raw encodings,
//! `u32`, `u64`, and `u128`, and binary floating-point values as their bits:
//! an x87 extended value in the low 80 bits of a `u128`.
//!
//! [`Bid32`], [`Bid64`], and [`Bid128`] implement [`Format`], which holds the
//! operations of one width. An operation returns its result and the flags
//! that it raised in an [`Outcome`]. [`Layout`] builds encodings and draws
//! random operands. [`Signals`], [`Value`], [`narrow`], [`encoding`], and
//! [`to_integer`] map floaty's results to the results of the library.

mod ffi;
mod mapping;
mod operands;

use core::cmp::Ordering;
use core::fmt;
use core::ops::{BitOr, BitOrAssign};

pub use ffi::{
    bid32_to_bid64, bid32_to_bid128, bid64_to_bid32, bid64_to_bid128, bid128_to_bid32,
    bid128_to_bid64,
};
pub use mapping::{
    EQUAL_OPERANDS, Extremum, Signals, Value, encoding, env, equal_operand_choice, narrow,
    to_integer,
};

/// A rounding direction of the library, with its `BID_ROUNDING_*` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// `BID_ROUNDING_TO_NEAREST`: to nearest, with a tie to an even digit.
    TiesToEven,
    /// `BID_ROUNDING_DOWN`: toward negative infinity.
    TowardNegative,
    /// `BID_ROUNDING_UP`: toward positive infinity.
    TowardPositive,
    /// `BID_ROUNDING_TO_ZERO`: toward zero.
    TowardZero,
    /// `BID_ROUNDING_TIES_AWAY`: to nearest, with a tie away from zero.
    TiesToAway,
}

impl Rounding {
    /// Every rounding direction, in the order of the library values.
    pub const ALL: [Self; 5] = [
        Self::TiesToEven,
        Self::TowardNegative,
        Self::TowardPositive,
        Self::TowardZero,
        Self::TiesToAway,
    ];

    /// Returns the `BID_ROUNDING_*` value, which the tests of the library
    /// also use.
    #[must_use]
    pub const fn code(self) -> u32 {
        match self {
            Self::TiesToEven => 0,
            Self::TowardNegative => 1,
            Self::TowardPositive => 2,
            Self::TowardZero => 3,
            Self::TiesToAway => 4,
        }
    }

    /// Returns the direction with a `BID_ROUNDING_*` value.
    #[must_use]
    pub fn from_code(code: u32) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|rounding| rounding.code() == code)
    }
}

impl From<Rounding> for floaty::Rounding {
    fn from(rounding: Rounding) -> Self {
        match rounding {
            Rounding::TiesToEven => Self::TiesToEven,
            Rounding::TowardNegative => Self::TowardNegative,
            Rounding::TowardPositive => Self::TowardPositive,
            Rounding::TowardZero => Self::TowardZero,
            Rounding::TiesToAway => Self::TiesToAway,
        }
    }
}

/// A set of the status flags of the library.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Flags(u32);

impl Flags {
    /// No flag.
    pub const NONE: Self = Self(0);
    /// `BID_INVALID_EXCEPTION`.
    pub const INVALID: Self = Self(0x01);
    /// `BID_DENORMAL_EXCEPTION`. In `readtest.in`, only the conversions from
    /// a subnormal binary value raise it.
    pub const DENORMAL: Self = Self(0x02);
    /// `BID_ZERO_DIVIDE_EXCEPTION`.
    pub const ZERO_DIVIDE: Self = Self(0x04);
    /// `BID_OVERFLOW_EXCEPTION`.
    pub const OVERFLOW: Self = Self(0x08);
    /// `BID_UNDERFLOW_EXCEPTION`.
    pub const UNDERFLOW: Self = Self(0x10);
    /// `BID_INEXACT_EXCEPTION`.
    pub const INEXACT: Self = Self(0x20);

    /// The names of the flags, by bit.
    pub const NAMES: [&str; 6] = [
        "invalid",
        "denormal",
        "zero-divide",
        "overflow",
        "underflow",
        "inexact",
    ];

    /// Returns the flags with the library bits, which `readtest.in` also
    /// uses. Unknown bits stay set.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// Returns the library bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns whether every flag of `other` is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the flags of `self` that are not in `other`.
    #[must_use]
    pub const fn difference(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }

    /// Returns the library flags of the floaty flags: the five IEEE flags,
    /// and [`Self::DENORMAL`] for [`floaty::Flags::DENORMAL_INPUT`]. The
    /// library has no flag for the other floaty flags.
    #[must_use]
    pub fn from_floaty(flags: floaty::Flags) -> Self {
        [
            (floaty::Flags::INVALID, Self::INVALID),
            (floaty::Flags::DENORMAL_INPUT, Self::DENORMAL),
            (floaty::Flags::DIVIDE_BY_ZERO, Self::ZERO_DIVIDE),
            (floaty::Flags::OVERFLOW, Self::OVERFLOW),
            (floaty::Flags::UNDERFLOW, Self::UNDERFLOW),
            (floaty::Flags::INEXACT, Self::INEXACT),
        ]
        .into_iter()
        .filter(|&(ours, _)| flags.contains(ours))
        .fold(Self::NONE, |set, (_, theirs)| set | theirs)
    }
}

impl BitOr for Flags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOrAssign for Flags {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl fmt::Debug for Flags {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Flags({:#04x})", self.0)
    }
}

/// The result of an operation and the flags it raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome<T> {
    /// The result.
    pub value: T,
    /// The flags that the operation raised.
    pub flags: Flags,
}

impl<T> Outcome<T> {
    /// Converts the result and keeps the flags.
    pub fn map<U>(self, convert: impl FnOnce(T) -> U) -> Outcome<U> {
        Outcome {
            value: convert(self.value),
            flags: self.flags,
        }
    }
}

/// The class of a value, in the order of the library's `class_types`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Class {
    /// A signaling NaN.
    SignalingNan,
    /// A quiet NaN.
    QuietNan,
    /// Negative infinity.
    NegativeInfinity,
    /// A negative normal value.
    NegativeNormal,
    /// A negative subnormal value.
    NegativeSubnormal,
    /// Negative zero.
    NegativeZero,
    /// Positive zero.
    PositiveZero,
    /// A positive subnormal value.
    PositiveSubnormal,
    /// A positive normal value.
    PositiveNormal,
    /// Positive infinity.
    PositiveInfinity,
}

impl Class {
    /// Every class.
    pub const ALL: [Self; 10] = [
        Self::SignalingNan,
        Self::QuietNan,
        Self::NegativeInfinity,
        Self::NegativeNormal,
        Self::NegativeSubnormal,
        Self::NegativeZero,
        Self::PositiveZero,
        Self::PositiveSubnormal,
        Self::PositiveNormal,
        Self::PositiveInfinity,
    ];

    /// Returns the `class_types` value.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::SignalingNan => 0,
            Self::QuietNan => 1,
            Self::NegativeInfinity => 2,
            Self::NegativeNormal => 3,
            Self::NegativeSubnormal => 4,
            Self::NegativeZero => 5,
            Self::PositiveZero => 6,
            Self::PositiveSubnormal => 7,
            Self::PositiveNormal => 8,
            Self::PositiveInfinity => 9,
        }
    }

    /// Returns the class with a `class_types` value.
    #[must_use]
    pub fn from_code(code: i32) -> Option<Self> {
        Self::ALL.into_iter().find(|class| class.code() == code)
    }

    /// Returns the class of a floaty class and sign, or `None` for a class
    /// that no decimal value has.
    #[must_use]
    pub fn from_floaty(class: floaty::Class, negative: bool) -> Option<Self> {
        use floaty::Class as Floaty;
        Some(match (class, negative) {
            (Floaty::SignalingNan, _) => Self::SignalingNan,
            (Floaty::QuietNan, _) => Self::QuietNan,
            (Floaty::Infinite, true) => Self::NegativeInfinity,
            (Floaty::Normal, true) => Self::NegativeNormal,
            (Floaty::Subnormal, true) => Self::NegativeSubnormal,
            (Floaty::Zero, true) => Self::NegativeZero,
            (Floaty::Zero, false) => Self::PositiveZero,
            (Floaty::Subnormal, false) => Self::PositiveSubnormal,
            (Floaty::Normal, false) => Self::PositiveNormal,
            (Floaty::Infinite, false) => Self::PositiveInfinity,
            _ => return None,
        })
    }
}

/// A comparison predicate of the library. A quiet predicate signals invalid
/// only for a signaling NaN operand; a signaling one for every NaN operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Predicate {
    /// `quiet_equal`.
    QuietEqual,
    /// `quiet_greater`.
    QuietGreater,
    /// `quiet_greater_equal`.
    QuietGreaterEqual,
    /// `quiet_greater_unordered`: greater, or unordered.
    QuietGreaterUnordered,
    /// `quiet_less`.
    QuietLess,
    /// `quiet_less_equal`.
    QuietLessEqual,
    /// `quiet_less_unordered`: less, or unordered.
    QuietLessUnordered,
    /// `quiet_not_equal`.
    QuietNotEqual,
    /// `quiet_not_greater`.
    QuietNotGreater,
    /// `quiet_not_less`.
    QuietNotLess,
    /// `quiet_ordered`: neither operand is a NaN.
    QuietOrdered,
    /// `quiet_unordered`: an operand is a NaN.
    QuietUnordered,
    /// `signaling_greater`.
    SignalingGreater,
    /// `signaling_greater_equal`.
    SignalingGreaterEqual,
    /// `signaling_greater_unordered`.
    SignalingGreaterUnordered,
    /// `signaling_less`.
    SignalingLess,
    /// `signaling_less_equal`.
    SignalingLessEqual,
    /// `signaling_less_unordered`.
    SignalingLessUnordered,
    /// `signaling_not_greater`.
    SignalingNotGreater,
    /// `signaling_not_less`.
    SignalingNotLess,
}

impl Predicate {
    /// Every predicate.
    pub const ALL: [Self; 20] = [
        Self::QuietEqual,
        Self::QuietGreater,
        Self::QuietGreaterEqual,
        Self::QuietGreaterUnordered,
        Self::QuietLess,
        Self::QuietLessEqual,
        Self::QuietLessUnordered,
        Self::QuietNotEqual,
        Self::QuietNotGreater,
        Self::QuietNotLess,
        Self::QuietOrdered,
        Self::QuietUnordered,
        Self::SignalingGreater,
        Self::SignalingGreaterEqual,
        Self::SignalingGreaterUnordered,
        Self::SignalingLess,
        Self::SignalingLessEqual,
        Self::SignalingLessUnordered,
        Self::SignalingNotGreater,
        Self::SignalingNotLess,
    ];

    /// Returns the library name without the format prefix, for example
    /// `quiet_equal`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::QuietEqual => "quiet_equal",
            Self::QuietGreater => "quiet_greater",
            Self::QuietGreaterEqual => "quiet_greater_equal",
            Self::QuietGreaterUnordered => "quiet_greater_unordered",
            Self::QuietLess => "quiet_less",
            Self::QuietLessEqual => "quiet_less_equal",
            Self::QuietLessUnordered => "quiet_less_unordered",
            Self::QuietNotEqual => "quiet_not_equal",
            Self::QuietNotGreater => "quiet_not_greater",
            Self::QuietNotLess => "quiet_not_less",
            Self::QuietOrdered => "quiet_ordered",
            Self::QuietUnordered => "quiet_unordered",
            Self::SignalingGreater => "signaling_greater",
            Self::SignalingGreaterEqual => "signaling_greater_equal",
            Self::SignalingGreaterUnordered => "signaling_greater_unordered",
            Self::SignalingLess => "signaling_less",
            Self::SignalingLessEqual => "signaling_less_equal",
            Self::SignalingLessUnordered => "signaling_less_unordered",
            Self::SignalingNotGreater => "signaling_not_greater",
            Self::SignalingNotLess => "signaling_not_less",
        }
    }

    /// Returns whether the predicate signals invalid for every NaN operand,
    /// not only for a signaling NaN.
    #[must_use]
    pub const fn is_signaling(self) -> bool {
        matches!(
            self,
            Self::SignalingGreater
                | Self::SignalingGreaterEqual
                | Self::SignalingGreaterUnordered
                | Self::SignalingLess
                | Self::SignalingLessEqual
                | Self::SignalingLessUnordered
                | Self::SignalingNotGreater
                | Self::SignalingNotLess
        )
    }

    /// Returns whether the predicate holds for the order of two operands,
    /// where `None` means unordered, as IEEE 754-2019 Table 5.1 to Table 5.3
    /// define the predicates.
    #[must_use]
    pub fn holds(self, order: Option<Ordering>) -> bool {
        use Ordering::{Equal, Greater, Less};
        match self {
            Self::QuietEqual => order == Some(Equal),
            Self::QuietNotEqual => order != Some(Equal),
            Self::QuietGreater | Self::SignalingGreater => order == Some(Greater),
            Self::QuietGreaterEqual | Self::SignalingGreaterEqual => {
                matches!(order, Some(Greater | Equal))
            }
            Self::QuietGreaterUnordered | Self::SignalingGreaterUnordered => {
                matches!(order, Some(Greater) | None)
            }
            Self::QuietLess | Self::SignalingLess => order == Some(Less),
            Self::QuietLessEqual | Self::SignalingLessEqual => matches!(order, Some(Less | Equal)),
            Self::QuietLessUnordered | Self::SignalingLessUnordered => {
                matches!(order, Some(Less) | None)
            }
            Self::QuietNotGreater | Self::SignalingNotGreater => order != Some(Greater),
            Self::QuietNotLess | Self::SignalingNotLess => order != Some(Less),
            Self::QuietOrdered => order.is_some(),
            Self::QuietUnordered => order.is_none(),
        }
    }
}

/// The integer type of a conversion to an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Integer {
    /// `int8`: `i8`.
    Int8,
    /// `int16`: `i16`.
    Int16,
    /// `int32`: `i32`.
    Int32,
    /// `int64`: `i64`.
    Int64,
    /// `uint8`: `u8`.
    UInt8,
    /// `uint16`: `u16`.
    UInt16,
    /// `uint32`: `u32`.
    UInt32,
    /// `uint64`: `u64`.
    UInt64,
}

impl Integer {
    /// Every integer type.
    pub const ALL: [Self; 8] = [
        Self::Int8,
        Self::Int16,
        Self::Int32,
        Self::Int64,
        Self::UInt8,
        Self::UInt16,
        Self::UInt32,
        Self::UInt64,
    ];

    /// Returns the name in the library function names, for example `uint16`.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
        }
    }

    /// Returns the width in bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        match self {
            Self::Int8 | Self::UInt8 => 8,
            Self::Int16 | Self::UInt16 => 16,
            Self::Int32 | Self::UInt32 => 32,
            Self::Int64 | Self::UInt64 => 64,
        }
    }

    /// Returns the result of an invalid conversion: a NaN, an infinity, or a
    /// value out of range. The library sets only the top bit of the type,
    /// signed or not, as every such line of `readtest.in` expects.
    #[must_use]
    pub const fn indefinite(self) -> i128 {
        let top = 1_i128 << (self.bits() - 1);
        match self {
            Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64 => -top,
            Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64 => top,
        }
    }
}

/// Whether a conversion to an integer signals an inexact result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Inexact {
    /// The functions without `x`, for example `to_int32_rnint`.
    Ignored,
    /// The functions with `x`, for example `to_int32_xrnint`.
    Signaled,
}

/// The operations of one BID format.
///
/// Each method names its library function without the `bidN_` prefix. A
/// method takes a [`Rounding`] when its function has a rounding argument,
/// and returns an [`Outcome`] when its function reports flags.
pub trait Format: sealed::Sealed {
    /// The encoding: `u32`, `u64`, or `u128`.
    type Bits: Copy + Eq + fmt::Debug + fmt::LowerHex + Into<u128> + TryFrom<u128>;

    /// The prefix of the library names, for example `bid64`.
    const NAME: &'static str;

    /// The field layout of the encoding.
    const LAYOUT: Layout;

    /// `add`: `x + y`.
    fn add(x: Self::Bits, y: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `sub`: `x - y`.
    fn sub(x: Self::Bits, y: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `mul`: `x * y`.
    fn mul(x: Self::Bits, y: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `div`: `x / y`.
    fn div(x: Self::Bits, y: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `fma`: `x * y + z`, rounded once.
    fn fma(x: Self::Bits, y: Self::Bits, z: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `sqrt`: the square root.
    fn sqrt(x: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `quantize`: `x` with the exponent of `y`.
    fn quantize(x: Self::Bits, y: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `scalbn`: `x * 10^n`.
    fn scalbn(x: Self::Bits, n: i32, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `round_integral_exact`: rounding to an integral value, with inexact.
    fn round_integral_exact(x: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `nearbyint`: rounding to an integral value, without inexact.
    fn nearbyint(x: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `round_integral_nearest_even`, `round_integral_negative`,
    /// `round_integral_positive`, `round_integral_zero`, or
    /// `round_integral_nearest_away`: rounding to an integral value in a
    /// fixed direction, without inexact.
    fn round_integral(x: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `rem`: the IEEE 754 remainder.
    fn rem(x: Self::Bits, y: Self::Bits) -> Outcome<Self::Bits>;
    /// `fmod`: the remainder of the quotient truncated toward zero.
    fn fmod(x: Self::Bits, y: Self::Bits) -> Outcome<Self::Bits>;
    /// `logb`: the exponent of the most significant digit, as a value.
    fn logb(x: Self::Bits) -> Outcome<Self::Bits>;
    /// `ilogb`: the exponent of the most significant digit, as an integer.
    fn ilogb(x: Self::Bits) -> Outcome<i32>;
    /// `quantum`: one unit in the last place of `x`, with the exponent of `x`.
    fn quantum(x: Self::Bits) -> Outcome<Self::Bits>;
    /// `quantexp`: the exponent of `x`.
    fn quantexp(x: Self::Bits) -> Outcome<i32>;
    /// `nextup`: the next value toward positive infinity.
    fn nextup(x: Self::Bits) -> Outcome<Self::Bits>;
    /// `nextdown`: the next value toward negative infinity.
    fn nextdown(x: Self::Bits) -> Outcome<Self::Bits>;
    /// `minnum`: the IEEE 754-2008 `minNum`.
    fn minnum(x: Self::Bits, y: Self::Bits) -> Outcome<Self::Bits>;
    /// `maxnum`: the IEEE 754-2008 `maxNum`.
    fn maxnum(x: Self::Bits, y: Self::Bits) -> Outcome<Self::Bits>;
    /// `minnum_mag`: the IEEE 754-2008 `minNumMag`.
    fn minnum_mag(x: Self::Bits, y: Self::Bits) -> Outcome<Self::Bits>;
    /// `maxnum_mag`: the IEEE 754-2008 `maxNumMag`.
    fn maxnum_mag(x: Self::Bits, y: Self::Bits) -> Outcome<Self::Bits>;
    /// A comparison predicate.
    fn compare(x: Self::Bits, y: Self::Bits, predicate: Predicate) -> Outcome<bool>;
    /// `totalOrder`: whether `x` is before or equal to `y` in the total order.
    fn total_order(x: Self::Bits, y: Self::Bits) -> bool;
    /// `totalOrderMag`: `total_order` of the absolute values.
    fn total_order_mag(x: Self::Bits, y: Self::Bits) -> bool;
    /// `sameQuantum`: whether the exponents are equal, or both operands are
    /// infinities, or both are NaNs.
    fn same_quantum(x: Self::Bits, y: Self::Bits) -> bool;
    /// `abs`: `x` with a positive sign.
    fn abs(x: Self::Bits) -> Self::Bits;
    /// `negate`: `x` with the opposite sign.
    fn negate(x: Self::Bits) -> Self::Bits;
    /// `copySign`: `x` with the sign of `y`.
    fn copy_sign(x: Self::Bits, y: Self::Bits) -> Self::Bits;
    /// `class`: the class of `x`.
    fn class(x: Self::Bits) -> Class;
    /// `isCanonical`: whether `x` is a canonical encoding.
    fn is_canonical(x: Self::Bits) -> bool;
    /// `to_int8_rnint` through `to_uint64_xceil`: the conversion to an
    /// integer type, rounding in `rounding`. The value widens to `i128`.
    fn to_integer(
        x: Self::Bits,
        integer: Integer,
        rounding: Rounding,
        inexact: Inexact,
    ) -> Outcome<i128>;
    /// `from_int32`. A format with room for every value has no rounding
    /// argument and no flags, and ignores `rounding`.
    fn from_int32(value: i32, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `from_uint32`, with `rounding` as for `from_int32`.
    fn from_uint32(value: u32, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `from_int64`, with `rounding` as for `from_int32`.
    fn from_int64(value: i64, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `from_uint64`, with `rounding` as for `from_int32`.
    fn from_uint64(value: u64, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `bid_to_dpdN`: the DPD encoding of the same value.
    fn to_dpd(x: Self::Bits) -> Self::Bits;
    /// `bid_dpd_to_bidN`: the BID encoding of a DPD encoding.
    fn from_dpd(x: Self::Bits) -> Self::Bits;
    /// `binary32_to_bidN`.
    fn from_binary32(bits: u32, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `binary64_to_bidN`.
    fn from_binary64(bits: u64, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `binary80_to_bidN`, from an x87 extended value in the low 80 bits.
    fn from_binary80(bits: u128, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `binary128_to_bidN`.
    fn from_binary128(bits: u128, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `to_binary32`.
    fn to_binary32(x: Self::Bits, rounding: Rounding) -> Outcome<u32>;
    /// `to_binary64`.
    fn to_binary64(x: Self::Bits, rounding: Rounding) -> Outcome<u64>;
    /// `to_binary80`, to an x87 extended value in the low 80 bits.
    fn to_binary80(x: Self::Bits, rounding: Rounding) -> Outcome<u128>;
    /// `to_binary128`.
    fn to_binary128(x: Self::Bits, rounding: Rounding) -> Outcome<u128>;
    /// `from_string`: the conversion of a numeric string.
    ///
    /// # Panics
    ///
    /// Panics when `text` holds a NUL character.
    fn from_string(text: &str, rounding: Rounding) -> Outcome<Self::Bits>;
    /// `to_string`: the string of an encoding, for example `+16E+0`.
    fn to_string(x: Self::Bits) -> Outcome<String>;
}

/// The field layout of a BID encoding, by IEEE 754-2019 Table 3.6, and
/// random encodings for the differential tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    /// The width `k` in bits: 32, 64, or 128.
    pub width: u32,
}

impl Layout {
    /// The precision `p` in digits, `9 * k / 32 - 2`.
    #[must_use]
    pub const fn precision(self) -> u32 {
        9 * self.width / 32 - 2
    }

    /// The largest adjusted exponent, `emax`, `3 * 2^(k / 16 + 3)`.
    #[must_use]
    pub const fn emax(self) -> i32 {
        3 << (self.width / 16 + 3)
    }

    /// The width `t` of the trailing significand field, `15 * k / 16 - 10`.
    #[must_use]
    pub const fn trailing(self) -> u32 {
        15 * self.width / 16 - 10
    }

    /// The bias of the exponent field, `emax + p - 2`.
    #[must_use]
    pub const fn bias(self) -> u32 {
        self.emax().unsigned_abs() + self.precision() - 2
    }

    /// The largest exponent field, `3 * 2^w - 1` for the `w` bits of the
    /// exponent continuation field.
    #[must_use]
    pub const fn largest_field(self) -> u32 {
        (3 << (self.width / 16 + 4)) - 1
    }

    /// Returns `10^digits`.
    #[must_use]
    pub const fn power_of_ten(digits: u32) -> u128 {
        10_u128.pow(digits)
    }

    /// Returns the encoding of a number with an exponent field and a
    /// coefficient. A coefficient above `10^p - 1` gives a non-canonical
    /// encoding. A coefficient of `2^(t + 3)` or more takes the form with the
    /// implicit bits `100`.
    ///
    /// # Panics
    ///
    /// Panics when no encoding holds the field or the coefficient.
    #[must_use]
    pub fn number(self, negative: bool, field: u32, coefficient: u128) -> u128 {
        let trailing = self.trailing();
        assert!(field <= self.largest_field(), "the exponent field fits");
        let sign = u128::from(negative) << (self.width - 1);
        let field = u128::from(field);
        if coefficient < 1 << (trailing + 3) {
            return sign | (field << (trailing + 3)) | coefficient;
        }
        let low = coefficient - (1 << (trailing + 3));
        assert!(
            low < 1 << (trailing + 1),
            "the coefficient fits the encoding"
        );
        sign | (0b11 << (self.width - 3)) | (field << (trailing + 1)) | low
    }

    /// Returns the exponent field and the coefficient of a number, which can
    /// be above `10^p - 1`, or `None` for an infinity or a NaN.
    #[must_use]
    pub fn fields(self, bits: u128) -> Option<(u32, u128)> {
        let trailing = self.trailing();
        let field_mask = (1 << (self.width / 16 + 6)) - 1;
        if (bits >> (self.width - 5)) & 0b1111 == 0b1111 {
            return None;
        }
        let (field, coefficient) = if (bits >> (self.width - 3)) & 0b11 == 0b11 {
            let low = bits & ((1 << (trailing + 1)) - 1);
            (
                (bits >> (trailing + 1)) & field_mask,
                (1 << (trailing + 3)) | low,
            )
        } else {
            let low = bits & ((1 << (trailing + 3)) - 1);
            ((bits >> (trailing + 3)) & field_mask, low)
        };
        // The mask leaves at most 14 bits, so the field fits.
        Some((u32::try_from(field).ok()?, coefficient))
    }

    /// Returns the canonical encoding of an infinity.
    #[must_use]
    pub const fn infinity(self, negative: bool) -> u128 {
        let sign = if negative { 1 << (self.width - 1) } else { 0 };
        sign | (0b1_1110 << (self.width - 6))
    }

    /// Returns a NaN encoding. `payload` fills the trailing significand
    /// field, and `extra` the bits between it and the signaling bit, which a
    /// canonical NaN has clear.
    #[must_use]
    pub const fn nan(self, negative: bool, signaling: bool, payload: u128, extra: u128) -> u128 {
        let sign = if negative { 1 << (self.width - 1) } else { 0 };
        let signaling = if signaling { 1 << (self.width - 7) } else { 0 };
        let extra_bits = self.width - 7 - self.trailing();
        let extra = (extra & ((1 << extra_bits) - 1)) << self.trailing();
        let payload = payload & ((1 << self.trailing()) - 1);
        sign | (0b1_1111 << (self.width - 6)) | signaling | extra | payload
    }

    /// Returns the canonical encoding of the datum of `bits`. A coefficient
    /// above `10^p - 1` becomes zero, and an infinity loses its trailing
    /// bits. A NaN loses the bits between its signaling bit and its trailing
    /// field, and a payload above `10^(p - 1) - 1` becomes zero.
    #[must_use]
    pub fn canonical(self, bits: u128) -> u128 {
        let negative = bits >> (self.width - 1) == 1;
        if let Some((field, coefficient)) = self.fields(bits) {
            let largest = Self::power_of_ten(self.precision()) - 1;
            let coefficient = if coefficient > largest {
                0
            } else {
                coefficient
            };
            return self.number(negative, field, coefficient);
        }
        if (bits >> (self.width - 6)) & 1 == 0 {
            return self.infinity(negative);
        }
        let signaling = (bits >> (self.width - 7)) & 1 == 1;
        let payload = bits & ((1 << self.trailing()) - 1);
        let largest = Self::power_of_ten(self.precision() - 1) - 1;
        let payload = if payload > largest { 0 } else { payload };
        self.nan(negative, signaling, payload, 0)
    }
}

/// decimal32 in the BID encoding.
#[derive(Clone, Copy, Debug)]
pub enum Bid32 {}

/// decimal64 in the BID encoding.
#[derive(Clone, Copy, Debug)]
pub enum Bid64 {}

/// decimal128 in the BID encoding.
#[derive(Clone, Copy, Debug)]
pub enum Bid128 {}

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::Bid32 {}
    impl Sealed for super::Bid64 {}
    impl Sealed for super::Bid128 {}
}

#[cfg(test)]
mod tests;
