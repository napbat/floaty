//! Safe wrappers over decNumber's fixed-size DPD formats.
//!
//! `build.rs` builds decNumber 3.68 from its pinned release archive.
//! `decSingle`, `decDouble`, and `decQuad` hold decimal32, decimal64, and
//! decimal128 in the DPD encoding. Every function here takes and returns raw
//! encodings: `u32`, `u64`, and `u128`.
//!
//! [`Double`] and [`Quad`] have the operations of decTest's `dd` and `dq`
//! files, through [`Arithmetic`]. `decSingle` has only conversions, so the
//! [`Arithmetic`] operations of [`Single`] run on decNumber's
//! arbitrary-precision numbers in the decimal32 context. [`Widening`]
//! converts [`Single`] and [`Double`] to and from the next wider format.
//! [`limited`] rounds an operation to fewer digits than a format has.
//!
//! A call takes a [`Rounding`] and returns the decNumber status of the
//! operation, the decTest conditions, in an [`Outcome`]. The fixed-size
//! formats ignore every other field of the decNumber context, and they
//! never report the conditions in [`Status::UNREPORTED`]. The operations of
//! [`Single`] drop those conditions too.

mod ffi;

use core::cmp::Ordering;
use core::fmt;
use core::ops::{BitOr, BitOrAssign};
use core::str::FromStr;

/// A rounding mode of decNumber, in the order of its `enum rounding`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// Round toward positive infinity: `DEC_ROUND_CEILING`.
    Ceiling,
    /// Round away from zero: `DEC_ROUND_UP`.
    Up,
    /// Round to nearest, with a tie away from zero: `DEC_ROUND_HALF_UP`.
    HalfUp,
    /// Round to nearest, with a tie to an even digit: `DEC_ROUND_HALF_EVEN`.
    HalfEven,
    /// Round to nearest, with a tie toward zero: `DEC_ROUND_HALF_DOWN`.
    HalfDown,
    /// Round toward zero: `DEC_ROUND_DOWN`.
    Down,
    /// Round toward negative infinity: `DEC_ROUND_FLOOR`.
    Floor,
    /// Round toward zero, then away from zero when the last digit is 0 or 5:
    /// `DEC_ROUND_05UP`.
    ZeroFiveUp,
}

impl Rounding {
    /// Every rounding mode.
    pub const ALL: [Self; 8] = [
        Self::Ceiling,
        Self::Up,
        Self::HalfUp,
        Self::HalfEven,
        Self::HalfDown,
        Self::Down,
        Self::Floor,
        Self::ZeroFiveUp,
    ];

    /// Returns the name that decTest's `rounding` directive uses.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Ceiling => "ceiling",
            Self::Up => "up",
            Self::HalfUp => "half_up",
            Self::HalfEven => "half_even",
            Self::HalfDown => "half_down",
            Self::Down => "down",
            Self::Floor => "floor",
            Self::ZeroFiveUp => "05up",
        }
    }

    /// Returns the value of the mode in decNumber's `enum rounding`.
    const fn code(self) -> u32 {
        match self {
            Self::Ceiling => 0,
            Self::Up => 1,
            Self::HalfUp => 2,
            Self::HalfEven => 3,
            Self::HalfDown => 4,
            Self::Down => 5,
            Self::Floor => 6,
            Self::ZeroFiveUp => 7,
        }
    }
}

/// The error for a rounding name that decTest does not define.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UnknownRoundingError(String);

impl fmt::Display for UnknownRoundingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unknown decTest rounding mode `{}`", self.0)
    }
}

impl core::error::Error for UnknownRoundingError {}

impl FromStr for Rounding {
    type Err = UnknownRoundingError;

    /// Parses a decTest rounding name. The name is case-independent.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .into_iter()
            .find(|rounding| rounding.name().eq_ignore_ascii_case(name))
            .ok_or_else(|| UnknownRoundingError(name.to_owned()))
    }
}

/// A set of decNumber status flags, which decTest calls conditions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Status(u32);

impl Status {
    /// No condition.
    pub const NONE: Self = Self(0);
    /// `DEC_Conversion_syntax`: a string is not a number.
    pub const CONVERSION_SYNTAX: Self = Self(0x1);
    /// `DEC_Division_by_zero`.
    pub const DIVISION_BY_ZERO: Self = Self(0x2);
    /// `DEC_Division_impossible`: an integer quotient has too many digits.
    pub const DIVISION_IMPOSSIBLE: Self = Self(0x4);
    /// `DEC_Division_undefined`: zero divided by zero.
    pub const DIVISION_UNDEFINED: Self = Self(0x8);
    /// `DEC_Insufficient_storage`.
    pub const INSUFFICIENT_STORAGE: Self = Self(0x10);
    /// `DEC_Inexact`.
    pub const INEXACT: Self = Self(0x20);
    /// `DEC_Invalid_context`.
    pub const INVALID_CONTEXT: Self = Self(0x40);
    /// `DEC_Invalid_operation`.
    pub const INVALID_OPERATION: Self = Self(0x80);
    /// `DEC_Lost_digits`. Only the X3.274 subset arithmetic sets it; this
    /// build of decNumber does not.
    pub const LOST_DIGITS: Self = Self(0x100);
    /// `DEC_Overflow`.
    pub const OVERFLOW: Self = Self(0x200);
    /// `DEC_Clamped`: the exponent of the result changed to fit the format.
    pub const CLAMPED: Self = Self(0x400);
    /// `DEC_Rounded`: the result has fewer digits than the exact value, which
    /// can still be exact.
    pub const ROUNDED: Self = Self(0x800);
    /// `DEC_Subnormal`: the result is subnormal.
    pub const SUBNORMAL: Self = Self(0x1000);
    /// `DEC_Underflow`.
    pub const UNDERFLOW: Self = Self(0x2000);

    /// The conditions that IEEE 754 reports as the invalid operation flag,
    /// as decNumber's `DEC_IEEE_754_Invalid_operation` groups them.
    pub const IEEE_INVALID_OPERATION: Self = Self(
        Self::CONVERSION_SYNTAX.0
            | Self::DIVISION_IMPOSSIBLE.0
            | Self::DIVISION_UNDEFINED.0
            | Self::INSUFFICIENT_STORAGE.0
            | Self::INVALID_CONTEXT.0
            | Self::INVALID_OPERATION.0,
    );

    /// The informational conditions that the fixed-size formats never set.
    /// The decNumber manual, in its decFloats chapter, states that
    /// `DEC_Inexact` is the only informational flag they set.
    pub const UNREPORTED: Self = Self(Self::CLAMPED.0 | Self::ROUNDED.0 | Self::SUBNORMAL.0);

    /// Each condition and its decTest name.
    const NAMES: [(Self, &'static str); 14] = [
        (Self::CLAMPED, "Clamped"),
        (Self::CONVERSION_SYNTAX, "Conversion_syntax"),
        (Self::DIVISION_BY_ZERO, "Division_by_zero"),
        (Self::DIVISION_IMPOSSIBLE, "Division_impossible"),
        (Self::DIVISION_UNDEFINED, "Division_undefined"),
        (Self::INEXACT, "Inexact"),
        (Self::INSUFFICIENT_STORAGE, "Insufficient_storage"),
        (Self::INVALID_CONTEXT, "Invalid_context"),
        (Self::INVALID_OPERATION, "Invalid_operation"),
        (Self::LOST_DIGITS, "Lost_digits"),
        (Self::OVERFLOW, "Overflow"),
        (Self::ROUNDED, "Rounded"),
        (Self::SUBNORMAL, "Subnormal"),
        (Self::UNDERFLOW, "Underflow"),
    ];

    /// Returns the condition with a decTest name. The name is
    /// case-independent.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::NAMES
            .into_iter()
            .find(|(_, known)| known.eq_ignore_ascii_case(name))
            .map(|(condition, _)| condition)
    }

    /// Returns the decNumber status bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Returns whether no condition is set.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns whether every condition of `other` is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns whether a condition of `other` is set.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Returns the conditions without those of `other`.
    #[must_use]
    pub const fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

impl BitOr for Status {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl BitOrAssign for Status {
    fn bitor_assign(&mut self, other: Self) {
        self.0 |= other.0;
    }
}

impl fmt::Display for Status {
    /// Writes the decTest names of the conditions, separated by spaces.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut separator = "";
        for (condition, name) in Self::NAMES {
            if self.contains(condition) {
                write!(formatter, "{separator}{name}")?;
                separator = " ";
            }
        }
        let unknown = self.0
            & !Self::NAMES
                .iter()
                .fold(0, |all, (condition, _)| all | condition.0);
        if unknown != 0 {
            write!(formatter, "{separator}{unknown:#x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Status {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Status({self})")
    }
}

/// The result of a decNumber operation and the conditions it raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome<T> {
    /// The result.
    pub value: T,
    /// The conditions that the operation raised.
    pub status: Status,
}

/// A decNumber operation with one operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unary {
    /// `Abs`: the absolute value, rounded.
    Abs,
    /// `Canonical`: the canonical encoding of the operand.
    Canonical,
    /// `Copy`: the operand, unchanged.
    Copy,
    /// `CopyAbs`: the operand with a positive sign.
    CopyAbs,
    /// `CopyNegate`: the operand with the opposite sign.
    CopyNegate,
    /// `Invert`: the digit-wise logical inversion.
    Invert,
    /// `LogB`: the adjusted exponent, as a decimal number.
    LogB,
    /// `Minus`: zero minus the operand.
    Minus,
    /// `NextMinus`: the next value toward negative infinity.
    NextMinus,
    /// `NextPlus`: the next value toward positive infinity.
    NextPlus,
    /// `Plus`: zero plus the operand.
    Plus,
    /// `Reduce`: the value with trailing zeros of the coefficient removed.
    Reduce,
    /// `ToIntegralExact`: rounding to an integral value, with `Inexact`.
    ToIntegralExact,
}

/// A decNumber operation with two operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binary {
    /// `Add`.
    Add,
    /// `And`: the digit-wise logical and.
    And,
    /// `Compare`: -1, 0, or 1, or a NaN.
    Compare,
    /// `CompareSignal`: `Compare`, and invalid for every NaN.
    CompareSignal,
    /// `CompareTotal`: the total order, as -1, 0, or 1.
    CompareTotal,
    /// `CompareTotalMag`: the total order of the absolute values.
    CompareTotalMag,
    /// `CopySign`: the first operand with the sign of the second.
    CopySign,
    /// `Divide`.
    Divide,
    /// `DivideInteger`: the integer part of the quotient.
    DivideInteger,
    /// `Max`.
    Max,
    /// `MaxMag`: the operand with the larger absolute value.
    MaxMag,
    /// `Min`.
    Min,
    /// `MinMag`: the operand with the smaller absolute value.
    MinMag,
    /// `Multiply`.
    Multiply,
    /// `NextToward`: the next value from the first operand toward the second.
    NextToward,
    /// `Or`: the digit-wise logical or.
    Or,
    /// `Quantize`: the first operand with the exponent of the second.
    Quantize,
    /// `Remainder`: the remainder of the truncated integer division.
    Remainder,
    /// `RemainderNear`: the IEEE 754 remainder.
    RemainderNear,
    /// `Rotate`: the coefficient digits rotated by the second operand.
    Rotate,
    /// `ScaleB`: the first operand times 10 to the power of the second.
    ScaleB,
    /// `Shift`: the coefficient digits shifted by the second operand.
    Shift,
    /// `Subtract`.
    Subtract,
    /// `Xor`: the digit-wise logical exclusive or.
    Xor,
}

/// A fixed-size DPD format of decNumber.
pub trait Format: sealed::Sealed {
    /// The encoding: `u32`, `u64`, or `u128`.
    type Bits: Copy + Eq + fmt::Debug + fmt::LowerHex + Into<u128> + TryFrom<u128>;

    /// The name of the format in decNumber, for example `decDouble`.
    const NAME: &'static str;
    /// The precision in digits.
    const PRECISION: u32;
    /// The largest adjusted exponent.
    const EMAX: i32;
    /// The smallest adjusted exponent of a normal value.
    const EMIN: i32;

    /// Converts a numeric string, rounding with `rounding`.
    ///
    /// # Panics
    ///
    /// Panics when `text` holds a NUL character.
    fn from_string(text: &str, rounding: Rounding) -> Outcome<Self::Bits>;

    /// Returns the scientific string of an encoding, as decTest's `toSci`
    /// writes it.
    fn to_string(x: Self::Bits) -> String;

    /// Returns the engineering string of an encoding, as decTest's `toEng`
    /// writes it.
    fn to_eng_string(x: Self::Bits) -> String;
}

/// The operations of a DPD format.
///
/// [`Double`] and [`Quad`] run the operations of `decDouble` and `decQuad`.
/// [`Single`] runs each operation on decNumber's arbitrary-precision numbers
/// in the decimal32 context: 7 digits, `emax` 96, `emin` -95, and the
/// clamp. A copy of a [`Single`] operand therefore gives its canonical
/// encoding, not its bits. `decSingle` has no fused multiply-add, and
/// `decNumberFMA` ignores a NaN addend when the product is invalid. So
/// [`fma`](Self::fma) of [`Single`] takes a special result from `decDoubleFMA`
/// of the operands widened exactly, and a finite result from
/// [`fma_wide`](Self::fma_wide).
pub trait Arithmetic: Format {
    /// Runs an operation with one operand.
    fn unary(operation: Unary, x: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;

    /// Runs an operation with two operands.
    fn binary(
        operation: Binary,
        x: Self::Bits,
        y: Self::Bits,
        rounding: Rounding,
    ) -> Outcome<Self::Bits>;

    /// Returns `x * y + z`, rounded once.
    fn fma(x: Self::Bits, y: Self::Bits, z: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits>;

    /// Returns whether two encodings have the same exponent, or are both
    /// infinities, or are both NaNs.
    fn same_quantum(x: Self::Bits, y: Self::Bits) -> bool;

    /// Returns the class of an encoding as decTest writes it, for example
    /// `+Normal` or `sNaN`.
    fn class(x: Self::Bits) -> &'static str;

    /// Returns the square root, correctly rounded with `rounding`.
    ///
    /// The fixed-size formats have no square root, and decNumber's
    /// `decNumberSquareRoot` rounds only to nearest even. So the root comes
    /// from `decNumberSquareRoot` at `2p + 4` digits with an unbounded
    /// exponent, and `from_string` rounds it to the format.
    ///
    /// The two roundings give the correctly rounded root. Take `x` with
    /// `p` digits and a root in `[10^k, 10^(k+1))`. The exponent of `x` is
    /// at least `2k - p + 1`, so `x - r^2` is a multiple of `10^(2(k-p+1))`
    /// for every `p`-digit value `r` of that binade. The same holds, divided
    /// by 4, for every midpoint `r` between two such values. A nonzero
    /// difference therefore puts the root at least `10^(k-2p+1) / 8` from
    /// `r`. That is more than half a unit in the last place at `2p + 1`
    /// digits, `10^(k-2p) / 2`. The wide root is then on the same side of
    /// every boundary as the exact root, and equal to a boundary only when
    /// the exact root is. A root is never subnormal and never overflows.
    ///
    /// An exact root has the preferred exponent `floor(e / 2)`, where `e`
    /// is the exponent of `x`, as decNumber and IEEE 754 give it. The
    /// status holds no condition of [`Status::UNREPORTED`].
    fn square_root(x: Self::Bits, rounding: Rounding) -> Outcome<Self::Bits> {
        ffi::square_root::<Self>(x, rounding)
    }

    /// Returns `x * y + z`, rounded once, through decNumber's
    /// arbitrary-precision `decNumberFMA`.
    ///
    /// [`fma`](Self::fma), and `decNumberFMA` at the precision of the
    /// format, round some cases in the wrong direction. When the addend
    /// lies far below a product of more than `p` digits, both take the
    /// addend as a rounding residue. The digits of the product below the
    /// rounding position then replace that residue. An addend of the other
    /// sign that is larger than those digits is lost. For example, decimal64
    /// `9999999999999999E+7 * 9999999999999999E+7 - 9999999999999999E+7`
    /// rounded down gives `9999999999999998E+30`, but the exact value is
    /// below it.
    ///
    /// This function runs `decNumberFMA` at `2p + 3` digits, where the
    /// product has no digits below the rounding position. It rounds with
    /// 05up, which keeps a later rounding to fewer digits correct, and
    /// `from_string` then rounds the value to the format. A zero result is
    /// exact, and its sign comes from a run in `rounding`. An infinite or
    /// NaN operand goes to [`fma`](Self::fma), which decTest checks.
    fn fma_wide(
        x: Self::Bits,
        y: Self::Bits,
        z: Self::Bits,
        rounding: Rounding,
    ) -> Outcome<Self::Bits> {
        let special = [x, y, z].into_iter().any(|bits| {
            let class = Self::class(bits);
            class.ends_with("Infinity") || class.ends_with("NaN")
        });
        if special {
            Self::fma(x, y, z, rounding)
        } else {
            ffi::fma_wide::<Self>(x, y, z, rounding)
        }
    }

    /// Orders two encodings as decNumber's arbitrary-precision
    /// `decNumberCompareTotal` does, or `decNumberCompareTotalMag` when
    /// `magnitude` is set.
    ///
    /// The `CompareTotal` and `CompareTotalMag` operations of the
    /// fixed-size formats order two NaNs of one kind and sign wrongly when
    /// their payloads differ in more than one group of four digits. The
    /// loop over the groups goes on after the first difference, and a later
    /// difference replaces its order. decDouble orders `NaN6039` above
    /// `NaN74744532`, for example.
    fn total_order(x: Self::Bits, y: Self::Bits, magnitude: bool) -> Ordering {
        ffi::total_order::<Self>(x, y, magnitude)
    }
}

/// A fixed-size format that converts to and from the next wider format:
/// [`Single`] and [`Double`], or [`Double`] and [`Quad`].
pub trait Widening: Format {
    /// The next wider format.
    type Wider: Format;

    /// Converts to the wider format, as decNumber's `ToWider` does.
    ///
    /// The conversion is exact and reports no condition. It copies the
    /// trailing significand field. A non-canonical declet or special value
    /// therefore stays non-canonical, and a signaling NaN stays signaling
    /// with the same payload digits.
    fn to_wider(x: Self::Bits) -> <Self::Wider as Format>::Bits;

    /// Converts from the wider format, rounding with `rounding`, as
    /// decNumber's `FromWider` does.
    ///
    /// A NaN is not made quiet and signals nothing. Its payload keeps the
    /// low-order digits that the format holds.
    fn from_wider(x: <Self::Wider as Format>::Bits, rounding: Rounding) -> Outcome<Self::Bits>;
}

/// An operation that [`limited`] rounds to a precision limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Limited {
    /// An operation with two operands: `Add`, `Subtract`, `Multiply`,
    /// `Divide`, or `ScaleB`.
    Binary(Binary),
    /// `x * y + z`.
    Fma,
    /// The square root.
    SquareRoot,
}

/// Returns the result of an operation on finite operands of format `F`,
/// rounded once to `digits` digits, and every condition that it raised.
///
/// The result is the scientific string of a decNumber in the context of a
/// format with `digits` digits and the exponent range of `F`, without the
/// clamp. That format has the rounding position of a precision limit: its
/// subnormal quantum is `10^(emin - digits + 1)`, a result overflows when
/// its adjusted exponent exceeds `emax`, and decNumber reports `Subnormal`
/// for a result that is tiny before rounding. The status keeps `Clamped`,
/// `Rounded`, and `Subnormal`.
///
/// The operation runs at `2p + 4` digits with an unbounded exponent range,
/// in 05up, and `decNumberFromString` rounds that value to the context. An
/// inexact value in 05up ends in a digit other than 0 or 5. It is then on
/// the same side of every boundary of fewer digits as the exact value, and
/// below `10^emin` exactly when the exact value is, so the second rounding
/// is correct. The square root rounds to nearest even at `2p + 4` digits,
/// which is correct as [`Arithmetic::square_root`] shows. A zero result is
/// exact, and its sign comes from a run in `rounding`.
///
/// The direct operations of decNumber are not correct here. An operand with
/// more digits than the context makes `decNumberAdd` lose a far smaller
/// term of the other sign, as [`Arithmetic::fma_wide`] describes, and
/// `decNumberScaleB` does not round such an operand.
///
/// # Panics
///
/// Panics when an operand is an infinity or a NaN, or when the operand
/// count does not match the operation.
#[must_use]
pub fn limited<F: Format>(
    operation: Limited,
    operands: &[F::Bits],
    digits: u32,
    rounding: Rounding,
) -> Outcome<String> {
    ffi::limited::<F>(operation, operands, digits, rounding)
}

/// decimal32 in the DPD encoding: decNumber's `decSingle`.
#[derive(Clone, Copy, Debug)]
pub enum Single {}

/// decimal64 in the DPD encoding: decNumber's `decDouble`.
#[derive(Clone, Copy, Debug)]
pub enum Double {}

/// decimal128 in the DPD encoding: decNumber's `decQuad`.
#[derive(Clone, Copy, Debug)]
pub enum Quad {}

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::Single {}
    impl Sealed for super::Double {}
    impl Sealed for super::Quad {}
}

#[cfg(test)]
mod tests {
    use super::{
        Arithmetic, Binary, Double, Format, Limited, Ordering, Quad, Rounding, Single, Status,
        Unary, Widening, limited,
    };

    #[test]
    fn reads_and_writes_names() {
        for rounding in Rounding::ALL {
            assert_eq!(rounding.name().parse(), Ok(rounding));
        }
        assert_eq!("HALF_EVEN".parse(), Ok(Rounding::HalfEven));
        assert!("half".parse::<Rounding>().is_err());
        assert_eq!(Status::from_name("inexact"), Some(Status::INEXACT));
        assert_eq!(Status::from_name("Inexactly"), None);
        assert_eq!(
            (Status::ROUNDED | Status::INEXACT).to_string(),
            "Inexact Rounded"
        );
        assert!(Status::IEEE_INVALID_OPERATION.contains(Status::CONVERSION_SYNTAX));
        assert_eq!(
            (Status::INEXACT | Status::ROUNDED).without(Status::UNREPORTED),
            Status::INEXACT
        );
    }

    #[test]
    fn converts_encodings() {
        // decTest ddcan003, dsEncode, and dqEncode give these encodings.
        let one = Double::from_string("1", Rounding::HalfEven);
        assert_eq!(one.value, 0x2238_0000_0000_0001);
        assert!(one.status.is_empty());
        assert_eq!(Double::to_string(one.value), "1");
        assert_eq!(
            Single::from_string("-7.50", Rounding::HalfEven).value,
            0xa230_03d0
        );
        assert_eq!(
            Quad::from_string("-7.50", Rounding::HalfEven).value,
            0xa207_8000_0000_0000_0000_0000_0000_03d0
        );
        assert_eq!(Single::to_eng_string(0xa230_03d0), "-7.50");
        let rounded = Single::from_string("1.2345678", Rounding::Down);
        assert_eq!(Single::to_string(rounded.value), "1.234567");
        assert_eq!(rounded.status, Status::INEXACT);
    }

    #[test]
    fn computes() {
        let number = |text| Double::from_string(text, Rounding::HalfEven).value;
        let third = Double::binary(Binary::Divide, number("1"), number("3"), Rounding::HalfEven);
        assert_eq!(Double::to_string(third.value), "0.3333333333333333");
        assert_eq!(third.status, Status::INEXACT);
        let sum = Double::fma(number("2"), number("3"), number("4"), Rounding::HalfEven);
        assert_eq!(Double::to_string(sum.value), "10");
        let minus = Double::unary(Unary::Minus, number("sNaN"), Rounding::HalfEven);
        assert_eq!(Double::to_string(minus.value), "NaN");
        assert_eq!(minus.status, Status::INVALID_OPERATION);
        assert!(Double::same_quantum(number("1.0"), number("2.0")));
        assert!(!Double::same_quantum(number("1.0"), number("2.00")));
        assert_eq!(Double::class(number("-0")), "-Zero");
        let quad = Quad::from_string("1E+6144", Rounding::HalfEven).value;
        assert_eq!(Quad::class(quad), "+Normal");
    }

    #[test]
    fn takes_square_roots() {
        let number = |text| Double::from_string(text, Rounding::HalfEven).value;
        let root = |text, rounding| {
            let outcome = Double::square_root(number(text), rounding);
            (Double::to_string(outcome.value), outcome.status)
        };
        // The root of 2 is 1.41421356237309504880..., so the directions
        // away from zero round up, and 05up does too, after the digit 5.
        for rounding in Rounding::ALL {
            let last = match rounding {
                Rounding::Ceiling | Rounding::Up | Rounding::ZeroFiveUp => '6',
                _ => '5',
            };
            let expected = format!("1.41421356237309{last}");
            assert_eq!(root("2", rounding), (expected, Status::INEXACT));
        }
        // An exact root has the exponent floor(e / 2).
        assert_eq!(
            root("1.00", Rounding::Down),
            (String::from("1.0"), Status::NONE)
        );
        assert_eq!(
            root("4E+2", Rounding::HalfEven),
            (String::from("2E+1"), Status::NONE)
        );
        assert_eq!(
            root("0E+3", Rounding::Floor),
            (String::from("0E+1"), Status::NONE)
        );
        assert_eq!(
            root("-0E-5", Rounding::Floor),
            (String::from("-0.000"), Status::NONE)
        );
        assert_eq!(
            root("-1", Rounding::HalfEven),
            (String::from("NaN"), Status::INVALID_OPERATION)
        );
        assert_eq!(
            root("sNaN12", Rounding::HalfEven),
            (String::from("NaN12"), Status::INVALID_OPERATION)
        );
        let quad = Quad::from_string("1E-6176", Rounding::HalfEven).value;
        assert_eq!(
            Quad::to_string(Quad::square_root(quad, Rounding::HalfEven).value),
            "1E-3088"
        );
    }

    #[test]
    fn works_around_the_fma_residue() {
        // The exact value is 9.99999999999999799999990000000010000001E+45.
        let number = |text| Double::from_string(text, Rounding::HalfEven).value;
        let x = number("9999999999999999E+7");
        let z = number("-9999999999999999E+7");
        let fixed = Double::fma(x, x, z, Rounding::Down);
        assert_eq!(Double::to_string(fixed.value), "9.999999999999998E+45");
        let wide = Double::fma_wide(x, x, z, Rounding::Down);
        assert_eq!(Double::to_string(wide.value), "9.999999999999997E+45");
        assert_eq!(wide.status, Status::INEXACT);
        let wide = Double::fma_wide(x, x, z, Rounding::HalfEven);
        assert_eq!(Double::to_string(wide.value), "9.999999999999998E+45");
        // An exact zero takes its sign from the rounding mode.
        let zero = Double::fma_wide(number("2"), number("3"), number("-6"), Rounding::Floor);
        assert_eq!(
            (Double::to_string(zero.value), zero.status),
            (String::from("-0"), Status::NONE)
        );
        // A special operand goes to decDoubleFMA: 0 * inf + NaN is the NaN.
        let nan = Double::fma_wide(
            number("0"),
            number("Inf"),
            number("NaN7"),
            Rounding::HalfEven,
        );
        assert_eq!(
            (Double::to_string(nan.value), nan.status),
            (String::from("NaN7"), Status::NONE)
        );
    }

    #[test]
    fn orders_nan_payloads() {
        let number = |text| Double::from_string(text, Rounding::HalfEven).value;
        let (small, large) = (number("NaN6039"), number("NaN74744532"));
        let fixed = Double::binary(Binary::CompareTotal, small, large, Rounding::HalfEven);
        assert_eq!(Double::to_string(fixed.value), "1");
        assert_eq!(Double::total_order(small, large, false), Ordering::Less);
        assert_eq!(
            Double::total_order(number("-NaN6039"), large, true),
            Ordering::Less
        );
        assert_eq!(
            Double::total_order(number("-1"), number("-1.0"), false),
            Ordering::Less
        );
        assert_eq!(
            Double::total_order(number("sNaN"), number("NaN"), false),
            Ordering::Less
        );
    }

    #[test]
    fn converts_between_widths() {
        // decSingle 1 widens to decDouble 1, as ddcan003 encodes it.
        let one = Single::from_string("1", Rounding::HalfEven).value;
        assert_eq!(Single::to_wider(one), 0x2238_0000_0000_0001);
        let wide = Double::to_wider(0x2238_0000_0000_0001);
        assert_eq!(wide, Quad::from_string("1", Rounding::HalfEven).value);
        // A signaling NaN widens unchanged.
        let nan = Double::from_string("-sNaN123", Rounding::HalfEven).value;
        assert_eq!(Quad::to_string(Double::to_wider(nan)), "-sNaN123");
        // Narrowing rounds in the direction of the rounding mode.
        let long = Quad::from_string("1.23456789012345678", Rounding::HalfEven).value;
        let narrowed = Double::from_wider(long, Rounding::Ceiling);
        assert_eq!(Double::to_string(narrowed.value), "1.234567890123457");
        assert_eq!(narrowed.status, Status::INEXACT);
        let large = Double::from_string("1E+200", Rounding::HalfEven).value;
        let overflow = Single::from_wider(large, Rounding::HalfEven);
        assert_eq!(Single::to_string(overflow.value), "Infinity");
        assert_eq!(overflow.status, Status::OVERFLOW | Status::INEXACT);
    }

    #[test]
    fn computes_in_decimal32() {
        let number = |text| Single::from_string(text, Rounding::HalfEven).value;
        let third = Single::binary(Binary::Divide, number("1"), number("3"), Rounding::HalfEven);
        assert_eq!(Single::to_string(third.value), "0.3333333");
        assert_eq!(third.status, Status::INEXACT);
        // The clamp folds the exponent 96 down to 90, and `Clamped` is
        // dropped.
        let large = Single::binary(
            Binary::Multiply,
            number("1E+90"),
            number("1E+6"),
            Rounding::HalfEven,
        );
        assert_eq!(
            (Single::to_string(large.value), large.status),
            (String::from("1.000000E+96"), Status::NONE)
        );
        let tiny = Single::binary(
            Binary::Divide,
            number("1E-95"),
            number("10"),
            Rounding::Down,
        );
        assert_eq!(
            (Single::to_string(tiny.value), tiny.status),
            (String::from("1E-96"), Status::NONE)
        );
        assert_eq!(Single::class(tiny.value), "+Subnormal");
        assert!(Single::same_quantum(number("1.0"), number("2.0")));
        let up = Single::unary(Unary::NextPlus, number("9999999E+90"), Rounding::HalfEven);
        assert_eq!(Single::to_string(up.value), "Infinity");
        // A special fused multiply-add follows decDoubleFMA: the quiet NaN
        // addend wins over the invalid product.
        let nan = Single::fma(
            number("0"),
            number("Inf"),
            number("-NaN7"),
            Rounding::HalfEven,
        );
        assert_eq!(
            (Single::to_string(nan.value), nan.status),
            (String::from("-NaN7"), Status::NONE)
        );
        let sum = Single::fma(number("2"), number("3"), number("4"), Rounding::HalfEven);
        assert_eq!(Single::to_string(sum.value), "10");
    }

    #[test]
    fn rounds_to_a_precision_limit() {
        let number = |text| Double::from_string(text, Rounding::HalfEven).value;
        let run = |operation, operands: &[u64], rounding| {
            let outcome = limited::<Double>(operation, operands, 3, rounding);
            (outcome.value, outcome.status)
        };
        // 1000000000000001 - 5 is 999999999999996, below 10^15.
        let add = Limited::Binary(Binary::Add);
        let sum = run(
            add,
            &[number("1000000000000001"), number("-5.0")],
            Rounding::Down,
        );
        assert_eq!(sum.0, "9.99E+14");
        assert!(sum.1.contains(Status::INEXACT));
        // The subnormal quantum is 10^(-383 - 3 + 1).
        let multiply = Limited::Binary(Binary::Multiply);
        let tiny = run(
            multiply,
            &[number("1E-383"), number("0.004")],
            Rounding::HalfEven,
        );
        assert_eq!(tiny.0, "0E-385");
        assert!(
            tiny.1
                .contains(Status::SUBNORMAL | Status::UNDERFLOW | Status::INEXACT)
        );
        let tiny = run(
            multiply,
            &[number("1E-383"), number("0.006")],
            Rounding::HalfEven,
        );
        assert_eq!(tiny.0, "1E-385");
        // An overflow toward zero gives the largest value of three digits.
        let large = run(multiply, &[number("1E+384"), number("10")], Rounding::Down);
        assert_eq!(large.0, "9.99E+384");
        assert!(large.1.contains(Status::OVERFLOW | Status::INEXACT));
        let root = run(Limited::SquareRoot, &[number("2")], Rounding::Ceiling);
        assert_eq!(
            root,
            (String::from("1.42"), Status::INEXACT | Status::ROUNDED)
        );
        // An exact zero takes its sign from the rounding mode.
        let zero = run(add, &[number("5"), number("-5")], Rounding::Floor);
        assert_eq!(zero, (String::from("-0"), Status::NONE));
    }
}
