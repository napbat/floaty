//! Arithmetic: the rounded operations, the remainders, and the neighbors of
//! a value, and the operators.

use super::Float;
use crate::env::{Flags, Mode, Override};
use crate::format::Standard;
use crate::format::internal::{Quotient, Step};
use crate::host::{self, Kind, Operation};

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    /// Adds `other`. Returns the sum and the flags.
    ///
    /// `behavior` is a [`Rounding`](crate::Rounding) that overrides only the
    /// rounding direction, or a behavior that replaces the whole behavior: an
    /// [`Env`](crate::Env) chosen at run time, or a mode fixed at compile time,
    /// such as [`mode::X86Sse`](crate::mode::X86Sse). A mode gives the fastest
    /// code, because its fields are constants.
    #[must_use]
    pub fn add_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::add(self.bits, other.bits, false, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Subtracts `other`. Returns the difference and the flags.
    #[must_use]
    pub fn sub_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::add(self.bits, other.bits, true, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Multiplies by `other`. Returns the product and the flags.
    #[must_use]
    pub fn mul_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::mul(self.bits, other.bits, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Divides by `other`. Returns the quotient and the flags.
    ///
    /// A division by zero gives an infinity with the sign of the quotient, or
    /// in a format without an infinity the NaN. A saturating behavior, or a
    /// format with neither, gives the largest finite value. The division
    /// signals divide-by-zero.
    #[must_use]
    pub fn div_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::div(self.bits, other.bits, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns the square root, with the default mode.
    #[must_use]
    #[inline]
    pub fn sqrt(self) -> Self {
        if !host::available(S::HOST, Kind::SquareRoot) {
            return self.sqrt_with(M::default()).0;
        }
        match host::sqrt::<S, W>(self.bits, &M::ENV) {
            Some(bits) => Self::from_masked(bits),
            None => sqrt_in_engine(self),
        }
    }

    /// Returns the square root and the flags.
    #[must_use]
    pub fn sqrt_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::sqrt(self.bits, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns `e^self`, with the default mode.
    #[must_use]
    pub fn exp(self) -> Self {
        self.exp_with(M::default()).0
    }

    /// Returns `e^self`, correctly rounded in the direction of the behavior,
    /// as IEEE 754-2019 `exp` does, and the flags.
    ///
    /// Every finite result but `e^0` is inexact. The result rounds once, so
    /// it has every flag of a rounding: overflow, underflow, `TINY`, and
    /// `ROUNDED_UP`. The special cases follow IEEE 754-2019 section 9.2.1:
    /// `e^±0` is 1 and exact, `e^+inf` is +inf, and `e^-inf` is +0. A NaN
    /// gives the NaN of the NaN rule. The exact decimal results, 1 and the
    /// +0 of `e^-inf`, have the exponent 0, and an inexact decimal result
    /// keeps every digit.
    ///
    /// The value evaluates in ball arithmetic, which bounds every error, at
    /// twice and then four times the bits of the storage. No input is known
    /// that needs more. For binary64 the second precision has 256 bits, and
    /// the published hardest cases need fewer than 160.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (e, flags) = F64::from_bits(0x3FF0_0000_0000_0000).exp_with(Env::IEEE);
    /// assert_eq!((e.to_bits(), flags), (0x4005_BF0A_8B14_5769, Flags::INEXACT));
    /// ```
    #[must_use]
    pub fn exp_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::exp(self.bits, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns `ln self`, with the default mode.
    #[must_use]
    pub fn log(self) -> Self {
        self.log_with(M::default()).0
    }

    /// Returns the natural logarithm `ln self`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `log` does, and the
    /// flags.
    ///
    /// Every finite result but `ln 1` is inexact, and rounds once as
    /// [`exp_with`](Self::exp_with) does. The special cases follow IEEE
    /// 754-2019 section 9.2.1: `ln 1` is +0 and exact, `ln ±0` is -inf and
    /// signals divide-by-zero, `ln +inf` is +inf, and a negative value or
    /// -inf gives the default NaN and signals invalid. A format without an
    /// infinity gives its NaN or its largest finite value for -inf. A NaN
    /// gives the NaN of the NaN rule. A decimal 0 has the exponent 0.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (ln2, flags) = F64::from_bits(0x4000_0000_0000_0000).log_with(Env::IEEE);
    /// assert_eq!((ln2.to_bits(), flags), (0x3FE6_2E42_FEFA_39EF, Flags::INEXACT));
    /// ```
    #[must_use]
    pub fn log_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::log(self.bits, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns `self * multiplier + addend`, rounded once, with the default
    /// mode.
    #[must_use]
    #[inline]
    pub fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        if !host::available(S::HOST, Kind::FusedMultiplyAdd) {
            return self.mul_add_with(multiplier, addend, M::default()).0;
        }
        match host::mul_add::<S, W>(self.bits, multiplier.bits, addend.bits, &M::ENV) {
            Some(bits) => Self::from_masked(bits),
            None => mul_add_in_engine(self, multiplier, addend),
        }
    }

    /// Returns `self * multiplier + addend`, rounded once, and the flags.
    ///
    /// The [`fused_order`](crate::env::NanRule::fused_order) field of the NaN
    /// rule orders the NaN operands, and the
    /// [`invalid_product`](crate::env::NanRule::invalid_product) field decides
    /// `0 * inf + NaN`.
    #[must_use]
    pub fn mul_add_with(
        self,
        multiplier: Self,
        addend: Self,
        behavior: impl Override,
    ) -> (Self, Flags) {
        let (bits, flags) = S::mul_add(
            self.bits,
            multiplier.bits,
            addend.bits,
            behavior.apply::<M>(),
        );
        (Self::from_masked(bits), flags)
    }

    /// Rounds to an integral value in the format, with the default mode.
    #[must_use]
    #[inline]
    pub fn round_to_integral(self) -> Self {
        if !host::available(S::HOST, Kind::RoundToIntegral) {
            return self.round_to_integral_with(M::default()).0;
        }
        match host::round_to_integral::<S, W>(self.bits, &M::ENV) {
            Some(bits) => Self::from_masked(bits),
            None => round_to_integral_in_engine(self),
        }
    }

    /// Rounds to an integral value in the format, in the rounding direction of
    /// the behavior. Returns the result and the flags.
    ///
    /// The flags include [`Flags::INEXACT`] when the result differs from the
    /// value, as IEEE 754 `roundToIntegralExact` signals. The other IEEE 754
    /// operations, such as `roundToIntegralTiesToEven`, are this operation
    /// with that flag ignored. The integer is exact at the full precision,
    /// so the precision limit and flush-to-zero do not apply. In a format
    /// whose largest finite value is below `2^(PRECISION - 1)`, an integer
    /// above it overflows by the usual rules.
    ///
    /// ```
    /// use floaty::{F32, Flags, Rounding};
    ///
    /// let (two, flags) = F32::from_bits(0x3FC0_0000).round_to_integral_with(Rounding::TiesToEven);
    /// assert_eq!((two.to_bits(), flags), (0x4000_0000, Flags::INEXACT | Flags::ROUNDED_UP));
    /// ```
    #[must_use]
    pub fn round_to_integral_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::round_to_integral(self.bits, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns the IEEE 754 remainder, with the default mode.
    #[must_use]
    #[inline]
    pub fn remainder(self, divisor: Self) -> Self {
        if !host::available(S::HOST, Kind::Remainder) {
            return self.remainder_with(divisor, M::default()).0;
        }
        match host::remainder::<S, W>(self.bits, divisor.bits, &M::ENV) {
            Some(bits) => Self::from_masked(bits),
            None => remainder_in_engine(self, divisor),
        }
    }

    /// Returns the IEEE 754 remainder `self - n * divisor`, and the flags.
    /// `n` is the integer nearest `self / divisor`, and the even one at a
    /// tie.
    ///
    /// The remainder is exact, so the rounding direction, the precision limit,
    /// and flush-to-zero do not apply. A zero remainder has the sign of
    /// `self`. An infinite `self` or a zero `divisor` is invalid. A subnormal
    /// remainder reports [`Flags::TINY`]. This is not the Rust `%` operator,
    /// which truncates the quotient:
    /// [`truncated_remainder_with`](Self::truncated_remainder_with).
    #[must_use]
    pub fn remainder_with(self, divisor: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::remainder(
            self.bits,
            divisor.bits,
            Quotient::Nearest,
            behavior.apply::<M>(),
        );
        (Self::from_masked(bits), flags)
    }

    /// Returns the truncated remainder, with the default mode. The `%`
    /// operator calls it.
    #[must_use]
    pub fn truncated_remainder(self, divisor: Self) -> Self {
        self.truncated_remainder_with(divisor, M::default()).0
    }

    /// Returns the truncated remainder `self - n * divisor`, and the flags.
    /// `n` is `self / divisor` rounded toward zero, as for the C `fmod`
    /// function and the Rust `%` operator.
    ///
    /// The remainder has the sign of `self` and a magnitude below that of
    /// `divisor`. The other rules are those of
    /// [`remainder_with`](Self::remainder_with): the result is exact, a zero
    /// has the sign of `self`, an infinite `self` or a zero `divisor` is
    /// invalid, and an infinite `divisor` gives `self`. A decimal result has
    /// the smaller exponent of the operands, as decNumber `remainder` gives.
    ///
    /// ```
    /// use floaty::F64;
    ///
    /// let (five, three) = (F64::from_bits(0x4014_0000_0000_0000), F64::from_bits(0x4008_0000_0000_0000));
    /// // 5 - 1 * 3 = 2, and the IEEE 754 remainder is 5 - 2 * 3 = -1.
    /// assert_eq!((five % three).to_bits(), 0x4000_0000_0000_0000);
    /// assert_eq!(five.remainder(three).to_bits(), 0xBFF0_0000_0000_0000);
    /// ```
    #[must_use]
    pub fn truncated_remainder_with(self, divisor: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::remainder(
            self.bits,
            divisor.bits,
            Quotient::Truncated,
            behavior.apply::<M>(),
        );
        (Self::from_masked(bits), flags)
    }

    /// Returns `self * RADIX^scale`, rounded with the default mode.
    #[must_use]
    pub fn scale_b(self, scale: i32) -> Self {
        self.scale_b_with(scale, M::default()).0
    }

    /// Returns `self * RADIX^scale`, rounded, as IEEE 754 `scaleB` does.
    /// Returns the result and the flags. A scale beyond `2^30` in magnitude
    /// acts as `2^30`, which overflows or underflows every format.
    #[must_use]
    pub fn scale_b_with(self, scale: i32, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::scale_b(self.bits, scale, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns the least value above `self`, with the default mode.
    #[must_use]
    pub fn next_up(self) -> Self {
        self.next_up_with(M::default()).0
    }

    /// Returns the least value above `self`, as IEEE 754 `nextUp` does, and
    /// the flags.
    ///
    /// The next value above a zero of either sign is the smallest positive
    /// subnormal value. The value above the largest finite value is the
    /// infinity, or in a format without an infinity the NaN. A saturating
    /// behavior, or a format with neither, gives the largest finite value
    /// itself, also for the step above an infinity. The step does not round, so the
    /// precision limit and flush-to-zero do not apply, and a subnormal result
    /// reports no [`Flags::TINY`]. Only a signaling NaN or an unsupported
    /// encoding signals invalid. A subnormal input reports
    /// [`Flags::DENORMAL_INPUT`], as for every operation.
    #[must_use]
    pub fn next_up_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::next(self.bits, Step::Up, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns the greatest value below `self`, with the default mode.
    #[must_use]
    pub fn next_down(self) -> Self {
        self.next_down_with(M::default()).0
    }

    /// Returns the greatest value below `self`, as IEEE 754 `nextDown` does,
    /// and the flags. It mirrors [`next_up_with`](Self::next_up_with).
    #[must_use]
    pub fn next_down_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::next(self.bits, Step::Down, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }
}

/// Implements an operator with the default mode of the type. The operator
/// drops the flags; the `_with` method returns them.
///
/// A format with a host fast path calls the engine out of line, only for a NaN
/// result or a host setting that differs from the mode. So the fast path
/// stays small enough to inline. The engine path of every other format
/// inlines.
macro_rules! operator {
    ($trait:ident, $method:ident, $with:ident, $engine:ident) => {
        impl<S: Standard<W>, const W: usize, M: Mode> core::ops::$trait for Float<S, W, M> {
            type Output = Self;

            #[inline]
            fn $method(self, other: Self) -> Self {
                if !host::available(S::HOST, Kind::Arithmetic) {
                    return self.$with(other, M::default()).0;
                }
                let host = host::binary::<S, W>(self.bits, other.bits, Operation::$trait, &M::ENV);
                match host {
                    Some(bits) => Self::from_masked(bits),
                    None => $engine(self, other),
                }
            }
        }

        /// Runs the operator in the engine, for a format whose host fast path
        /// does not apply.
        #[cold]
        #[inline(never)]
        fn $engine<S: Standard<W>, const W: usize, M: Mode>(
            left: Float<S, W, M>,
            right: Float<S, W, M>,
        ) -> Float<S, W, M> {
            left.$with(right, M::default()).0
        }
    };
}

operator!(Add, add, add_with, add_in_engine);
operator!(Sub, sub, sub_with, sub_in_engine);
operator!(Mul, mul, mul_with, mul_in_engine);
operator!(Div, div, div_with, div_in_engine);

/// The truncated remainder with the default mode of the type, as
/// [`Float::truncated_remainder`] computes it. The operator drops the flags.
impl<S: Standard<W>, const W: usize, M: Mode> core::ops::Rem for Float<S, W, M> {
    type Output = Self;

    fn rem(self, divisor: Self) -> Self {
        self.truncated_remainder(divisor)
    }
}

/// Implements a compound assignment operator from its binary operator.
macro_rules! assign {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl<S: Standard<W>, const W: usize, M: Mode> core::ops::$trait for Float<S, W, M> {
            #[inline]
            fn $method(&mut self, other: Self) {
                *self = *self $operator other;
            }
        }
    };
}

assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);
assign!(RemAssign, rem_assign, %);

/// Runs the square root in the engine, for a format whose host path does not
/// apply.
#[cold]
#[inline(never)]
fn sqrt_in_engine<S: Standard<W>, const W: usize, M: Mode>(
    value: Float<S, W, M>,
) -> Float<S, W, M> {
    value.sqrt_with(M::default()).0
}

/// Returns the remainder in the engine, for operands whose host path does
/// not apply.
#[cold]
#[inline(never)]
fn remainder_in_engine<S: Standard<W>, const W: usize, M: Mode>(
    dividend: Float<S, W, M>,
    divisor: Float<S, W, M>,
) -> Float<S, W, M> {
    dividend.remainder_with(divisor, M::default()).0
}

/// Runs the rounding to an integral value in the engine, for a format whose
/// host path does not apply.
#[cold]
#[inline(never)]
fn round_to_integral_in_engine<S: Standard<W>, const W: usize, M: Mode>(
    value: Float<S, W, M>,
) -> Float<S, W, M> {
    value.round_to_integral_with(M::default()).0
}

/// Runs the fused multiply-add in the engine, for a format whose host path
/// does not apply.
#[cold]
#[inline(never)]
fn mul_add_in_engine<S: Standard<W>, const W: usize, M: Mode>(
    left: Float<S, W, M>,
    multiplier: Float<S, W, M>,
    addend: Float<S, W, M>,
) -> Float<S, W, M> {
    left.mul_add_with(multiplier, addend, M::default()).0
}
