//! Arithmetic: the rounded operations, the remainder, and the neighbors of a
//! value.

use super::Float;
use crate::env::{Flags, Mode, Override};
use crate::format::Standard;
use crate::format::internal::Step;
use crate::host::{self, Operation};

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    /// Adds `other`. Returns the sum and the flags.
    ///
    /// `behavior` is a [`Rounding`](crate::Rounding) that overrides only the
    /// rounding direction, or an [`Env`](crate::Env) that replaces the whole
    /// behavior.
    #[must_use]
    pub fn add_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::add(self.bits, other.bits, false, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Subtracts `other`. Returns the difference and the flags.
    #[must_use]
    pub fn sub_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::add(self.bits, other.bits, true, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Multiplies by `other`. Returns the product and the flags.
    #[must_use]
    pub fn mul_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::mul(self.bits, other.bits, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Divides by `other`. Returns the quotient and the flags.
    #[must_use]
    pub fn div_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::div(self.bits, other.bits, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Returns the square root, with the default mode.
    #[must_use]
    pub fn sqrt(self) -> Self {
        self.sqrt_with(M::ENV).0
    }

    /// Returns the square root and the flags.
    #[must_use]
    pub fn sqrt_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::sqrt(self.bits, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Returns `self * multiplier + addend`, rounded once, with the default
    /// mode.
    #[must_use]
    pub fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        self.mul_add_with(multiplier, addend, M::ENV).0
    }

    /// Returns `self * multiplier + addend`, rounded once, and the flags.
    ///
    /// A NaN comes from `self` and `multiplier` first, and then from that
    /// result and `addend`, as in SoftFloat. The
    /// [`invalid_product`](crate::env::NanRule::invalid_product) field of the
    /// NaN rule decides `0 * inf + NaN`.
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
            &behavior.apply(M::ENV),
        );
        (Self::from_masked(bits), flags)
    }

    /// Rounds to an integral value in the format, with the default mode.
    #[must_use]
    pub fn round_to_integral(self) -> Self {
        self.round_to_integral_with(M::ENV).0
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
    /// let (two, flags) = F32::from_bits(0x3FC0_0000).round_to_integral_with(Rounding::NearestEven);
    /// assert_eq!((two.to_bits(), flags), (0x4000_0000, Flags::INEXACT | Flags::ROUNDED_UP));
    /// ```
    #[must_use]
    pub fn round_to_integral_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::round_to_integral(self.bits, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Returns the IEEE 754 remainder, with the default mode.
    #[must_use]
    pub fn remainder(self, divisor: Self) -> Self {
        self.remainder_with(divisor, M::ENV).0
    }

    /// Returns the IEEE 754 remainder `self - n * divisor`, and the flags.
    /// `n` is the integer nearest `self / divisor`, and the even one at a
    /// tie.
    ///
    /// The remainder is exact, so the rounding direction, the precision limit,
    /// and flush-to-zero do not apply. A zero remainder has the sign of
    /// `self`. An infinite `self` or a zero `divisor` is invalid. This is not
    /// the Rust `%` operator, which truncates the quotient.
    #[must_use]
    pub fn remainder_with(self, divisor: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::remainder(self.bits, divisor.bits, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Returns `self * RADIX^scale`, rounded with the default mode.
    #[must_use]
    pub fn scale_b(self, scale: i32) -> Self {
        self.scale_b_with(scale, M::ENV).0
    }

    /// Returns `self * RADIX^scale`, rounded, as IEEE 754 `scaleB` does.
    /// Returns the result and the flags.
    #[must_use]
    pub fn scale_b_with(self, scale: i32, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::scale_b(self.bits, scale, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Returns the least value above `self`, with the default mode.
    #[must_use]
    pub fn next_up(self) -> Self {
        self.next_up_with(M::ENV).0
    }

    /// Returns the least value above `self`, as IEEE 754 `nextUp` does, and
    /// the flags.
    ///
    /// The next value above a zero of either sign is the smallest positive
    /// subnormal value. In a format without an infinity, the value above the
    /// largest finite value is the NaN. When the behavior saturates, it is
    /// the largest finite value itself. The step does not round, so the precision
    /// limit and flush-to-zero do not apply. Only a signaling NaN or an unsupported encoding
    /// signals invalid. A subnormal input reports
    /// [`Flags::DENORMAL_INPUT`], as for every operation.
    #[must_use]
    pub fn next_up_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::next(self.bits, Step::Up, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Returns the greatest value below `self`, with the default mode.
    #[must_use]
    pub fn next_down(self) -> Self {
        self.next_down_with(M::ENV).0
    }

    /// Returns the greatest value below `self`, as IEEE 754 `nextDown` does,
    /// and the flags. It mirrors [`next_up_with`](Self::next_up_with).
    #[must_use]
    pub fn next_down_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::next(self.bits, Step::Down, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }
}

/// Implements an operator with the default mode of the type. The operator
/// drops the flags; the `_with` method returns them.
macro_rules! operator {
    ($trait:ident, $method:ident, $with:ident) => {
        impl<S: Standard<W>, const W: usize, M: Mode> core::ops::$trait for Float<S, W, M> {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                let host = host::binary::<S, W>(self.bits, other.bits, Operation::$trait, &M::ENV);
                match host {
                    Some(bits) => Self::from_masked(bits),
                    None => self.$with(other, M::ENV).0,
                }
            }
        }
    };
}

operator!(Add, add, add_with);
operator!(Sub, sub, sub_with);
operator!(Mul, mul, mul_with);
operator!(Div, div, div_with);
