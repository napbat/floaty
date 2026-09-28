//! Conversions between floats and integers.

use super::Float;
use crate::env::{Flags, Mode, Override};
use crate::exact::Unrounded;
use crate::format::Standard;
use crate::integer::{Integer, ToInt};

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    /// Converts an integer, rounding with the default mode.
    ///
    /// ```
    /// use floaty::{F32, Int};
    ///
    /// assert_eq!(F32::from_int(-3_i32).to_bits(), 0xC040_0000);
    /// assert_eq!(F32::from_int(Int::<24>::from_bits(0xFF_FFFF)).to_bits(), 0xBF80_0000);
    /// ```
    #[must_use]
    pub fn from_int<I: Integer>(value: I) -> Self {
        Self::from_int_with(value, M::ENV).0
    }

    /// Converts an integer, and returns the result and the flags.
    ///
    /// A zero converts to `+0`. The result rounds as every result does, so
    /// the precision limit applies.
    #[must_use]
    pub fn from_int_with<I: Integer>(value: I, behavior: impl Override) -> (Self, Flags) {
        let parts = value.to_parts();
        let exact = Unrounded {
            negative: parts.negative,
            exponent: 0,
            significand: parts.magnitude,
            sticky: false,
        };
        let (bits, flags) = S::round(&exact, &behavior.apply(M::ENV));
        (Self::from_masked(bits), flags)
    }

    /// Converts to an integer of type `I`, rounding with the default mode.
    ///
    /// [`mode::Ieee`](crate::mode::Ieee) rounds to nearest even. Rust `as`
    /// truncates; pass [`Rounding::TowardZero`](crate::Rounding::TowardZero)
    /// to [`to_int_with`](Self::to_int_with) for that.
    #[must_use]
    pub fn to_int<I: Integer>(self) -> ToInt<I> {
        self.to_int_with(M::ENV).0
    }

    /// Converts to an integer of type `I`, in the rounding direction of the
    /// behavior. Returns the result and the flags.
    ///
    /// A NaN or an unsupported encoding gives [`ToInt::Nan`]. An infinity, or
    /// a rounded value outside the range of `I`, gives
    /// [`ToInt::OutOfRange`]. Both signal invalid and not inexact. A value
    /// that fits signals [`Flags::INEXACT`] when it rounds, as the IEEE 754
    /// `convertToIntegerExact` operations do. The other IEEE 754 conversions
    /// are this operation with that flag ignored.
    ///
    /// ```
    /// use floaty::{F32, Flags, Rounding, ToInt};
    ///
    /// let value = F32::from_bits(0xC020_0000); // -2.5
    /// assert_eq!(value.to_int_with::<i8>(Rounding::TowardZero), (ToInt::Value(-2), Flags::INEXACT));
    /// assert_eq!(value.to_int_with::<u8>(F32::ENV), (ToInt::OutOfRange { negative: true }, Flags::INVALID));
    /// ```
    #[must_use]
    pub fn to_int_with<I: Integer>(self, behavior: impl Override) -> (ToInt<I>, Flags) {
        S::to_int(self.bits, &behavior.apply(M::ENV))
    }
}
