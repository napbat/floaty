//! Conversions between floats and integers.

use super::Float;
use crate::env::{Flags, Mode, Override};
use crate::exact::Unrounded;
use crate::format::Standard;
use crate::host::{self, Kind};
use crate::integer::{Integer, Parts, ToInt, fit};

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
    #[inline]
    pub fn from_int<I: Integer>(value: I) -> Self {
        if !host::available(S::HOST, Kind::FromInt) {
            return Self::from_int_with(value, M::default()).0;
        }
        let host = signed_64(value.to_parts())
            .and_then(|integer| host::from_int::<S, W>(integer, &M::ENV));
        match host {
            Some(bits) => Self::from_masked(bits),
            None => from_int_in_engine(value),
        }
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
        let (bits, flags) = S::round(&exact, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Converts to an integer of type `I`, rounding with the default mode.
    ///
    /// [`mode::Ieee`](crate::mode::Ieee) rounds to nearest even. Rust `as`
    /// truncates; pass [`Rounding::TowardZero`](crate::Rounding::TowardZero)
    /// to [`to_int_with`](Self::to_int_with) for that.
    #[must_use]
    #[inline]
    pub fn to_int<I: Integer>(self) -> ToInt<I> {
        if !host::available(S::HOST, Kind::ToInt) {
            return self.to_int_with(M::default()).0;
        }
        match host::to_int::<S, W>(self.bits, &M::ENV) {
            Some(integer) => from_host_integer(integer),
            None => to_int_in_engine(self),
        }
    }

    /// Converts to an integer of type `I`, in the rounding direction of the
    /// behavior. Returns the result and the flags.
    ///
    /// A NaN or an unsupported encoding gives [`ToInt::Nan`]. An infinity, or
    /// a rounded value outside the range of `I`, gives
    /// [`ToInt::OutOfRange`]. Both signal invalid and not inexact. A value
    /// that fits signals [`Flags::INEXACT`] when it rounds, as the IEEE 754
    /// `convertToIntegerExact` operations do, and [`Flags::ROUNDED_UP`] when
    /// the magnitude grows. The other IEEE 754 conversions are this operation
    /// with that flag ignored. The value rounds to an integer, not to the
    /// format, so the precision limit does not apply.
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
        S::to_int(self.bits, behavior.apply::<M>())
    }
}

/// Returns the result of a conversion to `I` of the integer that the host
/// rounds a value to. The range of `I` decides the result, as the engine
/// decides it.
#[inline]
pub(crate) fn from_host_integer<I: Integer>(integer: i64) -> ToInt<I> {
    let negative = integer < 0;
    match fit::<I, _>(negative, &[integer.unsigned_abs()]) {
        Some(parts) => ToInt::Value(I::from_parts(parts)),
        None => ToInt::OutOfRange { negative },
    }
}

/// Returns an integer as an `i64`, or `None` when it does not fit.
#[inline]
fn signed_64(parts: Parts) -> Option<i64> {
    let [low, rest @ ..] = parts.magnitude;
    if rest.iter().any(|&limb| limb != 0) {
        return None;
    }
    if parts.negative {
        0_i64.checked_sub_unsigned(low)
    } else {
        i64::try_from(low).ok()
    }
}

/// Runs a conversion from an integer in the engine, for a format or a value
/// whose host path does not apply.
#[cold]
#[inline(never)]
fn from_int_in_engine<S: Standard<W>, const W: usize, M: Mode, I: Integer>(
    value: I,
) -> Float<S, W, M> {
    Float::from_int_with(value, M::default()).0
}

/// Runs a conversion to an integer in the engine, for a value whose host path
/// does not apply.
#[cold]
#[inline(never)]
fn to_int_in_engine<S: Standard<W>, const W: usize, M: Mode, I: Integer>(
    value: Float<S, W, M>,
) -> ToInt<I> {
    value.to_int_with(M::default()).0
}
