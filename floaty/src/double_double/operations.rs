//! The operations of a double-double on its exact value: the conversion to
//! an integer, rounding to an integral value, `scale_b`, the classification,
//! the total order, and the minimum and maximum operations.
//!
//! No reference implements these operations for both algorithms, so each
//! one follows the rule that its documentation states. They read the exact
//! value `hi + lo`, so denormals-are-zero does not apply, and they report no
//! `DENORMAL_INPUT`.

use core::cmp::Ordering;

use super::convert::{BINARY, round_value};
use super::{Algorithm, DoubleDouble, LOWEST, Magnitude};
use crate::env::{Behavior, Env, Flags, Mode, Override};
use crate::exact::{self, Integral, Unrounded};
use crate::float::{Class, F64};
use crate::format::internal::MinMax;
use crate::format::{Binary, Standard};
use crate::integer::{Integer, Parts, ToInt, fit};
use crate::limbs::Limbs;
use crate::unpacked::{SCALE_LIMIT, Unpacked};

/// The limbs of the integer part of a finite pair, below 2^1025, with the
/// carry of a rounding.
type Whole = [u64; 17];

/// The bit length of `2^-969 * 2^1074`, the magnitude of `LDBL_MIN`: a
/// finite value with a shorter [`Magnitude`] is below `LDBL_MIN`.
const NORMAL_BITS: u32 = 106;

/// Returns the integer that a finite exact value rounds to in the direction
/// `rounding`.
fn integral(negative: bool, magnitude: &Magnitude, env: &Env) -> Integral<Whole> {
    let value = Unrounded {
        negative,
        exponent: LOWEST,
        significand: *magnitude,
        sticky: false,
    };
    exact::round_to_integer(&value, env.rounding)
}

/// Defines a minimum or maximum operation: a method with the default mode and
/// a `_with` method that returns the flags.
macro_rules! min_max {
    ($name:ident, $with:ident, $operation:ident, $summary:literal) => {
        #[doc = concat!("Returns ", $summary, ", with the default mode.")]
        #[must_use]
        pub fn $name(self, other: Self) -> Self {
            self.$with(other, M::default()).0
        }

        #[doc = concat!("Returns ", $summary, ", and the flags.")]
        ///
        /// The operands order by their exact values, and `-0` orders below
        /// `+0`. Two pairs of one value order by
        /// [`total_cmp`](Self::total_cmp). The result is an operand as it
        /// is, or a NaN with a `+0` low half. That NaN, and the flags, are
        /// those of the binary64 operation on the halves that hold the NaNs,
        /// with a zero for a number.
        #[must_use]
        pub fn $with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
            self.min_max(other, MinMax::$operation, behavior.apply::<M>())
        }
    };
}

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// Converts to an integer of type `I`, rounding with the default mode.
    #[must_use]
    pub fn to_int<I: Integer>(self) -> ToInt<I> {
        self.to_int_with(M::default()).0
    }

    /// Converts the exact value to an integer of type `I`, in the rounding
    /// direction of the behavior. Returns the result and the flags.
    ///
    /// The rules are those of [`Float::to_int_with`](crate::Float::to_int_with):
    /// a NaN gives [`ToInt::Nan`], and an infinity or a rounded value outside
    /// the range of `I` gives [`ToInt::OutOfRange`]. Both signal invalid. A
    /// value that fits signals [`Flags::INEXACT`] when it rounds, and
    /// [`Flags::ROUNDED_UP`] when the magnitude grows.
    ///
    /// ```
    /// use floaty::{DoubleDouble, F64, Flags, Gcc, Rounding, ToInt};
    ///
    /// // 2^53 + 0.5 has its fraction in the low half.
    /// let value = DoubleDouble::<Gcc>::from_parts(
    ///     F64::from_bits(0x4340_0000_0000_0000),
    ///     F64::from_bits(0x3FE0_0000_0000_0000),
    /// );
    /// let expected = (ToInt::Value(9_007_199_254_740_993_u64), Flags::INEXACT | Flags::ROUNDED_UP);
    /// assert_eq!(value.to_int_with(Rounding::TowardPositive), expected);
    /// ```
    #[must_use]
    pub fn to_int_with<I: Integer>(self, behavior: impl Override) -> (ToInt<I>, Flags) {
        let env = behavior.apply::<M>().env();
        match self.exact() {
            Unpacked::Zero { .. } => {
                let zero = Parts {
                    negative: false,
                    magnitude: [0; 8],
                };
                (ToInt::Value(I::from_parts(zero)), Flags::NONE)
            }
            Unpacked::Finite {
                negative,
                significand,
                ..
            } => {
                let integral = integral(negative, &significand, &env);
                match fit::<I, _>(negative, &integral.magnitude) {
                    Some(parts) => (
                        ToInt::Value(I::from_parts(parts)),
                        integral.flags(),
                    ),
                    None => (ToInt::OutOfRange { negative }, Flags::INVALID),
                }
            }
            Unpacked::Infinity { negative } => (ToInt::OutOfRange { negative }, Flags::INVALID),
            Unpacked::Nan { .. } => (ToInt::Nan, Flags::INVALID),
            Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
        }
    }

    /// Rounds to an integral value, with the default mode.
    #[must_use]
    pub fn round_to_integral(self) -> Self {
        self.round_to_integral_with(M::default()).0
    }

    /// Rounds the exact value to an integral value, in the rounding
    /// direction of the behavior. Returns the result and the flags.
    ///
    /// The integer rounds to a pair by the rule of the type documentation,
    /// without the precision limit and flush-to-zero. So the result is the
    /// canonical pair of the integer, and exact. Only the integer of a
    /// non-canonical pair above the largest finite pair overflows. A zero
    /// result has the sign of the value. The flags are those of
    /// [`Float::round_to_integral_with`](crate::Float::round_to_integral_with):
    /// [`Flags::INEXACT`] when the result differs from the value, as IEEE
    /// 754 `roundToIntegralExact` signals, and [`Flags::ROUNDED_UP`] when the
    /// magnitude grows. A NaN converts to binary64 with a `+0` low half, and
    /// an infinity keeps its high half with a `+0` low half.
    #[must_use]
    pub fn round_to_integral_with(self, behavior: impl Override) -> (Self, Flags) {
        let env = behavior.apply::<M>().env();
        let exact_env = Env {
            precision: None,
            flush_to_zero: false,
            ..env
        };
        let (value, flags) = match self.exact() {
            Unpacked::Finite {
                negative,
                significand,
                ..
            } => {
                let integral = integral(negative, &significand, &env);
                let value = if integral.magnitude.is_zero() {
                    Unpacked::Zero {
                        negative,
                        exponent: 0,
                    }
                } else {
                    Unpacked::Finite {
                        negative,
                        exponent: 0,
                        significand: integral.magnitude.resize::<Magnitude>(),
                    }
                };
                (value, integral.flags())
            }
            other => (other, Flags::NONE),
        };
        let ((hi, lo), round_flags) = round_value(value, BINARY, exact_env);
        (Self::new(hi, lo), flags | round_flags)
    }

    /// Returns `self * 2^scale`, rounded with the default mode.
    #[must_use]
    pub fn scale_b(self, scale: i32) -> Self {
        self.scale_b_with(scale, M::default()).0
    }

    /// Returns the exact value times `2^scale`, rounded to a pair by the
    /// rule of the type documentation, and the flags. A scale beyond `2^30`
    /// in magnitude acts as `2^30`, which overflows or underflows every pair.
    /// A NaN converts to binary64 with a `+0` low half.
    ///
    /// A pair scales exactly unless a half of its canonical form leaves the
    /// exponent range of binary64. So the result is the canonical pair with
    /// both halves scaled.
    #[must_use]
    pub fn scale_b_with(self, scale: i32, behavior: impl Override) -> (Self, Flags) {
        let value = match self.exact() {
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => Unpacked::Finite {
                negative,
                exponent: exponent + scale.clamp(-SCALE_LIMIT, SCALE_LIMIT),
                significand,
            },
            other => other,
        };
        let ((hi, lo), flags) = round_value(value, BINARY, behavior.apply::<M>());
        (Self::new(hi, lo), flags)
    }

    /// Returns the class of the exact value.
    ///
    /// A NaN or an infinite high half gives its class. With a finite high
    /// half, a NaN or an infinite low half gives its class. A finite value
    /// below `2^-969` in magnitude, the `LDBL_MIN` of the IBM `long double`,
    /// is subnormal: below it, the low half of a pair cannot hold 53 bits.
    /// A pair whose halves cancel, such as `(1, -1)`, is a zero.
    #[must_use]
    pub fn classify(self) -> Class {
        match self.exact() {
            Unpacked::Zero { .. } => Class::Zero,
            Unpacked::Finite { significand, .. } => {
                if significand.bit_length() < NORMAL_BITS {
                    Class::Subnormal
                } else {
                    Class::Normal
                }
            }
            Unpacked::Infinity { .. } => Class::Infinite,
            Unpacked::Nan {
                signaling: true, ..
            } => Class::SignalingNan,
            Unpacked::Nan {
                signaling: false, ..
            } => Class::QuietNan,
            Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
        }
    }

    /// Returns `true` when the exact value has a negative sign: a negative
    /// number, `-0`, a negative infinity, or a NaN with a negative sign.
    #[must_use]
    pub fn is_sign_negative(self) -> bool {
        match self.exact() {
            Unpacked::Zero { negative, .. }
            | Unpacked::Finite { negative, .. }
            | Unpacked::Infinity { negative }
            | Unpacked::Nan { negative, .. } => negative,
            Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
        }
    }

    /// Returns `true` when the exact value has a positive sign.
    #[must_use]
    pub fn is_sign_positive(self) -> bool {
        !self.is_sign_negative()
    }

    /// Returns `true` for a quiet or a signaling NaN.
    #[must_use]
    pub fn is_nan(self) -> bool {
        matches!(self.classify(), Class::QuietNan | Class::SignalingNan)
    }

    /// Returns `true` for a signaling NaN.
    #[must_use]
    pub fn is_signaling_nan(self) -> bool {
        self.classify() == Class::SignalingNan
    }

    /// Returns `true` for a positive or a negative infinity.
    #[must_use]
    pub fn is_infinite(self) -> bool {
        self.classify() == Class::Infinite
    }

    /// Returns `true` for a zero, a subnormal, or a normal value.
    #[must_use]
    pub fn is_finite(self) -> bool {
        matches!(
            self.classify(),
            Class::Zero | Class::Subnormal | Class::Normal
        )
    }

    /// Returns `true` for a zero.
    #[must_use]
    pub fn is_zero(self) -> bool {
        self.classify() == Class::Zero
    }

    /// Returns `true` for a subnormal value.
    #[must_use]
    pub fn is_subnormal(self) -> bool {
        self.classify() == Class::Subnormal
    }

    /// Returns `true` for a normal value.
    #[must_use]
    pub fn is_normal(self) -> bool {
        self.classify() == Class::Normal
    }

    /// Returns `true` when the pair is canonical, as `iscanonicall` of the
    /// IBM `long double` in glibc 2.43 decides.
    ///
    /// A pair with a zero low half is canonical, and so is a pair with a NaN
    /// high half. A pair with an infinite high half and a nonzero low half is
    /// not. Otherwise the low half must be below half an ulp of the high half
    /// in magnitude, or exactly half an ulp with an even high half. Without
    /// a precision limit, the rounding to a pair of the type documentation
    /// gives a canonical pair.
    #[must_use]
    pub fn is_canonical(self) -> bool {
        // `sysdeps/ieee754/ldbl-128ibm/s_iscanonicall.c` of glibc 2.43, on the
        // magnitudes of the halves.
        let high = self.hi.to_bits() & !(1 << 63);
        let low = self.lo.to_bits() & !(1 << 63);
        if low == 0 {
            return true;
        }
        let high_field = high >> 52;
        if high_field == 0x7FF {
            return high != 0x7FF0_0000_0000_0000;
        }
        // glibc compares the exponent fields: the high field must exceed the
        // low field by more than 53, or by 53 when the low half is a power of
        // two and the high half is even. A subnormal low half has the field
        // of its leading bit, `leading - 51`, so the limit is `leading + 2`.
        let (limit, low_power) = match low >> 52 {
            0 => {
                let leading = low.ilog2();
                (u64::from(leading) + 2, low == 1 << leading)
            }
            field => (field + 53, low.trailing_zeros() >= 52),
        };
        high_field > limit || (high_field == limit && low_power && high & 1 == 0)
    }

    /// Orders the pairs by a total order: by their exact values as IEEE 754
    /// `totalOrder` orders values, then by the halves.
    ///
    /// A negative NaN orders first, then the numbers from negative infinity
    /// to positive infinity with `-0` below `+0`, then a positive NaN. A
    /// signaling NaN orders nearer the numbers than a quiet NaN, and a larger
    /// payload farther. The NaN and the sign of a zero are those of the
    /// exact value. Two pairs of one exact value, such as `(1, 0)` and
    /// `(2, -1)`, order by the binary64 total order of their high halves,
    /// then of their low halves. The operation signals nothing.
    #[must_use]
    pub fn total_cmp(self, other: Self) -> Ordering {
        order_values(&self.exact(), &other.exact())
            .then_with(|| self.hi.total_cmp(other.hi))
            .then_with(|| self.lo.total_cmp(other.lo))
    }

    /// Returns the minimum or the maximum of one family.
    fn min_max<B: Behavior>(self, other: Self, operation: MinMax, behavior: B) -> (Self, Flags) {
        let (first, second) = (self.exact(), other.exact());
        if first.is_nan() || second.is_nan() {
            let (bits, flags) = <Binary<11> as Standard<64>>::min_max(
                self.nan_half().to_bits(),
                other.nan_half().to_bits(),
                operation,
                behavior,
            );
            let result = F64::from_bits(bits);
            if result.is_nan() {
                return (Self::new(result, F64::from_bits(0)), flags);
            }
            let number = if first.is_nan() { other } else { self };
            return (number, flags);
        }
        let first_is_smaller = self.total_cmp(other) != Ordering::Greater;
        let result = if first_is_smaller == operation.is_minimum() {
            self
        } else {
            other
        };
        (result, Flags::NONE)
    }

    /// Returns the half that holds the NaN of a NaN pair, or a zero for a
    /// number.
    fn nan_half(self) -> F64 {
        if self.hi.is_nan() {
            self.hi
        } else if self.hi.is_finite() && self.lo.is_nan() {
            self.lo
        } else {
            F64::from_bits(0)
        }
    }

    min_max!(
        minimum,
        minimum_with,
        Minimum,
        "the IEEE 754-2019 `minimum`: a NaN operand gives a NaN"
    );
    min_max!(
        maximum,
        maximum_with,
        Maximum,
        "the IEEE 754-2019 `maximum`: a NaN operand gives a NaN"
    );
    min_max!(
        minimum_number,
        minimum_number_with,
        MinimumNumber,
        "the IEEE 754-2019 `minimumNumber`: a NaN operand gives the other operand, and a signaling NaN also signals invalid"
    );
    min_max!(
        maximum_number,
        maximum_number_with,
        MaximumNumber,
        "the IEEE 754-2019 `maximumNumber`: a NaN operand gives the other operand, and a signaling NaN also signals invalid"
    );
    min_max!(
        min_num,
        min_num_with,
        MinNum,
        "the IEEE 754-2008 `minNum`: a quiet NaN operand gives the other operand, and a signaling NaN gives a NaN"
    );
    min_max!(
        max_num,
        max_num_with,
        MaxNum,
        "the IEEE 754-2008 `maxNum`: a quiet NaN operand gives the other operand, and a signaling NaN gives a NaN"
    );
}

/// Orders two exact values as IEEE 754 `totalOrder` orders values of one
/// format.
fn order_values(first: &Unpacked<Magnitude>, second: &Unpacked<Magnitude>) -> Ordering {
    let negative = |value: &Unpacked<Magnitude>| match *value {
        Unpacked::Zero { negative, .. }
        | Unpacked::Finite { negative, .. }
        | Unpacked::Infinity { negative }
        | Unpacked::Nan { negative, .. } => negative,
        Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
    };
    let (first_negative, second_negative) = (negative(first), negative(second));
    if first_negative != second_negative {
        return if first_negative {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let order = rank(first)
        .cmp(&rank(second))
        .then_with(|| magnitude(first).compare(&magnitude(second)));
    if first_negative {
        order.reverse()
    } else {
        order
    }
}

/// Returns the place of a class of values in the total order of the
/// magnitudes: zero, a finite value, an infinity, a signaling NaN, a quiet
/// NaN.
fn rank(value: &Unpacked<Magnitude>) -> u8 {
    match value {
        Unpacked::Zero { .. } => 0,
        Unpacked::Finite { .. } => 1,
        Unpacked::Infinity { .. } => 2,
        Unpacked::Nan {
            signaling: true, ..
        } => 3,
        Unpacked::Nan {
            signaling: false, ..
        } => 4,
        Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
    }
}

/// Returns the magnitude of a finite value, or the payload of a NaN, which
/// orders values of one rank.
fn magnitude(value: &Unpacked<Magnitude>) -> Magnitude {
    match *value {
        Unpacked::Finite { significand, .. } => significand,
        Unpacked::Nan { payload, .. } => payload,
        Unpacked::Zero { .. } | Unpacked::Infinity { .. } | Unpacked::Unsupported => {
            Magnitude::ZERO
        }
    }
}
