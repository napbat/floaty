//! The operations of a double-double on its exact value that round or select
//! a value: the conversion to an integer, rounding to an integral value,
//! `scale_b`, and the minimum and maximum operations.
//!
//! No reference implements these operations for both algorithms, so each
//! one follows the rule that its documentation states. They read the exact
//! value `hi + lo`, so denormals-are-zero does not apply, and they report no
//! `DENORMAL_INPUT`.

use core::cmp::Ordering;

use super::value::Magnitude;
use super::{Algorithm, DoubleDouble};
use crate::env::{Behavior, Flags, Mode, Override, Rounding};
use crate::exact::{self, Unrounded};
use crate::float::F64;
use crate::format::internal::MinMax;
use crate::format::{Binary, Standard};
use crate::integer::{Integer, ToInt};
use crate::limbs::Limbs;
use crate::rounding::Integral;
use crate::unpacked::{SCALE_LIMIT, Unpacked};

/// The limbs of the integer part of a finite pair, below 2^1025, with the
/// carry of a rounding.
type Whole = [u64; 17];

/// Returns the integer that a finite exact value rounds to in the direction
/// `rounding`.
fn integral(
    negative: bool,
    exponent: i32,
    magnitude: &Magnitude,
    rounding: Rounding,
) -> Integral<Whole> {
    let value = Unrounded {
        negative,
        exponent,
        significand: *magnitude,
        sticky: false,
    };
    exact::round_to_integer(&value, rounding)
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
        let value = self.exact();
        if let Some(special) = ToInt::special(&value) {
            return special;
        }
        let Unpacked::Finite {
            negative,
            exponent,
            significand,
        } = value
        else {
            unreachable!("ToInt::special takes every value that is not finite");
        };
        ToInt::from_integral(
            negative,
            &integral(negative, exponent, &significand, env.rounding),
        )
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
        let exact_env = env.for_exact_result();
        let (value, flags) = match self.exact() {
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let integral = integral(negative, exponent, &significand, env.rounding);
                let value = if integral.magnitude.is_zero() {
                    Unpacked::zero(negative)
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
        let (pair, round_flags) = Self::from_exact(value, exact_env);
        (pair, flags | round_flags)
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
        Self::from_exact(value, behavior.apply::<M>())
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
        self.special_half()
            .filter(|half| half.is_nan())
            .unwrap_or(F64::from_bits(0))
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
