//! The operations of a double-double on its exact value that round or select
//! a value: the conversion to an integer, rounding to an integral value,
//! `scale_b`, `log_b`, the minimum and maximum operations, and the fused
//! multiply-add and the next values of [`Qd`].
//!
//! No reference implements these operations for both algorithms, so each
//! one follows the rule that its documentation states. For a canonical pair,
//! glibc 2.43 gives the results of these rules but in the cases that the
//! tests of `floaty-verify` record. The operations read the exact value
//! `hi + lo`, so denormals-are-zero does not apply, and they report no
//! `DENORMAL_INPUT`.

use core::cmp::Ordering;

use super::value::{Magnitude, order_values};
use super::{Algorithm, DoubleDouble, Pair, Qd, convert};
use crate::env::{Behavior, Env, Flags, Mode, Override, Rounding};
use crate::exact::{self, Unrounded};
use crate::float::{Decoded, F64};
use crate::format::internal::{MinMax, Step};
use crate::format::{Binary, Standard};
use crate::integer::{Integer, ToInt};
use crate::limbs::{self, Limbs};
use crate::nan::{self, default_nan};
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

/// The limbs of the exact `a * b + c` of three finite pairs, with
/// [`FUSED_LOWEST`] as the weight of the lowest bit. A product of two halves
/// is below 2^2048, so the sum of the four products and the two halves of
/// the addend is below 2^2051 and needs 4,199 bits.
type Fused = [u64; 66];

/// The weight of the lowest bit of [`Fused`]: the product of two subnormal
/// quanta of 2^-1074.
const FUSED_LOWEST: i32 = -2148;

/// Returns the sign, the exponent of the lowest bit, and the significand of
/// a nonzero finite half, or `None` for a zero half.
fn term(half: F64) -> Option<(bool, i32, u64)> {
    match half.decode::<1>() {
        Decoded::Finite {
            negative,
            exponent,
            significand: [significand],
        } => Some((negative, exponent, significand)),
        Decoded::Zero { .. } => None,
        _ => unreachable!("the caller passes the halves of a pair with a finite value"),
    }
}

/// Returns the exact `a * b + c` of three pairs with finite values, from the
/// products of their halves. An exact zero takes the sign `zero_negative`.
fn fused_sum(
    (a0, a1): Pair,
    (b0, b1): Pair,
    (c0, c1): Pair,
    zero_negative: bool,
) -> Unpacked<Fused> {
    let products = [a0, a1].into_iter().filter_map(term).flat_map(|x| {
        [b0, b1]
            .into_iter()
            .filter_map(term)
            .map(move |y| (x.0 != y.0, x.1 + y.1, u128::from(x.2) * u128::from(y.2)))
    });
    let addends = [c0, c1]
        .into_iter()
        .filter_map(term)
        .map(|(negative, exponent, significand)| (negative, exponent, u128::from(significand)));
    let (positive, negative) = products.chain(addends).fold(
        (Fused::ZERO, Fused::ZERO),
        |(positive, negative), (sign, exponent, magnitude)| {
            let shift = u32::try_from(exponent - FUSED_LOWEST)
                .expect("every term lies at or above the lowest weight");
            let term = limbs::from_u128::<Fused>(magnitude).shl(shift);
            if sign {
                (positive, negative.add(term))
            } else {
                (positive.add(term), negative)
            }
        },
    );
    match positive.compare(&negative) {
        Ordering::Equal => Unpacked::zero(zero_negative),
        Ordering::Greater => Unpacked::Finite {
            negative: false,
            exponent: FUSED_LOWEST,
            significand: positive.sub(negative),
        },
        Ordering::Less => Unpacked::Finite {
            negative: true,
            exponent: FUSED_LOWEST,
            significand: negative.sub(positive),
        },
    }
}

/// Returns the infinity of the sign `negative`, with a `+0` low half.
fn infinite_pair(negative: bool) -> Pair {
    (
        F64::from_bits((u64::from(negative) << 63) | 0x7FF0_0000_0000_0000),
        F64::from_bits(0),
    )
}

/// Returns the largest finite pair of the sign `negative`, at the full
/// precision of binary64.
fn largest_pair(negative: bool) -> Pair {
    convert::largest(negative, &Env::IEEE)
}

/// Returns the negation of a pair whose low half is `+0` when it has no
/// rest. The low half stays `+0`, as in every canonical pair.
fn negate_pair((hi, lo): Pair) -> Pair {
    let lo = if lo.is_zero() { lo } else { -lo };
    (-hi, lo)
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
        /// [`total_cmp`](Self::total_cmp). A magnitude operation first orders
        /// the magnitudes of the exact values. The result is an operand as it
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

    /// Returns the exponent of the exact value, with the default mode.
    #[must_use]
    pub fn log_b(self) -> Self {
        self.log_b_with(M::default()).0
    }

    /// Returns the exponent of the exact value `hi + lo`, as IEEE 754-2019
    /// `logB` does: the integer `e` with `2^e <= |hi + lo| < 2^(e + 1)`, as a
    /// pair, and the flags. A zero gives -inf and signals divide-by-zero, and
    /// an infinity gives +inf. A NaN converts to binary64 with a `+0` low
    /// half. For a canonical pair, the result is that of glibc 2.43 `logbl`,
    /// which reads the low half when the high half is a power of two.
    ///
    /// ```
    /// use floaty::{DoubleDouble, Env, F64, Gcc};
    ///
    /// // 2 - 2^-60 lies below 2, so its exponent is 0.
    /// let value = DoubleDouble::<Gcc>::from_parts(
    ///     F64::from_bits(0x4000_0000_0000_0000),
    ///     F64::from_bits(0xBC30_0000_0000_0000),
    /// );
    /// let (exponent, _) = value.log_b_with(Env::IEEE);
    /// assert_eq!(exponent.hi().to_bits(), 0);
    /// ```
    #[must_use]
    pub fn log_b_with(self, behavior: impl Override) -> (Self, Flags) {
        let (value, flags) = match self.exact() {
            Unpacked::Zero { .. } => (Unpacked::Infinity { negative: true }, Flags::DIVIDE_BY_ZERO),
            Unpacked::Infinity { .. } => (Unpacked::Infinity { negative: false }, Flags::NONE),
            Unpacked::Finite {
                exponent,
                significand,
                ..
            } => {
                let top = i64::from(exponent) + i64::from(significand.bit_length()) - 1;
                let value = if top == 0 {
                    Unpacked::zero(false)
                } else {
                    Unpacked::Finite {
                        negative: top < 0,
                        exponent: 0,
                        significand: Magnitude::ZERO.with_limb(0, top.unsigned_abs()),
                    }
                };
                (value, Flags::NONE)
            }
            nan => (nan, Flags::NONE),
        };
        let (pair, round_flags) = Self::from_exact(value, behavior.apply::<M>());
        (pair, flags | round_flags)
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
        let order = self.total_cmp(other);
        let order = if operation.is_magnitude() {
            order_values(&first.abs(), &second.abs()).then(order)
        } else {
            order
        };
        let first_is_smaller = order != Ordering::Greater;
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
    min_max!(
        minimum_magnitude,
        minimum_magnitude_with,
        MinimumMagnitude,
        "the IEEE 754-2019 `minimumMagnitude`: the operand of the smaller magnitude, or the `minimum` of operands of one magnitude"
    );
    min_max!(
        maximum_magnitude,
        maximum_magnitude_with,
        MaximumMagnitude,
        "the IEEE 754-2019 `maximumMagnitude`: the operand of the larger magnitude, or the `maximum` of operands of one magnitude"
    );
    min_max!(
        minimum_magnitude_number,
        minimum_magnitude_number_with,
        MinimumMagnitudeNumber,
        "the IEEE 754-2019 `minimumMagnitudeNumber`: the operand of the smaller magnitude, or the `minimumNumber` of operands of one magnitude"
    );
    min_max!(
        maximum_magnitude_number,
        maximum_magnitude_number_with,
        MaximumMagnitudeNumber,
        "the IEEE 754-2019 `maximumMagnitudeNumber`: the operand of the larger magnitude, or the `maximumNumber` of operands of one magnitude"
    );
}

impl<M: Mode> DoubleDouble<Qd, M> {
    /// Returns `self * multiplier + addend`, rounded once with the default
    /// mode.
    #[must_use]
    pub fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        self.mul_add_with(multiplier, addend, M::default()).0
    }

    /// Returns `self * multiplier + addend` of the exact values, rounded once
    /// to a pair by the rule of the type documentation, as IEEE 754
    /// `fusedMultiplyAdd` rounds once. Returns the result and the flags.
    ///
    /// QD has no fused multiply-add, so `Qd` follows IEEE 754 on the exact
    /// values `hi + lo`. The special cases follow the binary formats. A NaN
    /// operand gives the NaN that the NaN rule of the behavior selects.
    /// `0 * inf` and `inf - inf` signal invalid. An exact zero sum takes the
    /// sign that IEEE 754-2019 section 6.3 gives. A NaN converts to binary64
    /// with a `+0` low half. [`Gcc`](crate::Gcc) follows `fmal` of glibc
    /// instead.
    ///
    /// ```
    /// use floaty::{DoubleDouble, Env, F64, Flags, Qd};
    ///
    /// // (1 + 2^-60)^2 - 1 = 2^-59 + 2^-120 needs the exact product.
    /// let x = DoubleDouble::<Qd>::from_parts(
    ///     F64::from_bits(0x3FF0_0000_0000_0000),
    ///     F64::from_bits(0x3C30_0000_0000_0000),
    /// );
    /// let minus_one = DoubleDouble::<Qd>::from_int(-1);
    /// let (result, flags) = x.mul_add_with(x, minus_one, Env::IEEE);
    /// assert_eq!(result.hi().to_bits(), 0x3C40_0000_0000_0000); // 2^-59
    /// assert_eq!(result.lo().to_bits(), 0x3870_0000_0000_0000); // 2^-120
    /// assert_eq!(flags, Flags::NONE);
    /// ```
    #[must_use]
    pub fn mul_add_with(
        self,
        multiplier: Self,
        addend: Self,
        behavior: impl Override,
    ) -> (Self, Flags) {
        let env = behavior.apply::<M>().env();
        let (first, second, third) = (self.exact(), multiplier.exact(), addend.exact());
        if let Some((nan, flags)) = nan::fused_special(&first, &second, &third, &env) {
            let (pair, round_flags) = Self::from_exact(nan, env);
            return (pair, flags | round_flags);
        }
        let product_negative = first.is_negative() != second.is_negative();
        let product_infinite = matches!(first, Unpacked::Infinity { .. })
            || matches!(second, Unpacked::Infinity { .. });
        match third {
            Unpacked::Infinity { negative } if product_infinite && negative != product_negative => {
                let (pair, round_flags) = Self::from_exact(default_nan::<Magnitude>(&env), env);
                (pair, Flags::INVALID | round_flags)
            }
            _ if product_infinite => {
                let infinity = Unpacked::<Magnitude>::Infinity {
                    negative: product_negative,
                };
                Self::from_exact(infinity, env)
            }
            Unpacked::Infinity { .. } => Self::from_exact(third, env),
            _ => {
                let zero_negative = env.zero_sum_sign(third.is_negative(), product_negative);
                let sum = fused_sum(
                    (self.hi, self.lo),
                    (multiplier.hi, multiplier.lo),
                    (addend.hi, addend.lo),
                    zero_negative,
                );
                Self::from_exact(sum, env)
            }
        }
    }

    /// Returns the next pair up, with the default mode.
    #[must_use]
    pub fn next_up(self) -> Self {
        self.next_up_with(M::default()).0
    }

    /// Returns the least canonical pair above the exact value, as IEEE 754
    /// `nextUp` does on the values of the canonical pairs. Returns the result
    /// and the flags.
    ///
    /// QD has no such function, so `Qd` follows IEEE 754 on the exact value.
    /// The value of every finite pair is the value of a canonical pair,
    /// unless the high half of that pair overflows. The next value of the
    /// canonical pair `(hi, lo)` is `hi + lo.next_up()`, except at the tie
    /// between `hi` and the next binary64 value: that tie takes the next high
    /// half. The value above a zero is `2^-1074`. The value past the largest
    /// finite pair is +inf, or with saturation the largest finite pair.
    ///
    /// The step does not round, so the precision limit and flush-to-zero do
    /// not apply. Only a signaling NaN signals invalid, and a NaN converts to
    /// binary64 with a `+0` low half. [`Gcc`](crate::Gcc) follows `nextupl`
    /// of glibc instead.
    ///
    /// ```
    /// use floaty::{DoubleDouble, F64, Qd};
    ///
    /// // The next pair above 1 adds the smallest subnormal value.
    /// let one = DoubleDouble::<Qd>::from_int(1);
    /// let next = one.next_up();
    /// assert_eq!(next.hi().to_bits(), 0x3FF0_0000_0000_0000);
    /// assert_eq!(next.lo().to_bits(), 1);
    /// ```
    #[must_use]
    pub fn next_up_with(self, behavior: impl Override) -> (Self, Flags) {
        self.next(Step::Up, behavior.apply::<M>().env())
    }

    /// Returns the next pair down, with the default mode.
    #[must_use]
    pub fn next_down(self) -> Self {
        self.next_down_with(M::default()).0
    }

    /// Returns the greatest canonical pair below the exact value, as IEEE 754
    /// `nextDown` does. It mirrors [`next_up_with`](Self::next_up_with).
    #[must_use]
    pub fn next_down_with(self, behavior: impl Override) -> (Self, Flags) {
        self.next(Step::Down, behavior.apply::<M>().env())
    }

    /// Returns the next canonical pair in the direction `step`.
    fn next(self, step: Step, env: Env) -> (Self, Flags) {
        let value = self.exact();
        if value.is_nan() {
            return Self::from_exact(value, env);
        }
        let past_largest = |negative: bool| {
            if env.saturate {
                largest_pair(negative)
            } else {
                infinite_pair(negative)
            }
        };
        let (hi, lo) = match step {
            Step::Up => Self::successor(&value).unwrap_or_else(|| past_largest(false)),
            Step::Down => {
                Self::successor(&value.negate()).map_or_else(|| past_largest(true), negate_pair)
            }
        };
        // The step above +inf, or below -inf, is the infinity itself, which a
        // saturating behavior clamps as it clamps the step past the largest
        // pair.
        let (hi, lo) = if env.saturate && hi.is_infinite() {
            largest_pair(hi.is_sign_negative())
        } else {
            (hi, lo)
        };
        (Self::new(hi, lo), Flags::NONE)
    }

    /// Returns the least canonical pair above a value that is not a NaN, or
    /// `None` past the largest finite pair.
    fn successor(value: &Unpacked<Magnitude>) -> Option<Pair> {
        let negative = match *value {
            Unpacked::Zero { .. } => return Some((F64::from_bits(1), F64::from_bits(0))),
            Unpacked::Infinity { negative: false } => return Some(infinite_pair(false)),
            Unpacked::Infinity { negative: true } => return Some(largest_pair(true)),
            Unpacked::Finite { negative, .. } => negative,
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the caller handles every NaN")
            }
        };
        let (canonical, _) = Self::from_exact(*value, Env::IEEE);
        if canonical.hi.is_infinite() {
            // Only a pair that is not canonical has a value at or above the
            // overflow threshold of the high half.
            return negative.then(|| largest_pair(true));
        }
        let (hi, lo) = (canonical.hi, canonical.lo);
        // The tie between `hi` and the next binary64 value is the last value
        // of `hi` when `hi` is even. The step above it continues from the next
        // high half, whose low halves below zero are finer.
        let above = hi.next_up();
        let (hi, lo) = if lo > F64::from_bits(0) && lo + lo == above - hi {
            (above, -lo)
        } else {
            (hi, lo)
        };
        let (next, _) = Self::from_exact(Self::new(hi, lo.next_up()).exact(), Env::IEEE);
        if next.hi.is_infinite() {
            return None;
        }
        Some((next.hi, next.lo))
    }
}
