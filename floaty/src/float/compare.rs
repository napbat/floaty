//! Comparison, total order, and the minimum and maximum operations.

use core::cmp::Ordering;

use super::Float;
use crate::env::{Behavior, Flags, Mode, Override};
use crate::format::Standard;
use crate::format::internal::MinMax;
use crate::host::{self, Kind};

/// Defines a minimum or maximum operation: a method with the default mode and
/// a `_with` method that returns the flags.
macro_rules! min_max {
    ($name:ident, $with:ident, $operation:ident, $summary:literal) => {
        #[doc = concat!("Returns ", $summary, ", with the default mode.")]
        #[must_use]
        #[inline]
        pub fn $name(self, other: Self) -> Self {
            if !host::available(S::HOST, Kind::Comparison) {
                return self.$with(other, M::default()).0;
            }
            match host::min_max::<S, W>(self.bits, other.bits, MinMax::$operation, &M::ENV) {
                Some(bits) => Self::from_masked(bits),
                None => min_max_in_engine(self, other, MinMax::$operation),
            }
        }

        #[doc = concat!("Returns ", $summary, ", and the flags.")]
        ///
        /// The result is an operand in its canonical encoding, or a NaN that
        /// the NaN rule selects. `-0` orders below `+0`. The operation does
        /// not round.
        #[must_use]
        pub fn $with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
            let (bits, flags) = S::min_max(
                self.bits,
                other.bits,
                MinMax::$operation,
                behavior.apply::<M>(),
            );
            (Self::from_masked(bits), flags)
        }
    };
}

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    /// Compares with `other` as the IEEE 754 quiet predicates do, such as
    /// `compareQuietLess`. Returns the order, or `None` when the values are
    /// unordered, and the flags.
    ///
    /// Zeros of either sign are equal. A NaN or an unsupported encoding is
    /// unordered. A signaling NaN or an unsupported encoding signals invalid.
    ///
    /// ```
    /// use core::cmp::Ordering;
    /// use floaty::{F32, Flags};
    ///
    /// let (one, nan) = (F32::from_bits(0x3F80_0000), F32::from_bits(0x7FC0_0000));
    /// assert_eq!(one.compare_quiet_with(nan, F32::ENV), (None, Flags::NONE));
    /// assert_eq!(one.compare_signaling_with(nan, F32::ENV), (None, Flags::INVALID));
    /// assert_eq!(one.compare_quiet_with(one, F32::ENV), (Some(Ordering::Equal), Flags::NONE));
    /// ```
    #[must_use]
    pub fn compare_quiet_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Option<Ordering>, Flags) {
        S::compare(self.bits, other.bits, behavior.apply::<M>())
    }

    /// Compares with `other` as the IEEE 754 signaling predicates do, such as
    /// `compareSignalingLess`. Every unordered comparison signals invalid.
    #[must_use]
    pub fn compare_signaling_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Option<Ordering>, Flags) {
        let (order, flags) = self.compare_quiet_with(other, behavior);
        match order {
            Some(_) => (order, flags),
            None => (order, flags | Flags::INVALID),
        }
    }

    /// Orders the encodings as IEEE 754 `totalOrder` does, with the rule of
    /// the default mode for two encodings of one datum.
    #[must_use]
    pub fn total_cmp(self, other: Self) -> Ordering {
        self.total_cmp_with(other, M::default())
    }

    /// Orders the encodings as IEEE 754 `totalOrder` does, with the
    /// [`TotalOrder`](crate::TotalOrder) of the behavior for two encodings of
    /// one datum.
    ///
    /// A negative NaN orders first, then the numbers from negative infinity
    /// to positive infinity with `-0` below `+0`, then a positive NaN. A
    /// signaling NaN orders nearer the numbers than a quiet NaN, and a larger
    /// payload farther. The NaN of [`Fnuz`](crate::Fnuz) orders first. In a
    /// decimal format, members of one cohort order by exponent.
    ///
    /// An x87 pseudo-denormal and a non-canonical decimal encoding each have
    /// a canonical twin of the same datum. [`TotalOrder::Datum`] makes the
    /// two equal, and [`TotalOrder::Encoding`] orders them by their bits. An
    /// unsupported x87 encoding orders by its exponent field and significand.
    /// The operation reads only the total-order rule, and signals nothing.
    ///
    /// [`TotalOrder::Datum`]: crate::TotalOrder::Datum
    /// [`TotalOrder::Encoding`]: crate::TotalOrder::Encoding
    ///
    /// ```
    /// use core::cmp::Ordering;
    /// use floaty::{D64Bid, TotalOrder};
    ///
    /// let infinity = D64Bid::from_bits(0x7800_0000_0000_0000);
    /// // An infinity with a trailing bit set is a non-canonical twin.
    /// let twin = D64Bid::from_bits(0x7800_0000_0000_0001);
    /// assert_eq!(twin.total_cmp(infinity), Ordering::Equal);
    /// assert_eq!(twin.total_cmp_with(infinity, TotalOrder::Encoding), Ordering::Greater);
    /// ```
    #[must_use]
    pub fn total_cmp_with(self, other: Self, behavior: impl Override) -> Ordering {
        S::total_cmp(
            self.bits,
            other.bits,
            behavior.apply::<M>().env().total_order,
        )
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

/// Equality as the IEEE 754 `compareQuietEqual` predicate, with the default
/// mode: zeros of either sign are equal, and a NaN equals nothing. Compare
/// [`to_bits`](Float::to_bits) for equal encodings.
impl<S: Standard<W>, const W: usize, M: Mode> PartialEq for Float<S, W, M> {
    fn eq(&self, other: &Self) -> bool {
        self.partial_cmp(other) == Some(Ordering::Equal)
    }
}

/// Order as the IEEE 754 quiet predicates, with the default mode. A NaN is
/// unordered.
impl<S: Standard<W>, const W: usize, M: Mode> PartialOrd for Float<S, W, M> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if !host::available(S::HOST, Kind::Comparison) {
            return self.compare_quiet_with(*other, M::default()).0;
        }
        match host::compare::<S, W>(self.bits, other.bits, &M::ENV) {
            Some(order) => Some(order),
            None => compare_in_engine(*self, *other),
        }
    }
}

/// Compares two values in the engine, for an unordered pair or a pair whose
/// host path does not apply.
#[cold]
#[inline(never)]
fn compare_in_engine<S: Standard<W>, const W: usize, M: Mode>(
    left: Float<S, W, M>,
    right: Float<S, W, M>,
) -> Option<Ordering> {
    left.compare_quiet_with(right, M::default()).0
}

/// Returns the minimum or maximum operation `operation` of two values in the
/// engine, for a pair whose host path does not apply.
#[cold]
#[inline(never)]
fn min_max_in_engine<S: Standard<W>, const W: usize, M: Mode>(
    left: Float<S, W, M>,
    right: Float<S, W, M>,
    operation: MinMax,
) -> Float<S, W, M> {
    let (bits, _) = S::min_max(left.bits, right.bits, operation, M::default());
    Float::from_masked(bits)
}
