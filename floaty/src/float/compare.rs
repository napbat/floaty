//! Comparison, total order, and the minimum and maximum operations.

use core::cmp::Ordering;

use super::Float;
use crate::env::{Flags, Mode, Override};
use crate::format::Standard;
use crate::format::internal::MinMax;

/// Defines a minimum or maximum operation: a method with the default mode and
/// a `_with` method that returns the flags.
macro_rules! min_max {
    ($name:ident, $with:ident, $operation:ident, $summary:literal) => {
        #[doc = concat!("Returns ", $summary, ", with the default mode.")]
        #[must_use]
        pub fn $name(self, other: Self) -> Self {
            self.$with(other, M::ENV).0
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
                &behavior.apply(M::ENV),
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
        S::compare(self.bits, other.bits, &behavior.apply(M::ENV))
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

    /// Orders the encodings as IEEE 754 `totalOrder` does.
    ///
    /// A negative NaN orders first, then the numbers from negative infinity
    /// to positive infinity with `-0` below `+0`, then a positive NaN. A
    /// signaling NaN orders nearer the numbers than a quiet NaN, and a larger
    /// payload farther. The NaN of [`Fnuz`](crate::Fnuz) orders first. An x87
    /// encoding that is not canonical orders by its exponent field and
    /// significand. The operation reads no behavior and signals nothing.
    #[must_use]
    pub fn total_cmp(self, other: Self) -> Ordering {
        S::total_cmp(self.bits, other.bits)
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
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.compare_quiet_with(*other, M::ENV).0
    }
}
