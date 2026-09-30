//! The comparisons, the total order, and the minimum and maximum operations
//! of `Lanes`.

use core::cmp::Ordering;

use super::Lanes;
use crate::env::{Flags, Mode, Override};
use crate::float::Float;
use crate::format::Standard;
use crate::format::internal::MinMax;
use crate::host::{self, Kind};

impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
    /// Applies `operation` to each pair of lanes, and returns its results.
    fn each<R>(
        self,
        other: Self,
        mut operation: impl FnMut(Float<S, W, M>, Float<S, W, M>) -> R,
    ) -> [R; N] {
        core::array::from_fn(|index| operation(self.lanes[index], other.lanes[index]))
    }

    /// Compares each pair of lanes in the engine or the scalar paths, for
    /// lanes whose packed host path declines.
    #[cold]
    #[inline(never)]
    fn compare_out_of_line(self, other: Self) -> [Option<Ordering>; N] {
        self.each(other, |left, right| left.partial_cmp(&right))
    }

    /// Compares each pair of lanes as the IEEE 754 quiet predicates do, with
    /// the default mode, as `PartialOrd` compares one pair. A lane holds the
    /// order, or `None` for an unordered pair.
    #[must_use]
    #[inline]
    pub fn compare_quiet(self, other: Self) -> [Option<Ordering>; N] {
        if !host::packed::available(S::HOST, Kind::Comparison) {
            return self.each(other, |left, right| left.partial_cmp(&right));
        }
        match host::packed::compare(&self.lanes, &other.lanes, &M::ENV) {
            Some(orders) => orders,
            None => self.compare_out_of_line(other),
        }
    }

    /// Compares each pair of lanes as the IEEE 754 quiet predicates do, and
    /// returns the orders and the union of the flags.
    #[must_use]
    pub fn compare_quiet_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> ([Option<Ordering>; N], Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let orders = self.each(other, |left, right| {
            let (order, lane_flags) = left.compare_quiet_with(right, behavior);
            flags |= lane_flags;
            order
        });
        (orders, flags)
    }

    /// Compares each pair of lanes as the IEEE 754 signaling predicates do,
    /// and returns the orders and the union of the flags.
    #[must_use]
    pub fn compare_signaling_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> ([Option<Ordering>; N], Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let orders = self.each(other, |left, right| {
            let (order, lane_flags) = left.compare_signaling_with(right, behavior);
            flags |= lane_flags;
            order
        });
        (orders, flags)
    }

    /// Orders each pair of lanes as IEEE 754 `totalOrder` does, with the rule
    /// of the default mode for two encodings of one datum.
    #[must_use]
    pub fn total_cmp(self, other: Self) -> [Ordering; N] {
        self.each(other, Float::total_cmp)
    }

    /// Orders each pair of lanes as IEEE 754 `totalOrder` does, with the
    /// total-order rule of the behavior.
    #[must_use]
    pub fn total_cmp_with(self, other: Self, behavior: impl Override) -> [Ordering; N] {
        let behavior = behavior.apply::<M>();
        self.each(other, |left, right| left.total_cmp_with(right, behavior))
    }
}

/// Defines a lane-wise minimum or maximum operation, which takes a packed
/// host path where the build has one, and its `_with` method, which returns
/// the union of the flags.
macro_rules! min_max {
    ($name:ident, $with:ident, $operation:ident, $summary:literal) => {
        impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
            #[doc = concat!("Returns ", $summary, " of each pair of lanes, with the default mode.")]
            #[must_use]
            #[inline]
            pub fn $name(self, other: Self) -> Self {
                if !host::packed::available(S::HOST, Kind::Comparison)
                {
                    return self.zip(other, Float::$name);
                }
                let operation = MinMax::$operation;
                match host::packed::min_max(&self.lanes, &other.lanes, operation, &M::ENV) {
                    Some(lanes) => Self::new(lanes),
                    None => self.zip_out_of_line(other, Float::$name),
                }
            }

            #[doc = concat!("Returns ", $summary, " of each pair of lanes, and the union of the flags.")]
            #[must_use]
            pub fn $with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
                let behavior = behavior.apply::<M>();
                self.zip_with(other, |left, right| left.$with(right, behavior))
            }
        }
    };
}

min_max!(
    minimum,
    minimum_with,
    Minimum,
    "the IEEE 754-2019 `minimum`"
);
min_max!(
    maximum,
    maximum_with,
    Maximum,
    "the IEEE 754-2019 `maximum`"
);
min_max!(
    minimum_number,
    minimum_number_with,
    MinimumNumber,
    "the IEEE 754-2019 `minimumNumber`"
);
min_max!(
    maximum_number,
    maximum_number_with,
    MaximumNumber,
    "the IEEE 754-2019 `maximumNumber`"
);
min_max!(min_num, min_num_with, MinNum, "the IEEE 754-2008 `minNum`");
min_max!(max_num, max_num_with, MaxNum, "the IEEE 754-2008 `maxNum`");
