//! The elementwise slice operations of `Lanes`: a store of the values of a
//! vector, their conversion to integers, and their minimum and maximum.

use super::Lanes;
use super::elementwise::Output;
use super::vector::{Single, Vector};
use crate::env::{Behavior, Flags, Mode, Override};
use crate::float::from_host_integer;
use crate::format::internal::MinMax;
use crate::host::{self, Host, Kind};
use crate::integer::{Integer, ToInt};

/// The elementwise slice operations. Each one reads a [`Vector`], often an
/// [elementwise view](crate::elementwise), value by value:
///
/// - [`store`](Self::store) writes value `i` into element `i` of a
///   destination.
/// - [`to_int_slice`](Self::to_int_slice) writes value `i` converted to an
///   integer type, as [`Float::to_int`](crate::Float::to_int) converts it.
/// - [`minimum_of`](Self::minimum_of), [`maximum_of`](Self::maximum_of),
///   [`minimum_number_of`](Self::minimum_number_of), and
///   [`maximum_number_of`](Self::maximum_number_of) reduce the values to one
///   in the order of the kernels: every lane starts at +∞ for a minimum and
///   at -∞ for a maximum, value `i` combines into lane `i % N` in
///   increasing order of `i`, and then lane `j` of the low half combines
///   with lane `j` of the high half while more than one lane is left. An
///   empty vector gives the start value, and so does a vector of NaNs for
///   the `_number` operations.
///
/// The methods without flags take the packed host path where the build has
/// one. The path checks the mode and the environment once for the call and
/// computes `N` values at a time. A store and `to_int_slice` compute the
/// values past the last chunk of `N` eight at a time. The engine computes a
/// chunk that holds a NaN. A `_with` method always runs the engine, and
/// returns the union of the flags of every step.
///
/// ```
/// use floaty::elementwise::{Product, RoundToIntegral, Splat};
/// use floaty::{F32, Lanes, Rounding, ToInt};
///
/// let x = [0.5_f32, 1.5, -2.5];
/// let two = F32::from_bits(0x4000_0000);
/// let mut codes = [ToInt::Value(0_i8); 3];
/// Lanes::<F32, 8>::to_int_slice(
///     RoundToIntegral { values: &x[..], rounding: Rounding::TiesToAway },
///     &mut codes,
/// );
/// assert_eq!(codes, [1, 2, -3].map(ToInt::Value));
/// let largest = Lanes::<F32, 8>::maximum_of(Product(&x[..], Splat::new(two, 3)));
/// assert_eq!(largest.to_bits(), 0x4040_0000); // 3
/// ```
///
/// # Panics
///
/// The methods panic when the destination does not hold one element for
/// each value.
impl<M: Mode, const N: usize> Lanes<Single<M>, N> {
    /// Writes value `i` of `values` into element `i` of `out`, with the
    /// default mode.
    #[inline]
    pub fn store<T: From<Single<M>>>(values: impl Vector, mut out: impl Output<T>) {
        let count = Self::check_destination(values, out.count());
        if Self::stores_on_host() {
            let done = host::packed::elementwise::store::<N>(
                values,
                &M::ENV,
                |start, end, bits| match bits {
                    Some(bits) => out.write(
                        start,
                        bits.iter().map(|&bits| Single::from_bits(bits).into()),
                    ),
                    None => Self::store_out_of_line(values, &mut out, start, end),
                },
            );
            if done.is_some() {
                return;
            }
        }
        Self::store_engine(values, &mut out, 0, count, M::default());
    }

    /// Writes value `i` of `values` into element `i` of `out`, as
    /// [`store`](Self::store) does, and returns the union of the flags.
    pub fn store_with<T: From<Single<M>>>(
        values: impl Vector,
        mut out: impl Output<T>,
        behavior: impl Override,
    ) -> Flags {
        let count = Self::check_destination(values, out.count());
        Self::store_engine(values, &mut out, 0, count, behavior.apply::<M>())
    }

    /// Writes value `i` of `values` converted to the integer type `I` into
    /// `out[i]`, rounding with the default mode, as
    /// [`Float::to_int`](crate::Float::to_int) converts it.
    #[inline]
    pub fn to_int_slice<I: Integer>(values: impl Vector, out: &mut [ToInt<I>]) {
        Self::check_destination(values, out.len());
        if const { host::packed::available(Host::Single, Kind::ToInt) } {
            let done =
                host::packed::elementwise::to_int::<N>(values, &M::ENV, |start, end, integers| {
                    let results = &mut out[start..end];
                    let Some(integers) = integers else {
                        for (index, result) in (start..end).zip(results) {
                            *result = Self::to_int_out_of_line(values, index);
                        }
                        return;
                    };
                    for ((index, result), &integer) in (start..end).zip(results).zip(integers) {
                        // `i32::MIN` marks a value that the engine converts.
                        *result = if integer == i32::MIN {
                            Self::to_int_out_of_line(values, index)
                        } else {
                            from_host_integer(i64::from(integer))
                        };
                    }
                });
            if done.is_some() {
                return;
            }
        }
        Self::to_int_engine(values, out, M::default());
    }

    /// Writes value `i` of `values` converted to the integer type `I` into
    /// `out[i]`, as [`to_int_slice`](Self::to_int_slice) does, and returns
    /// the union of the flags.
    pub fn to_int_slice_with<I: Integer>(
        values: impl Vector,
        out: &mut [ToInt<I>],
        behavior: impl Override,
    ) -> Flags {
        Self::check_destination(values, out.len());
        Self::to_int_engine(values, out, behavior.apply::<M>())
    }

    /// Returns the count of `values`, after a check that a destination of
    /// `elements` elements holds one element for each value.
    fn check_destination(values: impl Vector, elements: usize) -> usize {
        const { assert!(N.is_power_of_two(), "the lane count is a power of two") };
        let count = values.count();
        assert_eq!(count, elements, "the destination holds each value");
        count
    }

    /// Returns `true` when this build has the host path of the stores and
    /// the reductions. The answer is a constant.
    #[inline]
    fn stores_on_host() -> bool {
        const { host::packed::available(Host::Single, Kind::Arithmetic) }
    }

    /// Writes values `start` to `end - 1` in the engine with the default
    /// mode, for a chunk whose host path declines. The call stays out of
    /// line, so that the host path inlines into its caller.
    #[cold]
    #[inline(never)]
    fn store_out_of_line<T: From<Single<M>>>(
        values: impl Vector,
        out: &mut impl Output<T>,
        start: usize,
        end: usize,
    ) {
        Self::store_engine(values, out, start, end, M::default());
    }

    /// Writes values `start` to `end - 1` in the engine, and returns the
    /// union of the flags.
    fn store_engine<T: From<Single<M>>, B: Behavior>(
        values: impl Vector,
        out: &mut impl Output<T>,
        start: usize,
        end: usize,
        behavior: B,
    ) -> Flags {
        let mut flags = Flags::NONE;
        out.write(
            start,
            (start..end).map(|index| {
                let (value, value_flags) = values.value_with::<M, B>(index, behavior);
                flags |= value_flags;
                value.into()
            }),
        );
        flags
    }

    /// Converts value `index` to an integer in the engine with the default
    /// mode, for a value whose host path declines.
    #[cold]
    #[inline(never)]
    fn to_int_out_of_line<I: Integer>(values: impl Vector, index: usize) -> ToInt<I> {
        let behavior = M::default();
        let (value, _) = values.value_with::<M, M>(index, behavior);
        value.to_int_with(behavior).0
    }

    /// Converts each value to an integer in the engine, and returns the
    /// union of the flags.
    fn to_int_engine<I: Integer, B: Behavior>(
        values: impl Vector,
        out: &mut [ToInt<I>],
        behavior: B,
    ) -> Flags {
        out.iter_mut()
            .enumerate()
            .fold(Flags::NONE, |flags, (index, result)| {
                let (value, value_flags) = values.value_with::<M, B>(index, behavior);
                let (integer, integer_flags) = value.to_int_with(behavior);
                *result = integer;
                flags | value_flags | integer_flags
            })
    }

    /// Runs a reduction with the default mode: on the host unit where the
    /// build and the environment allow it, and in the engine otherwise.
    #[inline]
    fn reduce(values: impl Vector, operation: MinMax) -> Single<M> {
        const { assert!(N.is_power_of_two(), "the lane count is a power of two") };
        if Self::stores_on_host()
            && let Some(bits) = host::packed::elementwise::reduce::<N>(values, operation, &M::ENV)
        {
            return Single::from_bits(bits);
        }
        Self::reduce_out_of_line(values, operation)
    }

    /// Runs a reduction in the engine with the default mode, for a call
    /// whose host path declines.
    #[cold]
    #[inline(never)]
    fn reduce_out_of_line(values: impl Vector, operation: MinMax) -> Single<M> {
        Self::reduce_engine(values, operation, M::default()).0
    }

    /// Runs a reduction in the engine, in the order of the reductions.
    fn reduce_engine<B: Behavior>(
        values: impl Vector,
        operation: MinMax,
        behavior: B,
    ) -> (Single<M>, Flags) {
        const { assert!(N.is_power_of_two(), "the lane count is a power of two") };
        let start = if operation.is_minimum() {
            0x7F80_0000
        } else {
            0xFF80_0000
        };
        let mut lanes = [Single::<M>::from_bits(start); N];
        let mut flags = Flags::NONE;
        let mut combine = |lane: Single<M>, (value, value_flags): (Single<M>, Flags)| {
            let (result, step_flags) = match operation {
                MinMax::Minimum => lane.minimum_with(value, behavior),
                MinMax::Maximum => lane.maximum_with(value, behavior),
                MinMax::MinimumNumber => lane.minimum_number_with(value, behavior),
                _ => lane.maximum_number_with(value, behavior),
            };
            flags |= value_flags | step_flags;
            result
        };
        for index in 0..values.count() {
            let lane = &mut lanes[index % N];
            *lane = combine(*lane, values.value_with::<M, B>(index, behavior));
        }
        let mut half = N / 2;
        while half > 0 {
            let (low, high) = lanes.split_at_mut(half);
            for (low, &high) in low.iter_mut().zip(&*high) {
                *low = combine(*low, (high, Flags::NONE));
            }
            half /= 2;
        }
        (lanes[0], flags)
    }
}

/// Defines a reduction and its `_with` method.
macro_rules! reduction {
    ($name:ident, $with:ident, $operation:ident, $summary:literal) => {
        impl<M: Mode, const N: usize> Lanes<Single<M>, N> {
            #[doc = concat!("Returns ", $summary, " of the values of `values` in the order of the reductions, with the default mode.")]
            #[must_use]
            #[inline]
            pub fn $name(values: impl Vector) -> Single<M> {
                Self::reduce(values, MinMax::$operation)
            }

            #[doc = concat!("Returns ", $summary, " of the values of `values` as [`", stringify!($name), "`](Self::", stringify!($name), ") computes it, and the flags.")]
            #[must_use]
            pub fn $with(values: impl Vector, behavior: impl Override) -> (Single<M>, Flags) {
                Self::reduce_engine(values, MinMax::$operation, behavior.apply::<M>())
            }
        }
    };
}

reduction!(
    minimum_of,
    minimum_of_with,
    Minimum,
    "the IEEE 754-2019 `minimum`"
);
reduction!(
    maximum_of,
    maximum_of_with,
    Maximum,
    "the IEEE 754-2019 `maximum`"
);
reduction!(
    minimum_number_of,
    minimum_number_of_with,
    MinimumNumber,
    "the IEEE 754-2019 `minimumNumber`"
);
reduction!(
    maximum_number_of,
    maximum_number_of_with,
    MaximumNumber,
    "the IEEE 754-2019 `maximumNumber`"
);
