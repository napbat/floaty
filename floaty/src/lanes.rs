//! Values of one float type as the lanes of a vector register.

mod order;

use core::ops;

use crate::env::{Flags, Mode, Override};
use crate::float::{Class, Float, FloatType, from_host_integer};
use crate::format::Standard;
use crate::host::{self, Kind, Operation};
use crate::integer::{Integer, ToInt};

/// `N` values of one float type, as the lanes of a vector register.
///
/// `Lanes<F32, 4>` holds an SSE or NEON register of binary32 values, and
/// `Lanes<F32, 8>` an AVX register. Each operation applies the operation of
/// the type to every lane, so each lane gives the bits of the scalar
/// operation. A `_with` method returns the union of the flags of the lanes,
/// as a vector unit accumulates them in its status register.
///
/// ```
/// use floaty::{F32, Lanes};
///
/// let x = Lanes::<F32, 4>::from_bits([0x3F80_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000]);
/// let sum = x + x;
/// assert_eq!(sum.to_bits(), [0x4000_0000, 0x4080_0000, 0x40C0_0000, 0x4100_0000]);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Lanes<T, const N: usize> {
    lanes: [T; N],
}

impl<T, const N: usize> Lanes<T, N> {
    /// Makes lanes from an array, lane 0 first.
    #[must_use]
    pub const fn new(lanes: [T; N]) -> Self {
        Self { lanes }
    }

    /// Returns the lanes as an array, lane 0 first.
    #[must_use]
    pub fn into_array(self) -> [T; N] {
        self.lanes
    }
}

impl<T, const N: usize> From<[T; N]> for Lanes<T, N> {
    fn from(lanes: [T; N]) -> Self {
        Self::new(lanes)
    }
}

impl<T, const N: usize> From<Lanes<T, N>> for [T; N] {
    fn from(lanes: Lanes<T, N>) -> Self {
        lanes.into_array()
    }
}

impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
    /// Makes lanes from their encodings, lane 0 first.
    #[must_use]
    pub fn from_bits(bits: [S::Bits; N]) -> Self {
        Self::new(bits.map(Float::from_bits))
    }

    /// Returns the encodings of the lanes, lane 0 first.
    #[must_use]
    pub fn to_bits(self) -> [S::Bits; N] {
        self.lanes.map(Float::to_bits)
    }

    /// Applies `operation` to each lane.
    fn map(self, operation: impl FnMut(Float<S, W, M>) -> Float<S, W, M>) -> Self {
        Self::new(self.lanes.map(operation))
    }

    /// Applies `operation` to each pair of lanes.
    fn zip(
        self,
        other: Self,
        mut operation: impl FnMut(Float<S, W, M>, Float<S, W, M>) -> Float<S, W, M>,
    ) -> Self {
        let mut lanes = self.lanes;
        lanes
            .iter_mut()
            .zip(other.lanes)
            .for_each(|(lane, other)| *lane = operation(*lane, other));
        Self::new(lanes)
    }

    /// Applies `operation` to each lane, for lanes whose packed host path
    /// declines at run time or gives a NaN lane. Each lane can still take a
    /// scalar host path, which sends a NaN to the engine. The call stays out
    /// of line, so that the packed path inlines into its caller.
    #[cold]
    #[inline(never)]
    fn map_out_of_line(self, operation: impl FnMut(Float<S, W, M>) -> Float<S, W, M>) -> Self {
        self.map(operation)
    }

    /// Applies `operation` to each pair of lanes, for lanes whose packed host
    /// path declines, as `map_out_of_line` does.
    #[cold]
    #[inline(never)]
    fn zip_out_of_line(
        self,
        other: Self,
        operation: impl FnMut(Float<S, W, M>, Float<S, W, M>) -> Float<S, W, M>,
    ) -> Self {
        self.zip(other, operation)
    }

    /// Returns the fused multiply-add of each triple of lanes, lane by lane.
    fn mul_add_lane_wise(self, multiplier: Self, addend: Self) -> Self {
        let mut lanes = self.lanes;
        lanes
            .iter_mut()
            .zip(multiplier.lanes.into_iter().zip(addend.lanes))
            .for_each(|(lane, (multiplier, addend))| *lane = lane.mul_add(multiplier, addend));
        Self::new(lanes)
    }

    /// Returns the fused multiply-add of each triple of lanes, for lanes
    /// whose packed host path declines, as `map_out_of_line` does.
    #[cold]
    #[inline(never)]
    fn mul_add_out_of_line(self, multiplier: Self, addend: Self) -> Self {
        self.mul_add_lane_wise(multiplier, addend)
    }

    /// Returns the square root of each lane, with the default mode.
    #[must_use]
    #[inline]
    pub fn sqrt(self) -> Self {
        if !host::packed::available(S::HOST, Kind::SquareRoot) {
            return self.map(Float::sqrt);
        }
        match host::packed::sqrt(&self.lanes, &M::ENV) {
            Some(lanes) => Self::new(lanes),
            None => self.map_out_of_line(Float::sqrt),
        }
    }

    /// Returns the square root of each lane, and the union of the flags.
    #[must_use]
    pub fn sqrt_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let lanes = self.map(|lane| {
            let (value, lane_flags) = lane.sqrt_with(behavior);
            flags |= lane_flags;
            value
        });
        (lanes, flags)
    }

    /// Returns `self * multiplier + addend` of each triple of lanes, each
    /// rounded once, with the default mode.
    #[must_use]
    #[inline]
    pub fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        if !host::packed::available(S::HOST, Kind::FusedMultiplyAdd) {
            return self.mul_add_lane_wise(multiplier, addend);
        }
        match host::packed::mul_add(&self.lanes, &multiplier.lanes, &addend.lanes, &M::ENV) {
            Some(lanes) => Self::new(lanes),
            None => self.mul_add_out_of_line(multiplier, addend),
        }
    }

    /// Returns `self * multiplier + addend` of each triple of lanes, each
    /// rounded once, and the union of the flags.
    #[must_use]
    pub fn mul_add_with(
        self,
        multiplier: Self,
        addend: Self,
        behavior: impl Override,
    ) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let mut lanes = self.lanes;
        lanes
            .iter_mut()
            .zip(multiplier.lanes.into_iter().zip(addend.lanes))
            .for_each(|(lane, (multiplier, addend))| {
                let (value, lane_flags) = lane.mul_add_with(multiplier, addend, behavior);
                flags |= lane_flags;
                *lane = value;
            });
        (Self::new(lanes), flags)
    }

    /// Rounds each lane to an integral value, with the default mode.
    #[must_use]
    #[inline]
    pub fn round_to_integral(self) -> Self {
        if !host::packed::available(S::HOST, Kind::RoundToIntegral) {
            return self.map(Float::round_to_integral);
        }
        match host::packed::round_to_integral(&self.lanes, &M::ENV) {
            Some(lanes) => Self::new(lanes),
            None => self.map_out_of_line(Float::round_to_integral),
        }
    }

    /// Rounds each lane to an integral value, in the rounding direction of
    /// the behavior, and returns the union of the flags.
    #[must_use]
    pub fn round_to_integral_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let lanes = self.map(|lane| {
            let (value, lane_flags) = lane.round_to_integral_with(behavior);
            flags |= lane_flags;
            value
        });
        (lanes, flags)
    }

    /// Converts each lane to the float type `T`, rounding with the mode of
    /// `T`.
    #[must_use]
    #[inline]
    pub fn convert<T: FloatType>(self) -> Lanes<T, N> {
        if !host::packed::converts(S::HOST, T::HOST) {
            return Lanes::new(self.lanes.map(Float::convert::<T>));
        }
        match host::packed::convert(&self.lanes, T::HOST, &<T::Mode as Mode>::ENV) {
            Some(bits) => {
                // A loop, not `array::map`, which LLVM calls out of line
                // for four lanes.
                let mut lanes = [T::from_host([0, 0]); N];
                lanes
                    .iter_mut()
                    .zip(bits)
                    .for_each(|(lane, bits)| *lane = T::from_host([bits, 0]));
                Lanes::new(lanes)
            }
            None => convert_out_of_line(self),
        }
    }

    /// Converts each lane to the float type `T`, with an override of the
    /// behavior of `T`, and returns the union of the flags.
    #[must_use]
    pub fn convert_with<T: FloatType>(self, behavior: impl Override) -> (Lanes<T, N>, Flags) {
        let behavior = behavior.apply::<T::Mode>();
        let mut flags = Flags::NONE;
        let lanes = self.lanes.map(|lane| {
            let (value, lane_flags) = lane.convert_with::<T>(behavior);
            flags |= lane_flags;
            value
        });
        (Lanes::new(lanes), flags)
    }
}

/// Defines a lane-wise operator, which takes a packed host path where the
/// build has one, and its `_with` method, which returns the union of the
/// flags.
macro_rules! operator {
    ($trait:ident, $method:ident, $with:ident, $summary:literal) => {
        impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> ops::$trait
            for Lanes<Float<S, W, M>, N>
        {
            type Output = Self;

            #[inline]
            fn $method(self, other: Self) -> Self {
                if !host::packed::available(S::HOST, Kind::Arithmetic)
                {
                    return self.zip(other, ops::$trait::$method);
                }
                match host::packed::binary(&self.lanes, &other.lanes, Operation::$trait, &M::ENV) {
                    Some(lanes) => Self::new(lanes),
                    None => self.zip_out_of_line(other, ops::$trait::$method),
                }
            }
        }

        impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
            #[doc = concat!("Returns ", $summary, " of each pair of lanes, and the union of the flags.")]
            #[must_use]
            pub fn $with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
                let behavior = behavior.apply::<M>();
                let mut flags = Flags::NONE;
                let lanes = self.zip(other, |left, right| {
                    let (value, lane_flags) = left.$with(right, behavior);
                    flags |= lane_flags;
                    value
                });
                (lanes, flags)
            }
        }
    };
}

operator!(Add, add, add_with, "the sum");
operator!(Sub, sub, sub_with, "the difference");
operator!(Mul, mul, mul_with, "the product");
operator!(Div, div, div_with, "the quotient");

/// The truncated remainder of each pair of lanes, as
/// [`Lanes::truncated_remainder`] computes it.
impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> ops::Rem
    for Lanes<Float<S, W, M>, N>
{
    type Output = Self;

    fn rem(self, divisor: Self) -> Self {
        self.truncated_remainder(divisor)
    }
}

/// Implements a lane-wise compound assignment operator from its binary
/// operator.
macro_rules! assign {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> ops::$trait
            for Lanes<Float<S, W, M>, N>
        {
            #[inline]
            fn $method(&mut self, other: Self) {
                *self = *self $operator other;
            }
        }
    };
}

assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);
assign!(RemAssign, rem_assign, %);

impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> ops::Neg
    for Lanes<Float<S, W, M>, N>
{
    type Output = Self;

    /// Negates each lane. Only the sign bit changes.
    fn neg(self) -> Self {
        self.map(ops::Neg::neg)
    }
}

/// Defines a method that applies a predicate of `Float` to each lane.
macro_rules! predicates {
    ($($name:ident: $summary:literal),* $(,)?) => {
        impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
            $(
                #[doc = concat!("Returns `true` in each lane that holds ", $summary, ".")]
                #[must_use]
                pub fn $name(self) -> [bool; N] {
                    self.lanes.map(Float::$name)
                }
            )*
        }
    };
}

predicates!(
    is_sign_negative: "a value with the sign bit set",
    is_sign_positive: "a value with the sign bit clear",
    is_nan: "a quiet or a signaling NaN",
    is_signaling_nan: "a signaling NaN",
    is_infinite: "a positive or a negative infinity",
    is_finite: "a zero, a subnormal, or a normal value",
    is_zero: "a positive or a negative zero",
    is_subnormal: "a subnormal encoding",
    is_normal: "a normal value",
    is_canonical: "a canonical encoding",
);

impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
    /// Applies `operation`, which returns flags, to each lane, and returns
    /// the lanes and the union of the flags.
    fn map_with(
        self,
        mut operation: impl FnMut(Float<S, W, M>) -> (Float<S, W, M>, Flags),
    ) -> (Self, Flags) {
        let mut flags = Flags::NONE;
        let lanes = self.map(|lane| {
            let (value, lane_flags) = operation(lane);
            flags |= lane_flags;
            value
        });
        (lanes, flags)
    }

    /// Applies `operation`, which returns flags, to each pair of lanes, and
    /// returns the lanes and the union of the flags.
    fn zip_with(
        self,
        other: Self,
        mut operation: impl FnMut(Float<S, W, M>, Float<S, W, M>) -> (Float<S, W, M>, Flags),
    ) -> (Self, Flags) {
        let mut flags = Flags::NONE;
        let lanes = self.zip(other, |left, right| {
            let (value, lane_flags) = operation(left, right);
            flags |= lane_flags;
            value
        });
        (lanes, flags)
    }

    /// Returns the absolute value of each lane. Only the sign bit changes.
    #[must_use]
    pub fn abs(self) -> Self {
        self.map(Float::abs)
    }

    /// Returns each lane with the sign of the lane of `sign`. Only the sign
    /// bit changes.
    #[must_use]
    pub fn copy_sign(self, sign: Self) -> Self {
        self.zip(sign, Float::copy_sign)
    }

    /// Returns the class of each lane.
    #[must_use]
    pub fn classify(self) -> [Class; N] {
        self.lanes.map(Float::classify)
    }

    /// Converts each lane to the integer type `I`, rounding with the default
    /// mode. binary32 and binary64 lanes take a packed path where the build
    /// has one, and each lane that the packed conversion does not decide takes
    /// the scalar conversion.
    #[must_use]
    #[inline]
    pub fn to_int<I: Integer>(self) -> [ToInt<I>; N] {
        if !host::packed::available(S::HOST, Kind::ToInt) {
            return self.lanes.map(Float::to_int::<I>);
        }
        let Some(integers) = host::packed::to_int_i32(&self.lanes, &M::ENV) else {
            return self.lanes.map(Float::to_int::<I>);
        };
        let mut results = [ToInt::Nan; N];
        for ((result, lane), integer) in results.iter_mut().zip(self.lanes).zip(integers) {
            *result = if integer == i32::MIN {
                lane.to_int()
            } else {
                from_host_integer(i64::from(integer))
            };
        }
        results
    }

    /// Converts each lane to the integer type `I` with the behavior, and
    /// returns the union of the flags.
    #[must_use]
    pub fn to_int_with<I: Integer>(self, behavior: impl Override) -> ([ToInt<I>; N], Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let results = self.lanes.map(|lane| {
            let (result, lane_flags) = lane.to_int_with::<I>(behavior);
            flags |= lane_flags;
            result
        });
        (results, flags)
    }

    /// Makes lanes from integers, each rounded with the default mode, lane 0
    /// first. Each lane takes the scalar conversion: a packed conversion of
    /// 32-bit integers measured no faster.
    #[must_use]
    pub fn from_int<I: Integer>(values: [I; N]) -> Self {
        Self::new(values.map(Float::from_int))
    }

    /// Makes lanes from integers, each rounded with the behavior, and returns
    /// the union of the flags.
    #[must_use]
    pub fn from_int_with<I: Integer>(values: [I; N], behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let lanes = values.map(|value| {
            let (lane, lane_flags) = Float::from_int_with(value, behavior);
            flags |= lane_flags;
            lane
        });
        (Self::new(lanes), flags)
    }

    /// Returns the least value above each lane, with the default mode.
    #[must_use]
    pub fn next_up(self) -> Self {
        self.map(Float::next_up)
    }

    /// Returns the least value above each lane, and the union of the flags.
    #[must_use]
    pub fn next_up_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.next_up_with(behavior))
    }

    /// Returns the greatest value below each lane, with the default mode.
    #[must_use]
    pub fn next_down(self) -> Self {
        self.map(Float::next_down)
    }

    /// Returns the greatest value below each lane, and the union of the
    /// flags.
    #[must_use]
    pub fn next_down_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.next_down_with(behavior))
    }

    /// Returns each lane times `RADIX` to the power of the scale of its lane,
    /// rounded with the default mode.
    #[must_use]
    pub fn scale_b(self, scales: [i32; N]) -> Self {
        let mut lanes = self.lanes;
        lanes
            .iter_mut()
            .zip(scales)
            .for_each(|(lane, scale)| *lane = lane.scale_b(scale));
        Self::new(lanes)
    }

    /// Returns each lane times `RADIX` to the power of the scale of its lane,
    /// rounded with the behavior, and the union of the flags.
    #[must_use]
    pub fn scale_b_with(self, scales: [i32; N], behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        let mut flags = Flags::NONE;
        let mut lanes = self.lanes;
        lanes.iter_mut().zip(scales).for_each(|(lane, scale)| {
            let (value, lane_flags) = lane.scale_b_with(scale, behavior);
            flags |= lane_flags;
            *lane = value;
        });
        (Self::new(lanes), flags)
    }

    /// Returns the IEEE 754 remainder of each pair of lanes, with the default
    /// mode. Each lane takes the scalar remainder.
    #[must_use]
    pub fn remainder(self, divisor: Self) -> Self {
        self.zip(divisor, Float::remainder)
    }

    /// Returns the IEEE 754 remainder of each pair of lanes, and the union of
    /// the flags.
    #[must_use]
    pub fn remainder_with(self, divisor: Self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.zip_with(divisor, |dividend, divisor| {
            dividend.remainder_with(divisor, behavior)
        })
    }

    /// Returns the truncated remainder of each pair of lanes, with the
    /// default mode. The `%` operator calls it.
    #[must_use]
    pub fn truncated_remainder(self, divisor: Self) -> Self {
        self.zip(divisor, Float::truncated_remainder)
    }

    /// Returns the truncated remainder of each pair of lanes, as
    /// [`Float::truncated_remainder_with`] computes it, and the union of the
    /// flags.
    #[must_use]
    pub fn truncated_remainder_with(self, divisor: Self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.zip_with(divisor, |dividend, divisor| {
            dividend.truncated_remainder_with(divisor, behavior)
        })
    }
}

/// Converts each lane to the float type `T`, for lanes whose packed host
/// path declines, as `Lanes::map_out_of_line` does.
#[cold]
#[inline(never)]
fn convert_out_of_line<S: Standard<W>, const W: usize, M: Mode, const N: usize, T: FloatType>(
    lanes: Lanes<Float<S, W, M>, N>,
) -> Lanes<T, N> {
    Lanes::new(lanes.lanes.map(Float::convert::<T>))
}
