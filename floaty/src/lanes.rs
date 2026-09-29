//! Values of one float type as the lanes of a vector register.

use core::ops;

use crate::env::{Flags, Mode, Override};
use crate::float::{Float, FloatType};
use crate::format::Standard;
use crate::host::{self, Kind, Operation};

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
        if !host::available(S::HOST, Kind::SquareRoot) || !host::packed::lanes_of(S::HOST) {
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
        if !host::available(S::HOST, Kind::FusedMultiplyAdd) || !host::packed::lanes_of(S::HOST) {
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
        if !host::available(S::HOST, Kind::RoundToIntegral) || !host::packed::lanes_of(S::HOST) {
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
                if !host::available(S::HOST, Kind::Arithmetic) || !host::packed::lanes_of(S::HOST) {
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

/// Converts each lane to the float type `T`, for lanes whose packed host
/// path declines, as `Lanes::map_out_of_line` does.
#[cold]
#[inline(never)]
fn convert_out_of_line<S: Standard<W>, const W: usize, M: Mode, const N: usize, T: FloatType>(
    lanes: Lanes<Float<S, W, M>, N>,
) -> Lanes<T, N> {
    Lanes::new(lanes.lanes.map(Float::convert::<T>))
}
