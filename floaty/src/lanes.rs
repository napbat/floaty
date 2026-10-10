//! Values of one float type as the lanes of a vector register.

mod binary;
mod decimal;
pub mod elementwise;
mod kernel;
mod order;
mod store;
mod vector;

pub use self::vector::{LittleEndian, ScaledCodes, Vector, Widen};

use core::ops;

use crate::env::{Flags, Mode, Override};
use crate::float::{Class, Float, FloatType, from_host_integer};
use crate::format::{Binary, Standard};
use crate::host::{self, Kind, Operation, Unit};
use crate::integer::{Integer, ToInt};
use crate::sealed::Sealed;

/// An element of a slice that holds the values of the float type `F`: `F`
/// itself, `f32` for a binary32 type, or `f64` for a binary64 type. A host
/// element holds each value by its bits, as `From` reads them. The trait is
/// sealed.
pub trait Element<F>: Sealed + Copy {
    /// Returns the elements as values of `F`.
    #[doc(hidden)]
    fn floats(elements: &[Self]) -> &[F];

    /// Returns the elements as values of `F` to write.
    #[doc(hidden)]
    fn floats_mut(elements: &mut [Self]) -> &mut [F];
}

impl<S: Standard<W>, const W: usize, M: Mode> Element<Self> for Float<S, W, M> {
    #[inline]
    fn floats(elements: &[Self]) -> &[Self] {
        elements
    }

    #[inline]
    fn floats_mut(elements: &mut [Self]) -> &mut [Self] {
        elements
    }
}

impl Sealed for f32 {}

impl<M: Mode> Element<Float<Binary<8>, 32, M>> for f32 {
    #[inline]
    fn floats(elements: &[Self]) -> &[Float<Binary<8>, 32, M>] {
        host::floats_of_singles(elements)
    }

    #[inline]
    fn floats_mut(elements: &mut [Self]) -> &mut [Float<Binary<8>, 32, M>] {
        host::floats_of_singles_mut(elements)
    }
}

impl Sealed for f64 {}

impl<M: Mode> Element<Float<Binary<11>, 64, M>> for f64 {
    #[inline]
    fn floats(elements: &[Self]) -> &[Float<Binary<11>, 64, M>] {
        host::floats_of_doubles(elements)
    }

    #[inline]
    fn floats_mut(elements: &mut [Self]) -> &mut [Float<Binary<11>, 64, M>] {
        host::floats_of_doubles_mut(elements)
    }
}

/// `N` values of one float type, as the lanes of a vector register.
///
/// `Lanes<F32, 4>` holds an SSE or NEON register of binary32 values, and
/// `Lanes<F32, 8>` an AVX register. Each operation applies the operation of
/// the type to every lane, so each lane gives the bits of the scalar
/// operation. An operation that takes the scalar host path in each lane
/// reads the floating-point environment of the host once for all lanes. A
/// `_with` method returns the union of the flags of the lanes, as a vector
/// unit accumulates them in its status register.
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

    /// Applies `operation` to each pair of lanes, and returns its results.
    fn pairs<R>(
        self,
        other: Self,
        mut operation: impl FnMut(Float<S, W, M>, Float<S, W, M>) -> R,
    ) -> [R; N] {
        core::array::from_fn(|index| operation(self.lanes[index], other.lanes[index]))
    }

    /// Reads once the environment of the host unit that the scalar host path
    /// of `kind` reads, for the lanes of one operation, where the build has
    /// that path. Each lane that takes the scalar path then reads nothing.
    #[inline]
    fn unit(kind: Kind) -> Unit {
        if host::available(S::HOST, kind) {
            Unit::read(S::HOST)
        } else {
            Unit::Read
        }
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
        let unit = Self::unit(Kind::FusedMultiplyAdd);
        let mut lanes = self.lanes;
        lanes
            .iter_mut()
            .zip(multiplier.lanes.into_iter().zip(addend.lanes))
            .for_each(|(lane, (multiplier, addend))| {
                *lane = lane.mul_add_in(multiplier, addend, unit);
            });
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
        let root = |unit: Unit| move |lane: Float<S, W, M>| lane.sqrt_in(unit);
        if !host::packed::available(S::HOST, Kind::SquareRoot) {
            return self.map(root(Self::unit(Kind::SquareRoot)));
        }
        match host::packed::sqrt(&self.lanes, &M::ENV) {
            Some(lanes) => Self::new(lanes),
            None => self.map_out_of_line(root(Self::unit(Kind::SquareRoot))),
        }
    }

    /// Returns the square root of each lane, and the union of the flags.
    #[must_use]
    pub fn sqrt_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.sqrt_with(behavior))
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
        let triples: [_; N] = core::array::from_fn(|index| {
            (
                self.lanes[index],
                multiplier.lanes[index],
                addend.lanes[index],
            )
        });
        let (lanes, flags) = with_flags(triples, |(lane, multiplier, addend)| {
            lane.mul_add_with(multiplier, addend, behavior)
        });
        (Self::new(lanes), flags)
    }

    /// Rounds each lane to an integral value, with the default mode.
    #[must_use]
    #[inline]
    pub fn round_to_integral(self) -> Self {
        let integral = |unit: Unit| move |lane: Float<S, W, M>| lane.round_to_integral_in(unit);
        if !host::packed::available(S::HOST, Kind::RoundToIntegral) {
            return self.map(integral(Self::unit(Kind::RoundToIntegral)));
        }
        match host::packed::round_to_integral(&self.lanes, &M::ENV) {
            Some(lanes) => Self::new(lanes),
            None => self.map_out_of_line(integral(Self::unit(Kind::RoundToIntegral))),
        }
    }

    /// Rounds each lane to an integral value, in the rounding direction of
    /// the behavior, and returns the union of the flags.
    #[must_use]
    pub fn round_to_integral_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.round_to_integral_with(behavior))
    }

    /// Converts each lane to the float type `T`, rounding with the mode of
    /// `T`.
    #[must_use]
    #[inline]
    pub fn convert<T: FloatType>(self) -> Lanes<T, N> {
        if !host::packed::convertible(S::HOST, T::HOST) {
            let unit = conversion_unit::<S, W, T>();
            return Lanes::new(self.lanes.map(|lane| lane.convert_in::<T>(unit)));
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
        let (lanes, flags) = self.each_with(|lane| lane.convert_with::<T>(behavior));
        (Lanes::new(lanes), flags)
    }

    /// Converts each value of `values` to the float type `T` into the same
    /// index of `out`, rounding with the mode of `T`, as
    /// [`Float::convert`] does. The values are of the lane type, or host
    /// `f32` values for binary32 lanes, as [`Element`] states. The packed
    /// host path converts `N` values at a time, with one check of the
    /// environment for the call, where the build has the path. A chunk that
    /// holds a NaN, and the values after the last full chunk, convert one at
    /// a time.
    ///
    /// ```
    /// use floaty::{BF16, F32, Lanes};
    ///
    /// let values = [0x3F80_0000, 0x3F80_8000, 0x7FA0_0000].map(F32::from_bits);
    /// let mut out = [BF16::from_bits(0); 3];
    /// Lanes::<F32, 8>::convert_slice(&values, &mut out);
    /// // 1, the tie 1 + 2^-8 to even, and the quiet signaling NaN.
    /// assert_eq!(out.map(BF16::to_bits), [0x3F80, 0x3F80, 0x7FE0]);
    ///
    /// // Host values convert by their bits.
    /// Lanes::<F32, 8>::convert_slice(&[1.0_f32, 2.0, -0.0], &mut out);
    /// assert_eq!(out.map(BF16::to_bits), [0x3F80, 0x4000, 0x8000]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics when `values` and `out` have different lengths.
    #[inline]
    pub fn convert_slice<T: FloatType, E: Element<Float<S, W, M>>>(values: &[E], out: &mut [T]) {
        let values = E::floats(values);
        assert_eq!(
            values.len(),
            out.len(),
            "the slices of a conversion have one length"
        );
        if const { host::packed::convertible(S::HOST, T::HOST) } {
            let done = host::packed::convert_chunks::<S, W, M, T, N>(
                values,
                out,
                &<T::Mode as Mode>::ENV,
                convert_each,
            );
            if done.is_some() {
                let rest = values.len() - values.len() % N;
                convert_each(&values[rest..], &mut out[rest..]);
                return;
            }
        }
        convert_each(values, out);
    }

    /// Converts each value of `values` to the float type `T` into the same
    /// index of `out`, with an override of the behavior of `T`, and returns
    /// the union of the flags.
    ///
    /// # Panics
    ///
    /// Panics when `values` and `out` have different lengths.
    pub fn convert_slice_with<T: FloatType, E: Element<Float<S, W, M>>>(
        values: &[E],
        out: &mut [T],
        behavior: impl Override,
    ) -> Flags {
        let values = E::floats(values);
        assert_eq!(
            values.len(),
            out.len(),
            "the slices of a conversion have one length"
        );
        let behavior = behavior.apply::<T::Mode>();
        values
            .iter()
            .zip(out)
            .fold(Flags::NONE, |flags, (value, result)| {
                let (converted, value_flags) = value.convert_with::<T>(behavior);
                *result = converted;
                flags | value_flags
            })
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
                let operation = |unit: Unit| {
                    move |left: Float<S, W, M>, right| left.binary_in(right, Operation::$trait, unit)
                };
                if !host::packed::available(S::HOST, Kind::Arithmetic) {
                    return self.zip(other, operation(Self::unit(Kind::Arithmetic)));
                }
                match host::packed::binary(&self.lanes, &other.lanes, Operation::$trait, &M::ENV) {
                    Some(lanes) => Self::new(lanes),
                    None => self.zip_out_of_line(other, operation(Self::unit(Kind::Arithmetic))),
                }
            }
        }

        impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
            #[doc = concat!("Returns ", $summary, " of each pair of lanes, and the union of the flags.")]
            #[must_use]
            pub fn $with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
                let behavior = behavior.apply::<M>();
                self.zip_with(other, |left, right| left.$with(right, behavior))
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

/// Defines the methods that apply an exponential, a logarithm, a hyperbolic
/// function, a function scaled by pi, or an inverse trigonometric function
/// of `Float` to each lane, with the default mode and with a behavior.
macro_rules! elementary {
    ($($name:ident, $name_with:ident: $summary:literal),* $(,)?) => {
        impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
            $(
                #[doc = concat!(
                    "Returns ", $summary, " of each lane, with the default mode. Each lane ",
                    "takes [`Float::", stringify!($name), "`]."
                )]
                #[must_use]
                pub fn $name(self) -> Self {
                    self.map(Float::$name)
                }

                #[doc = concat!(
                    "Returns ", $summary, " of each lane, correctly rounded, and the union ",
                    "of the flags."
                )]
                #[must_use]
                pub fn $name_with(self, behavior: impl Override) -> (Self, Flags) {
                    let behavior = behavior.apply::<M>();
                    self.map_with(|lane| lane.$name_with(behavior))
                }
            )*
        }
    };
}

elementary!(
    exp, exp_with: "`e^x`",
    exp_m1, exp_m1_with: "`e^x - 1`",
    exp2, exp2_with: "`2^x`",
    exp2_m1, exp2_m1_with: "`2^x - 1`",
    exp10, exp10_with: "`10^x`",
    exp10_m1, exp10_m1_with: "`10^x - 1`",
    log, log_with: "the natural logarithm `ln x`",
    log2, log2_with: "`log_2 x`",
    log10, log10_with: "`log_10 x`",
    log_p1, log_p1_with: "`ln(1 + x)`",
    log2_p1, log2_p1_with: "`log_2(1 + x)`",
    log10_p1, log10_p1_with: "`log_10(1 + x)`",
    sinh, sinh_with: "`sinh x`",
    cosh, cosh_with: "`cosh x`",
    tanh, tanh_with: "`tanh x`",
    asinh, asinh_with: "`asinh x`",
    acosh, acosh_with: "`acosh x`",
    atanh, atanh_with: "`atanh x`",
    sin_pi, sin_pi_with: "`sin(pi x)`",
    cos_pi, cos_pi_with: "`cos(pi x)`",
    tan_pi, tan_pi_with: "`tan(pi x)`",
    asin, asin_with: "`asin x`",
    acos, acos_with: "`acos x`",
    atan, atan_with: "`atan x`",
    asin_pi, asin_pi_with: "`asin(x) / pi`",
    acos_pi, acos_pi_with: "`acos(x) / pi`",
    atan_pi, atan_pi_with: "`atan(x) / pi`",
);

impl<S: Standard<W>, const W: usize, M: Mode, const N: usize> Lanes<Float<S, W, M>, N> {
    /// Applies `operation`, which returns flags, to each lane, and returns
    /// the lanes and the union of the flags.
    fn map_with(
        self,
        operation: impl FnMut(Float<S, W, M>) -> (Float<S, W, M>, Flags),
    ) -> (Self, Flags) {
        let (lanes, flags) = self.each_with(operation);
        (Self::new(lanes), flags)
    }

    /// Applies `operation`, which returns a result and flags, to each lane,
    /// and returns the results and the union of the flags.
    fn each_with<R>(self, operation: impl FnMut(Float<S, W, M>) -> (R, Flags)) -> ([R; N], Flags) {
        with_flags(self.lanes, operation)
    }

    /// Applies `operation`, which returns a result and flags, to each pair of
    /// lanes, and returns the results and the union of the flags.
    fn pairs_with<R>(
        self,
        other: Self,
        mut operation: impl FnMut(Float<S, W, M>, Float<S, W, M>) -> (R, Flags),
    ) -> ([R; N], Flags) {
        let pairs: [_; N] = self.pairs(other, |left, right| (left, right));
        with_flags(pairs, |(left, right)| operation(left, right))
    }

    /// Applies `operation`, which returns flags, to each pair of lanes, and
    /// returns the lanes and the union of the flags.
    fn zip_with(
        self,
        other: Self,
        operation: impl FnMut(Float<S, W, M>, Float<S, W, M>) -> (Float<S, W, M>, Flags),
    ) -> (Self, Flags) {
        let (lanes, flags) = self.pairs_with(other, operation);
        (Self::new(lanes), flags)
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
            let unit = Self::unit(Kind::ToInt);
            return self.lanes.map(|lane| lane.to_int_in::<I>(unit));
        }
        let Some(integers) = host::packed::to_int_i32(&self.lanes, &M::ENV) else {
            let unit = Self::unit(Kind::ToInt);
            return self.lanes.map(|lane| lane.to_int_in::<I>(unit));
        };
        // A lane that the packed path does not decide takes the scalar path.
        // The first such lane reads the environment for the others.
        let mut unit = None;
        let mut results = [ToInt::Nan; N];
        for ((result, lane), integer) in results.iter_mut().zip(self.lanes).zip(integers) {
            *result = match integer {
                Some(integer) => from_host_integer(i64::from(integer)),
                None => lane.to_int_in(*unit.get_or_insert_with(|| Self::unit(Kind::ToInt))),
            };
        }
        results
    }

    /// Converts each lane to the integer type `I` with the behavior, and
    /// returns the union of the flags.
    #[must_use]
    pub fn to_int_with<I: Integer>(self, behavior: impl Override) -> ([ToInt<I>; N], Flags) {
        let behavior = behavior.apply::<M>();
        self.each_with(|lane| lane.to_int_with::<I>(behavior))
    }

    /// Makes lanes from integers, each rounded with the default mode, lane 0
    /// first. Each lane takes the scalar conversion: a packed conversion of
    /// 32-bit integers measured no faster.
    #[must_use]
    pub fn from_int<I: Integer>(values: [I; N]) -> Self {
        let unit = Self::unit(Kind::FromInt);
        Self::new(values.map(|value| Float::from_int_in(value, unit)))
    }

    /// Makes lanes from integers, each rounded with the behavior, and returns
    /// the union of the flags.
    #[must_use]
    pub fn from_int_with<I: Integer>(values: [I; N], behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        let (lanes, flags) = with_flags(values, |value| Float::from_int_with(value, behavior));
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
        let pairs: [_; N] = core::array::from_fn(|index| (self.lanes[index], scales[index]));
        let (lanes, flags) = with_flags(pairs, |(lane, scale)| lane.scale_b_with(scale, behavior));
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

/// Returns the environment of the host unit that converts the lanes of one
/// conversion from `S` to `T`, read once, where the build has the scalar
/// path, as `Lanes::unit` reads it.
#[inline]
fn conversion_unit<S: Standard<W>, const W: usize, T: FloatType>() -> Unit {
    if host::convertible(S::HOST, T::HOST) {
        Unit::read(host::conversion_unit(S::HOST, T::HOST))
    } else {
        Unit::Read
    }
}

/// Converts each lane to the float type `T`, for lanes whose packed host
/// path declines, as `Lanes::map_out_of_line` does.
#[cold]
#[inline(never)]
fn convert_out_of_line<S: Standard<W>, const W: usize, M: Mode, const N: usize, T: FloatType>(
    lanes: Lanes<Float<S, W, M>, N>,
) -> Lanes<T, N> {
    let unit = conversion_unit::<S, W, T>();
    Lanes::new(lanes.lanes.map(|lane| lane.convert_in::<T>(unit)))
}

/// Converts each value to the float type `T` into the same index of `out`,
/// with the mode of `T`, one value at a time: for a build without the
/// packed path, for a chunk that holds a NaN, and for the values after the
/// last full chunk. The function runs out of line, so that the engine stays
/// out of the loop of the packed path, which calls it for a chunk that
/// holds a NaN.
#[inline(never)]
fn convert_each<S: Standard<W>, const W: usize, M: Mode, T: FloatType>(
    values: &[Float<S, W, M>],
    out: &mut [T],
) {
    let unit = conversion_unit::<S, W, T>();
    values
        .iter()
        .zip(out)
        .for_each(|(value, result)| *result = value.convert_in(unit));
}

/// Applies `operation`, which returns a result and flags, to each item, and
/// returns the results and the union of the flags. Every `_with` method of
/// `Lanes` reports its flags through this function.
fn with_flags<T, R, const N: usize>(
    items: [T; N],
    mut operation: impl FnMut(T) -> (R, Flags),
) -> ([R; N], Flags) {
    let mut flags = Flags::NONE;
    let results = items.map(|item| {
        let (result, item_flags) = operation(item);
        flags |= item_flags;
        result
    });
    (results, flags)
}
