//! The elementwise views of the slice operations of [`Lanes`](crate::Lanes),
//! and the destinations of their stores.
//!
//! A view is a [`Vector`] whose value `i` is one operation of value `i` of
//! its operands, computed when a kernel or a store reads it. Views nest, so
//! one pass over the operands evaluates an expression without a buffer for
//! each step:
//!
//! ```
//! use floaty::elementwise::{Difference, Product, Splat};
//! use floaty::{F32, Lanes};
//!
//! let x = [1.0_f32, 2.0, 3.0];
//! let lo = [0.5_f32, 0.5, 0.5];
//! let half = F32::from_bits(0x3F00_0000);
//! let mut out = [0.0_f32; 3];
//! // out[i] = (x[i] - lo[i]) * 0.5, each step rounded.
//! Lanes::<F32, 8>::store(
//!     Product(Difference(&x[..], &lo[..]), Splat::new(half, 3)),
//!     &mut out[..],
//! );
//! assert_eq!(out, [0.25, 0.75, 1.25]);
//! ```
//!
//! A view of two vectors panics, when a kernel or a store reads it, if the
//! vectors have different counts of values. Each operation rounds in the
//! behavior of the call that reads the view.

use core::cell::Cell;

use super::vector::{Single, Vector, Widen};
use crate::env::{Behavior, Flags, Mode, Rounding};
use crate::float::{F32, Float};
use crate::format::Binary;
use crate::format::internal::MinMax;
use crate::host::{self, Isa, Load, Operation};
use crate::sealed::Sealed;

/// Returns the count of two vectors of one view.
///
/// # Panics
///
/// Panics when the vectors have different counts.
#[inline]
fn common_count(x: impl Load, y: impl Load) -> usize {
    let count = x.count();
    assert_eq!(count, y.count(), "the vectors of a view have one count");
    count
}

/// Returns a chunk with +0 in the lanes from `end`, the lanes past the last
/// value, as `Load::load_rest` requires.
#[inline]
fn padded<const N: usize>(mut bits: [u32; N], end: usize) -> [u32; N] {
    bits[end..].fill(0);
    bits
}

/// Defines a view of one operation of two vectors. `$host` computes a chunk
/// of encodings on the host unit with the operand `$operand`, and `$engine`
/// computes one value in the engine.
macro_rules! binary_view {
    ($(#[$doc:meta])* $name:ident, $host:ident($operand:expr), $engine:ident) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name<X, Y>(pub X, pub Y);

        impl<X: Vector, Y: Vector> Sealed for $name<X, Y> {}

        impl<X: Vector, Y: Vector> Load for $name<X, Y> {
            #[inline]
            fn count(self) -> usize {
                common_count(self.0, self.1)
            }

            #[inline]
            fn load_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
                let (x, y) = (self.0.load::<H, N>(start)?, self.1.load::<H, N>(start)?);
                host::packed::elementwise::$host::<H, N>(x, y, $operand)
            }

            #[inline]
            fn load_rest_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
                let (x, y) = (
                    self.0.load_rest::<H, N>(start)?,
                    self.1.load_rest::<H, N>(start)?,
                );
                let chunk = host::packed::elementwise::$host::<H, N>(x, y, $operand)?;
                Some(padded(chunk, self.count() - start))
            }

            #[inline]
            fn part(self, start: usize, count: usize) -> Self {
                Self(self.0.part(start, count), self.1.part(start, count))
            }
        }

        impl<X: Vector, Y: Vector> Vector for $name<X, Y> {
            #[inline]
            fn value_with<M: Mode, B: Behavior>(
                self,
                index: usize,
                behavior: B,
            ) -> (Single<M>, Flags) {
                let (x, x_flags) = self.0.value_with::<M, B>(index, behavior);
                let (y, y_flags) = self.1.value_with::<M, B>(index, behavior);
                let (value, flags) = x.$engine(y, behavior);
                (value, x_flags | y_flags | flags)
            }
        }
    };
}

binary_view!(
    /// The sum of two vectors: value `i` is `x[i] + y[i]`, rounded.
    Sum,
    binary(Operation::Add),
    add_with
);
binary_view!(
    /// The difference of two vectors: value `i` is `x[i] - y[i]`, rounded.
    Difference,
    binary(Operation::Sub),
    sub_with
);
binary_view!(
    /// The product of two vectors: value `i` is `x[i] * y[i]`, rounded.
    Product,
    binary(Operation::Mul),
    mul_with
);
binary_view!(
    /// The quotient of two vectors: value `i` is `x[i] / y[i]`, rounded.
    Quotient,
    binary(Operation::Div),
    div_with
);
binary_view!(
    /// The IEEE 754-2019 `minimum` of two vectors: value `i` is the smaller
    /// of `x[i]` and `y[i]`, -0 below +0, and a NaN when either is a NaN.
    Minimum,
    min_max(MinMax::Minimum),
    minimum_with
);
binary_view!(
    /// The IEEE 754-2019 `maximum` of two vectors: value `i` is the larger
    /// of `x[i]` and `y[i]`, +0 above -0, and a NaN when either is a NaN.
    Maximum,
    min_max(MinMax::Maximum),
    maximum_with
);
binary_view!(
    /// The IEEE 754-2019 `minimumNumber` of two vectors: value `i` is the
    /// smaller of `x[i]` and `y[i]`, -0 below +0, and the other value when
    /// one is a NaN.
    MinimumNumber,
    min_max(MinMax::MinimumNumber),
    minimum_number_with
);
binary_view!(
    /// The IEEE 754-2019 `maximumNumber` of two vectors: value `i` is the
    /// larger of `x[i]` and `y[i]`, +0 above -0, and the other value when
    /// one is a NaN.
    MaximumNumber,
    min_max(MinMax::MaximumNumber),
    maximum_number_with
);

/// One value repeated: value `i` is `value`, widened to binary32 as
/// [`Widen`] states, for each `i` below `count`.
#[derive(Clone, Copy, Debug)]
pub struct Splat<T> {
    /// The value.
    pub value: T,
    /// The count of values.
    pub count: usize,
}

impl<T: Widen> Splat<T> {
    /// Makes a vector of `count` copies of `value`.
    #[must_use]
    pub const fn new(value: T, count: usize) -> Self {
        Self { value, count }
    }
}

impl<T: Widen> Sealed for Splat<T> {}

impl<T: Widen> Load for Splat<T> {
    #[inline]
    fn count(self) -> usize {
        self.count
    }

    #[inline]
    fn load_on<H: Isa, const N: usize>(self, _start: usize) -> Option<[u32; N]> {
        T::widen_on_host::<H, N>(&[self.value; N])
    }

    #[inline]
    fn load_rest_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
        Some(padded(
            T::widen_on_host::<H, N>(&[self.value; N])?,
            self.count - start,
        ))
    }

    #[inline]
    fn part(self, _start: usize, count: usize) -> Self {
        Self::new(self.value, count)
    }
}

impl<T: Widen> Vector for Splat<T> {
    #[inline]
    fn value_with<M: Mode, B: Behavior>(self, _index: usize, behavior: B) -> (Single<M>, Flags) {
        self.value.widen_with(behavior)
    }
}

/// The values of a vector rounded to integral values in a fixed direction:
/// the IEEE 754 operations `roundToIntegralTiesToEven`,
/// `roundToIntegralTiesToAway`, `roundToIntegralTowardZero`, and the others
/// of [`Rounding`]. The direction replaces the rounding direction of the
/// behavior of the call, and every other field of the behavior applies.
#[derive(Clone, Copy, Debug)]
pub struct RoundToIntegral<X> {
    /// The vector.
    pub values: X,
    /// The rounding direction.
    pub rounding: Rounding,
}

impl<X: Vector> Sealed for RoundToIntegral<X> {}

impl<X: Vector> Load for RoundToIntegral<X> {
    #[inline]
    fn count(self) -> usize {
        self.values.count()
    }

    #[inline]
    fn load_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
        let values = self.values.load::<H, N>(start)?;
        host::packed::elementwise::round_to_integral::<H, N>(values, self.rounding)
    }

    #[inline]
    fn load_rest_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
        let values = self.values.load_rest::<H, N>(start)?;
        host::packed::elementwise::round_to_integral::<H, N>(values, self.rounding)
    }

    #[inline]
    fn part(self, start: usize, count: usize) -> Self {
        Self {
            values: self.values.part(start, count),
            rounding: self.rounding,
        }
    }
}

impl<X: Vector> Vector for RoundToIntegral<X> {
    #[inline]
    fn value_with<M: Mode, B: Behavior>(self, index: usize, behavior: B) -> (Single<M>, Flags) {
        let (value, value_flags) = self.values.value_with::<M, B>(index, behavior);
        let (rounded, flags) = value.round_to_integral_with(behavior.rounded(self.rounding));
        (rounded, value_flags | flags)
    }
}

/// The absolute values of a vector: value `i` is `x[i]` with the sign bit
/// clear. Only the sign bit changes, so a signaling NaN stays signaling.
#[derive(Clone, Copy, Debug)]
pub struct Abs<X>(pub X);

impl<X: Vector> Sealed for Abs<X> {}

impl<X: Vector> Load for Abs<X> {
    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn load_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
        Some(magnitudes(self.0.load::<H, N>(start)?))
    }

    #[inline]
    fn load_rest_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
        Some(magnitudes(self.0.load_rest::<H, N>(start)?))
    }

    #[inline]
    fn part(self, start: usize, count: usize) -> Self {
        Self(self.0.part(start, count))
    }
}

impl<X: Vector> Vector for Abs<X> {
    #[inline]
    fn value_with<M: Mode, B: Behavior>(self, index: usize, behavior: B) -> (Single<M>, Flags) {
        let (value, flags) = self.0.value_with::<M, B>(index, behavior);
        (value.abs(), flags)
    }
}

/// Returns binary32 encodings with the sign bit clear.
#[inline]
fn magnitudes<const N: usize>(mut bits: [u32; N]) -> [u32; N] {
    for bits in &mut bits {
        *bits &= 0x7FFF_FFFF;
    }
    bits
}

/// Returns the binary32 encodings of the `N` cells of `cells` from `start`.
/// `cells` holds at least `start + N` cells.
#[inline]
fn cell_encodings<T: Copy, const N: usize>(
    cells: &[Cell<T>],
    start: usize,
    encode: impl Fn(T) -> u32,
) -> [u32; N] {
    let chunk: &[Cell<T>; N] = cells[start..start + N]
        .try_into()
        .expect("the caller loads a chunk inside the vector");
    let mut bits = [0; N];
    bits.iter_mut()
        .zip(chunk)
        .for_each(|(bits, cell)| *bits = encode(cell.get()));
    bits
}

/// Returns the binary32 encodings of the cells of `cells` from `start`, with
/// +0 past the last cell. `cells` holds more than `start` cells.
#[inline]
fn cell_encodings_rest<T: Copy, const N: usize>(
    cells: &[Cell<T>],
    start: usize,
    encode: impl Fn(T) -> u32,
) -> [u32; N] {
    let mut bits = [0; N];
    bits.iter_mut()
        .zip(&cells[start..])
        .for_each(|(bits, cell)| *bits = encode(cell.get()));
    bits
}

/// Implements [`Vector`] for cells of a binary32 type `$element`, whose
/// encoding `$encode` returns. Cells let a store write a vector that its
/// values read.
macro_rules! cells {
    ($element:ty, $encode:expr, $value:expr $(, $generic:ident: $bound:path)?) => {
        impl$(<$generic: $bound>)? Load for &[Cell<$element>] {
            #[inline]
            fn count(self) -> usize {
                self.len()
            }

            #[inline]
            fn load_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
                Some(cell_encodings(self, start, $encode))
            }

            #[inline]
            fn load_rest_on<H: Isa, const N: usize>(self, start: usize) -> Option<[u32; N]> {
                Some(cell_encodings_rest(self, start, $encode))
            }

            #[inline]
            fn part(self, start: usize, count: usize) -> Self {
                &self[start..start + count]
            }
        }

        impl$(<$generic: $bound>)? Vector for &[Cell<$element>] {
            #[inline]
            fn value_with<M: Mode, B: Behavior>(
                self,
                index: usize,
                behavior: B,
            ) -> (Single<M>, Flags) {
                $value(self[index].get()).widen_with(behavior)
            }
        }
    };
}

cells!(
    Float<Binary<8>, 32, I>,
    Float::to_bits,
    |value: Float<Binary<8>, 32, I>| value,
    I: Mode
);
cells!(f32, f32::to_bits, F32::from);

/// A destination of the stores of [`Lanes`](crate::Lanes): `&mut [T]`, or
/// `&[Cell<T>]` for a store into a vector that its values also read. Value
/// `i` goes to element `i`. The trait is sealed.
pub trait Output<T>: Sealed {
    /// Returns the number of elements.
    #[doc(hidden)]
    fn count(&self) -> usize;

    /// Writes `values` into the elements from `start`. The destination holds
    /// at least `start + values.len()` elements.
    #[doc(hidden)]
    fn write(&mut self, start: usize, values: impl ExactSizeIterator<Item = T>);
}

impl<T> Sealed for &mut [T] {}

impl<T> Output<T> for &mut [T] {
    #[inline]
    fn count(&self) -> usize {
        self.len()
    }

    #[inline]
    fn write(&mut self, start: usize, values: impl ExactSizeIterator<Item = T>) {
        let end = start + values.len();
        self[start..end]
            .iter_mut()
            .zip(values)
            .for_each(|(element, value)| *element = value);
    }
}

impl<T> Sealed for &[Cell<T>] {}

impl<T> Output<T> for &[Cell<T>] {
    #[inline]
    fn count(&self) -> usize {
        self.len()
    }

    #[inline]
    fn write(&mut self, start: usize, values: impl ExactSizeIterator<Item = T>) {
        let end = start + values.len();
        self[start..end]
            .iter()
            .zip(values)
            .for_each(|(element, value)| element.set(value));
    }
}
