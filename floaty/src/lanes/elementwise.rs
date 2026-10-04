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
use crate::host::{self, Direction, Isa, Load, Operation};
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

/// An operation of two vectors that a view computes on the host unit, one
/// chunk at a time.
trait Combine: Copy {
    /// Returns the operation of each pair of binary32 encodings of `x` and
    /// `y` in the instruction set `I`. A result that rounds rounds in the
    /// direction `direction`.
    fn combine<I: Isa, const N: usize>(
        self,
        x: [u32; N],
        y: [u32; N],
        direction: impl Direction,
    ) -> Option<[u32; N]>;
}

impl Combine for Operation {
    #[inline]
    fn combine<I: Isa, const N: usize>(
        self,
        x: [u32; N],
        y: [u32; N],
        direction: impl Direction,
    ) -> Option<[u32; N]> {
        host::packed::elementwise::binary::<I, N, _>(x, y, self, direction)
    }
}

impl Combine for MinMax {
    /// The minimum and maximum select an operand, so no direction applies.
    #[inline]
    fn combine<I: Isa, const N: usize>(
        self,
        x: [u32; N],
        y: [u32; N],
        _direction: impl Direction,
    ) -> Option<[u32; N]> {
        host::packed::elementwise::min_max::<I, N>(x, y, self)
    }
}

/// Defines a view of one operation of two vectors. The operation `$operand`
/// computes a chunk of encodings on the host unit, as `Combine` states, and
/// `$engine` computes one value in the engine.
macro_rules! binary_view {
    ($(#[$doc:meta])* $name:ident, $operand:expr, $engine:ident) => {
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
            fn load_on<H: Isa, const N: usize>(
                self,
                start: usize,
                direction: impl Direction,
            ) -> Option<[u32; N]> {
                let x = self.0.load::<H, N>(start, direction)?;
                let y = self.1.load::<H, N>(start, direction)?;
                $operand.combine::<H, N>(x, y, direction)
            }

            #[inline]
            fn load_rest_on<H: Isa, const N: usize>(
                self,
                start: usize,
                direction: impl Direction,
            ) -> Option<[u32; N]> {
                let x = self.0.load_rest::<H, N>(start, direction)?;
                let y = self.1.load_rest::<H, N>(start, direction)?;
                let chunk = $operand.combine::<H, N>(x, y, direction)?;
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

            #[inline]
            fn check_store<T>(self, out: &[Cell<T>]) {
                self.0.check_store(out);
                self.1.check_store(out);
            }
        }
    };
}

binary_view!(
    /// The sum of two vectors: value `i` is `x[i] + y[i]`, rounded.
    Sum,
    Operation::Add,
    add_with
);
binary_view!(
    /// The difference of two vectors: value `i` is `x[i] - y[i]`, rounded.
    Difference,
    Operation::Sub,
    sub_with
);
binary_view!(
    /// The product of two vectors: value `i` is `x[i] * y[i]`, rounded.
    Product,
    Operation::Mul,
    mul_with
);
binary_view!(
    /// The quotient of two vectors: value `i` is `x[i] / y[i]`, rounded.
    Quotient,
    Operation::Div,
    div_with
);
binary_view!(
    /// The IEEE 754-2019 `minimum` of two vectors: value `i` is the smaller
    /// of `x[i]` and `y[i]`, -0 below +0, and a NaN when either is a NaN.
    Minimum,
    MinMax::Minimum,
    minimum_with
);
binary_view!(
    /// The IEEE 754-2019 `maximum` of two vectors: value `i` is the larger
    /// of `x[i]` and `y[i]`, +0 above -0, and a NaN when either is a NaN.
    Maximum,
    MinMax::Maximum,
    maximum_with
);
binary_view!(
    /// The IEEE 754-2019 `minimumNumber` of two vectors: value `i` is the
    /// smaller of `x[i]` and `y[i]`, -0 below +0, and the other value when
    /// one is a NaN.
    MinimumNumber,
    MinMax::MinimumNumber,
    minimum_number_with
);
binary_view!(
    /// The IEEE 754-2019 `maximumNumber` of two vectors: value `i` is the
    /// larger of `x[i]` and `y[i]`, +0 above -0, and the other value when
    /// one is a NaN.
    MaximumNumber,
    MinMax::MaximumNumber,
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
    fn load_on<H: Isa, const N: usize>(
        self,
        _start: usize,
        _direction: impl Direction,
    ) -> Option<[u32; N]> {
        T::widen_on_host::<H, N>(&[self.value; N])
    }

    #[inline]
    fn load_rest_on<H: Isa, const N: usize>(
        self,
        start: usize,
        _direction: impl Direction,
    ) -> Option<[u32; N]> {
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
    fn load_on<H: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]> {
        let values = self.values.load::<H, N>(start, direction)?;
        host::packed::elementwise::round_to_integral::<H, N>(values, self.rounding)
    }

    #[inline]
    fn load_rest_on<H: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]> {
        let values = self.values.load_rest::<H, N>(start, direction)?;
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

    #[inline]
    fn check_store<T>(self, out: &[Cell<T>]) {
        self.values.check_store(out);
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
    fn load_on<H: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]> {
        Some(magnitudes(self.0.load::<H, N>(start, direction)?))
    }

    #[inline]
    fn load_rest_on<H: Isa, const N: usize>(
        self,
        start: usize,
        direction: impl Direction,
    ) -> Option<[u32; N]> {
        Some(magnitudes(self.0.load_rest::<H, N>(start, direction)?))
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

    #[inline]
    fn check_store<T>(self, out: &[Cell<T>]) {
        self.0.check_store(out);
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

/// Rejects a cell operand that overlaps the destination at different
/// indices. Empty ranges and zero-sized destinations do not overlap.
#[inline]
fn check_cell_store<T, U>(values: &[Cell<T>], out: &[Cell<U>]) {
    let source = values.as_ptr_range();
    let destination = out.as_ptr_range();
    let same_indices =
        source.start.addr() == destination.start.addr() && size_of::<T>() == size_of::<U>();
    let disjoint = source.start == source.end
        || destination.start == destination.end
        || source.start.addr() >= destination.end.addr()
        || destination.start.addr() >= source.end.addr();
    assert!(
        same_indices || disjoint,
        "a cell store must not shift an overlapping operand"
    );
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
            fn load_on<H: Isa, const N: usize>(
                self,
                start: usize,
                _direction: impl Direction,
            ) -> Option<[u32; N]> {
                Some(cell_encodings(self, start, $encode))
            }

            #[inline]
            fn load_rest_on<H: Isa, const N: usize>(
                self,
                start: usize,
                _direction: impl Direction,
            ) -> Option<[u32; N]> {
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

            #[inline]
            fn check_store<T>(self, out: &[Cell<T>]) {
                check_cell_store(self, out);
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
/// `i` goes to element `i`. Cell operands must be disjoint from the
/// destination, or use the same cells at the same indices. A store rejects
/// shifted overlap before the first write. The trait is sealed.
pub trait Output<T>: Sealed {
    /// Returns the number of elements.
    #[doc(hidden)]
    fn count(&self) -> usize;

    /// Checks that cell operands do not overlap the destination at
    /// different indices. An exclusive slice cannot alias an operand.
    #[doc(hidden)]
    #[inline]
    fn check_values(&self, _values: impl Vector) {}

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
    fn check_values(&self, values: impl Vector) {
        values.check_store(self);
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
