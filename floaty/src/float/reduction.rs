//! The reduction operations of IEEE 754-2019 section 9.4, for the binary
//! formats alone.

use super::Float;
use crate::binary::{Factor, Layout, Summand};
use crate::env::{Behavior, Flags, Mode, Override};
use crate::format::internal::LimbConversion;
use crate::format::{Binary, Encoding, Standard, Storage, Width};

/// The limbs of an encoding of width `W`.
type StorageLimbs<const W: usize> = <<Width<W> as Storage>::Bits as LimbConversion>::Limbs;

/// A product as a value and a power of two: `value * 2^scale`.
///
/// [`scaled_product`](Float::scaled_product),
/// [`scaled_product_sum`](Float::scaled_product_sum), and
/// [`scaled_product_difference`](Float::scaled_product_difference) of a
/// binary format return it, as IEEE 754-2019 `scaledProd`,
/// `scaledProdSum`, and `scaledProdDiff` do.
///
/// Each step multiplies the product so far by the next factor, and rounds the
/// result to the precision in the direction of the behavior, without an
/// exponent limit. So the value of a product depends on the order of its
/// factors. A finite nonzero product has a value in `[1, 2)` or `(-2, -1]`,
/// and the flags report `INEXACT` when a step rounds. The flags never report
/// `OVERFLOW`, `UNDERFLOW`, `TINY`, or `ROUNDED_UP`.
///
/// The special cases follow ISO/IEC TS 18661-4, in this order:
///
/// - An unsupported operand gives the default NaN and signals invalid.
/// - A NaN operand gives the NaN that the NaN rule selects among the NaN
///   operands, in order, and a signaling NaN signals invalid.
/// - A sum of infinities of different signs, or a zero factor and an
///   infinite factor, give the default NaN and signal invalid.
/// - An infinite factor gives an infinity, and a zero factor a zero, with the
///   sign of the product of the factors.
/// - A scale outside the range of `i64` gives the default NaN and signals
///   invalid.
///
/// A zero, an infinity, or a NaN has the scale 0. An empty product is 1 with
/// the scale 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Scaled<F> {
    /// The scaled product.
    pub value: F,
    /// The power of two that scales `value` to the product.
    pub scale: i64,
}

impl<const E: u32, Enc: Encoding, const W: usize, M: Mode> Float<Binary<E, Enc>, W, M>
where
    Width<W>: Storage,
    Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    /// Returns the sum of `values`, rounded once, with the default mode.
    #[must_use]
    pub fn sum(values: impl IntoIterator<Item = Self, IntoIter: Clone>) -> Self {
        Self::sum_with(values, M::default()).0
    }

    /// Returns the sum of `values`, as IEEE 754-2019 `sum` does, and the
    /// flags.
    ///
    /// The exact sum rounds once in the direction of the behavior, so the
    /// order of the values does not change the result. The flags are those of
    /// that rounding, and `DENORMAL_INPUT` for a subnormal value. An empty sum
    /// is +0. An exact zero sum of nonzero terms is +0, or -0 when the
    /// rounding is toward negative, and a sum of zeros alone takes the sign
    /// rule of an exact zero sum of two zeros.
    ///
    /// A NaN gives the NaN that the NaN rule selects among the NaN values, in
    /// order, and a signaling NaN signals invalid. Infinities of both signs
    /// give the default NaN and signal invalid. Otherwise an infinity gives
    /// that infinity. An unsupported value gives the default NaN and signals
    /// invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let values = [0x7FE0_0000_0000_0000, 0x3FF0_0000_0000_0000, 0xFFE0_0000_0000_0000];
    /// // 2^1023 + 1 - 2^1023 is exactly 1.
    /// let (sum, flags) = F64::sum_with(values.map(F64::from_bits), Env::IEEE);
    /// assert_eq!((sum.to_bits(), flags), (0x3FF0_0000_0000_0000, Flags::NONE));
    /// ```
    #[must_use]
    pub fn sum_with(
        values: impl IntoIterator<Item = Self, IntoIter: Clone>,
        behavior: impl Override,
    ) -> (Self, Flags) {
        Self::reduction(values, Summand::Value, behavior)
    }

    /// Returns the sum of the magnitudes of `values`, rounded once, with the
    /// default mode.
    #[must_use]
    pub fn sum_abs(values: impl IntoIterator<Item = Self, IntoIter: Clone>) -> Self {
        Self::sum_abs_with(values, M::default()).0
    }

    /// Returns the sum of the magnitudes of `values`, as IEEE 754-2019
    /// `sumAbs` does, and the flags.
    ///
    /// The sum rounds as [`sum_with`](Self::sum_with) does, and a zero sum is
    /// +0. An infinity gives +inf, even with a quiet NaN, as `hypot` does.
    /// Otherwise a NaN gives the NaN of the NaN rule, and a signaling NaN
    /// signals invalid.
    #[must_use]
    pub fn sum_abs_with(
        values: impl IntoIterator<Item = Self, IntoIter: Clone>,
        behavior: impl Override,
    ) -> (Self, Flags) {
        Self::reduction(values, Summand::Magnitude, behavior)
    }

    /// Returns the sum of the squares of `values`, rounded once, with the
    /// default mode.
    #[must_use]
    pub fn sum_square(values: impl IntoIterator<Item = Self, IntoIter: Clone>) -> Self {
        Self::sum_square_with(values, M::default()).0
    }

    /// Returns the sum of the squares of `values`, as IEEE 754-2019
    /// `sumSquare` does, and the flags. The exact squares add and round once,
    /// and the special cases are those of [`sum_abs_with`](Self::sum_abs_with).
    #[must_use]
    pub fn sum_square_with(
        values: impl IntoIterator<Item = Self, IntoIter: Clone>,
        behavior: impl Override,
    ) -> (Self, Flags) {
        Self::reduction(values, Summand::Square, behavior)
    }

    /// Returns the sum of the products of `pairs`, rounded once, with the
    /// default mode.
    #[must_use]
    pub fn dot(pairs: impl IntoIterator<Item = (Self, Self), IntoIter: Clone>) -> Self {
        Self::dot_with(pairs, M::default()).0
    }

    /// Returns the sum of the products of `pairs`, as IEEE 754-2019 `dot`
    /// does, and the flags.
    ///
    /// The exact products add and round once, as in
    /// [`sum_with`](Self::sum_with). A NaN gives the NaN that the NaN rule
    /// selects among the NaN operands, in order, and signals invalid for a
    /// signaling NaN or a product `0 * inf`. Otherwise a product `0 * inf`,
    /// or infinite products of both signs, give the default NaN and signal
    /// invalid, and an infinite product gives that infinity.
    ///
    /// ```
    /// use floaty::{Env, F32};
    ///
    /// let pair = |x, y| (F32::from_bits(x), F32::from_bits(y));
    /// // (1 + 2^-23)(1 - 2^-23) - 1 is exactly -2^-46, which a product
    /// // rounded first would lose.
    /// let pairs = [pair(0x3F80_0001, 0x3F7F_FFFE), pair(0x3F80_0000, 0xBF80_0000)];
    /// assert_eq!(F32::dot_with(pairs, Env::IEEE).0.to_bits(), 0xA880_0000);
    /// ```
    #[must_use]
    pub fn dot_with(
        pairs: impl IntoIterator<Item = (Self, Self), IntoIter: Clone>,
        behavior: impl Override,
    ) -> (Self, Flags) {
        let limbs = pairs
            .into_iter()
            .map(|(x, y)| (x.bits.to_limbs(), y.bits.to_limbs()));
        let (bits, flags) = Layout::<E, Enc, W>::dot(limbs, &behavior.apply::<M>().env());
        (Self::from_masked(LimbConversion::from_limbs(bits)), flags)
    }

    /// Returns the product of `values` with a scale, with the default mode.
    #[must_use]
    pub fn scaled_product(values: impl IntoIterator<Item = Self, IntoIter: Clone>) -> Scaled<Self> {
        Self::scaled_product_with(values, M::default()).0
    }

    /// Returns the product of `values` as a value and a scale, as IEEE
    /// 754-2019 `scaledProd` does, and the flags. [`Scaled`] states the rules.
    ///
    /// ```
    /// use floaty::{Env, F64};
    ///
    /// // 2^1000 * 2^1000 * 3 overflows binary64, but its scaled product does
    /// // not: 1.5 * 2^2001.
    /// let values = [0x7E70_0000_0000_0000, 0x7E70_0000_0000_0000, 0x4008_0000_0000_0000];
    /// let (product, _) = F64::scaled_product_with(values.map(F64::from_bits), Env::IEEE);
    /// assert_eq!((product.value.to_bits(), product.scale), (0x3FF8_0000_0000_0000, 2001));
    /// ```
    #[must_use]
    pub fn scaled_product_with(
        values: impl IntoIterator<Item = Self, IntoIter: Clone>,
        behavior: impl Override,
    ) -> (Scaled<Self>, Flags) {
        let values = values.into_iter().map(|value| value.bits.to_limbs());
        let factors = values.clone().map(Factor::Value);
        let (bits, scale, flags) =
            Layout::<E, Enc, W>::scaled_product(factors, values, &behavior.apply::<M>().env());
        (Self::scaled(bits, scale), flags)
    }

    /// Returns the product of the sums of `pairs` with a scale, with the
    /// default mode.
    #[must_use]
    pub fn scaled_product_sum(
        pairs: impl IntoIterator<Item = (Self, Self), IntoIter: Clone>,
    ) -> Scaled<Self> {
        Self::scaled_product_sum_with(pairs, M::default()).0
    }

    /// Returns the product of the sums of `pairs` as a value and a scale, as
    /// IEEE 754-2019 `scaledProdSum` does, and the flags. Each sum is exact.
    /// [`Scaled`] states the rules.
    #[must_use]
    pub fn scaled_product_sum_with(
        pairs: impl IntoIterator<Item = (Self, Self), IntoIter: Clone>,
        behavior: impl Override,
    ) -> (Scaled<Self>, Flags) {
        Self::scaled_pairs(pairs, Factor::Sum, behavior)
    }

    /// Returns the product of the differences of `pairs` with a scale, with
    /// the default mode.
    #[must_use]
    pub fn scaled_product_difference(
        pairs: impl IntoIterator<Item = (Self, Self), IntoIter: Clone>,
    ) -> Scaled<Self> {
        Self::scaled_product_difference_with(pairs, M::default()).0
    }

    /// Returns the product of the differences of `pairs` as a value and a
    /// scale, as IEEE 754-2019 `scaledProdDiff` does, and the flags. Each
    /// difference is exact. [`Scaled`] states the rules.
    #[must_use]
    pub fn scaled_product_difference_with(
        pairs: impl IntoIterator<Item = (Self, Self), IntoIter: Clone>,
        behavior: impl Override,
    ) -> (Scaled<Self>, Flags) {
        Self::scaled_pairs(pairs, Factor::Difference, behavior)
    }

    /// Runs a sum of one vector.
    fn reduction(
        values: impl IntoIterator<Item = Self, IntoIter: Clone>,
        summand: Summand,
        behavior: impl Override,
    ) -> (Self, Flags) {
        let limbs = values.into_iter().map(|value| value.bits.to_limbs());
        let (bits, flags) =
            Layout::<E, Enc, W>::reduce(limbs, summand, &behavior.apply::<M>().env());
        (Self::from_masked(LimbConversion::from_limbs(bits)), flags)
    }

    /// Runs a scaled product of the factors that `factor` makes from pairs.
    fn scaled_pairs(
        pairs: impl IntoIterator<Item = (Self, Self), IntoIter: Clone>,
        factor: fn(StorageLimbs<W>, StorageLimbs<W>) -> Factor<StorageLimbs<W>>,
        behavior: impl Override,
    ) -> (Scaled<Self>, Flags) {
        let pairs = pairs
            .into_iter()
            .map(|(x, y)| (x.bits.to_limbs(), y.bits.to_limbs()));
        let factors = pairs.clone().map(move |(x, y)| factor(x, y));
        let operands = pairs.flat_map(|(x, y)| [x, y]);
        let (bits, scale, flags) =
            Layout::<E, Enc, W>::scaled_product(factors, operands, &behavior.apply::<M>().env());
        (Self::scaled(bits, scale), flags)
    }

    /// Returns a scaled product from its limbs.
    fn scaled(bits: StorageLimbs<W>, scale: i64) -> Scaled<Self> {
        Scaled {
            value: Self::from_masked(LimbConversion::from_limbs(bits)),
            scale,
        }
    }
}
