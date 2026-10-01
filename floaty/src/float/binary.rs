//! The operations of the binary formats alone: `log_b` and the augmented
//! operations.

use super::Float;
use crate::binary::Layout;
use crate::env::{Behavior, Flags, Mode, Override};
use crate::format::internal::LimbConversion;
use crate::format::{Binary, Encoding, Standard, Storage, Width};

/// The two results of an augmented operation of IEEE 754-2019 section 9.5:
/// a head and a tail.
///
/// [`augmented_add`](Float::augmented_add),
/// [`augmented_sub`](Float::augmented_sub), and
/// [`augmented_mul`](Float::augmented_mul) of a binary format return it. The
/// head is the exact result rounded to nearest with ties toward zero. The
/// tail is the exact result minus the head, rounded the same way. So the
/// head and the tail hold the exact result, except when the head overflows
/// or the tail of a product underflows. The rounding direction of the
/// behavior does not apply. The other fields apply to both roundings, so
/// flush-to-zero and a precision limit can also drop bits of the tail.
///
/// The special cases follow IEEE 754-2019:
///
/// - A NaN, an infinite, or an unsupported operand gives the result of the
///   plain operation in both fields. Two zero operands do the same, and a
///   zero factor of a product does too.
/// - A zero head gives the head in both fields.
/// - An overflow of the head gives the head in both fields: the infinity, or
///   the NaN or the largest finite value of a format without an infinity or
///   with saturation.
/// - An exact head gives a zero tail with the sign of the head.
///
/// The flags describe the pair as one result. When the tail is the head,
/// they are the flags of the rounding of the head. Otherwise `INEXACT` and
/// `UNDERFLOW` come from the rounding of the tail, `TINY` reports a tiny head
/// or tail, and `ROUNDED_UP` reports that `head + tail` is above the exact
/// result in magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Augmented<F> {
    /// The exact result rounded to nearest, with ties toward zero.
    pub head: F,
    /// The exact result minus the head, rounded to nearest, with ties toward
    /// zero.
    pub tail: F,
}

impl<const E: u32, Enc: Encoding, const W: usize, M: Mode> Float<Binary<E, Enc>, W, M>
where
    Width<W>: Storage,
    Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    /// Returns the exponent of the leading bit, with the default mode.
    #[must_use]
    pub fn log_b(self) -> Self {
        self.log_b_with(M::default()).0
    }

    /// Returns the exponent of the leading bit, `floor(log2(|self|))`, as an
    /// integral value, as IEEE 754 `logB` does, and the flags.
    ///
    /// The integral value rounds to the format at its full precision, in the
    /// direction of the behavior, so the precision limit does not apply. Only
    /// a format with fewer significand bits than the integer needs rounds it,
    /// and then signals inexact. A zero gives what `-1 / 0` gives: negative
    /// infinity and divide-by-zero, or the NaN or the largest finite value of
    /// a format without an infinity. An infinity gives positive infinity.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let exponent = |bits| F64::from_bits(bits).log_b_with(Env::IEEE);
    /// // 0.75 is 1.5 * 2^-1.
    /// assert_eq!(exponent(0x3FE8_0000_0000_0000).0.to_bits(), 0xBFF0_0000_0000_0000);
    /// // The smallest subnormal value is 2^-1074.
    /// let (smallest, flags) = exponent(1);
    /// assert_eq!(smallest.to_bits(), 0xC090_C800_0000_0000);
    /// assert_eq!(flags, Flags::DENORMAL_INPUT);
    /// // A zero gives negative infinity.
    /// assert_eq!(exponent(0), (F64::from_bits(0xFFF0_0000_0000_0000), Flags::DIVIDE_BY_ZERO));
    /// ```
    #[must_use]
    pub fn log_b_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) =
            Layout::<E, Enc, W>::log_b(self.bits.to_limbs(), &behavior.apply::<M>().env());
        (Self::from_masked(LimbConversion::from_limbs(bits)), flags)
    }

    /// Returns `self + other` as a head and a tail, with the default mode.
    #[must_use]
    pub fn augmented_add(self, other: Self) -> Augmented<Self> {
        self.augmented_add_with(other, M::default()).0
    }

    /// Returns `self + other` as a head and a tail, as IEEE 754-2019
    /// `augmentedAddition` does, and the flags. [`Augmented`] states the
    /// rules.
    ///
    /// ```
    /// use floaty::{Env, F32, Flags};
    ///
    /// // 1 + 2^-23 plus 2^-24 is halfway between 1 + 2^-23 and 1 + 2^-22.
    /// let x = F32::from_bits(0x3F80_0001);
    /// let (sum, flags) = x.augmented_add_with(F32::from_bits(0x3380_0000), Env::IEEE);
    /// // The tie rounds toward zero, and the tail holds the rest exactly.
    /// assert_eq!(sum.head.to_bits(), 0x3F80_0001);
    /// assert_eq!(sum.tail.to_bits(), 0x3380_0000);
    /// assert_eq!(flags, Flags::NONE);
    /// ```
    #[must_use]
    pub fn augmented_add_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Augmented<Self>, Flags) {
        let (head, tail, flags) = Layout::<E, Enc, W>::augmented_add(
            self.bits.to_limbs(),
            other.bits.to_limbs(),
            false,
            &behavior.apply::<M>().env(),
        );
        (Self::augmented(head, tail), flags)
    }

    /// Returns `self - other` as a head and a tail, with the default mode.
    #[must_use]
    pub fn augmented_sub(self, other: Self) -> Augmented<Self> {
        self.augmented_sub_with(other, M::default()).0
    }

    /// Returns `self - other` as a head and a tail, as IEEE 754-2019
    /// `augmentedSubtraction` does, and the flags. [`Augmented`] states the
    /// rules.
    #[must_use]
    pub fn augmented_sub_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Augmented<Self>, Flags) {
        let (head, tail, flags) = Layout::<E, Enc, W>::augmented_add(
            self.bits.to_limbs(),
            other.bits.to_limbs(),
            true,
            &behavior.apply::<M>().env(),
        );
        (Self::augmented(head, tail), flags)
    }

    /// Returns `self * other` as a head and a tail, with the default mode.
    #[must_use]
    pub fn augmented_mul(self, other: Self) -> Augmented<Self> {
        self.augmented_mul_with(other, M::default()).0
    }

    /// Returns `self * other` as a head and a tail, as IEEE 754-2019
    /// `augmentedMultiplication` does, and the flags. [`Augmented`] states
    /// the rules.
    ///
    /// ```
    /// use floaty::{Env, F32, Flags};
    ///
    /// // 3.5 * 0.5714286 is 2 + 1.5 * 2^-24.
    /// let x = F32::from_bits(0x4060_0000);
    /// let (product, flags) = x.augmented_mul_with(F32::from_bits(0x3F12_4925), Env::IEEE);
    /// assert_eq!(product.head.to_bits(), 0x4000_0000);
    /// assert_eq!(product.tail.to_bits(), 0x33C0_0000);
    /// assert_eq!(flags, Flags::NONE);
    /// // The tail of a product can fall below the smallest subnormal value.
    /// let tiny = F32::from_bits(0x0080_0001);
    /// let (product, flags) = tiny.augmented_mul_with(F32::from_bits(0x3F00_0001), Env::IEEE);
    /// assert_eq!(product.head.to_bits(), 0x0040_0001);
    /// assert_eq!(product.tail.to_bits(), 0);
    /// assert_eq!(flags, Flags::UNDERFLOW | Flags::INEXACT | Flags::TINY);
    /// ```
    #[must_use]
    pub fn augmented_mul_with(
        self,
        other: Self,
        behavior: impl Override,
    ) -> (Augmented<Self>, Flags) {
        let (head, tail, flags) = Layout::<E, Enc, W>::augmented_mul(
            self.bits.to_limbs(),
            other.bits.to_limbs(),
            &behavior.apply::<M>().env(),
        );
        (Self::augmented(head, tail), flags)
    }

    /// Returns the results of an augmented operation from their limbs.
    fn augmented(
        head: <<Width<W> as Storage>::Bits as LimbConversion>::Limbs,
        tail: <<Width<W> as Storage>::Bits as LimbConversion>::Limbs,
    ) -> Augmented<Self> {
        Augmented {
            head: Self::from_masked(LimbConversion::from_limbs(head)),
            tail: Self::from_masked(LimbConversion::from_limbs(tail)),
        }
    }
}
