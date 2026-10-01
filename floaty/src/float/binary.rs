//! The operations of the binary formats alone: `log_b`.

use super::Float;
use crate::binary::Layout;
use crate::env::{Behavior, Flags, Mode, Override};
use crate::format::internal::LimbConversion;
use crate::format::{Binary, Encoding, Standard, Storage, Width};

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
}
