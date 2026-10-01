//! The operations of the decimal formats alone: the quantum operations of
//! IEEE 754-2019 section 5.3.2, `log_b`, and the NaN payload operations.

use super::Float;
use crate::decimal::DecimalLayout;
use crate::env::{Behavior, Flags, Mode, Override};
use crate::format::internal::LimbConversion;
use crate::format::{Decimal, DecimalEncoding, Standard, Storage, Width};

impl<Enc: DecimalEncoding, const W: usize, M: Mode> Float<Decimal<Enc>, W, M>
where
    Width<W>: Storage,
    Decimal<Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    /// Returns `self` with the exponent of `exponent_of`, rounded with the
    /// default mode.
    #[must_use]
    pub fn quantize(self, exponent_of: Self) -> Self {
        self.quantize_with(exponent_of, M::default()).0
    }

    /// Returns `self` with the exponent of `exponent_of`, as IEEE 754
    /// `quantize` does, and the flags.
    ///
    /// Digits that the new exponent drops round in the direction of the
    /// behavior and signal inexact. A coefficient that the new exponent makes
    /// longer than the precision, or an infinity with a number, signals
    /// invalid. The operation signals no underflow or overflow.
    ///
    /// ```
    /// use floaty::{D64Bid, Decoded, Env, Exact, Rounding};
    ///
    /// let value = |coefficient, exponent| {
    ///     let exact = Exact { negative: false, exponent, significand: [coefficient], sticky: false };
    ///     D64Bid::round(exact, Env::IEEE).0
    /// };
    /// // 2.17 quantized to 0.1 is 2.2.
    /// let (rounded, _) = value(217, -2).quantize_with(value(1, -1), Rounding::TiesToEven);
    /// assert_eq!(rounded.decode::<1>(), Decoded::Finite { negative: false, exponent: -1, significand: [22] });
    /// ```
    #[must_use]
    pub fn quantize_with(self, exponent_of: Self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::quantize(
            self.bits.to_limbs(),
            exponent_of.bits.to_limbs(),
            &behavior.apply::<M>().env(),
        );
        (Self::from_masked(LimbConversion::from_limbs(bits)), flags)
    }

    /// Returns `true` when both values have the same exponent, or both are
    /// infinities, or both are NaNs, as IEEE 754 `sameQuantum` does. The
    /// operation signals nothing.
    #[must_use]
    pub fn same_quantum(self, other: Self) -> bool {
        DecimalLayout::<Enc, W>::same_quantum(self.bits.to_limbs(), other.bits.to_limbs())
    }

    /// Returns the quantum of the value, `1 * 10^exponent`, with the default
    /// mode.
    #[must_use]
    pub fn quantum(self) -> Self {
        self.quantum_with(M::default()).0
    }

    /// Returns the quantum of the value, `1 * 10^exponent`, as IEEE 754
    /// `quantum` does, and the flags. An infinity gives positive infinity.
    #[must_use]
    pub fn quantum_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::quantum(self.bits.to_limbs(), &behavior.apply::<M>().env());
        (Self::from_masked(LimbConversion::from_limbs(bits)), flags)
    }

    /// Returns the exponent of the leading digit, with the default mode.
    #[must_use]
    pub fn log_b(self) -> Self {
        self.log_b_with(M::default()).0
    }

    /// Returns the exponent of the leading digit, as an integral value, as
    /// IEEE 754 `logB` does, and the flags. A zero gives negative infinity and
    /// signals divide-by-zero, and an infinity gives positive infinity.
    #[must_use]
    pub fn log_b_with(self, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::log_b(self.bits.to_limbs(), &behavior.apply::<M>().env());
        (Self::from_masked(LimbConversion::from_limbs(bits)), flags)
    }

    /// Returns the payload of a NaN, as IEEE 754-2019 `getPayload` does.
    ///
    /// The payload is the value of the trailing significand field, as an
    /// integer with exponent 0. A payload above `10^(p - 1) - 1` is not
    /// canonical and gives 0. Every encoding that is not a NaN gives -1. The
    /// operation reads no behavior and signals nothing.
    ///
    /// ```
    /// use floaty::D64Bid;
    ///
    /// let nan = D64Bid::from_bits(0x7C00_0000_0000_007B);
    /// assert_eq!(nan.payload().to_bits(), 0x31C0_0000_0000_007B); // 123
    /// ```
    #[must_use]
    pub fn payload(self) -> Self {
        let bits = DecimalLayout::<Enc, W>::get_payload(self.bits.to_limbs());
        Self::from_masked(LimbConversion::from_limbs(bits))
    }

    /// Returns the quiet NaN with the payload `payload`, as IEEE 754-2019
    /// `setPayload` does.
    ///
    /// The admissible payloads are +0 and the positive integers up to
    /// `10^(p - 1) - 1`, with any exponent. Every other value, -0 included,
    /// gives +0 with exponent 0. The NaN is positive and canonical. The
    /// operation reads no behavior and signals nothing.
    ///
    /// ```
    /// use floaty::D64Bid;
    ///
    /// let payload = D64Bid::from_bits(0x31C0_0000_0000_007B); // 123
    /// assert_eq!(D64Bid::from_payload(payload).to_bits(), 0x7C00_0000_0000_007B);
    /// ```
    #[must_use]
    pub fn from_payload(payload: Self) -> Self {
        let bits = DecimalLayout::<Enc, W>::set_payload(payload.bits.to_limbs(), false);
        Self::from_masked(LimbConversion::from_limbs(bits))
    }

    /// Returns the signaling NaN with the payload `payload`, as IEEE 754-2019
    /// `setPayloadSignaling` does. The admissible payloads are those of
    /// [`from_payload`](Self::from_payload), +0 included.
    #[must_use]
    pub fn from_payload_signaling(payload: Self) -> Self {
        let bits = DecimalLayout::<Enc, W>::set_payload(payload.bits.to_limbs(), true);
        Self::from_masked(LimbConversion::from_limbs(bits))
    }
}
