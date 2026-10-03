//! The operations of `Lanes` of the decimal formats alone: the quantum
//! operations, `log_b`, and the NaN payload operations. Each lane takes the
//! scalar operation.

use super::Lanes;
use crate::env::{Flags, Mode, Override};
use crate::float::Float;
use crate::format::{Decimal, DecimalEncoding, Standard, Storage, Width};

/// A float of a decimal format.
type DecimalFloat<Enc, const W: usize, M> = Float<Decimal<Enc>, W, M>;

impl<Enc: DecimalEncoding, const W: usize, M: Mode, const N: usize>
    Lanes<DecimalFloat<Enc, W, M>, N>
where
    Width<W>: Storage,
    Decimal<Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    /// Returns each lane with the exponent of the lane of `exponent_of`, as
    /// [`Float::quantize`] does.
    #[must_use]
    pub fn quantize(self, exponent_of: Self) -> Self {
        self.zip(exponent_of, Float::quantize)
    }

    /// Returns each lane with the exponent of the lane of `exponent_of`, as
    /// [`Float::quantize_with`] does, and the union of the flags.
    #[must_use]
    pub fn quantize_with(self, exponent_of: Self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.zip_with(exponent_of, |x, exponent_of| {
            x.quantize_with(exponent_of, behavior)
        })
    }

    /// Returns `true` in each lane whose exponent is that of the lane of
    /// `other`, as [`Float::same_quantum`] does.
    #[must_use]
    pub fn same_quantum(self, other: Self) -> [bool; N] {
        self.pairs(other, Float::same_quantum)
    }

    /// Returns the quantum of each lane, as [`Float::quantum`] does.
    #[must_use]
    pub fn quantum(self) -> Self {
        self.map(Float::quantum)
    }

    /// Returns the quantum of each lane, as [`Float::quantum_with`] does, and
    /// the union of the flags.
    #[must_use]
    pub fn quantum_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.quantum_with(behavior))
    }

    /// Returns the exponent of the leading digit of each lane, as the
    /// `log_b` of a decimal [`Float`] does.
    #[must_use]
    pub fn log_b(self) -> Self {
        self.map(<DecimalFloat<Enc, W, M>>::log_b)
    }

    /// Returns the exponent of the leading digit of each lane, as the
    /// `log_b_with` of a decimal [`Float`] does, and the union of the flags.
    #[must_use]
    pub fn log_b_with(self, behavior: impl Override) -> (Self, Flags) {
        let behavior = behavior.apply::<M>();
        self.map_with(|lane| lane.log_b_with(behavior))
    }

    /// Returns the payload of each lane, as the `payload` of a decimal
    /// [`Float`] does.
    #[must_use]
    pub fn payload(self) -> Self {
        self.map(<DecimalFloat<Enc, W, M>>::payload)
    }

    /// Returns the quiet NaN with the payload of each lane, as the
    /// `from_payload` of a decimal [`Float`] does.
    #[must_use]
    pub fn from_payload(payload: Self) -> Self {
        payload.map(<DecimalFloat<Enc, W, M>>::from_payload)
    }

    /// Returns the signaling NaN with the payload of each lane, as the
    /// `from_payload_signaling` of a decimal [`Float`] does.
    #[must_use]
    pub fn from_payload_signaling(payload: Self) -> Self {
        payload.map(<DecimalFloat<Enc, W, M>>::from_payload_signaling)
    }
}
