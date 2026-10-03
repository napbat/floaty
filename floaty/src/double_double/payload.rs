//! The NaN payload operations of IEEE 754-2019, section 9.7.

use crate::env::{Env, Flags, Mode};
use crate::float::F64;

use super::{Algorithm, DoubleDouble};

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// Returns the payload of a NaN, as IEEE 754-2019 `getPayload` does.
    ///
    /// The payload is the 51 fraction bits below the binary64 quiet bit,
    /// as an integer. The first special half defines the value: high before
    /// low. A value that is not a NaN gives -1. The low result half is +0.
    /// The operation reads no behavior and signals nothing.
    ///
    /// ```
    /// use floaty::{DoubleDouble, F64, Qd};
    ///
    /// let nan = DoubleDouble::<Qd>::from_f64(F64::from_bits(0xFFF8_0000_0000_0005));
    /// assert_eq!(nan.payload().hi().to_bits(), 0x4014_0000_0000_0000); // 5
    /// assert_eq!(nan.payload().lo().to_bits(), 0);
    /// ```
    #[must_use]
    pub fn payload(self) -> Self {
        let payload = self
            .special_half()
            .map_or(F64::from_bits(0xBFF0_0000_0000_0000), F64::payload);
        Self::from_f64(payload)
    }

    /// Returns the positive quiet NaN with the payload `payload`, as IEEE
    /// 754-2019 `setPayload` does.
    ///
    /// The admissible exact values are +0 and the positive integers below
    /// `2^51`. Every other value, -0 included, gives +0. The high half holds
    /// the NaN, and the low half is +0. The operation reads no behavior and
    /// signals nothing.
    ///
    /// ```
    /// use floaty::{DoubleDouble, Gcc};
    ///
    /// let five = DoubleDouble::<Gcc>::from_int(5);
    /// let nan = DoubleDouble::from_payload(five);
    /// assert_eq!(nan.hi().to_bits(), 0x7FF8_0000_0000_0005);
    /// assert_eq!(nan.lo().to_bits(), 0);
    /// ```
    #[must_use]
    pub fn from_payload(payload: Self) -> Self {
        Self::from_f64(
            payload
                .payload_operand()
                .map_or(F64::from_bits(0), F64::from_payload),
        )
    }

    /// Returns the positive signaling NaN with the payload `payload`, as
    /// IEEE 754-2019 `setPayloadSignaling` does.
    ///
    /// The admissible exact values are the positive integers below `2^51`.
    /// Zero is not admissible, because its encoding is an infinity. Every
    /// other value gives +0. The high half holds the NaN, and the low half
    /// is +0. The operation reads no behavior and signals nothing.
    #[must_use]
    pub fn from_payload_signaling(payload: Self) -> Self {
        Self::from_f64(
            payload
                .payload_operand()
                .map_or(F64::from_bits(0), F64::from_payload_signaling),
        )
    }

    /// Every admissible payload fits binary64 exactly. An inexact conversion
    /// cannot hide a fraction in the low half and make that fraction admissible.
    fn payload_operand(self) -> Option<F64> {
        if self.lo.is_zero() {
            return Some(self.hi);
        }
        let (value, flags) = self.convert_with::<F64>(Env::IEEE);
        (!flags.contains(Flags::INEXACT)).then_some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Gcc, Qd};

    fn bits<Alg: Algorithm>(value: DoubleDouble<Alg>) -> (u64, u64) {
        (value.hi().to_bits(), value.lo().to_bits())
    }

    fn constructors<Alg: Algorithm>() {
        let pair = |hi, lo| DoubleDouble::<Alg>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        for (hi, lo, quiet, signaling) in [
            (0, 0, 0x7FF8_0000_0000_0000, 0),
            (0x8000_0000_0000_0000, 0, 0, 0),
            (
                0x4000_0000_0000_0000,
                0xBFF0_0000_0000_0000,
                0x7FF8_0000_0000_0001,
                0x7FF0_0000_0000_0001,
            ),
            (
                0x4320_0000_0000_0000,
                0xBFF0_0000_0000_0000,
                0x7FFF_FFFF_FFFF_FFFF,
                0x7FF7_FFFF_FFFF_FFFF,
            ),
            (0x4320_0000_0000_0000, 0, 0, 0),
            (0x3FF0_0000_0000_0000, 1, 0, 0),
            (0x3FF0_0000_0000_0000, 0x8000_0000_0000_0001, 0, 0),
            (0, 1, 0, 0),
            (
                0x3FF0_0000_0000_0000,
                0xBFF0_0000_0000_0000,
                0x7FF8_0000_0000_0000,
                0,
            ),
            (0xBFF0_0000_0000_0000, 0x3FF0_0000_0000_0000, 0, 0),
            (
                0x3FE0_0000_0000_0000,
                0x3FE0_0000_0000_0000,
                0x7FF8_0000_0000_0001,
                0x7FF0_0000_0000_0001,
            ),
            (0x7FF0_0000_0000_0001, 0, 0, 0),
            (0, 0x7FF8_0000_0000_0001, 0, 0),
        ] {
            let payload = pair(hi, lo);
            assert_eq!(
                bits(DoubleDouble::from_payload(payload)),
                (quiet, 0),
                "{payload:?}"
            );
            assert_eq!(
                bits(DoubleDouble::from_payload_signaling(payload)),
                (signaling, 0),
                "{payload:?}"
            );
        }
    }

    #[test]
    fn constructors_read_the_exact_value() {
        constructors::<Gcc>();
        constructors::<Qd>();
    }

    #[test]
    fn getter_uses_the_first_special_half() {
        let pair = |hi, lo| DoubleDouble::<Gcc>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        for (hi, lo, expected) in [
            (0xFFF8_0000_0000_0005, 0, 0x4014_0000_0000_0000),
            (0, 0x7FF0_0000_0000_0003, 0x4008_0000_0000_0000),
            (
                0x7FF0_0000_0000_0000,
                0x7FF8_0000_0000_0005,
                0xBFF0_0000_0000_0000,
            ),
            (0x7FF8_0000_0000_0000, 0x7FF8_0000_0000_0005, 0),
            (0x3FF0_0000_0000_0000, 0, 0xBFF0_0000_0000_0000),
        ] {
            assert_eq!(bits(pair(hi, lo).payload()), (expected, 0));
        }
    }
}
