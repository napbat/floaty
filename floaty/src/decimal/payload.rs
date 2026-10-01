//! The NaN payload operations of IEEE 754-2019 section 9.7 for the decimal
//! formats. The payload of a NaN is the value of its trailing significand
//! field.

use super::DecimalLayout;
use super::digits::power_of_ten_u128;
use crate::format::{DecimalEncoding, Storage, Width};
use crate::limbs::{self, Limbs};
use crate::unpacked::Unpacked;

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the payload of a NaN as an integral value with exponent 0, or
    /// -1 for every other encoding.
    pub fn get_payload<L: Limbs>(bits: L) -> L {
        let value = match Self::decode(bits) {
            Unpacked::Nan { payload, .. } if payload.is_zero() => Unpacked::Zero {
                negative: false,
                exponent: 0,
            },
            Unpacked::Nan { payload, .. } => Unpacked::Finite {
                negative: false,
                exponent: 0,
                significand: payload,
            },
            _ => Unpacked::Finite {
                negative: true,
                exponent: 0,
                significand: L::ZERO.with_bit(0),
            },
        };
        Self::encode(value)
    }

    /// Returns the quiet or the signaling NaN whose payload is the integral
    /// value of `bits`, or +0 with exponent 0 when the value is no admissible
    /// payload.
    pub fn set_payload<L: Limbs>(bits: L, signaling: bool) -> L {
        let payload = match Self::decode(bits) {
            Unpacked::Zero {
                negative: false, ..
            } => Some(0),
            Unpacked::Finite {
                negative: false,
                exponent,
                significand,
            } => integer(exponent, limbs::to_u128(&significand)),
            _ => None,
        };
        match payload.filter(|&payload| payload <= Self::LARGEST_PAYLOAD) {
            Some(payload) => Self::encode(Unpacked::Nan {
                negative: false,
                signaling,
                payload: limbs::from_u128(payload),
            }),
            None => Self::encode(Unpacked::<L>::Zero {
                negative: false,
                exponent: 0,
            }),
        }
    }
}

/// Returns `coefficient * 10^exponent` when it is an integer that fits a
/// `u128`.
fn integer(exponent: i32, coefficient: u128) -> Option<u128> {
    let scale = exponent.unsigned_abs();
    if scale > 38 {
        return None;
    }
    let power = power_of_ten_u128(scale);
    if exponent >= 0 {
        coefficient.checked_mul(power)
    } else {
        coefficient
            .is_multiple_of(power)
            .then(|| coefficient / power)
    }
}

#[cfg(test)]
mod tests {
    use crate::float::{D32Bid, D64Bid, D128Dpd};

    #[test]
    fn decimal64_payloads_have_15_digits() {
        let from_payload = |bits: u64| {
            let payload = D64Bid::from_bits(bits);
            (
                D64Bid::from_payload(payload).to_bits(),
                D64Bid::from_payload_signaling(payload).to_bits(),
            )
        };
        // 10^15 - 1 with exponent 0, and 1.2E+2 = 120 with exponent 1.
        let largest = 0x31C3_8D7E_A4C6_7FFF;
        assert_eq!(
            from_payload(largest),
            (0x7C03_8D7E_A4C6_7FFF, 0x7E03_8D7E_A4C6_7FFF)
        );
        assert_eq!(
            from_payload(0x31E0_0000_0000_000C),
            (0x7C00_0000_0000_0078, 0x7E00_0000_0000_0078)
        );
        assert_eq!(
            from_payload(largest + 1),
            (0x31C0_0000_0000_0000, 0x31C0_0000_0000_0000)
        );
        // Zero is admissible for both NaNs, and 1.5 for neither.
        assert_eq!(
            from_payload(0x0000_0000_0000_0000),
            (0x7C00_0000_0000_0000, 0x7E00_0000_0000_0000)
        );
        assert_eq!(
            from_payload(0x31A0_0000_0000_000F),
            (0x31C0_0000_0000_0000, 0x31C0_0000_0000_0000)
        );
        assert_eq!(
            from_payload(0xB1C0_0000_0000_0000),
            (0x31C0_0000_0000_0000, 0x31C0_0000_0000_0000)
        );
    }

    #[test]
    fn non_canonical_payloads_read_as_zero_and_numbers_as_minus_one() {
        // 10^6 in the trailing field of a decimal32 NaN is above 10^6 - 1.
        let nan = D32Bid::from_bits(0x7C0F_4240);
        assert_eq!(nan.payload().to_bits(), 0x3280_0000);
        // An infinity is no NaN, so its payload is -1.
        let infinity = D128Dpd::from_bits(0x7800_0000_0000_0000_0000_0000_0000_0000);
        assert_eq!(
            infinity.payload().to_bits(),
            0xA208_0000_0000_0000_0000_0000_0000_0001
        );
    }
}
