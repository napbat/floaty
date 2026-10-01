//! The NaN payload operations of IEEE 754-2019 section 9.7 for the binary
//! formats. The payload of a NaN is its fraction below the quiet bit.

use super::Layout;
use crate::env::Env;
use crate::exact::{self, Unrounded};
use crate::format::{Encoding, Storage, Width};
use crate::limbs::Limbs;
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the payload of a NaN as an integral value, or -1 for every
    /// other encoding.
    pub fn get_payload<L: Limbs>(bits: L) -> L {
        let integer = match Self::decode(bits) {
            Unpacked::Nan { payload, .. } => Unrounded {
                negative: false,
                exponent: 0,
                significand: payload,
                sticky: false,
            },
            _ => Unrounded {
                negative: true,
                exponent: 0,
                significand: L::ZERO.with_bit(0),
                sticky: false,
            },
        };
        // A payload has at most p - 2 bits, so the rounding is exact.
        let (value, _) = exact::round::<L, L, Self, Env>(&integer, Env::IEEE);
        Self::encode(value)
    }

    /// Returns the quiet or the signaling NaN whose payload is the integral
    /// value of `bits`, or +0 when the value is no admissible payload.
    pub fn set_payload<L: Limbs>(bits: L, signaling: bool) -> L {
        let payload = match Self::decode(bits) {
            Unpacked::Zero {
                negative: false, ..
            } => Some(L::ZERO),
            Unpacked::Finite {
                negative: false,
                exponent,
                significand,
            } => Self::integer(exponent, significand),
            _ => None,
        };
        // A signaling NaN needs a nonzero payload, or it encodes an infinity.
        // A format without a signaling NaN has no payload bits, so it admits
        // no signaling payload.
        let admissible =
            payload.filter(|payload| Self::TARGET.has_nan && !(signaling && payload.is_zero()));
        match admissible {
            Some(payload) => Self::encode(Unpacked::Nan {
                negative: false,
                signaling,
                payload,
            }),
            None => Self::encode(Unpacked::<L>::zero(false)),
        }
    }

    /// Returns `significand * 2^exponent` when it is an integer of at most
    /// [`Self::PAYLOAD_BITS`] bits.
    fn integer<L: Limbs>(exponent: i32, significand: L) -> Option<L> {
        // A value whose lowest significand bit weighs 1 or more is at least
        // 2^(p - 1), above every payload.
        if exponent >= 0 {
            return None;
        }
        let width = significand.bit_length();
        let shift = exponent.unsigned_abs();
        let integral = shift < width && !significand.any_below(shift);
        (integral && width - shift <= Self::PAYLOAD_BITS).then(|| significand.shr(shift))
    }
}

#[cfg(test)]
mod tests {
    use crate::float::{F4E2M1Fn, F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F64, F80};

    const MINUS_ONE: u64 = 0xBFF0_0000_0000_0000;

    fn from_payload(bits: u64) -> (u64, u64) {
        let payload = F64::from_bits(bits);
        (
            F64::from_payload(payload).to_bits(),
            F64::from_payload_signaling(payload).to_bits(),
        )
    }

    #[test]
    fn binary64_payloads_have_51_bits() {
        let largest = 0x431F_FFFF_FFFF_FFFC; // 2^51 - 1
        assert_eq!(
            from_payload(largest),
            (0x7FFF_FFFF_FFFF_FFFF, 0x7FF7_FFFF_FFFF_FFFF)
        );
        assert_eq!(from_payload(0x4320_0000_0000_0000), (0, 0)); // 2^51
        assert_eq!(from_payload(0), (0x7FF8_0000_0000_0000, 0));
        assert_eq!(from_payload(1 << 63), (0, 0)); // -0
        assert_eq!(from_payload(0x3FF8_0000_0000_0000), (0, 0)); // 1.5
        assert_eq!(from_payload(1), (0, 0)); // the smallest subnormal value
        assert_eq!(from_payload(0x7FF0_0000_0000_0000), (0, 0));
        assert_eq!(from_payload(0x7FF8_0000_0000_0001), (0, 0));
        let nan = F64::from_bits(0xFFF7_FFFF_FFFF_FFFF).payload();
        assert_eq!(nan.to_bits(), largest);
        assert_eq!(
            F64::from_bits(0x7FF0_0000_0000_0000).payload().to_bits(),
            MINUS_ONE
        );
    }

    #[test]
    fn small_formats_admit_the_payloads_of_their_nans() {
        let one = F8E5M2::from_bits(0x3C);
        assert_eq!(F8E5M2::from_payload(one).to_bits(), 0x7F);
        assert_eq!(F8E5M2::from_payload_signaling(one).to_bits(), 0x7D);
        assert_eq!(F8E5M2::from_bits(0x7D).payload().to_bits(), 0x3C);
        // The NaN of E4M3FN and E4M3FNUZ has no payload and is quiet.
        let zero = F8E4M3Fn::from_bits(0);
        assert_eq!(F8E4M3Fn::from_payload(zero).to_bits(), 0x7F);
        assert_eq!(F8E4M3Fn::from_payload_signaling(zero).to_bits(), 0);
        assert_eq!(
            F8E4M3Fn::from_payload(F8E4M3Fn::from_bits(0x38)).to_bits(),
            0
        );
        assert_eq!(F8E4M3Fn::from_bits(0xFF).payload().to_bits(), 0);
        let fnuz = F8E4M3Fnuz::from_payload(F8E4M3Fnuz::from_bits(0));
        assert_eq!(fnuz.to_bits(), 0x80);
        // FP4 E2M1 has no NaN.
        assert_eq!(F4E2M1Fn::from_payload(F4E2M1Fn::from_bits(0)).to_bits(), 0);
        assert_eq!(F4E2M1Fn::from_bits(0).payload().to_bits(), 0xA);
    }

    #[test]
    fn x87_payloads_have_62_bits_and_unsupported_encodings_are_no_nan() {
        let nan = F80::from_bits(0x7FFF_C000_0000_0000_0007);
        let payload = nan.payload();
        assert_eq!(F80::from_payload(payload).to_bits(), nan.to_bits());
        assert_eq!(
            F80::from_payload_signaling(payload).to_bits(),
            0x7FFF_8000_0000_0000_0007
        );
        // A pseudo-NaN, without the integer bit, is unsupported.
        let pseudo = F80::from_bits(0x7FFF_4000_0000_0000_0007);
        assert_eq!(pseudo.payload().to_bits(), 0xBFFF_8000_0000_0000_0000);
    }
}
