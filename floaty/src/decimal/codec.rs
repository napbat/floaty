//! The BID and DPD codec of the decimal formats.
//!
//! The codec works on a `u128`, which holds every coefficient and every
//! encoding of these widths.

use super::digits::{power_divisor, power_of_ten_u128};
use super::{DecimalLayout, declet, signed};
use crate::format::{DecimalEncoding, DecimalKind, Storage, Width};
use crate::limbs::{self, Limbs};
use crate::unpacked::Unpacked;

/// The weight of the second group of six declets, 10^18.
const GROUP_WEIGHT: u128 = power_of_ten_u128(18);

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the `width` bits of `bits` that start at bit `low`.
    const fn field(bits: u128, low: u32, width: u32) -> u128 {
        (bits >> low) & ((1 << width) - 1)
    }

    /// Decodes an encoding.
    pub fn decode<L: Limbs>(bits: L) -> Unpacked<L> {
        let () = Self::VALID;
        let bits = limbs::to_u128(&bits);
        let negative = (bits >> (Self::WIDTH - 1)) & 1 == 1;
        let leading = Self::field(bits, Self::WIDTH - 6, 5);
        let trailing = Self::field(bits, 0, Self::TRAILING);
        if leading == 0b11111 {
            let payload = Self::trailing_value(trailing);
            return Unpacked::Nan {
                negative,
                signaling: (bits >> Self::SIGNALING_BIT) & 1 == 1,
                payload: limbs::from_u128(Self::canonical_payload(payload)),
            };
        }
        if leading == 0b11110 {
            return Unpacked::Infinity { negative };
        }
        let (field, coefficient) = match Enc::KIND {
            DecimalKind::Bid => Self::bid_fields(bits),
            DecimalKind::Dpd => Self::dpd_fields(bits),
        };
        let exponent =
            signed(u32::try_from(field).expect("an exponent field fits a u32")) - Self::BIAS;
        if coefficient == 0 || coefficient > Self::LARGEST {
            // A BID coefficient above 10^p - 1 is not canonical and reads as
            // zero, as IEEE 754-2019 section 3.5.2 requires.
            return Unpacked::Zero { negative, exponent };
        }
        Unpacked::Finite {
            negative,
            exponent,
            significand: limbs::from_u128(coefficient),
        }
    }

    /// Returns the value of a trailing significand field: a binary integer
    /// for BID, and digits in declets for DPD.
    fn trailing_value(trailing: u128) -> u128 {
        match Enc::KIND {
            DecimalKind::Bid => trailing,
            DecimalKind::Dpd => {
                // Six declets hold 18 digits, which fit a u64. One 128-bit
                // multiplication joins the two groups of declets.
                let group = |declets: core::ops::Range<u32>| {
                    declets.rev().fold(0, |value, index| {
                        let declet = Self::field(trailing, 10 * index, 10);
                        let digits =
                            declet::VALUES[usize::try_from(declet).expect("a declet fits a usize")];
                        value * 1000 + u64::from(digits)
                    })
                };
                let split = Self::DECLETS.min(6);
                u128::from(group(split..Self::DECLETS)) * GROUP_WEIGHT + u128::from(group(0..split))
            }
        }
    }

    /// Returns the exponent field and the coefficient of a BID number.
    fn bid_fields(bits: u128) -> (u128, u128) {
        let exponent_bits = Self::CONTINUATION + 2;
        if Self::field(bits, Self::WIDTH - 3, 2) == 0b11 {
            // The coefficient has the implicit prefix 100.
            let field = Self::field(bits, Self::TRAILING + 1, exponent_bits);
            let coefficient =
                (1 << (Self::TRAILING + 3)) | Self::field(bits, 0, Self::TRAILING + 1);
            (field, coefficient)
        } else {
            let field = Self::field(bits, Self::TRAILING + 3, exponent_bits);
            (field, Self::field(bits, 0, Self::TRAILING + 3))
        }
    }

    /// Returns the exponent field and the coefficient of a DPD number.
    fn dpd_fields(bits: u128) -> (u128, u128) {
        let combination = Self::field(bits, Self::TRAILING, Self::CONTINUATION + 5);
        let continuation = Self::field(combination, 0, Self::CONTINUATION);
        let (high, digit) = if Self::field(combination, Self::CONTINUATION + 3, 2) == 0b11 {
            (
                Self::field(combination, Self::CONTINUATION + 1, 2),
                8 + Self::field(combination, Self::CONTINUATION, 1),
            )
        } else {
            (
                Self::field(combination, Self::CONTINUATION + 3, 2),
                Self::field(combination, Self::CONTINUATION, 3),
            )
        };
        let field = (high << Self::CONTINUATION) | continuation;
        let rest = Self::trailing_value(Self::field(bits, 0, Self::TRAILING));
        (field, digit * Self::DECLET_WEIGHT + rest)
    }

    /// Returns the trailing significand field of a value below
    /// `10^(3 * declets)`.
    fn trailing_field(value: u128) -> u128 {
        match Enc::KIND {
            DecimalKind::Bid => value,
            DecimalKind::Dpd => Self::declets(value).1,
        }
    }

    /// Returns the digit of `value` above its declets, and the declets as a
    /// trailing significand field. `value` is below `10^(3 * declets + 1)`.
    fn declets(value: u128) -> (u128, u128) {
        // A u64 holds six groups of three digits, and divides by 1000 with a
        // multiplication. A value of at least 10^18 takes one division by
        // 10^18, which splits it into two such parts.
        let (high, low) = match u64::try_from(value) {
            Ok(low) if value < GROUP_WEIGHT => (0, low),
            _ => {
                let ([high, above], low) =
                    limbs::divide_small(limbs::split_u128(value), power_divisor(18));
                debug_assert!(above == 0, "the value has at most 34 digits");
                (high, low)
            }
        };
        let mut groups = [low, high].into_iter().flat_map(|part| {
            (0..6).scan(part, |rest, _| {
                let group = *rest % 1000;
                *rest /= 1000;
                Some(group)
            })
        });
        let field = (0..Self::DECLETS)
            .zip(&mut groups)
            .fold(0, |field, (index, group)| {
                let declet = declet::DECLETS[usize::try_from(group).expect("a group fits a usize")];
                field | (u128::from(declet) << (10 * index))
            });
        let digit = groups
            .next()
            .expect("two parts of six groups hold every digit");
        (u128::from(digit), field)
    }

    /// Encodes a value in its canonical encoding. The value must be
    /// representable in the format.
    pub fn encode<L: Limbs>(value: Unpacked<L>) -> L {
        let () = Self::VALID;
        let sign = |negative: bool| u128::from(negative) << (Self::WIDTH - 1);
        let special =
            |negative: bool, leading: u128| sign(negative) | (leading << (Self::WIDTH - 6));
        let bits = match value {
            Unpacked::Zero { negative, exponent } => Self::encode_number(negative, exponent, 0),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => Self::encode_number(negative, exponent, limbs::to_u128(&significand)),
            Unpacked::Infinity { negative } => special(negative, 0b11110),
            Unpacked::Nan {
                negative,
                signaling,
                payload,
            } => {
                let payload = limbs::to_u128(&payload);
                debug_assert!(payload <= Self::LARGEST_PAYLOAD, "the payload fits");
                special(negative, 0b11111)
                    | (u128::from(signaling) << Self::SIGNALING_BIT)
                    | Self::trailing_field(payload)
            }
            Unpacked::Unsupported => unreachable!("a decimal format has no unsupported encoding"),
        };
        limbs::from_u128(bits)
    }

    /// Encodes a zero or a finite number.
    fn encode_number(negative: bool, exponent: i32, coefficient: u128) -> u128 {
        debug_assert!(coefficient <= Self::LARGEST, "the coefficient fits");
        let field =
            u128::from(u32::try_from(exponent + Self::BIAS).expect("the exponent is in range"));
        let sign = u128::from(negative) << (Self::WIDTH - 1);
        let continuation = Self::CONTINUATION;
        match Enc::KIND {
            DecimalKind::Bid => {
                if coefficient >> (Self::TRAILING + 3) == 0 {
                    sign | (field << (Self::TRAILING + 3)) | coefficient
                } else {
                    let low = coefficient & ((1 << (Self::TRAILING + 1)) - 1);
                    sign | (0b11 << (Self::WIDTH - 3)) | (field << (Self::TRAILING + 1)) | low
                }
            }
            DecimalKind::Dpd => {
                let (digit, trailing) = Self::declets(coefficient);
                let (high, low) = (field >> continuation, field & ((1 << continuation) - 1));
                let combination = if digit < 8 {
                    (high << (continuation + 3)) | (digit << continuation) | low
                } else {
                    (0b11 << (continuation + 3))
                        | (high << (continuation + 1))
                        | ((digit & 1) << continuation)
                        | low
                };
                sign | (combination << Self::TRAILING) | trailing
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::float::{Class, D32Bid, D32Dpd, D64Bid, D64Dpd, D128Bid, D128Dpd, Decoded};

    #[test]
    fn known_encodings_decode_and_encode() {
        // 1 in each format and encoding: coefficient 1, exponent 0.
        let one = Decoded::<2>::Finite {
            negative: false,
            exponent: 0,
            significand: [1, 0],
        };
        assert_eq!(D32Bid::from_bits(0x3280_0001).decode::<2>(), one);
        assert_eq!(D32Dpd::from_bits(0x2250_0001).decode::<2>(), one);
        assert_eq!(D64Bid::from_bits(0x31C0_0000_0000_0001).decode::<2>(), one);
        assert_eq!(D64Dpd::from_bits(0x2238_0000_0000_0001).decode::<2>(), one);
        assert_eq!(
            D128Bid::from_bits(0x3040_0000_0000_0000_0000_0000_0000_0001).decode::<2>(),
            one
        );
        assert_eq!(
            D128Dpd::from_bits(0x2208_0000_0000_0000_0000_0000_0000_0001).decode::<2>(),
            one
        );
        // The largest decimal64 value, 9999999999999999E+369, in both
        // encodings, and its round trip through the encoder.
        let largest = Decoded::<1>::Finite {
            negative: false,
            exponent: 369,
            significand: [9_999_999_999_999_999],
        };
        assert_eq!(
            D64Bid::from_bits(0x77FB_86F2_6FC0_FFFF).decode::<1>(),
            largest
        );
        assert_eq!(
            D64Dpd::from_bits(0x77FC_FF3F_CFF3_FCFF).decode::<1>(),
            largest
        );
        for bits in [
            0x77FB_86F2_6FC0_FFFF_u64,
            0x31C0_0000_0000_0001,
            0x7800_0000_0000_0000,
            0x7C00_0000_0000_0000,
        ] {
            assert!(D64Bid::from_bits(bits).is_canonical(), "{bits:#x}");
        }
        assert!(D64Dpd::from_bits(0x77FC_FF3F_CFF3_FCFF).is_canonical());
    }

    #[test]
    fn non_canonical_encodings_read_as_their_canonical_values() {
        // A BID coefficient above 10^16 - 1 reads as zero with its exponent.
        let wide = D64Bid::from_bits(0x6CB8_8000_0000_0000 | 0x0003_86F2_6FC1_0000);
        assert_eq!(wide.classify(), Class::Zero);
        assert!(!wide.is_canonical());
        // The non-canonical declet 0x3FF reads as 999.
        let declet = D64Dpd::from_bits(0x2238_0000_0000_03FF);
        assert_eq!(
            declet.decode::<1>(),
            Decoded::Finite {
                negative: false,
                exponent: 0,
                significand: [999]
            }
        );
        assert!(!declet.is_canonical());
        // Subnormal: the smallest value 1E-398 is below 10^-383.
        assert_eq!(D64Bid::from_bits(1).classify(), Class::Subnormal);
        assert_eq!(
            D64Bid::from_bits(0x7C00_0000_0000_0000).classify(),
            Class::QuietNan
        );
        assert_eq!(
            D64Bid::from_bits(0x7E00_0000_0000_0000).classify(),
            Class::SignalingNan
        );
    }
}
