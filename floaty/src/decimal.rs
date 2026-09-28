//! The decimal formats: IEEE 754 decimal32, decimal64, and decimal128 in the
//! BID and DPD encodings.
//!
//! The codec works on a `u128`, which holds every coefficient and every
//! encoding of these widths.

use core::cmp::Ordering;
use core::marker::PhantomData;

mod arithmetic;
mod compare;
mod convert;
mod declet;
mod digits;
mod integral;
mod round;
mod scale;

/// The limbs of an exact decimal result: 512 bits hold 154 digits. The widest
/// intermediate is the radicand of a decimal128 square root, which has at
/// most `2p + 3` digits, 71. A sum keeps at most `2p + 2` digits.
type Wide = [u64; 8];

use self::digits::{digit_count_u128, power_of_ten_u128};
use self::round::DecimalTarget;
use crate::env::{Env, Flags, TotalOrder};
use crate::exact::Unrounded;
use crate::float::Class;
use crate::format::internal::{Host, LimbConversion, MinMax, Source, Step};
use crate::format::{Decimal, DecimalEncoding, DecimalKind, Standard, Storage, Width};
use crate::integer::{Integer, ToInt};
use crate::limbs::{self, Limbs};
use crate::unpacked::Unpacked;

/// The layout constants and codec of `Decimal<Enc>` at width `W`.
pub struct DecimalLayout<Enc, const W: usize> {
    encoding: PhantomData<Enc>,
}

/// Returns `value` as an `i32` in a constant.
const fn signed(value: u32) -> i32 {
    match 0_i32.checked_add_unsigned(value) {
        Some(result) => result,
        None => panic!("the value exceeds i32::MAX"),
    }
}

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// The width in bits.
    const WIDTH: u32 = <Width<W> as Storage>::WIDTH;

    /// Rejects a width that has no decimal format at compile time.
    const VALID: () = assert!(
        Self::WIDTH == 32 || Self::WIDTH == 64 || Self::WIDTH == 128,
        "a decimal format is 32, 64, or 128 bits wide"
    );

    /// The precision in digits, `9 * k / 32 - 2` (IEEE 754-2019 Table 3.6).
    const PRECISION: u32 = {
        let () = Self::VALID;
        9 * Self::WIDTH / 32 - 2
    };

    /// The largest adjusted exponent, `3 * 2^(k / 16 + 3)`.
    const EMAX: i32 = {
        let () = Self::VALID;
        3 << (Self::WIDTH / 16 + 3)
    };

    /// The smallest adjusted exponent of a normal value.
    const EMIN: i32 = 1 - Self::EMAX;

    /// The width of the exponent continuation field, `w = k / 16 + 4`.
    const CONTINUATION: u32 = Self::WIDTH / 16 + 4;

    /// The width of the trailing significand field, `t = 15 * k / 16 - 10`.
    const TRAILING: u32 = 15 * Self::WIDTH / 16 - 10;

    /// The number of declets in the trailing significand field.
    const DECLETS: u32 = Self::TRAILING / 10;

    /// The bias of the exponent field.
    const BIAS: i32 = Self::EMAX + signed(Self::PRECISION) - 2;

    /// The largest coefficient, `10^p - 1`.
    const LARGEST: u128 = power_of_ten_u128(Self::PRECISION) - 1;

    /// The largest NaN payload, `10^(p - 1) - 1`.
    const LARGEST_PAYLOAD: u128 = power_of_ten_u128(Self::PRECISION - 1) - 1;

    /// The bits of the largest coefficient.
    const SIGNIFICAND_BITS: u32 = 128 - Self::LARGEST.leading_zeros();

    /// The parameters that the rounding routine rounds to.
    const TARGET: DecimalTarget = DecimalTarget {
        precision: Self::PRECISION,
        emin: Self::EMIN,
        emax: Self::EMAX,
    };

    /// The bits of the combination field below its five leading bits, and
    /// the bit that marks a signaling NaN.
    const SIGNALING_BIT: u32 = Self::WIDTH - 7;

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
                payload: limbs::from_u128(if payload > Self::LARGEST_PAYLOAD {
                    0
                } else {
                    payload
                }),
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
            DecimalKind::Dpd => (0..Self::DECLETS).rev().fold(0, |value, index| {
                let declet = Self::field(trailing, 10 * index, 10);
                value * 1000
                    + u128::from(
                        declet::VALUES[usize::try_from(declet).expect("a declet fits a usize")],
                    )
            }),
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
        (field, digit * power_of_ten_u128(3 * Self::DECLETS) + rest)
    }

    /// Returns the trailing significand field of a value below
    /// `10^(3 * declets)`.
    fn trailing_field(value: u128) -> u128 {
        match Enc::KIND {
            DecimalKind::Bid => value,
            DecimalKind::Dpd => (0..Self::DECLETS).fold(0, |field, index| {
                let group = (value / power_of_ten_u128(3 * index)) % 1000;
                let declet = declet::DECLETS[usize::try_from(group).expect("a group fits a usize")];
                field | (u128::from(declet) << (10 * index))
            }),
        }
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
                let unit = power_of_ten_u128(3 * Self::DECLETS);
                let (digit, rest) = (coefficient / unit, coefficient % unit);
                let (high, low) = (field >> continuation, field & ((1 << continuation) - 1));
                let combination = if digit < 8 {
                    (high << (continuation + 3)) | (digit << continuation) | low
                } else {
                    (0b11 << (continuation + 3))
                        | (high << (continuation + 1))
                        | ((digit & 1) << continuation)
                        | low
                };
                sign | (combination << Self::TRAILING) | Self::trailing_field(rest)
            }
        }
    }

    /// Returns the class of an encoding.
    pub fn classify<L: Limbs>(bits: L) -> Class {
        match Self::decode(bits) {
            Unpacked::Zero { .. } => Class::Zero,
            Unpacked::Finite {
                exponent,
                significand,
                ..
            } => {
                let digits = digit_count_u128(limbs::to_u128(&significand));
                if exponent + signed(digits) - 1 < Self::EMIN {
                    Class::Subnormal
                } else {
                    Class::Normal
                }
            }
            Unpacked::Infinity { .. } => Class::Infinite,
            Unpacked::Nan {
                signaling: true, ..
            } => Class::SignalingNan,
            Unpacked::Nan { .. } => Class::QuietNan,
            Unpacked::Unsupported => Class::Unsupported,
        }
    }

    /// Returns `true` when the encoding is the canonical encoding of its
    /// value, cohort member, and payload.
    pub fn is_canonical<L: Limbs>(bits: L) -> bool {
        Self::encode(Self::decode(bits)) == bits
    }

    /// Returns the encoding with its sign set to `negative`.
    pub fn with_sign<L: Limbs>(bits: L, negative: bool) -> L {
        let magnitude = bits.low_bits(Self::WIDTH - 1);
        if negative {
            magnitude.with_bit(Self::WIDTH - 1)
        } else {
            magnitude
        }
    }

    /// Rounds an exact value to the format. The preferred exponent is the
    /// exponent of the value.
    pub fn round<In: Limbs, Out: Limbs>(value: &Unrounded<In>, env: &Env) -> (Out, Flags) {
        let (rounded, flags) = round::round::<In, Out>(value, value.exponent, &Self::TARGET, env);
        (Self::encode(rounded), flags)
    }
}

impl<Enc: DecimalEncoding, const W: usize> Standard<W> for Decimal<Enc>
where
    Width<W>: Storage,
{
    type Bits = <Width<W> as Storage>::Bits;

    const RADIX: u32 = 10;
    const PRECISION: u32 = DecimalLayout::<Enc, W>::PRECISION;
    const EMAX: i32 = DecimalLayout::<Enc, W>::EMAX;
    const EMIN: i32 = DecimalLayout::<Enc, W>::EMIN;
    const SIGNIFICAND_BITS: u32 = DecimalLayout::<Enc, W>::SIGNIFICAND_BITS;
    const PAYLOAD_DIGITS: u32 = DecimalLayout::<Enc, W>::PRECISION - 1;
    const HOST: Host = Host::None;

    fn mask(bits: Self::Bits) -> Self::Bits {
        let () = DecimalLayout::<Enc, W>::VALID;
        bits
    }

    fn unpack(bits: Self::Bits) -> Unpacked<<Self::Bits as LimbConversion>::Limbs> {
        DecimalLayout::<Enc, W>::decode(bits.to_limbs())
    }

    fn classify(bits: Self::Bits) -> Class {
        DecimalLayout::<Enc, W>::classify(bits.to_limbs())
    }

    fn is_canonical(bits: Self::Bits) -> bool {
        DecimalLayout::<Enc, W>::is_canonical(bits.to_limbs())
    }

    fn is_sign_negative(bits: Self::Bits) -> bool {
        bits.to_limbs().bit(DecimalLayout::<Enc, W>::WIDTH - 1)
    }

    fn round<L: Limbs>(value: &Unrounded<L>, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::round(value, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn convert_from<L: Limbs>(
        value: Unpacked<L>,
        source: Source,
        env: &Env,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::convert_from(value, source, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn add(left: Self::Bits, right: Self::Bits, subtract: bool, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::add(left.to_limbs(), right.to_limbs(), subtract, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn mul(left: Self::Bits, right: Self::Bits, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::mul(left.to_limbs(), right.to_limbs(), env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn div(left: Self::Bits, right: Self::Bits, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::div(left.to_limbs(), right.to_limbs(), env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn sqrt(value: Self::Bits, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::sqrt(value.to_limbs(), env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn mul_add(
        left: Self::Bits,
        right: Self::Bits,
        addend: Self::Bits,
        env: &Env,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::mul_add(
            left.to_limbs(),
            right.to_limbs(),
            addend.to_limbs(),
            env,
        );
        (Self::Bits::from_limbs(bits), flags)
    }

    fn with_sign(bits: Self::Bits, negative: bool) -> Self::Bits {
        Self::Bits::from_limbs(DecimalLayout::<Enc, W>::with_sign(
            bits.to_limbs(),
            negative,
        ))
    }

    fn compare(left: Self::Bits, right: Self::Bits, env: &Env) -> (Option<Ordering>, Flags) {
        DecimalLayout::<Enc, W>::compare(left.to_limbs(), right.to_limbs(), env)
    }

    fn total_cmp(left: Self::Bits, right: Self::Bits, order: TotalOrder) -> Ordering {
        DecimalLayout::<Enc, W>::total_cmp(left.to_limbs(), right.to_limbs(), order)
    }

    fn min_max(
        left: Self::Bits,
        right: Self::Bits,
        operation: MinMax,
        env: &Env,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::min_max(left.to_limbs(), right.to_limbs(), operation, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn round_to_integral(value: Self::Bits, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::round_to_integral(value.to_limbs(), env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn to_int<I: Integer>(value: Self::Bits, env: &Env) -> (ToInt<I>, Flags) {
        DecimalLayout::<Enc, W>::to_int(value.to_limbs(), env)
    }

    fn remainder(left: Self::Bits, right: Self::Bits, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::remainder(left.to_limbs(), right.to_limbs(), env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn scale_b(value: Self::Bits, scale: i32, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::scale_b(value.to_limbs(), scale, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn next(value: Self::Bits, step: Step, env: &Env) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::next(value.to_limbs(), step, env);
        (Self::Bits::from_limbs(bits), flags)
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
