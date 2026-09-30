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

/// The limbs of an exact decimal result: twice the storage limbs `L`. An
/// addition is the exception: its sums stay in the storage limbs.
///
/// The widest intermediate is the radicand of a square root, which has at
/// most `2p + 3` digits. decimal32 and decimal64 compute on 128 bits, which
/// hold 38 digits. decimal128 computes on 256 bits, which hold 77 digits. A
/// decimal128 radicand has at most 71 digits.
type Wide<L> = <L as Widen>::Double;

use self::digits::{power_divisor, power_of_ten_u128};
use self::round::DecimalTarget;
use crate::env::{Behavior, Env, Flags, TotalOrder};
use crate::exact::Unrounded;
use crate::float::Class;
use crate::format::internal::{LimbConversion, MinMax, Quotient, Source, Step};
use crate::format::{Decimal, DecimalEncoding, DecimalKind, Standard, Storage, Width};
use crate::host::Host;
use crate::integer::{Integer, ToInt};
use crate::limbs::{self, Limbs, Widen};
use crate::unpacked::Unpacked;

/// The weight of the second group of six declets, 10^18.
const GROUP_WEIGHT: u128 = power_of_ten_u128(18);

/// The layout constants and codec of `Decimal<Enc>` at width `W`.
pub struct DecimalLayout<Enc, const W: usize> {
    encoding: PhantomData<Enc>,
}

/// Returns `value` as an `i32` in a constant.
#[inline]
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

    /// The weight of the digit above the declets, `10^(3 * declets)`.
    const DECLET_WEIGHT: u128 = power_of_ten_u128(3 * Self::DECLETS);

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

    /// Returns `true` when a nonzero `coefficient * 10^exponent` is below
    /// `10^emin`.
    ///
    /// At an exponent of at least emin, the leading digit is at emin or
    /// above. Below emin, the leading digit is below emin when the
    /// coefficient has at most `emin - exponent` digits. That count is below
    /// the precision, so the power of 10 fits.
    fn is_subnormal(exponent: i32, coefficient: u128) -> bool {
        exponent < Self::EMIN && coefficient < power_of_ten_u128(Self::EMIN.abs_diff(exponent))
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
                if Self::is_subnormal(exponent, limbs::to_u128(&significand)) {
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
    #[inline]
    pub fn round<In: Limbs, Out: Limbs, B: Behavior>(
        value: &Unrounded<In>,
        behavior: B,
    ) -> (Out, Flags) {
        let (rounded, flags) = round::round::<In, Out, Self, B>(value, value.exponent, behavior);
        (Self::encode(rounded), flags)
    }
}

impl<Enc: DecimalEncoding, const W: usize> round::DecimalRoundingTarget for DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    const TARGET: DecimalTarget = DecimalLayout::<Enc, W>::TARGET;
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
    // The trailing significand field holds the payload.
    const SOURCE: Source = Source {
        radix: 10,
        payload_digits: DecimalLayout::<Enc, W>::PRECISION - 1,
        payload_bits: DecimalLayout::<Enc, W>::TRAILING,
    };
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

    // `Float::convert_with` reads its operand here. Out of line, the decoded
    // operand went through memory and blocked store forwarding: a decimal32
    // to decimal64 `convert` took 29.7 ns instead of 22.8 ns.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn operand(
        bits: Self::Bits,
        env: &Env,
    ) -> (Unpacked<<Self::Bits as LimbConversion>::Limbs>, Flags) {
        let mut flags = Flags::NONE;
        let value = DecimalLayout::<Enc, W>::operand(bits.to_limbs(), env, &mut flags);
        (value, flags)
    }

    fn is_canonical(bits: Self::Bits) -> bool {
        DecimalLayout::<Enc, W>::is_canonical(bits.to_limbs())
    }

    fn is_sign_negative(bits: Self::Bits) -> bool {
        bits.to_limbs().bit(DecimalLayout::<Enc, W>::WIDTH - 1)
    }

    fn round<L: Limbs, B: Behavior>(value: &Unrounded<L>, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::round(value, behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn convert_from<L: Limbs, B: Behavior>(
        value: Unpacked<L>,
        source: Source,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) = DecimalLayout::<Enc, W>::convert_from(value, source, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn add<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        subtract: bool,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::add(left.to_limbs(), right.to_limbs(), subtract, behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn mul<B: Behavior>(left: Self::Bits, right: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::mul(left.to_limbs(), right.to_limbs(), behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn div<B: Behavior>(left: Self::Bits, right: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::div(left.to_limbs(), right.to_limbs(), behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn sqrt<B: Behavior>(value: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::sqrt(value.to_limbs(), behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn mul_add<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        addend: Self::Bits,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) = DecimalLayout::<Enc, W>::mul_add(
            left.to_limbs(),
            right.to_limbs(),
            addend.to_limbs(),
            behavior,
        );
        (Self::Bits::from_limbs(bits), flags)
    }

    fn with_sign(bits: Self::Bits, negative: bool) -> Self::Bits {
        Self::Bits::from_limbs(DecimalLayout::<Enc, W>::with_sign(
            bits.to_limbs(),
            negative,
        ))
    }

    fn compare<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        behavior: B,
    ) -> (Option<Ordering>, Flags) {
        let env = &behavior.env();
        DecimalLayout::<Enc, W>::compare(left.to_limbs(), right.to_limbs(), env)
    }

    fn total_cmp(left: Self::Bits, right: Self::Bits, order: TotalOrder) -> Ordering {
        DecimalLayout::<Enc, W>::total_cmp(left.to_limbs(), right.to_limbs(), order)
    }

    fn min_max<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        operation: MinMax,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) =
            DecimalLayout::<Enc, W>::min_max(left.to_limbs(), right.to_limbs(), operation, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn round_to_integral<B: Behavior>(value: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) = DecimalLayout::<Enc, W>::round_to_integral(value.to_limbs(), env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn to_int<I: Integer, B: Behavior>(value: Self::Bits, behavior: B) -> (ToInt<I>, Flags) {
        let env = &behavior.env();
        DecimalLayout::<Enc, W>::to_int(value.to_limbs(), env)
    }

    fn remainder<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        quotient: Quotient,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) =
            DecimalLayout::<Enc, W>::remainder(left.to_limbs(), right.to_limbs(), quotient, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn scale_b<B: Behavior>(value: Self::Bits, scale: i32, behavior: B) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) = DecimalLayout::<Enc, W>::scale_b(value.to_limbs(), scale, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn next<B: Behavior>(value: Self::Bits, step: Step, behavior: B) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) = DecimalLayout::<Enc, W>::next(value.to_limbs(), step, env);
        (Self::Bits::from_limbs(bits), flags)
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
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

    #[test]
    fn the_class_changes_at_the_smallest_normal_magnitude() {
        // A BID encoding with a small coefficient has its exponent field
        // above the coefficient field. The smallest normal magnitude,
        // 10^emin, is 10^(p - 1) at the smallest exponent and 1 at emin.
        let d32 = |field: u32, coefficient: u32| D32Bid::from_bits((field << 23) | coefficient);
        assert_eq!(d32(0, 1_000_000).classify(), Class::Normal);
        assert_eq!(d32(0, 999_999).classify(), Class::Subnormal);
        assert_eq!(d32(6, 1).classify(), Class::Normal);
        assert_eq!(d32(5, 1).classify(), Class::Subnormal);
        let d64 = |field: u64, coefficient: u64| D64Bid::from_bits((field << 53) | coefficient);
        assert_eq!(d64(0, 10_u64.pow(15)).classify(), Class::Normal);
        assert_eq!(d64(0, 10_u64.pow(15) - 1).classify(), Class::Subnormal);
        assert_eq!(d64(15, 1).classify(), Class::Normal);
        assert_eq!(d64(14, 1).classify(), Class::Subnormal);
        let d128 =
            |field: u128, coefficient: u128| D128Bid::from_bits((field << 113) | coefficient);
        assert_eq!(d128(0, 10_u128.pow(33)).classify(), Class::Normal);
        assert_eq!(d128(0, 10_u128.pow(33) - 1).classify(), Class::Subnormal);
        assert_eq!(d128(33, 1).classify(), Class::Normal);
        assert_eq!(d128(32, 1).classify(), Class::Subnormal);
        // An operation reports the subnormal operand, and DAZ reads it as a
        // zero with its exponent.
        let (normal, subnormal) = (d64(0, 10_u64.pow(15)), d64(0, 10_u64.pow(15) - 1));
        let (_, flags) = normal.add_with(normal, Env::IEEE);
        assert!(!flags.contains(Flags::DENORMAL_INPUT));
        let (_, flags) = subnormal.add_with(normal, Env::IEEE);
        assert!(flags.contains(Flags::DENORMAL_INPUT));
        let (sum, flags) = subnormal.add_with(normal, Env::IEEE.with_denormals_are_zero(true));
        assert_eq!(
            (sum.to_bits(), flags),
            (normal.to_bits(), Flags::DENORMAL_INPUT)
        );
    }
}
