//! The decimal formats: IEEE 754 decimal32, decimal64, and decimal128 in the
//! BID and DPD encodings.

use core::cmp::Ordering;
use core::marker::PhantomData;

mod arithmetic;
mod codec;
mod compare;
mod convert;
mod declet;
mod digits;
mod elementary;
mod integral;
mod payload;
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

use self::digits::power_of_ten_u128;
use self::round::DecimalTarget;
use crate::elementary::Transcendental;
use crate::env::{Behavior, Env, Flags, TotalOrder};
use crate::exact::Unrounded;
use crate::float::Class;
use crate::format::internal::{LimbConversion, MinMax, Quotient, Source, Step};
use crate::format::{Decimal, DecimalEncoding, Standard, Storage, Width};
use crate::host::Host;
use crate::integer::{Integer, ToInt};
use crate::limbs::{self, Limbs, Widen};
use crate::unpacked::Unpacked;

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

    /// Returns a NaN payload, or zero for a payload above `10^(p - 1) - 1`,
    /// which is not canonical.
    fn canonical_payload(payload: u128) -> u128 {
        if payload > Self::LARGEST_PAYLOAD {
            0
        } else {
            payload
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
            Unpacked::Unsupported => unreachable!("a decimal format has no unsupported encoding"),
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

    /// Decodes an operand, reports a subnormal operand, and applies DAZ. A
    /// zero from DAZ keeps the exponent of the subnormal value.
    // `Float::convert_with` and every decimal operation read their operands
    // here. With `#[inline]` only on the forwarder of `Standard::operand`, a
    // decimal32 to decimal64 `convert` took 24.3 ns instead of 22.8 ns.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn operand<L: Limbs>(bits: L, env: &Env, flags: &mut Flags) -> Unpacked<L> {
        let value = Self::decode(bits);
        let Unpacked::Finite {
            negative,
            exponent,
            significand,
        } = value
        else {
            return value;
        };
        if !Self::is_subnormal(exponent, limbs::to_u128(&significand)) {
            return value;
        }
        *flags |= Flags::DENORMAL_INPUT;
        if env.denormals_are_zero {
            Unpacked::Zero { negative, exponent }
        } else {
            value
        }
    }

    /// Rounds an exact result with a preferred exponent, and encodes it.
    #[inline]
    fn finish<L: Limbs, In: Limbs, B: Behavior>(
        value: &Unrounded<In>,
        preferred: i64,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        let preferred = i32::try_from(preferred.clamp(i64::from(i32::MIN), i64::from(i32::MAX)))
            .expect("a clamped exponent fits an i32");
        let (rounded, round_flags) = round::round::<In, L, Self, B>(value, preferred, behavior);
        (Self::encode(rounded), flags | round_flags)
    }

    /// Encodes a result that needs no rounding.
    fn exact<L: Limbs>(value: Unpacked<L>, flags: Flags) -> (L, Flags) {
        (Self::encode(value), flags)
    }

    /// Returns a zero with the exponent nearest `exponent` in the range of the
    /// format.
    fn zero<L>(negative: bool, exponent: i64) -> Unpacked<L> {
        round::zero(negative, exponent, &Self::TARGET)
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
    const SOURCE: Source = Source::Decimal {
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
        let preferred = i64::from(value.exponent);
        let (bits, flags) =
            DecimalLayout::<Enc, W>::finish(value, preferred, behavior, Flags::NONE);
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

    fn elementary<B: Behavior>(
        value: Self::Bits,
        function: Transcendental,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) =
            DecimalLayout::<Enc, W>::elementary(value.to_limbs(), function, behavior);
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
    use crate::float::{Class, D32Bid, D64Bid, D128Bid};

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
