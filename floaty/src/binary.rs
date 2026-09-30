//! Unpacking, packing, and classification of the binary formats.

use core::cmp::Ordering;
use core::marker::PhantomData;

mod arithmetic;
mod compare;
mod convert;
mod from_decimal;
mod integral;
mod remainder;
mod scale;

use crate::env::{Behavior, Env, Flags, TotalOrder};
use crate::exact::{self, RoundingTarget, Target, Unrounded};
use crate::float::Class;
use crate::format::internal::{LimbConversion, MinMax, Quotient, Source, Step};
use crate::format::{Binary, Encoding, EncodingKind, Standard, Storage, Width};
use crate::host::Host;
use crate::integer::{Integer, ToInt};
use crate::limbs::Limbs;
use crate::unpacked::{Number, Unpacked};

/// The layout constants and codec of `Binary<E, Enc>` at width `W`.
pub struct Layout<const E: u32, Enc, const W: usize> {
    encoding: PhantomData<Enc>,
}

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// The width in bits.
    const WIDTH: u32 = <Width<W> as Storage>::WIDTH;

    /// Rejects a layout that has no valid format at compile time.
    const VALID: () = {
        assert!(
            E >= 2 && E <= 28,
            "a binary format has 2 to 28 exponent bits"
        );
        assert!(
            Self::WIDTH >= E + 2,
            "a binary format has at least one fraction bit"
        );
        assert!(
            match Enc::FIXED_LAYOUT {
                Some(layout) => E == layout.exponent_bits && Self::WIDTH == layout.width,
                None => true,
            },
            "an encoding with a fixed layout exists only at its exponent bits and width"
        );
    };

    const IS_X87: bool = matches!(Enc::KIND, EncodingKind::X87);

    /// The fraction field width. For `X87` it includes the integer bit.
    const FRACTION_BITS: u32 = Self::WIDTH.saturating_sub(E + 1);

    /// The largest exponent field value.
    const FIELD_MAX: u64 = (1 << E) - 1;

    const BIAS: i32 = match (Enc::BIAS, Enc::KIND) {
        (Some(bias), _) => bias,
        (None, EncodingKind::Fnuz) => 1 << (E - 1),
        (
            None,
            EncodingKind::Ieee | EncodingKind::NoInf | EncodingKind::X87 | EncodingKind::Finite,
        ) => (1 << (E - 1)) - 1,
    };

    const PRECISION: u32 = {
        let () = Self::VALID;
        if Self::IS_X87 {
            Self::FRACTION_BITS
        } else {
            Self::FRACTION_BITS + 1
        }
    };

    const EMAX: i32 = {
        let () = Self::VALID;
        match Enc::KIND {
            EncodingKind::Ieee | EncodingKind::X87 => (1 << E) - 2 - Self::BIAS,
            EncodingKind::NoInf | EncodingKind::Fnuz | EncodingKind::Finite => {
                (1 << E) - 1 - Self::BIAS
            }
        }
    };

    const EMIN: i32 = {
        let () = Self::VALID;
        1 - Self::BIAS
    };

    /// The number of significand bits below the leading bit.
    const SHIFT: i32 = signed(Self::PRECISION - 1);

    /// The parameters that the rounding routine rounds to.
    const TARGET: Target = Target {
        precision: Self::PRECISION,
        emin: Self::EMIN,
        emax: Self::EMAX,
        has_infinity: matches!(Enc::KIND, EncodingKind::Ieee | EncodingKind::X87),
        all_ones_is_nan: matches!(Enc::KIND, EncodingKind::NoInf),
        has_nan: !matches!(Enc::KIND, EncodingKind::Finite),
    };

    /// The host format with the same encoding.
    const HOST: Host = Host::of_binary(E, Self::WIDTH, Enc::KIND);

    /// The number of NaN payload bits, below the quiet bit.
    const PAYLOAD_BITS: u32 = match Enc::KIND {
        EncodingKind::Ieee => Self::FRACTION_BITS - 1,
        EncodingKind::X87 => 62,
        EncodingKind::NoInf | EncodingKind::Fnuz | EncodingKind::Finite => 0,
    };

    /// Decodes an encoding.
    #[inline]
    pub fn decode<L: Limbs>(bits: L) -> Unpacked<L> {
        let () = Self::VALID;
        let negative = bits.bit(Self::WIDTH - 1);
        let field = bits.field(Self::FRACTION_BITS, E);
        let fraction = bits.low_bits(Self::FRACTION_BITS);
        match Enc::KIND {
            EncodingKind::Ieee if field == Self::FIELD_MAX => {
                if fraction.is_zero() {
                    Unpacked::Infinity { negative }
                } else {
                    Unpacked::Nan {
                        negative,
                        signaling: !fraction.bit(Self::FRACTION_BITS - 1),
                        payload: fraction.low_bits(Self::FRACTION_BITS - 1),
                    }
                }
            }
            EncodingKind::NoInf
                if field == Self::FIELD_MAX && fraction == L::ones(Self::FRACTION_BITS) =>
            {
                Unpacked::Nan {
                    negative,
                    signaling: false,
                    payload: L::ZERO,
                }
            }
            EncodingKind::Fnuz if field == 0 && fraction.is_zero() => {
                if negative {
                    Unpacked::Nan {
                        negative,
                        signaling: false,
                        payload: L::ZERO,
                    }
                } else {
                    Unpacked::zero(negative)
                }
            }
            EncodingKind::X87 => Self::decode_x87(negative, field, fraction),
            EncodingKind::Ieee
            | EncodingKind::NoInf
            | EncodingKind::Fnuz
            | EncodingKind::Finite => Self::decode_finite(negative, field, fraction),
        }
    }

    /// Returns the sign, the exponent, and the significand of a normal
    /// encoding, or `None` for any other encoding. A normal operand needs no
    /// special case, no denormal flag, and no DAZ, so an operation reads it
    /// without the full decode. A `NoInf` number with the largest exponent
    /// field takes the full decode.
    #[inline]
    pub(crate) fn normal<L: Limbs>(bits: L) -> Option<Number<L>> {
        let field = bits.field(Self::FRACTION_BITS, E);
        let normal = match Enc::KIND {
            EncodingKind::Ieee | EncodingKind::NoInf => field != 0 && field != Self::FIELD_MAX,
            EncodingKind::Fnuz | EncodingKind::Finite => field != 0,
            EncodingKind::X87 => field != 0 && field != Self::FIELD_MAX && bits.bit(63),
        };
        if !normal {
            return None;
        }
        let fraction = bits.low_bits(Self::FRACTION_BITS);
        let significand = if Self::IS_X87 {
            fraction
        } else {
            fraction.with_bit(Self::FRACTION_BITS)
        };
        Some(Number {
            negative: bits.bit(Self::WIDTH - 1),
            exponent: Self::unbiased(field) - Self::SHIFT,
            significand,
        })
    }

    /// Decodes a number of a format with an implicit integer bit.
    #[inline]
    fn decode_finite<L: Limbs>(negative: bool, field: u64, fraction: L) -> Unpacked<L> {
        if field != 0 {
            Unpacked::Finite {
                negative,
                exponent: Self::unbiased(field) - Self::SHIFT,
                significand: fraction.with_bit(Self::FRACTION_BITS),
            }
        } else if fraction.is_zero() {
            Unpacked::zero(negative)
        } else {
            Unpacked::Finite {
                negative,
                exponent: Self::EMIN - Self::SHIFT,
                significand: fraction,
            }
        }
    }

    /// Decodes an x87 encoding. `fraction` holds the integer bit as bit 63.
    fn decode_x87<L: Limbs>(negative: bool, field: u64, fraction: L) -> Unpacked<L> {
        let integer = fraction.bit(63);
        let tail = fraction.low_bits(63);
        if field == Self::FIELD_MAX {
            if !integer {
                Unpacked::Unsupported
            } else if tail.is_zero() {
                Unpacked::Infinity { negative }
            } else {
                Unpacked::Nan {
                    negative,
                    signaling: !tail.bit(62),
                    payload: tail.low_bits(62),
                }
            }
        } else if field == 0 {
            // A pseudo-denormal has the integer bit set. Its value equals the
            // normal value with exponent field 1, so both share one exponent.
            if fraction.is_zero() {
                Unpacked::zero(negative)
            } else {
                Unpacked::Finite {
                    negative,
                    exponent: Self::EMIN - Self::SHIFT,
                    significand: fraction,
                }
            }
        } else if integer {
            Unpacked::Finite {
                negative,
                exponent: Self::unbiased(field) - Self::SHIFT,
                significand: fraction,
            }
        } else {
            Unpacked::Unsupported
        }
    }

    /// Returns the class of an encoding.
    pub fn classify<L: Limbs>(bits: L) -> Class {
        match Self::decode(bits) {
            Unpacked::Zero { .. } => Class::Zero,
            Unpacked::Finite { .. } if bits.field(Self::FRACTION_BITS, E) == 0 => Class::Subnormal,
            Unpacked::Finite { .. } => Class::Normal,
            Unpacked::Infinity { .. } => Class::Infinite,
            Unpacked::Nan {
                signaling: true, ..
            } => Class::SignalingNan,
            Unpacked::Nan {
                signaling: false, ..
            } => Class::QuietNan,
            Unpacked::Unsupported => Class::Unsupported,
        }
    }

    /// Returns `true` when the encoding is the canonical encoding of its value:
    /// encoding the decoded value gives the same bits.
    pub fn is_canonical<L: Limbs>(bits: L) -> bool {
        match Self::decode(bits) {
            Unpacked::Unsupported => false,
            value => Self::encode(value) == bits,
        }
    }

    /// Returns the encoding with its sign set to `negative`. The zero and the
    /// NaN of [`Fnuz`](crate::Fnuz) each have one encoding, so they keep it.
    pub fn with_sign<L: Limbs>(bits: L, negative: bool) -> L {
        let magnitude = bits.low_bits(Self::WIDTH - 1);
        if matches!(Enc::KIND, EncodingKind::Fnuz) && magnitude.is_zero() {
            bits
        } else if negative {
            magnitude.with_bit(Self::WIDTH - 1)
        } else {
            magnitude
        }
    }

    #[inline]
    fn unbiased(field: u64) -> i32 {
        i32::try_from(field).expect("an exponent field has at most 28 bits") - Self::BIAS
    }
}

impl<const E: u32, Enc: Encoding, const W: usize> RoundingTarget for Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    const TARGET: Target = Layout::<E, Enc, W>::TARGET;
}

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// The largest exponent field value of a finite number.
    const FINITE_FIELD_MAX: u64 = match Enc::KIND {
        EncodingKind::Ieee | EncodingKind::X87 => Self::FIELD_MAX - 1,
        EncodingKind::NoInf | EncodingKind::Fnuz | EncodingKind::Finite => Self::FIELD_MAX,
    };

    /// Encodes a value in its canonical encoding.
    ///
    /// The value must be representable in the format. A negative zero in
    /// [`Fnuz`](crate::Fnuz) encodes as positive zero, because the format has
    /// no negative zero. A NaN in [`Finite`](crate::Finite) encodes as
    /// positive zero, because the format has no NaN; the operation that gives
    /// the NaN signals invalid.
    #[inline]
    pub fn encode<L: Limbs>(value: Unpacked<L>) -> L {
        let () = Self::VALID;
        match value {
            Unpacked::Zero { negative, .. } => {
                let negative = negative && !matches!(Enc::KIND, EncodingKind::Fnuz);
                Self::assemble(negative, 0, L::ZERO)
            }
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => Self::encode_finite(negative, exponent, significand),
            Unpacked::Infinity { negative } => match Enc::KIND {
                EncodingKind::Ieee => Self::assemble(negative, Self::FIELD_MAX, L::ZERO),
                EncodingKind::X87 => {
                    Self::assemble(negative, Self::FIELD_MAX, L::ZERO.with_bit(63))
                }
                EncodingKind::NoInf | EncodingKind::Fnuz | EncodingKind::Finite => {
                    unreachable!("no caller gives an infinity to a format without one")
                }
            },
            Unpacked::Nan {
                negative,
                signaling,
                payload,
            } => Self::encode_nan(negative, signaling, payload),
            Unpacked::Unsupported => unreachable!("an unsupported encoding has no canonical form"),
        }
    }

    #[inline]
    fn encode_finite<L: Limbs>(negative: bool, exponent: i32, significand: L) -> L {
        debug_assert!(
            significand.low_bits(Self::PRECISION) == significand && !significand.is_zero(),
            "the significand is nonzero and below 2^PRECISION"
        );
        if significand.bit(Self::PRECISION - 1) {
            let field = u64::try_from(exponent + Self::SHIFT + Self::BIAS)
                .expect("a normal value has a positive exponent field");
            debug_assert!(
                (1..=Self::FINITE_FIELD_MAX).contains(&field),
                "a normal value has an exponent in range"
            );
            let fraction = if Self::IS_X87 {
                significand
            } else {
                significand.low_bits(Self::FRACTION_BITS)
            };
            debug_assert!(
                !matches!(Enc::KIND, EncodingKind::NoInf)
                    || field != Self::FIELD_MAX
                    || fraction != L::ones(Self::FRACTION_BITS),
                "a NoInf number is not the NaN encoding"
            );
            Self::assemble(negative, field, fraction)
        } else {
            debug_assert!(
                exponent == Self::EMIN - Self::SHIFT,
                "a subnormal value has the minimum exponent"
            );
            Self::assemble(negative, 0, significand)
        }
    }

    fn encode_nan<L: Limbs>(negative: bool, signaling: bool, payload: L) -> L {
        match Enc::KIND {
            EncodingKind::Ieee => {
                let quiet_bit = Self::FRACTION_BITS - 1;
                debug_assert!(
                    payload.low_bits(quiet_bit) == payload,
                    "the payload fits below the quiet bit"
                );
                debug_assert!(
                    !signaling || !payload.is_zero(),
                    "a signaling NaN has a payload"
                );
                let fraction = if signaling {
                    payload
                } else {
                    payload.with_bit(quiet_bit)
                };
                Self::assemble(negative, Self::FIELD_MAX, fraction)
            }
            EncodingKind::X87 => {
                debug_assert!(
                    payload.low_bits(62) == payload,
                    "the payload fits below the quiet bit"
                );
                debug_assert!(
                    !signaling || !payload.is_zero(),
                    "a signaling NaN has a payload"
                );
                let fraction = payload.with_bit(63);
                let fraction = if signaling {
                    fraction
                } else {
                    fraction.with_bit(62)
                };
                Self::assemble(negative, Self::FIELD_MAX, fraction)
            }
            EncodingKind::NoInf => {
                debug_assert!(!signaling, "the format has no signaling NaN");
                Self::assemble(negative, Self::FIELD_MAX, L::ones(Self::FRACTION_BITS))
            }
            EncodingKind::Fnuz => {
                debug_assert!(!signaling, "the format has no signaling NaN");
                Self::assemble(true, 0, L::ZERO)
            }
            EncodingKind::Finite => Self::assemble(false, 0, L::ZERO),
        }
    }

    /// Decodes an operand, reports a subnormal operand, and applies DAZ.
    // `Float::convert_with` and the special paths read their operands here.
    // With only `#[inline]`, a binary32 to binary32 `convert` took 8.0 ns
    // instead of 7.3 ns.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn operand<L: Limbs>(bits: L, env: &Env, flags: &mut Flags) -> Unpacked<L> {
        let value = Self::decode(bits);
        let subnormal =
            matches!(value, Unpacked::Finite { .. }) && bits.field(Self::FRACTION_BITS, E) == 0;
        if !subnormal {
            return value;
        }
        *flags |= Flags::DENORMAL_INPUT;
        if env.denormals_are_zero {
            Unpacked::zero(bits.bit(Self::WIDTH - 1))
        } else {
            value
        }
    }

    /// Rounds an exact value, encodes it, and adds the flags of the rounding to
    /// `flags`.
    #[inline]
    fn finish<In: Limbs, L: Limbs, B: Behavior>(
        value: &Unrounded<In>,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        let (rounded, round_flags) = exact::round::<In, L, Self, B>(value, behavior);
        (Self::encode(rounded), flags | round_flags)
    }

    /// Encodes a result that needs no rounding.
    #[inline]
    fn exact<L: Limbs>(value: Unpacked<L>, flags: Flags) -> (L, Flags) {
        (Self::encode(value), flags)
    }

    /// Returns an infinity, or for a format without one the NaN or, when the
    /// behavior saturates or the format has no NaN, the largest finite value.
    fn infinity<L: Limbs>(negative: bool, env: &Env) -> Unpacked<L> {
        if Self::TARGET.has_infinity {
            Unpacked::Infinity { negative }
        } else if env.saturate || !Self::TARGET.has_nan {
            exact::largest(
                negative,
                env.precision_within(Self::TARGET.precision),
                &Self::TARGET,
            )
        } else {
            Unpacked::Nan {
                negative,
                signaling: false,
                payload: L::ZERO,
            }
        }
    }

    /// Places the sign, the exponent field, and the fraction in an encoding.
    #[inline]
    fn assemble<L: Limbs>(negative: bool, field: u64, fraction: L) -> L {
        // The sign is a field of one bit, so the encoder does not branch on
        // it.
        fraction
            .with_field(Self::FRACTION_BITS, E, field)
            .with_field(Self::WIDTH - 1, 1, u64::from(negative))
    }
}

impl<const E: u32, Enc: Encoding, const W: usize> Standard<W> for Binary<E, Enc>
where
    Width<W>: Storage,
{
    type Bits = <Width<W> as Storage>::Bits;

    const RADIX: u32 = 2;
    const PRECISION: u32 = Layout::<E, Enc, W>::PRECISION;
    const SIGNIFICAND_BITS: u32 = Layout::<E, Enc, W>::PRECISION;
    const EMAX: i32 = Layout::<E, Enc, W>::EMAX;
    const EMIN: i32 = Layout::<E, Enc, W>::EMIN;
    const SOURCE: Source = Source::Binary {
        payload_bits: Layout::<E, Enc, W>::PAYLOAD_BITS,
    };
    const HOST: Host = Layout::<E, Enc, W>::HOST;

    #[inline]
    fn mask(bits: Self::Bits) -> Self::Bits {
        let () = Layout::<E, Enc, W>::VALID;
        Self::Bits::from_limbs(bits.to_limbs().low_bits(Layout::<E, Enc, W>::WIDTH))
    }

    #[inline]
    fn unpack(bits: Self::Bits) -> Unpacked<<Self::Bits as LimbConversion>::Limbs> {
        Layout::<E, Enc, W>::decode(bits.to_limbs())
    }

    fn classify(bits: Self::Bits) -> Class {
        Layout::<E, Enc, W>::classify(bits.to_limbs())
    }

    // `Float::convert_with` reads its operand here. LLVM does not inline this
    // forwarder for `#[inline]`: a binary32 to binary32 `convert` took 12.4
    // ns instead of 7.4 ns, and an FP8 to binary32 `convert` 12.1 ns instead
    // of 7.4 ns.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    fn operand(
        bits: Self::Bits,
        env: &Env,
    ) -> (Unpacked<<Self::Bits as LimbConversion>::Limbs>, Flags) {
        let mut flags = Flags::NONE;
        let value = Layout::<E, Enc, W>::operand(bits.to_limbs(), env, &mut flags);
        (value, flags)
    }

    fn is_canonical(bits: Self::Bits) -> bool {
        Layout::<E, Enc, W>::is_canonical(bits.to_limbs())
    }

    fn is_sign_negative(bits: Self::Bits) -> bool {
        bits.to_limbs().bit(Layout::<E, Enc, W>::WIDTH - 1)
    }

    #[inline]
    fn round<L: Limbs, B: Behavior>(value: &Unrounded<L>, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) = Layout::<E, Enc, W>::finish(value, behavior, Flags::NONE);
        (Self::Bits::from_limbs(bits), flags)
    }

    #[inline]
    fn convert_from<L: Limbs, B: Behavior>(
        value: Unpacked<L>,
        source: Source,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) = Layout::<E, Enc, W>::convert_from(value, source, behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    #[inline]
    fn add<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        subtract: bool,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) =
            Layout::<E, Enc, W>::add(left.to_limbs(), right.to_limbs(), subtract, behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    #[inline]
    fn mul<B: Behavior>(left: Self::Bits, right: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) = Layout::<E, Enc, W>::mul(left.to_limbs(), right.to_limbs(), behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    #[inline]
    fn div<B: Behavior>(left: Self::Bits, right: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) = Layout::<E, Enc, W>::div(left.to_limbs(), right.to_limbs(), behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    #[inline]
    fn sqrt<B: Behavior>(value: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let (bits, flags) = Layout::<E, Enc, W>::sqrt(value.to_limbs(), behavior);
        (Self::Bits::from_limbs(bits), flags)
    }

    #[inline]
    fn mul_add<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        addend: Self::Bits,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let (bits, flags) = Layout::<E, Enc, W>::mul_add(
            left.to_limbs(),
            right.to_limbs(),
            addend.to_limbs(),
            behavior,
        );
        (Self::Bits::from_limbs(bits), flags)
    }

    fn with_sign(bits: Self::Bits, negative: bool) -> Self::Bits {
        Self::Bits::from_limbs(Layout::<E, Enc, W>::with_sign(bits.to_limbs(), negative))
    }

    #[inline]
    fn compare<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        behavior: B,
    ) -> (Option<Ordering>, Flags) {
        let env = &behavior.env();
        Layout::<E, Enc, W>::compare(left.to_limbs(), right.to_limbs(), env)
    }

    fn total_cmp(left: Self::Bits, right: Self::Bits, order: TotalOrder) -> Ordering {
        Layout::<E, Enc, W>::total_cmp(left.to_limbs(), right.to_limbs(), order)
    }

    #[inline]
    fn min_max<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        operation: MinMax,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) =
            Layout::<E, Enc, W>::min_max(left.to_limbs(), right.to_limbs(), operation, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn round_to_integral<B: Behavior>(value: Self::Bits, behavior: B) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) = Layout::<E, Enc, W>::round_to_integral(value.to_limbs(), env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn to_int<I: Integer, B: Behavior>(value: Self::Bits, behavior: B) -> (ToInt<I>, Flags) {
        let env = &behavior.env();
        Layout::<E, Enc, W>::to_int(value.to_limbs(), env)
    }

    fn remainder<B: Behavior>(
        left: Self::Bits,
        right: Self::Bits,
        quotient: Quotient,
        behavior: B,
    ) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) =
            Layout::<E, Enc, W>::remainder(left.to_limbs(), right.to_limbs(), quotient, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn scale_b<B: Behavior>(value: Self::Bits, scale: i32, behavior: B) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) = Layout::<E, Enc, W>::scale_b(value.to_limbs(), scale, env);
        (Self::Bits::from_limbs(bits), flags)
    }

    fn next<B: Behavior>(value: Self::Bits, step: Step, behavior: B) -> (Self::Bits, Flags) {
        let env = &behavior.env();
        let (bits, flags) = Layout::<E, Enc, W>::next(value.to_limbs(), step, env);
        (Self::Bits::from_limbs(bits), flags)
    }
}

/// Converts a `u32` to an `i32` in a constant.
const fn signed(value: u32) -> i32 {
    match 0_i32.checked_add_unsigned(value) {
        Some(result) => result,
        None => panic!("the value exceeds i32::MAX"),
    }
}

#[cfg(test)]
mod tests;
