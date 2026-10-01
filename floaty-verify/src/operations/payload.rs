//! An oracle for the NaN payload operations of IEEE 754-2019 section 9.7 on
//! the binary formats: `getPayload`, `setPayload`, and
//! `setPayloadSignaling`, by the documented rule of floaty.
//!
//! The oracle finds a NaN and its payload in the bits of the encoding, by
//! the field layout, without floaty's decoder. The payload is the fraction
//! below the quiet bit. A NaN of `NoInf` or `Fnuz` has the payload 0, and an
//! x87 encoding without its integer bit is no NaN. A payload argument reads
//! by [`Operand::read`](crate::mpfr::Operand::read). The admissible payloads
//! are +0 and the positive integers below `2^b`, for the `b` payload bits,
//! without 0 for a signaling NaN. Only an IEEE or x87 encoding has a
//! signaling NaN.

use floaty::format::{Encoding, Standard, Storage, Width};
use floaty::{Binary, Env, Flags, Float};
use rug::{Float as BigFloat, Integer};

use super::{Outcome, Sample};
use crate::encodings::IntegerBit;
use crate::mpfr::{Format, Read, Specials};

/// A NaN payload operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Payload {
    /// `getPayload`: [`Float::payload`].
    Get,
    /// `setPayload`: [`Float::from_payload`].
    Set,
    /// `setPayloadSignaling`: [`Float::from_payload_signaling`].
    SetSignaling,
}

impl Payload {
    /// Every payload operation.
    pub const ALL: [Self; 3] = [Self::Get, Self::Set, Self::SetSignaling];

    /// Runs the operation of floaty on `x`.
    #[must_use]
    pub fn apply<const E: u32, Enc: Encoding, const W: usize>(
        self,
        x: Float<Binary<E, Enc>, W>,
    ) -> Float<Binary<E, Enc>, W>
    where
        Width<W>: Storage,
        Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
    {
        match self {
            Self::Get => x.payload(),
            Self::Set => Float::<Binary<E, Enc>, W>::from_payload(x),
            Self::SetSignaling => Float::<Binary<E, Enc>, W>::from_payload_signaling(x),
        }
    }
}

/// Returns the expected result of a payload operation on a sample.
#[must_use]
pub fn payload(operation: Payload, sample: &Sample<8>, format: &Format) -> Outcome {
    match operation {
        Payload::Get => get(sample, format),
        Payload::Set => set(sample, format, false),
        Payload::SetSignaling => set(sample, format, true),
    }
}

/// Returns the payload of a NaN as an integral value, or -1.
fn get(sample: &Sample<8>, format: &Format) -> Outcome {
    match nan_payload(sample, format) {
        Some(payload) if payload == 0 => Outcome::Zero { negative: false },
        Some(payload) => Outcome::Finite(BigFloat::with_val(format.precision, payload)),
        None => Outcome::Finite(BigFloat::with_val(format.precision, -1)),
    }
}

/// Returns the NaN with the payload of the sample, or +0.
fn set(sample: &Sample<8>, format: &Format, signaling: bool) -> Outcome {
    let mut flags = Flags::NONE;
    let payload = match sample.operand.read(&Env::IEEE, &mut flags) {
        Read::Number(value) if value.is_integer() && !value.is_sign_negative() => {
            value.to_integer()
        }
        Read::Number(_) | Read::Nan(_) | Read::Unsupported => None,
    };
    let signaling_exists = format.specials == Specials::Ieee;
    let admissible = payload.filter(|payload| match payload_bits(sample, format) {
        Some(bits) => {
            payload.significant_bits() <= bits
                && !(signaling && (*payload == 0 || !signaling_exists))
        }
        None => false,
    });
    match admissible {
        Some(payload) => Outcome::Nan {
            negative: format.specials == Specials::Fnuz,
            signaling,
            payload,
        },
        None => Outcome::Zero { negative: false },
    }
}

/// Returns the number of payload bits, or `None` for a format without a
/// NaN.
fn payload_bits(sample: &Sample<8>, format: &Format) -> Option<u32> {
    match (format.specials, sample.layout.integer_bit) {
        (Specials::Ieee, IntegerBit::Implicit) => Some(sample.layout.fraction_bits() - 1),
        (Specials::Ieee, IntegerBit::Explicit) => Some(sample.layout.fraction_bits() - 2),
        (Specials::NoInf | Specials::Fnuz, _) => Some(0),
        (Specials::Finite, _) => None,
    }
}

/// Returns the payload of a NaN encoding, read from its fields, or `None`
/// for every other encoding.
fn nan_payload(sample: &Sample<8>, format: &Format) -> Option<Integer> {
    let layout = sample.layout;
    let fraction_bits = layout.fraction_bits();
    let bits = &sample.bits;
    let field = Integer::from(bits >> fraction_bits).keep_bits(layout.exponent_bits);
    let fraction = Integer::from(bits.keep_bits_ref(fraction_bits));
    let all_ones = field == layout.largest_field();
    let payload_bits = payload_bits(sample, format)?;
    let payload = Integer::from(fraction.keep_bits_ref(payload_bits));
    let nan = match (format.specials, layout.integer_bit) {
        // An x87 NaN has its integer bit and a nonzero fraction below it.
        (Specials::Ieee, IntegerBit::Explicit) => {
            all_ones
                && fraction.get_bit(fraction_bits - 1)
                && Integer::from(fraction.keep_bits_ref(fraction_bits - 1)) != 0
        }
        (Specials::Ieee, IntegerBit::Implicit) => all_ones && fraction != 0,
        (Specials::NoInf, _) => {
            all_ones && fraction == Integer::from(Integer::u_pow_u(2, fraction_bits)) - 1
        }
        (Specials::Fnuz, _) => *bits == Integer::from(Integer::u_pow_u(2, layout.width - 1)),
        (Specials::Finite, _) => false,
    };
    nan.then_some(payload)
}
