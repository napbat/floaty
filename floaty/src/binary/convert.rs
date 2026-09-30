//! Conversion of a decoded value of another format to a binary format.

use super::Layout;
use crate::env::{Behavior, Env, Flags, NanPropagation};
use crate::exact::Unrounded;
use crate::format::internal::Source;
use crate::format::{Encoding, Storage, Width};
use crate::limbs::Limbs;
use crate::nan;
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Converts a decoded value of another format, binary or decimal.
    ///
    /// A NaN payload keeps its high-order bits. The payload of a decimal
    /// source is the value of its trailing significand field, as the Intel
    /// decimal library converts it.
    #[inline]
    pub fn convert_from<In: Limbs, Out: Limbs, B: Behavior>(
        value: Unpacked<In>,
        source: Source,
        behavior: B,
    ) -> (Out, Flags) {
        let env = &behavior.env();
        match value {
            Unpacked::Zero { negative, .. } => {
                (Self::encode(Unpacked::zero(negative)), Flags::NONE)
            }
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let value = Unrounded {
                    negative,
                    exponent,
                    significand,
                    sticky: false,
                };
                match source {
                    Source::Decimal { .. } => Self::from_decimal(&value, env),
                    Source::Binary { .. } => Self::finish(&value, behavior, Flags::NONE),
                }
            }
            Unpacked::Infinity { negative } => Self::convert_infinity(negative, env),
            Unpacked::Nan {
                negative,
                signaling,
                payload,
            } => Self::convert_nan(negative, signaling, payload, source, env),
            Unpacked::Unsupported => (Self::encode(nan::default_nan(env)), Flags::INVALID),
        }
    }

    /// Converts a NaN. The payload keeps its high-order bits. A format without
    /// a NaN gives positive zero and signals invalid. The function stays out
    /// of line, so that the number path of
    /// [`convert_from`](Self::convert_from) inlines into its caller.
    #[inline(never)]
    fn convert_nan<In: Limbs, Out: Limbs>(
        negative: bool,
        signaling: bool,
        payload: In,
        source: Source,
        env: &Env,
    ) -> (Out, Flags) {
        if !Self::TARGET.has_nan {
            return (Self::encode(Unpacked::zero(false)), Flags::INVALID);
        }
        let flags = if signaling {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        let nan = match env.nan.propagation {
            NanPropagation::DefaultNan => Self::encode(nan::default_nan(env)),
            NanPropagation::SignalingFirst
            | NanPropagation::FirstOperand
            | NanPropagation::LargerSignificand => Self::encode(Unpacked::Nan {
                negative,
                signaling: false,
                payload: nan::align_payload(payload, source.payload_bits(), Self::PAYLOAD_BITS),
            }),
        };
        (nan, flags)
    }

    /// Converts an infinity. A format without an infinity gives the result
    /// of [`infinity`](Self::infinity) and signals invalid.
    #[inline(never)]
    fn convert_infinity<Out: Limbs>(negative: bool, env: &Env) -> (Out, Flags) {
        let flags = if Self::TARGET.has_infinity {
            Flags::NONE
        } else {
            Flags::INVALID
        };
        (Self::encode(Self::infinity(negative, env)), flags)
    }
}
