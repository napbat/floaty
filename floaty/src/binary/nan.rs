//! The NaN that an operation returns, by the NaN rule of its behavior.

use core::cmp::Ordering;

use super::Unpacked;
use crate::env::{Env, Flags, NanPropagation};
use crate::limbs::Limbs;

impl<L: Limbs> Unpacked<L> {
    /// Returns `true` for a NaN.
    pub fn is_nan(&self) -> bool {
        matches!(self, Self::Nan { .. })
    }

    /// Returns `true` for a signaling NaN.
    pub fn is_signaling(&self) -> bool {
        matches!(
            self,
            Self::Nan {
                signaling: true,
                ..
            }
        )
    }

    /// Returns the value with the other sign. A NaN keeps its sign.
    #[must_use]
    pub fn negate(self) -> Self {
        match self {
            Self::Zero { negative } => Self::Zero {
                negative: !negative,
            },
            Self::Finite {
                negative,
                exponent,
                significand,
            } => Self::Finite {
                negative: !negative,
                exponent,
                significand,
            },
            Self::Infinity { negative } => Self::Infinity {
                negative: !negative,
            },
            Self::Nan { .. } | Self::Unsupported => self,
        }
    }

    /// Returns the value made quiet, when it is a NaN.
    #[must_use]
    fn quiet(self) -> Self {
        match self {
            Self::Nan {
                negative, payload, ..
            } => Self::Nan {
                negative,
                signaling: false,
                payload,
            },
            other => other,
        }
    }
}

/// Returns the default NaN of the NaN rule.
pub fn default_nan<L: Limbs>(env: &Env) -> Unpacked<L> {
    Unpacked::Nan {
        negative: env.nan.default_negative,
        signaling: false,
        payload: L::ZERO,
    }
}

/// Returns the result of an operation with an unsupported operand or a NaN
/// operand, or `None` when every operand is a number.
///
/// An unsupported operand signals invalid and gives the default NaN. A NaN
/// operand gives the NaN that the rule selects. The operands are in operand
/// order.
pub fn special<L: Limbs>(
    first: &Unpacked<L>,
    second: &Unpacked<L>,
    env: &Env,
) -> Option<(Unpacked<L>, Flags)> {
    if matches!(first, Unpacked::Unsupported) || matches!(second, Unpacked::Unsupported) {
        return Some((default_nan(env), Flags::INVALID));
    }
    if first.is_nan() || second.is_nan() {
        return Some(propagate(first, second, env));
    }
    None
}

/// Selects the NaN that a two-operand operation returns, made quiet. At least
/// one operand is a NaN. A signaling NaN operand signals invalid.
///
/// The rules follow the SoftFloat specializations, and TestFloat checks each
/// one: ARM-VFPv2 for `SignalingFirst`, 8086-SSE for `FirstOperand`, 8086 for
/// `LargerSignificand`, and ARM-VFPv2-defaultNaN for `DefaultNan`. SoftFloat's 8086-SSE code
/// for x87 extended precision is its 8086 code, so TestFloat checks the
/// `FirstOperand` rule on the other formats only. No oracle checks it on x87
/// values: the x87 unit follows the `LargerSignificand` rule.
pub fn propagate<L: Limbs>(
    first: &Unpacked<L>,
    second: &Unpacked<L>,
    env: &Env,
) -> (Unpacked<L>, Flags) {
    debug_assert!(first.is_nan() || second.is_nan(), "an operand is a NaN");
    let (first_signaling, second_signaling) = (first.is_signaling(), second.is_signaling());
    let flags = if first_signaling || second_signaling {
        Flags::INVALID
    } else {
        Flags::NONE
    };
    let chosen = match env.nan.propagation {
        NanPropagation::DefaultNan => return (default_nan(env), flags),
        NanPropagation::SignalingFirst => {
            if first_signaling || (!second_signaling && first.is_nan()) {
                first
            } else {
                second
            }
        }
        NanPropagation::FirstOperand => {
            if first.is_nan() {
                first
            } else {
                second
            }
        }
        NanPropagation::LargerSignificand => match (first_signaling, second_signaling) {
            (true, false) if second.is_nan() => second,
            (true, false) => first,
            (false, true) if first.is_nan() => first,
            (false, true) => second,
            _ => larger(first, second),
        },
    };
    (chosen.quiet(), flags)
}

/// Returns the NaN with the larger significand, as the x87 unit does. A NaN is
/// larger than a number. Between equal significands the positive NaN wins.
fn larger<'a, L: Limbs>(first: &'a Unpacked<L>, second: &'a Unpacked<L>) -> &'a Unpacked<L> {
    let key = |value: &Unpacked<L>| match value {
        Unpacked::Nan {
            signaling,
            payload,
            negative,
        } => Some((!signaling, *payload, *negative)),
        _ => None,
    };
    match (key(first), key(second)) {
        (Some(_), None) => first,
        (None, _) => second,
        (
            Some((first_quiet, first_payload, first_negative)),
            Some((second_quiet, second_payload, second_negative)),
        ) => {
            let order = first_quiet
                .cmp(&second_quiet)
                .then_with(|| first_payload.compare(&second_payload));
            match order {
                Ordering::Greater => first,
                Ordering::Equal if !first_negative && second_negative => first,
                Ordering::Less | Ordering::Equal => second,
            }
        }
    }
}
