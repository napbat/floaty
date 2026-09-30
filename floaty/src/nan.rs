//! The NaN that an operation returns, by the NaN rule of its behavior.

use core::cmp::Ordering;

use crate::env::{Env, Flags, FusedNanOrder, InvalidProduct, NanPropagation};
use crate::format::internal::MinMax;
use crate::limbs::Limbs;
use crate::unpacked::Unpacked;

impl<L: Limbs> Unpacked<L> {
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

/// Moves a NaN payload of `from` bits to a payload of `to` bits and keeps its
/// high-order bits.
pub fn align_payload<In: Limbs, Out: Limbs>(payload: In, from: u32, to: u32) -> Out {
    if to >= from {
        payload.resize::<Out>().shl(to - from)
    } else {
        payload.shr(from - to).resize()
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

/// Returns the result of a one-operand operation with an unsupported or a NaN
/// operand, or `None` for a number, as [`special`] gives it.
pub fn special_unary<L: Limbs>(value: &Unpacked<L>, env: &Env) -> Option<(Unpacked<L>, Flags)> {
    special(value, &Unpacked::zero(false), env)
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
    select(&[first, second], env)
}

/// Returns the NaN of a fused multiply-add with a NaN factor, made quiet, in
/// the order of [`FusedNanOrder`]. A signaling NaN operand signals invalid.
pub fn fused<L: Limbs>(
    first: &Unpacked<L>,
    second: &Unpacked<L>,
    addend: &Unpacked<L>,
    env: &Env,
) -> (Unpacked<L>, Flags) {
    debug_assert!(first.is_nan() || second.is_nan(), "a factor is a NaN");
    match env.nan.fused_order {
        FusedNanOrder::ProductFirst => {
            let (product, product_flags) = propagate(first, second, env);
            let (value, flags) = propagate(&product, addend, env);
            (value, product_flags | flags)
        }
        FusedNanOrder::AddendFirst => select(&[addend, first, second], env),
        FusedNanOrder::AddendSecond => select(&[first, addend, second], env),
    }
}

/// Returns the result of a fused multiply-add with an unsupported or a NaN
/// operand, or with the invalid product `0 * inf`, or `None` otherwise.
///
/// The cases apply in order: an unsupported operand signals invalid and gives
/// the default NaN, a NaN factor gives the NaN of [`fused`], the invalid
/// product follows [`invalid_product`], and a NaN addend gives that NaN,
/// made quiet. The binary and the decimal engines share this order.
#[inline]
pub fn fused_special<L: Limbs>(
    first: &Unpacked<L>,
    second: &Unpacked<L>,
    addend: &Unpacked<L>,
    env: &Env,
) -> Option<(Unpacked<L>, Flags)> {
    if [first, second, addend]
        .iter()
        .any(|value| matches!(value, Unpacked::Unsupported))
    {
        return Some((default_nan(env), Flags::INVALID));
    }
    if first.is_nan() || second.is_nan() {
        return Some(fused(first, second, addend, env));
    }
    if matches!(
        (first, second),
        (Unpacked::Infinity { .. }, Unpacked::Zero { .. })
            | (Unpacked::Zero { .. }, Unpacked::Infinity { .. })
    ) {
        return Some(invalid_product(addend, env));
    }
    if addend.is_nan() {
        // Every rule picks the NaN over the number that the product is.
        return Some(propagate(&Unpacked::zero(false), addend, env));
    }
    None
}

/// Returns the result of a fused multiply-add whose product is the invalid
/// `0 * inf`, by the [`InvalidProduct`] rule. An addend that is not a NaN
/// gives the default NaN.
pub fn invalid_product<L: Limbs>(addend: &Unpacked<L>, env: &Env) -> (Unpacked<L>, Flags) {
    if !addend.is_nan() {
        return (default_nan(env), Flags::INVALID);
    }
    match env.nan.invalid_product {
        InvalidProduct::Signals => {
            let (value, flags) = propagate(&default_nan(env), addend, env);
            (value, flags | Flags::INVALID)
        }
        InvalidProduct::YieldsToNan => propagate(&Unpacked::zero(false), addend, env),
        InvalidProduct::SignalsAndYieldsToNan => {
            let (value, flags) = propagate(&Unpacked::zero(false), addend, env);
            (value, flags | Flags::INVALID)
        }
    }
}

/// Selects the NaN that the rule gives among operands in operand order, made
/// quiet. At least one operand is a NaN. A signaling NaN operand signals
/// invalid.
fn select<L: Limbs>(operands: &[&Unpacked<L>], env: &Env) -> (Unpacked<L>, Flags) {
    debug_assert!(
        operands.iter().any(|value| value.is_nan()),
        "an operand is a NaN"
    );
    let flags = if operands.iter().any(|value| value.is_signaling()) {
        Flags::INVALID
    } else {
        Flags::NONE
    };
    let propagation = env.nan.propagation;
    if propagation == NanPropagation::DefaultNan {
        return (default_nan(env), flags);
    }
    // Each rule picks one of two operands, and the pick of a list is the
    // pick of each operand against the earlier picks. The quieting waits
    // until the end, so a signaling NaN keeps its priority.
    let chosen = operands
        .iter()
        .copied()
        .reduce(|chosen, next| choose(chosen, next, propagation))
        .expect("an operation has an operand");
    (chosen.quiet(), flags)
}

/// Returns the operand that a propagation rule picks from two, in operand
/// order. At least one is a NaN.
fn choose<'a, L: Limbs>(
    first: &'a Unpacked<L>,
    second: &'a Unpacked<L>,
    propagation: NanPropagation,
) -> &'a Unpacked<L> {
    let (first_signaling, second_signaling) = (first.is_signaling(), second.is_signaling());
    match propagation {
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
        NanPropagation::DefaultNan => unreachable!("the caller gives the default NaN"),
    }
}

/// Returns the result of a minimum or maximum operation with a NaN operand,
/// or `None` when both operands are numbers.
///
/// `minimum` and `maximum` return a NaN for a NaN operand. `minimumNumber`
/// and `maximumNumber` return the number, and signal invalid for a signaling
/// NaN. `minNum` and `maxNum` return the number for a quiet NaN, and a NaN for
/// a signaling NaN. The NaN rule selects a NaN result.
pub fn min_max<L: Limbs>(
    first: &Unpacked<L>,
    second: &Unpacked<L>,
    operation: MinMax,
    env: &Env,
) -> Option<(Unpacked<L>, Flags)> {
    let signaling = first.is_signaling() || second.is_signaling();
    match (first.is_nan(), second.is_nan()) {
        (false, false) => None,
        (true, true) => Some(propagate(first, second, env)),
        (true, false) | (false, true) => {
            let returns_nan = match operation {
                MinMax::Minimum | MinMax::Maximum => true,
                MinMax::MinNum | MinMax::MaxNum => signaling,
                MinMax::MinimumNumber | MinMax::MaximumNumber => false,
            };
            if returns_nan {
                return Some(propagate(first, second, env));
            }
            let flags = if signaling {
                Flags::INVALID
            } else {
                Flags::NONE
            };
            let number = if first.is_nan() { *second } else { *first };
            Some((number, flags))
        }
    }
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
