//! Add, subtract, multiply, divide, square root, and fused multiply-add for
//! the binary formats.
//!
//! Each operation handles the special values, computes an exact result or an
//! exact result with a sticky bit, and rounds it once through the rounding
//! routine. The special cases follow IEEE 754 and, where IEEE 754 leaves a
//! choice, SoftFloat, which TestFloat checks, or the NaN rule of the
//! behavior.

use core::cmp::Ordering;

use super::{Layout, Number, Unpacked};
use crate::env::{Behavior, Env, Flags, Rounding};
use crate::exact::{self, Unrounded};
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};

/// A nonzero finite value: `significand * 2^exponent`.
#[derive(Clone, Copy)]
struct Term<L> {
    negative: bool,
    exponent: i64,
    significand: L,
}

impl<L: Limbs> Term<L> {
    /// Returns the weight of the highest significand bit.
    #[inline]
    fn top(&self) -> i64 {
        self.exponent + i64::from(self.significand.bit_length()) - 1
    }
}

/// The exact sum of two terms.
enum Sum<L> {
    /// A nonzero sum. The lowest bit can mark a value that is not exact.
    Value(Unrounded<L>),
    /// An exact zero.
    Zero,
}

/// Adds two terms. The sum keeps every bit that decides its rounding.
///
/// The terms share the lowest weight that keeps both exact, unless the width
/// of `L` cannot hold that. Then the lower term shifts right and sets its
/// lowest bit when it loses bits. That happens only when the lower term is at
/// least four times smaller than the other, so the lost bits stay far below
/// the rounding position, where a set lowest bit rounds like the true value.
///
/// The width of `L` must hold each term plus 3 bits, for the carry and the
/// jammed bit. It must also hold the rounding precision plus 5 bits. Then a
/// jammed bit stays below the round bit and the bit below it, after a
/// cancellation of one bit and a carry.
// Every addition of every format shares this function, so LLVM does not
// inline it for `#[inline]`. Forced with the rounding routine, it takes
// binary128 addition with a mode from 25.8 to 19.7 ns in the benchmark.
#[allow(clippy::inline_always)]
#[inline(always)]
fn sum<L: Limbs>(first: Term<L>, second: Term<L>) -> Sum<L> {
    let highest = first.top().max(second.top());
    let floor = highest - i64::from(L::BITS - 3);
    let common = first.exponent.min(second.exponent).max(floor);
    // Both terms usually shift left, so the branch predicts well. An
    // order of the terms by exponent would branch on random data.
    let align = |term: Term<L>| {
        if term.exponent >= common {
            let shift =
                u32::try_from(term.exponent - common).expect("the shift stays inside the width");
            term.significand.shl(shift)
        } else {
            let shift = u32::try_from(common - term.exponent).unwrap_or(u32::MAX);
            term.significand.shr_jam(shift)
        }
    };
    let (left, right) = (align(first), align(second));
    let (negative, significand) = if first.negative == second.negative {
        (first.negative, left.add(right))
    } else {
        match left.compare(&right) {
            Ordering::Greater => (first.negative, left.sub(right)),
            Ordering::Less => (second.negative, right.sub(left)),
            Ordering::Equal => return Sum::Zero,
        }
    };
    Sum::Value(Unrounded {
        negative,
        exponent: i32::try_from(common).expect("a sum exponent fits an i32"),
        significand,
        sticky: false,
    })
}

/// Returns the sign of an exact zero sum of operands with different signs:
/// negative only when rounding toward negative.
#[inline]
fn zero_sum_sign(env: &Env) -> bool {
    env.rounding == Rounding::TowardNegative
}

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Decodes an operand, reports a subnormal operand, and applies DAZ.
    pub(super) fn operand<L: Limbs>(bits: L, env: &Env, flags: &mut Flags) -> Unpacked<L> {
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

    /// Rounds an exact value and encodes it.
    #[inline]
    fn finish<In: Limbs, L: Limbs, B: Behavior>(
        value: &Unrounded<In>,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        let (rounded, round_flags) = exact::round::<In, L, Self, B>(value, behavior);
        (Self::encode(rounded), flags | round_flags)
    }

    /// Rounds a sum, or encodes an exact zero sum.
    #[inline]
    fn finish_sum<In: Limbs, L: Limbs, B: Behavior>(
        sum: &Sum<In>,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        match sum {
            Sum::Value(value) => Self::finish(value, behavior, flags),
            Sum::Zero => Self::exact(Unpacked::zero(zero_sum_sign(&behavior.env())), flags),
        }
    }

    /// Encodes a result that needs no rounding.
    #[inline]
    pub(super) fn exact<L: Limbs>(value: Unpacked<L>, flags: Flags) -> (L, Flags) {
        (Self::encode(value), flags)
    }

    /// Returns an infinity, or for a format without one the NaN or, when the
    /// behavior saturates or the format has no NaN, the largest finite value.
    pub(super) fn infinity<L: Limbs>(negative: bool, env: &Env) -> Unpacked<L> {
        if Self::TARGET.has_infinity {
            Unpacked::Infinity { negative }
        } else if env.saturate || !Self::TARGET.has_nan {
            exact::largest(negative, Self::TARGET.precision_in(env), &Self::TARGET)
        } else {
            Unpacked::Nan {
                negative,
                signaling: false,
                payload: L::ZERO,
            }
        }
    }

    /// Returns `significand * 2^exponent` of a finite operand, widened.
    #[inline]
    fn term<L: Widen>(negative: bool, exponent: i32, significand: L) -> Term<L::Double> {
        Term {
            negative,
            exponent: i64::from(exponent),
            significand: significand.resize(),
        }
    }

    /// Returns the square root of a positive finite operand.
    #[inline]
    fn sqrt_finite<L: Widen, B: Behavior>(
        number: Number<L>,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        // Shift the radicand to an even exponent and to 2p + 3 or 2p + 4 bits,
        // so that the root has p + 2 bits.
        let mut shift = 2 * Self::PRECISION + 3 - number.significand.bit_length();
        if (i64::from(number.exponent) - i64::from(shift)) % 2 != 0 {
            shift += 1;
        }
        let radicand = number.significand.resize::<L::Double>().shl(shift);
        let (root, inexact) = limbs::square_root(radicand);
        let shift = i32::try_from(shift).expect("a shift fits an i32");
        let value = Unrounded {
            negative: false,
            exponent: (number.exponent - shift) / 2,
            significand: root,
            sticky: inexact,
        };
        Self::finish(&value, behavior, flags)
    }

    /// Adds `left` and `right`, or subtracts `right` when `subtract` is set.
    #[inline]
    pub fn add<L: Widen, B: Behavior>(
        left: L,
        right: L,
        subtract: bool,
        behavior: B,
    ) -> (L, Flags) {
        if let (Some(a), Some(b)) = (Self::normal(left), Self::normal(right)) {
            let b = Number {
                negative: b.negative != subtract,
                ..b
            };
            return Self::add_finite(a, b, behavior, Flags::NONE);
        }
        Self::add_special(left, right, subtract, &behavior.env())
    }

    /// Adds operands that are not both normal: a zero, a subnormal number, an
    /// infinity, a NaN, or an unsupported encoding.
    ///
    /// The function stays out of line, so that the normal path of [`Self::add`]
    /// inlines into its caller.
    #[inline(never)]
    fn add_special<L: Widen>(left: L, right: L, subtract: bool, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        let y = if subtract { y.negate() } else { y };
        if let (Some(a), Some(b)) = (x.number(), y.number()) {
            return Self::add_finite(a, b, *env, flags);
        }
        match (x, y) {
            (Unpacked::Infinity { negative: a }, Unpacked::Infinity { negative: b }) if a != b => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) => Self::exact(x, flags),
            (_, Unpacked::Infinity { .. }) => Self::exact(y, flags),
            (Unpacked::Zero { negative: a, .. }, Unpacked::Zero { negative: b, .. }) => {
                let negative = if a == b { a } else { zero_sum_sign(env) };
                Self::exact(Unpacked::zero(negative), flags)
            }
            (
                Unpacked::Zero { .. },
                Unpacked::Finite {
                    negative,
                    exponent,
                    significand,
                },
            )
            | (
                Unpacked::Finite {
                    negative,
                    exponent,
                    significand,
                },
                Unpacked::Zero { .. },
            ) => {
                let value = Unrounded {
                    negative,
                    exponent,
                    significand,
                    sticky: false,
                };
                Self::finish(&value, *env, flags)
            }
            _ => unreachable!(
                "the earlier cases handle NaNs, unsupported operands, and finite pairs"
            ),
        }
    }

    /// Adds two nonzero finite operands.
    #[inline]
    fn add_finite<L: Widen, B: Behavior>(
        a: Number<L>,
        b: Number<L>,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        if Self::PRECISION + 5 <= L::BITS {
            // The limb width of the storage holds the sum and its jammed bit.
            let term = |number: Number<L>| Term {
                negative: number.negative,
                exponent: i64::from(number.exponent),
                significand: number.significand,
            };
            Self::finish_sum(&sum(term(a), term(b)), behavior, flags)
        } else {
            let term = |number: Number<L>| {
                Self::term(number.negative, number.exponent, number.significand)
            };
            Self::finish_sum(&sum(term(a), term(b)), behavior, flags)
        }
    }

    /// Multiplies `left` by `right`.
    #[inline]
    pub fn mul<L: Widen, B: Behavior>(left: L, right: L, behavior: B) -> (L, Flags) {
        if let (Some(a), Some(b)) = (Self::normal(left), Self::normal(right)) {
            return Self::mul_finite(a, b, behavior, Flags::NONE);
        }
        Self::mul_special(left, right, &behavior.env())
    }

    /// Multiplies operands that are not both normal.
    ///
    /// The function stays out of line, so that the normal path of [`Self::mul`]
    /// inlines into its caller.
    #[inline(never)]
    fn mul_special<L: Widen>(left: L, right: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        if let (Some(a), Some(b)) = (x.number(), y.number()) {
            return Self::mul_finite(a, b, *env, flags);
        }
        let negative = sign(&x) != sign(&y);
        match (x, y) {
            (Unpacked::Infinity { .. }, Unpacked::Zero { .. })
            | (Unpacked::Zero { .. }, Unpacked::Infinity { .. }) => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) | (_, Unpacked::Infinity { .. }) => {
                Self::exact(Unpacked::Infinity { negative }, flags)
            }
            (Unpacked::Zero { .. }, _) | (_, Unpacked::Zero { .. }) => {
                Self::exact(Unpacked::zero(negative), flags)
            }
            _ => unreachable!(
                "the earlier cases handle NaNs, unsupported operands, and finite pairs"
            ),
        }
    }

    /// Multiplies two nonzero finite operands.
    #[inline]
    fn mul_finite<L: Widen, B: Behavior>(
        a: Number<L>,
        b: Number<L>,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        let product = Unrounded {
            negative: a.negative != b.negative,
            exponent: a.exponent + b.exponent,
            significand: a.significand.widening_mul(b.significand),
            sticky: false,
        };
        Self::finish(&product, behavior, flags)
    }

    /// Divides `left` by `right`.
    #[inline]
    pub fn div<L: Widen, B: Behavior>(left: L, right: L, behavior: B) -> (L, Flags) {
        if let (Some(a), Some(b)) = (Self::normal(left), Self::normal(right)) {
            return Self::div_finite(a, b, behavior, Flags::NONE);
        }
        Self::div_special(left, right, &behavior.env())
    }

    /// Divides operands that are not both normal.
    ///
    /// The function stays out of line, so that the normal path of [`Self::div`]
    /// inlines into its caller.
    #[inline(never)]
    fn div_special<L: Widen>(left: L, right: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        if let (Some(a), Some(b)) = (x.number(), y.number()) {
            return Self::div_finite(a, b, *env, flags);
        }
        let negative = sign(&x) != sign(&y);
        match (x, y) {
            (Unpacked::Infinity { .. }, Unpacked::Infinity { .. })
            | (Unpacked::Zero { .. }, Unpacked::Zero { .. }) => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) => Self::exact(Unpacked::Infinity { negative }, flags),
            (_, Unpacked::Infinity { .. }) | (Unpacked::Zero { .. }, _) => {
                Self::exact(Unpacked::zero(negative), flags)
            }
            (_, Unpacked::Zero { .. }) => {
                Self::exact(Self::infinity(negative, env), flags | Flags::DIVIDE_BY_ZERO)
            }
            _ => unreachable!(
                "the earlier cases handle NaNs, unsupported operands, and finite pairs"
            ),
        }
    }

    /// Divides two nonzero finite operands.
    #[inline]
    fn div_finite<L: Widen, B: Behavior>(
        a: Number<L>,
        b: Number<L>,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        // Shift the dividend so that the quotient has at least p + 2 bits.
        // The numerator then has at most 2p + 2 bits.
        let shift = Self::PRECISION + 2 + b.significand.bit_length() - a.significand.bit_length();
        let numerator = a.significand.resize::<L::Double>().shl(shift);
        let (quotient, remainder) = limbs::divide(numerator, b.significand.resize());
        let shift = i32::try_from(shift).expect("a shift fits an i32");
        let value = Unrounded {
            negative: a.negative != b.negative,
            exponent: a.exponent - b.exponent - shift,
            significand: quotient,
            sticky: !remainder.is_zero(),
        };
        Self::finish(&value, behavior, flags)
    }

    /// Returns the square root of `value`.
    #[inline]
    pub fn sqrt<L: Widen, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
        if let Some(number) = Self::normal(value).filter(|number| !number.negative) {
            return Self::sqrt_finite(number, behavior, Flags::NONE);
        }
        Self::sqrt_special(value, &behavior.env())
    }

    /// Returns the square root of an operand that is not a positive normal
    /// number.
    ///
    /// The function stays out of line, so that the normal path of [`Self::sqrt`]
    /// inlines into its caller.
    #[inline(never)]
    fn sqrt_special<L: Widen>(value: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(value, env, &mut flags);
        // A one-operand NaN propagates as SoftFloat does, against a zero.
        if let Some((result, special)) = nan::special(&x, &Unpacked::zero(false), env) {
            return Self::exact(result, flags | special);
        }
        match x {
            Unpacked::Zero { .. } | Unpacked::Infinity { negative: false } => Self::exact(x, flags),
            Unpacked::Infinity { negative: true } | Unpacked::Finite { negative: true, .. } => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            Unpacked::Finite {
                negative: false,
                exponent,
                significand,
            } => {
                let positive = Number {
                    negative: false,
                    exponent,
                    significand,
                };
                Self::sqrt_finite(positive, *env, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }

    /// Returns `left * right + addend`, rounded once.
    ///
    /// The `fused_order` field of the NaN rule orders the NaN operands, and
    /// the `invalid_product` field decides `0 * inf + NaN`.
    #[inline]
    pub fn mul_add<L: Widen, B: Behavior>(left: L, right: L, addend: L, behavior: B) -> (L, Flags) {
        if let (Some(a), Some(b), Some(c)) = (
            Self::normal(left),
            Self::normal(right),
            Self::normal(addend),
        ) {
            let product = Self::product(a, b);
            let addend = Self::term(c.negative, c.exponent, c.significand);
            return Self::finish_sum(&sum(product, addend), behavior, Flags::NONE);
        }
        Self::mul_add_special(left, right, addend, &behavior.env())
    }

    /// Returns `left * right + addend` for operands that are not all normal.
    ///
    /// The function stays out of line, so that the normal path of [`Self::mul_add`]
    /// inlines into its caller.
    #[inline(never)]
    fn mul_add_special<L: Widen>(left: L, right: L, addend: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        let third = Self::operand(addend, env, &mut flags);
        if [&first, &second, &third]
            .iter()
            .any(|value| matches!(value, Unpacked::Unsupported))
        {
            return Self::exact(default_nan(env), flags | Flags::INVALID);
        }
        if first.is_nan() || second.is_nan() {
            let (value, special) = nan::fused(&first, &second, &third, env);
            return Self::exact(value, flags | special);
        }
        let product_negative = sign(&first) != sign(&second);
        // The product of the first two operands, as a value to combine with
        // the addend: an infinity, an invalid product, or a number.
        let product = {
            match (first, second) {
                (Unpacked::Infinity { .. }, Unpacked::Zero { .. })
                | (Unpacked::Zero { .. }, Unpacked::Infinity { .. }) => {
                    let (value, special) = nan::invalid_product(&third, env);
                    return Self::exact(value, flags | special);
                }
                (Unpacked::Infinity { .. }, _) | (_, Unpacked::Infinity { .. }) => {
                    Unpacked::Infinity {
                        negative: product_negative,
                    }
                }
                _ => {
                    if let Some((value, special)) =
                        nan::special(&Unpacked::zero(false), &third, env)
                    {
                        return Self::exact(value, flags | special);
                    }
                    if matches!(third, Unpacked::Infinity { .. }) {
                        return Self::exact(third, flags);
                    }
                    return Self::finite_mul_add(
                        first,
                        second,
                        third,
                        product_negative,
                        env,
                        flags,
                    );
                }
            }
        };
        if third.is_nan() {
            let (value, special) = nan::propagate(&product, &third, env);
            return Self::exact(value, flags | special);
        }
        match (product, third) {
            (
                Unpacked::Infinity {
                    negative: product_sign,
                },
                Unpacked::Infinity {
                    negative: addend_sign,
                },
            ) if product_sign != addend_sign => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) => Self::exact(product, flags),
            _ => unreachable!("an infinite product is the only other case"),
        }
    }

    /// Returns the exact product of two nonzero finite operands.
    #[inline]
    fn product<L: Widen>(a: Number<L>, b: Number<L>) -> Term<L::Double> {
        Term {
            negative: a.negative != b.negative,
            exponent: i64::from(a.exponent) + i64::from(b.exponent),
            significand: a.significand.widening_mul(b.significand),
        }
    }

    /// Returns `x * y + z` for finite or zero operands and a finite or zero
    /// addend.
    fn finite_mul_add<L: Widen>(
        x: Unpacked<L>,
        y: Unpacked<L>,
        z: Unpacked<L>,
        product_negative: bool,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let product = match (x.number(), y.number()) {
            (Some(first), Some(second)) => Some(Self::product(first, second)),
            _ => None,
        };
        match (product, z) {
            (None, Unpacked::Zero { negative, .. }) => {
                let negative = if negative == product_negative {
                    negative
                } else {
                    zero_sum_sign(env)
                };
                Self::exact(Unpacked::zero(negative), flags)
            }
            (
                None,
                Unpacked::Finite {
                    negative,
                    exponent,
                    significand,
                },
            ) => Self::finish(
                &Unrounded::from_term(Self::term(negative, exponent, significand)),
                *env,
                flags,
            ),
            (Some(product), Unpacked::Zero { .. }) => {
                Self::finish(&Unrounded::from_term(product), *env, flags)
            }
            (
                Some(product),
                Unpacked::Finite {
                    negative,
                    exponent,
                    significand,
                },
            ) => Self::finish_sum(
                &sum(product, Self::term(negative, exponent, significand)),
                *env,
                flags,
            ),
            _ => unreachable!("the addend is zero or finite"),
        }
    }
}

/// Returns the sign of a zero, finite, or infinite value.
fn sign<L: Limbs>(value: &Unpacked<L>) -> bool {
    match value {
        Unpacked::Zero { negative, .. }
        | Unpacked::Finite { negative, .. }
        | Unpacked::Infinity { negative } => *negative,
        Unpacked::Nan { .. } | Unpacked::Unsupported => false,
    }
}

impl<L: Limbs> Unrounded<L> {
    /// Returns an exact term as a value to round.
    fn from_term(term: Term<L>) -> Self {
        Self {
            negative: term.negative,
            exponent: i32::try_from(term.exponent)
                .expect("an operand or product exponent fits an i32"),
            significand: term.significand,
            sticky: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{
        Env, Flags, FusedNanOrder, InvalidProduct, NanPropagation, NanRule, Rounding,
    };
    use crate::float::{F8E4M3Fn, F32, F64, F80};

    const F64_TWO: u64 = 0x4000_0000_0000_0000;

    #[test]
    fn basic_operations_round_once() {
        let (tenth, fifth) = (
            F64::from_bits(0x3FB9_9999_9999_999A),
            F64::from_bits(0x3FC9_9999_9999_999A),
        );
        let (sum, flags) = tenth.add_with(fifth, Env::IEEE);
        assert_eq!(
            (sum.to_bits(), flags),
            (0x3FD3_3333_3333_3334, Flags::INEXACT | Flags::ROUNDED_UP)
        );
        let (root, flags) = F64::from_bits(F64_TWO).sqrt_with(Env::IEEE);
        assert_eq!(
            (root.to_bits(), flags),
            (0x3FF6_A09E_667F_3BCD, Flags::INEXACT | Flags::ROUNDED_UP)
        );
        let (third, _) = F64::from_bits(0x3FF0_0000_0000_0000)
            .div_with(F64::from_bits(0x4008_0000_0000_0000), Env::IEEE);
        assert_eq!(third.to_bits(), 0x3FD5_5555_5555_5555);
        let (product, flags) =
            F32::from_bits(0x3FC0_0000).mul_with(F32::from_bits(0x4000_0000), Env::IEEE);
        assert_eq!((product.to_bits(), flags), (0x4040_0000, Flags::NONE));
    }

    #[test]
    fn exact_zero_sums_take_the_sign_of_the_direction() {
        let one = F32::from_bits(0x3F80_0000);
        assert_eq!(one.sub_with(one, Env::IEEE).0.to_bits(), 0);
        assert_eq!(
            one.sub_with(one, Rounding::TowardNegative).0.to_bits(),
            0x8000_0000
        );
        let (negative_zero, zero) = (F32::from_bits(0x8000_0000), F32::from_bits(0));
        assert_eq!(
            negative_zero.add_with(negative_zero, Env::IEEE).0.to_bits(),
            0x8000_0000
        );
        assert_eq!(negative_zero.add_with(zero, Env::IEEE).0.to_bits(), 0);
    }

    #[test]
    fn invalid_and_divide_by_zero_cases() {
        let infinity = F32::from_bits(0x7F80_0000);
        let (nan, flags) = infinity.sub_with(infinity, Env::IEEE);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0000, Flags::INVALID));
        let (quotient, flags) = F32::from_bits(0xBF80_0000).div_with(F32::from_bits(0), Env::IEEE);
        assert_eq!(
            (quotient.to_bits(), flags),
            (0xFF80_0000, Flags::DIVIDE_BY_ZERO)
        );
        let (root, flags) = F32::from_bits(0xBF80_0000).sqrt_with(Env::IEEE);
        assert_eq!((root.to_bits(), flags), (0x7FC0_0000, Flags::INVALID));
        assert_eq!(
            F32::from_bits(0x8000_0000).sqrt_with(Env::IEEE).0.to_bits(),
            0x8000_0000
        );
        // A format without an infinity gives its NaN for a division by zero.
        let (nan, flags) = F8E4M3Fn::from_bits(0x38).div_with(F8E4M3Fn::from_bits(0), Env::IEEE);
        assert_eq!((nan.to_bits(), flags), (0x7F, Flags::DIVIDE_BY_ZERO));
        // An exact infinity is no overflow, so saturation keeps it.
        let saturate = Env::IEEE.with_saturate(true);
        let (quotient, flags) = F32::from_bits(0x3F80_0000).div_with(F32::from_bits(0), saturate);
        assert_eq!(
            (quotient.to_bits(), flags),
            (0x7F80_0000, Flags::DIVIDE_BY_ZERO)
        );
        let infinity = F32::from_bits(0x7F80_0000);
        let (sum, flags) = infinity.add_with(F32::from_bits(0x3F80_0000), saturate);
        assert_eq!((sum.to_bits(), flags), (0x7F80_0000, Flags::NONE));
    }

    #[test]
    fn fused_multiply_add_rounds_once_and_follows_softfloat_for_invalid_cases() {
        // (1 + 2^-23)^2 - (1 + 2^-22) = 2^-46, which a separate multiply loses.
        let a = F32::from_bits(0x3F80_0001);
        let (result, flags) = a.mul_add_with(a, F32::from_bits(0xBF80_0002), Env::IEEE);
        assert_eq!((result.to_bits(), flags), (0x2880_0000, Flags::NONE));
        let (zero, infinity, quiet) = (
            F32::from_bits(0),
            F32::from_bits(0x7F80_0000),
            F32::from_bits(0x7FC0_1234),
        );
        let (nan, flags) = zero.mul_add_with(infinity, quiet, Env::IEEE);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0000, Flags::INVALID));
        let (addend, flags) = F32::from_bits(0x3F80_0000).mul_add_with(
            F32::from_bits(0x3F80_0000),
            infinity,
            Env::IEEE,
        );
        assert_eq!((addend.to_bits(), flags), (0x7F80_0000, Flags::NONE));
    }

    #[test]
    fn the_fused_order_selects_among_the_nan_operands() {
        let one = F32::from_bits(0x3F80_0000);
        let (factor, addend) = (F32::from_bits(0x7FC0_0001), F32::from_bits(0x7FC0_0002));
        let signaling = F32::from_bits(0x7F80_0003);
        let addend_first = Env::IEEE.with_nan(
            NanRule::new(NanPropagation::SignalingFirst)
                .with_fused_order(FusedNanOrder::AddendFirst),
        );
        // Two quiet NaNs: the factor first by default, the addend first in
        // the Arm order.
        let (nan, flags) = factor.mul_add_with(one, addend, Env::IEEE);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0001, Flags::NONE));
        let (nan, flags) = factor.mul_add_with(one, addend, addend_first);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0002, Flags::NONE));
        // A signaling second factor comes before a quiet addend in both.
        let (nan, flags) = one.mul_add_with(signaling, addend, addend_first);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0003, Flags::INVALID));
        // With the first-operand rule, the addend is the first operand.
        let first = Env::IEEE.with_nan(
            NanRule::new(NanPropagation::FirstOperand).with_fused_order(FusedNanOrder::AddendFirst),
        );
        let (nan, _) = one.mul_add_with(signaling, addend, first);
        assert_eq!(nan.to_bits(), 0x7FC0_0002);
        // The PowerPC order puts the addend between the two factors.
        let addend_second = Env::IEEE.with_nan(
            NanRule::new(NanPropagation::FirstOperand)
                .with_fused_order(FusedNanOrder::AddendSecond),
        );
        let (nan, flags) = one.mul_add_with(signaling, addend, addend_second);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0002, Flags::INVALID));
        let (nan, flags) = factor.mul_add_with(signaling, addend, addend_second);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0001, Flags::INVALID));
    }

    #[test]
    fn a_nan_addend_can_take_precedence_over_an_invalid_product() {
        let (zero, infinity) = (F32::from_bits(0), F32::from_bits(0x7F80_0000));
        let yields = Env::IEEE.with_nan(
            Env::IEEE
                .nan
                .with_invalid_product(InvalidProduct::YieldsToNan),
        );
        let (nan, flags) = zero.mul_add_with(infinity, F32::from_bits(0x7FC0_1234), yields);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_1234, Flags::NONE));
        let (nan, flags) = infinity.mul_add_with(zero, F32::from_bits(0xFF80_0001), yields);
        assert_eq!((nan.to_bits(), flags), (0xFFC0_0001, Flags::INVALID));
        // Without a NaN addend the product still signals invalid.
        let (nan, flags) = zero.mul_add_with(infinity, F32::from_bits(0x3F80_0000), yields);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0000, Flags::INVALID));
        // The PowerPC rule signals invalid and still returns the addend.
        let signals = Env::IEEE.with_nan(
            Env::IEEE
                .nan
                .with_invalid_product(InvalidProduct::SignalsAndYieldsToNan),
        );
        let (nan, flags) = zero.mul_add_with(infinity, F32::from_bits(0xFFC0_1234), signals);
        assert_eq!((nan.to_bits(), flags), (0xFFC0_1234, Flags::INVALID));
        let (nan, flags) = infinity.mul_add_with(zero, F32::from_bits(0x3F80_0000), signals);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0000, Flags::INVALID));
    }

    #[test]
    fn each_nan_rule_selects_its_nan() {
        let quiet = F32::from_bits(0x7FC0_0001);
        let signaling = F32::from_bits(0x7F80_0002);
        let rule =
            |propagation| Env::IEEE.with_nan(NanRule::new(propagation).with_default_negative(true));
        let cases = [
            (NanPropagation::SignalingFirst, 0x7FC0_0002),
            (NanPropagation::FirstOperand, 0x7FC0_0001),
            (NanPropagation::LargerSignificand, 0x7FC0_0001),
            (NanPropagation::DefaultNan, 0xFFC0_0000),
        ];
        for (propagation, bits) in cases {
            let (nan, flags) = quiet.add_with(signaling, rule(propagation));
            assert_eq!(
                (nan.to_bits(), flags),
                (bits, Flags::INVALID),
                "{propagation:?}"
            );
        }
        // Between two quiet NaNs, the larger significand wins.
        let larger = F80::from_bits(0x7FFF_C000_0000_0000_0009);
        let smaller = F80::from_bits(0xFFFF_C000_0000_0000_0001);
        let (nan, _) = smaller.mul_with(larger, rule(NanPropagation::LargerSignificand));
        assert_eq!(nan.to_bits(), 0x7FFF_C000_0000_0000_0009);
    }
}
