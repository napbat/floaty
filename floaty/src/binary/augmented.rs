//! The augmented operations of IEEE 754-2019 section 9.5 for the binary
//! formats. [`Augmented`](crate::Augmented) states the rules.

use super::Layout;
use super::arithmetic::{Sum, Term, sum};
use crate::env::{Env, Flags, Rounding};
use crate::exact::{self, Unrounded};
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{Limbs, Widen};
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the head and the tail of `left + right`, or of `left - right`
    /// when `subtract` is set, and the flags.
    pub fn augmented_add<L: Widen>(left: L, right: L, subtract: bool, env: &Env) -> (L, L, Flags) {
        let env = env.with_rounding(Rounding::TiesTowardZero);
        let mut flags = Flags::NONE;
        let x = Self::operand(left, &env, &mut flags);
        let y = Self::operand(right, &env, &mut flags);
        let y = if subtract { y.negate() } else { y };
        let term = |value: Unpacked<L>| {
            value
                .number()
                .map(|number| Self::term(number.negative, number.exponent, number.significand))
        };
        let (big, small) = match (term(x), term(y)) {
            (Some(first), Some(second)) if first.top() >= second.top() => (first, Some(second)),
            (Some(first), Some(second)) => (second, Some(first)),
            (Some(first), None) if matches!(y, Unpacked::Zero { .. }) => (first, None),
            (None, Some(second)) if matches!(x, Unpacked::Zero { .. }) => (second, None),
            _ => {
                // A NaN, an infinity, an unsupported operand, or two zeros:
                // both results are the sum.
                let (bits, special) = Self::add(left, right, subtract, env);
                return (bits, bits, special);
            }
        };
        let value = small.map_or(Sum::Value(Unrounded::from_term(big)), |small| {
            sum(big, small)
        });
        // An inexact head is at most one binade above `big`, and at most
        // p binades below it, so `big - head` spans at most 2p bits and is
        // exact in the doubled width. The error then rounds as every sum of
        // two terms does.
        let error = |negated_head: Term<L::Double>| match (sum(big, negated_head), small) {
            (Sum::Value(difference), Some(small)) => sum(Term::from_exact(&difference), small),
            (Sum::Zero, Some(small)) => Sum::Value(Unrounded::from_term(small)),
            (difference, None) => difference,
        };
        Self::augment(&value, error, &env, flags)
    }

    /// Returns the head and the tail of `left * right`, and the flags.
    pub fn augmented_mul<L: Widen>(left: L, right: L, env: &Env) -> (L, L, Flags) {
        let env = env.with_rounding(Rounding::TiesTowardZero);
        let mut flags = Flags::NONE;
        let x = Self::operand(left, &env, &mut flags);
        let y = Self::operand(right, &env, &mut flags);
        let (Some(a), Some(b)) = (x.number(), y.number()) else {
            // A NaN, an infinity, an unsupported operand, or a zero: both
            // results are the product.
            let (bits, special) = Self::mul(left, right, env);
            return (bits, bits, special);
        };
        let product = Self::product(a, b);
        // The product has at most 2p bits, and the head is at most one binade
        // above it, so `product - head` is exact in the doubled width.
        let error = |negated_head: Term<L::Double>| sum(product, negated_head);
        Self::augment(
            &Sum::Value(Unrounded::from_term(product)),
            error,
            &env,
            flags,
        )
    }

    /// Rounds the head of an exact value, and the tail from the error that
    /// `error` returns for the negated head.
    fn augment<L: Widen>(
        value: &Sum<L::Double>,
        error: impl FnOnce(Term<L::Double>) -> Sum<L::Double>,
        env: &Env,
        flags: Flags,
    ) -> (L, L, Flags) {
        let (head, head_flags) = match value {
            Sum::Value(value) => exact::round::<L::Double, L, Self, Env>(value, *env),
            Sum::Zero => (Unpacked::zero(env.zero_sum_is_negative()), Flags::NONE),
        };
        let head_bits = Self::encode(head);
        let number = match head.number() {
            Some(number) if !head_flags.contains(Flags::OVERFLOW) => number,
            // A zero head, or an overflow: the tail is the head.
            _ => return (head_bits, head_bits, flags | head_flags),
        };
        if !head_flags.contains(Flags::INEXACT) {
            let tail = Self::encode(Unpacked::zero(number.negative));
            return (head_bits, tail, flags | head_flags);
        }
        let negated = Self::term(!number.negative, number.exponent, number.significand);
        let Sum::Value(error) = error(negated) else {
            unreachable!("an inexact head leaves a nonzero error");
        };
        let (tail, tail_flags) = exact::round::<L::Double, L, Self, Env>(&error, *env);
        // The tail holds the error of the head, so the flags describe the
        // pair, and the flags of the head drop out. A tiny inexact head leaves
        // a tiny tail, so the tail reports `TINY` for both. `head + tail` is
        // above the exact value in magnitude when the tail moves past the
        // error toward the sign of the head.
        let rounded_up = tail_flags.contains(Flags::INEXACT)
            && tail_flags.contains(Flags::ROUNDED_UP) == (error.negative == number.negative);
        let pair = if rounded_up {
            Flags::ROUNDED_UP
        } else {
            Flags::NONE
        };
        let flags = flags | tail_flags.difference(Flags::ROUNDED_UP) | pair;
        (head_bits, Self::encode(tail), flags)
    }
}

impl<L: Limbs> Term<L> {
    /// Returns an exact value as a term.
    fn from_exact(value: &Unrounded<L>) -> Self {
        debug_assert!(!value.sticky, "the value is exact");
        Self {
            negative: value.negative,
            exponent: i64::from(value.exponent),
            significand: value.significand,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
    use crate::float::{F32, F64};

    /// Returns the bits of the head and the tail, and the flags.
    fn bits<T>(
        result: (crate::Augmented<T>, Flags),
        to_bits: impl Fn(T) -> u64,
    ) -> (u64, u64, Flags) {
        let (pair, flags) = result;
        (to_bits(pair.head), to_bits(pair.tail), flags)
    }

    // The pairs are from the tests of the `fp-ieee` 0.1.0.6 Haskell package,
    // `test/AugmentedArithSpec.hs`, which compares them with exact
    // rationals. A tail that rounds to zero keeps the sign of the error.
    #[test]
    fn products_match_fp_ieee() {
        let binary64 = [
            (
                0x8009_EF76_B935_56A0,
                0x3FEE_179B_DE0A_1DD2,
                0x8009_57D3_CFCE_9C63,
                0,
            ),
            (
                0x8018_EB0E_0204_4F68,
                0xBFDC_93B8_3A57_51C8,
                0x000B_2059_BF8D_CE81,
                1 << 63,
            ),
            (
                0x000D_C3BD_0E6B_0A3C,
                0xBFE7_A77B_B9DF_06DA,
                0x800A_2CBA_9D52_695F,
                1 << 63,
            ),
            (
                0xBFED_25F2_402F_E726,
                0xBFE0_B42F_4E9E_B842,
                0x3FDE_6E33_5433_C1F9,
                0xBC5B_B70C_80F1_8340,
            ),
        ];
        for (x, y, head, tail) in binary64 {
            let result = F64::from_bits(x).augmented_mul_with(F64::from_bits(y), Env::IEEE);
            let (ours_head, ours_tail, _) = bits(result, F64::to_bits);
            assert_eq!((ours_head, ours_tail), (head, tail), "{x:#x} * {y:#x}");
        }
        let binary32 = [
            (0x000D_C284, 0xBDC9_CCA0, 0x8001_5B17, 0x8000_0000),
            (0x00AA_19DE, 0xBF34_D020, 0x8078_247A, 0x8000_0000),
            (0x0038_E6C6, 0xBF62_E8B2, 0x8032_6F73, 0x8000_0000),
            (0xBFD1_8CA3, 0x8040_0000, 0x0068_C651, 0),
            (0x4060_0000, 0x3F12_4925, 0x4000_0000, 0x33C0_0000),
            (0x4060_0000, 0xBF12_4925, 0xC000_0000, 0xB3C0_0000),
        ];
        for (x, y, head, tail) in binary32 {
            let result = F32::from_bits(x).augmented_mul_with(F32::from_bits(y), Env::IEEE);
            let (ours_head, ours_tail, _) = bits(result, |value| u64::from(value.to_bits()));
            assert_eq!((ours_head, ours_tail), (head, tail), "{x:#x} * {y:#x}");
        }
    }

    #[test]
    fn sums_keep_far_operands_and_round_ties_toward_zero() {
        let sum = |x: u64, y: u64| {
            bits(
                F64::from_bits(x).augmented_add_with(F64::from_bits(y), Env::IEEE),
                F64::to_bits,
            )
        };
        // 1 plus or minus the smallest subnormal value, an exact tiny tail.
        let tiny = Flags::TINY | Flags::DENORMAL_INPUT;
        assert_eq!(
            sum(0x3FF0_0000_0000_0000, 1),
            (0x3FF0_0000_0000_0000, 1, tiny)
        );
        let below = sum(0x3FF0_0000_0000_0000, 0x8000_0000_0000_0001);
        assert_eq!(below, (0x3FF0_0000_0000_0000, 0x8000_0000_0000_0001, tiny));
        // 1 + 2^-52 + 2^-53 is a tie, and the head keeps the odd value.
        let tie = sum(0x3FF0_0000_0000_0001, 0x3CA0_0000_0000_0000);
        assert_eq!(
            tie,
            (0x3FF0_0000_0000_0001, 0x3CA0_0000_0000_0000, Flags::NONE)
        );
    }

    #[test]
    fn zeros_and_overflows_give_the_head_twice() {
        const NEGATIVE_ZERO: u64 = 1 << 63;
        const MAX: u64 = 0x7FEF_FFFF_FFFF_FFFF;
        const INFINITY: u64 = 0x7FF0_0000_0000_0000;
        let sum = |x: u64, y: u64, env: Env| {
            bits(
                F64::from_bits(x).augmented_add_with(F64::from_bits(y), env),
                F64::to_bits,
            )
        };
        assert_eq!(
            sum(NEGATIVE_ZERO, NEGATIVE_ZERO, Env::IEEE),
            (NEGATIVE_ZERO, NEGATIVE_ZERO, Flags::NONE)
        );
        assert_eq!(sum(0, NEGATIVE_ZERO, Env::IEEE), (0, 0, Flags::NONE));
        assert_eq!(
            sum(0x3FF0_0000_0000_0000, 0xBFF0_0000_0000_0000, Env::IEEE),
            (0, 0, Flags::NONE)
        );
        let overflow = Flags::OVERFLOW | Flags::INEXACT;
        assert_eq!(
            sum(MAX, MAX, Env::IEEE),
            (INFINITY, INFINITY, overflow | Flags::ROUNDED_UP)
        );
        assert_eq!(
            sum(MAX, MAX, Env::IEEE.with_saturate(true)),
            (MAX, MAX, overflow)
        );
    }
}
