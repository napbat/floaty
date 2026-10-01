//! The IEEE 754 remainder and the truncated remainder of the binary formats.

use core::cmp::Ordering;

use super::Layout;
use crate::env::{Env, Flags};
use crate::exact::{self, Unrounded};
use crate::format::internal::Quotient;
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{self, Divisor, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the remainder `left - n * right`. `n` is `left / right`
    /// rounded to an integer as `quotient` says: the IEEE 754 remainder, or
    /// the truncated remainder of C `fmod`.
    ///
    /// The remainder is exact, so the rounding direction, the precision
    /// limit, and flush-to-zero do not apply. A zero remainder has the sign
    /// of `left`. `inf % y` and `x % 0` are invalid. A subnormal remainder
    /// reports `TINY`, as a subnormal result of a rounded operation does.
    pub fn remainder<L: Widen>(left: L, right: L, quotient: Quotient, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&first, &second, env) {
            return Self::exact(value, flags | special);
        }
        match (first, second) {
            (Unpacked::Infinity { .. }, _) | (_, Unpacked::Zero { .. }) => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Zero { .. }, _) => Self::exact(first, flags),
            (
                Unpacked::Finite {
                    negative,
                    exponent,
                    significand,
                },
                Unpacked::Infinity { .. },
            ) => Self::exact_remainder(
                Unrounded {
                    negative,
                    exponent,
                    significand: significand.resize::<L::Double>(),
                    sticky: false,
                },
                env,
                flags,
            ),
            (
                Unpacked::Finite {
                    negative,
                    exponent,
                    significand,
                },
                Unpacked::Finite {
                    exponent: divisor_exponent,
                    significand: divisor_significand,
                    ..
                },
            ) => Self::finite_remainder(
                (negative, exponent, significand),
                (divisor_exponent, divisor_significand),
                quotient,
                env,
                flags,
            ),
            _ => unreachable!("the special cases handle every NaN and unsupported operand"),
        }
    }

    /// Returns the remainder of two finite values: the dividend as its sign,
    /// exponent, and significand, and the divisor as its exponent and
    /// significand.
    fn finite_remainder<L: Widen>(
        (negative, exponent, significand): (bool, i32, L),
        (divisor_exponent, divisor_significand): (i32, L),
        quotient: Quotient,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let dividend = significand.resize::<L::Double>();
        // The remainder of the magnitudes, the divisor at the same weight, the
        // weight of their lowest bits, and the lowest bit of the quotient.
        let (rest, divisor, lowest, odd) = if exponent >= divisor_exponent {
            // The dividend is significand * 2^steps at the weight of the
            // divisor. Its residue modulo twice the divisor gives the
            // remainder and the lowest bit of the quotient. With
            // significand * 2^steps = q * divisor + r, the residue is
            // (q mod 2) * divisor + r.
            let steps =
                u32::try_from(exponent - divisor_exponent).expect("the difference is not negative");
            // Twice the divisor has PRECISION + 1 bits, and every storage type
            // holds PRECISION + 2 bits.
            let modulus = divisor_significand.shl(1);
            let residue = Self::shifted_mod(significand, steps, modulus);
            let odd = residue.compare(&divisor_significand) != Ordering::Less;
            let rest = if odd {
                residue.sub(divisor_significand)
            } else {
                residue
            };
            (
                rest.resize::<L::Double>(),
                divisor_significand.resize::<L::Double>(),
                divisor_exponent,
                odd,
            )
        } else {
            let shift =
                u32::try_from(divisor_exponent - exponent).expect("the divisor exponent is larger");
            if shift > Self::PRECISION {
                // |left| < 2^(exponent + PRECISION) <= |right| / 2, so n is 0.
                let value = Unrounded {
                    negative,
                    exponent,
                    significand: dividend,
                    sticky: false,
                };
                return Self::exact_remainder(value, env, flags);
            }
            // The shifted divisor has at most 2 * PRECISION bits.
            let divisor = divisor_significand.resize::<L::Double>().shl(shift);
            let (quotient, rest) = limbs::divide(dividend, divisor);
            (rest, divisor, exponent, quotient.bit(0))
        };
        // `rest` is the remainder of the truncated quotient. When n rounds up,
        // the remainder is `rest - divisor`, with the other sign.
        let rounded_up = quotient.rounds_up(&rest, &divisor, odd);
        let magnitude = if rounded_up { divisor.sub(rest) } else { rest };
        if magnitude.is_zero() {
            return Self::exact(Unpacked::zero(negative), flags);
        }
        let value = Unrounded {
            negative: negative != rounded_up,
            exponent: lowest,
            significand: magnitude,
            sticky: false,
        };
        Self::exact_remainder(value, env, flags)
    }

    /// Returns `value * 2^exponent mod modulus`. The modulus is twice a
    /// divisor significand, so it has at most `PRECISION + 1` bits.
    ///
    /// The product of two residues needs the library division of 128 bits
    /// when the precision has 32 to 62 bits, as in binary64. Such a modulus
    /// reduces each product with its reciprocal instead, when the exponent
    /// needs a few squarings. It cut the remainder of binary64 operands
    /// `EMAX / 2` apart from 172 to 90 ns. A narrower precision divides
    /// natively, which is faster than a reciprocal.
    fn shifted_mod<L: Widen>(value: L, exponent: u32, modulus: L) -> L {
        if L::BITS == 64 && Self::PRECISION >= 32 && exponent >= RECIPROCAL_EXPONENT {
            let shifted = shifted_mod_limb(value.limb(0), exponent, modulus.limb(0));
            return L::ZERO.with_limb(0, shifted);
        }
        multiply_mod(
            limbs::divide(value, modulus).1,
            power_of_two_mod(exponent, modulus),
            modulus,
        )
    }

    /// Encodes a nonzero remainder, which the format holds exactly. The
    /// rounding routine gives the canonical form and reports `TINY` for a
    /// subnormal remainder.
    fn exact_remainder<L: Widen>(
        value: Unrounded<L::Double>,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let exact_behavior = env.for_exact_result();
        let (rounded, round_flags) =
            exact::round::<L::Double, L, Self, Env>(&value, exact_behavior);
        debug_assert!(
            !round_flags.contains(Flags::INEXACT),
            "the remainder is exact"
        );
        (Self::encode(rounded), flags | round_flags)
    }
}

/// Returns `left * right mod modulus`. Both factors are below the modulus.
fn multiply_mod<L: Widen>(left: L, right: L, modulus: L) -> L {
    let (_, rest) = limbs::divide(left.widening_mul(right), modulus.resize::<L::Double>());
    rest.resize()
}

/// The smallest exponent that takes four squarings. From there on, the
/// reciprocal of a modulus of one limb repays its cost. Below it, the
/// remainder divides at each step.
const RECIPROCAL_EXPONENT: u32 = 8;

/// Returns `value * 2^exponent mod modulus` for a modulus of one limb, with
/// each product reduced by the reciprocal of the modulus. Twice a residue
/// fits a limb.
fn shifted_mod_limb(value: u64, exponent: u32, modulus: u64) -> u64 {
    let divisor = Divisor::new(modulus);
    let multiply =
        |left: [u64; 1], right: [u64; 1]| [multiply_mod_limb(left[0], right[0], divisor)];
    let power = power_of_two(exponent, [modulus], multiply);
    multiply_mod_limb(divisor.divide_limb(0, value).1, power[0], divisor)
}

/// Returns `left * right mod divisor` for factors below a divisor of one
/// limb. The product is below the divisor times 2^64, so one step of the
/// division by the reciprocal gives the remainder, without the library
/// division of 128 bits.
fn multiply_mod_limb(left: u64, right: u64, divisor: Divisor) -> u64 {
    let [low, high] = limbs::split_u128(u128::from(left) * u128::from(right));
    divisor.divide_limb(high, low).1
}

/// Barrett's reduction modulo a fixed modulus of `width` bits: the quotient
/// of a product comes from two products with `factor = floor(4^width /
/// modulus)`, not from a long division. The factor costs one long division.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Barrett<L> {
    modulus: L,
    factor: L,
    width: u32,
}

impl<L: Widen> Barrett<L> {
    /// Returns the reduction for `modulus`, or `None` when its factor does
    /// not fit `L`. The factor has at most `width + 2` bits.
    fn new(modulus: L) -> Option<Self> {
        let width = modulus.bit_length();
        let power = L::Double::ZERO.with_bit(2 * width);
        let (factor, _) = limbs::divide(power, modulus.resize());
        (factor.bit_length() <= L::BITS).then(|| Self {
            modulus,
            factor: factor.resize(),
            width,
        })
    }

    /// Returns `left * right mod modulus`. Both factors are below the modulus,
    /// so the product is below `4^width`, and the estimated quotient is at
    /// most two below the quotient.
    fn multiply(&self, left: L, right: L) -> L {
        let product = left.widening_mul(right);
        let top: L = product.shr(self.width - 1).resize();
        let quotient: L = top.widening_mul(self.factor).shr(self.width + 1).resize();
        let modulus = self.modulus.resize::<L::Double>();
        let mut rest = product.sub(quotient.widening_mul(self.modulus));
        while rest.compare(&modulus) != Ordering::Less {
            rest = rest.sub(modulus);
        }
        rest.resize()
    }
}

/// The number of squarings from which Barrett's reduction repays its long
/// division. With fewer, the remainder divides at each step. For binary128,
/// a Barrett step saved about 21 ns and the factor cost about 190 ns in the
/// benchmark, so the two meet near 9 squarings.
const BARRETT_SQUARINGS: u32 = 10;

/// Returns `2^exponent mod modulus` by square and multiply, so the cost grows
/// with the bit length of the exponent, not with the exponent. The modulus
/// is at least 2, and twice a residue fits `L`.
///
/// A modulus of two limbs, as in binary128 and x87 extended precision, takes
/// Barrett's reduction when the exponent needs many squarings. It cut the
/// remainder of binary128 operands `EMAX / 2` apart from 390 to 308 ns. A
/// modulus of one limb divides natively, and a longer modulus divided as fast
/// as Barrett's two products in the benchmark.
fn power_of_two_mod<L: Widen>(exponent: u32, modulus: L) -> L {
    let squarings = u32::BITS - exponent.leading_zeros();
    if squarings >= BARRETT_SQUARINGS && (65..=128).contains(&modulus.bit_length()) {
        if let Some(barrett) = Barrett::new(modulus) {
            return power_of_two(exponent, modulus, |left, right| {
                barrett.multiply(left, right)
            });
        }
    }
    power_of_two(exponent, modulus, |left, right| {
        multiply_mod(left, right, modulus)
    })
}

/// Returns `2^exponent mod modulus` by square and multiply, with `multiply`
/// as the product modulo the modulus.
fn power_of_two<L: Widen>(exponent: u32, modulus: L, multiply: impl Fn(L, L) -> L) -> L {
    let mut power = L::ZERO.with_bit(0);
    for position in (0..u32::BITS - exponent.leading_zeros()).rev() {
        power = multiply(power, power);
        if (exponent >> position) & 1 == 1 {
            power = power.shl(1);
            if power.compare(&modulus) != Ordering::Less {
                power = power.sub(modulus);
            }
        }
    }
    power
}

#[cfg(test)]
mod tests {
    use super::{Barrett, multiply_mod, multiply_mod_limb};
    use crate::env::{Env, Flags};
    use crate::float::F64;
    use crate::limbs::{self, Divisor, Limbs};

    fn remainder(left: u64, right: u64) -> (u64, Flags) {
        let (value, flags) = F64::from_bits(left).remainder_with(F64::from_bits(right), Env::IEEE);
        (value.to_bits(), flags)
    }

    const TWO: u64 = 0x4000_0000_0000_0000;
    const THREE: u64 = 0x4008_0000_0000_0000;
    const FIVE: u64 = 0x4014_0000_0000_0000;
    const SEVEN: u64 = 0x401C_0000_0000_0000;
    const ONE: u64 = 0x3FF0_0000_0000_0000;
    const MINUS_ONE: u64 = 0xBFF0_0000_0000_0000;

    #[test]
    fn the_quotient_rounds_to_nearest_even() {
        assert_eq!(remainder(FIVE, THREE), (MINUS_ONE, Flags::NONE));
        // 7 / 2 = 3.5 rounds to 4, and 5 / 2 = 2.5 rounds to 2.
        assert_eq!(remainder(SEVEN, TWO), (MINUS_ONE, Flags::NONE));
        assert_eq!(remainder(FIVE, TWO), (ONE, Flags::NONE));
        // 2^100 = 1 (mod 3), far past the width of the significands.
        assert_eq!(remainder(0x4630_0000_0000_0000, THREE), (ONE, Flags::NONE));
        // -4 % 2 is -0.
        assert_eq!(
            remainder(0xC010_0000_0000_0000, TWO),
            (0x8000_0000_0000_0000, Flags::NONE)
        );
    }

    #[test]
    fn the_truncated_quotient_rounds_toward_zero() {
        let truncated = |left: u64, right: u64| {
            let (value, flags) =
                F64::from_bits(left).truncated_remainder_with(F64::from_bits(right), Env::IEEE);
            (value.to_bits(), flags)
        };
        // 5 / 3 truncates to 1, and 7 / 2 = 3.5 truncates to 3.
        assert_eq!(truncated(FIVE, THREE), (TWO, Flags::NONE));
        assert_eq!(truncated(SEVEN, TWO), (ONE, Flags::NONE));
        // The remainder has the sign of the dividend, whatever the sign of
        // the divisor.
        assert_eq!(truncated(FIVE, THREE | (1 << 63)), (TWO, Flags::NONE));
        assert_eq!(
            truncated(FIVE | (1 << 63), THREE),
            (TWO | (1 << 63), Flags::NONE)
        );
        // 2^100 = 1 (mod 3), and the quotient parity does not matter.
        assert_eq!(truncated(0x4630_0000_0000_0000, THREE), (ONE, Flags::NONE));
        // A dividend below the divisor is its own remainder.
        assert_eq!(truncated(TWO, THREE), (TWO, Flags::NONE));
    }

    #[test]
    fn special_operands_follow_ieee_754() {
        let infinity = 0x7FF0_0000_0000_0000;
        assert_eq!(remainder(FIVE, infinity), (FIVE, Flags::NONE));
        assert_eq!(
            remainder(infinity, FIVE),
            (0x7FF8_0000_0000_0000, Flags::INVALID)
        );
        assert_eq!(remainder(FIVE, 0), (0x7FF8_0000_0000_0000, Flags::INVALID));
        // The smallest subnormal is its own remainder by 1.
        assert_eq!(remainder(1, ONE), (1, Flags::DENORMAL_INPUT | Flags::TINY));
    }

    #[test]
    fn reciprocal_products_match_the_division() {
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // The moduli of 33 to 64 bits at both ends, one above a power of two,
        // and random moduli of each width.
        let fixed = [(1_u64 << 32) + 1, u64::MAX, (1 << 63) | 1, (1 << 53) | 1];
        let random: [u64; 32] = core::array::from_fn(|index| {
            let width = 33 + u32::try_from(index).expect("an index fits a u32");
            (next() >> (64 - width)) | (1 << (width - 1))
        });
        for modulus in fixed.into_iter().chain(random) {
            let divisor = Divisor::new(modulus);
            let edges = [0, 1, modulus - 1, modulus - 2];
            let lefts: [u64; 40] = core::array::from_fn(|_| next() % modulus);
            for left in edges.into_iter().chain(lefts) {
                for right in edges.into_iter().chain(lefts.into_iter().take(8)) {
                    assert_eq!(
                        [multiply_mod_limb(left, right, divisor)],
                        multiply_mod([left], [right], [modulus]),
                        "{left:x} * {right:x} mod {modulus:x}"
                    );
                }
            }
        }
    }

    #[test]
    fn barrett_products_match_the_division() {
        // 2^126 has the factor 2^128, which does not fit two limbs.
        assert_eq!(Barrett::new([0_u64, 1 << 62]), None);
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        // The smallest modulus of two limbs, all ones at 127 bits, one above
        // a power of two, and random moduli.
        let random: [[u64; 2]; 20] =
            core::array::from_fn(|_| [next(), (next() >> (next() % 63 + 1)) | 1]);
        let fixed = [[1_u64, 1], [u64::MAX, u64::MAX >> 1], [1, 1 << 62]];
        for modulus in fixed.into_iter().chain(random) {
            let barrett = Barrett::new(modulus).expect("the factor fits two limbs");
            let below = |value: [u64; 2]| limbs::divide(value, modulus).1;
            let one = [1_u64, 0];
            let edges = [[0_u64, 0], one, modulus.sub(one), modulus.sub([2, 0])];
            let lefts: [[u64; 2]; 50] = core::array::from_fn(|_| below([next(), next()]));
            let rights: [[u64; 2]; 5] = core::array::from_fn(|_| below([next(), next()]));
            for left in edges.into_iter().chain(lefts) {
                for right in edges.into_iter().chain(rights) {
                    assert_eq!(
                        barrett.multiply(left, right),
                        multiply_mod(left, right, modulus),
                        "{left:x?} * {right:x?} mod {modulus:x?}"
                    );
                }
            }
        }
    }
}
