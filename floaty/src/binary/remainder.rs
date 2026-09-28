//! The IEEE 754 remainder of the binary formats.

use core::cmp::Ordering;

use super::{Layout, Unpacked};
use crate::env::{Env, Flags};
use crate::exact::{self, Unrounded};
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the IEEE 754 remainder `left - n * right`. `n` is the integer
    /// nearest `left / right`, and the even one at a tie.
    ///
    /// The remainder is exact, so the rounding direction, the precision
    /// limit, and flush-to-zero do not apply. A zero remainder has the sign
    /// of `left`. `inf % y` and `x % 0` are invalid. A subnormal remainder
    /// reports `TINY`, as a subnormal result of a rounded operation does.
    pub fn remainder<L: Widen>(left: L, right: L, env: &Env) -> (L, Flags) {
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
            let residue = multiply_mod(
                limbs::divide(significand, modulus).1,
                power_of_two_mod(steps, modulus),
                modulus,
            );
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
        // Round the quotient to nearest even: past half of the divisor, n
        // grows by one and the remainder changes sign.
        let above_half = match rest.shl(1).compare(&divisor) {
            Ordering::Greater => true,
            Ordering::Equal => odd,
            Ordering::Less => false,
        };
        let magnitude = if above_half { divisor.sub(rest) } else { rest };
        if magnitude.is_zero() {
            return Self::exact(Unpacked::zero(negative), flags);
        }
        let value = Unrounded {
            negative: negative != above_half,
            exponent: lowest,
            significand: magnitude,
            sticky: false,
        };
        Self::exact_remainder(value, env, flags)
    }

    /// Encodes a nonzero remainder, which the format holds exactly. The
    /// rounding routine gives the canonical form and reports `TINY` for a
    /// subnormal remainder.
    fn exact_remainder<L: Widen>(
        value: Unrounded<L::Double>,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let exact_behavior = Env {
            precision: None,
            flush_to_zero: false,
            ..*env
        };
        let (rounded, round_flags) =
            exact::round::<L::Double, L>(&value, &Self::TARGET, &exact_behavior);
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

/// Returns `2^exponent mod modulus` by square and multiply, so the cost grows
/// with the bit length of the exponent, not with the exponent. The modulus
/// is at least 2, and twice a residue fits `L`.
fn power_of_two_mod<L: Widen>(exponent: u32, modulus: L) -> L {
    let mut power = L::ZERO.with_bit(0);
    for position in (0..u32::BITS - exponent.leading_zeros()).rev() {
        power = multiply_mod(power, power, modulus);
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
    use crate::env::{Env, Flags};
    use crate::float::F64;

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
}
