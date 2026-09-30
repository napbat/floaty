//! Rounding to an integral value, conversion to an integer, and the IEEE 754
//! and truncated remainders, for the decimal formats.

use core::cmp::Ordering;

use super::digits::{digit_count, power_of_ten};
use super::round;
use super::{DecimalLayout, Wide};
use crate::env::{Env, Flags};
use crate::exact::{Integral, Unrounded};
use crate::format::internal::Quotient;
use crate::format::{DecimalEncoding, Storage, Width};
use crate::integer::{Integer, ToInt};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

/// The largest adjusted exponent of a value that can fit a 512-bit integer:
/// 2^512 is below 10^155.
const LARGEST_TOP: i64 = 154;

/// Returns `left * right mod modulus`. Both factors are below the modulus,
/// and the square of the modulus fits `L`.
fn multiply_mod<L: Limbs>(left: L, right: L, modulus: L) -> L {
    limbs::divide(limbs::multiply_fit(left, right), modulus).1
}

/// Returns `10^exponent mod modulus` by square and multiply, so the cost grows
/// with the bit length of the exponent.
fn power_of_ten_mod<L: Limbs>(exponent: u32, modulus: L) -> L {
    let ten = limbs::divide(power_of_ten::<L>(1), modulus).1;
    let mut power = limbs::divide(power_of_ten::<L>(0), modulus).1;
    for position in (0..u32::BITS - exponent.leading_zeros()).rev() {
        power = multiply_mod(power, power, modulus);
        if (exponent >> position) & 1 == 1 {
            power = multiply_mod(power, ten, modulus);
        }
    }
    power
}

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Rounds to an integral value in the direction of the behavior. The
    /// preferred exponent is the larger of the exponent and zero, as IEEE
    /// 754-2019 Table 5.1 gives it. The result is exact, so the precision
    /// limit and flush-to-zero do not apply.
    pub fn round_to_integral<L: Limbs>(bits: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        match value {
            Unpacked::Nan { .. } => {
                let (nan, special) =
                    nan::special_unary(&value, env).expect("a NaN has a special result");
                Self::exact(nan, flags | special)
            }
            Unpacked::Zero { negative, exponent } if exponent < 0 => Self::exact(
                Unpacked::Zero {
                    negative,
                    exponent: 0,
                },
                flags,
            ),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } if exponent < 0 => {
                let value = Unrounded {
                    negative,
                    exponent,
                    significand,
                    sticky: false,
                };
                let integral: Integral<L> = round::round_to_integer(&value, env.rounding);
                flags |= integral.flags();
                let result = if integral.magnitude.is_zero() {
                    Unpacked::Zero {
                        negative,
                        exponent: 0,
                    }
                } else {
                    Unpacked::Finite {
                        negative,
                        exponent: 0,
                        significand: integral.magnitude,
                    }
                };
                Self::exact(result, flags)
            }
            // Zeros, infinities, and values without a fraction.
            number => Self::exact(number, flags),
        }
    }

    /// Rounds to an integer of type `I`, in the direction of the behavior.
    /// The rules are those of the binary formats.
    pub fn to_int<L: Limbs, I: Integer>(bits: L, env: &Env) -> (ToInt<I>, Flags) {
        let mut flags = Flags::NONE;
        let (negative, exponent, significand) = match Self::operand(bits, env, &mut flags) {
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => (negative, exponent, significand),
            special => {
                let (result, special_flags) =
                    ToInt::special(&special).expect("a value that is not finite has a result");
                return (result, flags | special_flags);
            }
        };
        let top = i64::from(exponent) + i64::from(digit_count(&significand)) - 1;
        if top > LARGEST_TOP {
            return (ToInt::OutOfRange { negative }, flags | Flags::INVALID);
        }
        let integral: Integral<[u64; 9]> = if exponent >= 0 {
            // The integer has at most 155 digits, which fit 9 limbs.
            let shift = u32::try_from(exponent).expect("the exponent is small");
            let scaled: [u64; 9] = significand.resize();
            let product = (0..shift).fold(scaled, |value, _| limbs::multiply_small(value, 10));
            Integral {
                magnitude: product,
                inexact: false,
                rounded_up: false,
            }
        } else {
            let value = Unrounded {
                negative,
                exponent,
                significand,
                sticky: false,
            };
            round::round_to_integer(&value, env.rounding)
        };
        let (result, integer_flags) = ToInt::from_integral(negative, &integral);
        (result, flags | integer_flags)
    }

    /// Returns the remainder `left - n * right`, where `n` is `left / right`
    /// rounded to an integer as `quotient` says: the IEEE 754 remainder, or
    /// the truncated remainder of C `fmod` and decNumber `remainder`. The
    /// result is exact. Its preferred exponent is the smaller exponent of the
    /// operands.
    ///
    /// IEEE 754 has no limit on `n`. decNumber signals invalid when `n` has
    /// more digits than the precision; this function follows IEEE 754.
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
            (
                Unpacked::Finite {
                    negative,
                    exponent,
                    significand,
                },
                Unpacked::Infinity { .. },
            ) => {
                // The result is the dividend. The rounding routine reports
                // `TINY` for a subnormal dividend.
                let value = Unrounded {
                    negative,
                    exponent,
                    significand: significand.resize::<Wide<L>>(),
                    sticky: false,
                };
                Self::exact_result(&value, i64::from(exponent), env, flags)
            }
            (_, Unpacked::Infinity { .. }) => Self::exact(first, flags),
            (
                Unpacked::Zero { negative, exponent },
                Unpacked::Finite {
                    exponent: divisor, ..
                },
            ) => Self::exact(
                Self::zero(negative, i64::from(exponent.min(divisor))),
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
                (negative, exponent, significand.resize()),
                (divisor_exponent, divisor_significand.resize()),
                quotient,
                env,
                flags,
            ),
            _ => unreachable!("the special cases handle every NaN"),
        }
    }

    /// Returns the remainder of two finite values: the dividend as its sign,
    /// exponent, and coefficient, and the divisor as its exponent and
    /// coefficient.
    fn finite_remainder<L: Widen>(
        (negative, exponent, coefficient): (bool, i32, Wide<L>),
        (divisor_exponent, divisor_coefficient): (i32, Wide<L>),
        quotient: Quotient,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let lowest = exponent.min(divisor_exponent);
        let (rest, divisor, odd) = if exponent >= divisor_exponent {
            // The dividend is coefficient * 10^steps at the weight of the
            // divisor. Its residue modulo twice the divisor gives the
            // remainder and the lowest bit of the quotient.
            let steps =
                u32::try_from(exponent - divisor_exponent).expect("the difference is not negative");
            let modulus = limbs::multiply_small(divisor_coefficient, 2);
            let residue = multiply_mod(
                limbs::divide(coefficient, modulus).1,
                power_of_ten_mod(steps, modulus),
                modulus,
            );
            let odd = residue.compare(&divisor_coefficient) != Ordering::Less;
            let rest = if odd {
                residue.sub(divisor_coefficient)
            } else {
                residue
            };
            (rest, divisor_coefficient, odd)
        } else {
            let top = |exponent: i32, coefficient: &Wide<L>| {
                i64::from(exponent) + i64::from(digit_count(coefficient)) - 1
            };
            if top(exponent, &coefficient) < top(divisor_exponent, &divisor_coefficient) - 1 {
                // |left| < |right| / 10, so n is 0 and the result is left.
                let value = Unrounded {
                    negative,
                    exponent,
                    significand: coefficient,
                    sticky: false,
                };
                return Self::exact_result(&value, i64::from(lowest), env, flags);
            }
            let shift = u32::try_from(divisor_exponent - exponent).expect("the shift is small");
            let divisor = limbs::multiply_fit(divisor_coefficient, power_of_ten(shift));
            let (quotient, rest) = limbs::divide(coefficient, divisor);
            (rest, divisor, quotient.bit(0))
        };
        // `rest` is the remainder of the truncated quotient. When n rounds up,
        // the remainder is `rest - divisor`, with the other sign.
        let rounded_up = quotient.rounds_up(&rest, &divisor, odd);
        let magnitude = if rounded_up { divisor.sub(rest) } else { rest };
        if magnitude.is_zero() {
            return Self::exact(Self::zero(negative, i64::from(lowest)), flags);
        }
        let value = Unrounded {
            negative: negative != rounded_up,
            exponent: lowest,
            significand: magnitude,
            sticky: false,
        };
        Self::exact_result(&value, i64::from(lowest), env, flags)
    }

    /// Encodes an exact result through the rounding routine, which gives its
    /// canonical form and reports `TINY` for a subnormal value. The precision
    /// limit and flush-to-zero do not apply.
    fn exact_result<L: Widen>(
        value: &Unrounded<Wide<L>>,
        preferred: i64,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let exact_behavior = env.for_exact_result();
        let (result, round_flags) = Self::finish(value, preferred, exact_behavior, flags);
        debug_assert!(!round_flags.contains(Flags::INEXACT), "the result is exact");
        (result, round_flags)
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
    use crate::float::D64Bid;

    /// Returns the truncated and the IEEE 754 remainder of two decimal64
    /// encodings, with the flags.
    fn remainders(left: u64, right: u64) -> [(u64, Flags); 2] {
        let (x, y) = (D64Bid::from_bits(left), D64Bid::from_bits(right));
        let (truncated, truncated_flags) = x.truncated_remainder_with(y, Env::IEEE);
        let (nearest, nearest_flags) = x.remainder_with(y, Env::IEEE);
        [
            (truncated.to_bits(), truncated_flags),
            (nearest.to_bits(), nearest_flags),
        ]
    }

    const THREE: u64 = 0x31C0_0000_0000_0003;
    const EIGHT: u64 = 0x31C0_0000_0000_0008;

    #[test]
    fn the_truncated_quotient_keeps_the_sign_of_the_dividend() {
        // 8 = 2 * 3 + 2 and 8 = 3 * 3 - 1.
        assert_eq!(
            remainders(EIGHT, THREE),
            [
                (0x31C0_0000_0000_0002, Flags::NONE),
                (0xB1C0_0000_0000_0001, Flags::NONE)
            ]
        );
        assert_eq!(
            remainders(EIGHT | (1 << 63), THREE)[0],
            (0xB1C0_0000_0000_0002, Flags::NONE)
        );
        // 8.0 remainder 3 is 2.0, at the smaller exponent.
        assert_eq!(
            remainders(0x31A0_0000_0000_0050, THREE)[0],
            (0x31A0_0000_0000_0014, Flags::NONE)
        );
    }

    #[test]
    fn a_subnormal_dividend_over_infinity_is_tiny() {
        // 1E-398 is subnormal. Before the fix, the remainder by an infinity
        // returned it without `TINY`, unlike the binary formats.
        let expected = (0x0000_0000_0000_0001, Flags::TINY | Flags::DENORMAL_INPUT);
        assert_eq!(
            remainders(0x0000_0000_0000_0001, 0x7800_0000_0000_0000),
            [expected, expected]
        );
    }
}
