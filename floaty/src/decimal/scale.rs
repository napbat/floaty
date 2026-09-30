//! Scaling by a power of 10, the next value up or down, and the quantum
//! operations of IEEE 754-2019 section 5.3.2, for the decimal formats.

use super::digits::digit_count;
use super::round;
use super::{DecimalLayout, Wide};
use crate::env::{Env, Flags};
use crate::exact::Unrounded;
use crate::format::internal::Step;
use crate::format::{DecimalEncoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::rounding::Integral;
use crate::unpacked::{SCALE_LIMIT, Unpacked};

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Returns `value * 10^scale`, rounded. The preferred exponent is the
    /// exponent of `value` plus `scale`.
    pub fn scale_b<L: Widen>(bits: L, scale: i32, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        let scale = i64::from(scale.clamp(-SCALE_LIMIT, SCALE_LIMIT));
        match value {
            Unpacked::Nan { .. } => {
                let (nan, special) =
                    nan::special_unary(&value, env).expect("a NaN has a special result");
                Self::exact(nan, flags | special)
            }
            Unpacked::Zero { negative, exponent } => {
                Self::exact(Self::zero(negative, i64::from(exponent) + scale), flags)
            }
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let exponent = i64::from(exponent) + scale;
                let scaled = Unrounded {
                    negative,
                    exponent: i32::try_from(exponent).expect("a scaled exponent fits an i32"),
                    significand: significand.resize::<Wide<L>>(),
                    sticky: false,
                };
                Self::finish(&scaled, exponent, *env, flags)
            }
            Unpacked::Infinity { .. } => Self::exact(value, flags),
            Unpacked::Unsupported => unreachable!("a decimal format has no unsupported encoding"),
        }
    }

    /// Returns the next value up or down, as IEEE 754 `nextUp` and `nextDown`
    /// do. A finite result has every digit, so the least possible exponent.
    /// The step does not round, so the precision limit and flush-to-zero do
    /// not apply.
    pub fn next<L: Limbs>(bits: L, step: Step, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        if let Some((nan, special)) = nan::special_unary(&value, env) {
            return Self::exact(nan, flags | special);
        }
        let result = match step {
            Step::Up => Self::successor(value).unwrap_or(Unpacked::Infinity { negative: false }),
            Step::Down => Self::successor(value.negate())
                .map_or(Unpacked::Infinity { negative: true }, Unpacked::negate),
        };
        Self::exact(result, flags)
    }

    /// Returns the least number above a number, with every digit, or `None`
    /// above the largest finite value.
    fn successor<L: Limbs>(value: Unpacked<L>) -> Option<Unpacked<L>> {
        let (lowest, highest) = Self::exponent_range();
        let narrow =
            |exponent: i64| i32::try_from(exponent).expect("an exponent of the format fits an i32");
        let (negative, coefficient, exponent) = match value {
            Unpacked::Zero { .. } => {
                return Some(Unpacked::Finite {
                    negative: false,
                    exponent: narrow(lowest),
                    significand: L::ZERO.with_bit(0),
                });
            }
            Unpacked::Infinity { negative: false } => return Some(value),
            Unpacked::Infinity { negative: true } => {
                return Some(round::largest(true, Self::PRECISION, &Self::TARGET));
            }
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => (negative, significand, i64::from(exponent)),
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the caller handles every NaN")
            }
        };
        let (coefficient, mut exponent) = round::least_exponent(
            coefficient,
            digit_count(&coefficient),
            exponent,
            &Self::TARGET,
        );
        let one = L::ZERO.with_bit(0);
        if negative {
            let next = coefficient.sub(one);
            if next.is_zero() {
                return Some(Unpacked::Zero {
                    negative: true,
                    exponent: narrow(exponent),
                });
            }
            if digit_count(&next) < Self::PRECISION && exponent > lowest {
                // Below a power of ten, the spacing shrinks tenfold.
                let full = limbs::multiply_small(coefficient, 10).sub(one);
                return Some(Unpacked::Finite {
                    negative: true,
                    exponent: narrow(exponent - 1),
                    significand: full,
                });
            }
            return Some(Unpacked::Finite {
                negative: true,
                exponent: narrow(exponent),
                significand: next,
            });
        }
        let mut next = coefficient.add(one);
        if digit_count(&next) > Self::PRECISION {
            next = limbs::divide(next, limbs::from_u128(10)).0;
            exponent += 1;
        }
        if exponent > highest {
            return None;
        }
        Some(Unpacked::Finite {
            negative: false,
            exponent: narrow(exponent),
            significand: next,
        })
    }

    /// Returns `left` with the exponent of `right`, as IEEE 754 `quantize`
    /// does. Digits that the new exponent drops round in the direction of the
    /// behavior and signal inexact. A coefficient that the new exponent makes
    /// longer than the precision signals invalid. The operation signals no
    /// underflow or overflow.
    pub fn quantize<L: Widen>(left: L, right: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let first = Self::operand(left, env, &mut flags);
        let second = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&first, &second, env) {
            return Self::exact(value, flags | special);
        }
        let exponent = match (first, second) {
            (Unpacked::Infinity { .. }, Unpacked::Infinity { .. }) => {
                return Self::exact(first, flags);
            }
            (Unpacked::Infinity { .. }, _) | (_, Unpacked::Infinity { .. }) => {
                return Self::exact(default_nan(env), flags | Flags::INVALID);
            }
            (_, Unpacked::Zero { exponent, .. } | Unpacked::Finite { exponent, .. }) => exponent,
            _ => unreachable!("the special cases handle every NaN"),
        };
        match first {
            Unpacked::Zero { negative, .. } => {
                Self::exact(Unpacked::Zero { negative, exponent }, flags)
            }
            Unpacked::Finite {
                negative,
                exponent: from,
                significand,
            } => {
                let value = Unrounded {
                    negative,
                    exponent: from,
                    significand: significand.resize::<Wide<L>>(),
                    sticky: false,
                };
                let drop = i64::from(exponent) - i64::from(from);
                let coefficient: Wide<L> = if drop >= 0 {
                    let integral: Integral<Wide<L>> =
                        round::round_digits(&value, drop, env.rounding);
                    flags |= integral.flags();
                    integral.magnitude
                } else {
                    let shift = u32::try_from(-drop).unwrap_or(u32::MAX);
                    if shift > Self::PRECISION {
                        return Self::exact(default_nan(env), flags | Flags::INVALID);
                    }
                    (0..shift).fold(value.significand, |coefficient, _| {
                        limbs::multiply_small(coefficient, 10)
                    })
                };
                if digit_count(&coefficient) > Self::PRECISION {
                    return Self::exact(default_nan(env), flags | Flags::INVALID);
                }
                let result = if coefficient.is_zero() {
                    Unpacked::Zero { negative, exponent }
                } else {
                    Unpacked::Finite {
                        negative,
                        exponent,
                        significand: coefficient.resize(),
                    }
                };
                Self::exact(result, flags)
            }
            _ => unreachable!("the special cases handle every NaN and infinity"),
        }
    }

    /// Returns `true` when both values have the same exponent, or both are
    /// infinities, or both are NaNs, as IEEE 754 `sameQuantum` does.
    pub fn same_quantum<L: Limbs>(left: L, right: L) -> bool {
        match (Self::decode(left), Self::decode(right)) {
            (
                Unpacked::Zero { exponent: a, .. } | Unpacked::Finite { exponent: a, .. },
                Unpacked::Zero { exponent: b, .. } | Unpacked::Finite { exponent: b, .. },
            ) => a == b,
            (Unpacked::Infinity { .. }, Unpacked::Infinity { .. })
            | (Unpacked::Nan { .. }, Unpacked::Nan { .. }) => true,
            _ => false,
        }
    }

    /// Returns the quantum of a value, `1 * 10^exponent`, as IEEE 754
    /// `quantum` does. An infinity gives positive infinity.
    pub fn quantum<L: Limbs>(bits: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        match value {
            Unpacked::Nan { .. } => {
                let (nan, special) =
                    nan::special_unary(&value, env).expect("a NaN has a special result");
                Self::exact(nan, flags | special)
            }
            Unpacked::Infinity { .. } => Self::exact(Unpacked::Infinity { negative: false }, flags),
            Unpacked::Zero { exponent, .. } | Unpacked::Finite { exponent, .. } => {
                let quantum = Unpacked::Finite {
                    negative: false,
                    exponent,
                    significand: L::ZERO.with_bit(0),
                };
                Self::exact(quantum, flags)
            }
            Unpacked::Unsupported => unreachable!("a decimal format has no unsupported encoding"),
        }
    }

    /// Returns the exponent of the leading digit of a value, as an integral
    /// value, as IEEE 754 `logB` does. A zero gives negative infinity and
    /// signals divide-by-zero, and an infinity gives positive infinity.
    pub fn log_b<L: Limbs>(bits: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        match value {
            Unpacked::Nan { .. } => {
                let (nan, special) =
                    nan::special_unary(&value, env).expect("a NaN has a special result");
                Self::exact(nan, flags | special)
            }
            Unpacked::Infinity { .. } => Self::exact(Unpacked::Infinity { negative: false }, flags),
            Unpacked::Zero { .. } => Self::exact(
                Unpacked::Infinity { negative: true },
                flags | Flags::DIVIDE_BY_ZERO,
            ),
            Unpacked::Finite {
                exponent,
                significand,
                ..
            } => {
                let top = i64::from(exponent) + i64::from(digit_count(&significand)) - 1;
                let magnitude = u128::from(top.unsigned_abs());
                let result = if magnitude == 0 {
                    Unpacked::Zero {
                        negative: false,
                        exponent: 0,
                    }
                } else {
                    Unpacked::Finite {
                        negative: top < 0,
                        exponent: 0,
                        significand: limbs::from_u128(magnitude),
                    }
                };
                Self::exact(result, flags)
            }
            Unpacked::Unsupported => unreachable!("a decimal format has no unsupported encoding"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
    use crate::float::{D64Bid, Decoded};

    #[test]
    fn quantum_reads_a_subnormal_operand_through_daz() {
        let subnormal = D64Bid::from_bits(0x0000_0000_0000_0005);
        let daz = Env::IEEE.with_denormals_are_zero(true);
        let (quantum, flags) = subnormal.quantum_with(daz);
        assert_eq!(flags, Flags::DENORMAL_INPUT);
        assert_eq!(
            quantum.decode::<1>(),
            Decoded::Finite {
                negative: false,
                exponent: -398,
                significand: [1]
            }
        );
    }
}
