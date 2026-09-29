//! Scaling by a power of two, and the next value up or down, for the binary
//! formats.

use super::{Layout, Unpacked};
use crate::env::{Env, Flags};
use crate::exact::{self, Unrounded};
use crate::format::internal::Step;
use crate::format::{Encoding, Storage, Width};
use crate::limbs::Limbs;
use crate::nan;

/// The largest scale that `scale_b` applies. Every format overflows or
/// underflows at a smaller scale, and the limit keeps every exponent inside
/// an `i32`.
const SCALE_LIMIT: i32 = 1 << 30;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns `value * 2^scale`, rounded, as IEEE 754 `scaleB` does.
    pub fn scale_b<L: Limbs>(bits: L, scale: i32, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        match value {
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let scaled = Unrounded {
                    negative,
                    exponent: exponent + scale.clamp(-SCALE_LIMIT, SCALE_LIMIT),
                    significand,
                    sticky: false,
                };
                let (rounded, round_flags) = exact::round::<L, L, Self, Env>(&scaled, *env);
                (Self::encode(rounded), flags | round_flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                let (nan, special) = nan::special(&value, &Unpacked::zero(false), env)
                    .expect("a NaN or an unsupported operand has a special result");
                Self::exact(nan, flags | special)
            }
            Unpacked::Zero { .. } | Unpacked::Infinity { .. } => Self::exact(value, flags),
        }
    }

    /// Returns the next value up or down, as IEEE 754 `nextUp` and `nextDown`
    /// do.
    ///
    /// The step does not round, so the precision limit and flush-to-zero do
    /// not apply. In a format without an infinity, the value past the largest
    /// finite value is the NaN. When the behavior saturates, it is the largest
    /// finite value itself. Only a signaling NaN or an unsupported operand
    /// signals invalid. A subnormal operand reports `DENORMAL_INPUT`.
    pub fn next<L: Limbs>(bits: L, step: Step, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        if let Some((nan, special)) = nan::special(&value, &Unpacked::zero(false), env) {
            return Self::exact(nan, flags | special);
        }
        let past_largest = |negative| {
            if env.saturate && !Self::TARGET.has_infinity {
                exact::largest(negative, Self::PRECISION, &Self::TARGET)
            } else {
                Self::infinity(negative, env)
            }
        };
        let result = match step {
            Step::Up => Self::successor(value).unwrap_or_else(|| past_largest(false)),
            Step::Down => {
                Self::successor(value.negate()).map_or_else(|| past_largest(true), Unpacked::negate)
            }
        };
        Self::exact(result, flags)
    }

    /// Returns the least number above a number, or `None` above the largest
    /// finite value.
    fn successor<L: Limbs>(value: Unpacked<L>) -> Option<Unpacked<L>> {
        let one = L::ZERO.with_bit(0);
        let smallest_exponent = Self::EMIN - Self::SHIFT;
        let largest = exact::largest(false, Self::PRECISION, &Self::TARGET);
        let next = match value {
            Unpacked::Zero { .. } => Unpacked::Finite {
                negative: false,
                exponent: smallest_exponent,
                significand: one,
            },
            Unpacked::Infinity { negative: false } => value,
            Unpacked::Infinity { negative: true } => largest.negate(),
            Unpacked::Finite {
                negative: false, ..
            } if value == largest => return None,
            Unpacked::Finite {
                negative: false,
                exponent,
                significand,
            } => {
                let next = significand.add(one);
                if next.bit_length() > Self::PRECISION {
                    Unpacked::Finite {
                        negative: false,
                        exponent: exponent + 1,
                        significand: next.shr(1),
                    }
                } else {
                    Unpacked::Finite {
                        negative: false,
                        exponent,
                        significand: next,
                    }
                }
            }
            Unpacked::Finite {
                negative: true,
                exponent,
                significand,
            } => {
                let next = significand.sub(one);
                if next.is_zero() {
                    Unpacked::zero(true)
                } else if next.bit_length() < Self::PRECISION && exponent > smallest_exponent {
                    // Below a power of two, the spacing halves: the next value
                    // has a full significand at the next lower exponent.
                    Unpacked::Finite {
                        negative: true,
                        exponent: exponent - 1,
                        significand: next.shl(1).with_bit(0),
                    }
                } else {
                    Unpacked::Finite {
                        negative: true,
                        exponent,
                        significand: next,
                    }
                }
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the caller handles every NaN and unsupported operand")
            }
        };
        Some(next)
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
    use crate::float::{F8E4M3Fn, F8E4M3Fnuz, F32, F80};

    fn f32(bits: u32) -> F32 {
        F32::from_bits(bits)
    }

    #[test]
    fn scaling_rounds_at_the_ends_of_the_range() {
        let one = f32(0x3F80_0000);
        assert_eq!(one.scale_b(10).to_bits(), 0x4480_0000);
        assert_eq!(one.scale_b(-149).to_bits(), 1);
        let (zero, flags) = one.scale_b_with(-150, Env::IEEE);
        assert_eq!(
            (zero.to_bits(), flags),
            (0, Flags::UNDERFLOW | Flags::INEXACT | Flags::TINY)
        );
        assert_eq!(one.scale_b(i32::MAX).to_bits(), 0x7F80_0000);
        assert_eq!(one.scale_b(i32::MIN).to_bits(), 0);
    }

    #[test]
    fn next_up_and_next_down_step_through_every_boundary() {
        let up = |bits| f32(bits).next_up().to_bits();
        let down = |bits| f32(bits).next_down().to_bits();
        assert_eq!(up(0x8000_0000), 1);
        assert_eq!(up(0x8000_0001), 0x8000_0000);
        assert_eq!(up(0x7F7F_FFFF), 0x7F80_0000);
        assert_eq!(up(0xFF80_0000), 0xFF7F_FFFF);
        assert_eq!(up(0x3F7F_FFFF), 0x3F80_0000);
        assert_eq!(down(0x3F80_0000), 0x3F7F_FFFF);
        assert_eq!(down(0xFF7F_FFFF), 0xFF80_0000);
        assert_eq!(down(0), 0x8000_0001);
        assert_eq!(down(0x0080_0000), 0x007F_FFFF);
        let (nan, flags) = f32(0x7F80_0001).next_up_with(Env::IEEE);
        assert_eq!((nan.to_bits(), flags), (0x7FC0_0001, Flags::INVALID));

        // A format without an infinity: the NaN, or the largest value when
        // saturating.
        let largest = F8E4M3Fn::from_bits(0x7E);
        assert_eq!(largest.next_up().to_bits(), 0x7F);
        assert_eq!(
            largest
                .next_up_with(Env::IEEE.with_saturate(true))
                .0
                .to_bits(),
            0x7E
        );
        assert_eq!(F8E4M3Fn::from_bits(0xFE).next_down().to_bits(), 0xFF);
        // The precision limit does not shrink the largest value.
        let limited = Env::IEEE
            .with_saturate(true)
            .with_precision(core::num::NonZeroU32::new(2));
        assert_eq!(largest.next_up_with(limited).0.to_bits(), 0x7E);
        assert_eq!(
            F8E4M3Fn::from_bits(0xFE)
                .next_down_with(limited)
                .0
                .to_bits(),
            0xFE
        );
        assert_eq!(F8E4M3Fnuz::from_bits(0x81).next_up().to_bits(), 0);

        // An x87 pseudo-denormal steps from the value of the normal encoding.
        let pseudo = F80::from_bits(0x0000_8000_0000_0000_0000);
        assert_eq!(pseudo.next_up().to_bits(), 0x0001_8000_0000_0000_0001);
        assert_eq!(pseudo.next_down().to_bits(), 0x0000_7FFF_FFFF_FFFF_FFFF);
    }
}
