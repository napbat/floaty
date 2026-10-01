//! The algebraic functions of IEEE 754-2019 section 9.2 for the binary
//! formats, correctly rounded: `hypot` and `rSqrt`.

use super::Layout;
use crate::env::{Env, Flags};
use crate::exact::Unrounded;
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::unpacked::{Number, Unpacked};

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns `sqrt(left^2 + right^2)`, rounded once, and the flags.
    pub fn hypot<L: Widen>(left: L, right: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if matches!(x, Unpacked::Unsupported) || matches!(y, Unpacked::Unsupported) {
            return Self::exact(default_nan(env), flags | Flags::INVALID);
        }
        // An infinity gives +inf even with a quiet NaN, as IEEE 754-2019
        // section 9.2.1 requires. A signaling NaN gives a NaN first.
        let infinite =
            matches!(x, Unpacked::Infinity { .. }) || matches!(y, Unpacked::Infinity { .. });
        if x.is_signaling() || y.is_signaling() || (!infinite && (x.is_nan() || y.is_nan())) {
            let (value, special) = nan::propagate(&x, &y, env);
            return Self::exact(value, flags | special);
        }
        if infinite {
            return Self::exact(Unpacked::Infinity { negative: false }, flags);
        }
        let magnitude = |number: Number<L>| Number {
            negative: false,
            ..number
        };
        match (x.number().map(magnitude), y.number().map(magnitude)) {
            (Some(a), Some(b)) => Self::hypot_finite(a, b, env, flags),
            (Some(number), None) | (None, Some(number)) => {
                let value = Unrounded {
                    negative: false,
                    exponent: number.exponent,
                    significand: number.significand,
                    sticky: false,
                };
                Self::finish(&value, *env, flags)
            }
            (None, None) => Self::exact(Unpacked::zero(false), flags),
        }
    }

    /// Returns the hypotenuse of two positive finite numbers.
    ///
    /// The larger square sits two bits below the top of the padded width, so
    /// the sum has room for its carry. The bits of the smaller square below
    /// the width only set the sticky bit: both squares are positive, so the
    /// truncated sum lies below the true sum by less than one unit. The
    /// radicand then takes 2p + 3 or 2p + 4 bits and an even exponent, so the
    /// root has p + 2 bits.
    fn hypot_finite<L: Widen>(a: Number<L>, b: Number<L>, env: &Env, flags: Flags) -> (L, Flags) {
        let (first, second) = (Self::product(a, a), Self::product(b, b));
        let (big, small) = if first.top() >= second.top() {
            (first, second)
        } else {
            (second, first)
        };
        let padded_bits = i64::from(<L::Padded as Limbs>::BITS);
        let big_width = i64::from(big.significand.bit_length());
        let big_shift = padded_bits - 2 - big_width;
        let low = big.exponent - big_shift;
        let big_part = big
            .significand
            .resize::<L::Padded>()
            .shl(u32::try_from(big_shift).expect("the padded width holds a square"));
        let shift = small.exponent - low;
        let (small_part, mut sticky) = if shift >= 0 {
            let shift = u32::try_from(shift).expect("the smaller square sits below the larger");
            (small.significand.resize::<L::Padded>().shl(shift), false)
        } else {
            let dropped = u32::try_from(-shift).unwrap_or(u32::MAX);
            let part = small.significand.shr(dropped).resize::<L::Padded>();
            (part, small.significand.any_below(dropped))
        };
        let total = big_part.add(small_part);
        let precision = i64::from(Self::PRECISION);
        let mut dropped = i64::from(total.bit_length()) - (2 * precision + 4);
        if (low + dropped) % 2 != 0 {
            dropped += 1;
        }
        let dropped_bits = u32::try_from(dropped).expect("the sum has more bits than the radicand");
        sticky |= total.any_below(dropped_bits);
        let radicand = total.shr(dropped_bits).resize::<L::Double>();
        let (root, inexact) = limbs::square_root_of_double::<L>(radicand);
        // The exponent of the radicand, `low + dropped`, is even, so the root
        // takes exactly half of it.
        let value = Unrounded {
            negative: false,
            exponent: i32::try_from(i64::midpoint(low, dropped))
                .expect("a root exponent fits an i32"),
            significand: root,
            sticky: sticky || inexact,
        };
        Self::finish(&value, *env, flags)
    }

    /// Returns `1 / sqrt(value)`, rounded once, and the flags.
    pub fn reciprocal_sqrt<L: Widen>(value: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(value, env, &mut flags);
        if let Some((result, special)) = nan::special_unary(&x, env) {
            return Self::exact(result, flags | special);
        }
        match x {
            Unpacked::Zero { negative, .. } => {
                Self::exact(Self::infinity(negative, env), flags | Flags::DIVIDE_BY_ZERO)
            }
            Unpacked::Infinity { negative: false } => Self::exact(Unpacked::zero(false), flags),
            Unpacked::Infinity { negative: true } | Unpacked::Finite { negative: true, .. } => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            Unpacked::Finite {
                negative: false,
                exponent,
                significand,
            } => Self::reciprocal_sqrt_finite(exponent, significand, env, flags),
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }

    /// Returns `1 / sqrt(significand * 2^exponent)` of a positive number.
    ///
    /// With an even exponent `e` and `m` of `w` bits, the value is
    /// `sqrt(2^n / m) * 2^-((e + n) / 2)` for an even `n`. The floor of the
    /// root of the floor of `2^n / m` is the floor of the root of `2^n / m`,
    /// and the root is exact only when both steps are. `n` gives the quotient
    /// 2p + 3 or 2p + 4 bits, so the root has p + 2 bits. Two divisions in the
    /// double width give the quotient.
    fn reciprocal_sqrt_finite<L: Widen>(
        exponent: i32,
        significand: L,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let (exponent, significand) = if exponent % 2 == 0 {
            (exponent, significand.resize::<L::Double>())
        } else {
            (exponent - 1, significand.resize::<L::Double>().shl(1))
        };
        let width = significand.bit_length();
        let power_of_two = !significand.any_below(width - 1);
        // The quotient of a power of two is exactly a power of two, one bit
        // longer than the quotient of a larger significand of the same width.
        let base = 2 * Self::PRECISION + 2 + width;
        let shift = if power_of_two {
            base & !1
        } else {
            (base + 1) & !1
        };
        let first = width + Self::PRECISION + 1;
        let second = shift - first;
        let one = L::Double::ZERO.with_bit(0);
        let (high, rest) = limbs::divide(one.shl(first), significand);
        let (low, remainder) = limbs::divide(rest.shl(second), significand);
        let quotient = high.shl(second).add(low);
        let (root, inexact) = limbs::square_root_of_double::<L>(quotient);
        let shift = i32::try_from(shift).expect("a shift fits an i32");
        let value = Unrounded {
            negative: false,
            exponent: -(exponent + shift) / 2,
            significand: root,
            sticky: inexact || !remainder.is_zero(),
        };
        Self::finish(&value, *env, flags)
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
    use crate::float::{F32, F64, Float};
    use crate::format::Binary;

    /// A layout whose precision is two bits below its storage.
    type Tight = Float<Binary<2>, 64>;

    fn hypot(x: u64, y: u64) -> (u64, Flags) {
        let (result, flags) = F64::from_bits(x).hypot_with(F64::from_bits(y), Env::IEEE);
        (result.to_bits(), flags)
    }

    #[test]
    fn hypot_keeps_the_far_smaller_operand_as_a_sticky_bit() {
        // 2^1000 and 2^-1000: the result is 2^1000 and a little more.
        let (large, small) = (0x7E70_0000_0000_0000, 0x0170_0000_0000_0000);
        assert_eq!(hypot(large, small), (large, Flags::INEXACT));
        let up = F64::from_bits(large)
            .hypot_with(F64::from_bits(small), crate::Rounding::TowardPositive);
        assert_eq!(up.0.to_bits(), large + 1);
        // 5 and 12 give 13 exactly.
        assert_eq!(
            hypot(0x4014_0000_0000_0000, 0x4028_0000_0000_0000),
            (0x402A_0000_0000_0000, Flags::NONE)
        );
    }

    #[test]
    fn hypot_of_an_infinity_is_infinite_before_a_quiet_nan() {
        let (infinity, quiet, signaling) = (
            0x7FF0_0000_0000_0000,
            0x7FF8_0000_0000_0001,
            0x7FF0_0000_0000_0001,
        );
        assert_eq!(hypot(quiet, infinity | (1 << 63)), (infinity, Flags::NONE));
        assert_eq!(
            hypot(signaling, infinity),
            (0x7FF8_0000_0000_0001, Flags::INVALID)
        );
        assert_eq!(hypot(1 << 63, 1 << 63), (0, Flags::NONE));
    }

    #[test]
    fn reciprocal_sqrt_of_zeros_and_powers_of_two() {
        let rsqrt = |bits: u32| {
            let (result, flags) = F32::from_bits(bits).reciprocal_sqrt_with(Env::IEEE);
            (result.to_bits(), flags)
        };
        assert_eq!(rsqrt(0x8000_0000), (0xFF80_0000, Flags::DIVIDE_BY_ZERO));
        assert_eq!(rsqrt(0x7F80_0000), (0, Flags::NONE));
        assert_eq!(rsqrt(0xBF80_0000), (0x7FC0_0000, Flags::INVALID));
        // 2^-126 gives 2^63 exactly, and the smallest subnormal value 2^-149
        // gives 2^74.5.
        assert_eq!(rsqrt(0x0080_0000), (0x5F00_0000, Flags::NONE));
        let (root, flags) = rsqrt(1);
        assert_eq!(flags, Flags::INEXACT | Flags::DENORMAL_INPUT);
        assert_eq!(root, 0x64B5_04F3);
        // A precision two bits below the storage takes the longest quotient:
        // the subnormal 0.5 of this layout gives sqrt(2).
        let half = Tight::from_bits(0x1000_0000_0000_0000);
        let (root, flags) = half.reciprocal_sqrt_with(Env::IEEE);
        let flags_expected = Flags::INEXACT | Flags::DENORMAL_INPUT;
        assert_eq!(
            (root.to_bits(), flags),
            (0x2D41_3CCC_FE77_9921, flags_expected)
        );
    }
}
