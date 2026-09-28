//! Conversion of a finite decimal value to a binary format, correctly rounded.

use super::Layout;
use crate::env::{Env, Flags};
use crate::exact::Unrounded;
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{self, Limbs};
use crate::radix;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Converts `significand * 10^exponent`, correctly rounded.
    ///
    /// A nonnegative exponent gives the exact integer `significand * 5^q` at
    /// binary exponent `q`. A negative exponent `-k` gives the quotient of
    /// `significand * 2^t` by `5^k`, with `t` large enough for the precision
    /// plus two bits, and a sticky bit from the remainder. The binary rounding
    /// routine rounds either value once.
    pub(super) fn from_decimal<In: Limbs, Out: Limbs>(
        value: &Unrounded<In>,
        env: &Env,
    ) -> (Out, Flags) {
        let bits = i64::from(value.significand.bit_length())
            + i64::from(Self::PRECISION)
            + 4
            + radix::power_of_five_bits(value.exponent.unsigned_abs());
        if radix::fits_small(bits) {
            Self::from_decimal_in::<{ radix::SMALL }, In, Out>(value, env)
        } else {
            Self::from_decimal_in::<{ radix::LARGE }, In, Out>(value, env)
        }
    }

    /// Converts `significand * 10^exponent` in numbers of `N` limbs.
    fn from_decimal_in<const N: usize, In: Limbs, Out: Limbs>(
        value: &Unrounded<In>,
        env: &Env,
    ) -> (Out, Flags) {
        let exponent = value.exponent;
        let power = radix::power_of_five::<N>(exponent.unsigned_abs());
        let exact = if exponent >= 0 {
            Unrounded {
                negative: value.negative,
                exponent,
                significand: radix::multiply(&power, &value.significand),
                sticky: false,
            }
        } else {
            let coefficient: [u64; N] = value.significand.resize();
            let room = Self::PRECISION + 3 + power.bit_length();
            let shift = room.saturating_sub(coefficient.bit_length());
            let (quotient, remainder) = limbs::divide(coefficient.shl(shift), power);
            Unrounded {
                negative: value.negative,
                exponent: exponent - i32::try_from(shift).expect("the shift fits an i32"),
                significand: quotient,
                sticky: !remainder.is_zero(),
            }
        };
        Self::round(&exact, env)
    }
}

#[cfg(test)]
mod tests {
    use crate::float::{D64Bid, D128Bid, Decoded, F16, F32, F64};
    use crate::{Env, Exact, Flags, Rounding};

    fn decimal(coefficient: u64, exponent: i32) -> D64Bid {
        let exact = Exact {
            negative: false,
            exponent,
            significand: [coefficient],
            sticky: false,
        };
        D64Bid::round(exact, Env::IEEE).0
    }

    #[test]
    fn decimal_values_convert_correctly_rounded() {
        let to_f64 = |value: D64Bid| -> F64 { value.convert() };
        assert_eq!(to_f64(decimal(1, 0)).to_bits(), 0x3FF0_0000_0000_0000);
        assert_eq!(to_f64(decimal(5, -1)).to_bits(), 0x3FE0_0000_0000_0000);
        assert_eq!(to_f64(decimal(1, -1)).to_bits(), 0x3FB9_9999_9999_999A);
        assert_eq!(to_f64(decimal(1, 300)).to_bits(), 0x7E37_E43C_8800_759C);
        let (small, flags) = decimal(1, -1).convert_with::<F32>(Rounding::TowardZero);
        assert_eq!(small.to_bits(), 0x3DCC_CCCC);
        assert_eq!(flags, Flags::INEXACT);
        let (large, flags) = decimal(1, 5).convert_with::<F16>(Env::IEEE);
        assert!(large.is_infinite());
        assert!(flags.contains(Flags::OVERFLOW));
    }

    #[test]
    fn extreme_exponents_round_to_zero_or_overflow() {
        let tiny = D128Bid::round(
            Exact {
                negative: true,
                exponent: -6176,
                significand: [1_u64, 0],
                sticky: false,
            },
            Env::IEEE,
        )
        .0;
        let (zero, flags) = tiny.convert_with::<F64>(Env::IEEE);
        assert_eq!(zero.to_bits(), 0x8000_0000_0000_0000);
        assert!(flags.contains(Flags::UNDERFLOW | Flags::INEXACT));
        let huge: D128Bid = D128Bid::round(
            Exact {
                negative: false,
                exponent: 6111,
                significand: [9_999_999_999_999_999_999, 0],
                sticky: false,
            },
            Env::IEEE,
        )
        .0;
        assert!(huge.convert::<F64>().is_infinite());
        assert!(matches!(huge.decode::<2>(), Decoded::Finite { .. }));
    }
}
