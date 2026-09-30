//! Rounding to an integral value, and conversion to an integer, for the
//! binary formats.
//!
//! Both operations round with the integer step of the rounding routine, and
//! report `INEXACT` whenever the result differs from the operand. The IEEE 754
//! operations that do not signal inexact, such as `roundToIntegralTiesToEven`,
//! are these operations with `INEXACT` ignored.

use super::Layout;
use crate::env::{Env, Flags};
use crate::exact::{self, Unrounded};
use crate::format::{Encoding, Storage, Width};
use crate::integer::{Integer, Parts, ToInt, fit};
use crate::limbs::Limbs;
use crate::nan;
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Rounds to an integral value in the format, in the rounding direction.
    ///
    /// The integer is exact at the full precision, so the precision limit
    /// and flush-to-zero do not apply. SoftFloat and the x87 `FRNDINT`
    /// instruction also ignore the precision limit. In a format whose
    /// largest finite value is below `2^(PRECISION - 1)`, the integer can
    /// exceed that value. It then overflows by the usual rules.
    pub fn round_to_integral<L: Limbs>(bits: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let value = Self::operand(bits, env, &mut flags);
        let (negative, exponent, significand) = match value {
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } if exponent < 0 => (negative, exponent, significand),
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                let (nan, special) = nan::special_unary(&value, env)
                    .expect("a NaN or an unsupported operand has a special result");
                return Self::exact(nan, flags | special);
            }
            // Zeros, infinities, and values without a fraction.
            number => return Self::exact(number, flags),
        };
        let integral = exact::round_to_integer::<L, L>(
            &Unrounded {
                negative,
                exponent,
                significand,
                sticky: false,
            },
            env.rounding,
        );
        flags |= integral.flags();
        if integral.magnitude.is_zero() {
            return Self::exact(Unpacked::zero(negative), flags);
        }
        // The integer has at most PRECISION + 1 bits. With PRECISION + 1 bits
        // it is a power of two. So it is exact at the full precision, but it
        // can still overflow the exponent range.
        let integer = Unrounded {
            negative,
            exponent: 0,
            significand: integral.magnitude,
            sticky: false,
        };
        // An integer of magnitude 1 or more is never tiny, so flush-to-zero,
        // which the exact behavior also clears, changes nothing here.
        let (rounded, overflow_flags) =
            exact::round::<L, L, Self, Env>(&integer, env.for_exact_result());
        // An overflow gives an infinity, which is above the value, or the NaN
        // or the largest finite value, which are not. So the overflow decides
        // `ROUNDED_UP`.
        let flags = if overflow_flags.contains(Flags::OVERFLOW) {
            flags.difference(Flags::ROUNDED_UP)
        } else {
            flags
        };
        Self::exact(rounded, flags | overflow_flags)
    }

    /// Rounds to an integer of type `I`, in the rounding direction.
    ///
    /// A NaN or an unsupported operand gives [`ToInt::Nan`], and an infinity
    /// or a rounded value outside the range of `I` gives
    /// [`ToInt::OutOfRange`]. Both signal invalid and not inexact.
    pub fn to_int<L: Limbs, I: Integer>(bits: L, env: &Env) -> (ToInt<I>, Flags) {
        let mut flags = Flags::NONE;
        let (negative, exponent, significand) = match Self::operand(bits, env, &mut flags) {
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => (negative, exponent, significand),
            Unpacked::Zero { .. } => {
                let zero = Parts {
                    negative: false,
                    magnitude: [0; 8],
                };
                return (ToInt::Value(I::from_parts(zero)), flags);
            }
            Unpacked::Infinity { negative } => {
                return (ToInt::OutOfRange { negative }, flags | Flags::INVALID);
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                return (ToInt::Nan, flags | Flags::INVALID);
            }
        };
        let top = i64::from(exponent) + i64::from(significand.bit_length()) - 1;
        if top >= i64::from(I::BITS) {
            // The magnitude is at least 2^BITS, above every integer of the
            // type, in every rounding direction.
            return (ToInt::OutOfRange { negative }, flags | Flags::INVALID);
        }
        let value = Unrounded {
            negative,
            exponent,
            significand,
            sticky: false,
        };
        // The integer part is below 2^BITS, so it fits twice the width of a
        // 64-bit integer with its carry, and nine limbs for any integer.
        if I::BITS <= 64 {
            Self::fit_integer::<L, [u64; 2], I>(&value, env, flags)
        } else {
            Self::fit_integer::<L, [u64; 9], I>(&value, env, flags)
        }
    }

    /// Rounds a finite value to an integer in `M`, and returns it when it
    /// fits `I`. The integer part of the value must fit `M` with its carry.
    fn fit_integer<L: Limbs, M: Limbs, I: Integer>(
        value: &Unrounded<L>,
        env: &Env,
        mut flags: Flags,
    ) -> (ToInt<I>, Flags) {
        let negative = value.negative;
        let integral = exact::round_to_integer::<L, M>(value, env.rounding);
        let Some(parts) = fit::<I, M>(negative, &integral.magnitude) else {
            return (ToInt::OutOfRange { negative }, flags | Flags::INVALID);
        };
        flags |= integral.flags();
        (ToInt::Value(I::from_parts(parts)), flags)
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags, Rounding};
    use crate::exact::Exact;
    use crate::float::{F6E2M3Fn, F32, F80, F128, F512, Float};
    use crate::format::Binary;
    use crate::integer::{Int, ToInt, UInt};

    #[test]
    fn a_carry_to_two_to_the_64_is_out_of_range() {
        // 2^64 - 0.5 is halfway between 2^64 - 1 and 2^64. The carry of the
        // rounding to 2^64 needs the second limb of the integer buffer.
        let halfway = Exact {
            negative: false,
            exponent: -1,
            significand: [u64::MAX, 1],
            sticky: false,
        };
        let (value, _) = F128::round(halfway, Env::IEEE);
        assert_eq!(
            value.to_int_with::<u64>(Rounding::TiesToEven),
            (ToInt::OutOfRange { negative: false }, Flags::INVALID)
        );
        assert_eq!(
            value.to_int_with::<u64>(Rounding::TowardZero),
            (ToInt::Value(u64::MAX), Flags::INEXACT)
        );
    }

    fn round(bits: u32, rounding: Rounding) -> (u32, Flags) {
        let (value, flags) = F32::from_bits(bits).round_to_integral_with(rounding);
        (value.to_bits(), flags)
    }

    #[test]
    fn rounding_to_an_integral_value_follows_the_direction() {
        // 2.5 and -0.5.
        assert_eq!(
            round(0x4020_0000, Rounding::TiesToEven),
            (0x4000_0000, Flags::INEXACT)
        );
        assert_eq!(
            round(0x4020_0000, Rounding::TiesToAway),
            (0x4040_0000, Flags::INEXACT | Flags::ROUNDED_UP)
        );
        assert_eq!(
            round(0x4020_0000, Rounding::ToOdd),
            (0x4040_0000, Flags::INEXACT | Flags::ROUNDED_UP)
        );
        assert_eq!(
            round(0xBF00_0000, Rounding::TiesToEven),
            (0x8000_0000, Flags::INEXACT)
        );
        assert_eq!(
            round(0x4B00_0001, Rounding::TowardZero),
            (0x4B00_0001, Flags::NONE)
        );
        // Precision control does not apply: 2^40 + 1 keeps its low bit.
        let wide = F80::from_bits(0x4027_8000_0000_8000_0000);
        let limited = Env::IEEE.with_precision(core::num::NonZeroU32::new(24));
        assert_eq!(
            wide.round_to_integral_with(limited).0.to_bits(),
            wide.to_bits()
        );
    }

    #[test]
    fn an_integer_above_a_small_range_overflows() {
        // Binary<3> at width 8 has 5 bits of precision and a largest value of
        // 15.5, so 15.5 rounds to 16, past the range.
        let largest = Float::<Binary<3>, 8>::from_bits(0x6F);
        let (infinity, flags) = largest.round_to_integral_with(Rounding::TiesToEven);
        assert_eq!(
            (infinity.to_bits(), flags),
            (0x70, Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP)
        );
        let (truncated, flags) = largest.round_to_integral_with(Rounding::TowardZero);
        assert_eq!((truncated.to_bits(), flags), (0x6E, Flags::INEXACT));
        // A saturated overflow gives the largest value, which is not above
        // the operand.
        let saturate = Env::IEEE.with_saturate(true);
        let (saturated, flags) = largest.round_to_integral_with(saturate);
        assert_eq!(
            (saturated.to_bits(), flags),
            (0x6F, Flags::OVERFLOW | Flags::INEXACT)
        );
        let (mx, flags) = F6E2M3Fn::from_bits(0x1F).round_to_integral_with(Rounding::TiesToEven);
        assert_eq!(
            (mx.to_bits(), flags),
            (0x1F, Flags::OVERFLOW | Flags::INEXACT)
        );
    }

    #[test]
    fn conversion_to_an_integer_checks_the_exact_width() {
        let two_31 = F32::from_bits(0x4F00_0000);
        assert_eq!(
            two_31.to_int_with::<i32>(Env::IEEE),
            (ToInt::OutOfRange { negative: false }, Flags::INVALID)
        );
        assert_eq!((-two_31).to_int::<i32>(), ToInt::Value(i32::MIN));
        assert_eq!(two_31.to_int::<u32>(), ToInt::Value(1 << 31));
        assert_eq!(
            F32::from_bits(0xBF00_0000).to_int_with::<u8>(Env::IEEE),
            (ToInt::Value(0), Flags::INEXACT)
        );
        let two_23 = F32::from_bits(0x4B00_0000);
        assert!(matches!(
            two_23.to_int::<Int<24>>(),
            ToInt::OutOfRange { negative: false }
        ));
        assert_eq!(
            (-two_23).to_int::<Int<24>>(),
            ToInt::Value(Int::<24>::from_bits(0x80_0000))
        );
        assert_eq!(
            F32::from_bits(0x7FC0_0000).to_int_with::<i64>(Env::IEEE),
            (ToInt::Nan, Flags::INVALID)
        );
        let huge = F512::from_bits([0, 0, 0, 0, 0, 0, 0, 0x7000_0000_0000_0000]);
        assert!(matches!(
            huge.to_int::<UInt<512>>(),
            ToInt::OutOfRange { negative: false }
        ));
    }
}
