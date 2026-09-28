//! Exact values and the one rounding routine that every operation uses.

use crate::binary::Unpacked;
use crate::env::{Env, Flags, Rounding, Tininess};
use crate::limbs::Limbs;

/// An exact value to round: `significand * 2^exponent`, plus a sticky bit.
///
/// `exponent` is the weight of the lowest significand bit. The significand is
/// an integer of any width, and the caller never normalizes it. `sticky`
/// says that the true magnitude is above `significand * 2^exponent` by less
/// than one lowest bit.
///
/// When `sticky` is set, the significand must have at least `p + 2`
/// significant bits, where `p` is the target precision. With fewer bits the
/// value is not known well enough to round correctly. A debug build checks
/// this contract.
///
/// ```
/// use floaty::{Exact, F32, Flags, Rounding};
///
/// // 1 + 2^-24 is halfway between two binary32 values.
/// let halfway = Exact { negative: false, exponent: -24, significand: [(1 << 24) + 1], sticky: false };
/// let (even, flags) = F32::round(halfway, F32::ENV);
/// assert_eq!(even.to_bits(), 0x3F80_0000);
/// assert_eq!(flags, Flags::INEXACT);
/// let (away, _) = F32::round(halfway, Rounding::NearestAway);
/// assert_eq!(away.to_bits(), 0x3F80_0001);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Exact<const N: usize> {
    /// The sign.
    pub negative: bool,
    /// The weight of the lowest significand bit.
    pub exponent: i32,
    /// The significand, least significant limb first.
    pub significand: [u64; N],
    /// `true` when the true magnitude is above the value by less than one
    /// lowest bit.
    pub sticky: bool,
}

/// The parameters of the format that the rounding routine rounds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target {
    /// The precision of the format.
    pub precision: u32,
    /// The exponent of the smallest normal value.
    pub emin: i32,
    /// The exponent of the largest finite value.
    pub emax: i32,
    /// `true` when the format has an infinity.
    pub has_infinity: bool,
    /// `true` when an all-ones significand at `emax` is the NaN, as in `NoInf`.
    pub all_ones_is_nan: bool,
}

impl Target {
    /// Returns the precision that `env` rounds to: the precision limit of the
    /// behavior, when it is below the format precision.
    pub fn precision_in(&self, env: &Env) -> u32 {
        env.precision
            .map_or(self.precision, |limit| limit.get().min(self.precision))
    }
}

/// A value to round, with a significand of any limb type.
///
/// [`Exact`] names its significand by a limb count. Generic engine code knows
/// only the limb type, so it uses this form.
#[derive(Clone, Copy, Debug)]
pub struct Unrounded<L> {
    /// The sign.
    pub negative: bool,
    /// The weight of the lowest significand bit.
    pub exponent: i32,
    /// The significand.
    pub significand: L,
    /// The sticky bit.
    pub sticky: bool,
}

/// The bits of a value at one rounding position.
#[derive(Clone, Copy)]
struct Cut<L> {
    /// The bits at and above the rounding position.
    kept: L,
    /// The first bit below the rounding position.
    round: bool,
    /// Any bit below the round bit, or the sticky bit.
    rest: bool,
}

/// Cuts `significand * 2^exponent` at the bit of weight `lowest`.
///
/// The kept bits must fit in `Out`.
fn cut<In: Limbs, Out: Limbs>(value: &Unrounded<In>, lowest: i64) -> Cut<Out> {
    let shift = lowest - i64::from(value.exponent);
    if shift <= 0 {
        let left = u32::try_from(-shift).expect("a left shift keeps the value inside the target");
        return Cut {
            kept: value.significand.resize::<Out>().shl(left),
            round: false,
            rest: value.sticky,
        };
    }
    match u32::try_from(shift) {
        Ok(shift) if shift <= In::BITS => Cut {
            kept: value.significand.shr(shift).resize(),
            round: value.significand.bit(shift - 1),
            rest: value.significand.any_below(shift - 1) || value.sticky,
        },
        _ => Cut {
            kept: Out::ZERO,
            round: false,
            rest: !value.significand.is_zero() || value.sticky,
        },
    }
}

/// Applies a rounding direction to a cut. Returns the rounded bits and `true`
/// when the magnitude grew.
fn apply<L: Limbs>(cut: Cut<L>, negative: bool, rounding: Rounding) -> (L, bool) {
    let inexact = cut.round || cut.rest;
    let increment = match rounding {
        Rounding::NearestEven => cut.round && (cut.rest || cut.kept.bit(0)),
        Rounding::NearestAway => cut.round,
        Rounding::TowardPositive => inexact && !negative,
        Rounding::TowardNegative => inexact && negative,
        Rounding::TowardZero => false,
        Rounding::ToOdd => {
            return if inexact && !cut.kept.bit(0) {
                (cut.kept.with_bit(0), true)
            } else {
                (cut.kept, false)
            };
        }
    };
    if increment {
        (cut.kept.increment(), true)
    } else {
        (cut.kept, false)
    }
}

/// Rounds an exact value to a format.
///
/// Returns the rounded value in the canonical form that the encoder takes,
/// and the flags. `Out` must hold the format precision plus one bit, because
/// a carry out of the rounded bits happens before the shift that removes it.
/// Every storage type has at least two bits more than its precision.
pub fn round<In: Limbs, Out: Limbs>(
    value: &Unrounded<In>,
    target: &Target,
    env: &Env,
) -> (Unpacked<Out>, Flags) {
    let negative = value.negative;
    let width = value.significand.bit_length();
    if width == 0 {
        debug_assert!(!value.sticky, "a sticky bit needs a nonzero significand");
        return (Unpacked::Zero { negative }, Flags::NONE);
    }
    let precision = target.precision_in(env);
    debug_assert!(
        !value.sticky || width >= precision + 2,
        "a sticky value has at least p + 2 significant bits"
    );
    let p = i64::from(precision);
    let emin = i64::from(target.emin);
    let top = i64::from(value.exponent) + i64::from(width) - 1;
    let normal_lowest = top - (p - 1);
    let mut lowest = normal_lowest.max(emin - (p - 1));

    let first = cut::<In, Out>(value, lowest);
    let inexact = first.round || first.rest;
    let (mut kept, rounded_up) = apply(first, negative, env.rounding);
    if kept.bit_length() > precision {
        kept = kept.shr(1);
        lowest += 1;
    }

    let tiny = match env.tininess {
        Tininess::BeforeRounding => top < emin,
        Tininess::AfterRounding => {
            top < emin - 1
                || (top == emin - 1 && {
                    let (unbounded, _) =
                        apply(cut::<In, Out>(value, normal_lowest), negative, env.rounding);
                    unbounded.bit_length() <= precision
                })
        }
    };
    let mut flags = Flags::NONE;
    if tiny {
        flags |= Flags::TINY;
        if env.flush_to_zero {
            return (
                Unpacked::Zero { negative },
                flags | Flags::UNDERFLOW | Flags::INEXACT,
            );
        }
    }
    if inexact {
        flags |= Flags::INEXACT;
        if tiny {
            flags |= Flags::UNDERFLOW;
        }
    }
    if kept.is_zero() {
        return (Unpacked::Zero { negative }, flags);
    }
    if rounded_up {
        flags |= Flags::ROUNDED_UP;
    }

    let length = kept.bit_length();
    let result_top = lowest + i64::from(length) - 1;
    let emax = i64::from(target.emax);
    let nan_pattern = target.all_ones_is_nan
        && result_top == emax
        && precision == target.precision
        && kept == Out::ones(precision);
    if result_top > emax || nan_pattern {
        return overflow(negative, precision, target, env);
    }
    (normalize(negative, kept, lowest, target), flags)
}

/// The integer that a value rounds to.
#[derive(Clone, Copy, Debug)]
pub struct Integral<L> {
    /// The magnitude of the integer.
    pub magnitude: L,
    /// `true` when the integer differs from the value.
    pub inexact: bool,
    /// `true` when the magnitude of the integer is above the magnitude of the
    /// value.
    pub rounded_up: bool,
}

/// Rounds a value to an integer in the direction of `rounding`, with the same
/// rounding step as [`round`]. The sticky bit of `value` must be clear.
///
/// `Out` must hold the integer part of the value plus one bit, for the carry
/// of the rounding.
pub fn round_to_integer<In: Limbs, Out: Limbs>(
    value: &Unrounded<In>,
    rounding: Rounding,
) -> Integral<Out> {
    debug_assert!(!value.sticky, "the value is exact");
    let first = cut::<In, Out>(value, 0);
    let inexact = first.round || first.rest;
    let (magnitude, rounded_up) = apply(first, value.negative, rounding);
    Integral {
        magnitude,
        inexact,
        rounded_up,
    }
}

/// Returns the canonical form of `kept * 2^lowest`.
fn normalize<L: Limbs>(negative: bool, kept: L, lowest: i64, target: &Target) -> Unpacked<L> {
    let length = kept.bit_length();
    let full = i64::from(target.precision);
    let subnormal_lowest = i64::from(target.emin) - (full - 1);
    let top = lowest + i64::from(length) - 1;
    let exponent = if top >= i64::from(target.emin) {
        top - (full - 1)
    } else {
        subnormal_lowest
    };
    let shift =
        u32::try_from(lowest - exponent).expect("the canonical exponent is not above the value");
    Unpacked::Finite {
        negative,
        exponent: i32::try_from(exponent).expect("a finite exponent fits an i32"),
        significand: kept.shl(shift),
    }
}

/// Returns the result of an overflow.
fn overflow<L: Limbs>(
    negative: bool,
    precision: u32,
    target: &Target,
    env: &Env,
) -> (Unpacked<L>, Flags) {
    let flags = Flags::OVERFLOW | Flags::INEXACT;
    let to_infinity = match env.rounding {
        Rounding::NearestEven | Rounding::NearestAway => true,
        Rounding::TowardPositive => !negative,
        Rounding::TowardNegative => negative,
        Rounding::TowardZero | Rounding::ToOdd => false,
    };
    if to_infinity {
        if target.has_infinity {
            return (Unpacked::Infinity { negative }, flags | Flags::ROUNDED_UP);
        }
        if !env.saturate {
            return (
                Unpacked::Nan {
                    negative,
                    signaling: false,
                    payload: L::ZERO,
                },
                flags,
            );
        }
    }
    (largest(negative, precision, target), flags)
}

/// Returns the largest finite value at `precision` bits.
pub fn largest<L: Limbs>(negative: bool, precision: u32, target: &Target) -> Unpacked<L> {
    let full = target.precision;
    let significand = if target.all_ones_is_nan && precision == full {
        L::ones(full).shr(1).shl(1)
    } else {
        L::ones(precision).shl(full - precision)
    };
    let exponent = target.emax - i32::try_from(full - 1).expect("a precision fits an i32");
    Unpacked::Finite {
        negative,
        exponent,
        significand,
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use crate::env::{Env, Flags, Rounding, Tininess};
    use crate::exact::Exact;
    use crate::float::{F8E4M3, F8E4M3Fnuz, F16, F32, F80, F512};

    fn exact<const N: usize>(negative: bool, exponent: i32, significand: [u64; N]) -> Exact<N> {
        Exact {
            negative,
            exponent,
            significand,
            sticky: false,
        }
    }

    /// 1 + 2^-24: halfway between 1.0 and the next binary32 value.
    const HALFWAY: Exact<1> = Exact {
        negative: false,
        exponent: -24,
        significand: [(1 << 24) + 1],
        sticky: false,
    };

    #[test]
    fn every_direction_at_a_halfway_value() {
        let cases = [
            (Rounding::NearestEven, 0x3F80_0000, Flags::INEXACT),
            (
                Rounding::NearestAway,
                0x3F80_0001,
                Flags::INEXACT | Flags::ROUNDED_UP,
            ),
            (
                Rounding::TowardPositive,
                0x3F80_0001,
                Flags::INEXACT | Flags::ROUNDED_UP,
            ),
            (Rounding::TowardNegative, 0x3F80_0000, Flags::INEXACT),
            (Rounding::TowardZero, 0x3F80_0000, Flags::INEXACT),
            (
                Rounding::ToOdd,
                0x3F80_0001,
                Flags::INEXACT | Flags::ROUNDED_UP,
            ),
        ];
        for (rounding, bits, flags) in cases {
            let (value, reported) = F32::round(HALFWAY, rounding);
            assert_eq!((value.to_bits(), reported), (bits, flags), "{rounding:?}");
        }
        let negative = Exact {
            negative: true,
            ..HALFWAY
        };
        assert_eq!(
            F32::round(negative, Rounding::TowardNegative).0.to_bits(),
            0xBF80_0001
        );
        assert_eq!(
            F32::round(negative, Rounding::TowardPositive).0.to_bits(),
            0xBF80_0000
        );
    }

    #[test]
    fn nearest_even_keeps_an_odd_value_below_halfway_with_sticky() {
        // 1 + 2^-23 + 2^-25, sticky: just above a quarter step; nearest is 1 + 2^-23.
        let value = Exact {
            negative: false,
            exponent: -25,
            significand: [(1 << 25) + 5],
            sticky: true,
        };
        let (result, flags) = F32::round(value, Rounding::NearestEven);
        assert_eq!((result.to_bits(), flags), (0x3F80_0001, Flags::INEXACT));
    }

    #[test]
    fn a_carry_moves_to_the_next_binade() {
        // 2 - 2^-25 rounds up to 2.0.
        let value = exact(false, -25, [(1 << 26) - 1]);
        let (result, flags) = F32::round(value, Rounding::NearestEven);
        assert_eq!(
            (result.to_bits(), flags),
            (0x4000_0000, Flags::INEXACT | Flags::ROUNDED_UP)
        );
    }

    #[test]
    fn exact_values_of_any_width_round_trip() {
        let (one, flags) = F32::round(exact(false, -3, [8]), Env::IEEE);
        assert_eq!((one.to_bits(), flags), (0x3F80_0000, Flags::NONE));
        let mut wide = [0_u64; 8];
        wide[7] = 1 << 10;
        let (from_wide, _) = F16::round(exact(false, -458, wide), Env::IEEE);
        assert_eq!(from_wide.to_bits(), 0x3C00);
        let (into_wide, flags) = F512::round(exact(true, 0, [3]), Env::IEEE);
        assert_eq!(flags, Flags::NONE);
        assert_eq!(into_wide.to_bits()[7], 0xC000_0080_0000_0000);
        let (zero, flags) = F32::round(exact(true, 5, [0]), Env::IEEE);
        assert_eq!((zero.to_bits(), flags), (0x8000_0000, Flags::NONE));
    }

    #[test]
    fn overflow_follows_the_direction_and_the_encoding() {
        let huge = exact(false, 128, [1]);
        let cases = [
            (Rounding::NearestEven, 0x7F80_0000),
            (Rounding::TowardPositive, 0x7F80_0000),
            (Rounding::TowardZero, 0x7F7F_FFFF),
            (Rounding::TowardNegative, 0x7F7F_FFFF),
            (Rounding::ToOdd, 0x7F7F_FFFF),
        ];
        for (rounding, bits) in cases {
            let (value, flags) = F32::round(huge, rounding);
            assert_eq!(value.to_bits(), bits, "{rounding:?}");
            assert!(
                flags.contains(Flags::OVERFLOW | Flags::INEXACT),
                "{rounding:?}"
            );
        }
        // OCP E4M3 has no infinity. 480 rounds to the NaN encoding, so it overflows.
        let above = exact(false, 5, [15]);
        assert_eq!(F8E4M3::round(above, Env::IEEE).0.to_bits(), 0x7F);
        let saturate = Env::IEEE.with_saturate(true);
        assert_eq!(F8E4M3::round(above, saturate).0.to_bits(), 0x7E);
        let (largest, flags) = F8E4M3::round(exact(false, 5, [14]), Env::IEEE);
        assert_eq!((largest.to_bits(), flags), (0x7E, Flags::NONE));
        assert_eq!(
            F8E4M3Fnuz::round(exact(true, 8, [1]), Env::IEEE)
                .0
                .to_bits(),
            0x80
        );
        assert_eq!(
            F8E4M3Fnuz::round(exact(true, 8, [1]), saturate).0.to_bits(),
            0xFF
        );
    }

    /// 2^-126 - 2^-151: below the smallest normal binary32 value, but it rounds
    /// up to it both at the subnormal quantum and with an unbounded exponent.
    const JUST_BELOW_NORMAL: Exact<1> = Exact {
        negative: false,
        exponent: -151,
        significand: [(1 << 25) - 1],
        sticky: false,
    };

    #[test]
    fn tininess_before_and_after_rounding_differ() {
        let after = Env::IEEE;
        let before = Env::IEEE.with_tininess(Tininess::BeforeRounding);
        let (value, flags) = F32::round(JUST_BELOW_NORMAL, after);
        assert_eq!(
            (value.to_bits(), flags),
            (0x0080_0000, Flags::INEXACT | Flags::ROUNDED_UP)
        );
        let (value, flags) = F32::round(JUST_BELOW_NORMAL, before);
        let tiny = Flags::TINY | Flags::UNDERFLOW | Flags::INEXACT | Flags::ROUNDED_UP;
        assert_eq!((value.to_bits(), flags), (0x0080_0000, tiny));
    }

    #[test]
    fn flush_to_zero_uses_the_tininess_rule() {
        let flush = Env::IEEE.with_flush_to_zero(true);
        let (value, flags) = F32::round(JUST_BELOW_NORMAL, flush);
        assert_eq!(value.to_bits(), 0x0080_0000, "not tiny after rounding");
        assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
        let (value, flags) = F32::round(
            JUST_BELOW_NORMAL,
            flush.with_tininess(Tininess::BeforeRounding),
        );
        assert_eq!(
            (value.to_bits(), flags),
            (0, Flags::TINY | Flags::UNDERFLOW | Flags::INEXACT)
        );
        let (value, flags) = F32::round(exact(true, -149, [1]), flush);
        assert_eq!(
            (value.to_bits(), flags),
            (0x8000_0000, Flags::TINY | Flags::UNDERFLOW | Flags::INEXACT)
        );
    }

    #[test]
    fn exact_subnormals_are_tiny_but_do_not_underflow() {
        let (value, flags) = F32::round(exact(false, -149, [3]), Env::IEEE);
        assert_eq!((value.to_bits(), flags), (3, Flags::TINY));
    }

    #[test]
    fn a_tiny_value_can_round_to_zero() {
        let (value, flags) = F32::round(exact(true, -151, [1]), Rounding::TowardZero);
        assert_eq!(
            (value.to_bits(), flags),
            (0x8000_0000, Flags::TINY | Flags::UNDERFLOW | Flags::INEXACT)
        );
        let (value, flags) = F32::round(exact(false, -151, [1]), Rounding::TowardPositive);
        assert_eq!(value.to_bits(), 1);
        assert!(flags.contains(Flags::ROUNDED_UP | Flags::UNDERFLOW));
    }

    #[test]
    fn precision_control_keeps_the_exponent_range() {
        let pc24 = Env::IEEE.with_precision(NonZeroU32::new(24));
        // 1 + 2^-24 rounds to 1.0 at 24 bits, in the x87 format.
        let (value, flags) = F80::round(HALFWAY, pc24);
        assert_eq!(
            (value.to_bits(), flags),
            (0x3FFF_8000_0000_0000_0000, Flags::INEXACT)
        );
        // 1 + 2^-23 is exact at 24 bits.
        let (value, flags) = F80::round(exact(false, -23, [(1 << 23) + 1]), pc24);
        assert_eq!(
            (value.to_bits(), flags),
            (0x3FFF_8000_0100_0000_0000, Flags::NONE)
        );
        // A tiny value keeps 24 bits of quantum below 2^emin: 2^-16382 * 2^-23 is exact.
        let (value, flags) = F80::round(exact(false, -16405, [1]), pc24);
        assert_eq!(
            (value.to_bits(), flags),
            (0x0000_0000_0100_0000_0000, Flags::TINY)
        );
        // 2^-16382 * 2^-24 is below that quantum and rounds to even: zero.
        let (value, flags) = F80::round(exact(false, -16406, [1]), pc24);
        assert_eq!(
            (value.to_bits(), flags),
            (0, Flags::TINY | Flags::UNDERFLOW | Flags::INEXACT)
        );
    }

    #[test]
    fn saturating_overflow_of_the_largest_values() {
        // x87 at 24 bits: the largest value has 24 significand bits.
        let pc24 = Env::IEEE
            .with_precision(NonZeroU32::new(24))
            .with_rounding(Rounding::TowardZero);
        let (value, flags) = F80::round(exact(false, 16384, [1]), pc24);
        assert_eq!(value.to_bits(), 0x7FFE_FFFF_FF00_0000_0000);
        assert_eq!(flags, Flags::OVERFLOW | Flags::INEXACT);
    }
}
