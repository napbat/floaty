//! The decimal rounding routine: an exact coefficient times a power of 10,
//! rounded to a decimal format with the preferred exponent of its operation.
//!
//! The routine shares the choice of the direction with the binary routine,
//! [`exact::rounds_up`](crate::exact::rounds_up). IEEE 754 detects the
//! tininess of a decimal result before rounding, so the routine ignores the
//! tininess rule of the behavior.

use core::cmp::Ordering;

use super::digits::{self, digit_count, last_digit, power_of_ten};
use crate::env::{Behavior, Env, Flags, Rounding};
use crate::exact::{self, Dropped, Integral, Unrounded};
use crate::limbs::{self, Limbs};
use crate::unpacked::Unpacked;

/// The parameters of the decimal format that the routine rounds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecimalTarget {
    /// The precision in digits.
    pub precision: u32,
    /// The adjusted exponent of the smallest normal value.
    pub emin: i32,
    /// The adjusted exponent of the largest finite value.
    pub emax: i32,
}

impl DecimalTarget {
    /// Returns the precision that `env` rounds to: the precision limit of the
    /// behavior, in digits, when it is below the format precision.
    pub fn precision_in(&self, env: &Env) -> u32 {
        env.precision
            .map_or(self.precision, |limit| limit.get().min(self.precision))
    }

    /// Returns the smallest and the largest exponent of a coefficient of
    /// `precision` digits.
    fn exponents(&self, precision: u32) -> (i64, i64) {
        let digits = i64::from(precision);
        (
            i64::from(self.emin) - digits + 1,
            i64::from(self.emax) - digits + 1,
        )
    }
}

/// A type whose constant is the target of the decimal rounding routine.
///
/// Each decimal format is its own target type. So each format gets its own
/// copy of [`round`], with the parameters of the format as constants.
pub trait DecimalRoundingTarget {
    /// The parameters of the format.
    const TARGET: DecimalTarget;
}

/// Returns an exponent that fits an `i32`.
fn narrow(exponent: i64) -> i32 {
    i32::try_from(exponent).expect("a decimal exponent of a format fits an i32")
}

/// Rounds an exact value to a decimal format.
///
/// `value` is `significand * 10^exponent`, and its sticky bit says that the
/// true magnitude is above it by less than one unit of its last digit. With
/// the sticky bit set, the significand must have more digits than the
/// precision, so that the rounding sees the first dropped digit.
///
/// The precision limit of the behavior moves only the rounding position. The
/// result then takes the member of its cohort by the rules of the format at
/// its full precision. An exact result takes the member whose exponent is
/// nearest `preferred`. An inexact result has the least possible exponent,
/// so it keeps every digit. An exponent above the range of a full
/// coefficient clamps down when the coefficient has room for trailing zeros.
#[inline]
pub fn round<In: Limbs, Out: Limbs, F: DecimalRoundingTarget, B: Behavior>(
    value: &Unrounded<In>,
    preferred: i32,
    behavior: B,
) -> (Unpacked<Out>, Flags) {
    let target = &F::TARGET;
    let env = &behavior.env();
    let negative = value.negative;
    let precision = target.precision_in(env);
    let (lowest, _) = target.exponents(precision);
    let (full_lowest, full_highest) = target.exponents(target.precision);
    let digits = digit_count(&value.significand);
    if digits == 0 {
        debug_assert!(!value.sticky, "a sticky bit needs a nonzero significand");
        let exponent = i64::from(preferred).clamp(full_lowest, full_highest);
        return (
            Unpacked::Zero {
                negative,
                exponent: narrow(exponent),
            },
            Flags::NONE,
        );
    }
    debug_assert!(
        !value.sticky || digits > precision,
        "a sticky value has more digits than the precision"
    );
    let exponent = i64::from(value.exponent);
    let top = exponent + i64::from(digits) - 1;
    // IEEE 754 detects decimal tininess before rounding.
    let tiny = top < i64::from(target.emin);
    let position = exponent.max(top - i64::from(precision) + 1).max(lowest);
    let (kept, dropped) = cut::<In, Out>(value, position - exponent, digits);
    let inexact = dropped.is_inexact();
    let step = exact::rounds_up(env.rounding, negative, dropped, last_digit(&kept), 10);
    let (mut kept, mut position) = (kept, position);
    if step {
        kept = kept.increment();
        if digit_count(&kept) > precision {
            // The carry reached 10^precision.
            kept = limbs::divide_small(kept, 10).0;
            position += 1;
        }
    }
    let mut flags = Flags::NONE;
    let zero = Unpacked::Zero {
        negative,
        exponent: narrow(full_lowest),
    };
    if tiny {
        flags |= Flags::TINY;
        if env.flush_to_zero {
            return (zero, flags | Flags::UNDERFLOW | Flags::INEXACT);
        }
    }
    if inexact {
        flags |= Flags::INEXACT;
        if tiny {
            flags |= Flags::UNDERFLOW;
        }
    }
    if step {
        flags |= Flags::ROUNDED_UP;
    }
    if kept.is_zero() {
        return (zero, flags);
    }
    if position + i64::from(digit_count(&kept)) - 1 > i64::from(target.emax) {
        return overflow(negative, precision, target, env);
    }
    let (kept, position) = if inexact {
        least_exponent(kept, position, target)
    } else {
        nearest_to_preferred(kept, position, preferred, target)
    };
    let finite = Unpacked::Finite {
        negative,
        exponent: narrow(position),
        significand: kept,
    };
    (finite, flags)
}

/// The most digits that one division drops: 10^19 fits a `u64`.
const CHUNK: u32 = 19;

/// Cuts `significand * 10^exponent` at the digit `drop` places above its
/// last digit. Returns the kept digits and the dropped part, with the sticky
/// bit.
///
/// The cut divides by at most 10^19 at a time, so it needs no room above the
/// significand.
fn cut<In: Limbs, Out: Limbs>(value: &Unrounded<In>, drop: i64, digits: u32) -> (Out, Dropped) {
    let Ok(drop) = u32::try_from(drop) else {
        unreachable!("the rounding position is not below the last digit");
    };
    if drop == 0 {
        return (
            value.significand.resize(),
            Dropped::new(Ordering::Less, value.sticky),
        );
    }
    if drop > digits {
        // Every digit is dropped, and the value is below a tenth of a unit.
        return (Out::ZERO, Dropped::BelowHalf);
    }
    let mut quotient = value.significand;
    let mut rest = value.sticky;
    let mut first = 0;
    let mut left = drop;
    while left > 0 {
        let chunk = left.min(CHUNK);
        let divisor = u64::try_from(digits::power_of_ten_u128(chunk)).expect("10^19 fits a u64");
        let (next, remainder) = limbs::divide_small(quotient, divisor);
        left -= chunk;
        if left == 0 {
            // The last chunk holds the first dropped digit.
            let unit =
                u64::try_from(digits::power_of_ten_u128(chunk - 1)).expect("10^18 fits a u64");
            first = remainder / unit;
            rest |= remainder % unit != 0;
        } else {
            rest |= remainder != 0;
        }
        quotient = next;
    }
    let dropped = match first {
        0 => Dropped::new(Ordering::Less, rest),
        1..=4 => Dropped::BelowHalf,
        _ => Dropped::new(first.cmp(&5), rest),
    };
    (quotient.resize(), dropped)
}

/// Rounds `significand * 10^exponent`, with a negative exponent, to an
/// integer in the direction of `rounding`, with the same cut and choice of
/// the direction as [`round`]. `Out` must hold the integer part plus one.
pub fn round_to_integer<In: Limbs, Out: Limbs>(
    value: &Unrounded<In>,
    rounding: Rounding,
) -> Integral<Out> {
    debug_assert!(value.exponent < 0, "the value has a fraction");
    round_digits(value, -i64::from(value.exponent), rounding)
}

/// Drops the last `drop` digits of the significand, rounding in the direction
/// of `rounding`. `Out` must hold the kept digits plus one.
pub fn round_digits<In: Limbs, Out: Limbs>(
    value: &Unrounded<In>,
    drop: i64,
    rounding: Rounding,
) -> Integral<Out> {
    let digits = digit_count(&value.significand);
    let (kept, dropped) = cut::<In, Out>(value, drop, digits);
    let rounded_up = exact::rounds_up(rounding, value.negative, dropped, last_digit(&kept), 10);
    let magnitude = if rounded_up { kept.increment() } else { kept };
    Integral {
        magnitude,
        inexact: dropped.is_inexact(),
        rounded_up,
    }
}

/// Moves an exact coefficient toward the preferred exponent: it drops
/// trailing zeros to raise the exponent, and adds trailing zeros to lower it
/// while the coefficient has room. An exponent above the range of the format
/// clamps down by trailing zeros. The coefficient has room, because the value
/// is below the overflow bound.
fn nearest_to_preferred<L: Limbs>(
    mut kept: L,
    mut position: i64,
    preferred: i32,
    target: &DecimalTarget,
) -> (L, i64) {
    let (lowest, highest) = target.exponents(target.precision);
    let preferred = i64::from(preferred).clamp(lowest, highest);
    while position < preferred && last_digit(&kept) == 0 {
        kept = limbs::divide_small(kept, 10).0;
        position += 1;
    }
    while position > preferred && digit_count(&kept) < target.precision {
        kept = limbs::multiply_small(kept, 10);
        position -= 1;
    }
    debug_assert!(position <= highest, "the value is below the overflow bound");
    (kept, position)
}

/// Adds trailing zeros to an inexact coefficient, down to the least possible
/// exponent. A precision limit leaves fewer digits than the format holds.
fn least_exponent<L: Limbs>(mut kept: L, mut position: i64, target: &DecimalTarget) -> (L, i64) {
    let (lowest, highest) = target.exponents(target.precision);
    while position > lowest && digit_count(&kept) < target.precision {
        kept = limbs::multiply_small(kept, 10);
        position -= 1;
    }
    debug_assert!(position <= highest, "the value is below the overflow bound");
    (kept, position)
}

/// Returns the result of an overflow: an infinity, or the largest finite value
/// of `precision` digits in the directions that round toward zero.
fn overflow<L: Limbs>(
    negative: bool,
    precision: u32,
    target: &DecimalTarget,
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
        return (Unpacked::Infinity { negative }, flags | Flags::ROUNDED_UP);
    }
    (largest(negative, precision, target), flags)
}

/// Returns the largest finite value of `precision` digits, with the least
/// possible exponent: `(10^precision - 1) * 10^(p - precision)` at the largest
/// exponent of a full coefficient.
pub fn largest<L: Limbs>(negative: bool, precision: u32, target: &DecimalTarget) -> Unpacked<L> {
    let (_, highest) = target.exponents(target.precision);
    let significand =
        power_of_ten::<L>(target.precision).sub(power_of_ten::<L>(target.precision - precision));
    Unpacked::Finite {
        negative,
        exponent: narrow(highest),
        significand,
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use crate::env::{Env, Flags, Rounding};
    use crate::exact::Exact;
    use crate::float::{D64Bid, Decoded};

    fn parts(value: D64Bid) -> (u64, i32) {
        match value.decode::<1>() {
            Decoded::Finite {
                exponent,
                significand: [coefficient],
                ..
            } => (coefficient, exponent),
            other => panic!("{other:?} is not finite"),
        }
    }

    fn rounded(coefficient: u64, exponent: i32, env: Env) -> (D64Bid, Flags) {
        let exact = Exact {
            negative: false,
            exponent,
            significand: [coefficient],
            sticky: false,
        };
        D64Bid::round(exact, env)
    }

    #[test]
    fn a_precision_limit_moves_only_the_rounding_position() {
        let limit = Env::IEEE.with_precision(NonZeroU32::new(3));
        // An exact value above the largest exponent of 16 digits clamps down.
        let (high, flags) = rounded(123, 380, limit);
        assert_eq!(
            (parts(high), flags),
            ((12_300_000_000_000, 369), Flags::NONE)
        );
        // An inexact value has the least exponent of the format.
        let (third, flags) = rounded(3_333_333, -7, limit);
        assert_eq!(parts(third), (3_330_000_000_000_000, -16));
        assert_eq!(flags, Flags::INEXACT);
        // An overflow toward zero gives the largest value of three digits.
        let (largest, flags) = rounded(1, 385, limit.with_rounding(Rounding::TowardZero));
        assert_eq!(parts(largest), (9_990_000_000_000_000, 369));
        assert_eq!(flags, Flags::OVERFLOW | Flags::INEXACT);
        // A value that rounds to zero has the least exponent of the format.
        let (zero, _) = rounded(1, -390, limit);
        assert_eq!(
            zero.decode::<1>(),
            Decoded::Zero {
                negative: false,
                exponent: -398
            }
        );
    }

    #[test]
    fn a_significand_of_any_width_rounds() {
        // 2^19136 * 10^-6000 is 3.235914409624613E-240, rounded. The
        // significand is wider than the buffer of `limbs::divide`.
        let mut wide = [0_u64; 300];
        wide[299] = 1;
        let exact = Exact {
            negative: false,
            exponent: -6000,
            significand: wide,
            sticky: false,
        };
        let (value, flags) = D64Bid::round(exact, Env::IEEE);
        assert_eq!(parts(value), (3_235_914_409_624_613, -255));
        assert_eq!(flags, Flags::INEXACT);
    }
}
