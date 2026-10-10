//! The rational values of the exponentials and the logarithms in base 2 and
//! 10, and the exact integer arithmetic of `compound`.
//!
//! `b^x` for `b = 2` or `10` and a rational `x = m / n` in lowest terms is
//! rational only for `n = 1`: the exponent of each prime of `b^m` would be a
//! multiple of `n`. In the same way, `log_b y` of a positive rational `y` is
//! rational only when `y` is a power of `b` with an integer exponent. A
//! rational value can lie on the grid of the truncation, where no ball
//! decides it, so this module computes those values exactly.
//!
//! A power of the base in its own radix, and `b^n - 1` there, has a short
//! exact form. A power of the other base fits the second working width for a
//! small `n` only. A larger one is an odd integer of more than `p + 3` bits,
//! or an integer with a last digit other than 0 of more than `p + 2` digits,
//! or the reciprocal of a power of 5 in binary: it lies off the grid, and a
//! ball decides it as for an irrational value.

use super::{Argument, Base, Radix, Target, power};
use crate::exact::Unrounded;
use crate::limbs::{self, Divisor, Limbs};

/// Returns the integer part of `|x|`, saturated at `u64::MAX`, and `true`
/// when `x` has a nonzero fraction.
fn split<L: Limbs>(x: &Argument<L>, radix: Radix) -> (u64, bool) {
    let shift = x.exponent.unsigned_abs();
    match radix {
        Radix::Binary if x.exponent >= 0 => {
            if x.significand.bit_length() + shift > 64 {
                return (u64::MAX, false);
            }
            (x.significand.limb(0) << shift, false)
        }
        Radix::Binary => {
            if shift >= L::BITS {
                // The significand is below `2^shift`, so `|x| < 1`.
                return (0, true);
            }
            let whole = x.significand.shr(shift);
            let fraction = x.significand.any_below(shift);
            if whole.bit_length() > 64 {
                return (u64::MAX, fraction);
            }
            (whole.limb(0), fraction)
        }
        Radix::Decimal => {
            // A decimal coefficient has at most 34 digits.
            let coefficient = limbs::to_u128(&x.significand);
            let unit = 10_u128.checked_pow(shift);
            if x.exponent >= 0 {
                let whole = unit.and_then(|unit| coefficient.checked_mul(unit));
                let whole = whole.and_then(|whole| u64::try_from(whole).ok());
                return (whole.unwrap_or(u64::MAX), false);
            }
            let Some(unit) = unit else {
                // `10^shift` passes 2^128, above the coefficient.
                return (0, true);
            };
            let whole = u64::try_from(coefficient / unit).unwrap_or(u64::MAX);
            (whole, !coefficient.is_multiple_of(unit))
        }
    }
}

/// Returns `x` when it is an integer of magnitude below 2^63.
pub(super) fn integer<L: Limbs>(x: &Argument<L>, radix: Radix) -> Option<i64> {
    let (whole, fraction) = split(x, radix);
    if fraction {
        return None;
    }
    let magnitude = i64::try_from(whole).ok()?;
    Some(if x.negative { -magnitude } else { magnitude })
}

/// Returns `true` when `|x|` is at least `bound`.
pub(super) fn at_least<L: Limbs>(x: &Argument<L>, radix: Radix, bound: u64) -> bool {
    split(x, radix).0 >= bound
}

/// Returns the exact integer `k`.
pub(super) fn integer_value<W: Limbs>(k: i64) -> Unrounded<W> {
    Unrounded {
        negative: k < 0,
        exponent: 0,
        significand: W::ZERO.with_limb(0, k.unsigned_abs()),
        sticky: false,
    }
}

/// Returns `b^n`, or `b^n - 1` when `minus_one` is set, for an integer `n`
/// other than zero inside the range of the shortcuts of `exponential`.
///
/// With the base equal to the radix, `b^n` is `1 * RADIX^n`, and `b^n - 1`
/// has `|n|` digits `RADIX - 1`. Those digits truncate to `p + 3` bits, or
/// `p + 2` digits, with the sticky bit. A power of the other base is exact
/// when it fits `W`. Returns `None` for a power that does not, and for
/// `10^n` with `n < 0` in binary, which lie off the grid.
pub(super) fn power_value<W: Limbs>(
    base: Base,
    n: i64,
    minus_one: bool,
    target: &Target,
) -> Option<Unrounded<W>> {
    let radix = target.radix;
    let one = W::ZERO.with_bit(0);
    if base.is_radix(radix) {
        if !minus_one {
            return Some(exact(false, one, n, target));
        }
        let count = n.unsigned_abs();
        let kept = u64::from(target.kept());
        let digits = count.min(kept);
        let lowest = n.min(0) + i64::try_from(count - digits).ok()?;
        let digits = u32::try_from(digits).expect("the kept digits fit a u32");
        return Some(Unrounded {
            negative: n < 0,
            exponent: clamped(i128::from(lowest), target),
            significand: power::<W>(radix, digits).sub(one),
            sticky: count > kept,
        });
    }
    // `b^n = value * RADIX^exponent`.
    let count = n.unsigned_abs();
    let (value, exponent) = match (radix, n >= 0) {
        // `10^n = 5^n 2^n` in binary, and `2^n = 5^|n| 10^n` in decimal.
        (Radix::Binary, true) | (Radix::Decimal, false) => (small_power::<W>(5, count)?, n),
        (Radix::Binary, false) => return None,
        (Radix::Decimal, true) => (radix_power::<W>(Radix::Binary, count)?, 0),
    };
    if !minus_one {
        return Some(exact(false, value, exponent, target));
    }
    if exponent >= 0 {
        let whole = scaled(value, exponent, radix)?;
        Some(exact(false, whole.sub(one), 0, target))
    } else {
        let unit = radix_power::<W>(radix, exponent.unsigned_abs())?;
        Some(exact(true, unit.sub(value), exponent, target))
    }
}

/// Returns `k` when `x`, or `1 + x` when `plus_one` is set, is `b^k` for a
/// base of 2 or 10, and `None` when the logarithm is irrational.
pub(super) fn logarithm<L: Limbs, W: Limbs>(
    x: &Argument<L>,
    base: Base,
    plus_one: bool,
    radix: Radix,
) -> Option<i64> {
    let (value, exponent) = if plus_one {
        one_plus::<L, W>(x, radix)?
    } else {
        (x.significand.resize::<W>(), i64::from(x.exponent))
    };
    let (value, exponent) = stripped(value, exponent, radix);
    match (base, radix) {
        (Base::E, _) => None,
        (Base::Two, Radix::Binary) | (Base::Ten, Radix::Decimal) => {
            (value == W::ZERO.with_bit(0)).then_some(exponent)
        }
        // `M 2^E = 10^k = 5^k 2^k` with `M` odd.
        (Base::Ten, Radix::Binary) => {
            let count = u64::try_from(exponent).ok()?;
            (small_power::<W>(5, count)? == value).then_some(exponent)
        }
        // `M 10^E = 2^k` with `M` not a multiple of 10: `E = 0` and
        // `M = 2^k`, or `E = k < 0` and `M = 5^-k`.
        (Base::Two, Radix::Decimal) => match exponent {
            0 => {
                let top = value.bit_length() - 1;
                (!value.any_below(top)).then_some(i64::from(top))
            }
            ..0 => (small_power::<W>(5, exponent.unsigned_abs())? == value).then_some(exponent),
            _ => None,
        },
    }
}

/// Returns the exact value `value * RADIX^exponent`.
fn exact<W: Limbs>(negative: bool, value: W, exponent: i64, target: &Target) -> Unrounded<W> {
    Unrounded {
        negative,
        exponent: clamped(i128::from(exponent), target),
        significand: value,
        sticky: false,
    }
}

/// Returns `base^count` for a base below 8 when it fits `W`: the power is
/// below `2^(3 count)`.
fn small_power<W: Limbs>(base: u64, count: u64) -> Option<W> {
    debug_assert!(base < 8, "a small power has a base below 2^3");
    if count.checked_mul(3)? >= u64::from(W::BITS) {
        return None;
    }
    Some(integer_power(W::ZERO.with_limb(0, base), count))
}

/// Returns `RADIX^count` when it fits `W`. `10^count` is below
/// `2^(4 count)`.
pub(super) fn radix_power<W: Limbs>(radix: Radix, count: u64) -> Option<W> {
    let bits = match radix {
        Radix::Binary => count,
        Radix::Decimal => count.checked_mul(4)?,
    };
    let count = u32::try_from(count).ok()?;
    (bits < u64::from(W::BITS)).then(|| power(radix, count))
}

/// Returns `value * RADIX^exponent` for `exponent >= 0` when it fits `W`.
fn scaled<W: Limbs>(value: W, exponent: i64, radix: Radix) -> Option<W> {
    let count = u64::try_from(exponent).ok()?;
    let unit = radix_power::<W>(radix, count)?;
    let bits = u64::from(value.bit_length()) + u64::from(unit.bit_length());
    (bits < u64::from(W::BITS)).then(|| limbs::multiply_fit(value, unit))
}

/// Returns `1 + x` exactly as `value * RADIX^exponent`, when it fits the
/// width `W` with a bit to spare. `x` is above -1.
///
/// A `1 + x` that does not fit is not a power of 2 or 10: a large `x` makes
/// it an integer whose last digit is 1, and a tiny `x` puts it near 1.
pub(super) fn one_plus<L: Limbs, W: Limbs>(x: &Argument<L>, radix: Radix) -> Option<(W, i64)> {
    let exponent = i64::from(x.exponent);
    let significand = x.significand.resize::<W>();
    if exponent >= 0 {
        // `x` is at least 1, so it is positive: `1 + x` is the integer
        // `m RADIX^e + 1`.
        debug_assert!(
            !x.negative,
            "an argument above -1 and at least 1 is positive"
        );
        return Some((scaled(significand, exponent, radix)?.increment(), 0));
    }
    // `1 + x` is `(RADIX^k ± m) RADIX^-k`, and `m < RADIX^k` for a negative
    // `x`, because `|x| < 1`. A binary sum below `2^(W::BITS - 1)`, and a
    // decimal one below `2^(0.84 W::BITS + 2)`, fits.
    let shift = exponent.unsigned_abs();
    if radix == Radix::Binary && shift >= u64::from(W::BITS) - 1 {
        return None;
    }
    let unit = radix_power::<W>(radix, shift)?;
    let sum = if x.negative {
        unit.sub(significand)
    } else {
        unit.add(significand)
    };
    Some((sum, exponent))
}

/// Returns `x - 1` exactly as `value * RADIX^exponent` for an argument above
/// 1, when it fits the width `W`. A value that does not fit belongs to a
/// large `x`, where `x - 1` keeps its digits in a ball.
pub(super) fn minus_one<L: Limbs, W: Limbs>(x: &Argument<L>, radix: Radix) -> Option<(W, i64)> {
    debug_assert!(!x.negative, "an argument above 1 is positive");
    let exponent = i64::from(x.exponent);
    let significand = x.significand.resize::<W>();
    if exponent >= 0 {
        let one = W::ZERO.with_bit(0);
        return Some((scaled(significand, exponent, radix)?.sub(one), 0));
    }
    // `x - 1` is `(m - RADIX^k) RADIX^-k`, and `m > RADIX^k`, because
    // `x > 1`.
    let unit = radix_power::<W>(radix, exponent.unsigned_abs())?;
    Some((significand.sub(unit), exponent))
}

/// Returns `value * RADIX^exponent`, a nonzero number, with every factor of
/// the radix moved from `value` to the exponent.
pub(super) fn stripped<W: Limbs>(value: W, exponent: i64, radix: Radix) -> (W, i64) {
    match radix {
        Radix::Binary => {
            let zeros = (0..W::BITS)
                .find(|&position| value.bit(position))
                .expect("the value is not zero");
            (value.shr(zeros), exponent + i64::from(zeros))
        }
        Radix::Decimal => {
            let ten = Divisor::new(10);
            let (mut value, mut exponent) = (value, exponent);
            loop {
                let (quotient, rest) = limbs::divide_small(value, ten);
                if rest != 0 {
                    return (value, exponent);
                }
                value = quotient;
                exponent += 1;
            }
        }
    }
}

/// Returns `base^count`, which must fit `W`, by squaring.
pub(super) fn integer_power<W: Limbs>(base: W, count: u64) -> W {
    let mut result = W::ZERO.with_bit(0);
    let mut square = base;
    let mut rest = count;
    loop {
        if rest & 1 == 1 {
            result = limbs::multiply_fit(result, square);
        }
        rest >>= 1;
        if rest == 0 {
            return result;
        }
        square = limbs::multiply_fit(square, square);
    }
}

/// Returns an exponent of a result, clamped far past the range of the format,
/// where a clamped exponent rounds as the true one does.
pub(super) fn clamped(exponent: i128, target: &Target) -> i32 {
    let bound = 2 * i128::from(target.range) + i128::from(target.precision);
    i32::try_from(exponent.clamp(-bound, bound)).expect("the bound fits an i32")
}
