//! Base-two exponential and logarithm, with one final format rounding.
//!
//! Integer exponentials and logarithms of powers of two are evaluated as
//! rationals before ball arithmetic. Other arguments use certified balls
//! of `x ln 2` or `ln x / ln 2`, never a rounded intermediate float.

use super::ball::Ball;
use super::constants::ln2;
use super::dynamic::Interval;
use super::{
    Argument, CertifiedFunction, Elementary, Function, Radix, Target, argument, certified_ziv,
    leading, near_one, out_of_range, series,
};
use crate::exact::Unrounded;
use crate::limbs::{self, Divisor, Limbs, Widen};

/// Returns `2^x`, exact when its rational coefficient fits the working
/// width, and otherwise truncated with a certified sticky bit.
pub(crate) fn exp2<L: Elementary>(x: &Argument<L>, target: &Target) -> Unrounded<L::Second> {
    let top = leading(x, target.radix);
    let precision = i64::from(target.precision);
    let tiny = match target.radix {
        Radix::Binary => top < -(precision + 4),
        Radix::Decimal => top < -(precision + 3),
    };
    // `ln 2 < 1`, so the near-one bound for exp also bounds exp2.
    if tiny {
        return near_one(x.negative, target);
    }
    // `2^(4 range) > RADIX^range` for either radix. This also bounds
    // the argument of the exponential series far inside its 2^30 limit.
    let limit = 4 * u64::from(target.range);
    let huge = match target.radix {
        Radix::Binary => top >= i64::from(64 - limit.leading_zeros()),
        Radix::Decimal => top >= i64::from(super::digit_count(&[limit])),
    };
    if huge {
        return out_of_range(!x.negative, target);
    }
    if let Some(n) = integer(x, target.radix)
        && let Some(exact) = integer_exp2::<L::Second>(n, target.radix)
    {
        return exact;
    }
    certified_ziv::<L, _>(&Exp2(*x, target.radix), target)
}

/// Returns `log2 x` of a positive finite argument, including its exact
/// integer value when the argument is a power of two.
pub(crate) fn log2<L: Elementary>(x: &Argument<L>, target: &Target) -> Unrounded<L::Second> {
    debug_assert!(!x.negative, "the logarithm takes a positive argument");
    if let Some(exponent) = power_of_two(x, target.radix) {
        return Unrounded {
            negative: exponent < 0,
            exponent: 0,
            significand: [exponent.unsigned_abs()].resize(),
            sticky: false,
        };
    }
    certified_ziv::<L, _>(&Log2(*x, target.radix), target)
}

struct Exp2<L>(Argument<L>, Radix);
struct Log2<L>(Argument<L>, Radix);

impl<L: Limbs> Function for Exp2<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let x = argument::<L, W>(&self.0, self.1)?;
        Some(series::exp(&x.mul(&ln2::<W>())))
    }
}

impl<L: Limbs> Function for Log2<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let x = &self.0;
        let logarithm = match self.1 {
            Radix::Binary => series::log(&Ball::new(false, x.significand, i64::from(x.exponent)))?,
            Radix::Decimal => {
                series::log_decimal(&Ball::new(false, x.significand, 0), i64::from(x.exponent))?
            }
        };
        logarithm.div(&ln2::<W>())
    }
}

impl<L: Elementary> CertifiedFunction for Exp2<L> {
    fn refine<W: Limbs>(&self, target: &Target) -> Unrounded<W> {
        // This approximate integer chooses a reduction only: subtracting it
        // from the exact interval argument preserves the enclosure. At the
        // first width the conversion error is far below 1/4, so the residual
        // is below 1 in magnitude, even beside a half-integer.
        let nearest = argument::<L, L::First>(&self.0, self.1)
            .expect("the first width bounds an exp2 argument")
            .center()
            .round_to_i64();
        let mut bits = W::BITS
            .max(128)
            .checked_mul(2)
            .expect("a working width fits u32");
        loop {
            let residual =
                Interval::argument(&self.0, self.1, bits).sub(&Interval::integer(nearest, bits));
            let result = dynamic_exp(&residual.mul(&dynamic_ln2(bits)));
            if let Some(result) = result.truncate_scaled(target, nearest) {
                return result;
            }
            bits = bits
                .checked_mul(2)
                .expect("the refinement precision fits u32");
        }
    }
}

impl<L: Limbs> CertifiedFunction for Log2<L> {
    fn refine<W: Limbs>(&self, target: &Target) -> Unrounded<W> {
        let x = &self.0;
        let top = x.significand.bit_length() - 1;
        let normalized = Argument {
            negative: false,
            exponent: -i32::try_from(top).expect("a significand width fits i32"),
            significand: x.significand,
        };
        let mut bits = W::BITS
            .max(128)
            .checked_mul(2)
            .expect("a working width fits u32");
        loop {
            let ln2 = dynamic_ln2(bits);
            let m = Interval::argument(&normalized, Radix::Binary, bits);
            let logarithm = dynamic_log_normalized(&m);
            let coefficient_exponent = i64::from(top);
            let logarithm = match self.1 {
                Radix::Binary => logarithm.add(&ln2.mul(&Interval::integer(
                    coefficient_exponent + i64::from(x.exponent),
                    bits,
                ))),
                Radix::Decimal => {
                    // ln 10 = ln(5/4) + 3 ln 2. Only the small normalized
                    // coefficient enters log; extreme decimal exponents do
                    // not materialize an enormous argument.
                    let five_fourths = Interval::integer(5, bits).scale(-2);
                    let ln10 = dynamic_log_normalized(&five_fourths).add(&ln2.mul_small(3));
                    logarithm
                        .add(&ln2.mul(&Interval::integer(coefficient_exponent, bits)))
                        .add(&ln10.mul(&Interval::integer(i64::from(x.exponent), bits)))
                }
            };
            let result = logarithm.div(&ln2).expect("ln 2 is bounded away from zero");
            if let Some(result) = result.truncate(target) {
                return result;
            }
            bits = bits
                .checked_mul(2)
                .expect("the refinement precision fits u32");
        }
    }
}

/// The exponential of an argument below 1 in magnitude. Taylor terms at
/// `x / 2^s` have a tail below the last term: the next-term ratio is below
/// 1/4. Outward-rounded squarings preserve the interval after adding it.
fn dynamic_exp(x: &Interval) -> Interval {
    let bits = x.bits();
    let squarings = i64::from(bits.isqrt() / 2 + 2);
    let t = x.scale(-squarings);
    let mut term = Interval::integer(1, bits);
    let mut sum = term.clone();
    for index in 1_u64.. {
        term = term.mul(&t).div_small(index);
        sum = sum.add(&term);
        // Fixed-point terms eventually reach a one-unit endpoint rather
        // than tending to zero. Stop above that floor, then bound the tail.
        if term.upper_log2() <= 4 - i64::from(bits) {
            sum = sum.widen(&term);
            for _ in 0..squarings {
                sum = sum.mul(&sum);
            }
            return sum;
        }
    }
    unreachable!()
}

/// `ln m` for `1 <= m <= 2`: with `t = (m-1)/(m+1)`, `|t| <= 1/3`,
/// so the tail of the atanh series is below its last term in magnitude.
fn dynamic_log_normalized(m: &Interval) -> Interval {
    let bits = m.bits();
    let one = Interval::integer(1, bits);
    let t = m.sub(&one).div(&m.add(&one)).expect("m + 1 is positive");
    let t2 = t.mul(&t);
    let mut power = t.clone();
    let mut sum = t;
    for index in 1_u64.. {
        power = power.mul(&t2);
        let term = power.div_small(2 * index + 1);
        sum = sum.add(&term);
        if power.upper_log2() <= 4 - i64::from(bits) {
            return sum.widen(&term).scale(1);
        }
    }
    unreachable!()
}

fn dynamic_ln2(bits: u32) -> Interval {
    dynamic_log_normalized(&Interval::integer(2, bits))
}

/// Decodes a small integral argument without floating-point conversion.
/// The caller has already rejected huge magnitudes.
fn integer<L: Limbs>(x: &Argument<L>, radix: Radix) -> Option<i64> {
    let magnitude = match radix {
        Radix::Binary if x.exponent >= 0 => {
            let shift = u32::try_from(x.exponent).ok()?;
            if x.significand.bit_length().checked_add(shift)? > 62 {
                return None;
            }
            x.significand.resize::<[u64; 1]>().shl(shift).limb(0)
        }
        Radix::Binary => {
            let shift = x.exponent.unsigned_abs();
            if x.significand.any_below(shift) {
                return None;
            }
            let whole = x.significand.shr(shift);
            if whole.bit_length() > 62 {
                return None;
            }
            whole.limb(0)
        }
        Radix::Decimal => {
            let mut whole = x.significand;
            if x.exponent < 0 {
                // More decimal divisions than bits must reach a nonzero
                // remainder, since the coefficient is nonzero.
                if x.exponent.unsigned_abs() > whole.bit_length() {
                    return None;
                }
                for _ in 0..x.exponent.unsigned_abs() {
                    let (quotient, remainder) = limbs::divide_small(whole, Divisor::new(10));
                    if remainder != 0 {
                        return None;
                    }
                    whole = quotient;
                }
                if whole.bit_length() > 62 {
                    return None;
                }
                whole.limb(0)
            } else {
                if whole.bit_length() > 62 {
                    return None;
                }
                let mut magnitude = whole.limb(0);
                for _ in 0..x.exponent {
                    magnitude = magnitude.checked_mul(10)?;
                }
                magnitude
            }
        }
    };
    let magnitude = i64::try_from(magnitude).ok()?;
    Some(if x.negative { -magnitude } else { magnitude })
}

/// Returns a rational power directly when its coefficient fits `W`.
/// For decimal, `2^-n = 5^n 10^-n`; the format rounding routine can discard
/// coefficient digits itself, without introducing an approximate result.
fn integer_exp2<W: Limbs>(n: i64, radix: Radix) -> Option<Unrounded<W>> {
    let (exponent, significand) = match radix {
        Radix::Binary => (i32::try_from(n).ok()?, W::ZERO.with_bit(0)),
        Radix::Decimal if n >= 0 => {
            let bit = u32::try_from(n).ok()?;
            if bit >= W::BITS {
                return None;
            }
            (0, W::ZERO.with_bit(bit))
        }
        Radix::Decimal => {
            let count = u32::try_from(n.unsigned_abs()).ok()?;
            if count >= W::BITS {
                return None;
            }
            let mut coefficient = W::ZERO.with_bit(0);
            for _ in 0..count {
                if coefficient.bit_length() + 3 > W::BITS {
                    return None;
                }
                coefficient = limbs::multiply_small(coefficient, 5);
            }
            (i32::try_from(n).ok()?, coefficient)
        }
    };
    Some(Unrounded {
        negative: false,
        exponent,
        significand,
        sticky: false,
    })
}

/// Recognizes powers of two in either radix, including subnormals and
/// decimal cohorts. A decimal denominator contributes a factor `5^-q`,
/// which must cancel exactly before the remaining coefficient is dyadic.
fn power_of_two<L: Limbs>(x: &Argument<L>, radix: Radix) -> Option<i64> {
    let mut coefficient = x.significand;
    if radix == Radix::Decimal {
        if x.exponent > 0 {
            return None;
        }
        let count = x.exponent.unsigned_abs();
        if count > coefficient.bit_length() {
            return None;
        }
        for _ in 0..count {
            let (quotient, remainder) = limbs::divide_small(coefficient, Divisor::new(5));
            if remainder != 0 {
                return None;
            }
            coefficient = quotient;
        }
    }
    let top = coefficient.bit_length().checked_sub(1)?;
    if coefficient.any_below(top) {
        return None;
    }
    Some(i64::from(x.exponent) + i64::from(top))
}
