//! Certified sine and cosine, with integer quarter-period reduction.
//!
//! The usual path uses stack balls. If their intervals cannot certify the
//! answer (including when the input's exponent exceeds their width), the
//! fixed-point interval evaluator increases both reduction and evaluation
//! precision. No result is taken from an interval's center alone.

use super::ball::{Ball, integer_part};
use super::dynamic::{self, Interval};
use super::{Argument, Elementary, Radix, Target, argument, leading, near_one, power, truncate};
use crate::exact::Unrounded;
use crate::limbs::{Limbs, Widen};

/// Returns the sine of a nonzero finite argument.
pub(crate) fn sin<L: Elementary>(x: &Argument<L>, target: &Target) -> Unrounded<L::Second> {
    evaluate(x, target, false)
}

/// Returns the cosine of a nonzero finite argument.
pub(crate) fn cos<L: Elementary>(x: &Argument<L>, target: &Target) -> Unrounded<L::Second> {
    evaluate(x, target, true)
}

fn evaluate<L: Elementary>(x: &Argument<L>, target: &Target, cosine: bool) -> Unrounded<L::Second> {
    // |sin(x)| is strictly below |x| by less than |x|^3/6; cos(x) is
    // strictly below 1 by less than x^2/2. Here those differences are
    // smaller than one unit of the extra-digit truncation grid.
    if leading(x, target.radix) < -i64::from(target.precision + 4) {
        if cosine {
            return near_one(true, target);
        }
        let digits = match target.radix {
            Radix::Binary => x.significand.bit_length(),
            Radix::Decimal => super::digit_count(&x.significand),
        };
        let extra = target.precision + 3 - digits;
        let significand = match target.radix {
            Radix::Binary => x.significand.resize::<L::Second>().shl(extra),
            Radix::Decimal => {
                let factor = power::<L::Second>(Radix::Decimal, extra);
                x.significand
                    .resize::<L::Second>()
                    .widening_mul(factor)
                    .resize()
            }
        };
        return Unrounded {
            negative: x.negative,
            exponent: x.exponent - i32::try_from(extra).expect("a precision fits an i32"),
            significand: significand.sub(L::Second::ZERO.with_bit(0)),
            sticky: true,
        };
    }
    if let Some(value) =
        ball::<L, L::First>(x, target.radix, cosine).and_then(|value| truncate(&value, target))
    {
        return Unrounded {
            negative: value.negative,
            exponent: value.exponent,
            significand: value.significand.resize(),
            sticky: value.sticky,
        };
    }
    if let Some(value) =
        ball::<L, L::Second>(x, target.radix, cosine).and_then(|value| truncate(&value, target))
    {
        return value;
    }
    refine::<L, L::Second>(x, target, cosine)
}

/// Machin's identity: pi/2 = 8 atan(1/5) - 2 atan(1/239).
fn half_pi<W: Widen>() -> Ball<W> {
    fn atan<W: Widen>(denominator: u64) -> Ball<W> {
        let t = Ball::<W>::one().div_small(denominator);
        let square = t.mul(&t).negate();
        let mut power = t;
        let mut sum = t;
        for index in 1_u64.. {
            power = power.mul(&square);
            let term = power.div_small(2 * index + 1);
            sum = sum.add(&term);
            if term.upper().ceiling_log2() < -i64::from(W::BITS) - 8 {
                // The alternating remainder is smaller than this term.
                return sum.widened(term.upper());
            }
        }
        unreachable!()
    }
    atan::<W>(5).scale(3).sub(&atan::<W>(239).scale(1))
}

fn ball<L: Limbs, W: Widen>(x: &Argument<L>, radix: Radix, cosine: bool) -> Option<Ball<W>> {
    let magnitude = Argument {
        negative: false,
        ..*x
    };
    let x_ball = argument::<L, W>(&magnitude, radix)?;
    // Leave enough absolute precision for subtraction of the entire
    // quarter-period product. Large exponents use the adaptive evaluator.
    if x_ball.center().top() > i64::from(W::BITS / 2) {
        return None;
    }
    let half_pi = half_pi::<W>();
    let quotient = x_ball.div(&half_pi)?.add(&Ball::one().scale(-1));
    let quadrant = integer_part(&quotient)?;
    let remainder = x_ball.sub(&half_pi.mul(&Ball::new(false, quadrant, 0)));
    if remainder.upper().ceiling_log2() > 0 {
        return None;
    }
    let quarter = quadrant.limb(0) & 3;
    let use_cosine = cosine ^ (quarter & 1 != 0);
    let negative = if cosine {
        quarter == 1 || quarter == 2
    } else {
        quarter >= 2
    };
    let square = remainder.mul(&remainder).negate();
    let mut term = if use_cosine { Ball::one() } else { remainder };
    let mut sum = term;
    let limit = if use_cosine {
        0
    } else {
        remainder.center().top()
    } - i64::from(W::BITS)
        - 8;
    for index in 1_u64.. {
        let first = 2 * index - u64::from(use_cosine);
        term = term.mul(&square).div_small(first).div_small(first + 1);
        sum = sum.add(&term);
        if term.upper().ceiling_log2() < limit {
            // For |r| <= 1 each later term is at most 1/6 of its
            // predecessor. One extra last-term bound covers the tail.
            sum = sum.widened(term.upper());
            break;
        }
    }
    if negative ^ (!cosine && x.negative) {
        sum = sum.negate();
    }
    Some(sum)
}

fn refine<L: Limbs, W: Limbs>(x: &Argument<L>, target: &Target, cosine: bool) -> Unrounded<W> {
    let top = leading(x, target.radix);
    // log2(10) < 4. This deliberately conservative exponent estimate
    // also covers the extra significand digit in a decimal decade.
    let reduction = match target.radix {
        Radix::Binary => top.max(0),
        Radix::Decimal => (4 * (top + 1)).max(0),
    };
    let magnitude = Argument {
        negative: false,
        ..*x
    };
    let mut working = W::BITS.max(128);
    loop {
        let bits = working
            .checked_add(u32::try_from(reduction).expect("a format exponent fits u32"))
            .and_then(|value| value.checked_add(32))
            .expect("the working precision fits the address space");
        let argument = Interval::argument(&magnitude, target.radix, bits);
        let half_pi = dynamic::half_pi(bits);
        if let Some((quarter, integer)) = argument
            .div(&half_pi)
            .and_then(|quotient| quotient.nearest_integer())
        {
            let remainder = argument.sub(&half_pi.mul(&integer));
            if remainder.upper_log2() <= 0 {
                let use_cosine = cosine ^ (quarter & 1 != 0);
                let negative = if cosine {
                    quarter == 1 || quarter == 2
                } else {
                    quarter >= 2
                };
                let square = remainder.mul(&remainder).negate();
                let mut term = if use_cosine {
                    Interval::integer(1, bits)
                } else {
                    remainder
                };
                let mut sum = term.clone();
                for index in 1_u64.. {
                    let first = 2 * index - u64::from(use_cosine);
                    term = term.mul(&square).div_small(first).div_small(first + 1);
                    sum = sum.add(&term);
                    if term.upper_log2() < -i64::from(working) - 8 {
                        sum = sum.widen(&term);
                        break;
                    }
                }
                if negative ^ (!cosine && x.negative) {
                    sum = sum.negate();
                }
                if let Some(value) = sum.truncate(target) {
                    return value;
                }
            }
        }
        working = working
            .checked_mul(2)
            .expect("the working precision fits the address space");
    }
}
