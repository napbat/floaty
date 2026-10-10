//! The balls of `exp`, `e^x - 1`, `log`, `ln(1 + x)`, `sin`, `cos`, `atan`,
//! and the powers of 10 at one working width.

use super::ball::Ball;
use super::constants::{ln2, ln10, pi};
use crate::limbs::Widen;

/// `floor(sqrt(2) * 2^63)`: a mantissa whose top 64 bits are at or above it
/// lies at or above `sqrt(2)` in its binade.
const SQRT2: u64 = 0xB504_F333_F9DE_6484;

/// The terms that a series sums at most. Each series stops far earlier.
const TERM_LIMIT: u64 = 4096;

/// Returns the number of squarings that ends the evaluation of `exp`: about
/// half the square root of the width, which balances the terms of the series
/// against the squarings.
fn squarings<W: Widen>() -> i64 {
    i64::from(W::BITS.isqrt() / 2 + 2)
}

/// Returns the ball of `e^x` for a ball of `|x|` below 2^30.
///
/// `x = k ln 2 + r` with `k` the integer nearest `x / ln 2`, so `|r|` is
/// about `ln 2 / 2` at most. The series of `e^t` runs on `t = r / 2^s` and
/// stops at a term below `2^-(W::BITS + 8)`. Each later term is at most a
/// quarter of the one before, so the rest of the series is below that term,
/// which the radius takes. `s` squarings then give `e^r`, and `2^k` scales
/// it.
pub(super) fn exp<W: Widen>(x: &Ball<W>) -> Ball<W> {
    let ln2 = ln2::<W>();
    let (quotient, _) = x.center().div(ln2.center());
    let k = quotient.round_to_i64();
    let r = x.sub(&ln2.mul(&Ball::integer(k)));
    let squarings = squarings::<W>();
    let t = r.scale(-squarings);
    let limit = -i64::from(W::BITS) - 8;
    let mut sum = Ball::one();
    let mut term = Ball::one();
    // `|t|` is below 1/2, so each term is at most a quarter of the one before
    // and the rest of the series lies below the last term, also when the
    // limit stops the loop.
    for index in 1..=TERM_LIMIT {
        term = term.mul(&t).div_small(index);
        sum = sum.add(&term);
        if term.upper().ceiling_log2() < limit {
            break;
        }
    }
    sum = sum.widened(term.upper());
    for _ in 0..squarings {
        sum = sum.mul(&sum);
    }
    sum.scale(k)
}

/// Returns the ball of `e^x - 1` for a ball of `|x|` below 2^30.
///
/// From `|x| = 1/2` up, `|e^x - 1|` is at least a third of the larger of
/// `e^x` and 1, so the difference with 1 loses at most two bits. Below it,
/// the series of `e^t - 1` runs on `t = x / 2^s` without its first term,
/// which keeps the error relative to the value. It stops at a term below
/// `|t| 2^-(W::BITS + 8)`, and the radius takes the rest, as in [`exp`].
/// `s` steps of `e^(2t) - 1 = u (u + 2)`, with `u = e^t - 1`, then give the
/// result.
pub(super) fn exp_minus_one<W: Widen>(x: &Ball<W>) -> Ball<W> {
    let center = x.center();
    if center.is_zero() || center.top() >= -1 {
        return exp(x).sub(&Ball::one());
    }
    let squarings = squarings::<W>();
    let t = x.scale(-squarings);
    let limit = t.center().top() - i64::from(W::BITS) - 8;
    let mut sum = t;
    let mut term = t;
    // `|t|` is below 1/4, so each term is at most a quarter of the one before.
    for index in 2..=TERM_LIMIT {
        term = term.mul(&t).div_small(index);
        sum = sum.add(&term);
        if term.upper().ceiling_log2() < limit {
            break;
        }
    }
    sum = sum.widened(term.upper());
    let two = Ball::integer(2);
    for _ in 0..squarings {
        sum = sum.mul(&sum.add(&two));
    }
    sum
}

/// Returns the ball of `ln y` for a ball of positive values, or `None` when
/// the ball comes near zero.
///
/// `y = m 2^e` with `m` from `1 / sqrt(2)` to `sqrt(2)`, so
/// `t = (m - 1) / (m + 1)` is at most 0.172 in magnitude, and
/// `ln m = 2 atanh(t)`.
pub(super) fn log<W: Widen>(y: &Ball<W>) -> Option<Ball<W>> {
    if !y.is_positive() {
        return None;
    }
    let center = y.center();
    let exponent = if center.leading() >= SQRT2 {
        center.top() + 1
    } else {
        center.top()
    };
    let m = y.scale(-exponent);
    let one = Ball::one();
    let t = m.sub(&one).div(&m.add(&one))?;
    Some(twice_atanh(&t).add(&ln2::<W>().mul(&Ball::integer(exponent))))
}

/// Returns the ball of `ln(1 + x)` for a ball of values above -1, or `None`
/// when the ball comes near -1.
///
/// From `|x| = 1/4` up, `|ln(1 + x)|` is above 0.2, so the sum `1 + x` loses
/// at most three bits of the logarithm. Below it, `ln(1 + x) = 2 atanh(t)`
/// with `t = x / (2 + x)`, at most 1/7 in magnitude, which keeps the error
/// relative to the value.
pub(super) fn log_one_plus<W: Widen>(x: &Ball<W>) -> Option<Ball<W>> {
    let center = x.center();
    if center.is_zero() || center.top() >= -2 {
        return log(&Ball::one().add(x));
    }
    let t = x.div(&x.add(&Ball::integer(2)))?;
    Some(twice_atanh(&t))
}

/// Returns the ball of `2 atanh(t)`, the sum of `2 t^(2j + 1) / (2j + 1)`,
/// for a ball of `|t|` at most 0.172. The series stops at a term below
/// `|t| 2^-(W::BITS + 8)`. The rest is below twice that term, because `t^2`
/// is below 0.03, and the radius takes it.
fn twice_atanh<W: Widen>(t: &Ball<W>) -> Ball<W> {
    if t.center().is_zero() {
        // `t` is within its radius of 0. The terms past the first are below
        // the cube of the radius.
        return t.widened(t.radius()).scale(1);
    }
    let t2 = t.mul(t);
    let limit = t.center().top() - i64::from(W::BITS) - 8;
    let mut sum = *t;
    let mut power = *t;
    let mut term = *t;
    for index in 1..=TERM_LIMIT {
        power = power.mul(&t2);
        term = power.div_small(2 * index + 1);
        sum = sum.add(&term);
        if term.upper().ceiling_log2() < limit {
            break;
        }
    }
    sum.widened(term.upper().scale(1)).scale(1)
}

/// Returns the ball of `ln c + q ln 10` for a positive integer `c`.
pub(super) fn log_decimal<W: Widen>(c: &Ball<W>, q: i64) -> Option<Ball<W>> {
    let log_c = log(c)?;
    Some(log_c.add(&ln10::<W>().mul(&Ball::integer(q))))
}

/// Returns the ball of `sin y` for a ball of `|y|` at most 1.
///
/// The Taylor series alternates, and each term is below a sixth of the one
/// before, so the rest lies below the last term, which the radius takes. The
/// series stops at a term below `|y| 2^-(W::BITS + 8)`.
pub(super) fn sine<W: Widen>(y: &Ball<W>) -> Ball<W> {
    if y.center().is_zero() {
        // `|sin y|` is at most `|y|`, within the radius of 0.
        return y.widened(y.radius());
    }
    let square = y.mul(y);
    let limit = y.center().top() - i64::from(W::BITS) - 8;
    let mut sum = *y;
    let mut term = *y;
    for index in 1..=TERM_LIMIT {
        term = term
            .mul(&square)
            .div_small(2 * index * (2 * index + 1))
            .negate();
        sum = sum.add(&term);
        if term.upper().ceiling_log2() < limit {
            break;
        }
    }
    sum.widened(term.upper())
}

/// Returns the ball of `cos y` for a ball of `|y|` at most 1, as [`sine`]
/// does. The series stops at a term below `2^-(W::BITS + 8)`, because
/// `cos y` is above 1/2.
pub(super) fn cosine<W: Widen>(y: &Ball<W>) -> Ball<W> {
    let square = y.mul(y);
    let limit = -i64::from(W::BITS) - 8;
    let mut sum = Ball::one();
    let mut term = Ball::one();
    for index in 1..=TERM_LIMIT {
        term = term
            .mul(&square)
            .div_small((2 * index - 1) * 2 * index)
            .negate();
        sum = sum.add(&term);
        if term.upper().ceiling_log2() < limit {
            break;
        }
    }
    sum.widened(term.upper())
}

/// Returns the ball of `atan y` for a ball of positive values, or `None`
/// when the ball comes near zero.
///
/// From `y = 1` up, `atan y = pi / 2 - atan(1 / y)`, with `1 / y` at most 1.
/// Each step of `atan t = 2 atan(t / (1 + sqrt(1 + t^2)))` then about halves
/// the argument, until it lies below `2^-s`, with `s` from [`squarings`].
/// The series runs there, and the `k` steps scale its ball by `2^k`, which
/// keeps the error relative to the value.
pub(super) fn arctangent<W: Widen>(y: &Ball<W>) -> Option<Ball<W>> {
    let one = Ball::one();
    let reflected = !y.center().is_zero() && y.center().top() >= 0;
    let mut t = if reflected { one.div(y)? } else { *y };
    let limit = -squarings::<W>();
    let mut halvings = 0;
    while !t.center().is_zero() && t.center().top() >= limit {
        let root = one.add(&t.mul(&t)).sqrt()?;
        t = t.div(&one.add(&root))?;
        halvings += 1;
    }
    let value = arctangent_series(&t).scale(halvings);
    Some(if reflected {
        pi::<W>().scale(-1).sub(&value)
    } else {
        value
    })
}

/// Returns the ball of `atan t`, the sum of `(-1)^j t^(2j + 1) / (2j + 1)`,
/// for a ball of `|t|` below 1/2. The series alternates, and its terms
/// decrease, so the rest lies below the last term, which the radius takes.
/// The series stops at a term below `|t| 2^-(W::BITS + 8)`.
fn arctangent_series<W: Widen>(t: &Ball<W>) -> Ball<W> {
    if t.center().is_zero() {
        // `|atan t|` is at most `|t|`, within the radius of 0.
        return t.widened(t.radius());
    }
    let square = t.mul(t);
    let limit = t.center().top() - i64::from(W::BITS) - 8;
    let mut sum = *t;
    let mut power = *t;
    let mut term = *t;
    for index in 1..=TERM_LIMIT {
        power = power.mul(&square).negate();
        term = power.div_small(2 * index + 1);
        sum = sum.add(&term);
        if term.upper().ceiling_log2() < limit {
            break;
        }
    }
    sum.widened(term.upper())
}

/// Returns the ball of `10^n` for `n >= 0`, by squaring.
pub(super) fn power_of_ten<W: Widen>(n: u64) -> Ball<W> {
    let mut result = Ball::one();
    let mut base = Ball::integer(10);
    let mut rest = n;
    while rest != 0 {
        if rest & 1 == 1 {
            result = result.mul(&base);
        }
        rest >>= 1;
        if rest != 0 {
            base = base.mul(&base);
        }
    }
    result
}
