//! `compound(x, n) = (1 + x)^n` of IEEE 754-2019 section 9.2 for a binary
//! argument `x > -1`, other than zero, and an integer `n` other than zero.
//!
//! The result is rational, so it can lie on the grid of the truncation, or
//! closer to the grid than any ball resolves, and then no ball decides the
//! truncation. With `1 + x = M 2^E` and `M` odd, a result on the grid is a
//! power of two, or `M^n` with few bits for `n > 0`. An exact integer power,
//! or the exact quotient `2^s / M^|n|` for `n < 0`, gives each result whose
//! power fits the second working width. For a large `x`, the result lies
//! within `2 |n| / x` of `x^n` in ratio, and `x^n` can lie on the grid; the
//! result then follows from the side of `x^n` that it lies on. A tiny `n x`
//! gives a result within one grid step of 1 in the same way. A huge
//! `n ln(1 + x)` gives an overflow or an underflow. Ziv's strategy evaluates
//! `e^(n ln(1 + x))` for every other result.

use super::ball::Ball;
use super::{
    Argument, Elementary, Function, Radix, Target, leading, near_one, out_of_range, series, ziv,
};
use crate::exact::Unrounded;
use crate::limbs::{self, Limbs, Widen};

/// Returns `(1 + x)^n` of a binary argument, truncated for the rounding
/// routine of the format. The result is exact when the power is, and has
/// the sticky bit set otherwise.
pub(crate) fn compound<L: Elementary>(
    x: &Argument<L>,
    n: i64,
    target: &Target,
) -> Unrounded<L::Second> {
    debug_assert!(
        target.radix == Radix::Binary,
        "compound takes a binary argument"
    );
    // `|n x|` is below `2^(leading + 1 + bits(n))`. When that is at most
    // `2^-(p + 5)`, `|n ln(1 + x)|`, below `2 |n x|`, is below the tiny bound
    // of `e^t`. `t` has the sign of `n x`.
    let count_bits = i64::from(u64::BITS - n.unsigned_abs().leading_zeros());
    let precision = i64::from(target.precision);
    if leading(x, Radix::Binary) + 1 + count_bits <= -(precision + 5) {
        return near_one((n < 0) != x.negative, target);
    }
    if let Some(exact) = exact_power::<L>(x, n, target) {
        return exact;
    }
    if leading(x, Radix::Binary) >= precision + 6 + count_bits
        && let Some(near) = near_dominant::<L>(x, n, target)
    {
        return near;
    }
    // Past `4 * range`, `e^t` overflows or underflows the range, as for
    // `exp`. Below it, `|t|` stays far inside the bound of `series::exp`.
    let limit = 4 * u64::from(target.range);
    let huge = i64::from(u64::BITS - limit.leading_zeros());
    let first = log_term::<L, L::First>(x, n);
    if let Some(t) = first.filter(|t| !t.center().is_zero() && t.center().top() >= huge) {
        return out_of_range(!t.center().is_negative(), target);
    }
    ziv::<L, _>(&Compound(*x, n), target)
}

/// Returns the truncation of `(1 + x)^n` for an `x` of at least
/// `2^(p + 6 + bits(n))` whose power `x^n` lies on the grid of the
/// truncation, or `None` when `x^n` does not.
///
/// `(1 + x)^n` is `x^n (1 + d)`, where `d` has the sign of `n` and `|d|` is
/// below `2 |n| / x`, at most `2^-(p + 5)`. So the result lies within one grid
/// step of `x^n`: above it for `n > 0`, and below it for `n < 0`. With
/// `x = m 2^e` and `m` odd, `x^n` is on the grid when `m^n` has at most
/// `p + 3` bits for `n > 0`, and when `m = 1` for `n < 0`.
fn near_dominant<L: Elementary>(
    x: &Argument<L>,
    n: i64,
    target: &Target,
) -> Option<Unrounded<L::Second>> {
    let zeros = (0..L::BITS)
        .find(|&position| x.significand.bit(position))
        .expect("the argument is not zero");
    let odd = x.significand.shr(zeros).resize::<L::Second>();
    let scale = (i128::from(x.exponent) + i128::from(zeros)) * i128::from(n);
    let kept = target.precision + 3;
    let one = L::Second::ZERO.with_bit(0);
    let power = if odd == one {
        one
    } else if n < 0
        || u64::from(odd.bit_length()).saturating_mul(n.unsigned_abs()) > u64::from(L::Second::BITS)
    {
        // A quotient by `m^|n|` is not dyadic. A power past the width has
        // more than half the width in bits, past `p + 3`.
        return None;
    } else {
        integer_power(odd, n.unsigned_abs())
    };
    let width = power.bit_length();
    if width > kept {
        return None;
    }
    let shift = kept - width;
    Some(if n > 0 {
        Unrounded {
            negative: false,
            exponent: clamped(scale - i128::from(shift), target),
            significand: power.shl(shift),
            sticky: true,
        }
    } else {
        // Just below the power of two `2^(e n)`: all ones in the binade below.
        Unrounded {
            negative: false,
            exponent: clamped(scale - i128::from(kept), target),
            significand: L::Second::ones(kept),
            sticky: true,
        }
    })
}

/// The function `(1 + x)^n`, as `e^(n ln(1 + x))`.
struct Compound<L>(Argument<L>, i64);

impl<L: Limbs> Function for Compound<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let t = log_term::<L, W>(&self.0, self.1)?;
        // `series::exp` takes an argument below 2^30. `compound` sends an
        // argument past `4 * range`, below 2^26, out of range first.
        (t.center().top() < 30).then(|| series::exp(&t))
    }
}

/// Returns the ball of `n ln(1 + x)` at the width `W`.
fn log_term<L: Limbs, W: Widen>(x: &Argument<L>, n: i64) -> Option<Ball<W>> {
    let argument = Ball::new(x.negative, x.significand, i64::from(x.exponent));
    let sum = Ball::<W>::one().add(&argument);
    Some(series::log(&sum)?.mul(&Ball::integer(n)))
}

/// Returns `(1 + x)^n` exactly, or truncated with the sticky bit for `n < 0`,
/// when `1 + x` is a power of two, or when the power of its odd part fits the
/// second working width with room for the quotient. Returns `None`
/// otherwise.
fn exact_power<L: Elementary>(
    x: &Argument<L>,
    n: i64,
    target: &Target,
) -> Option<Unrounded<L::Second>> {
    let (odd, exponent) = one_plus::<L, L::Second>(x)?;
    let scale = i128::from(exponent) * i128::from(n);
    let one = L::Second::ZERO.with_bit(0);
    if odd == one {
        return Some(Unrounded {
            negative: false,
            exponent: clamped(scale, target),
            significand: one,
            sticky: false,
        });
    }
    let precision = target.precision;
    let room = if n > 0 {
        L::Second::BITS
    } else {
        L::Second::BITS - precision - 3
    };
    let count = n.unsigned_abs();
    if u64::from(odd.bit_length()).saturating_mul(count) > u64::from(room) {
        return None;
    }
    let power = integer_power(odd, count);
    let width = power.bit_length();
    Some(if n > 0 {
        let dropped = width.saturating_sub(precision + 3);
        Unrounded {
            negative: false,
            exponent: clamped(scale + i128::from(dropped), target),
            significand: power.shr(dropped),
            sticky: power.any_below(dropped),
        }
    } else {
        // `2^s / M^|n|` lies in `(2^(p + 2), 2^(p + 3))`, because `M^|n|` is
        // not a power of two.
        let shift = width + precision + 2;
        let (quotient, rest) = limbs::divide(L::Second::ZERO.with_bit(shift), power);
        Unrounded {
            negative: false,
            exponent: clamped(scale - i128::from(shift), target),
            significand: quotient,
            sticky: !rest.is_zero(),
        }
    })
}

/// Returns `1 + x` as `M 2^E` with `M` odd, when `1 + x` fits the width `W`
/// with a bit to spare. `x` is above -1.
fn one_plus<L: Limbs, W: Limbs>(x: &Argument<L>) -> Option<(W, i64)> {
    let length = x.significand.bit_length();
    let exponent = i64::from(x.exponent);
    let (sum, lowest) = if exponent >= 0 {
        // `x` is at least 1, so it is positive: `1 + x` is the integer
        // `m 2^e + 1`.
        debug_assert!(
            !x.negative,
            "an argument above -1 and at least 1 is positive"
        );
        if i64::from(length) + exponent >= i64::from(W::BITS) - 1 {
            return None;
        }
        let shift = u32::try_from(exponent).ok()?;
        (x.significand.resize::<W>().shl(shift).increment(), 0)
    } else {
        // `1 + x` is `(2^k ± m) 2^-k`, and `m < 2^k`, because `|x| < 1`.
        let shift = u32::try_from(-exponent).ok()?;
        if shift >= W::BITS - 1 {
            return None;
        }
        let power = W::ZERO.with_bit(shift);
        let m = x.significand.resize::<W>();
        let sum = if x.negative {
            power.sub(m)
        } else {
            power.add(m)
        };
        (sum, exponent)
    };
    let zeros = (0..W::BITS)
        .find(|&position| sum.bit(position))
        .expect("1 + x is not zero");
    Some((sum.shr(zeros), lowest + i64::from(zeros)))
}

/// Returns `base^count`, which must fit `W`, by squaring.
fn integer_power<W: Limbs>(base: W, count: u64) -> W {
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
fn clamped(exponent: i128, target: &Target) -> i32 {
    let bound = 2 * i128::from(target.range) + i128::from(target.precision);
    i32::try_from(exponent.clamp(-bound, bound)).expect("the bound fits an i32")
}
