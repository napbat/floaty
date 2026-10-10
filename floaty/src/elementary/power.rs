//! `pow(x, y)` and `powr(x, y)` of IEEE 754-2019 section 9.2: `x^y`. `pow`
//! takes a negative `x` with an integer `y`, and `powr` is `e^(y ln x)` for
//! `x >= 0` alone.
//!
//! The special cases of section 9.2.1 give an exact result, an infinity, a
//! pole, or invalid. Two finite nonzero arguments give `|x|^y`, negative for
//! `pow` of a negative `x` and an odd `y`.
//!
//! `|x|^y` is rational only when `x^(1/d)` is, with `d` the denominator of
//! `y` in lowest terms: a power of 2, or `2^i 5^j` in decimal. A rational
//! value with more than `t'` digits, the digits of the truncation, lies off
//! its grid, and so does an infinite decimal fraction. So [`exact_power`]
//! gives the rational values with at most `t'` digits exactly, and a ball
//! decides every other value.
//!
//! The ball is `e^t` with `t = y ln |x|`. A `t` below `2^-(p + 5)`, or below
//! `10^-(p + 3)`, gives a value just beside 1, and a `t` past four times the
//! range gives an overflow or an underflow, as for `e^x`.

use core::cmp::Ordering;

use super::ball::Ball;
use super::bivariate::{Outcome, Point, point};
use super::exact::{clamped, exact, integer_power, radix_power, stripped};
use super::{
    Argument, Elementary, Function, Radix, Target, argument, compare_one, natural_log, near_one,
    out_of_range, power as radix_power_of, series, ziv_from,
};
use crate::exact::Unrounded;
use crate::limbs::{self, Divisor, Limbs, Widen};
use crate::unpacked::Unpacked;

/// Returns `pow(x, y)`, or `powr(x, y)` when `powr` is set, for operands
/// that are not NaNs.
pub(super) fn power<L: Elementary>(
    powr: bool,
    x: &Unpacked<L>,
    y: &Unpacked<L>,
    target: &Target,
) -> Outcome<L::Second> {
    let radix = target.radix;
    let (x_negative, x) = point(x);
    let (y_negative, y) = point(y);
    let one = || {
        Outcome::Truncated(exact(
            false,
            <L::Second as Limbs>::ZERO.with_bit(0),
            0,
            target,
        ))
    };
    let parity = match &y {
        Point::Finite(y) => parity::<L>(y, radix),
        Point::Zero | Point::Infinity => None,
    };
    let odd = parity == Some(true) && !powr;
    if matches!(y, Point::Zero) {
        // `powr(x, ±0)` is 1 for a finite `x > 0` alone.
        let positive = matches!(x, Point::Finite(_)) && !x_negative;
        return if powr && !positive {
            Outcome::Invalid
        } else {
            one()
        };
    }
    if powr && x_negative && !matches!(x, Point::Zero) {
        return Outcome::Invalid;
    }
    match x {
        Point::Zero => {
            let negative = x_negative && odd;
            match (y, y_negative) {
                (Point::Infinity, true) => Outcome::Infinity { negative: false },
                (_, true) => Outcome::Pole { negative },
                (_, false) => Outcome::Zero { negative },
            }
        }
        Point::Infinity => {
            // `pow(-inf, y)` is `pow(-0, -y)`.
            let negative = x_negative && odd;
            if y_negative {
                Outcome::Zero { negative }
            } else {
                Outcome::Infinity { negative }
            }
        }
        Point::Finite(x) => {
            let order = compare_one(&x, radix);
            match y {
                Point::Infinity => match (order, y_negative) {
                    (Ordering::Equal, _) if powr => Outcome::Invalid,
                    (Ordering::Equal, _) => one(),
                    (Ordering::Less, true) | (Ordering::Greater, false) => {
                        Outcome::Infinity { negative: false }
                    }
                    (Ordering::Less, false) | (Ordering::Greater, true) => {
                        Outcome::Zero { negative: false }
                    }
                },
                Point::Finite(y) => {
                    if x_negative && parity.is_none() {
                        return Outcome::Invalid;
                    }
                    let magnitude = Argument {
                        negative: false,
                        ..x
                    };
                    let value = if order == Ordering::Equal {
                        exact(false, <L::Second as Limbs>::ZERO.with_bit(0), 0, target)
                    } else {
                        magnitude_power::<L>(&magnitude, &y, target)
                    };
                    Outcome::Truncated(Unrounded {
                        negative: x_negative && odd,
                        ..value
                    })
                }
                Point::Zero => unreachable!("a zero exponent gives 1 or invalid above"),
            }
        }
    }
}

/// Returns `Some(true)` for an odd integer, `Some(false)` for an even one,
/// and `None` for a value with a fraction.
fn parity<L: Elementary>(y: &Argument<L>, radix: Radix) -> Option<bool> {
    let exponent = i64::from(y.exponent);
    if exponent > 0 {
        return Some(false);
    }
    let shift = u32::try_from(-exponent).ok()?;
    match radix {
        Radix::Binary => {
            if shift >= L::BITS || y.significand.any_below(shift) {
                return None;
            }
            Some(y.significand.bit(shift))
        }
        Radix::Decimal => {
            let unit = radix_power::<L::Second>(radix, u64::from(shift))?;
            let (quotient, rest) = limbs::divide(y.significand.resize::<L::Second>(), unit);
            rest.is_zero().then(|| quotient.bit(0))
        }
    }
}

/// Returns the truncation of `|x|^y` for a positive `x` other than 1 and a
/// finite nonzero `y`.
fn magnitude_power<L: Elementary>(
    x: &Argument<L>,
    y: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    if let Some(value) = exact_power::<L>(x, y, target) {
        return value;
    }
    let value = Value {
        x: *x,
        y: *y,
        radix: target.radix,
    };
    // The ball of `t` at the first width decides the shortcuts, and its
    // `e^t` is the first step of Ziv's strategy.
    let t = value
        .exponent::<L::First>()
        .expect("the first width bounds y ln x");
    // `t` is positive when `x` lies above 1 and `y` above 0, or both below.
    let positive = (compare_one(x, target.radix) == Ordering::Greater) != y.negative;
    let precision = i64::from(target.precision);
    let tiny = match target.radix {
        Radix::Binary => precision + 5,
        // `2^-(4p + 12)` lies below `10^-(p + 3)`.
        Radix::Decimal => 4 * (precision + 3),
    };
    if t.upper().ceiling_log2() < -tiny {
        return near_one(!positive, target);
    }
    // `|t|` from `2^k`, the power of two above `4 R`, puts `e^t` past
    // `2^(5 R)` and `10^(1.7 R)`, past the range. Below it, `|t|` lies below
    // 2^30, which `exp` takes.
    let range = 4 * u64::from(target.range);
    let center = t.center();
    if !center.is_zero() && center.top() >= i64::from(64 - range.leading_zeros()) {
        return out_of_range(positive, target);
    }
    ziv_from::<L, _>(Some(series::exp(&t)), &value, target)
}

/// `|x|^y` at a positive `x` and a finite nonzero `y`.
struct Value<L> {
    /// The base.
    x: Argument<L>,
    /// The exponent.
    y: Argument<L>,
    /// The radix of the arguments.
    radix: Radix,
}

impl<L: Limbs> Value<L> {
    /// Returns the ball of `t = y ln x` at the width `W`.
    fn exponent<W: Widen>(&self) -> Option<Ball<W>> {
        Some(argument::<L, W>(&self.y, self.radix)?.mul(&natural_log::<L, W>(&self.x, self.radix)?))
    }
}

impl<L: Limbs> Function for Value<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        Some(series::exp(&self.exponent::<W>()?))
    }
}

/// Returns `|x|^y` exactly when it is rational with at most `t'` digits, or
/// `None` for a value off the grid of the truncation.
fn exact_power<L: Elementary>(
    x: &Argument<L>,
    y: &Argument<L>,
    target: &Target,
) -> Option<Unrounded<L::Second>> {
    let radix = target.radix;
    let (base, base_exponent) = stripped(
        x.significand.resize::<L::Second>(),
        i64::from(x.exponent),
        radix,
    );
    let (numerator, exponent) = stripped(
        y.significand.resize::<L::Second>(),
        i64::from(y.exponent),
        radix,
    );
    // `y = ±m RADIX^f` with `m` free of the radix: an integer for `f >= 0`,
    // and otherwise `m' / d` in lowest terms.
    let (count, denominator) = match radix {
        Radix::Binary => {
            if exponent >= 0 {
                (scaled_count(&numerator, exponent, 2)?, 1)
            } else {
                let shift = u32::try_from(-exponent).ok().filter(|&shift| shift < 32)?;
                (u128::from(small(&numerator)?), 1_u64 << shift)
            }
        }
        Radix::Decimal => decimal_exponent(&numerator, exponent)?,
    };
    let (root, root_exponents) = match radix {
        Radix::Binary => binary_root(base, base_exponent, denominator)?,
        Radix::Decimal => decimal_root(base, base_exponent, denominator)?,
    };
    raise(root, root_exponents, count, y.negative, target)
}

/// Returns `m RADIX^f` for `f >= 0` as a count, or `None` past 2^64, where
/// the power has too many digits or lies far past the range.
fn scaled_count<W: Limbs>(numerator: &W, exponent: i64, radix: u64) -> Option<u128> {
    let mut count = u128::from(small(numerator)?);
    for _ in 0..exponent {
        count = count
            .checked_mul(u128::from(radix))
            .filter(|&count| count >> 64 == 0)?;
    }
    Some(count)
}

/// Returns a value below 2^64 as a `u64`.
fn small<W: Limbs>(value: &W) -> Option<u64> {
    (value.bit_length() <= 64).then(|| value.limb(0))
}

/// Returns the decimal exponent `±m 10^f` as `(m', d)` with `y = m' / d` in
/// lowest terms, or `None` for a count past 2^64.
fn decimal_exponent<W: Limbs>(numerator: &W, exponent: i64) -> Option<(u128, u64)> {
    if exponent >= 0 {
        return Some((scaled_count(numerator, exponent, 10)?, 1));
    }
    let shift = u32::try_from(-exponent).ok().filter(|&shift| shift < 16)?;
    let mut count = small(numerator)?;
    let mut denominator = 10_u64.pow(shift);
    // `m` has no factor 10, so only factors 2 or 5 of it cancel.
    for factor in [2, 5] {
        while count % factor == 0 && denominator % factor == 0 {
            count /= factor;
            denominator /= factor;
        }
    }
    Some((u128::from(count), denominator))
}

/// The `d`-th root of `|x|`: an odd integer, or one free of 2 and 5, and
/// the exponents of 2 and of 5 that multiply it.
type Root<W> = (W, [i128; 2]);

/// Returns the `d`-th root of a binary `|x| = a 2^e`, with `a` odd, when it
/// is rational: `a` a `d`-th power and `d` dividing `e`.
fn binary_root<W: Limbs>(base: W, exponent: i64, denominator: u64) -> Option<Root<W>> {
    let shift = denominator.trailing_zeros();
    if i128::from(exponent) % i128::from(denominator) != 0 {
        return None;
    }
    // A root of an odd `a` other than 1 is at least 3, so `a` lies at or
    // above `3^d`, past `2^d`.
    let one = base.bit_length() == 1;
    if !one && u64::from(base.bit_length()) <= denominator {
        return None;
    }
    let mut root = base;
    for _ in 0..shift {
        let (half, inexact) = limbs::square_root(root);
        if inexact {
            return None;
        }
        root = half;
    }
    let exponent = i128::from(exponent) / i128::from(denominator);
    Some((root, [exponent, 0]))
}

/// Returns the `d`-th root of a decimal `|x| = c 10^q`, with `c` free of
/// 10, when it is rational: `c = 2^i 5^j c''` with `c''` a `d`-th power and
/// `d` dividing `i + q` and `j + q`.
fn decimal_root<W: Limbs>(base: W, exponent: i64, denominator: u64) -> Option<Root<W>> {
    let twos = (0..W::BITS)
        .find(|&position| base.bit(position))
        .expect("the base is not zero");
    let mut rest = base.shr(twos);
    let mut fives = 0;
    loop {
        let (quotient, remainder) = limbs::divide_small(rest, Divisor::new(5));
        if remainder != 0 {
            break;
        }
        rest = quotient;
        fives += 1;
    }
    let exponents = [
        i128::from(twos) + i128::from(exponent),
        i128::from(fives) + i128::from(exponent),
    ];
    let d = i128::from(denominator);
    if exponents.iter().any(|&exponent| exponent % d != 0) {
        return None;
    }
    let root = integer_root(rest, denominator)?;
    Some((root, exponents.map(|exponent| exponent / d)))
}

/// Returns the `d`-th root of an integer below 2^128 when it is exact.
fn integer_root<W: Limbs>(value: W, denominator: u64) -> Option<W> {
    if denominator == 1 || (value.bit_length() == 1 && value.bit(0)) {
        return Some(value);
    }
    let value = small_u128(&value)?;
    // A root of at least 2 needs a value of at least `2^d`.
    let d = u32::try_from(denominator).ok().filter(|&d| d < 128)?;
    let (mut low, mut high) = (1_u128, 1_u128 << (128 / d + 1).min(64));
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        match middle.checked_pow(d) {
            Some(power) if power <= value => low = middle,
            _ => high = middle - 1,
        }
    }
    (low.checked_pow(d) == Some(value)).then(|| limbs::from_u128(low))
}

/// Returns a value below 2^128 as a `u128`.
fn small_u128<W: Limbs>(value: &W) -> Option<u128> {
    (value.bit_length() <= 128)
        .then(|| u128::from(value.limb(0)) | (u128::from(value.limb(1)) << 64))
}

/// Returns `(r 2^a 5^b)^(±n)` exactly when it has at most `t'` digits.
fn raise<W: Limbs>(
    root: W,
    [twos, fives]: [i128; 2],
    count: u128,
    negative: bool,
    target: &Target,
) -> Option<Unrounded<W>> {
    let one = root.bit_length() == 1 && root.bit(0);
    // A root other than 1 gives a power with at least `n` digits, and an
    // infinite fraction for a negative exponent.
    if !one && (negative || count > u128::from(target.kept())) {
        return None;
    }
    let count_signed = i128::try_from(count).ok()?;
    let signed = if negative {
        -count_signed
    } else {
        count_signed
    };
    let [twos, fives] = [twos.saturating_mul(signed), fives.saturating_mul(signed)];
    let mut significand = if one {
        root
    } else {
        let bits = u64::from(root.bit_length()) * u64::try_from(count).ok()?;
        if bits >= u64::from(W::BITS) / 2 {
            return None;
        }
        integer_power(root, u64::try_from(count).ok()?)
    };
    let exponent = match target.radix {
        Radix::Binary => twos,
        Radix::Decimal => {
            // `2^a 5^b` is `10^min(a, b)` times a power of 2 or of 5.
            let common = twos.min(fives);
            for (factor, power) in [(2, twos - common), (5, fives - common)] {
                for _ in 0..power.min(i128::from(W::BITS)) {
                    if significand.bit_length() + 3 >= W::BITS {
                        return None;
                    }
                    significand = limbs::multiply_small(significand, factor);
                }
                if power > i128::from(W::BITS) {
                    return None;
                }
            }
            common
        }
    };
    let limit = radix_power_of::<W>(target.radix, target.kept());
    if significand.compare(&limit) != Ordering::Less {
        return None;
    }
    Some(Unrounded {
        negative: false,
        exponent: clamped(exponent, target),
        significand,
        sticky: false,
    })
}
