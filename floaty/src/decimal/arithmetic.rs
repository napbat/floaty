//! Add, subtract, multiply, divide, square root, and fused multiply-add for
//! the decimal formats.
//!
//! Each operation handles the special values, computes an exact result, and
//! rounds it once with the preferred exponent that IEEE 754-2019 Table 5.1
//! gives the operation. A sum of values far apart cuts the smaller term below
//! the round digit, and keeps one more digit that stands for the lost part,
//! as the binary engine jams its lowest bit.

use super::digits::{self, digit_count, power_of_ten};
use super::{DecimalLayout, Wide};
use crate::env::{Behavior, Flags};
use crate::exact::Unrounded;
use crate::format::{DecimalEncoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

/// A term of a sum: `coefficient * 10^exponent`. A zero term has a zero
/// coefficient.
#[derive(Clone, Copy)]
struct Term<L> {
    negative: bool,
    exponent: i64,
    coefficient: L,
}

impl<L: Limbs> Term<L> {
    /// Returns the adjusted exponent: the weight of the leading digit.
    fn top(&self) -> Option<i64> {
        let digits = digit_count(&self.coefficient);
        (digits > 0).then(|| self.exponent + i64::from(digits) - 1)
    }

    /// Returns the term as an exact value.
    fn exact(&self) -> Unrounded<L> {
        Unrounded {
            negative: self.negative,
            exponent: narrow(self.exponent),
            significand: self.coefficient,
            sticky: false,
        }
    }
}

/// The exact sum of two terms.
enum Sum<L> {
    /// A nonzero sum. Its lowest digit can stand for a lost part.
    Value(Unrounded<L>),
    /// An exact zero.
    Zero,
}

/// Returns an exponent that fits an `i32`.
#[inline]
fn narrow(exponent: i64) -> i32 {
    i32::try_from(exponent).expect("an exponent of an exact decimal result fits an i32")
}

/// Adds two terms.
///
/// A zero term leaves the other term unchanged. Otherwise the term with the
/// higher leading digit dominates, and the sum keeps it exactly. The other
/// term is exact too, unless its leading digit is at least 2 places lower
/// and it has digits below `c`: the lower of the dominant exponent and
/// `precision` plus 1 places below the dominant leading digit. Then it is cut
/// at `c`, and a digit of 1 below `c` stands for the part that the cut lost.
/// The sum has its leading digit at most 1 place below the dominant one, so
/// its round digit is at `c` or above. The true sum and the cut sum then lie
/// strictly between the same two consecutive multiples of `10^c`, so they
/// round alike.
fn sum<L: Limbs>(first: Term<L>, second: Term<L>, precision: u32) -> Sum<L> {
    let (dominant, dominant_top, other, other_top) = match (first.top(), second.top()) {
        (None, None) => return Sum::Zero,
        (Some(_), None) => return Sum::Value(first.exact()),
        (None, Some(_)) => return Sum::Value(second.exact()),
        (Some(a), Some(b)) if a >= b => (first, a, second, b),
        (Some(a), Some(b)) => (second, b, first, a),
    };
    let cut = dominant
        .exponent
        .min(dominant_top - i64::from(precision) - 1);
    let jam = other_top <= dominant_top - 2 && other.exponent < cut;
    let common = if jam {
        cut - 1
    } else {
        dominant.exponent.min(other.exponent)
    };
    let scale = |term: &Term<L>| {
        let shift = u32::try_from(term.exponent - common)
            .expect("a term is at or above the common exponent");
        limbs::multiply_fit(term.coefficient, power_of_ten(shift))
    };
    let dominant_digits = scale(&dominant);
    let other_digits = if jam {
        let shift = u32::try_from(cut - other.exponent).unwrap_or(u32::MAX);
        let (kept, lost) = divide_by_power(other.coefficient, shift);
        limbs::multiply_small(kept, 10).add(limbs::from_u128(u128::from(lost)))
    } else {
        scale(&other)
    };
    let (negative, coefficient) = if dominant.negative == other.negative {
        (dominant.negative, dominant_digits.add(other_digits))
    } else {
        match dominant_digits.compare(&other_digits) {
            core::cmp::Ordering::Greater => (dominant.negative, dominant_digits.sub(other_digits)),
            core::cmp::Ordering::Less => (other.negative, other_digits.sub(dominant_digits)),
            core::cmp::Ordering::Equal => return Sum::Zero,
        }
    };
    Sum::Value(Unrounded {
        negative,
        exponent: narrow(common),
        significand: coefficient,
        sticky: false,
    })
}

/// Returns `true` when limbs `L` hold every result of `sum` of two terms of
/// at most `precision` digits.
///
/// Without a cut, the aligned terms have at most `precision + 2` digits.
/// After a cut, the dominant term has `precision + 3` digits, and the other
/// term fewer. So every sum is below `101 * 10^(precision + 1)`. Limbs of 128
/// bits or more hold every bound that a `u128` holds, and the compiler
/// rejects a bound that overflows a `u128`. The limbs of every decimal
/// format have at least 64 bits, which hold the bounds of decimal32 and
/// decimal64.
const fn holds_sums<L: Limbs>(precision: u32) -> bool {
    let bound = 101 * digits::power_of_ten_u128(precision + 1);
    L::BITS >= 128 || bound >> L::BITS == 0
}

/// Divides by `10^exponent`. Returns the quotient and `true` when the
/// remainder is not zero.
fn divide_by_power<L: Limbs>(value: L, exponent: u32) -> (L, bool) {
    if exponent > digit_count(&value) {
        return (L::ZERO, !value.is_zero());
    }
    let (quotient, remainder) = limbs::divide(value, power_of_ten(exponent));
    (quotient, !remainder.is_zero())
}

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the term of a zero or a finite value, in the limbs `Out`.
    fn term<L: Limbs, Out: Limbs>(value: &Unpacked<L>) -> Term<Out> {
        match *value {
            Unpacked::Zero { negative, exponent } => Term {
                negative,
                exponent: i64::from(exponent),
                coefficient: Out::ZERO,
            },
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => Term {
                negative,
                exponent: i64::from(exponent),
                coefficient: significand.resize(),
            },
            _ => unreachable!("only a zero or a finite value is a term"),
        }
    }

    /// Returns the exact product of two zero or finite values, with the sign
    /// `negative`.
    #[inline]
    fn product<L: Widen>(x: &Unpacked<L>, y: &Unpacked<L>, negative: bool) -> Term<Wide<L>> {
        Term {
            negative,
            exponent: Self::exponent_of(x) + Self::exponent_of(y),
            coefficient: limbs::multiply_fit(
                Self::term::<L, Wide<L>>(x).coefficient,
                Self::term::<L, Wide<L>>(y).coefficient,
            ),
        }
    }

    /// Returns the exponent of a zero or a finite value.
    fn exponent_of<L>(value: &Unpacked<L>) -> i64 {
        match *value {
            Unpacked::Zero { exponent, .. } | Unpacked::Finite { exponent, .. } => {
                i64::from(exponent)
            }
            _ => unreachable!("only a zero or a finite value has an exponent"),
        }
    }

    /// Rounds a sum with a preferred exponent, or encodes an exact zero sum.
    #[inline]
    fn finish_sum<L: Widen, In: Limbs, B: Behavior>(
        sum: &Sum<In>,
        negative_zero: bool,
        preferred: i64,
        behavior: B,
        flags: Flags,
    ) -> (L, Flags) {
        match sum {
            Sum::Value(value) => Self::finish(value, preferred, behavior, flags),
            Sum::Zero => Self::exact(Self::zero(negative_zero, preferred), flags),
        }
    }

    /// Adds `left` and `right`, or subtracts `right` when `subtract` is set.
    /// The preferred exponent is the smaller exponent of the operands.
    pub fn add<L: Widen, B: Behavior>(
        left: L,
        right: L,
        subtract: bool,
        behavior: B,
    ) -> (L, Flags) {
        let env = &behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        let y = if subtract { y.negate() } else { y };
        match (x, y) {
            (Unpacked::Infinity { negative: a }, Unpacked::Infinity { negative: b }) if a != b => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) => Self::exact(x, flags),
            (_, Unpacked::Infinity { .. }) => Self::exact(y, flags),
            _ => {
                let preferred = Self::exponent_of(&x).min(Self::exponent_of(&y));
                let (a, b) = (Self::term::<L, L>(&x), Self::term::<L, L>(&y));
                // An exact zero sum keeps the sign that the signs share, or
                // takes the sign of the rounding direction.
                let zero_sign = env.zero_sum_sign(a.negative, b.negative);
                // The sum stays in the limbs of the operands.
                const { assert!(holds_sums::<L>(Self::PRECISION), "the limbs hold every sum") };
                let sum = sum(a, b, Self::PRECISION);
                Self::finish_sum(&sum, zero_sign, preferred, behavior, flags)
            }
        }
    }

    /// Multiplies `left` by `right`. The preferred exponent is the sum of the
    /// exponents.
    pub fn mul<L: Widen, B: Behavior>(left: L, right: L, behavior: B) -> (L, Flags) {
        let env = &behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        let negative = x.is_negative() != y.is_negative();
        match (x, y) {
            (Unpacked::Infinity { .. }, Unpacked::Zero { .. })
            | (Unpacked::Zero { .. }, Unpacked::Infinity { .. }) => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) | (_, Unpacked::Infinity { .. }) => {
                Self::exact(Unpacked::Infinity { negative }, flags)
            }
            _ => {
                let product = Self::product(&x, &y, negative);
                let value = Unrounded {
                    negative,
                    exponent: narrow(product.exponent),
                    significand: product.coefficient,
                    sticky: false,
                };
                Self::finish(&value, product.exponent, behavior, flags)
            }
        }
    }

    /// Divides `left` by `right`. The preferred exponent is the exponent of
    /// `left` less the exponent of `right`.
    pub fn div<L: Widen, B: Behavior>(left: L, right: L, behavior: B) -> (L, Flags) {
        let env = &behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        let negative = x.is_negative() != y.is_negative();
        let (lowest, _) = Self::TARGET.exponent_range();
        match (x, y) {
            (Unpacked::Infinity { .. }, Unpacked::Infinity { .. })
            | (Unpacked::Zero { .. }, Unpacked::Zero { .. }) => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) => Self::exact(Unpacked::Infinity { negative }, flags),
            // A finite value over an infinity is a zero with the smallest
            // exponent, as decTest gives it.
            (_, Unpacked::Infinity { .. }) => Self::exact(Self::zero(negative, lowest), flags),
            (_, Unpacked::Zero { .. }) => Self::exact(
                Unpacked::Infinity { negative },
                flags | Flags::DIVIDE_BY_ZERO,
            ),
            (Unpacked::Zero { .. }, _) => {
                let exponent = Self::exponent_of(&x) - Self::exponent_of(&y);
                Self::exact(Self::zero(negative, exponent), flags)
            }
            _ => {
                let (dividend, divisor) = (
                    Self::term::<L, Wide<L>>(&x).coefficient,
                    Self::term::<L, Wide<L>>(&y).coefficient,
                );
                let precision = env.precision_within(Self::TARGET.precision);
                // Scale the dividend so that the quotient has at least
                // precision + 1 digits.
                let scale = (i64::from(precision) + 1 + i64::from(digit_count(&divisor))
                    - i64::from(digit_count(&dividend)))
                .max(0);
                let scale = u32::try_from(scale).expect("the scale is small");
                let numerator = limbs::multiply_fit(dividend, power_of_ten(scale));
                let (quotient, remainder) = limbs::divide(numerator, divisor);
                let preferred = Self::exponent_of(&x) - Self::exponent_of(&y);
                let value = Unrounded {
                    negative,
                    exponent: narrow(preferred - i64::from(scale)),
                    significand: quotient,
                    sticky: !remainder.is_zero(),
                };
                Self::finish(&value, preferred, behavior, flags)
            }
        }
    }

    /// Returns the square root. The preferred exponent is half the exponent
    /// of the operand, rounded down.
    pub fn sqrt<L: Widen, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
        let env = &behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, env, &mut flags);
        if let Some((result, special)) = nan::special_unary(&x, env) {
            return Self::exact(result, flags | special);
        }
        match x {
            Unpacked::Infinity { negative: false } => Self::exact(x, flags),
            Unpacked::Zero { negative, exponent } => Self::exact(
                Self::zero(negative, i64::from(exponent).div_euclid(2)),
                flags,
            ),
            Unpacked::Infinity { negative: true } | Unpacked::Finite { negative: true, .. } => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            Unpacked::Finite { exponent, .. } => {
                let coefficient = Self::term::<L, Wide<L>>(&x).coefficient;
                let precision = env.precision_within(Self::TARGET.precision);
                let exponent = i64::from(exponent);
                // Scale to at least 2 * precision + 2 digits and an even
                // exponent, so that the root has precision + 1 digits.
                let mut scale =
                    (2 * i64::from(precision) + 2 - i64::from(digit_count(&coefficient))).max(0);
                if (exponent - scale).rem_euclid(2) != 0 {
                    scale += 1;
                }
                let shift = u32::try_from(scale).expect("the scale is small");
                let radicand = limbs::multiply_fit(coefficient, power_of_ten(shift));
                let (root, inexact) = limbs::square_root(radicand);
                let result = Unrounded {
                    negative: false,
                    exponent: narrow((exponent - scale) / 2),
                    significand: root,
                    sticky: inexact,
                };
                Self::finish(&result, exponent.div_euclid(2), behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN")
            }
        }
    }

    /// Returns `left * right + addend`, rounded once. The preferred exponent
    /// is the smaller of the product exponent and the addend exponent. The NaN
    /// cases follow `nan::fused_special`, as in the binary engine.
    pub fn mul_add<L: Widen, B: Behavior>(left: L, right: L, addend: L, behavior: B) -> (L, Flags) {
        let env = &behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        let z = Self::operand(addend, env, &mut flags);
        if let Some((value, special)) = nan::fused_special(&x, &y, &z, env) {
            return Self::exact(value, flags | special);
        }
        let product_negative = x.is_negative() != y.is_negative();
        let product_infinite =
            matches!(x, Unpacked::Infinity { .. }) || matches!(y, Unpacked::Infinity { .. });
        match z {
            _ if product_infinite => match z {
                Unpacked::Infinity { negative } if negative != product_negative => {
                    Self::exact(default_nan(env), flags | Flags::INVALID)
                }
                _ => Self::exact(
                    Unpacked::Infinity {
                        negative: product_negative,
                    },
                    flags,
                ),
            },
            Unpacked::Infinity { .. } => Self::exact(z, flags),
            _ => {
                let product = Self::product(&x, &y, product_negative);
                let addend = Self::term::<L, Wide<L>>(&z);
                let preferred = product.exponent.min(addend.exponent);
                let zero_sign = env.zero_sum_sign(product.negative, addend.negative);
                Self::finish_sum(
                    &sum(product, addend, Self::PRECISION),
                    zero_sign,
                    preferred,
                    behavior,
                    flags,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
    use crate::exact::Exact;
    use crate::float::{D64Bid, D64Dpd, Decoded};

    #[test]
    fn the_largest_sum_stays_in_one_limb() {
        // 9999999999999999E3 + 9999999999999999E0 cuts the second term. The
        // cut sum, 10009999999999998991 at 10^0, is the largest of decimal64
        // and uses 64 bits. It rounds up to 1001000000000000E4.
        let largest = 9_999_999_999_999_999;
        let (sum, flags) = number(false, largest, 3).add_with(number(false, largest, 0), Env::IEEE);
        assert_eq!(parts(sum), (1_001_000_000_000_000, 4));
        assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
    }

    /// Returns `coefficient * 10^exponent` in decimal64, exactly.
    fn number(negative: bool, coefficient: u64, exponent: i32) -> D64Bid {
        let exact = Exact {
            negative,
            exponent,
            significand: [coefficient],
            sticky: false,
        };
        let (value, flags) = D64Bid::round(exact, Env::IEEE);
        assert_eq!(flags, Flags::NONE, "the value is exact");
        value
    }

    fn parts(value: D64Bid) -> (u64, i32) {
        match value.decode::<1>() {
            Decoded::Finite {
                exponent,
                significand: [coefficient],
                ..
            } => (coefficient, exponent),
            Decoded::Zero { exponent, .. } => (0, exponent),
            other => panic!("{other:?} is not a number"),
        }
    }

    #[test]
    fn results_take_the_preferred_exponent() {
        let one = number(false, 1, 0);
        assert_eq!(parts(one + one), (2, 0));
        assert_eq!(
            parts(number(false, 575, -2) + number(false, 33, -1)),
            (905, -2)
        );
        // 1.0 + 1.00 = 2.00, and 1.00 / 2 = 0.50.
        assert_eq!(
            parts(number(false, 10, -1) + number(false, 100, -2)),
            (200, -2)
        );
        assert_eq!(
            parts(number(false, 100, -2) / number(false, 2, 0)),
            (50, -2)
        );
        assert_eq!(parts(one / number(false, 4, 0)), (25, -2));
        assert_eq!(parts(number(false, 2, 0) / number(false, 2, 0)), (1, 0));
        assert_eq!(parts(number(false, 15, -1) * number(false, 2, 0)), (30, -1));
        assert_eq!(parts(number(false, 4, 0).sqrt()), (2, 0));
        assert_eq!(parts(number(false, 25, -2).sqrt()), (5, -1));
        assert_eq!(
            parts(number(false, 2, 0).mul_add(number(false, 3, 0), one)),
            (7, 0)
        );
        // x + 0E-10 pads x with zeros toward the smaller exponent.
        assert_eq!(
            parts(number(false, 123, -2) + number(false, 0, -10)),
            (12_300_000_000, -10)
        );
    }

    #[test]
    fn inexact_results_keep_every_digit() {
        let (third, flags) = number(false, 1, 0).div_with(number(false, 3, 0), Env::IEEE);
        assert_eq!(parts(third), (3_333_333_333_333_333, -16));
        assert_eq!(flags, Flags::INEXACT);
        assert_eq!(
            parts(number(false, 2, 0).sqrt()),
            (1_414_213_562_373_095, -15)
        );
        let largest = number(false, 9_999_999_999_999_999, 369);
        let (infinity, flags) = largest.add_with(number(false, 1, 369), Env::IEEE);
        assert!(infinity.is_infinite());
        assert_eq!(flags, Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP);
        // A far smaller addend only rounds: 1 + 1E-30 = 1 in each direction
        // but toward positive and away from zero, which give the next value up.
        let tiny = number(false, 1, -30);
        assert_eq!(
            parts(number(false, 1, 0) + tiny),
            (1_000_000_000_000_000, -15)
        );
        let (up, _) = number(false, 1, 0).add_with(tiny, crate::Rounding::TowardPositive);
        assert_eq!(parts(up), (1_000_000_000_000_001, -15));
        let (away, _) = number(false, 1, 0).add_with(tiny, crate::Rounding::AwayFromZero);
        assert_eq!(parts(away), (1_000_000_000_000_001, -15));
        let (down, _) = number(false, 1, 0).sub_with(tiny, crate::Rounding::TowardZero);
        assert_eq!(parts(down), (9_999_999_999_999_999, -16));
    }

    #[test]
    fn a_zero_term_leaves_the_other_term_unchanged() {
        let (sum, flags) = number(false, 1, 0).add_with(number(false, 0, 200), Env::IEEE);
        assert_eq!((parts(sum), flags), ((1, 0), Flags::NONE));
        // 1E-398, the smallest subnormal value.
        let smallest = D64Bid::from_bits(1);
        let (sum, flags) = smallest.add_with(number(false, 0, 369), Env::IEEE);
        let subnormal = Flags::DENORMAL_INPUT | Flags::TINY;
        assert_eq!((parts(sum), flags), ((1, -398), subnormal));
        let zero_product = number(false, 0, 300).mul_add(number(false, 0, 60), number(false, 1, 0));
        assert_eq!(parts(zero_product), (1, 0));
    }

    #[test]
    fn a_fused_multiply_add_keeps_the_product_exact() {
        let x = number(false, 1_234_567_890_123_456, 0);
        // Cancellation leaves the low digits of the product.
        let (difference, flags) =
            x.mul_add_with(x, number(true, 1_524_157_875_323_881, 15), Env::IEEE);
        assert_eq!(
            (parts(difference), flags),
            ((726_870_921_383_936, 0), Flags::NONE)
        );
        // A digit of the addend meets the low digits of the product without a
        // false tie: 99999999999999994999900000000001 rounds down.
        let nines = number(false, 9_999_999_999_999_999, 0);
        let (sum, flags) = nines.mul_add_with(nines, number(false, 149_999, 11), Env::IEEE);
        assert_eq!(parts(sum), (9_999_999_999_999_999, 16));
        assert_eq!(flags, Flags::INEXACT);
    }

    #[test]
    fn a_far_smaller_term_keeps_the_sum_inexact() {
        use crate::float::D32Bid;
        let value = |negative, coefficient: u64, exponent| {
            let exact = Exact {
                negative,
                exponent,
                significand: [coefficient],
                sticky: false,
            };
            D32Bid::round(exact, Env::IEEE).0
        };
        // 0.9999999 * 0.9999999 - 1E-64, toward positive.
        let x = value(false, 9_999_999, -7);
        let (sum, flags) = x.mul_add_with(x, value(true, 1, -64), crate::Rounding::TowardPositive);
        let expected = value(false, 9_999_999, -7);
        assert_eq!(sum.to_bits(), expected.to_bits());
        assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
    }

    #[test]
    fn a_precision_limit_does_not_cut_close_terms() {
        let limit = Env::IEEE.with_precision(core::num::NonZeroU32::new(3));
        let (difference, flags) = number(false, 1_234_567_890_123_456, 0)
            .sub_with(number(false, 1_234_567_890_123_000, 0), limit);
        assert_eq!((parts(difference), flags), ((456, 0), Flags::NONE));
        let up = limit.with_rounding(crate::Rounding::TowardPositive);
        let (difference, flags) = number(true, 1_000_000_000_000_000, -36)
            .sub_with(number(true, 999_999_999_999_997, -36), up);
        assert_eq!(difference.to_bits(), number(true, 3, -36).to_bits());
        assert_eq!(flags, Flags::NONE);
    }

    #[test]
    fn dpd_computes_the_same_values() {
        let one = D64Dpd::from_bits(0x2238_0000_0000_0001);
        assert_eq!((one + one).to_bits(), 0x2238_0000_0000_0002);
    }
}
