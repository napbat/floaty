//! Add, subtract, multiply, divide, square root, and fused multiply-add for
//! the decimal formats.
//!
//! Each operation handles the special values, computes an exact result, and
//! rounds it once with the preferred exponent that IEEE 754-2019 Table 5.1
//! gives the operation. A sum of values far apart cuts the smaller term below
//! the round digit, and keeps one more digit that stands for the lost part,
//! as the binary engine jams its lowest bit.

use super::digits::{digit_count, power_of_ten};
use super::{DecimalLayout, Wide, round};
use crate::env::{Env, Flags, Rounding};
use crate::exact::Unrounded;
use crate::format::{DecimalEncoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

/// A term of a sum: `coefficient * 10^exponent`. A zero term has a zero
/// coefficient.
#[derive(Clone, Copy)]
struct Term {
    negative: bool,
    exponent: i64,
    coefficient: Wide,
}

impl Term {
    /// Returns the adjusted exponent: the weight of the leading digit.
    fn top(&self) -> Option<i64> {
        let digits = digit_count(&self.coefficient);
        (digits > 0).then(|| self.exponent + i64::from(digits) - 1)
    }

    /// Returns the term as an exact value.
    fn exact(&self) -> Unrounded<Wide> {
        Unrounded {
            negative: self.negative,
            exponent: narrow(self.exponent),
            significand: self.coefficient,
            sticky: false,
        }
    }
}

/// The exact sum of two terms.
enum Sum {
    /// A nonzero sum. Its lowest digit can stand for a lost part.
    Value(Unrounded<Wide>),
    /// An exact zero.
    Zero,
}

/// Returns an exponent that fits an `i32`.
fn narrow(exponent: i64) -> i32 {
    i32::try_from(exponent).expect("an exponent of an exact decimal result fits an i32")
}

/// Returns the sign of an exact zero sum of values with different signs:
/// negative only when rounding toward negative.
fn zero_sum_sign(env: &Env) -> bool {
    env.rounding == Rounding::TowardNegative
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
fn sum(first: Term, second: Term, precision: u32) -> Sum {
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
    let scale = |term: &Term| {
        let shift = u32::try_from(term.exponent - common)
            .expect("a term is at or above the common exponent");
        multiply(term.coefficient, power_of_ten(shift))
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

/// Returns `value * factor` of two values whose product fits.
fn multiply(value: Wide, factor: Wide) -> Wide {
    let product = value.widening_mul(factor);
    debug_assert!(product.bit_length() <= Wide::BITS, "the product fits");
    product.resize()
}

/// Divides by `10^exponent`. Returns the quotient and `true` when the
/// remainder is not zero.
fn divide_by_power(value: Wide, exponent: u32) -> (Wide, bool) {
    if exponent > digit_count(&value) {
        return (Wide::ZERO, !value.is_zero());
    }
    let (quotient, remainder) = limbs::divide(value, power_of_ten(exponent));
    (quotient, !remainder.is_zero())
}

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Decodes an operand, reports a subnormal operand, and applies DAZ. A
    /// zero from DAZ keeps the exponent of the subnormal value.
    pub(super) fn operand<L: Limbs>(bits: L, env: &Env, flags: &mut Flags) -> Unpacked<L> {
        let value = Self::decode(bits);
        let Unpacked::Finite {
            negative, exponent, ..
        } = value
        else {
            return value;
        };
        if Self::classify(bits) != crate::float::Class::Subnormal {
            return value;
        }
        *flags |= Flags::DENORMAL_INPUT;
        if env.denormals_are_zero {
            Unpacked::Zero { negative, exponent }
        } else {
            value
        }
    }

    /// Rounds an exact result with a preferred exponent, and encodes it.
    pub(super) fn finish<L: Limbs>(
        value: &Unrounded<Wide>,
        preferred: i64,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let preferred = narrow(preferred.clamp(i64::from(i32::MIN), i64::from(i32::MAX)));
        let (rounded, round_flags) = round::round::<Wide, L>(value, preferred, &Self::TARGET, env);
        (Self::encode(rounded), flags | round_flags)
    }

    /// Encodes a result that needs no rounding.
    pub(super) fn exact<L: Limbs>(value: Unpacked<L>, flags: Flags) -> (L, Flags) {
        (Self::encode(value), flags)
    }

    /// Returns a zero with the exponent nearest `exponent` in the range of the
    /// format.
    pub(super) fn zero<L>(negative: bool, exponent: i64) -> Unpacked<L> {
        let lowest = i64::from(Self::EMIN) - i64::from(Self::PRECISION) + 1;
        let highest = i64::from(Self::EMAX) - i64::from(Self::PRECISION) + 1;
        Unpacked::Zero {
            negative,
            exponent: narrow(exponent.clamp(lowest, highest)),
        }
    }

    /// Returns the term of a zero or a finite value.
    fn term<L: Limbs>(value: &Unpacked<L>) -> Term {
        match *value {
            Unpacked::Zero { negative, exponent } => Term {
                negative,
                exponent: i64::from(exponent),
                coefficient: Wide::ZERO,
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

    /// Returns the exponent of a zero or a finite value.
    fn exponent_of<L>(value: &Unpacked<L>) -> i64 {
        match *value {
            Unpacked::Zero { exponent, .. } | Unpacked::Finite { exponent, .. } => {
                i64::from(exponent)
            }
            _ => unreachable!("only a zero or a finite value has an exponent"),
        }
    }

    /// Returns the sign of a zero, finite, or infinite value.
    fn sign<L>(value: &Unpacked<L>) -> bool {
        match *value {
            Unpacked::Zero { negative, .. }
            | Unpacked::Finite { negative, .. }
            | Unpacked::Infinity { negative } => negative,
            Unpacked::Nan { .. } | Unpacked::Unsupported => false,
        }
    }

    /// Rounds a sum with a preferred exponent, or encodes an exact zero sum.
    fn finish_sum<L: Limbs>(
        sum: &Sum,
        negative_zero: bool,
        preferred: i64,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        match sum {
            Sum::Value(value) => Self::finish(value, preferred, env, flags),
            Sum::Zero => Self::exact(Self::zero(negative_zero, preferred), flags),
        }
    }

    /// Adds `left` and `right`, or subtracts `right` when `subtract` is set.
    /// The preferred exponent is the smaller exponent of the operands.
    pub fn add<L: Limbs>(left: L, right: L, subtract: bool, env: &Env) -> (L, Flags) {
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
                let (a, b) = (Self::term(&x), Self::term(&y));
                // An exact zero sum keeps the sign that the signs share, or
                // takes the sign of the rounding direction.
                let zero_sign = if a.negative == b.negative {
                    a.negative
                } else {
                    zero_sum_sign(env)
                };
                let sum = sum(a, b, Self::PRECISION);
                Self::finish_sum(&sum, zero_sign, preferred, env, flags)
            }
        }
    }

    /// Multiplies `left` by `right`. The preferred exponent is the sum of the
    /// exponents.
    pub fn mul<L: Limbs>(left: L, right: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        let negative = Self::sign(&x) != Self::sign(&y);
        match (x, y) {
            (Unpacked::Infinity { .. }, Unpacked::Zero { .. })
            | (Unpacked::Zero { .. }, Unpacked::Infinity { .. }) => {
                Self::exact(default_nan(env), flags | Flags::INVALID)
            }
            (Unpacked::Infinity { .. }, _) | (_, Unpacked::Infinity { .. }) => {
                Self::exact(Unpacked::Infinity { negative }, flags)
            }
            _ => {
                let exponent = Self::exponent_of(&x) + Self::exponent_of(&y);
                let product = multiply(Self::term(&x).coefficient, Self::term(&y).coefficient);
                let value = Unrounded {
                    negative,
                    exponent: narrow(exponent),
                    significand: product,
                    sticky: false,
                };
                Self::finish(&value, exponent, env, flags)
            }
        }
    }

    /// Divides `left` by `right`. The preferred exponent is the exponent of
    /// `left` less the exponent of `right`.
    pub fn div<L: Limbs>(left: L, right: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        if let Some((value, special)) = nan::special(&x, &y, env) {
            return Self::exact(value, flags | special);
        }
        let negative = Self::sign(&x) != Self::sign(&y);
        let lowest = i64::from(Self::EMIN) - i64::from(Self::PRECISION) + 1;
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
                let (dividend, divisor) = (Self::term(&x).coefficient, Self::term(&y).coefficient);
                let precision = Self::TARGET.precision_in(env);
                // Scale the dividend so that the quotient has at least
                // precision + 1 digits.
                let scale = (i64::from(precision) + 1 + i64::from(digit_count(&divisor))
                    - i64::from(digit_count(&dividend)))
                .max(0);
                let scale = u32::try_from(scale).expect("the scale is small");
                let numerator = multiply(dividend, power_of_ten(scale));
                let (quotient, remainder) = limbs::divide(numerator, divisor);
                let preferred = Self::exponent_of(&x) - Self::exponent_of(&y);
                let value = Unrounded {
                    negative,
                    exponent: narrow(preferred - i64::from(scale)),
                    significand: quotient,
                    sticky: !remainder.is_zero(),
                };
                Self::finish(&value, preferred, env, flags)
            }
        }
    }

    /// Returns the square root. The preferred exponent is half the exponent
    /// of the operand, rounded down.
    pub fn sqrt<L: Limbs>(value: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(value, env, &mut flags);
        if let Some((result, special)) = nan::special(&x, &Unpacked::zero(false), env) {
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
                let coefficient = Self::term(&x).coefficient;
                let precision = Self::TARGET.precision_in(env);
                let exponent = i64::from(exponent);
                // Scale to at least 2 * precision + 2 digits and an even
                // exponent, so that the root has precision + 1 digits.
                let mut scale =
                    (2 * i64::from(precision) + 2 - i64::from(digit_count(&coefficient))).max(0);
                if (exponent - scale).rem_euclid(2) != 0 {
                    scale += 1;
                }
                let shift = u32::try_from(scale).expect("the scale is small");
                let radicand = multiply(coefficient, power_of_ten(shift));
                let (root, inexact) = limbs::square_root(radicand);
                let result = Unrounded {
                    negative: false,
                    exponent: narrow((exponent - scale) / 2),
                    significand: root,
                    sticky: inexact,
                };
                Self::finish(&result, exponent.div_euclid(2), env, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN")
            }
        }
    }

    /// Returns `left * right + addend`, rounded once. The preferred exponent
    /// is the smaller of the product exponent and the addend exponent. The NaN
    /// cases follow the binary engine and the fused order of the NaN rule.
    pub fn mul_add<L: Limbs>(left: L, right: L, addend: L, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(left, env, &mut flags);
        let y = Self::operand(right, env, &mut flags);
        let z = Self::operand(addend, env, &mut flags);
        let invalid_product = matches!(
            (x, y),
            (Unpacked::Infinity { .. }, Unpacked::Zero { .. })
                | (Unpacked::Zero { .. }, Unpacked::Infinity { .. })
        );
        if x.is_nan() || y.is_nan() {
            let (value, special) = nan::fused(&x, &y, &z, env);
            return Self::exact(value, flags | special);
        }
        if invalid_product {
            let (value, special) = nan::invalid_product(&z, env);
            return Self::exact(value, flags | special);
        }
        if z.is_nan() {
            let (value, special) = nan::propagate(&Unpacked::zero(false), &z, env);
            return Self::exact(value, flags | special);
        }
        let product_negative = Self::sign(&x) != Self::sign(&y);
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
                let product_exponent = Self::exponent_of(&x) + Self::exponent_of(&y);
                let product = Term {
                    negative: product_negative,
                    exponent: product_exponent,
                    coefficient: multiply(Self::term(&x).coefficient, Self::term(&y).coefficient),
                };
                let addend = Self::term(&z);
                let preferred = product_exponent.min(addend.exponent);
                let zero_sign = if product.negative == addend.negative {
                    product.negative
                } else {
                    zero_sum_sign(env)
                };
                Self::finish_sum(
                    &sum(product, addend, Self::PRECISION),
                    zero_sign,
                    preferred,
                    env,
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
        // but toward positive, which gives the next value up.
        let tiny = number(false, 1, -30);
        assert_eq!(
            parts(number(false, 1, 0) + tiny),
            (1_000_000_000_000_000, -15)
        );
        let (up, _) = number(false, 1, 0).add_with(tiny, crate::Rounding::TowardPositive);
        assert_eq!(parts(up), (1_000_000_000_000_001, -15));
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
