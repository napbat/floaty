//! Random operands of a BID layout for the differential tests.
//!
//! [`Layout::random`] draws an encoding biased to the edge cases. The other
//! generators draw operands for one operation: an operand near another one,
//! a radicand with an exact root, an operand near a bound of an integer type,
//! and two operands that compare equal.

use super::{Integer, Layout};
use crate::random::SplitMix64;

impl Layout {
    /// Returns a random encoding, biased to the edge cases: zeros, one-digit
    /// and full coefficients, exponents at both ends of the range,
    /// subnormal values, non-canonical coefficients, infinities, and quiet
    /// and signaling NaNs with canonical and non-canonical payloads.
    pub fn random(self, rng: &mut SplitMix64) -> u128 {
        let negative = rng.next_u64() & 1 == 1;
        match below(rng, 20) {
            0 | 1 => self.number(negative, self.random_field(rng), 0),
            2 => self.random_special(rng, negative),
            3 => self.random_nan(rng, negative),
            4 => self.random_non_canonical(rng, negative),
            5 => rng.next_u128() & ((u128::MAX) >> (128 - self.width)),
            _ => self.number(
                negative,
                self.random_field(rng),
                self.random_coefficient(rng),
            ),
        }
    }

    /// Returns a random canonical coefficient, biased to one digit, full
    /// precision, powers of 10, and tails of 5, 49..9, and 50..01.
    pub fn random_coefficient(self, rng: &mut SplitMix64) -> u128 {
        let precision = self.precision();
        let digits = |rng: &mut SplitMix64, count: u32| {
            let low = Self::power_of_ten(count - 1);
            low + below(rng, Self::power_of_ten(count) - low)
        };
        match below(rng, 9) {
            0 => 1 + below(rng, 9),
            1 => Self::power_of_ten(precision) - 1,
            2 => Self::power_of_ten(below_u32(rng, precision)),
            3 => 5 * Self::power_of_ten(below_u32(rng, precision - 1)),
            4 => Self::power_of_ten(precision - 1),
            5 | 6 => {
                // A tail of `50..0`, `49..9`, or `50..01` on random leading
                // digits: the halfway cases of a shorter coefficient.
                let tail = 1 + below_u32(rng, precision - 1);
                let head = digits(rng, precision - tail);
                let unit = Self::power_of_ten(tail);
                let half = unit / 2;
                let tail = match below(rng, 3) {
                    0 => half,
                    1 => half - 1,
                    _ => half + 1,
                };
                head * unit + tail
            }
            _ => {
                let count = 1 + below_u32(rng, precision);
                digits(rng, count)
            }
        }
    }

    /// Returns a random exponent field, biased to both ends of the range and
    /// to exponent 0.
    pub fn random_field(self, rng: &mut SplitMix64) -> u32 {
        let (largest, near) = (self.largest_field(), self.precision() + 3);
        match below(rng, 5) {
            0 => below_u32(rng, near),
            1 => largest - below_u32(rng, near),
            2 => self.bias() + below_u32(rng, 2 * near) - near,
            _ => below_u32(rng, largest + 1),
        }
    }

    /// Returns an infinity, with non-canonical trailing bits at times.
    fn random_special(self, rng: &mut SplitMix64, negative: bool) -> u128 {
        let garbage = if below(rng, 2) == 0 {
            0
        } else {
            rng.next_u128() & ((1 << (self.width - 6)) - 1)
        };
        self.infinity(negative) | garbage
    }

    /// Returns a quiet or signaling NaN with a zero, small, largest,
    /// random, or non-canonical payload, and at times extra bits.
    fn random_nan(self, rng: &mut SplitMix64, negative: bool) -> u128 {
        let largest = Self::power_of_ten(self.precision() - 1) - 1;
        let payload = match below(rng, 6) {
            0 => 0,
            1 => 1 + below(rng, 999),
            2 => largest,
            3 => largest + 1 + below(rng, (1 << self.trailing()) - largest - 1),
            _ => below(rng, largest + 1),
        };
        let extra = if below(rng, 4) == 0 {
            rng.next_u128()
        } else {
            0
        };
        self.nan(negative, below(rng, 2) == 0, payload, extra)
    }

    /// Returns a number with a coefficient above `10^p - 1`, which reads as
    /// a zero.
    fn random_non_canonical(self, rng: &mut SplitMix64, negative: bool) -> u128 {
        let smallest = Self::power_of_ten(self.precision());
        let largest = (1 << (self.trailing() + 3)) + (1 << (self.trailing() + 1)) - 1;
        let coefficient = smallest + below(rng, largest - smallest + 1);
        self.number(negative, self.random_field(rng), coefficient)
    }

    /// Returns an operand near `x`: the same exponent, a close exponent, a
    /// member of its cohort, half a unit in its last place, its negation,
    /// itself, or a new random operand.
    pub fn related(self, rng: &mut SplitMix64, x: u128) -> u128 {
        let negative = rng.next_u64() & 1 == 1;
        let Some((field, coefficient)) = self.fields(x) else {
            return self.random(rng);
        };
        let precision = self.precision();
        let span = precision + 3;
        let largest_field = self.largest_field();
        match below(rng, 10) {
            0 => x,
            1 => x ^ (1 << (self.width - 1)),
            2 => self.number(negative, field, self.random_coefficient(rng)),
            3 | 4 => {
                // A field up to `span` away, clamped to the range.
                let step = below_u32(rng, 2 * span + 1);
                let coefficient = self.random_coefficient(rng);
                let near = (field + step).saturating_sub(span).min(largest_field);
                self.number(negative, near, coefficient)
            }
            5 => {
                // A member of the cohort of `x`: the coefficient times 10^k,
                // at an exponent k lower.
                let digits = coefficient.checked_ilog10().map_or(1, |log| log + 1);
                let room = precision.saturating_sub(digits);
                let k = below_u32(rng, room + 1);
                if coefficient > Self::power_of_ten(precision) - 1 || field < k {
                    return x;
                }
                let wide = coefficient * Self::power_of_ten(k);
                self.number(x >> (self.width - 1) == 1, field - k, wide)
            }
            6 => self.number(negative, field.saturating_sub(1), 5),
            _ => self.random(rng),
        }
    }

    /// Returns a radicand: a random operand, or a square with an exact root.
    ///
    /// A square is a perfect square coefficient at an even exponent, or ten
    /// times a perfect square at an odd exponent, anywhere in the exponent
    /// range. The root has trailing zeros at times. An exact root takes the
    /// member of its cohort nearest the preferred exponent, the exponent of
    /// the radicand halved and rounded down. For the second form and for a
    /// root with trailing zeros, that member has trailing zeros.
    pub fn square(self, rng: &mut SplitMix64) -> u128 {
        if below(rng, 2) == 0 {
            return self.random(rng);
        }
        let field = self.random_field(rng);
        // The exponent `field - bias` is odd when the field and the bias
        // differ in parity.
        let scale = if (field ^ self.bias()) & 1 == 1 {
            10
        } else {
            1
        };
        let largest = ((Self::power_of_ten(self.precision()) - 1) / scale).isqrt();
        let root = if below(rng, 4) == 0 {
            // One digit and trailing zeros. `largest` has at least three
            // digits, so the root stays below it.
            let zeros = below_u32(rng, largest.ilog10());
            (1 + below(rng, 9)) * Self::power_of_ten(zeros)
        } else {
            1 + below(rng, largest)
        };
        self.number(false, field, root * root * scale)
    }

    /// Returns an operand near a bound of `integer`: a bound, one past it, or
    /// a random integer, with a fraction of zero, a half, or just below or
    /// above a half. Returns a random operand at times.
    ///
    /// A value with more digits than the precision keeps its leading digits,
    /// truncated or rounded up. So a narrow format gets its nearest values on
    /// both sides of each bound: decimal32 gets `2147483E+3` and `2147484E+3`
    /// for the bound `2^31 - 1` of `int32`.
    pub fn near_integer(self, rng: &mut SplitMix64, integer: Integer) -> u128 {
        if below(rng, 4) == 0 {
            return self.random(rng);
        }
        let top = 1_i128 << (integer.bits() - 1);
        let (low, high) = if integer.indefinite() < 0 {
            (-top, top - 1)
        } else {
            (0, 2 * top - 1)
        };
        let whole = match below(rng, 6) {
            0 => low,
            1 => high,
            2 => low - 1,
            3 => high + 1,
            _ => {
                let magnitude = i128::from(rng.next_u64() >> below_u32(rng, 64));
                if below(rng, 2) == 0 {
                    magnitude
                } else {
                    -magnitude
                }
            }
        };
        let fraction_digits = below_u32(rng, 4);
        let unit = Self::power_of_ten(fraction_digits);
        let half = unit / 2;
        let fraction = match below(rng, 4) {
            0 => 0,
            1 => half,
            2 => half.saturating_sub(1),
            _ => (half + 1).min(unit.saturating_sub(1)),
        };
        let magnitude = whole.unsigned_abs() * unit + fraction;
        let digits = magnitude.checked_ilog10().map_or(1, |log| log + 1);
        let mut dropped = digits.saturating_sub(self.precision());
        let round_up = dropped > 0 && below(rng, 2) == 0;
        let mut kept = magnitude / Self::power_of_ten(dropped) + u128::from(round_up);
        if kept > Self::power_of_ten(self.precision()) - 1 {
            kept /= 10;
            dropped += 1;
        }
        self.number(whole < 0, self.bias() + dropped - fraction_digits, kept)
    }

    /// Returns two operands that compare equal, in a random order: two zeros
    /// of any signs and exponents, a zero and a non-canonical coefficient,
    /// which reads as zero, two members of one cohort with one sign, such as
    /// 1.0 and 1.00, or two encodings of one infinity.
    pub fn equal_pair(self, rng: &mut SplitMix64) -> (u128, u128) {
        let sign = |rng: &mut SplitMix64| rng.next_u64() & 1 == 1;
        let zero = |rng: &mut SplitMix64| self.number(sign(rng), self.random_field(rng), 0);
        let (first, second) = match below(rng, 4) {
            0 => (zero(rng), zero(rng)),
            1 => {
                let negative = sign(rng);
                (zero(rng), self.random_non_canonical(rng, negative))
            }
            2 => {
                let negative = sign(rng);
                let precision = self.precision();
                let digits = 1 + below_u32(rng, precision);
                let low = Self::power_of_ten(digits - 1);
                let coefficient = low + below(rng, Self::power_of_ten(digits) - low);
                let room = precision - digits;
                let field = self.random_field(rng).max(room);
                let member = |rng: &mut SplitMix64| {
                    let zeros = below_u32(rng, room + 1);
                    let wide = coefficient * Self::power_of_ten(zeros);
                    self.number(negative, field - zeros, wide)
                };
                (member(rng), member(rng))
            }
            _ => {
                let negative = sign(rng);
                (self.infinity(negative), self.random_special(rng, negative))
            }
        };
        if below(rng, 2) == 0 {
            (first, second)
        } else {
            (second, first)
        }
    }
}

/// Returns a random value below `bound`, which must not be zero.
fn below(rng: &mut SplitMix64, bound: u128) -> u128 {
    rng.next_u128() % bound
}

/// Returns a random value below `bound`, which must not be zero.
fn below_u32(rng: &mut SplitMix64, bound: u32) -> u32 {
    u32::try_from(below(rng, u128::from(bound))).expect("the value is below a u32 bound")
}
