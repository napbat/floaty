//! Bounds on positive values in the bits of the storage, with directed
//! rounding. They decide most comparisons of `pown` and `rootn` before an
//! exact power does, and they give the estimate of a root.

use core::cmp::Ordering;

use crate::limbs::{self, Limbs, Widen};

/// The direction of a rounding of an [`Approx`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Direction {
    Down,
    Up,
}

/// A positive value `mantissa * 2^exponent`, with the top bit of the
/// mantissa set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Approx<L> {
    mantissa: L,
    exponent: i64,
}

impl<L: Widen> Approx<L> {
    /// Rounds `value * 2^exponent` for a nonzero value of any width.
    pub(super) fn new<V: Limbs>(value: V, exponent: i64, direction: Direction) -> Self {
        let width = value.bit_length();
        debug_assert!(width != 0, "the value is positive");
        if width <= L::BITS {
            let added = L::BITS - width;
            return Self {
                mantissa: value.resize::<L>().shl(added),
                exponent: exponent - i64::from(added),
            };
        }
        let dropped = width - L::BITS;
        let truncated = Self {
            mantissa: value.shr(dropped).resize(),
            exponent: exponent + i64::from(dropped),
        };
        if direction == Direction::Down || !value.any_below(dropped) {
            return truncated;
        }
        truncated.next_up()
    }

    /// Returns the next value up: one more unit in the last place.
    fn next_up(self) -> Self {
        let mantissa = self.mantissa.increment();
        if mantissa.is_zero() {
            // The mantissa was all ones, and becomes the next power of two.
            return Self {
                mantissa: L::ZERO.with_bit(L::BITS - 1),
                exponent: self.exponent + 1,
            };
        }
        Self { mantissa, ..self }
    }

    /// Returns 1.
    fn one() -> Self {
        Self::new(L::ZERO.with_bit(0), 0, Direction::Down)
    }

    pub(super) fn mul(self, other: Self, direction: Direction) -> Self {
        Self::new(
            self.mantissa.widening_mul(other.mantissa),
            self.exponent + other.exponent,
            direction,
        )
    }

    pub(super) fn pow(self, power: u32, direction: Direction) -> Self {
        let mut result = Self::one();
        for bit in (0..u32::BITS - power.leading_zeros()).rev() {
            result = result.mul(result, direction);
            if (power >> bit) & 1 == 1 {
                result = result.mul(self, direction);
            }
        }
        result
    }

    pub(super) fn div(self, other: Self, direction: Direction) -> Self {
        let numerator = self.mantissa.resize::<L::Double>().shl(L::BITS);
        let (quotient, remainder) = limbs::divide(numerator, other.mantissa.resize());
        let quotient = if direction == Direction::Up && !remainder.is_zero() {
            quotient.increment()
        } else {
            quotient
        };
        let exponent = self.exponent - other.exponent - i64::from(L::BITS);
        Self::new(quotient, exponent, direction)
    }

    /// Returns `self * numerator / denominator` for small factors.
    fn scale(self, numerator: u32, denominator: u32, direction: Direction) -> Self {
        let product =
            limbs::multiply_small(self.mantissa.resize::<L::Double>(), u64::from(numerator));
        let divisor = L::Double::ZERO.with_limb(0, u64::from(denominator));
        let (quotient, remainder) = limbs::divide(product, divisor);
        let quotient = if direction == Direction::Up && !remainder.is_zero() {
            quotient.increment()
        } else {
            quotient
        };
        Self::new(quotient, self.exponent, direction)
    }

    fn add(self, other: Self, direction: Direction) -> Self {
        let (high, low) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let gap = high.exponent - low.exponent;
        let Some(gap) = u32::try_from(gap).ok().filter(|&gap| gap < L::BITS) else {
            // The lower value is below one unit of the higher one.
            return match direction {
                Direction::Down => high,
                Direction::Up => high.next_up(),
            };
        };
        let sum = high
            .mantissa
            .resize::<L::Double>()
            .shl(gap)
            .add(low.mantissa.resize());
        Self::new(sum, low.exponent, direction)
    }

    /// Returns the weight of the top bit, as a power of two.
    pub(super) fn top(&self) -> i64 {
        self.exponent + i64::from(L::BITS) - 1
    }

    pub(super) fn compare(&self, other: &Self) -> Ordering {
        self.exponent
            .cmp(&other.exponent)
            .then_with(|| self.mantissa.compare(&other.mantissa))
    }

    /// Returns the integer part, and `true` when a fraction is left. The
    /// integer part must fit the double width.
    pub(super) fn floor(self) -> (L::Double, bool) {
        match u32::try_from(-self.exponent) {
            Ok(dropped) if dropped >= L::BITS => (L::Double::ZERO, true),
            Ok(dropped) => (
                self.mantissa.shr(dropped).resize(),
                self.mantissa.any_below(dropped),
            ),
            Err(_) => {
                let shift = u32::try_from(self.exponent).expect("the integer fits");
                (self.mantissa.resize::<L::Double>().shl(shift), false)
            }
        }
    }
}

/// An interval `[low, high]` around a positive value.
#[derive(Clone, Copy, Debug)]
pub(super) struct Interval<L> {
    pub(super) low: Approx<L>,
    pub(super) high: Approx<L>,
}

impl<L: Widen> Interval<L> {
    /// Returns the bounds of `value * 2^exponent` for a nonzero value of any
    /// width.
    pub(super) fn of<V: Limbs>(value: V, exponent: i64) -> Self {
        Self {
            low: Approx::new(value, exponent, Direction::Down),
            high: Approx::new(value, exponent, Direction::Up),
        }
    }

    pub(super) fn pow(&self, power: u32) -> Self {
        Self {
            low: self.low.pow(power, Direction::Down),
            high: self.high.pow(power, Direction::Up),
        }
    }

    pub(super) fn mul(&self, other: &Self) -> Self {
        Self {
            low: self.low.mul(other.low, Direction::Down),
            high: self.high.mul(other.high, Direction::Up),
        }
    }

    /// Returns the bounds of `self / other`.
    pub(super) fn div(&self, other: &Self) -> Self {
        Self {
            low: self.low.div(other.high, Direction::Down),
            high: self.high.div(other.low, Direction::Up),
        }
    }

    /// Returns the order of the two values, or `None` when the bounds do not
    /// decide it.
    pub(super) fn compare(&self, other: &Self) -> Option<Ordering> {
        if self.low.compare(&other.high) == Ordering::Greater {
            return Some(Ordering::Greater);
        }
        if self.high.compare(&other.low) == Ordering::Less {
            return Some(Ordering::Less);
        }
        let exact = self.low == self.high && other.low == other.high;
        (exact && self.low == other.low).then_some(Ordering::Equal)
    }

    /// Returns the integer part of the value, when the bounds decide it and
    /// a fraction is certainly left.
    pub(super) fn floor_inexact(&self) -> Option<L::Double> {
        let (low, fraction) = self.low.floor();
        let (high, _) = self.high.floor();
        (fraction && low == high).then_some(low)
    }
}

/// Returns an estimate of `floor(target^(1/power))` for a root of at most
/// `bits` bits: its top bits by bisection, and the rest by Newton's steps.
/// The estimate needs no bound, because exact powers check it.
pub(super) fn estimate_root<L: Widen>(power: u32, target: &Approx<L>, bits: u32) -> L::Double {
    if power == 1 {
        return target.floor().0;
    }
    let top = bits.min(12);
    let mut root = L::Double::ZERO;
    for bit in (bits - top..bits).rev() {
        let candidate = root.with_bit(bit);
        let estimate = Approx::<L>::new(candidate, 0, Direction::Down).pow(power, Direction::Down);
        if estimate.compare(target) != Ordering::Greater {
            root = candidate;
        }
    }
    if top == bits || root.is_zero() {
        return root;
    }
    let mut value = Approx::<L>::new(root.with_bit(bits - top - 1), 0, Direction::Down);
    for _ in 0..64 {
        // y <- ((k - 1) y + T / y^(k - 1)) / k
        let quotient = target.div(value.pow(power - 1, Direction::Down), Direction::Down);
        let next = value
            .scale(power - 1, 1, Direction::Down)
            .add(quotient, Direction::Down)
            .scale(1, power, Direction::Down);
        if next.floor().0 == value.floor().0 {
            return next.floor().0;
        }
        value = next;
    }
    value.floor().0
}
