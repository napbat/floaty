//! Ball arithmetic: a number known within a radius.
//!
//! A [`Ball`] is a center and a radius. The true value lies within the
//! radius of the center. Each operation truncates its center toward zero
//! and adds a bound on every error to the radius, so the ball of a result
//! holds the true result of the operation on every value of the operand
//! balls.

use core::cmp::Ordering;

use crate::limbs::{self, Divisor, Limbs, Widen};

/// An upper bound `mantissa * 2^exponent` on a nonnegative error.
///
/// The mantissa stays below 2^32, so that the product of two mantissas
/// fits a `u64`. Each operation rounds up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Radius {
    mantissa: u64,
    exponent: i64,
}

/// The width of the mantissa of a [`Radius`].
const RADIUS_BITS: u32 = 32;

impl Radius {
    /// The bound zero.
    pub(super) const ZERO: Self = Self {
        mantissa: 0,
        exponent: 0,
    };

    /// Returns the bound `2^exponent`.
    pub(super) fn power_of_two(exponent: i64) -> Self {
        Self {
            mantissa: 1,
            exponent,
        }
    }

    /// Returns `mantissa * 2^exponent` with the mantissa below 2^32, rounded
    /// up.
    fn new(mantissa: u128, exponent: i64) -> Self {
        let length = 128 - mantissa.leading_zeros();
        if length <= RADIUS_BITS {
            return Self {
                mantissa: u64::try_from(mantissa).expect("the mantissa has at most 32 bits"),
                exponent,
            };
        }
        let shift = length - RADIUS_BITS;
        let lost = mantissa & ((1 << shift) - 1) != 0;
        // A carry can add a bit, which the next call removes.
        Self::new(
            (mantissa >> shift) + u128::from(lost),
            exponent + i64::from(shift),
        )
    }

    /// Returns `true` for the bound zero.
    pub(super) fn is_zero(self) -> bool {
        self.mantissa == 0
    }

    /// Returns a bound on the sum.
    #[must_use]
    pub(super) fn add(self, other: Self) -> Self {
        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        let (high, low) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let gap = high.exponent - low.exponent;
        // The smaller bound moves to the exponent of the larger one, rounded
        // up. Past 64 bits it is below one unit of the larger one.
        let moved = match u32::try_from(gap) {
            Ok(gap) if gap < 64 => {
                let shifted = low.mantissa >> gap;
                shifted + u64::from(shifted << gap != low.mantissa)
            }
            _ => 1,
        };
        Self::new(u128::from(high.mantissa) + u128::from(moved), high.exponent)
    }

    /// Returns a bound on the product.
    #[must_use]
    pub(super) fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::ZERO;
        }
        Self::new(
            u128::from(self.mantissa) * u128::from(other.mantissa),
            self.exponent + other.exponent,
        )
    }

    /// Returns a bound on the quotient by a positive integer.
    #[must_use]
    pub(super) fn div_small(self, divisor: u64) -> Self {
        if self.is_zero() {
            return self;
        }
        let numerator = u128::from(self.mantissa) << 64;
        let divisor = u128::from(divisor);
        Self::new(numerator.div_ceil(divisor), self.exponent - 64)
    }

    /// Returns a bound on the quotient by a value of at least
    /// `lower * 2^exponent`, with `lower` nonzero.
    #[must_use]
    fn div_lower(self, lower: u64, exponent: i64) -> Self {
        if self.is_zero() {
            return self;
        }
        let numerator = u128::from(self.mantissa) << 64;
        Self::new(
            numerator.div_ceil(u128::from(lower)),
            self.exponent - 64 - exponent,
        )
    }

    /// Returns the bound times `2^shift`.
    #[must_use]
    pub(super) fn scale(self, shift: i64) -> Self {
        Self {
            exponent: self.exponent + shift,
            ..self
        }
    }

    /// Returns the least `e` with `self <= 2^e`, or `i64::MIN` for zero.
    pub(super) fn ceiling_log2(self) -> i64 {
        if self.is_zero() {
            return i64::MIN;
        }
        self.exponent + i64::from(64 - (self.mantissa - 1).leading_zeros())
    }

    /// Returns the bound as a count of units of `2^unit`, rounded up, or
    /// `None` when the count needs more than `limit` bits.
    fn units(self, unit: i64, limit: u32) -> Option<(u64, u32)> {
        if self.is_zero() {
            return Some((0, 0));
        }
        let shift = self.exponent - unit;
        if shift >= 0 {
            let shift = u32::try_from(shift).ok()?;
            let length = 64 - self.mantissa.leading_zeros();
            (length + shift <= limit).then_some((self.mantissa, shift))
        } else {
            let shift = u32::try_from(-shift).unwrap_or(u32::MAX);
            let count = if shift >= 64 {
                1
            } else {
                let shifted = self.mantissa >> shift;
                shifted + u64::from(shifted << shift != self.mantissa)
            };
            Some((count, 0))
        }
    }
}

/// A number `(-1)^negative * mantissa * 2^exponent`. The mantissa has its
/// top bit set, or is zero.
#[derive(Clone, Copy, Debug)]
pub(super) struct Big<W> {
    negative: bool,
    mantissa: W,
    exponent: i64,
}

impl<W: Widen> Big<W> {
    /// Returns zero.
    fn zero() -> Self {
        Self {
            negative: false,
            mantissa: W::ZERO,
            exponent: 0,
        }
    }

    /// Returns `value * 2^exponent` for an integer of any width, truncated
    /// toward zero to the width of `W`, and a bound on the dropped part.
    pub(super) fn new<V: Limbs>(negative: bool, value: V, exponent: i64) -> (Self, Radius) {
        let length = value.bit_length();
        if length == 0 {
            return (Self::zero(), Radius::ZERO);
        }
        if length <= W::BITS {
            let shift = W::BITS - length;
            let number = Self {
                negative,
                mantissa: value.resize::<W>().shl(shift),
                exponent: exponent - i64::from(shift),
            };
            return (number, Radius::ZERO);
        }
        let shift = length - W::BITS;
        let exponent = exponent + i64::from(shift);
        let error = if value.any_below(shift) {
            Radius::power_of_two(exponent)
        } else {
            Radius::ZERO
        };
        let number = Self {
            negative,
            mantissa: value.shr(shift).resize(),
            exponent,
        };
        (number, error)
    }

    /// Returns `true` for zero.
    pub(super) fn is_zero(&self) -> bool {
        self.mantissa.is_zero()
    }

    /// Returns `true` for a negative number.
    pub(super) fn is_negative(&self) -> bool {
        self.negative && !self.is_zero()
    }

    /// Returns the weight of the top bit. The number must not be zero.
    pub(super) fn top(&self) -> i64 {
        self.exponent + i64::from(W::BITS) - 1
    }

    /// Returns the top 64 bits of the mantissa.
    pub(super) fn leading(&self) -> u64 {
        self.mantissa.shr(W::BITS - 64).limb(0)
    }

    /// Returns the mantissa and the weight of its lowest bit.
    pub(super) fn parts(&self) -> (W, i64) {
        (self.mantissa, self.exponent)
    }

    /// Returns an upper bound on the magnitude.
    fn upper(&self) -> Radius {
        if self.is_zero() {
            return Radius::ZERO;
        }
        let top = self.mantissa.shr(W::BITS - RADIUS_BITS).limb(0);
        Radius::new(
            u128::from(top) + 1,
            self.exponent + i64::from(W::BITS - RADIUS_BITS),
        )
    }

    /// Returns a nonzero lower bound `mantissa * 2^exponent` on the
    /// magnitude of a number that is not zero.
    fn lower(&self) -> (u64, i64) {
        let top = self.mantissa.shr(W::BITS - RADIUS_BITS).limb(0);
        (top, self.exponent + i64::from(W::BITS - RADIUS_BITS))
    }

    /// Returns the negation.
    #[must_use]
    pub(super) fn negate(self) -> Self {
        Self {
            negative: !self.negative,
            ..self
        }
    }

    /// Returns the number times `2^shift`, exactly.
    #[must_use]
    pub(super) fn scale(self, shift: i64) -> Self {
        Self {
            exponent: self.exponent + shift,
            ..self
        }
    }

    /// Returns the truncated product and a bound on its error.
    pub(super) fn mul(&self, other: &Self) -> (Self, Radius) {
        let product = self.mantissa.widening_mul(other.mantissa);
        Big::new(
            self.negative != other.negative,
            product,
            self.exponent + other.exponent,
        )
    }

    /// Returns the truncated sum and a bound on its error.
    pub(super) fn add(&self, other: &Self) -> (Self, Radius) {
        if self.is_zero() {
            return (*other, Radius::ZERO);
        }
        if other.is_zero() {
            return (*self, Radius::ZERO);
        }
        // Both mantissas move to a common lowest weight in twice the width,
        // with a bit of room for a carry. The larger one keeps every bit.
        let top = self.top().max(other.top());
        let lowest = top - 2 * i64::from(W::BITS) + 2;
        let (left, left_lost) = self.aligned(lowest);
        let (right, right_lost) = other.aligned(lowest);
        let lost = u128::from(left_lost) + u128::from(right_lost);
        let alignment = Radius::new(lost, lowest);
        let (negative, magnitude) = if self.negative == other.negative {
            (self.negative, left.add(right))
        } else {
            match left.compare(&right) {
                Ordering::Greater | Ordering::Equal => (self.negative, left.sub(right)),
                Ordering::Less => (other.negative, right.sub(left)),
            }
        };
        let (sum, error) = Big::new(negative, magnitude, lowest);
        (sum, error.add(alignment))
    }

    /// Returns the mantissa at the lowest weight `lowest`, in twice the
    /// width, truncated, and `true` when bits drop.
    fn aligned(&self, lowest: i64) -> (W::Double, bool) {
        let wide = self.mantissa.resize::<W::Double>();
        let shift = self.exponent - lowest;
        if shift >= 0 {
            let shift = u32::try_from(shift).expect("the larger mantissa fits the double width");
            return (wide.shl(shift), false);
        }
        let count = u32::try_from(-shift).unwrap_or(u32::MAX);
        (wide.shr(count), self.mantissa.any_below(count))
    }

    /// Returns the truncated quotient by a positive integer and a bound on
    /// its error.
    pub(super) fn div_small(&self, divisor: u64) -> (Self, Radius) {
        let numerator = self.mantissa.resize::<W::Double>().shl(W::BITS);
        let (quotient, rest) = limbs::divide_small(numerator, Divisor::new(divisor));
        let exponent = self.exponent - i64::from(W::BITS);
        let (number, error) = Big::new(self.negative, quotient, exponent);
        let remainder = if rest == 0 {
            Radius::ZERO
        } else {
            Radius::power_of_two(exponent)
        };
        (number, error.add(remainder))
    }

    /// Returns the truncated quotient by a number that is not zero, and a
    /// bound on its error.
    pub(super) fn div(&self, other: &Self) -> (Self, Radius) {
        let numerator = self.mantissa.resize::<W::Double>().shl(W::BITS);
        let denominator = other.mantissa.resize::<W::Double>();
        let (quotient, rest) = limbs::divide(numerator, denominator);
        let exponent = self.exponent - i64::from(W::BITS) - other.exponent;
        let (number, error) = Big::new(self.negative != other.negative, quotient, exponent);
        let remainder = if rest.is_zero() {
            Radius::ZERO
        } else {
            Radius::power_of_two(exponent)
        };
        (number, error.add(remainder))
    }

    /// Returns the truncated square root of a positive number and a bound on
    /// its error. The mantissa moves up by `W::BITS` bits, or one bit less,
    /// to an even exponent, so the root keeps `W::BITS` bits.
    pub(super) fn sqrt(&self) -> (Self, Radius) {
        let shift = if (self.exponent - i64::from(W::BITS)) % 2 == 0 {
            W::BITS
        } else {
            W::BITS - 1
        };
        let wide = self.mantissa.resize::<W::Double>().shl(shift);
        let (root, inexact) = limbs::square_root_of_double::<W>(wide);
        let exponent = (self.exponent - i64::from(shift)) / 2;
        let (number, error) = Big::new(false, root, exponent);
        let rest = if inexact {
            Radius::power_of_two(exponent)
        } else {
            Radius::ZERO
        };
        (number, error.add(rest))
    }

    /// Returns the nearest integer, ties away from zero. The magnitude must
    /// be below 2^62.
    pub(super) fn round_to_i64(&self) -> i64 {
        if self.is_zero() || self.top() < -1 {
            return 0;
        }
        debug_assert!(self.top() < 62, "the integer fits an i64");
        let shift = u32::try_from(-self.exponent).expect("a value below 2^62 has a fraction");
        let whole = self.mantissa.shr(shift).limb(0);
        let half = self.mantissa.bit(shift - 1);
        let magnitude = i64::try_from(whole + u64::from(half)).expect("the integer is below 2^62");
        if self.negative { -magnitude } else { magnitude }
    }
}

/// A number known within a radius: the true value `v` meets
/// `|v - center| <= radius`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Ball<W> {
    center: Big<W>,
    radius: Radius,
}

impl<W: Widen> Ball<W> {
    /// Returns the exact number `value * 2^exponent` for an integer of any
    /// width. The radius bounds the truncation to the width of `W`.
    pub(super) fn new<V: Limbs>(negative: bool, value: V, exponent: i64) -> Self {
        let (center, radius) = Big::new(negative, value, exponent);
        Self { center, radius }
    }

    /// Returns the center as a ball of radius zero.
    #[must_use]
    pub(super) fn center_only(&self) -> Self {
        Self {
            center: self.center,
            radius: Radius::ZERO,
        }
    }

    /// Returns the exact integer `value`.
    pub(super) fn integer(value: i64) -> Self {
        Self::new(value < 0, [value.unsigned_abs()], 0)
    }

    /// Returns 1.
    pub(super) fn one() -> Self {
        Self::integer(1)
    }

    /// Returns the center.
    pub(super) fn center(&self) -> &Big<W> {
        &self.center
    }

    /// Returns the radius.
    pub(super) fn radius(&self) -> Radius {
        self.radius
    }

    /// Returns the ball with its radius grown by `extra`.
    #[must_use]
    pub(super) fn widened(self, extra: Radius) -> Self {
        Self {
            radius: self.radius.add(extra),
            ..self
        }
    }

    /// Returns an upper bound on the magnitude of every value of the ball.
    pub(super) fn upper(&self) -> Radius {
        self.center.upper().add(self.radius)
    }

    /// Returns `true` when every value of the ball is above zero.
    pub(super) fn is_positive(&self) -> bool {
        if self.center.is_zero() || self.center.negative {
            return false;
        }
        let (lower, exponent) = self.center.lower();
        // The radius is below the lower bound of the center.
        self.radius.ceiling_log2() < exponent + i64::from(lower.ilog2())
    }

    /// Returns the negation.
    #[must_use]
    pub(super) fn negate(self) -> Self {
        Self {
            center: self.center.negate(),
            ..self
        }
    }

    /// Returns the ball of the magnitude of a ball that does not hold zero.
    #[must_use]
    pub(super) fn abs(self) -> Self {
        Self {
            center: Big {
                negative: false,
                ..self.center
            },
            ..self
        }
    }

    /// Returns the ball times `2^shift`.
    #[must_use]
    pub(super) fn scale(self, shift: i64) -> Self {
        Self {
            center: self.center.scale(shift),
            radius: self.radius.scale(shift),
        }
    }

    /// Returns the sum.
    #[must_use]
    pub(super) fn add(&self, other: &Self) -> Self {
        let (center, error) = self.center.add(&other.center);
        Self {
            center,
            radius: self.radius.add(other.radius).add(error),
        }
    }

    /// Returns the difference.
    #[must_use]
    pub(super) fn sub(&self, other: &Self) -> Self {
        self.add(&other.negate())
    }

    /// Returns the product. With `a = x + d` and `b = y + e`, the error of
    /// `x * y` is at most `|x| e + |y| d + d e`.
    #[must_use]
    pub(super) fn mul(&self, other: &Self) -> Self {
        let (center, error) = self.center.mul(&other.center);
        let radius = self
            .center
            .upper()
            .mul(other.radius)
            .add(other.center.upper().mul(self.radius))
            .add(self.radius.mul(other.radius))
            .add(error);
        Self { center, radius }
    }

    /// Returns the quotient by a positive integer.
    #[must_use]
    pub(super) fn div_small(&self, divisor: u64) -> Self {
        let (center, error) = self.center.div_small(divisor);
        Self {
            center,
            radius: self.radius.div_small(divisor).add(error),
        }
    }

    /// Returns the quotient, or `None` when the divisor ball comes within
    /// half its center of zero.
    ///
    /// With `a = x + d` and `b = y + e`, `|a / b - x / y|` is at most
    /// `(d + |x / y| e) / (|y| - e)`, and `|y| - e` is at least `|y| / 2`.
    #[must_use]
    pub(super) fn div(&self, other: &Self) -> Option<Self> {
        if other.center.is_zero() {
            return None;
        }
        let (lower, exponent) = other.center.lower();
        // The radius of the divisor must stay below half its lower bound.
        if other.radius.scale(1).ceiling_log2() >= exponent + i64::from(lower.ilog2()) {
            return None;
        }
        let (center, error) = self.center.div(&other.center);
        let quotient = center.upper().add(error);
        let radius = self
            .radius
            .add(quotient.mul(other.radius))
            .scale(1)
            .div_lower(lower, exponent)
            .add(error);
        Some(Self { center, radius })
    }

    /// Returns the square root, or `None` when the ball comes within half its
    /// center of zero.
    ///
    /// With `a = x + d` and `|d|` at most `x / 2`, `|sqrt(a) - sqrt(x)|` is
    /// `|d| / (sqrt(a) + sqrt(x))`, at most `|d| / sqrt(x)`. The truncated
    /// root of the center is at most `sqrt(x)`.
    #[must_use]
    pub(super) fn sqrt(&self) -> Option<Self> {
        if self.center.is_zero() || self.center.negative {
            return None;
        }
        let (lower, exponent) = self.center.lower();
        if self.radius.scale(1).ceiling_log2() >= exponent + i64::from(lower.ilog2()) {
            return None;
        }
        let (center, error) = self.center.sqrt();
        let (root, root_exponent) = center.lower();
        let radius = self.radius.div_lower(root, root_exponent).add(error);
        Some(Self { center, radius })
    }
}

/// Returns the integer part of every value of a positive ball, when it is
/// the same for all of them, as a value of `W`.
pub(super) fn integer_part<W: Widen>(ball: &Ball<W>) -> Option<W> {
    let (mantissa, exponent) = ball.center.parts();
    if ball.center.is_negative() || exponent >= 0 {
        return None;
    }
    let shift = u32::try_from(-exponent).ok()?;
    let (units, units_shift) = ball.radius.units(exponent, W::BITS - 2)?;
    let units = W::ZERO.with_limb(0, units).shl(units_shift);
    if units.compare(&mantissa) != Ordering::Less {
        return None;
    }
    let low = mantissa.sub(units).resize::<W::Double>().shr(shift);
    let high = mantissa
        .resize::<W::Double>()
        .add(units.resize())
        .shr(shift);
    (low == high).then(|| low.resize())
}

/// Returns `floor(|v| / 2^lowest)` for every value `v` of a ball that does
/// not hold zero, when it is the same for all of them.
pub(super) fn truncation<W: Widen>(ball: &Ball<W>, lowest: i64) -> Option<W> {
    let magnitude = ball.abs();
    if !magnitude.is_positive() {
        return None;
    }
    integer_part(&magnitude.scale(-lowest))
}
