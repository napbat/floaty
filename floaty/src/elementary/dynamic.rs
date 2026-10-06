//! Alloc-backed, outward-rounded fixed-point intervals for certified refinement.
//!
//! Each endpoint is an integer times `2^-bits`. Integer arithmetic is exact;
//! every discarded fraction rounds toward the appropriate endpoint.

use super::{Argument, Radix, Target};
use crate::exact::Unrounded;
use crate::limbs::Limbs;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Big(Vec<u64>);

impl Big {
    fn zero() -> Self {
        Self(Vec::new())
    }
    fn small(x: u64) -> Self {
        if x == 0 { Self::zero() } else { Self(vec![x]) }
    }
    fn normalize(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }
    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }
    fn length(&self) -> u32 {
        self.0.last().map_or(0, |x| {
            let words =
                u32::try_from(self.0.len() - 1).expect("the working integer length fits u32");
            words
                .checked_mul(64)
                .and_then(|bits| bits.checked_add(64 - x.leading_zeros()))
                .expect("the working integer bit length fits u32")
        })
    }
    fn compare(&self, y: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&y.0.len())
            .then_with(|| self.0.iter().rev().cmp(y.0.iter().rev()))
    }
    fn add(&self, y: &Self) -> Self {
        let count = self.0.len().max(y.0.len());
        let mut out = Vec::with_capacity(count + 1);
        let mut carry = 0_u128;
        for i in 0..count {
            carry +=
                u128::from(*self.0.get(i).unwrap_or(&0)) + u128::from(*y.0.get(i).unwrap_or(&0));
            out.push(
                u64::try_from(carry & u128::from(u64::MAX)).expect("the mask leaves one limb"),
            );
            carry >>= 64;
        }
        if carry != 0 {
            out.push(u64::try_from(carry).expect("the carry fits one limb"));
        }
        Self(out)
    }
    fn sub(&self, y: &Self) -> Self {
        let mut out = self.clone();
        out.sub_assign(y);
        out
    }
    fn sub_assign(&mut self, y: &Self) {
        debug_assert!(self.compare(y).is_ge());
        let mut borrow = false;
        for (i, slot) in self.0.iter_mut().enumerate() {
            let (x, a) = slot.overflowing_sub(*y.0.get(i).unwrap_or(&0));
            let (x, b) = x.overflowing_sub(u64::from(borrow));
            *slot = x;
            borrow = a || b;
        }
        debug_assert!(!borrow);
        self.normalize();
    }
    fn shr_one(&mut self) {
        let mut carry = 0;
        for slot in self.0.iter_mut().rev() {
            let next = *slot << 63;
            *slot = (*slot >> 1) | carry;
            carry = next;
        }
        self.normalize();
    }
    fn shl(&self, bits: u32) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let limbs = (bits / 64) as usize;
        let shift = bits % 64;
        let mut out = vec![0; self.0.len() + limbs + usize::from(shift != 0)];
        for (i, &x) in self.0.iter().enumerate() {
            out[i + limbs] |= x << shift;
            if shift != 0 {
                out[i + limbs + 1] |= x >> (64 - shift);
            }
        }
        let mut out = Self(out);
        out.normalize();
        out
    }
    fn shr(&self, bits: u32) -> (Self, bool) {
        let limbs = (bits / 64) as usize;
        let shift = bits % 64;
        if limbs >= self.0.len() {
            return (Self::zero(), !self.is_zero());
        }
        let lost = self.0[..limbs].iter().any(|&x| x != 0)
            || (shift != 0 && self.0[limbs] << (64 - shift) != 0);
        let mut out = Vec::with_capacity(self.0.len() - limbs);
        for i in limbs..self.0.len() {
            out.push(
                (self.0[i] >> shift)
                    | if shift == 0 {
                        0
                    } else {
                        self.0.get(i + 1).copied().unwrap_or(0) << (64 - shift)
                    },
            );
        }
        let mut out = Self(out);
        out.normalize();
        (out, lost)
    }
    fn mul(&self, y: &Self) -> Self {
        if self.is_zero() || y.is_zero() {
            return Self::zero();
        }
        let mut out = vec![0; self.0.len() + y.0.len()];
        for (i, &x) in self.0.iter().enumerate() {
            let mut carry = 0_u128;
            for (j, &y) in y.0.iter().enumerate() {
                carry += u128::from(x) * u128::from(y) + u128::from(out[i + j]);
                out[i + j] =
                    u64::try_from(carry & u128::from(u64::MAX)).expect("the mask leaves one limb");
                carry >>= 64;
            }
            out[i + y.0.len()] = u64::try_from(carry).expect("the carry fits one limb");
        }
        let mut out = Self(out);
        out.normalize();
        out
    }
    fn div_small(&self, y: u64) -> (Self, bool) {
        assert!(y != 0);
        let mut out = self.clone();
        let mut remainder = 0_u128;
        for slot in out.0.iter_mut().rev() {
            remainder = (remainder << 64) | u128::from(*slot);
            *slot = u64::try_from(remainder / u128::from(y))
                .expect("a one-limb division quotient fits one limb");
            remainder %= u128::from(y);
        }
        out.normalize();
        (out, remainder != 0)
    }
    fn div(&self, y: &Self) -> (Self, bool) {
        assert!(!y.is_zero());
        if self.compare(y).is_lt() {
            return (Self::zero(), !self.is_zero());
        }
        if y.0.len() == 1 {
            return self.div_small(y.0[0]);
        }
        let shift = self.length() - y.length();
        let mut divisor = y.shl(shift);
        let mut rest = self.clone();
        let mut quotient = vec![0; (shift / 64 + 1) as usize];
        for bit in (0..=shift).rev() {
            if rest.compare(&divisor).is_ge() {
                rest.sub_assign(&divisor);
                quotient[(bit / 64) as usize] |= 1 << (bit % 64);
            }
            divisor.shr_one();
        }
        let mut quotient = Self(quotient);
        quotient.normalize();
        (quotient, !rest.is_zero())
    }
    fn from_limbs<L: Limbs>(x: L) -> Self {
        let mut out = Vec::with_capacity(x.bit_length().div_ceil(64) as usize);
        for i in 0..x.bit_length().div_ceil(64) {
            out.push(x.limb(i as usize));
        }
        Self(out)
    }
    fn to_limbs<W: Limbs>(&self) -> Option<W> {
        if self.length() > W::BITS {
            return None;
        }
        let mut out = W::ZERO;
        for (i, &x) in self.0.iter().enumerate() {
            out = out.with_limb(i, x);
        }
        Some(out)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Signed {
    negative: bool,
    magnitude: Big,
}
impl Signed {
    fn new(negative: bool, magnitude: Big) -> Self {
        Self {
            negative: negative && !magnitude.is_zero(),
            magnitude,
        }
    }
    fn negate(&self) -> Self {
        Self::new(!self.negative, self.magnitude.clone())
    }
    fn compare(&self, y: &Self) -> Ordering {
        match (self.negative, y.negative) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            (true, true) => y.magnitude.compare(&self.magnitude),
            (false, false) => self.magnitude.compare(&y.magnitude),
        }
    }
    fn add(&self, y: &Self) -> Self {
        if self.negative == y.negative {
            Self::new(self.negative, self.magnitude.add(&y.magnitude))
        } else if self.magnitude.compare(&y.magnitude).is_ge() {
            Self::new(self.negative, self.magnitude.sub(&y.magnitude))
        } else {
            Self::new(y.negative, y.magnitude.sub(&self.magnitude))
        }
    }
    fn rounded(negative: bool, value: Big, lost: bool, up: bool) -> Self {
        let value = if lost && (negative != up) {
            value.add(&Big::small(1))
        } else {
            value
        };
        Self::new(negative, value)
    }
    fn shr(&self, bits: u32, up: bool) -> Self {
        let (value, lost) = self.magnitude.shr(bits);
        Self::rounded(self.negative, value, lost, up)
    }
    fn mul(&self, y: &Self, bits: u32, up: bool) -> Self {
        let (value, lost) = self.magnitude.mul(&y.magnitude).shr(bits);
        Self::rounded(self.negative != y.negative, value, lost, up)
    }
    fn div(&self, y: &Self, bits: u32, up: bool) -> Self {
        let (value, lost) = self.magnitude.shl(bits).div(&y.magnitude);
        Self::rounded(self.negative != y.negative, value, lost, up)
    }
    fn div_small(&self, y: u64, up: bool) -> Self {
        let (value, lost) = self.magnitude.div_small(y);
        Self::rounded(self.negative, value, lost, up)
    }
}

/// A closed, outward-rounded interval of dyadic endpoints.
#[derive(Clone, Debug)]
pub(super) struct Interval {
    low: Signed,
    high: Signed,
    bits: u32,
}
impl Interval {
    pub(super) fn integer(value: i64, bits: u32) -> Self {
        let value = Signed::new(value < 0, Big::small(value.unsigned_abs()).shl(bits));
        Self {
            low: value.clone(),
            high: value,
            bits,
        }
    }
    pub(super) fn argument<L: Limbs>(x: &Argument<L>, radix: Radix, bits: u32) -> Self {
        let coefficient = Signed::new(x.negative, Big::from_limbs(x.significand).shl(bits));
        let mut result = Self {
            low: coefficient.clone(),
            high: coefficient,
            bits,
        };
        match radix {
            Radix::Binary => result.scale(i64::from(x.exponent)),
            Radix::Decimal => {
                if x.exponent >= 0 {
                    for _ in 0..x.exponent {
                        result = result.mul_small(10);
                    }
                } else {
                    // Repeated outward division encloses the exact rational;
                    // integer lower/upper endpoints need not be exact inputs.
                    for _ in 0..x.exponent.unsigned_abs() {
                        result = result.div_small(10);
                    }
                }
                result
            }
        }
    }
    pub(super) fn bits(&self) -> u32 {
        self.bits
    }
    pub(super) fn is_positive(&self) -> bool {
        !self.low.negative && !self.low.magnitude.is_zero()
    }
    pub(super) fn is_negative(&self) -> bool {
        self.high.negative
    }
    pub(super) fn upper_log2(&self) -> i64 {
        let maximum = self
            .low
            .magnitude
            .length()
            .max(self.high.magnitude.length());
        if maximum == 0 {
            i64::MIN
        } else {
            i64::from(maximum) - i64::from(self.bits)
        }
    }
    pub(super) fn add(&self, y: &Self) -> Self {
        assert_eq!(self.bits, y.bits);
        Self {
            low: self.low.add(&y.low),
            high: self.high.add(&y.high),
            bits: self.bits,
        }
    }
    pub(super) fn negate(&self) -> Self {
        Self {
            low: self.high.negate(),
            high: self.low.negate(),
            bits: self.bits,
        }
    }
    pub(super) fn sub(&self, y: &Self) -> Self {
        self.add(&y.negate())
    }
    pub(super) fn mul_small(&self, y: u64) -> Self {
        let multiply = |x: &Signed| Signed::new(x.negative, x.magnitude.mul(&Big::small(y)));
        Self {
            low: multiply(&self.low),
            high: multiply(&self.high),
            bits: self.bits,
        }
    }
    pub(super) fn mul(&self, y: &Self) -> Self {
        assert_eq!(self.bits, y.bits);
        let pairs = if !self.low.negative && !y.low.negative {
            [(&self.low, &y.low), (&self.high, &y.high)]
        } else if self.high.negative && y.high.negative {
            [(&self.high, &y.high), (&self.low, &y.low)]
        } else if !self.low.negative && y.high.negative {
            [(&self.high, &y.low), (&self.low, &y.high)]
        } else if self.high.negative && !y.low.negative {
            [(&self.low, &y.high), (&self.high, &y.low)]
        } else {
            // Only zero-straddling intervals need all four products.
            let mut low = self.low.mul(&y.low, self.bits, false);
            let mut high = self.low.mul(&y.low, self.bits, true);
            for (a, b) in [
                (&self.low, &y.high),
                (&self.high, &y.low),
                (&self.high, &y.high),
            ] {
                let below = a.mul(b, self.bits, false);
                let above = a.mul(b, self.bits, true);
                if below.compare(&low).is_lt() {
                    low = below;
                }
                if above.compare(&high).is_gt() {
                    high = above;
                }
            }
            return Self {
                low,
                high,
                bits: self.bits,
            };
        };
        Self {
            low: pairs[0].0.mul(pairs[0].1, self.bits, false),
            high: pairs[1].0.mul(pairs[1].1, self.bits, true),
            bits: self.bits,
        }
    }
    pub(super) fn div_small(&self, y: u64) -> Self {
        Self {
            low: self.low.div_small(y, false),
            high: self.high.div_small(y, true),
            bits: self.bits,
        }
    }
    pub(super) fn div(&self, y: &Self) -> Option<Self> {
        assert_eq!(self.bits, y.bits);
        if y.is_negative() {
            return self.negate().div(&y.negate());
        }
        if !y.is_positive() {
            return None;
        }
        let low_divisor = if self.low.negative { &y.low } else { &y.high };
        let high_divisor = if self.high.negative { &y.high } else { &y.low };
        Some(Self {
            low: self.low.div(low_divisor, self.bits, false),
            high: self.high.div(high_divisor, self.bits, true),
            bits: self.bits,
        })
    }
    pub(super) fn scale(&self, shift: i64) -> Self {
        if shift >= 0 {
            let shift = u32::try_from(shift).expect("the fixed-point exponent fits u32");
            let moved = |x: &Signed| Signed::new(x.negative, x.magnitude.shl(shift));
            Self {
                low: moved(&self.low),
                high: moved(&self.high),
                bits: self.bits,
            }
        } else {
            let shift = u32::try_from(shift.unsigned_abs()).unwrap_or(u32::MAX);
            Self {
                low: self.low.shr(shift, false),
                high: self.high.shr(shift, true),
                bits: self.bits,
            }
        }
    }
    /// Expands by a symmetric dyadic error bound.
    pub(super) fn widen(&self, error: &Self) -> Self {
        assert_eq!(self.bits, error.bits);
        let maximum = if error.low.magnitude.compare(&error.high.magnitude).is_gt() {
            error.low.magnitude.clone()
        } else {
            error.high.magnitude.clone()
        };
        let bound = Signed::new(false, maximum);
        Self {
            low: self.low.add(&bound.negate()),
            high: self.high.add(&bound),
            bits: self.bits,
        }
    }
    /// The same nearest integer (ties upward) for both endpoints, plus its quadrant.
    pub(super) fn nearest_integer(&self) -> Option<(u8, Self)> {
        let shifted = self.add(&Self::integer(1, self.bits).scale(-1));
        let low = shifted.low.shr(self.bits, false);
        let high = shifted.high.shr(self.bits, false);
        if low != high || low.negative {
            return None;
        }
        let quadrant = u8::try_from(low.magnitude.0.first().copied().unwrap_or(0) & 3)
            .expect("a quadrant has two bits");
        let integer = Signed::new(false, low.magnitude.shl(self.bits));
        Some((
            quadrant,
            Self {
                low: integer.clone(),
                high: integer,
                bits: self.bits,
            },
        ))
    }

    pub(super) fn truncate<W: Limbs>(&self, target: &Target) -> Option<Unrounded<W>> {
        self.truncate_scaled(target, 0)
    }

    /// Truncates a binary-scaled interval without materializing its exponent.
    pub(super) fn truncate_scaled<W: Limbs>(
        &self,
        target: &Target,
        binary_exponent: i64,
    ) -> Option<Unrounded<W>> {
        if !self.is_positive() && !self.is_negative() {
            return None;
        }
        let negative = self.is_negative();
        let (small, large) = if negative {
            (&self.high.magnitude, &self.low.magnitude)
        } else {
            (&self.low.magnitude, &self.high.magnitude)
        };
        let top = i64::from(small.length()) - 1 - i64::from(self.bits) + binary_exponent;
        let lowest = match target.radix {
            Radix::Binary => top - i64::from(target.precision) - 2,
            Radix::Decimal => {
                i64::try_from((i128::from(top) * 1_292_913_986).div_euclid(1 << 32)).ok()?
                    - i64::from(target.precision)
                    - 2
            }
        };
        let scaled = |value: &Big| -> Option<Big> {
            Some(match target.radix {
                Radix::Binary => {
                    let shift = lowest + i64::from(self.bits) - binary_exponent;
                    if shift >= 0 {
                        value.shr(u32::try_from(shift).ok()?).0
                    } else {
                        value.shl(u32::try_from(-shift).ok()?)
                    }
                }
                Radix::Decimal => {
                    let mut value = value.clone();
                    let weight = binary_exponent - i64::from(self.bits);
                    if lowest < 0 {
                        for _ in 0..lowest.unsigned_abs() {
                            value = value.mul(&Big::small(10));
                        }
                    }
                    value = if weight >= 0 {
                        value.shl(u32::try_from(weight).ok()?)
                    } else {
                        value.shr(u32::try_from(-weight).ok()?).0
                    };
                    if lowest >= 0 {
                        for _ in 0..lowest {
                            value = value.div_small(10).0;
                        }
                    }
                    value
                }
            })
        };
        let below: Big = scaled(small)?;
        let above: Big = scaled(large)?;
        if below != above || below.is_zero() {
            return None;
        }
        Some(Unrounded {
            negative,
            exponent: i32::try_from(lowest).ok()?,
            significand: below.to_limbs()?,
            sticky: true,
        })
    }
}

/// Machin's identity with a bounded alternating-series remainder.
pub(super) fn half_pi(bits: u32) -> Interval {
    fn atan(denominator: u64, bits: u32) -> Interval {
        let mut power = Interval::integer(1, bits).div_small(denominator);
        let mut sum = power.clone();
        for index in 1_u64.. {
            power = power.div_small(denominator * denominator);
            let term = power.div_small(2 * index + 1);
            sum = if index & 1 == 0 {
                sum.add(&term)
            } else {
                sum.sub(&term)
            };
            // Rounding stops powers at a one-unit upper endpoint. The
            // mathematical alternating tail is smaller than the last term.
            if power.upper_log2() <= 1 - i64::from(bits) {
                return sum.widen(&term);
            }
        }
        unreachable!()
    }
    atan(5, bits).scale(3).sub(&atan(239, bits).scale(1))
}
