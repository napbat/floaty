//! The argument reduction of `sin`, `cos`, and `tan`: `|x| 2 / pi` is
//! `4k + q + f`, with the quadrant `q` and `f` in about `[-1/2, 1/2]`, so the
//! angle `pi f / 2` lies within `pi / 4` of zero.
//!
//! An argument below 1 takes the product with a ball of `2 / pi` at the
//! working width, less the nearest integer. The product keeps the relative
//! precision of the width.
//!
//! An argument from 1 up takes the bits of 2/pi from `two_over_pi.bin`
//! (Payne and Hanek), as an integer `M` times `2^e`. A binary argument is
//! its significand times `2^e`. A decimal argument `c 10^q` is
//! `(c 5^q) 2^q` for `q >= 0`, and for `q < 0` the integer
//! `floor(c 2^F / 10^-q)` times `2^-F`, below it by less than `2^-F`. Each
//! 64-bit limb `M_i` of weight `2^w` multiplies the bits of 2/pi of weight
//! `2^(1 - w)` down to `2^(-w - F)`: the bits above them give multiples of 4,
//! and the bits below them add less than `2^(64 - F)`. The sum keeps
//! `F + 2` bits, the quadrant and `F` bits of the fraction, exactly. So the
//! fraction has the same absolute precision, `F = 2W - 66` bits, for every
//! argument from 1 up: a large argument loses no bits to its size.
//!
//! `two_over_pi.bin` holds `floor(2/pi 2^N)` with `N = 2^22 + 8192`, the
//! most significant byte first, which
//! `floaty-verify/src/bin/generate-two-over-pi.rs` writes. The bits reach
//! every argument below `2^(2^22)`, which holds every finite value of
//! binary512 and of decimal128.

use super::ball::{Ball, Radius};
use super::constants::pi;
use super::{Argument, Radix, argument, leading};
use crate::limbs::{self, Divisor, Limbs, Widen};

/// The bits of 2/pi after the point, the most significant byte first.
static TWO_OVER_PI: &[u8; 525_312] = include_bytes!("two_over_pi.bin");

/// The exponent of the least power of two that the reduction does not reach.
pub(super) const REACH: i64 = 1 << 22;

/// The limbs of the integer `M` of an argument: `c 5^q` of decimal128 lies
/// below `2^(113 + 6111 log2(5))`, below `2^14304`.
const PRODUCT_LIMBS: usize = 232;

/// A reduced argument: the angle `pi f / 2` and its quadrant.
pub(super) struct Reduced<W> {
    /// The quadrant `q`, from 0 to 3.
    pub(super) quadrant: u64,
    /// The angle, within about `pi / 4` of zero.
    pub(super) angle: Ball<W>,
}

/// Reduces the magnitude of a nonzero finite argument below `2^REACH`.
pub(super) fn reduce<L: Limbs, W: Widen>(x: &Argument<L>, radix: Radix) -> Option<Reduced<W>> {
    let exponent = i64::from(x.exponent);
    let (quadrant, fraction) = if leading(x, radix) < 0 {
        let magnitude = Argument {
            negative: false,
            ..*x
        };
        let product = argument::<L, W>(&magnitude, radix)?.mul(&Ball::integer(2).div(&pi::<W>())?);
        nearest(&product)
    } else if radix == Radix::Decimal && exponent < 0 {
        decimal_fraction_product::<L, W>(x.significand, exponent.unsigned_abs())
    } else {
        let mut limbs = [0_u64; PRODUCT_LIMBS];
        copy_limbs(&mut limbs, &x.significand);
        if radix == Radix::Decimal {
            times_power_of_five(&mut limbs, exponent.unsigned_abs());
        }
        integer_product::<W>(&limbs, exponent)
    };
    Some(Reduced {
        quadrant,
        angle: fraction.mul(&pi()).scale(-1),
    })
}

/// Copies the limbs of `value` to the low limbs of `limbs`.
fn copy_limbs<V: Limbs>(limbs: &mut [u64; PRODUCT_LIMBS], value: &V) {
    let count = usize::try_from(V::BITS.div_ceil(64)).expect("a limb count fits a usize");
    for (index, limb) in limbs.iter_mut().enumerate().take(count) {
        *limb = value.limb(index);
    }
}

/// Returns `c 10^-m 2 / pi mod 4` for a decimal argument from 1 up with a
/// fraction: the product of `X 2^-F` with `X = floor(c 2^F / 10^m)`, whose
/// ball grows by `2^-F` to hold the product of `c 10^-m`.
fn decimal_fraction_product<L: Limbs, W: Widen>(coefficient: L, digits: u64) -> (u64, Ball<W>) {
    let shift = fraction_bits::<W>();
    // `c` lies below `10^34`, below `2^113`, so `c 2^F` fits the `2W + 64`
    // bits of `W::Padded`.
    let mut scaled = coefficient.resize::<W::Padded>().shl(shift);
    let mut rest = digits;
    while rest > 0 {
        let step = rest.min(19);
        let power = 10_u64.pow(u32::try_from(step).expect("a step fits a u32"));
        scaled = limbs::divide_small(scaled, Divisor::new(power)).0;
        rest -= step;
    }
    let mut limbs = [0_u64; PRODUCT_LIMBS];
    copy_limbs(&mut limbs, &scaled);
    let (quadrant, fraction) = integer_product::<W>(&limbs, -i64::from(shift));
    (
        quadrant,
        fraction.widened(Radius::power_of_two(-i64::from(shift))),
    )
}

/// Returns `F`, the fraction bits of the product with the bits of 2/pi at
/// the width `W`.
fn fraction_bits<W: Widen>() -> u32 {
    2 * W::BITS - 66
}

/// Returns the integer `k` nearest the center of a ball of positive values
/// with a fraction, `k mod 4`, and the ball less `k`. A ball below 1/2 has
/// `k = 0`.
fn nearest<W: Widen>(product: &Ball<W>) -> (u64, Ball<W>) {
    let center = product.center();
    if center.is_zero() || center.top() < -1 {
        return (0, *product);
    }
    let (mantissa, exponent) = center.parts();
    let shift = u32::try_from(-exponent).expect("a product below 2^(W - 1) has a fraction");
    let half = <W::Double as Limbs>::ZERO.with_bit(shift - 1);
    let integer = mantissa.resize::<W::Double>().add(half).shr(shift);
    let quadrant = integer.field(0, 2);
    (quadrant, product.sub(&Ball::new(false, integer, 0)))
}

/// Returns `M 2^e 2 / pi mod 4` for an integer `M` of 64-bit limbs, least
/// significant first: the quadrant, and a ball of the fraction in `[0, 1]`.
/// The fraction moves to `[-1/2, 1/2)` with the next quadrant above 1/2.
/// A limb of negative weight takes the bits of 2/pi from above the point,
/// which are zero.
fn integer_product<W: Widen>(limbs: &[u64], exponent: i64) -> (u64, Ball<W>) {
    let fraction_bits = fraction_bits::<W>();
    let span = fraction_bits + 2;
    let mut sum = <W::Double as Limbs>::ZERO;
    for (index, &limb) in limbs.iter().enumerate().filter(|&(_, &limb)| limb != 0) {
        let weight = exponent + 64 * i64::try_from(index).expect("an index fits an i64");
        let window: W::Double = window(weight - 1, span);
        sum = sum.add(limbs::multiply_small(window, limb)).low_bits(span);
    }
    let quadrant = sum.field(fraction_bits, 2);
    let fraction = sum.low_bits(fraction_bits);
    let lowest = -i64::from(fraction_bits);
    // Each limb adds less than `2^(64 - F)` below the window, and there are
    // fewer than 2^8 limbs.
    let radius = Radius::power_of_two(64 + 8 + lowest);
    let ball = Ball::new(false, fraction, lowest).widened(radius);
    if fraction.bit(fraction_bits - 1) {
        ((quadrant + 1) & 3, ball.sub(&Ball::one()))
    } else {
        (quadrant, ball)
    }
}

/// Multiplies an integer of 64-bit limbs, least significant first, by
/// `5^count`, in steps of `5^27`, the largest power of 5 below 2^63.
fn times_power_of_five(limbs: &mut [u64; PRODUCT_LIMBS], count: u64) {
    let mut rest = count;
    while rest > 0 {
        let step = rest.min(27);
        let factor = 5_u64.pow(u32::try_from(step).expect("a step fits a u32"));
        let mut carry = 0_u64;
        for limb in limbs.iter_mut() {
            let product = u128::from(*limb) * u128::from(factor) + u128::from(carry);
            *limb = low_u64(product);
            carry = low_u64(product >> 64);
        }
        debug_assert_eq!(carry, 0, "c 5^q of a decimal format fits the product limbs");
        rest -= step;
    }
}

/// Returns the low 64 bits of a `u128`.
fn low_u64(value: u128) -> u64 {
    u64::try_from(value & u128::from(u64::MAX)).expect("the low 64 bits fit a u64")
}

/// Returns `count` bits of the fraction of 2/pi from the bit of weight
/// `2^-first` on, as an integer whose top bit weighs `2^-first`. The bits
/// before the point, at `first <= 0`, are zero.
fn window<D: Limbs>(first: i64, count: u32) -> D {
    let last = first + i64::from(count) - 1;
    debug_assert!(
        last <= i64::try_from(TWO_OVER_PI.len() * 8).expect("the table length fits"),
        "the window lies inside the table"
    );
    let mut value = D::ZERO;
    let mut taken = 0;
    let mut index = 0;
    while taken < count {
        let width = (count - taken).min(64);
        let start = last - i64::from(taken) - i64::from(width) + 1;
        value = value.with_limb(index, chunk(start) >> (64 - width));
        taken += width;
        index += 1;
    }
    value
}

/// Returns 64 bits of the fraction of 2/pi from the bit of weight `2^-start`
/// on, the first in the top bit.
fn chunk(start: i64) -> u64 {
    if start < 1 {
        let zeros = 1 - start;
        return if zeros >= 64 { 0 } else { chunk(1) >> zeros };
    }
    let bit = usize::try_from(start - 1).expect("a bit index fits a usize");
    let (byte, shift) = (bit / 8, bit % 8);
    let bytes = (0..9).fold(0_u128, |bytes, offset| {
        let value = TWO_OVER_PI.get(byte + offset).copied().unwrap_or(0);
        (bytes << 8) | u128::from(value)
    });
    low_u64(bytes >> (8 - shift))
}

#[cfg(test)]
mod tests {
    use super::{TWO_OVER_PI, chunk};
    use crate::elementary::ball::Ball;
    use crate::elementary::constants::pi;
    use crate::limbs::Limbs;

    /// The first 1984 bits of the table match `2 / pi` from the table of pi,
    /// which `constants` checks against Machin's formula. The last word of
    /// the 2048-bit quotient takes the truncation errors.
    #[test]
    fn two_over_pi_matches_pi() {
        let quotient = Ball::<[u64; 32]>::integer(2)
            .div(&pi())
            .expect("pi is far from zero");
        let (mantissa, exponent) = quotient.center().parts();
        // `2 / pi` lies in `[1/2, 1)`, so the top bit of the mantissa weighs
        // `2^-1`.
        assert_eq!(exponent + 32 * 64, 0);
        let words = (0..31).map(|index| chunk(1 + 64 * index));
        let mantissa_words = (0..31).map(|index| mantissa.limb(31 - index));
        assert!(words.eq(mantissa_words));
        assert_eq!(TWO_OVER_PI.len(), ((1 << 22) + 8192) / 8);
    }
}
