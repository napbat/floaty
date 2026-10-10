//! The trigonometric functions of an argument scaled by pi of IEEE 754-2019
//! section 9.2: `sinPi`, `cosPi`, and `tanPi`.
//!
//! `sin(pi x)` of a rational `x` is rational only where it is 0, ±1/2, or ±1,
//! and `tan(pi x)` only where it is 0 or ±1 (Niven). A binary or decimal
//! argument gives ±1/2 nowhere, because it is no odd multiple of 1/6. So
//! the exact values come from an argument whose `4x` is an integer, and
//! `4x mod 8` gives them, with the signs of IEEE 754-2019 section 9.2.1:
//!
//! | `4x mod 8` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
//! | --- | --- | --- | --- | --- | --- | --- | --- | --- |
//! | `sinPi` | ±0 | | 1 | | ±0 | | -1 | |
//! | `cosPi` | 1 | | +0 | | -1 | | +0 | |
//! | `tanPi` | ±0 | 1 | +inf | -1 | ∓0 | 1 | -inf | -1 |
//!
//! A zero of `sinPi`, and of `tanPi` at an even integer, takes the sign of
//! `x`. A zero of `tanPi` at an odd integer takes the other sign. An
//! infinity of `tanPi` signals divide-by-zero.
//!
//! Every other argument of at least 1/4 reduces exactly: `A = 4 (|x| mod 2)`
//! lies in `(0, 8)`, and the symmetries of the functions bring it to
//! `[0, 1]`, where the series of `sin` and `cos` at `pi A / 4` give the
//! value. `cosPi` of an argument below `2^-ceil((p + 5) / 2)`, or
//! `10^-ceil((p + 3) / 2)`, lies within one unit of the truncation below 1,
//! because `1 - cos(pi x)` is below `(pi x)^2 / 2`.

use super::ball::Ball;
use super::constants::pi;
use super::exact::radix_power;
use super::{
    Argument, Elementary, Function, Radix, Special, Target, argument, leading, near_one, number,
    series, ziv,
};
use crate::exact::Unrounded;
use crate::limbs::{self, Limbs, Widen};

/// A trigonometric function of an argument scaled by pi.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PiScaled {
    /// `sinPi`: `sin(pi x)`.
    Sin,
    /// `cosPi`: `cos(pi x)`.
    Cos,
    /// `tanPi`: `tan(pi x)`.
    Tan,
}

impl PiScaled {
    /// Returns the result at a zero with the sign `negative`.
    pub(super) fn at_zero<L>(self, negative: bool) -> Special<L> {
        match self {
            Self::Cos => Special::One { negative: false },
            Self::Sin | Self::Tan => Special::Zero { negative },
        }
    }
}

/// Returns the exact value at an argument whose `4x` is an integer, or the
/// argument to evaluate.
pub(super) fn special<L: Limbs>(function: PiScaled, x: Argument<L>, radix: Radix) -> Special<L> {
    let Some(octant) = octant(&x, radix) else {
        return Special::Evaluate(x);
    };
    let negative = x.negative;
    match (function, octant) {
        (PiScaled::Sin, 0 | 4) | (PiScaled::Tan, 0) => Special::Zero { negative },
        (PiScaled::Tan, 4) => Special::Zero {
            negative: !negative,
        },
        (PiScaled::Sin, 2) | (PiScaled::Cos, 0) | (PiScaled::Tan, 1 | 5) => {
            Special::One { negative: false }
        }
        (PiScaled::Sin, 6) | (PiScaled::Cos, 4) | (PiScaled::Tan, 3 | 7) => {
            Special::One { negative: true }
        }
        (PiScaled::Cos, 2 | 6) => Special::Zero { negative: false },
        (PiScaled::Tan, 2) => Special::Pole { negative: false },
        (PiScaled::Tan, 6) => Special::Pole { negative: true },
        _ => Special::Evaluate(x),
    }
}

/// Returns `4x mod 8`, from 0 to 7, when `4x` is an integer.
fn octant<L: Limbs>(x: &Argument<L>, radix: Radix) -> Option<u64> {
    let shift = x.exponent + 2;
    let low = match radix {
        Radix::Binary if shift >= 3 => 0,
        Radix::Binary if shift >= 0 => x.significand.limb(0) << shift,
        Radix::Binary => {
            let count = shift.unsigned_abs();
            if count >= x.significand.bit_length() || x.significand.any_below(count) {
                return None;
            }
            x.significand.shr(count).limb(0)
        }
        Radix::Decimal => {
            // A coefficient has at most 34 digits, so `4C` fits a `u128`, and
            // `4C 10^q` for `q` up to 2. `10^3` is a multiple of 8.
            let quadruple = 4 * limbs::to_u128(&x.significand);
            let exponent = x.exponent.unsigned_abs();
            let unit = 10_u128.checked_pow(exponent);
            let whole = if x.exponent >= 3 {
                0
            } else if x.exponent >= 0 {
                quadruple * unit.expect("10^2 fits a u128")
            } else {
                let unit = unit?;
                if !quadruple.is_multiple_of(unit) {
                    return None;
                }
                quadruple / unit
            };
            u64::try_from(whole & 7).expect("a value below 8 fits a u64")
        }
    };
    let low = low & 7;
    Some(if x.negative { (8 - low) & 7 } else { low })
}

/// Returns `function` of an argument that [`special`] gives, truncated to
/// `p + 3` bits or `p + 2` digits with the sticky bit set.
pub(super) fn evaluate<L: Elementary>(
    function: PiScaled,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    if function == PiScaled::Cos && leading(x, target.radix) < -target.half_tiny() {
        return near_one(true, target);
    }
    let value = Value {
        function,
        x: *x,
        radix: target.radix,
    };
    ziv::<L, _>(&value, target)
}

/// A function scaled by pi at an argument.
struct Value<L> {
    /// The function.
    function: PiScaled,
    /// The argument.
    x: Argument<L>,
    /// The radix of the argument.
    radix: Radix,
}

impl<L: Limbs> Function for Value<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let magnitude = Argument {
            negative: false,
            ..self.x
        };
        // An odd function takes the sign of `x`.
        let mut negative = self.x.negative && self.function != PiScaled::Cos;
        let Some((mut quarters, unit, exponent)) = quarters::<L, W>(&magnitude, self.radix) else {
            // `|x| < 1/4` needs no reduction.
            let angle = argument::<L, W>(&magnitude, self.radix)?.mul(&pi());
            let value = match self.function {
                PiScaled::Sin => series::sine(&angle),
                PiScaled::Cos => series::cosine(&angle),
                PiScaled::Tan => series::sine(&angle).div(&series::cosine(&angle))?,
            };
            return Some(if negative { value.negate() } else { value });
        };
        // `A = quarters / unit` lies in `(0, 8)`. A period of `tanPi`, and a
        // half period of `sinPi` and `cosPi`, is 4 in `A`. Past 2, `A` turns
        // to `4 - A`, which keeps `sinPi` and changes the sign of `cosPi`
        // and `tanPi`. Past 1, `A` turns to `2 - A`, which swaps the sine and
        // the cosine, and inverts the tangent.
        let four = unit.shl(2);
        if quarters.compare(&four).is_ge() {
            quarters = quarters.sub(four);
            negative ^= self.function != PiScaled::Tan;
        }
        if quarters.compare(&unit.shl(1)).is_gt() {
            quarters = four.sub(quarters);
            negative ^= self.function != PiScaled::Sin;
        }
        let swapped = quarters.compare(&unit).is_gt();
        if swapped {
            quarters = unit.shl(1).sub(quarters);
        }
        let angle = number::<W, W>(false, quarters, exponent, self.radix)?
            .mul(&pi())
            .scale(-2);
        let value = match (self.function, swapped) {
            (PiScaled::Sin, false) | (PiScaled::Cos, true) => series::sine(&angle),
            (PiScaled::Sin, true) | (PiScaled::Cos, false) => series::cosine(&angle),
            (PiScaled::Tan, false) => series::sine(&angle).div(&series::cosine(&angle))?,
            (PiScaled::Tan, true) => series::cosine(&angle).div(&series::sine(&angle))?,
        };
        Some(if negative { value.negate() } else { value })
    }
}

/// Returns `A = 4 (|x| mod 2)` of an argument of at least 1/4 as
/// `quarters / unit`, with `unit = RADIX^-exponent`, or `None` below 1/4.
/// The argument has a fraction, because `4x` is not an integer, so its
/// exponent is negative, and above `-(p + 3)` in binary and `-(p + 1)` in
/// decimal: the values fit `W`.
fn quarters<L: Limbs, W: Limbs>(x: &Argument<L>, radix: Radix) -> Option<(W, W, i64)> {
    let length = match radix {
        Radix::Binary => x.significand.bit_length(),
        Radix::Decimal => super::digit_count(&x.significand),
    };
    let top = i64::from(x.exponent) + i64::from(length);
    // `|x| < RADIX^top`, which is at most 1/4 for `top <= -2` in binary and
    // `top <= -1` in decimal, where `|x|` is below 1/10.
    let small = match radix {
        Radix::Binary => top <= -2,
        Radix::Decimal => top <= -1,
    };
    if small {
        return None;
    }
    debug_assert!(x.exponent < 0, "an argument whose 4x is not an integer");
    let exponent = i64::from(x.exponent);
    let count = exponent.unsigned_abs();
    let unit = radix_power::<W>(radix, count).expect("the unit of an argument fits the width");
    let (_, rest) = limbs::divide(x.significand.resize::<W>(), unit.shl(1));
    Some((rest.shl(2), unit, exponent))
}
