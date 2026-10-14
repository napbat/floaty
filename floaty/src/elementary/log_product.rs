//! `y log2(x)` and `y log2(1 + x)`, rounded once: the products of the x87
//! instructions `FYL2X` and `FYL2XP1`. IEEE 754-2019 defines neither.
//!
//! The special cases are those of the logarithm and of the product together.
//! A NaN and an unsupported operand take the NaN rule first. The logarithm
//! of a negative number, below -1 for `log2(1 + x)`, is invalid, and so is
//! a zero times an infinity. The logarithm is negative infinity at `x = 0`,
//! or at `x = -1` for `log2(1 + x)`. There, a finite nonzero `y` gives an
//! infinity from finite operands, which signals divide-by-zero, as IEEE
//! 754-2019 section 7.3 defines it. An infinite `y` gives the infinity
//! without the flag.
//!
//! The logarithm is rational only at a power of 2, where the product `y k`
//! is exact. Every other product of a finite nonzero `y` is irrational, and
//! a ball decides it. One argument lies too close to the grid for a ball:
//! `log2(1 + 2^k)` lies just above the integer `k` for `k >= p + 5`, so the
//! product lies just beyond `y k`.

use core::cmp::Ordering;

use super::ball::Ball;
use super::bivariate::{Outcome, Point, point};
use super::exact::{self, exact};
use super::{
    Argument, Base, Elementary, Function, Radix, Target, argument, beside, compare_one, leading,
    natural_log, series, ziv,
};
use crate::exact::Unrounded;
use crate::limbs::{self, Limbs, Widen};
use crate::unpacked::Unpacked;

/// The logarithm of an operand that is not a NaN.
enum Logarithm<L> {
    /// An exact zero with the sign.
    Zero(bool),
    /// Negative infinity at a pole.
    Pole,
    /// Positive infinity.
    Infinity,
    /// The logarithm of a negative number.
    Invalid,
    /// A finite nonzero logarithm of the argument, with its sign.
    Finite(bool, Argument<L>),
}

/// Returns `log2(x)`, or `log2(1 + x)` when `plus_one` is set, as a
/// [`Logarithm`].
fn logarithm<L: Limbs>(plus_one: bool, x: &Unpacked<L>, radix: Radix) -> Logarithm<L> {
    let (negative, x) = point(x);
    match (x, plus_one) {
        (Point::Zero, false) => Logarithm::Pole,
        (Point::Zero, true) => Logarithm::Zero(negative),
        // `log2` of a negative number, and `log2(1 - inf)`.
        (_, false) | (Point::Infinity, true) if negative => Logarithm::Invalid,
        (Point::Infinity, _) => Logarithm::Infinity,
        (Point::Finite(x), false) => match compare_one(&x, radix) {
            Ordering::Equal => Logarithm::Zero(false),
            order => Logarithm::Finite(order == Ordering::Less, x),
        },
        (Point::Finite(x), true) if negative => match compare_one(&x, radix) {
            Ordering::Greater => Logarithm::Invalid,
            Ordering::Equal => Logarithm::Pole,
            Ordering::Less => Logarithm::Finite(true, x),
        },
        (Point::Finite(x), true) => Logarithm::Finite(false, x),
    }
}

/// Returns `y log2(x)`, or `y log2(1 + x)` when `plus_one` is set, for
/// operands that are not NaNs.
pub(super) fn log_product<L: Elementary>(
    plus_one: bool,
    y: &Unpacked<L>,
    x: &Unpacked<L>,
    target: &Target,
) -> Outcome<L::Second> {
    let (y_negative, y) = point(y);
    match (logarithm(plus_one, x, target.radix), y) {
        (Logarithm::Invalid, _)
        | (Logarithm::Zero(_), Point::Infinity)
        | (Logarithm::Pole | Logarithm::Infinity, Point::Zero) => Outcome::Invalid,
        (Logarithm::Zero(negative) | Logarithm::Finite(negative, _), Point::Zero) => {
            Outcome::Zero {
                negative: y_negative != negative,
            }
        }
        (Logarithm::Zero(negative), Point::Finite(_)) => Outcome::Zero {
            negative: y_negative != negative,
        },
        (Logarithm::Pole, Point::Finite(_)) => Outcome::Pole {
            negative: !y_negative,
        },
        (Logarithm::Pole, Point::Infinity) => Outcome::Infinity {
            negative: !y_negative,
        },
        (Logarithm::Infinity, _) => Outcome::Infinity {
            negative: y_negative,
        },
        (Logarithm::Finite(negative, _), Point::Infinity) => Outcome::Infinity {
            negative: y_negative != negative,
        },
        (Logarithm::Finite(_, x), Point::Finite(y)) => {
            Outcome::Truncated(product::<L>(plus_one, &y, &x, target))
        }
    }
}

/// Returns the truncation of `y log2(x)`, or of `y log2(1 + x)`, for a
/// finite nonzero `y` and a finite nonzero logarithm.
fn product<L: Elementary>(
    plus_one: bool,
    y: &Argument<L>,
    x: &Argument<L>,
    target: &Target,
) -> Unrounded<L::Second> {
    let radix = target.radix;
    let times = |k: i64| {
        // `|k|` is below 2^63, so `y k` fits the second width, which has four
        // times the bits of the storage and at least 256.
        let factor = L::Second::ZERO.with_limb(0, k.unsigned_abs());
        let magnitude = limbs::multiply_fit(y.significand.resize::<L::Second>(), factor);
        (y.negative != (k < 0), magnitude)
    };
    if let Some(k) = exact::logarithm::<L, L::Second>(x, Base::Two, plus_one, radix) {
        let (negative, magnitude) = times(k);
        return exact(negative, magnitude, i64::from(y.exponent), target);
    }
    if plus_one
        && leading(x, radix) >= target.tiny()
        && let Some(k) = exact::logarithm::<L, L::Second>(x, Base::Two, false, radix)
    {
        // `log2(1 + x)` lies above the integer `k = log2 x` by less than
        // `1 / (x ln 2)`, so the product lies beyond `y k` by less than one
        // unit of the truncation of `y k`.
        let (negative, magnitude) = times(k);
        return beside(negative, magnitude, i64::from(y.exponent), false, target);
    }
    let function = Product {
        y: *y,
        x: *x,
        radix,
        plus_one,
    };
    ziv::<L, _>(&function, target)
}

/// The product `y log2(x)`, or `y log2(1 + x)`, of a finite nonzero `y`.
struct Product<L> {
    /// The factor.
    y: Argument<L>,
    /// The argument of the logarithm.
    x: Argument<L>,
    /// The radix of the arguments.
    radix: Radix,
    /// `true` for `y log2(1 + x)`.
    plus_one: bool,
}

impl<L: Limbs> Function for Product<L> {
    fn ball<W: Widen>(&self) -> Option<Ball<W>> {
        let log = if self.plus_one {
            series::log_one_plus(&argument::<L, W>(&self.x, self.radix)?)?
        } else {
            natural_log::<L, W>(&self.x, self.radix)?
        };
        Some(argument::<L, W>(&self.y, self.radix)?.mul(&Base::Two.over(&log)?))
    }
}
