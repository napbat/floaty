//! The functions of two arguments of IEEE 754-2019 section 9.2: `atan2` and
//! `atan2Pi`, which the module `atan2` evaluates, and `pow` and `powr`,
//! which the module `power` evaluates.

use core::cmp::Ordering;

use super::{Argument, Elementary, Radix, Target, atan2, compare_one, power};
use crate::exact::Unrounded;
use crate::limbs::Limbs;
use crate::unpacked::Unpacked;

/// A function of two arguments of IEEE 754-2019 section 9.2.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bivariate {
    /// `atan2(y, x)`: the angle of the point `(x, y)`.
    Atan2,
    /// `atan2Pi(y, x)`: `atan2(y, x) / pi`.
    Atan2Pi,
    /// `pow(x, y)`: `x^y`, with a negative `x` for an integer `y`.
    Pow,
    /// `powr(x, y)`: `e^(y ln x)`, for `x >= 0` alone.
    Powr,
}

/// The value of a function of two arguments at operands that are not NaNs.
pub(crate) enum Outcome<W> {
    /// An exact zero with the sign `negative`.
    Zero {
        /// The sign.
        negative: bool,
    },
    /// An exact infinity with the sign `negative`.
    Infinity {
        /// The sign.
        negative: bool,
    },
    /// An infinity at a pole, which signals divide-by-zero.
    Pole {
        /// The sign.
        negative: bool,
    },
    /// The default NaN, which signals invalid.
    Invalid,
    /// A truncation for the rounding routine, or an exact value.
    Truncated(Unrounded<W>),
}

/// An operand that is not a NaN, without its sign.
pub(super) enum Point<L> {
    /// A zero.
    Zero,
    /// A finite nonzero argument, with its sign.
    Finite(Argument<L>),
    /// An infinity.
    Infinity,
}

/// Returns the sign of an operand and its [`Point`].
pub(super) fn point<L: Limbs>(operand: &Unpacked<L>) -> (bool, Point<L>) {
    match *operand {
        Unpacked::Zero { negative, .. } => (negative, Point::Zero),
        Unpacked::Infinity { negative } => (negative, Point::Infinity),
        Unpacked::Finite {
            negative,
            exponent,
            significand,
        } => (
            negative,
            Point::Finite(Argument {
                negative,
                exponent,
                significand,
            }),
        ),
        Unpacked::Nan { .. } | Unpacked::Unsupported => {
            unreachable!("the NaN rule handles every NaN and unsupported operand")
        }
    }
}

/// Returns `true` when `function` gives exactly 1 whatever a quiet NaN
/// operand holds: `pow(x, ±0)` for any `x`, and `pow(+1, y)` for any `y`, by
/// IEEE 754-2019 section 9.2.1. A signaling NaN and an unsupported operand
/// still give a NaN.
pub(crate) fn is_one<L: Limbs>(
    function: Bivariate,
    first: &Unpacked<L>,
    second: &Unpacked<L>,
    radix: Radix,
) -> bool {
    let quiet = |operand: &Unpacked<L>| {
        !operand.is_signaling() && !matches!(operand, Unpacked::Unsupported)
    };
    let plus_one = match *first {
        Unpacked::Finite {
            negative: false,
            exponent,
            significand,
        } => {
            let x = Argument {
                negative: false,
                exponent,
                significand,
            };
            compare_one(&x, radix) == Ordering::Equal
        }
        _ => false,
    };
    function == Bivariate::Pow
        && quiet(first)
        && quiet(second)
        && (matches!(second, Unpacked::Zero { .. }) || plus_one)
}

/// Returns `function` of two operands that are not NaNs, in the order of
/// IEEE 754-2019: `y` first for `atan2`, and `x` first for `pow`. The value
/// is exact, or truncated for the rounding routine of the format as
/// [`super::evaluate`] truncates a value.
pub(crate) fn bivariate<L: Elementary>(
    function: Bivariate,
    first: &Unpacked<L>,
    second: &Unpacked<L>,
    target: &Target,
) -> Outcome<L::Second> {
    match function {
        Bivariate::Atan2 | Bivariate::Atan2Pi => {
            atan2::atan2::<L>(function == Bivariate::Atan2Pi, first, second, target)
        }
        Bivariate::Pow | Bivariate::Powr => {
            power::power::<L>(function == Bivariate::Powr, first, second, target)
        }
    }
}
