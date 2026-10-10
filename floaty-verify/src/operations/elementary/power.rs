//! The oracle of `pow` and `powr` of IEEE 754-2019 section 9.2 on the
//! binary and the decimal formats.
//!
//! The oracle states the special cases of section 9.2.1 itself. A finite
//! nonzero pair gives `|x|^y`, negative for `pow` of a negative `x` and an
//! odd `y`. MPFR's `pow` bounds it, exactly where the value is exact: a
//! binary pair is exact in MPFR. For a decimal pair, GMP gives `x^y`
//! exactly where `x^(1/d)` is rational for the denominator `d` of `y` and
//! the power is short. Any other value is irrational, or has more digits
//! than the grid. `|x|^y` is monotonic in each argument away from `|x| = 1`
//! and `y = 0`, so the least and the greatest MPFR value at the four corners
//! of the bounds of two decimal arguments bound it.
//!
//! With `t = y ln |x|`, MPFR takes a working precision near `log2(1 / |t|)`
//! bits where `x^y` lies next to 1, and its exponent range ends far past
//! the formats. So the oracle takes stand-ins on the same side of each point
//! of the grid: just beside 1 for `|t|` below `2^-(p + 11)` in binary and
//! below `2^-(4p + 70)` in decimal, and past the range for `|t|` from `4 R`,
//! as the exponentials do.

use core::cmp::Ordering;

use floaty::{Env, Flags};
use rug::float::Round;
use rug::ops::{Pow, PowAssignRound};
use rug::{Float as BigFloat, Integer, Rational};

use super::decimal::{
    DecimalRead, decimal_nan, decimal_out_of_range, decimal_truncation, read_decimal,
};
use super::{Bivariate, Shape, truncation};
use crate::mpfr::decimal::{self, DecimalFormat, DecimalValue};
use crate::mpfr::{self, Format, Input, Operand, Read};
use crate::operations::{Outcome, default_nan, special_operands};

/// The result of a special case of `pow` or `powr`.
enum Special {
    /// A finite nonzero pair: `|x|^y` with the sign.
    Evaluate(bool),
    /// An exact 1.
    One,
    /// An exact zero with the sign.
    Zero(bool),
    /// An exact infinity with the sign.
    Infinity(bool),
    /// An infinity with the sign at a pole, with divide-by-zero.
    Pole(bool),
    /// The default NaN, with invalid.
    Invalid,
}

/// Returns the special case of `pow`, or of `powr` when `powr` is set, at
/// two operands that are not NaNs. `parity` is `Some(true)` for an odd
/// integer `y`, `Some(false)` for an even one, and `None` otherwise.
fn special(powr: bool, x: &Shape, y: &Shape, parity: Option<bool>) -> Special {
    let odd = !powr && parity == Some(true);
    let (x_negative, y_negative) = (x.is_negative(), y.is_negative());
    // The order of `|x|` against 1.
    let magnitude = match x {
        Shape::Finite { minus_one, one, .. } if minus_one.is_eq() || one.is_eq() => Ordering::Equal,
        Shape::Finite { minus_one, one, .. } if minus_one.is_gt() && one.is_lt() => Ordering::Less,
        _ => Ordering::Greater,
    };
    let finite_x = matches!(x, Shape::Finite { .. });
    if matches!(y, Shape::Zero(_)) {
        return if !powr || (finite_x && !x_negative) {
            Special::One
        } else {
            Special::Invalid
        };
    }
    if powr && x_negative && !matches!(x, Shape::Zero(_)) {
        return Special::Invalid;
    }
    if finite_x && !x_negative && magnitude == Ordering::Equal {
        return if powr && matches!(y, Shape::Infinity(_)) {
            Special::Invalid
        } else {
            Special::One
        };
    }
    match (x, y) {
        (Shape::Zero(_), Shape::Infinity(true)) => Special::Infinity(false),
        (Shape::Zero(_), Shape::Infinity(false)) => Special::Zero(false),
        (Shape::Zero(_), _) if y_negative => Special::Pole(x_negative && odd),
        (Shape::Zero(_), _) => Special::Zero(x_negative && odd),
        (Shape::Infinity(_), _) if y_negative => Special::Zero(x_negative && odd),
        (Shape::Infinity(_), _) => Special::Infinity(x_negative && odd),
        (_, Shape::Infinity(_)) => match magnitude {
            Ordering::Equal => Special::One,
            Ordering::Less if y_negative => Special::Infinity(false),
            Ordering::Greater if !y_negative => Special::Infinity(false),
            _ => Special::Zero(false),
        },
        _ if x_negative && parity.is_none() => Special::Invalid,
        _ => Special::Evaluate(x_negative && odd),
    }
}

/// Returns `true` when `pow` gives exactly 1 whatever a quiet NaN operand
/// holds: `pow(x, ±0)`, and `pow(+1, y)`.
fn is_one(function: Bivariate, reads: &[Read; 2]) -> bool {
    let quiet = |read: &Read| !read.is_signaling() && !matches!(read, Read::Unsupported);
    let one = matches!(&reads[0], Read::Number(x) if *x == 1);
    let zero = matches!(&reads[1], Read::Number(y) if y.is_zero());
    function == Bivariate::Pow && reads.iter().all(quiet) && (one || zero)
}

/// Returns the expected result of `function` on two operands of a binary
/// format, `x` first, and the flags.
pub(super) fn expected<const N: usize>(
    function: Bivariate,
    x: &Operand<N>,
    y: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [x.read(env, &mut flags), y.read(env, &mut flags)];
    let one = || Outcome::Finite(BigFloat::with_val(2, 1));
    if is_one(function, &reads) {
        return (one(), flags);
    }
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    let [Read::Number(x), Read::Number(y)] = &reads else {
        unreachable!("every special operand has a result");
    };
    let parity = (y.is_finite() && y.is_integer())
        .then(|| !BigFloat::with_val(y.prec() + 1, y >> 1u32).is_integer());
    let shape = |value: &BigFloat| {
        if value.is_zero() {
            Shape::Zero(value.is_sign_negative())
        } else if value.is_infinite() {
            Shape::Infinity(value.is_sign_negative())
        } else {
            Shape::finite(value)
        }
    };
    let negative = match special(function == Bivariate::Powr, &shape(x), &shape(y), parity) {
        Special::Evaluate(negative) => negative,
        Special::One => return (one(), flags),
        Special::Zero(negative) => return (Outcome::from_value(format.zero(negative)), flags),
        Special::Infinity(negative) => {
            return (Outcome::from_value(format.infinity(negative, env)), flags);
        }
        Special::Pole(negative) => {
            let infinity = format.infinity(negative, env);
            return (Outcome::from_value(infinity), flags | Flags::DIVIDE_BY_ZERO);
        }
        Special::Invalid => return (default_nan(format, env), flags | Flags::INVALID),
    };
    let magnitude = x.clone().abs();
    // `(-1)^n` of an integer `n` is ±1 exactly.
    let input = if magnitude == 1 {
        Input {
            negative: false,
            exponent: 0,
            significand: Integer::from(1),
            sticky: false,
        }
    } else {
        binary_stand_in(&magnitude, y, format).unwrap_or_else(|| {
            truncation(format.precision, |precision, round| {
                Bivariate::Pow.mpfr(&magnitude, y, precision, round)
            })
        })
    };
    let input = Input { negative, ..input };
    let (value, more) = mpfr::round(&input, format, env);
    (Outcome::from_value(value), flags | more)
}

/// Returns the bounds of `t = y log2 |x|` at 64 bits. The logarithm reads
/// every bit of `|x|`, so a base next to 1 keeps its distance from 1.
fn exponent_bounds(magnitude: &BigFloat, y: &BigFloat) -> (BigFloat, BigFloat) {
    let bound = |round: Round| {
        let mut log = magnitude.clone();
        let _ = log.log2_round(round);
        // The product rounds away from zero in magnitude for the upper bound.
        BigFloat::with_val_round(64, &log * y, round).0.abs()
    };
    let (a, b) = (bound(Round::Down), bound(Round::Up));
    if a < b { (a, b) } else { (b, a) }
}

/// Returns a stand-in truncation of `|x|^y` for a binary pair: just beside
/// 1 for `|y log2 |x||` below `2^-(p + 11)`, where `1 + t` truncates to 1
/// with the sticky bit above 1, or to all ones below it, and `2^(p + 2)`
/// times `2^(±2^28)` with the sticky bit from `4 R`, past the range.
fn binary_stand_in(magnitude: &BigFloat, y: &BigFloat, format: &Format) -> Option<Input> {
    let (low, high) = exponent_bounds(magnitude, y);
    let above_one = (*magnitude > 1) == y.is_sign_positive();
    let precision = format.precision;
    let digits = i32::try_from(precision).ok()?;
    if high < (BigFloat::with_val(2, 1) >> (precision + 11)) {
        return Some(if above_one {
            Input {
                negative: false,
                exponent: -(digits + 2),
                significand: Integer::from(1) << (precision + 2),
                sticky: true,
            }
        } else {
            Input {
                negative: false,
                exponent: -(digits + 3),
                significand: (Integer::from(1) << (precision + 3)) - 1u32,
                sticky: true,
            }
        });
    }
    (low >= 4 * format.reach()).then(|| Input {
        negative: false,
        exponent: if above_one { 1 << 28 } else { -(1 << 28) },
        significand: Integer::from(1) << (precision + 2),
        sticky: true,
    })
}

/// Returns the expected result of `function` on two operands of a decimal
/// format, `x` first, and the flags.
pub(super) fn expected_decimal(
    function: Bivariate,
    x: &Operand<2>,
    y: &Operand<2>,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let mut flags = Flags::NONE;
    let reads = [
        read_decimal(x, env, &mut flags),
        read_decimal(y, env, &mut flags),
    ];
    let one = || DecimalValue::Finite {
        negative: false,
        coefficient: Integer::from(1),
        exponent: 0,
    };
    let quiet = reads.iter().all(|read| match read {
        DecimalRead::Nan(nan) => !nan.signaling,
        DecimalRead::Number(..) => true,
    });
    let x_one = matches!(&reads[0], DecimalRead::Number(_, Some(x)) if *x == 1);
    let y_zero = matches!(&reads[1], DecimalRead::Number(Shape::Zero(_), _));
    if function == Bivariate::Pow && quiet && (x_one || y_zero) {
        return (one(), flags);
    }
    let nans: Vec<_> = reads
        .iter()
        .filter_map(|read| match read {
            DecimalRead::Nan(nan) => Some(nan.clone()),
            DecimalRead::Number(..) => None,
        })
        .collect();
    if !nans.is_empty() {
        let (value, more) = decimal_nan(&nans, env);
        return (value, flags | more);
    }
    let [
        DecimalRead::Number(x_shape, x),
        DecimalRead::Number(y_shape, y),
    ] = reads
    else {
        unreachable!("an operand that is not a NaN is a number");
    };
    let parity = y
        .as_ref()
        .filter(|y| *y.denom() == 1)
        .map(|y| y.numer().is_odd());
    let powr = function == Bivariate::Powr;
    let (value, more) = match special(powr, &x_shape, &y_shape, parity) {
        Special::Evaluate(negative) => {
            let (x, y) = (
                x.expect("a finite operand has a value").abs(),
                y.expect("a finite operand has a value"),
            );
            decimal_power(&x, &y, negative, format, env)
        }
        Special::One => (one(), Flags::NONE),
        Special::Zero(negative) => (
            DecimalValue::Zero {
                negative,
                exponent: 0,
            },
            Flags::NONE,
        ),
        Special::Infinity(negative) => (DecimalValue::Infinity { negative }, Flags::NONE),
        Special::Pole(negative) => (DecimalValue::Infinity { negative }, Flags::DIVIDE_BY_ZERO),
        Special::Invalid => {
            let nan = DecimalValue::Nan {
                negative: env.nan.default_negative,
                payload: Integer::ZERO,
            };
            (nan, Flags::INVALID)
        }
    };
    (value, flags | more)
}

/// Returns `±x^y` for a positive decimal `x` and a finite nonzero `y`. `1^y`
/// is 1 exactly.
fn decimal_power(
    x: &Rational,
    y: &Rational,
    negative: bool,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let sign = |value: Rational| if negative { -value } else { value };
    if *x == 1 {
        return decimal::rational_to_decimal(&sign(Rational::from(1)), format, env);
    }
    let precision = format.precision;
    let magnitude = BigFloat::with_val(4 * precision + 64, x);
    let exponent = BigFloat::with_val(64, y);
    let (low, high) = exponent_bounds(&magnitude, &exponent);
    let above_one = (*x > 1) == (*y > 0);
    let signed = |value: BigFloat| if negative { -value } else { value };
    if high < (BigFloat::with_val(2, 1) >> (4 * precision + 70)) {
        // `1 ± 2^-(4p + 20)` lies beside 1 on the side of the value, closer
        // than one unit of `p + 2` digits.
        let tiny = BigFloat::with_val(2, 1) >> (4 * precision + 20);
        let value = if above_one {
            BigFloat::with_val(4 * precision + 24, 1u32 + &tiny)
        } else {
            BigFloat::with_val(4 * precision + 24, 1u32 - &tiny)
        };
        return decimal::to_decimal(&signed(value), format, env);
    }
    let bound = decimal_out_of_range(format);
    if low >= bound {
        let scale = i32::try_from(bound).expect("a decimal bound fits an i32");
        let value = BigFloat::with_val(2, 1) << if above_one { scale } else { -scale };
        return decimal::to_decimal(&signed(value), format, env);
    }
    // An exact value on the grid has at most `p` digits and an exponent in
    // the range, so its rational form has fewer than `4 (p + 2R)` bits.
    let limit = 4 * (u64::from(precision) + 2 * u64::try_from(bound).expect("a bound is positive"));
    if let Some(value) = rational_power(x, y, limit) {
        return decimal::rational_to_decimal(&sign(value), format, env);
    }
    let value = decimal_truncation(precision, |working, round| {
        let bound =
            |value: &Rational, direction| BigFloat::with_val_round(working, value, direction).0;
        [Round::Down, Round::Up]
            .into_iter()
            .flat_map(|x_round| {
                [Round::Down, Round::Up].map(|y_round| {
                    let mut value = bound(x, x_round);
                    let _ = value.pow_assign_round(&bound(y, y_round), round);
                    value
                })
            })
            .reduce(|chosen, value| match round {
                Round::Down if value < chosen => value,
                Round::Up if value > chosen => value,
                _ => chosen,
            })
            .expect("a box has four corners")
    });
    decimal::to_decimal(&signed(value), format, env)
}

/// Returns `x^y` when it is rational and its rational form has at most
/// `limit` bits: `y = m / d` in lowest terms with `x^(1/d)` rational. Any
/// other value is irrational, or longer than the grid of the format.
fn rational_power(x: &Rational, y: &Rational, limit: u64) -> Option<Rational> {
    let d = y.denom().to_u32()?;
    let root = |value: &Integer| {
        let (root, rest) = value.clone().root_rem(Integer::new(), d);
        rest.is_zero().then_some(root)
    };
    let base = Rational::from((root(x.numer())?, root(x.denom())?));
    let m = y.numer().to_i32()?;
    let bits = u64::from(base.numer().significant_bits() + base.denom().significant_bits());
    (bits * u64::from(m.unsigned_abs()) <= limit).then(|| base.pow(m))
}
