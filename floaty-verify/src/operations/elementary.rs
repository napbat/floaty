//! An oracle for the exponentials and the logarithms of IEEE 754-2019
//! section 9.2 on the binary and the decimal formats: `exp`, `expm1`,
//! `exp2`, `exp2m1`, `exp10`, `exp10m1`, `log`, `log2`, `log10`, `logp1`,
//! `log2p1`, and `log10p1`.
//!
//! MPFR evaluates the function with directed rounding, at a working precision
//! that doubles until the bounds below and above decide the truncation of the
//! value to `p + 3` bits, or to `p + 2` digits. An irrational value never
//! lies on a point of the grid, so that truncation with a sticky bit rounds
//! as the true value does. For a binary format, [`crate::mpfr::round`]
//! rounds it. For a decimal format, [`decimal::to_decimal`] rounds a binary
//! value strictly inside the truncation interval, which rounds as the true
//! value does. A decimal argument `C * 10^q` enters MPFR as two bounds, and
//! every function increases, so the bounds of the arguments give bounds of
//! the results.
//!
//! `b^x` for `b = 2` or `10` and a rational `x = m / n` in lowest terms is
//! rational only for `n = 1`: the exponent of each prime of `b^m` would be a
//! multiple of `n`. In the same way, `log_b y` is rational only for an
//! integer power `y` of `b`. `e^x` and `ln y` are irrational but at `x = 0`
//! and `y = 1` (Lindemann). GMP gives every rational logarithm, and every
//! rational decimal result, exactly: MPFR bounds of a value on the grid
//! never decide its truncation, and a decimal value such as `10^-3` has no
//! binary form. A rational binary exponential is exact at the working
//! precision, where the two bounds meet, or lies off the grid.
//!
//! An argument of an exponential at or above `4 R` in magnitude, with
//! `R = max(emax, p - emin) + 3` in powers of the radix, or `R` for `10^x`
//! in binary, gives a result past the range of the format, which the sign
//! of the argument decides without MPFR. `b^x - 1` of a negative `x` below
//! `-(p + 11)`, or `-(4p + 16)` for a decimal format, lies just above -1,
//! which the oracle also decides from that bound: MPFR takes a working
//! precision near `|x|` bits there. IEEE 754-2019 section 9.2.1 gives the
//! special cases, and floaty's operand rule gives the NaN and unsupported
//! cases. A decimal result that is exact takes the exponent nearest 0, by
//! the rule of floaty.

use core::cmp::Ordering;

use floaty::format::Standard;
use floaty::{Decoded, Env, Flags, Float};
use rug::float::Round;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer, Rational};

use super::{Outcome, default_nan, special_operands};
use crate::mpfr::decimal::{self, DecimalFormat, DecimalValue};
use crate::mpfr::{self, Format, Input, Nan, Operand, Read, select_nan};

/// A function of the oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Function {
    /// `e^x`.
    Exp,
    /// `e^x - 1`.
    ExpM1,
    /// `2^x`.
    Exp2,
    /// `2^x - 1`.
    Exp2M1,
    /// `10^x`.
    Exp10,
    /// `10^x - 1`.
    Exp10M1,
    /// The natural logarithm `ln x`.
    Log,
    /// `log_2 x`.
    Log2,
    /// `log_10 x`.
    Log10,
    /// `ln(1 + x)`.
    LogP1,
    /// `log_2(1 + x)`.
    Log2P1,
    /// `log_10(1 + x)`.
    Log10P1,
    /// `sinh x`.
    Sinh,
    /// `cosh x`.
    Cosh,
    /// `tanh x`.
    Tanh,
    /// `asinh x`.
    Asinh,
    /// `acosh x`.
    Acosh,
    /// `atanh x`.
    Atanh,
    /// `sin(pi x)`.
    SinPi,
    /// `cos(pi x)`.
    CosPi,
    /// `tan(pi x)`.
    TanPi,
}

impl Function {
    /// Every function.
    pub const ALL: [Self; 21] = [
        Self::Exp,
        Self::ExpM1,
        Self::Exp2,
        Self::Exp2M1,
        Self::Exp10,
        Self::Exp10M1,
        Self::Log,
        Self::Log2,
        Self::Log10,
        Self::LogP1,
        Self::Log2P1,
        Self::Log10P1,
        Self::Sinh,
        Self::Cosh,
        Self::Tanh,
        Self::Asinh,
        Self::Acosh,
        Self::Atanh,
        Self::SinPi,
        Self::CosPi,
        Self::TanPi,
    ];

    /// The exponentials and the logarithms, the first 12 of [`Self::ALL`].
    pub const EXPONENTIALS_AND_LOGARITHMS: &[Self] = Self::ALL.split_at(12).0;

    /// The hyperbolic and the trigonometric functions, the rest of
    /// [`Self::ALL`]. Their tests run apart from the first group, so that
    /// nextest runs both at once.
    pub const HYPERBOLIC_AND_TRIGONOMETRIC: &[Self] = Self::ALL.split_at(12).1;

    /// Returns floaty's result of the function on `x` in `env`, and the
    /// flags.
    #[must_use]
    pub fn floaty<S: Standard<W>, const W: usize>(
        self,
        x: Float<S, W>,
        env: Env,
    ) -> (Float<S, W>, Flags) {
        match self {
            Self::Exp => x.exp_with(env),
            Self::ExpM1 => x.exp_m1_with(env),
            Self::Exp2 => x.exp2_with(env),
            Self::Exp2M1 => x.exp2_m1_with(env),
            Self::Exp10 => x.exp10_with(env),
            Self::Exp10M1 => x.exp10_m1_with(env),
            Self::Log => x.log_with(env),
            Self::Log2 => x.log2_with(env),
            Self::Log10 => x.log10_with(env),
            Self::LogP1 => x.log_p1_with(env),
            Self::Log2P1 => x.log2_p1_with(env),
            Self::Log10P1 => x.log10_p1_with(env),
            Self::Sinh => x.sinh_with(env),
            Self::Cosh => x.cosh_with(env),
            Self::Tanh => x.tanh_with(env),
            Self::Asinh => x.asinh_with(env),
            Self::Acosh => x.acosh_with(env),
            Self::Atanh => x.atanh_with(env),
            Self::SinPi => x.sin_pi_with(env),
            Self::CosPi => x.cos_pi_with(env),
            Self::TanPi => x.tan_pi_with(env),
        }
    }

    /// Returns `true` for an exponential.
    fn is_exponential(self) -> bool {
        matches!(
            self,
            Self::Exp | Self::ExpM1 | Self::Exp2 | Self::Exp2M1 | Self::Exp10 | Self::Exp10M1
        )
    }

    /// Returns `true` for `b^x - 1` or `log_b(1 + x)`.
    fn is_shifted(self) -> bool {
        matches!(
            self,
            Self::ExpM1 | Self::Exp2M1 | Self::Exp10M1 | Self::LogP1 | Self::Log2P1 | Self::Log10P1
        )
    }

    /// Returns `true` for a trigonometric function scaled by pi.
    fn is_pi_scaled(self) -> bool {
        matches!(self, Self::SinPi | Self::CosPi | Self::TanPi)
    }

    /// Returns `true` for a hyperbolic function or its inverse.
    fn is_hyperbolic(self) -> bool {
        matches!(
            self,
            Self::Sinh | Self::Cosh | Self::Tanh | Self::Asinh | Self::Acosh | Self::Atanh
        )
    }

    /// Returns the base 2 or 10 of an exponential or a logarithm, or `None`
    /// for e and the hyperbolic functions.
    fn base(self) -> Option<u32> {
        match self {
            Self::Exp2 | Self::Exp2M1 | Self::Log2 | Self::Log2P1 => Some(2),
            Self::Exp10 | Self::Exp10M1 | Self::Log10 | Self::Log10P1 => Some(10),
            _ => None,
        }
    }

    /// Returns the value at `x` that MPFR rounds to `precision` bits in the
    /// direction `round`. `x` must have at most `precision` bits.
    fn mpfr(self, x: &BigFloat, precision: u32, round: Round) -> BigFloat {
        let mut value = BigFloat::with_val(precision, x);
        let _ = match self {
            Self::Exp => value.exp_round(round),
            Self::ExpM1 => value.exp_m1_round(round),
            Self::Exp2 => value.exp2_round(round),
            Self::Exp2M1 => value.exp2_m1_round(round),
            Self::Exp10 => value.exp10_round(round),
            Self::Exp10M1 => value.exp10_m1_round(round),
            Self::Log => value.ln_round(round),
            Self::Log2 => value.log2_round(round),
            Self::Log10 => value.log10_round(round),
            Self::LogP1 => value.ln_1p_round(round),
            Self::Log2P1 => value.log2_1p_round(round),
            Self::Log10P1 => value.log10_1p_round(round),
            Self::Sinh => value.sinh_round(round),
            Self::Cosh => value.cosh_round(round),
            Self::Tanh => value.tanh_round(round),
            Self::Asinh => value.asinh_round(round),
            Self::Acosh => value.acosh_round(round),
            Self::Atanh => value.atanh_round(round),
            Self::SinPi => value.sin_pi_round(round),
            Self::CosPi => value.cos_pi_round(round),
            Self::TanPi => value.tan_pi_round(round),
        };
        value
    }

    /// Returns the result of the special cases of IEEE 754-2019 section
    /// 9.2.1 at an operand, or [`Special::Evaluate`].
    fn special(self, x: &Shape) -> Special {
        if self.is_pi_scaled() {
            return match x {
                Shape::Zero(_) if self == Self::CosPi => Special::One(false),
                Shape::Zero(negative) => Special::Zero(*negative),
                Shape::Infinity(_) => Special::Invalid,
                Shape::Finite { .. } => Special::Evaluate,
            };
        }
        if self.is_hyperbolic() {
            return self.hyperbolic_special(x);
        }
        let (exponential, shifted) = (self.is_exponential(), self.is_shifted());
        match x {
            Shape::Zero(negative) if shifted => Special::Zero(*negative),
            Shape::Zero(_) if exponential => Special::One(false),
            Shape::Zero(_) => Special::Pole(true),
            Shape::Infinity(false) => Special::Infinity(false),
            Shape::Infinity(true) if !exponential => Special::Invalid,
            Shape::Infinity(true) if shifted => Special::One(true),
            Shape::Finite { .. } if exponential => Special::Evaluate,
            Shape::Finite { minus_one, .. } if shifted => match minus_one {
                Ordering::Less => Special::Invalid,
                Ordering::Equal => Special::Pole(true),
                Ordering::Greater => Special::Evaluate,
            },
            Shape::Finite { negative: true, .. } => Special::Invalid,
            // `e^-inf` of an exponential, and the logarithm of 1.
            Shape::Infinity(true)
            | Shape::Finite {
                one: Ordering::Equal,
                ..
            } => Special::Zero(false),
            Shape::Finite { .. } => Special::Evaluate,
        }
    }

    /// Returns the exact value of `sinPi`, `cosPi`, or `tanPi` at an
    /// argument reduced by [`quarter`]: 0, ±1, or a pole, which MPFR gives
    /// with the signs of IEEE 754-2019 section 9.2.1. Returns `None` for an
    /// irrational value, of `sinPi` and `cosPi` at an odd multiple of 1/4.
    fn at_quarter(self, reduced: &BigFloat) -> Option<Special> {
        let value = self.mpfr(reduced, 64, Round::Nearest);
        let negative = value.is_sign_negative();
        if value.is_zero() {
            Some(Special::Zero(negative))
        } else if value.is_infinite() {
            Some(Special::Pole(negative))
        } else if value.clone().abs() == 1 {
            Some(Special::One(negative))
        } else {
            None
        }
    }

    /// Returns the result of the special cases of IEEE 754-2019 section
    /// 9.2.1 for a hyperbolic function. `acosh` takes `x >= 1`, and `atanh`
    /// takes `|x| <= 1` with poles at ±1.
    fn hyperbolic_special(self, x: &Shape) -> Special {
        match (self, x) {
            (Self::Cosh, Shape::Zero(_)) => Special::One(false),
            (Self::Acosh, Shape::Zero(_)) => Special::Invalid,
            (_, Shape::Zero(negative)) => Special::Zero(*negative),
            (Self::Sinh | Self::Asinh, Shape::Infinity(negative)) => Special::Infinity(*negative),
            (Self::Cosh, Shape::Infinity(_)) | (Self::Acosh, Shape::Infinity(false)) => {
                Special::Infinity(false)
            }
            (Self::Tanh, Shape::Infinity(negative)) => Special::One(*negative),
            (_, Shape::Infinity(_)) => Special::Invalid,
            (Self::Acosh, Shape::Finite { one, .. }) => match one {
                Ordering::Less => Special::Invalid,
                Ordering::Equal => Special::Zero(false),
                Ordering::Greater => Special::Evaluate,
            },
            (Self::Atanh, Shape::Finite { minus_one, one, .. }) => match (minus_one, one) {
                (Ordering::Less, _) | (_, Ordering::Greater) => Special::Invalid,
                (Ordering::Equal, _) => Special::Pole(true),
                (_, Ordering::Equal) => Special::Pole(false),
                _ => Special::Evaluate,
            },
            (_, Shape::Finite { .. }) => Special::Evaluate,
        }
    }

    /// Returns the value at `x` when it is rational: a power of 2 or 10, or
    /// that power less 1, at an integer `x` of magnitude below `2^16`, or an
    /// integer logarithm.
    fn rational(self, x: &Rational) -> Option<Rational> {
        let base = self.base()?;
        if !self.is_exponential() {
            return self.logarithm(x).map(Rational::from);
        }
        if *x.denom() != 1 {
            return None;
        }
        let n = x.numer().to_i32().filter(|n| n.unsigned_abs() < 1 << 16)?;
        let power = Integer::from(Integer::u_pow_u(base, n.unsigned_abs()));
        let value = if n >= 0 {
            Rational::from(power)
        } else {
            Rational::from((1, power))
        };
        Some(if self.is_shifted() {
            value - 1u32
        } else {
            value
        })
    }

    /// Returns `k` for a logarithm in base 2 or 10 when `x`, or `1 + x` for
    /// a shifted logarithm, is `b^k`.
    fn logarithm(self, x: &Rational) -> Option<i64> {
        let base = self.base().filter(|_| !self.is_exponential())?;
        let y = if self.is_shifted() {
            x.clone() + 1u32
        } else {
            x.clone()
        };
        let (numerator, denominator) = y.into_numer_denom();
        let (power, negative) = if denominator == 1 {
            (numerator, false)
        } else if numerator == 1 {
            (denominator, true)
        } else {
            return None;
        };
        // `b^k = 2^k 5^k` for `b = 10`, so the zero bits at the bottom count
        // `k` in both bases. `5^k` has `floor(k log2 5) + 1` bits, and
        // `log2 5 = 2.3219...`, which rules out most values before the power.
        let k = power.find_one(0).expect("a power is not zero");
        let rest = power >> k;
        let is_power = if base == 2 {
            rest == 1
        } else {
            let bits = u64::from(rest.significant_bits());
            let estimate = u64::from(k) * 2_321_928 / 1_000_000 + 1;
            bits.abs_diff(estimate) <= 1 && rest == Integer::from(Integer::u_pow_u(5, k))
        };
        let k = i64::from(k);
        is_power.then_some(if negative { -k } else { k })
    }

    /// Returns `k` for a logarithm in base 2 or 10 of a binary format when
    /// `x`, or `1 + x` for a shifted logarithm, is `b^k`. Only the integer
    /// significand of `x` enters GMP, because `x` can reach `2^(2^27)`.
    fn binary_logarithm(self, x: &BigFloat, precision: u32) -> Option<i64> {
        let base = self.base().filter(|_| !self.is_exponential())?;
        if self.is_shifted() {
            // `x = b^k - 1` lies in `(-1, -1/2]`, or holds `k` bits of `b^k`
            // below `2^p`.
            let exponent = x.get_exp()?;
            if exponent.unsigned_abs() > precision + 1 {
                return None;
            }
            return self.logarithm(&x.to_rational()?);
        }
        let (significand, exponent) = x.to_integer_exp()?;
        let zeros = significand.find_one(0)?;
        let odd = significand >> zeros;
        let k = i64::from(exponent) + i64::from(zeros);
        // `x = m 2^k` with `m` odd is `2^k` for `m = 1`, and `10^k` for
        // `m = 5^k`.
        let is_power = if base == 2 {
            odd == 1
        } else {
            let count = u32::try_from(k).ok()?;
            count < 64 * precision && odd == Integer::from(Integer::u_pow_u(5, count))
        };
        is_power.then_some(k)
    }
}

/// An operand that is not a NaN.
enum Shape {
    /// A zero with its sign.
    Zero(bool),
    /// An infinity with its sign.
    Infinity(bool),
    /// A nonzero finite value, with its sign and its order against -1 and 1.
    Finite {
        /// The sign.
        negative: bool,
        /// The order of the value against -1.
        minus_one: Ordering,
        /// The order of the value against 1.
        one: Ordering,
    },
}

impl Shape {
    /// Returns the shape of a nonzero finite value.
    fn finite<T: PartialOrd<i32>>(x: &T) -> Self {
        let order = |bound| x.partial_cmp(&bound).expect("a finite value has an order");
        Self::Finite {
            negative: order(0) == Ordering::Less,
            minus_one: order(-1),
            one: order(1),
        }
    }
}

/// Returns `±(4|x| mod 8) / 4` with the sign of `x` when `4x` is an integer:
/// the argument of `sinPi`, `cosPi`, and `tanPi` less whole periods, which
/// keeps the sign of a zero result.
fn quarter(x: &Rational) -> Option<BigFloat> {
    let quadruple = x.clone().abs() * 4u32;
    if *quadruple.denom() != 1 {
        return None;
    }
    let reduced = BigFloat::with_val(8, quadruple.numer().mod_u(8)) >> 2u32;
    Some(if *x < 0 { -reduced } else { reduced })
}

/// Returns [`quarter`] of a value of a binary format. A value at or above
/// `2^(p + 2)` is a multiple of 8, which reduces to a zero, and `4x` of a
/// nonzero value below 1/4 is no integer: neither enters GMP, because a
/// value can reach `2^(±2^27)`.
fn binary_quarter(x: &BigFloat, precision: u32) -> Option<BigFloat> {
    let exponent = x.get_exp()?;
    if exponent >= i32::try_from(precision + 3).ok()? {
        let zero = BigFloat::new(2);
        return Some(if x.is_sign_negative() { -zero } else { zero });
    }
    if exponent <= -2 {
        return None;
    }
    quarter(&x.to_rational()?)
}

/// Returns the other directed rounding of `Down` and `Up`.
fn reverse(round: Round) -> Round {
    if round == Round::Down {
        Round::Up
    } else {
        Round::Down
    }
}

/// The result of a function by the special cases of IEEE 754-2019 section
/// 9.2.1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// No special case: the function evaluates.
    Evaluate,
    /// An exact 1 with the sign.
    One(bool),
    /// A zero with the sign.
    Zero(bool),
    /// An infinity with the sign.
    Infinity(bool),
    /// An infinity with the sign at a pole, with divide-by-zero.
    Pole(bool),
    /// The default NaN, with invalid.
    Invalid,
}

/// Returns the bound past which an argument of an exponential overflows or
/// underflows a binary format, from its [`Format::reach`] `R`: `4 R` bounds
/// `e^x` and `2^x`, and `R` bounds `10^x`. The bounds keep `b^x` inside the
/// exponent range of MPFR for every layout.
fn binary_out_of_range(function: Function, format: &Format) -> i64 {
    let factor = if function.base() == Some(10) { 1 } else { 4 };
    factor * format.reach()
}

/// Returns the bound `4 R` past which an argument of an exponential
/// overflows or underflows a decimal format, with
/// `R = max(emax, p - emin) + 3`: `b^|x| >= 2^(4R) = 16^R > 10^R`, and every
/// finite value of the format lies between `10^-R` and `10^R`.
fn decimal_out_of_range(format: DecimalFormat) -> i64 {
    let emin = 1 - i64::from(format.emax);
    let reach = i64::from(format.emax).max(i64::from(format.precision) - emin);
    4 * (reach + 3)
}

/// The working precision past which the oracle stops: no result of a format
/// needs anything near it.
const PRECISION_LIMIT: u32 = 1 << 20;

/// Returns the expected result of `function` on an operand, and the flags.
#[must_use]
pub fn expected<const N: usize>(
    function: Function,
    x: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let reads = [x.read(env, &mut flags)];
    if let Some((value, special)) = special_operands(&reads, format, env) {
        return (value, flags | special);
    }
    let [Read::Number(value)] = &reads else {
        unreachable!("every special operand has a result");
    };
    let (outcome, more) = number(function, value, format, env);
    (outcome, flags | more)
}

/// Returns the expected result of `function` on a number, and the flags.
fn number(function: Function, x: &BigFloat, format: &Format, env: &Env) -> (Outcome, Flags) {
    let negative = x.is_sign_negative();
    let shape = if x.is_zero() {
        Shape::Zero(negative)
    } else if x.is_infinite() {
        Shape::Infinity(negative)
    } else {
        Shape::finite(x)
    };
    let exact = |value| (Outcome::from_value(value), Flags::NONE);
    let mut special = function.special(&shape);
    if matches!(special, Special::Evaluate)
        && function.is_pi_scaled()
        && let Some(reduced) = binary_quarter(x, format.precision)
        && let Some(value) = function.at_quarter(&reduced)
    {
        special = value;
    }
    match special {
        Special::Evaluate => {}
        Special::One(negative) => {
            let one = BigFloat::with_val(2, if negative { -1 } else { 1 });
            return (Outcome::Finite(one), Flags::NONE);
        }
        Special::Zero(negative) => return exact(format.zero(negative)),
        Special::Infinity(negative) => return exact(format.infinity(negative, env)),
        Special::Pole(negative) => {
            let infinity = format.infinity(negative, env);
            return (Outcome::from_value(infinity), Flags::DIVIDE_BY_ZERO);
        }
        Special::Invalid => return (default_nan(format, env), Flags::INVALID),
    }
    let far = x.clone().abs() >= binary_out_of_range(function, format);
    // MPFR takes a working precision near `|x|` bits for `b^x - 1` of a
    // large negative `x`, and for `tanh` of a large `|x|`, so the oracle
    // takes those values from their bounds.
    let beyond = x.clone().abs() >= i64::from(format.precision + 11);
    let below_one = function.is_exponential() && function.is_shifted() && negative;
    let input = if (below_one || function == Function::Tanh) && beyond {
        // `b^x <= 2^x` lies below `2^-(p + 11)`, and so does
        // `1 - |tanh x| < 2 e^-2|x|`. So `b^x - 1` and `tanh x` lie inside
        // ±1 by less than one unit of `p + 3` bits: each truncates to all
        // ones below 1 in magnitude.
        Input {
            negative,
            exponent: -i32::try_from(format.precision + 3).expect("a precision fits an i32"),
            significand: (Integer::from(1) << (format.precision + 3)) - 1u32,
            sticky: true,
        }
    } else if far
        && !below_one
        && (function.is_exponential() || matches!(function, Function::Sinh | Function::Cosh))
    {
        // A value far past the range: `2^(p + 2)` times `2^(±2^28)`, with the
        // sticky bit. `b^x` overflows or underflows by the sign of `x`, and
        // `sinh` and `cosh` overflow, `sinh` with the sign of `x`. `b^x - 1`
        // of a negative `x` lies near -1, which MPFR bounds decide below
        // `|x| = p + 11`.
        let (sign, overflow) = match function {
            Function::Sinh => (negative, true),
            Function::Cosh => (false, true),
            _ => (false, !negative),
        };
        Input {
            negative: sign,
            exponent: if overflow { 1 << 28 } else { -(1 << 28) },
            significand: Integer::from(1) << (format.precision + 2),
            sticky: true,
        }
    } else if let Some(k) = function.binary_logarithm(x, format.precision) {
        Input {
            negative: k < 0,
            exponent: 0,
            significand: Integer::from(k.unsigned_abs()),
            sticky: false,
        }
    } else {
        truncation(format.precision, |precision, round| {
            function.mpfr(x, precision, round)
        })
    };
    let (value, flags) = mpfr::round(&input, format, env);
    (Outcome::from_value(value), flags)
}

/// Returns the truncation of a value to `precision + 3` bits, with the
/// sticky bit set, or the value itself where the bounds meet. `evaluate`
/// gives the value rounded at a working precision in a direction. The
/// working precision doubles until the bounds decide the truncation.
///
/// # Panics
///
/// Panics when the working precision reaches [`PRECISION_LIMIT`].
pub(super) fn truncation(precision: u32, evaluate: impl Fn(u32, Round) -> BigFloat) -> Input {
    let mut working = precision + 64;
    loop {
        let (low, high) = (evaluate(working, Round::Down), evaluate(working, Round::Up));
        if low == high {
            let (significand, exponent) = low
                .to_integer_exp()
                .expect("a bound of a finite value is finite");
            return Input {
                negative: significand < 0,
                exponent,
                significand: significand.abs(),
                sticky: false,
            };
        }
        if let Some(input) = common_truncation(&low, &high, precision) {
            return input;
        }
        working *= 2;
        assert!(
            working < PRECISION_LIMIT,
            "the bounds decide the truncation"
        );
    }
}

/// Returns the truncation to `precision + 3` bits of every value strictly
/// between two finite bounds of one sign, or `None` when the bounds do not
/// decide it. A bound can be a power of two while the value is in the
/// binade below, as `e^x` is for a tiny negative `x`.
fn common_truncation(low: &BigFloat, high: &BigFloat, precision: u32) -> Option<Input> {
    let negative = low.is_sign_negative();
    if negative != high.is_sign_negative() || low.is_zero() || high.is_zero() {
        return None;
    }
    let (small, large) = if negative { (high, low) } else { (low, high) };
    // The value is in the binade of `small` or above it. The magnitude of
    // `small` lies in `[2^top, 2^(top + 1))`.
    let top = small.get_exp()? - 1;
    let lowest = top - i32::try_from(precision + 2).ok()?;
    let below = (small.clone().abs() >> lowest).floor().to_integer()?;
    let above = (large.clone().abs() >> lowest).ceil().to_integer()? - 1u32;
    (below == above).then_some(Input {
        negative,
        exponent: lowest,
        significand: below,
        sticky: true,
    })
}

/// Returns the expected result of `function` on a decimal operand, and the
/// flags.
///
/// # Panics
///
/// Panics for an unsupported encoding, which a decimal format does not have.
#[must_use]
pub fn expected_decimal(
    function: Function,
    x: &Operand<2>,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let mut flags = if x.subnormal {
        Flags::DENORMAL_INPUT
    } else {
        Flags::NONE
    };
    let (shape, value) = match x.decoded {
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => {
            let offered = Nan {
                negative,
                signaling,
                payload: Integer::from_digits(&payload, Order::Lsf),
            };
            let nan = select_nan(&[offered], env);
            if signaling {
                flags |= Flags::INVALID;
            }
            let value = DecimalValue::Nan {
                negative: nan.negative,
                payload: nan.payload,
            };
            return (value, flags);
        }
        Decoded::Zero { negative, .. } => (Shape::Zero(negative), None),
        Decoded::Finite { negative, .. } if x.subnormal && env.denormals_are_zero => {
            (Shape::Zero(negative), None)
        }
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let coefficient = Integer::from_digits(&significand, Order::Lsf);
            let power = Integer::from(Integer::u_pow_u(10, exponent.unsigned_abs()));
            let magnitude = if exponent >= 0 {
                Rational::from(coefficient * power)
            } else {
                Rational::from((coefficient, power))
            };
            let value = if negative { -magnitude } else { magnitude };
            (Shape::finite(&value), Some(value))
        }
        Decoded::Infinity { negative } => (Shape::Infinity(negative), None),
        Decoded::Unsupported => panic!("a decimal format has no unsupported encoding"),
    };
    let (value, more) = decimal_number(function, &shape, value.as_ref(), format, env);
    (value, flags | more)
}

/// Returns the expected result of `function` on a decimal operand that is
/// not a NaN, and the flags.
fn decimal_number(
    function: Function,
    shape: &Shape,
    value: Option<&Rational>,
    format: DecimalFormat,
    env: &Env,
) -> (DecimalValue, Flags) {
    let mut special = function.special(shape);
    if matches!(special, Special::Evaluate)
        && function.is_pi_scaled()
        && let Some(reduced) = value.and_then(quarter)
        && let Some(at_quarter) = function.at_quarter(&reduced)
    {
        special = at_quarter;
    }
    if let Some(result) = decimal_special(special, env) {
        return result;
    }
    let x = value.expect("only a finite value evaluates");
    if let Some(result) = decimal_stand_in(function, x, format, env) {
        return result;
    }
    if let Some(value) = function.rational(x) {
        return decimal::rational_to_decimal(&value, format, env);
    }
    // `cosh` is even, and increases with `|x|`, so its argument enters MPFR
    // as `|x|`. The functions scaled by pi do not increase, but the bounds of
    // an argument lie closer than any extremum or pole, so the least and the
    // greatest value at the two bounds bound the value. Every other function
    // increases with `x`.
    let x = if function == Function::Cosh {
        x.clone().abs()
    } else {
        x.clone()
    };
    let value = decimal_truncation(format.precision, |precision, round| {
        let argument = BigFloat::with_val_round(precision, &x, round).0;
        let value = function.mpfr(&argument, precision, round);
        if !function.is_pi_scaled() {
            return value;
        }
        let other = BigFloat::with_val_round(precision, &x, reverse(round)).0;
        let other = function.mpfr(&other, precision, round);
        match round {
            Round::Down if other < value => other,
            Round::Up if other > value => other,
            _ => value,
        }
    });
    decimal::to_decimal(&value, format, env)
}

/// Returns the decimal result of a special case, or `None` to evaluate.
fn decimal_special(special: Special, env: &Env) -> Option<(DecimalValue, Flags)> {
    let value = match special {
        Special::Evaluate => return None,
        Special::One(negative) => DecimalValue::Finite {
            negative,
            coefficient: Integer::from(1),
            exponent: 0,
        },
        Special::Zero(negative) => DecimalValue::Zero {
            negative,
            exponent: 0,
        },
        Special::Infinity(negative) => DecimalValue::Infinity { negative },
        Special::Pole(negative) => {
            let infinity = DecimalValue::Infinity { negative };
            return Some((infinity, Flags::DIVIDE_BY_ZERO));
        }
        Special::Invalid => {
            let nan = DecimalValue::Nan {
                negative: env.nan.default_negative,
                payload: Integer::ZERO,
            };
            return Some((nan, Flags::INVALID));
        }
    };
    Some((value, Flags::NONE))
}

/// Returns the decimal result at an argument whose value the oracle takes
/// from a bound, where MPFR bounds of a decimal argument would need a
/// working precision near `|x|` or `log2(1 / |x|)` bits. A stand-in value
/// lies in the same unit of `p + 2` digits as the value, on the same side of
/// each point of the grid, so it rounds as the value does.
fn decimal_stand_in(
    function: Function,
    x: &Rational,
    format: DecimalFormat,
    env: &Env,
) -> Option<(DecimalValue, Flags)> {
    let negative = *x < 0;
    let bound = decimal_out_of_range(format);
    let precision = format.precision;
    // `b^x <= 2^x` lies below `2^-(4p + 16)`, below `10^-(p + 2)`, and so
    // does `1 - |tanh x| < 2 e^-2|x|` from `|x| = 2p + 12`. So `b^x - 1` and
    // `tanh x` lie inside ±1 by less than one unit of `p + 2` digits, as
    // `±(1 - 2^-(4p + 20))` does.
    let shifted_negative = function.is_exponential() && function.is_shifted() && negative;
    let near_minus_one = shifted_negative && *x <= -i64::from(4 * precision + 16);
    let near_one = function == Function::Tanh && x.clone().abs() >= 2 * precision + 12;
    if near_minus_one || near_one {
        let tiny = BigFloat::with_val(2, 1) >> (4 * precision + 20);
        let value = BigFloat::with_val(4 * precision + 24, 1u32 - &tiny);
        let value = if negative { -value } else { value };
        return Some(decimal::to_decimal(&value, format, env));
    }
    // `2^(±4R)` lies past the range of the format, as the value does: `b^x`
    // overflows or underflows by the sign of `x`, and `sinh` and `cosh`
    // overflow, `sinh` with the sign of `x`.
    let overflows =
        function.is_exponential() || matches!(function, Function::Sinh | Function::Cosh);
    if x.clone().abs() >= bound && !shifted_negative && overflows {
        let scale = i32::try_from(bound).expect("a decimal bound fits an i32");
        let underflow = negative && function.is_exponential();
        let value = BigFloat::with_val(2, 1) << if underflow { -scale } else { scale };
        let value = if negative && function == Function::Sinh {
            -value
        } else {
            value
        };
        return Some(decimal::to_decimal(&value, format, env));
    }
    // `e^x - 1` lies above `x`, and `ln(1 + x)` below it, by less than
    // `x^2`. `sinh x` and `atanh x` lie beyond `x`, and `tanh x` and
    // `asinh x` short of it, by less than `|x|^3`. Below
    // `|x| = 2^-(4p + 70)`, `x ± |x| 2^-(4p + 70)` lies on the same side of
    // `x` and closer to it than one unit of `p + 2` digits, as the value
    // does, and GMP gives its digits.
    let tiny = Rational::from((1, Integer::from(1) << (4 * precision + 70)));
    if x.clone().abs() >= tiny {
        return None;
    }
    let step = x.clone() * tiny;
    let value = match function {
        Function::ExpM1 => x.clone() + step.abs(),
        Function::LogP1 => x.clone() - step.abs(),
        Function::Sinh | Function::Atanh => x.clone() + step,
        Function::Tanh | Function::Asinh => x.clone() - step,
        _ => return None,
    };
    Some(decimal::rational_to_decimal(&value, format, env))
}

/// Returns a binary value with the truncation to `precision + 2` digits of
/// an irrational value, strictly inside the truncation interval. `evaluate`
/// gives a bound of the value at a working precision in a direction. The
/// working precision doubles until the bounds decide the truncation.
///
/// # Panics
///
/// Panics when the working precision reaches [`PRECISION_LIMIT`].
fn decimal_truncation(precision: u32, evaluate: impl Fn(u32, Round) -> BigFloat) -> BigFloat {
    let mut working = 4 * precision + 64;
    loop {
        let (low, high) = (evaluate(working, Round::Down), evaluate(working, Round::Up));
        if let Some(value) = inside_truncation(&low, &high, precision) {
            return value;
        }
        working *= 2;
        assert!(
            working < PRECISION_LIMIT,
            "the bounds decide the truncation"
        );
    }
}

/// Returns the midpoint of two finite bounds of one sign when every value
/// strictly between them has one truncation to `precision + 2` digits, or
/// `None`. The midpoint has that truncation too.
fn inside_truncation(low: &BigFloat, high: &BigFloat, precision: u32) -> Option<BigFloat> {
    let negative = low.is_sign_negative();
    if negative != high.is_sign_negative() || low.is_zero() || high.is_zero() {
        return None;
    }
    let (small, large) = if negative { (high, low) } else { (low, high) };
    // MPFR writes `|small|` as `0.d * 10^exponent`, so the value is in the
    // decade `[10^top, 10^(top + 1))` of `small` or above it.
    let (_, _, exponent) = small.to_sign_string_exp_round(10, Some(1), Round::Zero);
    let top = i64::from(exponent?) - 1;
    let lowest = top - i64::from(precision) - 1;
    let unit = Integer::from(Integer::u_pow_u(
        10,
        u32::try_from(lowest.unsigned_abs()).ok()?,
    ));
    let scaled = |bound: &BigFloat| {
        let (numerator, denominator) = bound.to_rational()?.abs().into_numer_denom();
        Some(if lowest >= 0 {
            (numerator, denominator * &unit)
        } else {
            (numerator * &unit, denominator)
        })
    };
    let (numerator, denominator) = scaled(small)?;
    let below = numerator.div_rem_floor(denominator).0;
    let (numerator, denominator) = scaled(large)?;
    let above = numerator.div_rem_ceil(denominator).0 - 1u32;
    if below != above {
        return None;
    }
    // The bounds lie in one decade, at most four binades apart.
    let bits = small.prec().max(large.prec()) + 8;
    Some(BigFloat::with_val(bits, small + large) >> 1u32)
}
