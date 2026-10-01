//! The reduction operations of IEEE 754-2019 section 9.4 for the binary
//! formats. The sums round their exact result once. The scaled products
//! round each step, without an exponent limit.

mod window;

use self::window::{ExactSum, exact_sum};
use super::Layout;
use super::arithmetic::{Sum, Term, sum};
use crate::env::{Env, Flags};
use crate::exact::{self, Unrounded};
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::unpacked::{Number, Unpacked};

/// What a reduction of one vector sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Summand {
    /// `sum`: the values.
    Value,
    /// `sumAbs`: the magnitudes.
    Magnitude,
    /// `sumSquare`: the squares.
    Square,
}

/// A factor of a scaled product.
#[derive(Clone, Copy)]
pub enum Factor<L> {
    /// A value.
    Value(L),
    /// The sum of two values.
    Sum(L, L),
    /// The difference of two values.
    Difference(L, L),
}

/// The NaN operands of a sum, by precedence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
enum Nans {
    #[default]
    None,
    Quiet,
    Signaling,
}

/// The signs of the infinite terms of a sum.
#[derive(Clone, Copy, Debug, Default)]
struct Infinities {
    positive: bool,
    negative: bool,
}

/// What the operands of a sum add up to before the finite terms.
#[derive(Default)]
struct Scan {
    unsupported: bool,
    nans: Nans,
    infinities: Infinities,
    /// The sum of the zero operands, by the sign rule of an exact zero sum.
    zero_negative: Option<bool>,
    /// An upper bound on the weight of the highest bit of a finite term.
    top: Option<i64>,
}

impl Scan {
    fn nan(&mut self, signaling: bool) {
        let kind = if signaling {
            Nans::Signaling
        } else {
            Nans::Quiet
        };
        self.nans = self.nans.max(kind);
    }

    fn infinity(&mut self, negative: bool) {
        if negative {
            self.infinities.negative = true;
        } else {
            self.infinities.positive = true;
        }
    }

    fn zero(&mut self, negative: bool, env: &Env) {
        self.zero_negative = Some(match self.zero_negative {
            None => negative,
            Some(sum) => env.zero_sum_sign(sum, negative),
        });
    }

    fn top(&mut self, top: i64) {
        self.top = Some(self.top.map_or(top, |highest| highest.max(top)));
    }
}

/// The special factors of a scaled product, by precedence: the last one
/// decides the result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Special {
    None,
    Zero,
    Infinity,
    /// The sum of two infinities of different signs, or a zero factor and an
    /// infinite factor.
    Invalid,
    Nan,
    Unsupported,
}

impl Special {
    /// Returns the special factors of two sets. A zero factor and an
    /// infinite factor are invalid together.
    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Zero, Self::Infinity) | (Self::Infinity, Self::Zero) => Self::Invalid,
            _ => self.max(other),
        }
    }
}

/// A factor of a scaled product, decoded.
enum FactorValue<L> {
    /// A special factor, and the sign of a zero or an infinity.
    Special(Special, bool),
    /// A nonzero finite factor: a number, or the exact sum of two numbers.
    Finite {
        negative: bool,
        first: Number<L>,
        second: Option<Number<L>>,
    },
}

/// The state of a scaled product.
struct Product<L> {
    /// The product of the finite factors, rounded, in `[1, 2)`.
    value: Number<L>,
    /// The scale of `value`, or `None` once it leaves the range of `i64`.
    scale: Option<i64>,
    /// The sign of the product of every factor.
    negative: bool,
    special: Special,
    /// `true` once a step rounds.
    inexact: bool,
}

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the sum of the values, the magnitudes, or the squares of
    /// `values`, rounded once, and the flags.
    pub fn reduce<L: Widen>(
        values: impl Iterator<Item = L> + Clone,
        summand: Summand,
        env: &Env,
    ) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let mut scan = Scan::default();
        let signed = summand == Summand::Value;
        for bits in values.clone() {
            match Self::operand(bits, env, &mut flags) {
                Unpacked::Unsupported => scan.unsupported = true,
                Unpacked::Nan { signaling, .. } => scan.nan(signaling),
                Unpacked::Infinity { negative } => scan.infinity(negative && signed),
                Unpacked::Zero { negative, .. } => scan.zero(negative && signed, env),
                Unpacked::Finite {
                    exponent,
                    significand,
                    ..
                } => {
                    let top = i64::from(exponent) + i64::from(significand.bit_length());
                    scan.top(if summand == Summand::Square {
                        2 * top - 1
                    } else {
                        top - 1
                    });
                }
            }
        }
        if scan.unsupported {
            return Self::exact(default_nan(env), flags | Flags::INVALID);
        }
        // `sumAbs` and `sumSquare` give +inf for an infinity before a quiet
        // NaN, as `hypot` does.
        let infinity_first = !signed && scan.nans != Nans::Signaling && scan.infinities.positive;
        if scan.nans != Nans::None && !infinity_first {
            let (value, special) = nan::select_all(values.map(Self::decode), env);
            return Self::exact(value, flags | special);
        }
        if let Some(result) = Self::infinite_sum(&scan, env, flags) {
            return result;
        }
        let terms = || {
            values.clone().filter_map(move |bits| {
                let number = Self::finite_operand(bits, env)?;
                Some(match summand {
                    Summand::Value => {
                        Self::term(number.negative, number.exponent, number.significand)
                    }
                    Summand::Magnitude => Self::term(false, number.exponent, number.significand),
                    Summand::Square => Self::product(number, number),
                })
            })
        };
        Self::round_sum(terms, &scan, env, flags)
    }

    /// Returns the sum of the products of `pairs`, rounded once, and the
    /// flags.
    pub fn dot<L: Widen>(pairs: impl Iterator<Item = (L, L)> + Clone, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let mut scan = Scan::default();
        let mut invalid_product = false;
        for (left, right) in pairs.clone() {
            let x = Self::operand(left, env, &mut flags);
            let y = Self::operand(right, env, &mut flags);
            let negative = x.is_negative() != y.is_negative();
            match (x, y) {
                (Unpacked::Unsupported, _) | (_, Unpacked::Unsupported) => scan.unsupported = true,
                (Unpacked::Nan { .. }, _) | (_, Unpacked::Nan { .. }) => {
                    scan.nan(x.is_signaling() || y.is_signaling());
                }
                (Unpacked::Infinity { .. }, Unpacked::Zero { .. })
                | (Unpacked::Zero { .. }, Unpacked::Infinity { .. }) => invalid_product = true,
                (Unpacked::Infinity { .. }, _) | (_, Unpacked::Infinity { .. }) => {
                    scan.infinity(negative);
                }
                (Unpacked::Zero { .. }, _) | (_, Unpacked::Zero { .. }) => scan.zero(negative, env),
                (
                    Unpacked::Finite {
                        exponent: a,
                        significand: x_significand,
                        ..
                    },
                    Unpacked::Finite {
                        exponent: b,
                        significand: y_significand,
                        ..
                    },
                ) => {
                    let width = x_significand.bit_length() + y_significand.bit_length();
                    scan.top(i64::from(a) + i64::from(b) + i64::from(width) - 1);
                }
            }
        }
        if scan.unsupported {
            return Self::exact(default_nan(env), flags | Flags::INVALID);
        }
        // A NaN operand gives a NaN. It also signals invalid for a product
        // `0 * inf`, as every other operation with that product does.
        if scan.nans != Nans::None {
            let operands =
                pairs.flat_map(|(left, right)| [Self::decode(left), Self::decode(right)]);
            let (value, special) = nan::select_all(operands, env);
            let invalid = if invalid_product {
                Flags::INVALID
            } else {
                Flags::NONE
            };
            return Self::exact(value, flags | special | invalid);
        }
        if invalid_product {
            return Self::exact(default_nan(env), flags | Flags::INVALID);
        }
        if let Some(result) = Self::infinite_sum(&scan, env, flags) {
            return result;
        }
        let terms = || {
            pairs.clone().filter_map(move |(left, right)| {
                let x = Self::finite_operand(left, env)?;
                let y = Self::finite_operand(right, env)?;
                Some(Self::product(x, y))
            })
        };
        Self::round_sum(terms, &scan, env, flags)
    }

    /// Returns the sum of infinite terms, or `None` without one. Infinities
    /// of both signs give the default NaN and signal invalid.
    fn infinite_sum<L: Limbs>(scan: &Scan, env: &Env, flags: Flags) -> Option<(L, Flags)> {
        match (scan.infinities.positive, scan.infinities.negative) {
            (true, true) => Some(Self::exact(default_nan(env), flags | Flags::INVALID)),
            (true, false) => Some(Self::exact(Unpacked::Infinity { negative: false }, flags)),
            (false, true) => Some(Self::exact(Unpacked::Infinity { negative: true }, flags)),
            (false, false) => None,
        }
    }

    /// Rounds the exact sum of the finite terms, or encodes its zero.
    fn round_sum<D: Limbs, I: Iterator<Item = Term<D>>, L: Limbs>(
        terms: impl Fn() -> I,
        scan: &Scan,
        env: &Env,
        flags: Flags,
    ) -> (L, Flags) {
        let exact = match scan.top {
            Some(top) => exact_sum(terms, top, Self::PRECISION),
            None => ExactSum::Zero,
        };
        match exact {
            ExactSum::Value(value) => Self::finish(&value, *env, flags),
            ExactSum::Zero => {
                // Finite terms that cancel give the zero of an exact zero
                // sum. Zero operands alone give the sum of their zeros, and
                // no operand gives +0.
                let negative = if scan.top.is_some() {
                    env.zero_sum_is_negative()
                } else {
                    scan.zero_negative.unwrap_or(false)
                };
                Self::exact(Unpacked::zero(negative), flags)
            }
        }
    }

    /// Returns the product of `factors`, rounded at each step, as a value in
    /// `[1, 2)` and a scale, and the flags. `operands` gives the operands in
    /// order, for the NaN rule.
    pub fn scaled_product<L: Widen>(
        factors: impl Iterator<Item = Factor<L>>,
        operands: impl Iterator<Item = L>,
        env: &Env,
    ) -> (L, i64, Flags) {
        let mut flags = Flags::NONE;
        let mut state = Product {
            value: Self::one(),
            scale: Some(0),
            negative: false,
            special: Special::None,
            inexact: false,
        };
        for factor in factors {
            match Self::factor(factor, env, &mut flags) {
                FactorValue::Special(special, negative) => {
                    state.special = state.special.and(special);
                    state.negative ^= negative;
                }
                FactorValue::Finite {
                    negative,
                    first,
                    second,
                } => {
                    state.negative ^= negative;
                    if state.special == Special::None && state.scale.is_some() {
                        Self::step(&mut state, first, second, env);
                    }
                }
            }
        }
        let special =
            |value: Unpacked<L>, special: Flags| (Self::encode(value), 0, flags | special);
        let scale = match (state.special, state.scale) {
            (Special::Unsupported | Special::Invalid, _) | (Special::None, None) => {
                return special(default_nan(env), Flags::INVALID);
            }
            (Special::Nan, _) => {
                let (value, nan_flags) = nan::select_all(operands.map(Self::decode), env);
                return special(value, nan_flags);
            }
            (Special::Infinity, _) => {
                return special(Self::infinity(state.negative, env), Flags::NONE);
            }
            (Special::Zero, _) => return special(Unpacked::zero(state.negative), Flags::NONE),
            (Special::None, Some(scale)) => scale,
        };
        let value = Unpacked::Finite {
            negative: state.value.negative,
            exponent: state.value.exponent,
            significand: state.value.significand,
        };
        // Only a finite product reports the rounding of its steps.
        let inexact = if state.inexact {
            Flags::INEXACT
        } else {
            Flags::NONE
        };
        (Self::encode(value), scale, flags | inexact)
    }

    /// Decodes a factor of a scaled product. A sum of two finite numbers is
    /// exact.
    fn factor<L: Widen>(factor: Factor<L>, env: &Env, flags: &mut Flags) -> FactorValue<L> {
        let (x, y) = match factor {
            Factor::Value(bits) => (Self::operand(bits, env, flags), None),
            Factor::Sum(left, right) => (
                Self::operand(left, env, flags),
                Some(Self::operand(right, env, flags)),
            ),
            Factor::Difference(left, right) => (
                Self::operand(left, env, flags),
                Some(Self::operand(right, env, flags).negate()),
            ),
        };
        let Some(y) = y else {
            return Self::single_factor(x);
        };
        match (x, y) {
            (Unpacked::Unsupported, _) | (_, Unpacked::Unsupported) => {
                FactorValue::Special(Special::Unsupported, false)
            }
            (Unpacked::Nan { .. }, _) | (_, Unpacked::Nan { .. }) => {
                FactorValue::Special(Special::Nan, false)
            }
            (Unpacked::Infinity { negative: a }, Unpacked::Infinity { negative: b }) if a != b => {
                FactorValue::Special(Special::Invalid, false)
            }
            (Unpacked::Infinity { negative }, _) | (_, Unpacked::Infinity { negative }) => {
                FactorValue::Special(Special::Infinity, negative)
            }
            (Unpacked::Zero { negative: a, .. }, Unpacked::Zero { negative: b, .. }) => {
                FactorValue::Special(Special::Zero, env.zero_sum_sign(a, b))
            }
            (Unpacked::Zero { .. }, number) | (number, Unpacked::Zero { .. }) => {
                Self::single_factor(number)
            }
            (Unpacked::Finite { .. }, Unpacked::Finite { .. }) => {
                let (Some(first), Some(second)) = (x.number(), y.number()) else {
                    unreachable!("both operands are finite");
                };
                let term = |number: Number<L>| {
                    Self::term(number.negative, number.exponent, number.significand)
                };
                match sum(term(first), term(second)) {
                    Sum::Zero => FactorValue::Special(Special::Zero, env.zero_sum_is_negative()),
                    Sum::Value(value) => FactorValue::Finite {
                        negative: value.negative,
                        first,
                        second: Some(second),
                    },
                }
            }
        }
    }

    /// Decodes a factor of one operand.
    fn single_factor<L: Limbs>(value: Unpacked<L>) -> FactorValue<L> {
        match value {
            Unpacked::Unsupported => FactorValue::Special(Special::Unsupported, false),
            Unpacked::Nan { .. } => FactorValue::Special(Special::Nan, false),
            Unpacked::Infinity { negative } => FactorValue::Special(Special::Infinity, negative),
            Unpacked::Zero { negative, .. } => FactorValue::Special(Special::Zero, negative),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => FactorValue::Finite {
                negative,
                first: Number {
                    negative,
                    exponent,
                    significand,
                },
                second: None,
            },
        }
    }

    /// Multiplies the product by a finite factor, `first` or `first + second`,
    /// and rounds the result in `[1, 2]` to the precision. The scale takes the
    /// exponent of the result.
    fn step<L: Widen>(
        state: &mut Product<L>,
        first: Number<L>,
        second: Option<Number<L>>,
        env: &Env,
    ) {
        let value = state.value;
        let exact = match second {
            None => Unrounded::from_term(Self::product(value, first)),
            Some(second) => match sum(Self::product(value, first), Self::product(value, second)) {
                Sum::Value(exact) => exact,
                Sum::Zero => unreachable!("a nonzero factor gives a nonzero product"),
            },
        };
        let top = exact.exponent
            + i32::try_from(exact.significand.bit_length()).expect("a width fits an i32")
            - 1;
        let scaled = Unrounded {
            exponent: exact.exponent - top,
            ..exact
        };
        let (rounded, round_flags) = exact::round::<L::Double, L, Self, Env>(&scaled, *env);
        if round_flags.contains(Flags::INEXACT) {
            state.inexact = true;
        }
        let Unpacked::Finite {
            negative,
            exponent,
            significand,
        } = rounded
        else {
            unreachable!("a value in [1, 2] rounds to a finite value");
        };
        // A value that rounds up to 2 moves to 1.
        let carry = exponent + i32::try_from(significand.bit_length()).expect("a width fits") - 1;
        state.value = Number {
            negative,
            exponent: exponent - carry,
            significand,
        };
        state.scale = state
            .scale
            .and_then(|scale| scale.checked_add(i64::from(top) + i64::from(carry)));
    }

    /// Returns an operand as a nonzero finite number, after DAZ, or `None`.
    /// The first pass over the operands reports their flags.
    fn finite_operand<L: Limbs>(bits: L, env: &Env) -> Option<Number<L>> {
        let mut flags = Flags::NONE;
        Self::operand(bits, env, &mut flags).number()
    }

    /// Returns 1 as a number.
    pub(super) fn one<L: Limbs>() -> Number<L> {
        Number {
            negative: false,
            exponent: -Self::SHIFT,
            significand: L::ZERO.with_bit(Self::PRECISION - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags, Rounding};
    use crate::float::{F64, F128};

    const ONE: u64 = 0x3FF0_0000_0000_0000;
    const SIGN: u64 = 1 << 63;

    fn sum(values: &[u64], env: Env) -> (u64, Flags) {
        let (sum, flags) = F64::sum_with(values.iter().map(|&bits| F64::from_bits(bits)), env);
        (sum.to_bits(), flags)
    }

    #[test]
    fn sums_round_the_exact_sum_once() {
        // 2^1000 + 1 - 2^1000 and 2^1000 + 2^-1000 - 2^1000.
        let large = 0x7E70_0000_0000_0000;
        assert_eq!(
            sum(&[large, ONE, large | SIGN], Env::IEEE),
            (ONE, Flags::NONE)
        );
        let small = 0x0170_0000_0000_0000;
        assert_eq!(
            sum(&[large, small, large | SIGN], Env::IEEE),
            (small, Flags::NONE)
        );
        // 1 + 2^-53 is a tie, which the smallest subnormal value decides.
        let half = 0x3CA0_0000_0000_0000;
        let inexact = Flags::INEXACT | Flags::DENORMAL_INPUT;
        let up = Flags::ROUNDED_UP | inexact;
        assert_eq!(sum(&[ONE, half, 1], Env::IEEE), (ONE + 1, up));
        assert_eq!(sum(&[ONE, half, 1 | SIGN], Env::IEEE), (ONE, inexact));
        assert_eq!(sum(&[ONE, half], Env::IEEE), (ONE, Flags::INEXACT));
    }

    #[test]
    fn far_terms_cancel_across_many_windows() {
        // 2^16000 + 2^-16000 - 2^16000 in binary128.
        let value = |field: u128| F128::from_bits(field << 112);
        let terms = [value(0x7E7F), value(0x017F), value(0xFE7F)];
        let (sum, flags) = F128::sum_with(terms, Env::IEEE);
        assert_eq!((sum.to_bits(), flags), (0x017F << 112, Flags::NONE));
    }

    #[test]
    fn zero_sums_take_the_sign_rule_of_addition() {
        let toward_negative = Env::IEEE.with_rounding(Rounding::TowardNegative);
        let negative_zero = SIGN;
        assert_eq!(sum(&[], toward_negative).0, 0);
        assert_eq!(
            sum(&[negative_zero, negative_zero], Env::IEEE).0,
            negative_zero
        );
        assert_eq!(sum(&[0, negative_zero], Env::IEEE).0, 0);
        assert_eq!(sum(&[0, negative_zero], toward_negative).0, negative_zero);
        assert_eq!(sum(&[ONE, ONE | SIGN], Env::IEEE).0, 0);
        assert_eq!(sum(&[ONE, ONE | SIGN], toward_negative).0, negative_zero);
    }

    #[test]
    fn scaled_products_keep_a_value_in_one_to_two() {
        let value = |bits: u64| F64::from_bits(bits);
        let product = |scaled: crate::Scaled<F64>| (scaled.value.to_bits(), scaled.scale);
        assert_eq!(product(F64::scaled_product([])), (ONE, 0));
        // 3 * 3 is 1.125 * 2^3, and 0.5 * 0.5 is 1 * 2^-2.
        let three = value(0x4008_0000_0000_0000);
        assert_eq!(
            product(F64::scaled_product([three, three])),
            (0x3FF2_0000_0000_0000, 3)
        );
        let half = value(0x3FE0_0000_0000_0000);
        assert_eq!(product(F64::scaled_product([half, half])), (ONE, -2));
        // (1 + 2)(-3 + 1) is -6, and (1 - 1) is +0.
        let (one, two) = (value(ONE), value(0x4000_0000_0000_0000));
        let pairs = [(one, two), (-three, one)];
        assert_eq!(
            product(F64::scaled_product_sum(pairs)),
            (0xBFF8_0000_0000_0000, 2)
        );
        assert_eq!(
            product(F64::scaled_product_difference([(one, one)])),
            (0, 0)
        );
    }
}
