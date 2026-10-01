//! `pown` and `rootn` of IEEE 754-2019 section 9.2 for the binary formats.
//!
//! For `|n|` up to [`EXACT_LIMIT`], both round their exact result once.
//! Bounds in the bits of the storage decide most results, and an exact
//! integer power in [`Big`] decides the rest. Past the limit, `pown`
//! multiplies by squaring and rounds each step, and `rootn` gives the default
//! NaN and signals invalid.

mod big;
mod bounds;

use core::cmp::Ordering;

use self::big::Big;
use self::bounds::{Approx, Direction, Interval, estimate_root};
use super::Layout;
use crate::env::{Env, Flags};
use crate::exact::{self, Unrounded};
use crate::format::{Encoding, Storage, Width};
use crate::limbs::{self, Limbs, Widen};
use crate::nan::{self, default_nan};
use crate::unpacked::{Number, Unpacked};

/// The largest `|n|` that `pown` and `rootn` round once. A correctly
/// rounded result needs about `|n| * p` bits in the worst case, so the exact
/// powers stay in a fixed array.
pub const EXACT_LIMIT: u64 = 64;

/// The bound on the exponent of a result past the range of every format.
/// A clamped exponent rounds as the true one does.
const EXPONENT_CLAMP: i128 = 1 << 30;

/// Returns the largest integer that `exceeds` does not reject, from an
/// estimate. `exceeds` must be monotonic, false for zero, and true for some
/// value that fits `D`. The steps double away from the estimate until they
/// pass the result, and then the gap halves, so a poor estimate only costs a
/// few checks.
fn settle<D: Limbs>(estimate: D, exceeds: impl Fn(&D) -> bool) -> D {
    let one = D::ZERO.with_bit(0);
    let mut step = one;
    let (mut low, mut high) = if exceeds(&estimate) {
        let mut high = estimate;
        loop {
            let candidate = if step.compare(&high) == Ordering::Less {
                high.sub(step)
            } else {
                D::ZERO
            };
            if !exceeds(&candidate) {
                break (candidate, high);
            }
            high = candidate;
            step = step.shl(1);
        }
    } else {
        let mut low = estimate;
        loop {
            let candidate = low.add(step);
            if exceeds(&candidate) {
                break (low, candidate);
            }
            low = candidate;
            step = step.shl(1);
        }
    };
    while high.sub(low) != one {
        let middle = low.add(high.sub(low).shr(1));
        if exceeds(&middle) {
            high = middle;
        } else {
            low = middle;
        }
    }
    low
}

/// Returns an exponent of a result, clamped far past the range of every
/// format.
fn clamped(exponent: i128) -> i32 {
    let exponent = exponent.clamp(-EXPONENT_CLAMP, EXPONENT_CLAMP);
    i32::try_from(exponent).expect("the clamp fits an i32")
}

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns `value^n`, and the flags.
    pub fn pown<L: Widen>(value: L, n: i64, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(value, env, &mut flags);
        // `pown(x, 0)` is 1 for every value but a signaling NaN, as IEEE
        // 754-2019 section 9.2.1 requires.
        let special = x.is_signaling() || matches!(x, Unpacked::Unsupported);
        if n == 0 && !special {
            let one = Self::one::<L>();
            return Self::exact(
                Unpacked::Finite {
                    negative: false,
                    exponent: one.exponent,
                    significand: one.significand,
                },
                flags,
            );
        }
        if let Some((result, special)) = nan::special_unary(&x, env) {
            return Self::exact(result, flags | special);
        }
        let odd = n % 2 != 0;
        match x {
            Unpacked::Zero { negative, .. } if n > 0 => {
                Self::exact(Unpacked::zero(negative && odd), flags)
            }
            Unpacked::Zero { negative, .. } => Self::exact(
                Self::infinity(negative && odd, env),
                flags | Flags::DIVIDE_BY_ZERO,
            ),
            Unpacked::Infinity { negative } if n > 0 => Self::exact(
                Unpacked::Infinity {
                    negative: negative && odd,
                },
                flags,
            ),
            Unpacked::Infinity { negative } => Self::exact(Unpacked::zero(negative && odd), flags),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let number = Number {
                    negative,
                    exponent,
                    significand,
                };
                if n.unsigned_abs() <= EXACT_LIMIT {
                    Self::pown_exact(number, n, env, flags)
                } else {
                    Self::pown_steps(number, n, env, flags)
                }
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }

    /// Returns `x^n` of a finite nonzero `x` and `0 < |n| <= 64`, rounded
    /// once. Bounds on `m^|n|` give the top bits of the power, or of the
    /// quotient `2^s / m^|n|` of a negative `n`, when no rounding boundary lies
    /// between them. Otherwise the exact power decides.
    fn pown_exact<L: Widen>(x: Number<L>, n: i64, env: &Env, flags: Flags) -> (L, Flags) {
        let power = u32::try_from(n.unsigned_abs()).expect("the power is at most 64");
        let negative = x.negative && power % 2 == 1;
        let exponent = i64::from(x.exponent) * i64::from(power);
        let precision = Self::PRECISION;
        let bounds = Interval::<L>::of(x.significand, 0).pow(power);
        let value = if n > 0 {
            // A unit one bit below the round bit of the power.
            let unit_weight = bounds.low.top() + 1 - i64::from(precision + 3);
            let unit = Interval::<L>::of(L::ZERO.with_bit(0), unit_weight);
            match bounds.div(&unit).floor_inexact() {
                Some(kept) => Unrounded {
                    negative,
                    exponent: clamped(i128::from(exponent) + i128::from(unit_weight)),
                    significand: kept,
                    sticky: true,
                },
                None => Self::pown_exact_positive(&x, power, negative, exponent),
            }
        } else {
            let width = u32::try_from(bounds.high.top() + 1).expect("a power width fits");
            let shift = width + precision + 2;
            let limit = Interval::<L>::of(L::ZERO.with_bit(0), i64::from(shift));
            match limit.div(&bounds).floor_inexact() {
                Some(quotient) => Unrounded {
                    negative,
                    exponent: clamped(-i128::from(exponent) - i128::from(shift)),
                    significand: quotient,
                    sticky: true,
                },
                None => Self::pown_exact_negative(&x, power, negative, exponent),
            }
        };
        Self::finish(&value, *env, flags)
    }

    /// Returns the exact power `x^n` of a positive `n`, with p + 3 bits and a
    /// sticky bit.
    fn pown_exact_positive<L: Widen>(
        x: &Number<L>,
        power: u32,
        negative: bool,
        exponent: i64,
    ) -> Unrounded<L::Double> {
        let exact = Big::of(&x.significand).pow(power);
        let dropped = exact.bit_length().saturating_sub(Self::PRECISION + 3);
        Unrounded {
            negative,
            exponent: clamped(i128::from(exponent) + i128::from(dropped)),
            significand: exact.shr_to::<L::Double>(dropped),
            sticky: exact.any_below(dropped),
        }
    }

    /// Returns `x^n` of a negative `n` from the exact quotient `2^s / m^|n|`
    /// of p + 3 or p + 4 bits, and a sticky bit.
    fn pown_exact_negative<L: Widen>(
        x: &Number<L>,
        power: u32,
        negative: bool,
        exponent: i64,
    ) -> Unrounded<L::Double> {
        let exact = Big::of(&x.significand).pow(power);
        let width = exact.bit_length();
        let shift = width + Self::PRECISION + 2;
        // The top bits of the divisor, one below the storage, give a
        // quotient at or below the true one: the divisor itself when it
        // fits, and its top bits plus one otherwise.
        let kept = width.min(L::BITS - 1);
        let dropped = width - kept;
        let top = exact.shr_to::<L>(dropped).resize::<L::Double>();
        let divisor = if dropped == 0 { top } else { top.increment() };
        let numerator = L::Double::ZERO.with_bit(shift - dropped);
        let (estimate, _) = limbs::divide(numerator, divisor);
        let target = Big::power_of_two(shift);
        let times = |quotient: &L::Double| Big::of(quotient).mul(&exact);
        let quotient = settle(estimate, |quotient| {
            times(quotient).compare(&target) == Ordering::Greater
        });
        Unrounded {
            negative,
            exponent: clamped(-i128::from(exponent) - i128::from(shift)),
            significand: quotient,
            sticky: times(&quotient).compare(&target) != Ordering::Equal,
        }
    }

    /// Returns `x^n` of a finite nonzero `x` and `|n| > 64`, by squaring
    /// from the top bit of `|n|`. Each product rounds to the precision in the
    /// direction of the behavior, without an exponent limit. The last
    /// product of a positive `n` rounds into the format, and a negative `n`
    /// ends with the reciprocal, rounded into the format.
    fn pown_steps<L: Widen>(x: Number<L>, n: i64, env: &Env, flags: Flags) -> (L, Flags) {
        let power = n.unsigned_abs();
        let width = x.significand.bit_length();
        let top = i32::try_from(width).expect("a width fits an i32") - 1;
        let base = (
            Number {
                exponent: -top,
                ..x
            },
            i128::from(x.exponent) + i128::from(top),
        );
        let mut value = base;
        let mut inexact = false;
        let bits = u64::BITS - power.leading_zeros();
        for bit in (0..bits - 1).rev() {
            let multiply = (power >> bit) & 1 == 1;
            let last = bit == 0 && n > 0;
            let squared = Self::power_step(value, value, !multiply && last, env, &mut inexact);
            value = match squared {
                Step::Scaled(next) => next,
                Step::Final(result) => return Self::with_inexact(result, flags, inexact),
            };
            if multiply {
                match Self::power_step(value, base, last, env, &mut inexact) {
                    Step::Scaled(next) => value = next,
                    Step::Final(result) => return Self::with_inexact(result, flags, inexact),
                }
            }
        }
        let (number, scale) = value;
        let length = number.significand.bit_length();
        let shift = length + Self::PRECISION + 2;
        let numerator = L::Double::ZERO.with_bit(shift);
        let (quotient, remainder) = limbs::divide(numerator, number.significand.resize());
        let reciprocal = Unrounded {
            negative: number.negative,
            exponent: clamped(-i128::from(number.exponent) - scale - i128::from(shift)),
            significand: quotient,
            sticky: !remainder.is_zero(),
        };
        Self::with_inexact(Self::finish(&reciprocal, *env, Flags::NONE), flags, inexact)
    }

    /// Multiplies two scaled values. The product rounds into the format when
    /// `last` is set, and otherwise to the precision in `[1, 2]`, as a value
    /// in `[1, 2)` and a scale.
    fn power_step<L: Widen>(
        (first, first_scale): (Number<L>, i128),
        (second, second_scale): (Number<L>, i128),
        last: bool,
        env: &Env,
        inexact: &mut bool,
    ) -> Step<L> {
        let product = Unrounded::from_term(Self::product(first, second));
        let scale = first_scale + second_scale;
        if last {
            let value = Unrounded {
                exponent: clamped(i128::from(product.exponent) + scale),
                ..product
            };
            return Step::Final(Self::finish(&value, *env, Flags::NONE));
        }
        let top = product.exponent
            + i32::try_from(product.significand.bit_length()).expect("a width fits an i32")
            - 1;
        let scaled = Unrounded {
            exponent: product.exponent - top,
            ..product
        };
        let (rounded, round_flags) = exact::round::<L::Double, L, Self, Env>(&scaled, *env);
        *inexact |= round_flags.contains(Flags::INEXACT);
        let Unpacked::Finite {
            negative,
            exponent,
            significand,
        } = rounded
        else {
            unreachable!("a value in [1, 2] rounds to a finite value");
        };
        let carry = exponent + i32::try_from(significand.bit_length()).expect("a width fits") - 1;
        Step::Scaled((
            Number {
                negative,
                exponent: exponent - carry,
                significand,
            },
            scale + i128::from(top) + i128::from(carry),
        ))
    }

    /// Adds `INEXACT` for a rounded earlier step to a result.
    fn with_inexact<L>((bits, last): (L, Flags), flags: Flags, inexact: bool) -> (L, Flags) {
        let earlier = if inexact { Flags::INEXACT } else { Flags::NONE };
        (bits, flags | last | earlier)
    }

    /// Returns `value^(1/n)`, and the flags.
    pub fn rootn<L: Widen>(value: L, n: i64, env: &Env) -> (L, Flags) {
        let mut flags = Flags::NONE;
        let x = Self::operand(value, env, &mut flags);
        if let Some((result, special)) = nan::special_unary(&x, env) {
            return Self::exact(result, flags | special);
        }
        let invalid = || Self::exact(default_nan(env), flags | Flags::INVALID);
        if n == 0 || n.unsigned_abs() > EXACT_LIMIT {
            return invalid();
        }
        let odd = n % 2 != 0;
        match x {
            Unpacked::Zero { negative, .. } if n > 0 => {
                Self::exact(Unpacked::zero(negative && odd), flags)
            }
            Unpacked::Zero { negative, .. } => Self::exact(
                Self::infinity(negative && odd, env),
                flags | Flags::DIVIDE_BY_ZERO,
            ),
            Unpacked::Infinity { negative: true } | Unpacked::Finite { negative: true, .. }
                if !odd =>
            {
                invalid()
            }
            Unpacked::Infinity { negative } if n > 0 => {
                Self::exact(Unpacked::Infinity { negative }, flags)
            }
            Unpacked::Infinity { negative } => Self::exact(Unpacked::zero(negative), flags),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let number = Number {
                    negative,
                    exponent,
                    significand,
                };
                Self::rootn_exact(number, n, env, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }

    /// Returns `x^(1/n)` of a finite nonzero `x` and `0 < |n| <= 64`,
    /// rounded once.
    ///
    /// With `x = m * 2^(k * j + r)` and `0 <= r < k` for `k = |n|`, the
    /// significand `m * 2^r` has `w` bits. A positive `n` takes the root of
    /// `m * 2^(r + k * t)`, and a negative `n` the largest `y` with
    /// `y^k * m * 2^r <= 2^(k * u)`. `t` and `u` give the root p + 2 to p + 4
    /// bits, and the exact power checks every bit.
    fn rootn_exact<L: Widen>(x: Number<L>, n: i64, env: &Env, flags: Flags) -> (L, Flags) {
        let power = u32::try_from(n.unsigned_abs()).expect("the root is at most 64");
        let k = i64::from(power);
        let (quotient, rest) = (
            i64::from(x.exponent).div_euclid(k),
            i64::from(x.exponent).rem_euclid(k),
        );
        let rest = u32::try_from(rest).expect("the rest is below the root");
        let significand = x.significand.resize::<L::Double>().shl(rest);
        let width = significand.bit_length();
        let precision = Self::PRECISION;
        let big_significand = Big::of(&significand);
        let bounds = Interval::<L>::of(significand, 0);
        // Bounds decide the order of a power and its target, and the exact
        // powers decide the rest.
        let (root, exponent, order) = if n > 0 {
            let scale = (power * (precision + 1) + 1 - width).div_ceil(power);
            let target = Interval::<L>::of(significand, i64::from(power * scale));
            let estimate = estimate_root(power, &target.low, precision + 3);
            let radicand = big_significand.shl(power * scale);
            let order = move |root: &L::Double| {
                Interval::<L>::of(*root, 0)
                    .pow(power)
                    .compare(&target)
                    .unwrap_or_else(|| Big::of(root).pow(power).compare(&radicand))
            };
            let root = settle(estimate, |root| order(root) == Ordering::Greater);
            (root, quotient - i64::from(scale), order(&root))
        } else {
            let scale = precision + 2 + width.div_ceil(power);
            let limit = Interval::<L>::of(L::ZERO.with_bit(0), i64::from(power * scale));
            let target = limit
                .low
                .div(Approx::new(significand, 0, Direction::Up), Direction::Down);
            let estimate = estimate_root(power, &target, precision + 4);
            let exact_limit = Big::power_of_two(power * scale);
            let order = move |root: &L::Double| {
                Interval::<L>::of(*root, 0)
                    .pow(power)
                    .mul(&bounds)
                    .compare(&limit)
                    .unwrap_or_else(|| {
                        Big::of(root)
                            .pow(power)
                            .mul(&big_significand)
                            .compare(&exact_limit)
                    })
            };
            let root = settle(estimate, |root| order(root) == Ordering::Greater);
            (root, -quotient - i64::from(scale), order(&root))
        };
        let sticky = order != Ordering::Equal;
        let value = Unrounded {
            negative: x.negative,
            exponent: clamped(i128::from(exponent)),
            significand: root,
            sticky,
        };
        Self::finish(&value, *env, flags)
    }
}

/// The result of a step of `pown`.
enum Step<L> {
    /// A product in `[1, 2)` and its scale.
    Scaled((Number<L>, i128)),
    /// The result, rounded into the format, and the flags of its rounding.
    Final((L, Flags)),
}

#[cfg(test)]
mod tests {
    use crate::env::{Env, Flags};
    use crate::float::{F64, Float};
    use crate::format::Binary;

    /// A layout whose precision is two bits below its storage.
    type Tight = Float<Binary<2>, 64>;

    const SIGN: u64 = 1 << 63;
    const INFINITY: u64 = 0x7FF0_0000_0000_0000;

    fn pown(bits: u64, n: i64) -> (u64, Flags) {
        let (result, flags) = F64::from_bits(bits).pown_with(n, Env::IEEE);
        (result.to_bits(), flags)
    }

    fn rootn(bits: u64, n: i64) -> (u64, Flags) {
        let (result, flags) = F64::from_bits(bits).rootn_with(n, Env::IEEE);
        (result.to_bits(), flags)
    }

    #[test]
    fn pown_special_cases_follow_ieee() {
        let one = 0x3FF0_0000_0000_0000;
        assert_eq!(pown(0x7FF8_0000_0000_0001, 0), (one, Flags::NONE));
        assert_eq!(pown(INFINITY | SIGN, 0), (one, Flags::NONE));
        assert_eq!(
            pown(0x7FF0_0000_0000_0001, 0),
            (0x7FF8_0000_0000_0001, Flags::INVALID)
        );
        assert_eq!(pown(SIGN, -3), (INFINITY | SIGN, Flags::DIVIDE_BY_ZERO));
        assert_eq!(pown(SIGN, -2), (INFINITY, Flags::DIVIDE_BY_ZERO));
        assert_eq!(pown(INFINITY | SIGN, 3), (INFINITY | SIGN, Flags::NONE));
        assert_eq!(pown(INFINITY | SIGN, -3), (SIGN, Flags::NONE));
    }

    #[test]
    fn pown_rounds_once_up_to_the_limit_and_each_step_past_it() {
        // 3^-40 rounds once.
        assert_eq!(
            pown(0x4008_0000_0000_0000, -40),
            (0x3BF8_46D5_50E3_7B50, Flags::INEXACT)
        );
        // 2^-1074 past the limit: every step is exact, and the reciprocal is
        // the smallest subnormal value.
        assert_eq!(pown(0x4000_0000_0000_0000, -1074), (1, Flags::TINY));
        assert_eq!(
            pown(0x4000_0000_0000_0000, 1024),
            (
                INFINITY,
                Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP
            )
        );
    }

    #[test]
    fn rootn_is_exact_where_it_can_be_and_invalid_past_the_limit() {
        let invalid = (0x7FF8_0000_0000_0000, Flags::INVALID);
        assert_eq!(
            rootn(0xC020_0000_0000_0000, 3),
            (0xC000_0000_0000_0000, Flags::NONE)
        );
        assert_eq!(rootn(0xC020_0000_0000_0000, 2), invalid);
        assert_eq!(rootn(0x4020_0000_0000_0000, 65), invalid);
        assert_eq!(rootn(0x4020_0000_0000_0000, 0), invalid);
        assert_eq!(rootn(SIGN, -3), (INFINITY | SIGN, Flags::DIVIDE_BY_ZERO));
        // The square root of 2^-1074 is 2^-537.
        assert_eq!(rootn(1, 2), (0x1E60_0000_0000_0000, Flags::DENORMAL_INPUT));
        // A precision two bits below the storage takes the longest powers:
        // the cube root of the subnormal 0.5 of that layout.
        let (root, flags) = Tight::from_bits(0x1000_0000_0000_0000).rootn_with(3, Env::IEEE);
        let expected = (
            0x1965_FEA5_3D6E_3C83,
            Flags::INEXACT
                | Flags::DENORMAL_INPUT
                | Flags::TINY
                | Flags::UNDERFLOW
                | Flags::ROUNDED_UP,
        );
        assert_eq!((root.to_bits(), flags), expected);
    }
}
