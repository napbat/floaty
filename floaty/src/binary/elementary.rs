//! The exponentials, the logarithms, the trigonometric functions, `atan2`,
//! `pow`, `powr`, and `compound` of IEEE 754-2019 section 9.2 for the binary
//! formats.
//!
//! The special cases follow section 9.2.1. The other arguments take the
//! truncation of `crate::elementary`, which the rounding routine rounds once.

use core::cmp::Ordering;

use super::Layout;
use crate::elementary::{
    self, Argument, Bivariate, Elementary, Outcome, Radix, Special, Target, Transcendental,
    compare_one,
};
use crate::env::{Behavior, Flags};
use crate::format::{Encoding, Storage, Width};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the format of a result of an elementary function.
    fn elementary_target() -> Target {
        let reach = i64::from(Self::EMAX).max(i64::from(Self::PRECISION) - i64::from(Self::EMIN));
        Target {
            radix: Radix::Binary,
            precision: Self::PRECISION,
            range: u32::try_from(reach + 3).expect("the exponent range of a format fits a u32"),
        }
    }

    /// Returns 1 with the sign `negative`.
    fn signed_one<L: Elementary>(negative: bool) -> Unpacked<L> {
        let one = Self::one::<L>();
        Unpacked::Finite {
            negative,
            exponent: one.exponent,
            significand: one.significand,
        }
    }

    /// Returns `function` of `value`, correctly rounded, and the flags.
    pub fn elementary<L: Elementary, B: Behavior>(
        value: L,
        function: Transcendental,
        behavior: B,
    ) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, &env, &mut flags);
        if let Some((nan, special)) = nan::special_unary(&x, &env) {
            return Self::exact(nan, flags | special);
        }
        let target = Self::elementary_target();
        let result = match elementary::special(function, x, Radix::Binary) {
            Special::Evaluate(argument) => {
                let truncated = elementary::evaluate(function, &argument, &target);
                return Self::finish(&truncated, behavior, flags);
            }
            Special::Angle { eighths, scaled } => {
                let truncated = elementary::angle::<L>(eighths, scaled, &target);
                return Self::finish(&truncated, behavior, flags);
            }
            Special::One { negative } => Self::signed_one(negative),
            Special::Zero { negative } => Unpacked::zero(negative),
            Special::Infinity { negative } => Unpacked::Infinity { negative },
            Special::Pole { negative } => {
                flags |= Flags::DIVIDE_BY_ZERO;
                Self::infinity(negative, &env)
            }
            Special::Invalid => {
                flags |= Flags::INVALID;
                default_nan(&env)
            }
        };
        Self::exact(result, flags)
    }

    /// Returns `function` of `left` and `right`, in the operand order of
    /// IEEE 754-2019, correctly rounded, and the flags.
    pub fn bivariate<L: Elementary, B: Behavior>(
        left: L,
        right: L,
        function: Bivariate,
        behavior: B,
    ) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let first = Self::operand(left, &env, &mut flags);
        let second = Self::operand(right, &env, &mut flags);
        if elementary::is_one(function, &first, &second, Radix::Binary) {
            return Self::exact(Self::signed_one(false), flags);
        }
        if let Some((nan, special)) = nan::special(&first, &second, &env) {
            return Self::exact(nan, flags | special);
        }
        let result =
            match elementary::bivariate(function, &first, &second, &Self::elementary_target()) {
                Outcome::Truncated(truncated) => return Self::finish(&truncated, behavior, flags),
                Outcome::Zero { negative } => Unpacked::zero(negative),
                Outcome::Infinity { negative } => Unpacked::Infinity { negative },
                Outcome::Pole { negative } => {
                    flags |= Flags::DIVIDE_BY_ZERO;
                    Self::infinity(negative, &env)
                }
                Outcome::Invalid => {
                    flags |= Flags::INVALID;
                    default_nan(&env)
                }
            };
        Self::exact(result, flags)
    }

    /// Returns `(1 + value)^n`, correctly rounded, and the flags.
    pub fn compound<L: Elementary, B: Behavior>(value: L, n: i64, behavior: B) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, &env, &mut flags);
        let one = || Self::signed_one(false);
        // `compound(x, 0)` is 1 for a quiet NaN too.
        if n == 0 && x.is_nan() && !x.is_signaling() {
            return Self::exact(one(), flags);
        }
        if let Some((nan, special)) = nan::special_unary(&x, &env) {
            return Self::exact(nan, flags | special);
        }
        let below_minus_one = match x {
            Unpacked::Infinity { negative } => negative,
            Unpacked::Finite {
                negative: true,
                exponent,
                significand,
            } => compare_one(&argument(true, exponent, significand), Radix::Binary).is_gt(),
            _ => false,
        };
        if below_minus_one {
            return Self::exact(default_nan(&env), flags | Flags::INVALID);
        }
        if n == 0 {
            return Self::exact(one(), flags);
        }
        match x {
            Unpacked::Zero { .. } => Self::exact(one(), flags),
            Unpacked::Infinity { .. } if n > 0 => Self::exact(x, flags),
            Unpacked::Infinity { .. } => Self::exact(Unpacked::zero(false), flags),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let argument = argument(negative, exponent, significand);
                if negative && compare_one(&argument, Radix::Binary) == Ordering::Equal {
                    return if n > 0 {
                        Self::exact(Unpacked::zero(false), flags)
                    } else {
                        let infinity = Self::infinity(false, &env);
                        Self::exact(infinity, flags | Flags::DIVIDE_BY_ZERO)
                    };
                }
                let truncated = elementary::compound(&argument, n, &Self::elementary_target());
                Self::finish(&truncated, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }
}

/// Returns the argument `significand * 2^exponent` with its sign.
fn argument<L>(negative: bool, exponent: i32, significand: L) -> Argument<L> {
    Argument {
        negative,
        exponent,
        significand,
    }
}
