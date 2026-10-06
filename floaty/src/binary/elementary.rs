//! `exp`, `log`, and `compound` of IEEE 754-2019 section 9.2 for the binary
//! formats.
//!
//! The special cases follow section 9.2.1. The other arguments take the
//! truncation of `crate::elementary`, which the rounding routine rounds once.

use core::cmp::Ordering;

use super::Layout;
use crate::elementary::{self, Argument, Elementary, Radix, Target};
use crate::env::{Behavior, Flags};
use crate::format::{Encoding, Storage, Width};
use crate::limbs::Limbs;
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the format of a result of `exp` or `log`.
    pub(super) fn elementary_target() -> Target {
        let reach = i64::from(Self::EMAX).max(i64::from(Self::PRECISION) - i64::from(Self::EMIN));
        Target {
            radix: Radix::Binary,
            precision: Self::PRECISION,
            range: u32::try_from(reach + 3).expect("the exponent range of a format fits a u32"),
        }
    }

    /// Returns `e^value`, correctly rounded, and the flags.
    pub fn exp<L: Elementary, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, &env, &mut flags);
        if let Some((nan, special)) = nan::special_unary(&x, &env) {
            return Self::exact(nan, flags | special);
        }
        match x {
            Unpacked::Zero { .. } => {
                let one = Self::one::<L>();
                let one = Unpacked::Finite {
                    negative: false,
                    exponent: one.exponent,
                    significand: one.significand,
                };
                Self::exact(one, flags)
            }
            Unpacked::Infinity { negative: false } => Self::exact(x, flags),
            Unpacked::Infinity { negative: true } => Self::exact(Unpacked::zero(false), flags),
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let argument = Argument {
                    negative,
                    exponent,
                    significand,
                };
                let truncated = elementary::exp(&argument, &Self::elementary_target());
                Self::finish(&truncated, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }

    /// Returns `ln value`, correctly rounded, and the flags.
    pub fn log<L: Elementary, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, &env, &mut flags);
        if let Some((nan, special)) = nan::special_unary(&x, &env) {
            return Self::exact(nan, flags | special);
        }
        match x {
            Unpacked::Zero { .. } => {
                Self::exact(Self::infinity(true, &env), flags | Flags::DIVIDE_BY_ZERO)
            }
            Unpacked::Infinity { negative: false } => Self::exact(x, flags),
            Unpacked::Infinity { negative: true } | Unpacked::Finite { negative: true, .. } => {
                Self::exact(default_nan(&env), flags | Flags::INVALID)
            }
            Unpacked::Finite {
                exponent,
                significand,
                ..
            } => {
                if is_one(exponent, &significand) {
                    return Self::exact(Unpacked::zero(false), flags);
                }
                let argument = Argument {
                    negative: false,
                    exponent,
                    significand,
                };
                let truncated = elementary::log(&argument, &Self::elementary_target());
                Self::finish(&truncated, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }

    /// Returns `(1 + value)^n`, correctly rounded, and the flags.
    pub fn compound<L: Elementary, B: Behavior>(value: L, n: i64, behavior: B) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, &env, &mut flags);
        let one = || {
            let one = Self::one::<L>();
            Unpacked::Finite {
                negative: false,
                exponent: one.exponent,
                significand: one.significand,
            }
        };
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
            } => compare_one(exponent, &significand) == Ordering::Greater,
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
                negative: true,
                exponent,
                significand,
            } if compare_one(exponent, &significand) == Ordering::Equal => {
                if n > 0 {
                    Self::exact(Unpacked::zero(false), flags)
                } else {
                    Self::exact(Self::infinity(false, &env), flags | Flags::DIVIDE_BY_ZERO)
                }
            }
            Unpacked::Finite {
                negative,
                exponent,
                significand,
            } => {
                let argument = Argument {
                    negative,
                    exponent,
                    significand,
                };
                let truncated = elementary::compound(&argument, n, &Self::elementary_target());
                Self::finish(&truncated, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }
}

/// Returns `true` when `significand * 2^exponent` is 1: a power of two whose
/// exponent cancels.
fn is_one<L: Limbs>(exponent: i32, significand: &L) -> bool {
    compare_one(exponent, significand) == Ordering::Equal
}

/// Compares the magnitude `significand * 2^exponent`, which is not zero,
/// with 1.
fn compare_one<L: Limbs>(exponent: i32, significand: &L) -> Ordering {
    let top = significand.bit_length() - 1;
    let leading = i64::from(exponent) + i64::from(top);
    match leading.cmp(&0) {
        Ordering::Equal if significand.any_below(top) => Ordering::Greater,
        order => order,
    }
}
