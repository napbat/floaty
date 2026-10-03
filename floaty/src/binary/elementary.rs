//! `exp` and `log` of IEEE 754-2019 section 9.2 for the binary formats.
//!
//! The special cases follow section 9.2.1. The other arguments take the
//! truncation of `crate::elementary`, which the rounding routine rounds once.

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
    fn elementary_target() -> Target {
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
}

/// Returns `true` when `significand * 2^exponent` is 1: a power of two whose
/// exponent cancels.
fn is_one<L: Limbs>(exponent: i32, significand: &L) -> bool {
    let top = significand.bit_length() - 1;
    i64::from(exponent) + i64::from(top) == 0 && !significand.any_below(top)
}
