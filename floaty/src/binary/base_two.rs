//! Base-two elementary functions for binary layouts.

use super::Layout;
use crate::elementary::{self, Argument, Elementary};
use crate::env::{Behavior, Flags};
use crate::format::{Encoding, Storage, Width};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

impl<const E: u32, Enc: Encoding, const W: usize> Layout<E, Enc, W>
where
    Width<W>: Storage,
{
    /// Returns `2^value`, correctly rounded, and the flags.
    pub fn exp2<L: Elementary, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, &env, &mut flags);
        if let Some((nan, special)) = nan::special_unary(&x, &env) {
            return Self::exact(nan, flags | special);
        }
        match x {
            Unpacked::Zero { .. } => {
                let one = Self::one::<L>();
                Self::exact(
                    Unpacked::Finite {
                        negative: false,
                        exponent: one.exponent,
                        significand: one.significand,
                    },
                    flags,
                )
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
                let truncated = elementary::exp2(&argument, &Self::elementary_target());
                Self::finish(&truncated, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }

    /// Returns `log2 value`, correctly rounded, and the flags.
    pub fn log2<L: Elementary, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
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
                let argument = Argument {
                    negative: false,
                    exponent,
                    significand,
                };
                let truncated = elementary::log2(&argument, &Self::elementary_target());
                Self::finish(&truncated, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }
}
