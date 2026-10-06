//! IEEE special values and final rounding for sine and cosine.

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
    /// Returns the sine and its exception flags.
    pub fn sin<L: Elementary, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
        Self::trigonometric(value, behavior, false)
    }

    /// Returns the cosine and its exception flags.
    pub fn cos<L: Elementary, B: Behavior>(value: L, behavior: B) -> (L, Flags) {
        Self::trigonometric(value, behavior, true)
    }

    fn trigonometric<L: Elementary, B: Behavior>(
        value: L,
        behavior: B,
        cosine: bool,
    ) -> (L, Flags) {
        let env = behavior.env();
        let mut flags = Flags::NONE;
        let x = Self::operand(value, &env, &mut flags);
        if let Some((nan, special)) = nan::special_unary(&x, &env) {
            return Self::exact(nan, flags | special);
        }
        match x {
            Unpacked::Zero { .. } if !cosine => Self::exact(x, flags),
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
            Unpacked::Infinity { .. } => Self::exact(default_nan(&env), flags | Flags::INVALID),
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
                let target = Self::elementary_target();
                let truncated = if cosine {
                    elementary::cos(&argument, &target)
                } else {
                    elementary::sin(&argument, &target)
                };
                Self::finish(&truncated, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }
}
