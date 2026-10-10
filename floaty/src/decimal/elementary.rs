//! The exponentials and the logarithms of IEEE 754-2019 section 9.2 for the
//! decimal formats.
//!
//! The special cases follow section 9.2.1. An exact result takes the
//! exponent nearest 0, as `logB` and a conversion from an integer do: the
//! results of the special cases have the exponent 0, and the rounding
//! routine gives an exact value the member of its cohort with the exponent
//! nearest 0. An inexact result keeps every digit, because the rounding
//! routine gives it the least exponent. The other arguments take the
//! truncation of `crate::elementary`, which the rounding routine rounds
//! once.

use super::DecimalLayout;
use crate::elementary::{self, Elementary, Radix, Special, Target, Transcendental};
use crate::env::{Behavior, Flags};
use crate::format::{DecimalEncoding, Storage, Width};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the format of a result of an elementary function.
    fn elementary_target() -> Target {
        let reach = i64::from(Self::EMAX).max(i64::from(Self::PRECISION) - i64::from(Self::EMIN));
        Target {
            radix: Radix::Decimal,
            precision: Self::PRECISION,
            range: u32::try_from(reach + 3).expect("the exponent range of a format fits a u32"),
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
        let result = match elementary::special(function, x, Radix::Decimal) {
            Special::Evaluate(argument) => {
                let truncated =
                    elementary::evaluate(function, &argument, &Self::elementary_target());
                return Self::finish(&truncated, 0, behavior, flags);
            }
            Special::One { negative } => Unpacked::Finite {
                negative,
                exponent: 0,
                significand: L::ZERO.with_bit(0),
            },
            Special::Zero { negative } => Self::zero(negative, 0),
            Special::Infinity { negative } => Unpacked::Infinity { negative },
            Special::Pole => {
                flags |= Flags::DIVIDE_BY_ZERO;
                Unpacked::Infinity { negative: true }
            }
            Special::Invalid => {
                flags |= Flags::INVALID;
                default_nan(&env)
            }
        };
        Self::exact(result, flags)
    }
}
