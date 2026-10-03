//! `exp` and `log` of IEEE 754-2019 section 9.2 for the decimal formats.
//!
//! The special cases follow section 9.2.1. The exact results, `e^0 = 1` and
//! `ln 1 = 0`, take the exponent 0, as `logB` and a conversion from an
//! integer do. Every other result is inexact and keeps every digit, because
//! the rounding routine gives an inexact result the least exponent. The
//! other arguments take the truncation of `crate::elementary`, which the
//! rounding routine rounds once.

use super::DecimalLayout;
use crate::elementary::{self, Argument, Elementary, Radix, Target};
use crate::env::{Behavior, Flags};
use crate::format::{DecimalEncoding, Storage, Width};
use crate::limbs::{self, Limbs};
use crate::nan::{self, default_nan};
use crate::unpacked::Unpacked;

impl<Enc: DecimalEncoding, const W: usize> DecimalLayout<Enc, W>
where
    Width<W>: Storage,
{
    /// Returns the format of a result of `exp` or `log`.
    fn elementary_target() -> Target {
        let reach = i64::from(Self::EMAX).max(i64::from(Self::PRECISION) - i64::from(Self::EMIN));
        Target {
            radix: Radix::Decimal,
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
                let one = Unpacked::Finite {
                    negative: false,
                    exponent: 0,
                    significand: L::ZERO.with_bit(0),
                };
                Self::exact(one, flags)
            }
            Unpacked::Infinity { negative: false } => Self::exact(x, flags),
            Unpacked::Infinity { negative: true } => Self::exact(Self::zero(false, 0), flags),
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
                Self::finish(&truncated, 0, behavior, flags)
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
            Unpacked::Zero { .. } => Self::exact(
                Unpacked::Infinity { negative: true },
                flags | Flags::DIVIDE_BY_ZERO,
            ),
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
                    return Self::exact(Self::zero(false, 0), flags);
                }
                let argument = Argument {
                    negative: false,
                    exponent,
                    significand,
                };
                let truncated = elementary::log(&argument, &Self::elementary_target());
                Self::finish(&truncated, 0, behavior, flags)
            }
            Unpacked::Nan { .. } | Unpacked::Unsupported => {
                unreachable!("the special cases handle every NaN and unsupported operand")
            }
        }
    }
}

/// Returns `true` when `coefficient * 10^exponent` is 1: the coefficient is
/// `10^-exponent`. A coefficient has at most 34 digits.
fn is_one<L: Limbs>(exponent: i32, coefficient: &L) -> bool {
    let Ok(digits) = u32::try_from(-i64::from(exponent)) else {
        return false;
    };
    if digits > 34 {
        return false;
    }
    let power = (0..digits).fold([1_u64, 0], |power, _| limbs::multiply_small(power, 10));
    coefficient.resize::<[u64; 2]>() == power
}
