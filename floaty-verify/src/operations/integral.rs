//! The oracle of rounding to an integral value, and of conversion to and
//! from an integer.

use floaty::format::{Storage, Width};
use floaty::{Env, Flags, Int, Rounding, ToInt, UInt};
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

use super::{Outcome, Read, number, read, special_operands, zero};
use crate::arithmetic::Operand;
use crate::mpfr::{self, Format, Input};

/// Rounds a nonzero finite value to an integral value in a rounding
/// direction. A zero result has the sign of the value, as IEEE 754-2019
/// section 6.3 requires.
fn integral(value: &BigFloat, rounding: Rounding) -> BigFloat {
    // The integral value has at most one bit more than the value.
    let precision = value.prec() + 2;
    let result = match rounding {
        Rounding::NearestEven => BigFloat::with_val(precision, value.round_even_ref()),
        Rounding::NearestAway => BigFloat::with_val(precision, value.round_ref()),
        Rounding::TowardPositive => BigFloat::with_val(precision, value.ceil_ref()),
        Rounding::TowardNegative => BigFloat::with_val(precision, value.floor_ref()),
        Rounding::TowardZero => BigFloat::with_val(precision, value.trunc_ref()),
        Rounding::ToOdd => {
            // Round to odd keeps an integral value, and otherwise takes the
            // odd one of the two nearest integers.
            let floor = BigFloat::with_val(precision, value.floor_ref());
            if floor == *value {
                floor
            } else if (floor.clone() >> 1u32).is_integer() {
                BigFloat::with_val(precision, value.ceil_ref())
            } else {
                floor
            }
        }
        _ => panic!("the oracle knows every rounding direction"),
    };
    if result.is_zero() {
        zero(value.is_sign_negative())
    } else {
        result
    }
}

/// Returns `INEXACT` when the integral value differs from the value, and
/// `ROUNDED_UP` when its magnitude is larger.
fn integral_flags(value: &BigFloat, integral: &BigFloat) -> Flags {
    if integral == value {
        Flags::NONE
    } else if *integral.as_abs() > *value.as_abs() {
        Flags::INEXACT | Flags::ROUNDED_UP
    } else {
        Flags::INEXACT
    }
}

/// Returns the expected result and flags of `round_to_integral_with`.
///
/// IEEE 754-2019 section 5.9 `roundToIntegralExact`: a zero or an infinity
/// does not change, and a result that differs from the operand signals
/// inexact. `DESIGN.md` adds `ROUNDED_UP`, and the precision limit and
/// flush-to-zero do not apply. An integer above the largest finite value of
/// the format overflows as the rounding oracle gives it.
///
/// # Panics
///
/// Panics when MPFR gives no integer for a finite value, which does not
/// happen.
#[must_use]
pub fn round_to_integral<const N: usize>(
    operand: &Operand<N>,
    format: &Format,
    env: &Env,
) -> (Outcome, Flags) {
    let mut flags = Flags::NONE;
    let operands = [read(operand, env, &mut flags)];
    if let Some((nan, special)) = special_operands(&operands, format, env) {
        return (nan, flags | special);
    }
    let [Read::Number(value)] = &operands else {
        unreachable!("every special operand has a result");
    };
    if !value.is_normal() {
        return (number(value, format), flags);
    }
    let result = integral(value, env.rounding);
    let flags = flags | integral_flags(value, &result);
    if result.is_zero() {
        return (number(&result, format), flags);
    }
    // The integer is exact at the full precision; the rounding oracle only
    // applies the exponent range.
    let full_precision = Env::IEEE
        .with_rounding(env.rounding)
        .with_tininess(env.tininess)
        .with_nan(env.nan)
        .with_saturate(env.saturate);
    let integer = result
        .to_integer()
        .expect("an integral value is an integer");
    let input = Input {
        negative: integer < 0,
        exponent: 0,
        significand: integer.abs(),
        sticky: false,
    };
    let (rounded, overflow_flags) = mpfr::round(&input, format, &full_precision);
    (Outcome::from_value(rounded), flags | overflow_flags)
}

/// Returns the expected result and flags of `to_int_with` into the integer
/// type `I`.
///
/// IEEE 754-2019 section 5.8 `convertToIntegerExact`: the value rounds in
/// the direction of the behavior. `DESIGN.md` gives `Nan` for a NaN or an
/// unsupported encoding, and `OutOfRange` with the sign of the float for an
/// infinity or a rounded value outside the range of `I`. Both signal invalid
/// and not inexact. A value that fits signals `INEXACT` and `ROUNDED_UP` as
/// `round_to_integral` does.
///
/// # Panics
///
/// Panics when MPFR gives no integer for a finite value, which does not
/// happen.
#[must_use]
pub fn to_int<I: IntegerValue, const N: usize>(
    operand: &Operand<N>,
    env: &Env,
) -> (ToInt<Integer>, Flags) {
    let mut flags = Flags::NONE;
    let value = match read(operand, env, &mut flags) {
        Read::Number(value) => value,
        Read::Nan { .. } | Read::Unsupported => return (ToInt::Nan, flags | Flags::INVALID),
    };
    let out_of_range = ToInt::OutOfRange {
        negative: value.is_sign_negative(),
    };
    if value.is_infinite() {
        return (out_of_range, flags | Flags::INVALID);
    }
    if value.is_zero() {
        return (ToInt::Value(Integer::ZERO), flags);
    }
    let result = integral(&value, env.rounding);
    let (low, high) = I::range();
    if result < low || result > high {
        return (out_of_range, flags | Flags::INVALID);
    }
    let integer = result.to_integer().expect("an integral value is finite");
    (
        ToInt::Value(integer),
        flags | integral_flags(&value, &result),
    )
}

/// Returns the expected result and flags of `from_int_with`: the rounding
/// oracle rounds the integer, and a zero converts to `+0`.
#[must_use]
pub fn from_int(value: &Integer, format: &Format, env: &Env) -> (Outcome, Flags) {
    let input = Input {
        negative: *value < 0,
        exponent: 0,
        significand: value.clone().abs(),
        sticky: false,
    };
    let (result, flags) = mpfr::round(&input, format, env);
    (Outcome::from_value(result), flags)
}

/// An integer type whose value converts to and from a `rug` integer.
pub trait IntegerValue: floaty::Integer {
    /// Returns the width in bits, from the definition of the type.
    fn width() -> u32;

    /// Returns `true` for a signed type, from the definition of the type.
    fn signed() -> bool;

    /// Returns the smallest and the largest value.
    ///
    /// # Panics
    ///
    /// Panics when floaty reports another width or signedness for the type
    /// than its definition gives.
    #[must_use]
    fn range() -> (Integer, Integer) {
        assert_eq!(
            (Self::width(), Self::signed()),
            (
                <Self as floaty::Integer>::BITS,
                <Self as floaty::Integer>::SIGNED
            ),
            "floaty reports the width and signedness of the definition"
        );
        let width = Self::width();
        if Self::signed() {
            let half = Integer::from(1) << (width - 1);
            (-half.clone(), half - 1u32)
        } else {
            (Integer::ZERO, (Integer::from(1) << width) - 1u32)
        }
    }

    /// Returns the value.
    fn to_integer(self) -> Integer;

    /// Makes a value. The value must fit the type.
    fn from_integer(value: &Integer) -> Self;
}

/// Returns a conversion result with its value as a `rug` integer.
pub fn to_int_value<I: IntegerValue>(result: ToInt<I>) -> ToInt<Integer> {
    match result {
        ToInt::Value(value) => ToInt::Value(value.to_integer()),
        ToInt::OutOfRange { negative } => ToInt::OutOfRange { negative },
        ToInt::Nan => ToInt::Nan,
    }
}

/// Implements [`IntegerValue`] for Rust primitive integers.
macro_rules! primitive {
    ($($integer:ty),*) => {
        $(
            impl IntegerValue for $integer {
                fn width() -> u32 {
                    <$integer>::BITS
                }

                fn signed() -> bool {
                    <$integer>::MIN != 0
                }

                fn to_integer(self) -> Integer {
                    Integer::from(self)
                }

                fn from_integer(value: &Integer) -> Self {
                    value.to_i128()
                        .and_then(|value| Self::try_from(value).ok())
                        .or_else(|| value.to_u128().and_then(|value| Self::try_from(value).ok()))
                        .expect("the value fits the type")
                }
            }
        )*
    };
}

primitive!(i8, i16, i32, i64, i128, u8, u16, u32, u64, u128);

/// A storage type of the width table, read as an unsigned `rug` integer.
pub trait StorageBits: Copy {
    /// Returns the bits as an unsigned integer.
    fn to_unsigned(self) -> Integer;

    /// Makes the bits from an unsigned integer that fits the storage.
    fn from_unsigned(value: &Integer) -> Self;
}

/// Implements [`StorageBits`] for the primitive storage types.
macro_rules! primitive_storage {
    ($($bits:ty),*) => {
        $(
            impl StorageBits for $bits {
                fn to_unsigned(self) -> Integer {
                    Integer::from(self)
                }

                fn from_unsigned(value: &Integer) -> Self {
                    value
                        .to_u128()
                        .and_then(|value| Self::try_from(value).ok())
                        .expect("the value fits the storage")
                }
            }
        )*
    };
}

primitive_storage!(u8, u16, u32, u64, u128);

impl<const N: usize> StorageBits for [u64; N] {
    fn to_unsigned(self) -> Integer {
        Integer::from_digits(&self, Order::Lsf)
    }

    fn from_unsigned(value: &Integer) -> Self {
        let digits = value.to_digits::<u64>(Order::Lsf);
        assert!(digits.len() <= N, "the value fits the storage");
        let mut limbs = [0; N];
        limbs[..digits.len()].copy_from_slice(&digits);
        limbs
    }
}

impl<const BITS: usize> IntegerValue for UInt<BITS>
where
    Width<BITS>: Storage,
    <Width<BITS> as Storage>::Bits: StorageBits,
{
    fn width() -> u32 {
        u32::try_from(BITS).expect("a width of the storage table fits a u32")
    }

    fn signed() -> bool {
        false
    }

    fn to_integer(self) -> Integer {
        self.to_bits().to_unsigned()
    }

    fn from_integer(value: &Integer) -> Self {
        assert!(
            *value >= 0 && value.significant_bits() <= <Self as floaty::Integer>::BITS,
            "the value fits the type"
        );
        Self::from_bits(StorageBits::from_unsigned(value))
    }
}

impl<const BITS: usize> IntegerValue for Int<BITS>
where
    Width<BITS>: Storage,
    <Width<BITS> as Storage>::Bits: StorageBits,
{
    fn width() -> u32 {
        u32::try_from(BITS).expect("a width of the storage table fits a u32")
    }

    fn signed() -> bool {
        true
    }

    fn to_integer(self) -> Integer {
        let width = <Self as floaty::Integer>::BITS;
        let bits = self.to_bits().to_unsigned();
        if bits.get_bit(width - 1) {
            bits - (Integer::from(1) << width)
        } else {
            bits
        }
    }

    fn from_integer(value: &Integer) -> Self {
        let width = <Self as floaty::Integer>::BITS;
        let half = Integer::from(1) << (width - 1);
        assert!(
            *value >= -half.clone() && *value < half,
            "the value fits the type"
        );
        let bits = if *value < 0 {
            value.clone() + (Integer::from(1) << width)
        } else {
            value.clone()
        };
        Self::from_bits(StorageBits::from_unsigned(&bits))
    }
}
