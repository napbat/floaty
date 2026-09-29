//! Conversions between the DPD formats and integers against decNumber, in
//! the eight rounding modes of decNumber. The Intel decimal library checks
//! both conversions for BID in its five directions only.
//!
//! - `to_int_with::<i128>` rounds as decNumber's `ToIntegralExact` does in
//!   the mode. The integral value gives `ToInt::Value` when it fits an
//!   `i128`, and `ToInt::OutOfRange` with the sign of the operand otherwise,
//!   as an infinity does. A NaN gives `ToInt::Nan`. Both of those signal
//!   invalid and not inexact. A value that fits takes `INEXACT` from
//!   decNumber, and `ROUNDED_UP` when the magnitude grew.
//! - `from_int_with` rounds the decimal string of a random `i128` with
//!   decNumber's `FromString` in the mode.

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Decoded, Env, Flags, ToInt};
use floaty_verify::decnumber::{self, Arithmetic, Double, Quad, Single, Unary};
use floaty_verify::dectest::Operation;
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

use super::operands::{Generator, Shape};
use super::{
    DpdFloat, Report, SHARED_ROUNDINGS, Tally, compared, describe_operands, direction, flags_of,
};

/// Returns the expected result and flags of `to_int_with::<i128>`, from
/// decNumber's integral value of `x` in the mode `rounding`.
fn expected_to_int<F: Arithmetic, const W: usize>(
    x: F::Bits,
    rounding: decnumber::Rounding,
) -> (ToInt<i128>, Flags)
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let class = F::class(x);
    let negative = class.starts_with('-');
    if class.ends_with("NaN") {
        return (ToInt::Nan, Flags::INVALID);
    }
    if class.ends_with("Infinity") {
        return (ToInt::OutOfRange { negative }, Flags::INVALID);
    }
    let integral = F::unary(Unary::ToIntegralExact, x, rounding);
    let toward_zero = F::unary(Unary::ToIntegralExact, x, decnumber::Rounding::Down);
    // floaty's `decode`, which the decTest `apply` vectors check, reads the
    // integral value that decNumber chose.
    let value = match DpdFloat::<W>::from_bits(integral.value).decode::<2>() {
        Decoded::Zero { .. } => Integer::ZERO,
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let shift = u32::try_from(exponent).expect("an integral value has no fraction digits");
            // A nonzero coefficient times 10^39 is beyond the range of i128.
            if shift >= 39 {
                return (ToInt::OutOfRange { negative }, Flags::INVALID);
            }
            let magnitude = Integer::from_digits(&significand, Order::Lsf)
                * Integer::from(Integer::u_pow_u(10, shift));
            if negative { -magnitude } else { magnitude }
        }
        other => panic!("{other:?} is not the integral value of a number"),
    };
    let Some(value) = value.to_i128() else {
        return (ToInt::OutOfRange { negative }, Flags::INVALID);
    };
    let flags = flags_of(integral.status)
        | Report::Magnitude.flags::<F>(
            (integral.value, integral.status),
            (toward_zero.value, toward_zero.status),
        );
    (ToInt::Value(value), flags)
}

/// Converts random operands of format `F` to `i128` in every rounding mode
/// of decNumber, and compares floaty with decNumber.
fn run_to_int<F: Arithmetic, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let mut tally = Tally::default();
    for _ in 0..count {
        let [x] = generator.operands(&Operation::Unary(Unary::ToIntegralExact))[..] else {
            panic!("rounding to an integral value takes one operand");
        };
        for rounding in SHARED_ROUNDINGS {
            let env = Env::IEEE.with_rounding(direction(rounding));
            let (result, flags) = DpdFloat::<W>::from_bits(x).to_int_with::<i128>(env);
            let flags = compared(flags);
            let (expected, expected_flags) = expected_to_int::<F, W>(x, rounding);
            if (result, flags) == (expected, expected_flags) {
                tally.passed += 1;
                continue;
            }
            tally.fail(|| {
                format!(
                    "to_int {rounding:?} [{}]: floaty gives {result:?} {flags:?}, decNumber \
                     {expected:?} {expected_flags:?}",
                    describe_operands::<F>(&[x]),
                )
            });
        }
    }
    tally
}

/// Returns a random integer: of 64 bits, of up to 127 bits, a tie at
/// `precision` digits, or an exact value of at most `precision` digits.
fn integer(random: &mut SplitMix64, precision: u32) -> i128 {
    let wide = |random: &mut SplitMix64| {
        u128::from(random.next_u64()) << 64 | u128::from(random.next_u64())
    };
    let magnitude = match random.next_u64() % 4 {
        0 => u128::from(random.next_u64()),
        1 => wide(random) >> 1,
        2 => {
            // `precision` digits and then the digit 5, scaled by a power of
            // ten: a tie of every nearest mode.
            let lowest = 10_u128.pow(precision - 1);
            let coefficient = lowest + wide(random) % (9 * lowest);
            let scale = u32::try_from(random.next_u64() % 3).expect("the scale is below 3");
            (coefficient * 10 + 5) * 10_u128.pow(scale)
        }
        _ => wide(random) % 10_u128.pow(precision),
    };
    let magnitude = i128::try_from(magnitude).expect("every magnitude is below 2^127");
    if random.next_u64() & 1 == 1 {
        -magnitude
    } else {
        magnitude
    }
}

/// Converts random integers to format `F` in every rounding mode of
/// decNumber, and compares floaty with decNumber.
fn run_from_int<F: Arithmetic, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut random = SplitMix64::new(seed);
    let mut tally = Tally::default();
    for _ in 0..count {
        let value = integer(&mut random, F::PRECISION);
        let text = value.to_string();
        for rounding in SHARED_ROUNDINGS {
            let env = Env::IEEE.with_rounding(direction(rounding));
            let (result, flags) = DpdFloat::<W>::from_int_with(value, env);
            let flags = compared(flags);
            let outcome = F::from_string(&text, rounding);
            let toward_zero = F::from_string(&text, decnumber::Rounding::Down);
            let expected_flags = flags_of(outcome.status)
                | Report::Rounded.flags::<F>(
                    (outcome.value, outcome.status),
                    (toward_zero.value, toward_zero.status),
                );
            if result.to_bits() == outcome.value && flags == expected_flags {
                tally.passed += 1;
                continue;
            }
            tally.fail(|| {
                format!(
                    "from_int {rounding:?} {value}: floaty gives {:#x} {flags:?}, decNumber \
                     {:#x} {expected_flags:?}",
                    result.to_bits(),
                    outcome.value,
                )
            });
        }
    }
    tally
}

#[test]
fn decimal32_integer_conversions() {
    let to_int = run_to_int::<Single, 32>(20_000, 0x0032_1170);
    to_int.report("decimal32 to_int");
    to_int.assert_counts(160_000, &[], &[]);
    let from_int = run_from_int::<Single, 32>(20_000, 0x0032_F170);
    from_int.report("decimal32 from_int");
    from_int.assert_counts(160_000, &[], &[]);
}

#[test]
fn decimal64_integer_conversions() {
    let to_int = run_to_int::<Double, 64>(20_000, 0x0064_1170);
    to_int.report("decimal64 to_int");
    to_int.assert_counts(160_000, &[], &[]);
    let from_int = run_from_int::<Double, 64>(20_000, 0x0064_F170);
    from_int.report("decimal64 from_int");
    from_int.assert_counts(160_000, &[], &[]);
}

#[test]
fn decimal128_integer_conversions() {
    let to_int = run_to_int::<Quad, 128>(20_000, 0x0128_1170);
    to_int.report("decimal128 to_int");
    to_int.assert_counts(160_000, &[], &[]);
    let from_int = run_from_int::<Quad, 128>(20_000, 0x0128_F170);
    from_int.report("decimal128 from_int");
    from_int.assert_counts(160_000, &[], &[]);
}
