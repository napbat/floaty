//! Conversions between the DPD formats and integers against decNumber, in
//! the eight rounding modes of decNumber. The integer types are `i128`,
//! `u128`, and `Int` and `UInt` of a narrow and a wide width. The Intel
//! decimal library checks both conversions for BID in its five directions,
//! and for the integer types of at most 64 bits only.
//!
//! - `to_int_with` rounds as decNumber's `ToIntegralExact` does in the mode.
//!   The integral value gives `ToInt::Value` when it fits the integer type,
//!   and `ToInt::OutOfRange` with the sign of the operand otherwise, as an
//!   infinity does. A NaN gives `ToInt::Nan`. Both of those signal invalid
//!   and not inexact. A value that fits takes `INEXACT` from decNumber, and
//!   `ROUNDED_UP` when the magnitude grew.
//! - `from_int_with` rounds the decimal string of a random integer with
//!   decNumber's `FromString` in the mode.

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Decoded, Env, Flags, Int, ToInt, UInt};
use floaty_verify::decnumber::{self, Arithmetic, Double, Quad, Single, Unary};
use floaty_verify::dectest::Operation;
use floaty_verify::operations::integral::{IntegerValue, to_int_value};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;
use rug::ops::RemRounding;

use super::operands::{Generator, Shape};
use super::{
    DpdFloat, Report, SHARED_ROUNDINGS, Tally, compared, describe_operands, direction, flags_of,
};

/// Returns the expected result and flags of `to_int_with::<I>`, from
/// decNumber's integral value of `x` in the mode `rounding`.
fn expected_to_int<F: Arithmetic, const W: usize, I: IntegerValue>(
    x: F::Bits,
    rounding: decnumber::Rounding,
) -> (ToInt<Integer>, Flags)
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
            // A nonzero coefficient times 10^shift is at least 2^shift, which
            // is beyond the range of an integer type of at most shift bits.
            if shift >= I::width() {
                return (ToInt::OutOfRange { negative }, Flags::INVALID);
            }
            let magnitude = Integer::from_digits(&significand, Order::Lsf)
                * Integer::from(Integer::u_pow_u(10, shift));
            if negative { -magnitude } else { magnitude }
        }
        other => panic!("{other:?} is not the integral value of a number"),
    };
    let (lowest, highest) = I::range();
    if value < lowest || value > highest {
        return (ToInt::OutOfRange { negative }, Flags::INVALID);
    }
    let flags = flags_of(integral.status)
        | Report::Magnitude.flags::<F>(
            (integral.value, integral.status),
            (toward_zero.value, toward_zero.status),
        );
    (ToInt::Value(value), flags)
}

/// Returns operands of format `F` at the range edges of the integer type
/// `I`: each bound minus one, the bound, and the bound plus one, each with a
/// fraction of one half, rounded to the format toward zero and away from
/// zero. A bound with more digits than the format then gives a value just
/// inside the range and a value just outside it.
fn range_edges<F: Arithmetic, I: IntegerValue>() -> Vec<F::Bits> {
    let (lowest, highest) = I::range();
    let texts: Vec<String> = [lowest, highest]
        .iter()
        .flat_map(|bound| {
            [-1, 0, 1].map(|step| {
                let integer = Integer::from(bound + step);
                // The sign comes first, so that -0.5 keeps it.
                let sign = if integer < 0 { "-" } else { "" };
                format!("{sign}{}.5", integer.abs())
            })
        })
        .collect();
    texts
        .iter()
        .flat_map(|text| {
            [decnumber::Rounding::Down, decnumber::Rounding::Up]
                .map(|rounding| F::from_string(text, rounding).value)
        })
        .collect()
}

/// Converts random operands of format `F`, and the operands of
/// [`range_edges`], to the integer type `I` in every rounding mode of
/// decNumber, and compares floaty with decNumber.
fn run_to_int<F: Arithmetic, const W: usize, I: IntegerValue>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let random = (0..count).map(|_| {
        let [x] = generator.operands(&Operation::Unary(Unary::ToIntegralExact))[..] else {
            panic!("rounding to an integral value takes one operand");
        };
        x
    });
    let operands: Vec<F::Bits> = random.chain(range_edges::<F, I>()).collect();
    let mut tally = Tally::default();
    for x in operands {
        for rounding in SHARED_ROUNDINGS {
            let env = Env::IEEE.with_rounding(direction(rounding));
            let (result, flags) = DpdFloat::<W>::from_bits(x).to_int_with::<I>(env);
            let (result, flags) = (to_int_value(result), compared(flags));
            let (expected, expected_flags) = expected_to_int::<F, W, I>(x, rounding);
            if (&result, flags) == (&expected, expected_flags) {
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

/// Returns a random integer of the type `I`: of up to 64 bits, of up to the
/// width of `I`, a tie at `precision` digits, or an exact value of at most
/// `precision` digits. A value outside the range of `I` wraps into it.
fn integer<I: IntegerValue>(random: &mut SplitMix64, precision: u32) -> Integer {
    let bits = |random: &mut SplitMix64, count: u32| {
        let digits: Vec<u64> = (0..count.div_ceil(64)).map(|_| random.next_u64()).collect();
        Integer::from_digits(&digits, Order::Lsf).keep_bits(count)
    };
    let width = I::width();
    let magnitude = match random.below(4) {
        0 => bits(random, width.min(64)),
        1 => bits(random, width),
        2 => {
            // `precision` digits and then the digit 5, scaled by a power of
            // ten: a tie of every nearest mode.
            let lowest = Integer::from(Integer::u_pow_u(10, precision - 1));
            let coefficient = bits(random, 128) % (Integer::from(9) * &lowest) + &lowest;
            let scale = u32::try_from(random.below(3)).expect("the scale is below 3");
            (coefficient * 10u32 + 5u32) * Integer::from(Integer::u_pow_u(10, scale))
        }
        _ => bits(random, 128) % Integer::from(Integer::u_pow_u(10, precision)),
    };
    let value = if I::signed() && random.coin_flip() {
        -magnitude
    } else {
        magnitude
    };
    let (lowest, highest) = I::range();
    let span = Integer::from(&highest - &lowest) + 1u32;
    (value - &lowest).rem_euc(span) + lowest
}

/// Converts random integers of the type `I` to format `F` in every rounding
/// mode of decNumber, and compares floaty with decNumber.
fn run_from_int<F: Arithmetic, const W: usize, I: IntegerValue>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut random = SplitMix64::new(seed);
    let mut tally = Tally::default();
    for _ in 0..count {
        let value = integer::<I>(&mut random, F::PRECISION);
        let text = value.to_string();
        for rounding in SHARED_ROUNDINGS {
            let env = Env::IEEE.with_rounding(direction(rounding));
            let (result, flags) = DpdFloat::<W>::from_int_with(I::from_integer(&value), env);
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
    let to_int = run_to_int::<Single, 32, i128>(20_000, 0x0032_1170);
    to_int.report("decimal32 to_int");
    to_int.assert_counts(160_096, &[], &[]);
    let from_int = run_from_int::<Single, 32, i128>(20_000, 0x0032_F170);
    from_int.report("decimal32 from_int");
    from_int.assert_counts(160_000, &[], &[]);
}

#[test]
fn decimal64_integer_conversions() {
    let to_int = run_to_int::<Double, 64, i128>(20_000, 0x0064_1170);
    to_int.report("decimal64 to_int");
    to_int.assert_counts(160_096, &[], &[]);
    let from_int = run_from_int::<Double, 64, i128>(20_000, 0x0064_F170);
    from_int.report("decimal64 from_int");
    from_int.assert_counts(160_000, &[], &[]);
}

#[test]
fn decimal128_integer_conversions() {
    let to_int = run_to_int::<Quad, 128, i128>(20_000, 0x0128_1170);
    to_int.report("decimal128 to_int");
    to_int.assert_counts(160_096, &[], &[]);
    let from_int = run_from_int::<Quad, 128, i128>(20_000, 0x0128_F170);
    from_int.report("decimal128 from_int");
    from_int.assert_counts(160_000, &[], &[]);
}

/// Converts the format `$format` of `$width` bits to and from `u128`, and
/// `Int` and `UInt` of a narrow and a wide width.
macro_rules! other_integer_types {
    ($format:ty, $width:literal, $seed:literal) => {{
        // Each conversion to an integer also runs the twelve range edges.
        let to_int = [
            ("u128", run_to_int::<$format, $width, u128>(2_000, $seed)),
            (
                "Int<20>",
                run_to_int::<$format, $width, Int<20>>(2_000, $seed + 1),
            ),
            (
                "UInt<20>",
                run_to_int::<$format, $width, UInt<20>>(2_000, $seed + 2),
            ),
            (
                "Int<200>",
                run_to_int::<$format, $width, Int<200>>(2_000, $seed + 3),
            ),
            (
                "UInt<512>",
                run_to_int::<$format, $width, UInt<512>>(2_000, $seed + 4),
            ),
        ];
        for (name, tally) in to_int {
            tally.report(&format!("decimal{} to {name}", $width));
            tally.assert_counts(16_096, &[], &[]);
        }
        let from_int = [
            (
                "u128",
                run_from_int::<$format, $width, u128>(2_000, $seed + 5),
            ),
            (
                "Int<20>",
                run_from_int::<$format, $width, Int<20>>(2_000, $seed + 6),
            ),
            (
                "UInt<20>",
                run_from_int::<$format, $width, UInt<20>>(2_000, $seed + 7),
            ),
            (
                "Int<200>",
                run_from_int::<$format, $width, Int<200>>(2_000, $seed + 8),
            ),
            (
                "UInt<512>",
                run_from_int::<$format, $width, UInt<512>>(2_000, $seed + 9),
            ),
        ];
        for (name, tally) in from_int {
            tally.report(&format!("decimal{} from {name}", $width));
            tally.assert_counts(16_000, &[], &[]);
        }
    }};
}

#[test]
fn conversions_with_the_other_integer_types() {
    other_integer_types!(Single, 32, 0x0032_0170);
    other_integer_types!(Double, 64, 0x0064_0170);
    other_integer_types!(Quad, 128, 0x0128_0170);
}
