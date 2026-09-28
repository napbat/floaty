//! Conversions between the DPD widths against decNumber's `ToWider` and
//! `FromWider`.
//!
//! decNumber converts a NaN without quieting it, and a narrowing keeps the
//! low-order payload digits. IEEE 754 `convertFormat` quiets a signaling
//! NaN and signals invalid, and floaty keeps the high-order payload digits,
//! as `DESIGN.md` records under "NaN Payloads in Conversions". So for a NaN
//! the test takes the sign and the payload digits of decNumber's string of
//! the operand, and applies that rule.
//!
//! A widening copies a non-canonical declet or special value, and floaty
//! gives the canonical encoding of the number. So the expected result is
//! decNumber's encoding of its own string of the result, which is
//! canonical. The test also compares `TINY` and `ROUNDED_UP`, as for a
//! rounded operation, from decNumber's conversion rounded toward zero.

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Env, Flags};
use floaty_verify::decnumber::{self, Arithmetic, Double, Format, Outcome, Quad, Single, Widening};

use super::operands::{Generator, Shape, power_of_ten};
use super::{
    Answer, DpdFloat, Report, SHARED_ROUNDINGS, Tally, compared, describe, describe_operands,
    direction, flags_of,
};

/// The note of a NaN operand, which the payload rule of `DESIGN.md`
/// checks.
const NAN_NOTE: &str = "a NaN, with the payload rule of DESIGN.md";

/// Returns the NaN that a conversion from format `S` to format `T` gives
/// for a NaN operand, from decNumber's string of the operand. The result is
/// quiet. A widening multiplies the payload by a power of 10, and a
/// narrowing keeps its high-order digits.
fn converted_nan<S: Format, T: Format>(text: &str) -> T::Bits {
    let (sign, rest) = match text.strip_prefix('-') {
        Some(rest) => ("-", rest),
        None => ("", text),
    };
    let digits = rest.trim_start_matches('s').trim_start_matches("NaN");
    let payload: u128 = if digits.is_empty() {
        0
    } else {
        digits
            .parse()
            .expect("decNumber writes the payload in digits")
    };
    let payload = if T::PRECISION >= S::PRECISION {
        payload * power_of_ten(T::PRECISION - S::PRECISION)
    } else {
        payload / power_of_ten(S::PRECISION - T::PRECISION)
    };
    let text = if payload == 0 {
        format!("{sign}NaN")
    } else {
        format!("{sign}NaN{payload}")
    };
    T::from_string(&text, decnumber::Rounding::HalfEven).value
}

/// Returns decNumber's encoding of its own string of an encoding: the
/// canonical encoding of the same number.
fn canonical<F: Format>(bits: F::Bits) -> F::Bits {
    let outcome = F::from_string(&F::to_string(bits), decnumber::Rounding::HalfEven);
    assert!(outcome.status.is_empty(), "an encoding converts exactly");
    outcome.value
}

/// Converts `x` from format `S` to format `T` with floaty in a rounding
/// mode, and compares the result with `oracle`, which is decNumber's
/// conversion.
fn check<S: Format, T: Arithmetic, const FROM: usize, const TO: usize>(
    tally: &mut Tally,
    x: S::Bits,
    rounding: decnumber::Rounding,
    oracle: impl Fn(S::Bits, decnumber::Rounding) -> Outcome<T::Bits>,
) where
    Width<FROM>: Storage<Bits = S::Bits>,
    Decimal<Dpd>: Standard<FROM, Bits = S::Bits>,
    Width<TO>: Storage<Bits = T::Bits>,
    Decimal<Dpd>: Standard<TO, Bits = T::Bits>,
{
    let direction = direction(rounding).expect("a shared mode has a direction");
    let (result, flags) = DpdFloat::<FROM>::from_bits(x)
        .convert_with::<DpdFloat<TO>>(Env::IEEE.with_rounding(direction));
    let flags = compared(flags);
    let source = S::to_string(x);
    let (expected, expected_flags) = if source.contains("NaN") {
        tally.note(NAN_NOTE);
        let expected = converted_nan::<S, T>(&source);
        let signaling = if source.contains("sNaN") {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        (expected, signaling)
    } else {
        let outcome = oracle(x, rounding);
        let toward_zero = oracle(x, decnumber::Rounding::Down);
        let details = Report::Rounded.flags::<T>(
            (outcome.value, outcome.status),
            (toward_zero.value, toward_zero.status),
        );
        (
            canonical::<T>(outcome.value),
            flags_of(outcome.status) | details,
        )
    };
    if result.to_bits() == expected && flags == expected_flags {
        tally.passed += 1;
        return;
    }
    tally.fail(|| {
        format!(
            "{} to {} {rounding:?} [{}]: floaty gives {} {flags:?}, expected {} \
             {expected_flags:?}",
            S::NAME,
            T::NAME,
            describe_operands::<S>(&[x]),
            describe::<T>(&Answer::Encoding(result.to_bits())),
            describe::<T>(&Answer::Encoding(expected)),
        )
    });
}

/// Returns `count` random operands of format `F`, biased toward the edges of
/// a shape.
fn operands<F: Format>(count: usize, seed: u64, shape: Shape) -> Vec<F::Bits> {
    let mut generator = Generator::<F>::new(seed, shape);
    (0..count).map(|_| generator.operand().0).collect()
}

/// Returns the shape of wide operands at the edges of the narrow format `N`:
/// the coefficient length of `N::Wider`, and the exponent range of `N`.
fn narrowing_shape<N: Widening>() -> Shape {
    Shape {
        digits: <N::Wider as Format>::PRECISION,
        ..Shape::of::<N>()
    }
}

#[test]
fn widening_conversions() {
    let mut tally = Tally::default();
    for x in operands::<Single>(50_000, 0x0032_0001, Shape::of::<Single>()) {
        for rounding in SHARED_ROUNDINGS {
            let to_double = |x, _| Outcome {
                value: Single::to_wider(x),
                status: decnumber::Status::NONE,
            };
            check::<Single, Double, 32, 64>(&mut tally, x, rounding, to_double);
            let to_quad = |x, _| Outcome {
                value: Double::to_wider(Single::to_wider(x)),
                status: decnumber::Status::NONE,
            };
            check::<Single, Quad, 32, 128>(&mut tally, x, rounding, to_quad);
        }
    }
    for x in operands::<Double>(50_000, 0x0064_0001, Shape::of::<Double>()) {
        for rounding in SHARED_ROUNDINGS {
            let to_quad = |x, _| Outcome {
                value: Double::to_wider(x),
                status: decnumber::Status::NONE,
            };
            check::<Double, Quad, 64, 128>(&mut tally, x, rounding, to_quad);
        }
    }
    tally.report("widening conversions");
    tally.assert_counts(900_000, &[], &[(NAN_NOTE, 63_876)]);
}

#[test]
fn narrowing_conversions() {
    let mut tally = Tally::default();
    for x in operands::<Double>(50_000, 0x0064_0002, narrowing_shape::<Single>()) {
        for rounding in SHARED_ROUNDINGS {
            check::<Double, Single, 64, 32>(&mut tally, x, rounding, Single::from_wider);
        }
    }
    for x in operands::<Quad>(50_000, 0x0128_0004, narrowing_shape::<Double>()) {
        for rounding in SHARED_ROUNDINGS {
            check::<Quad, Double, 128, 64>(&mut tally, x, rounding, Double::from_wider);
        }
    }
    tally.report("narrowing conversions");
    tally.assert_counts(600_000, &[], &[(NAN_NOTE, 42_534)]);
}
