//! Conversions between the DPD widths against decNumber's `ToWider` and
//! `FromWider`.
//!
//! decNumber converts a NaN without quieting it, and a narrowing keeps the
//! low-order payload digits. IEEE 754 `convertFormat` quiets a signaling
//! NaN and signals invalid, and floaty keeps the high-order payload digits,
//! as the Intel decimal library does. So for a NaN the test takes the sign
//! and the payload digits of decNumber's string of the operand, and applies
//! that rule.
//!
//! A widening copies a non-canonical declet or special value, and floaty
//! gives the canonical encoding of the number. So the expected result is
//! decNumber's encoding of its own string of the result, which is
//! canonical. The test also compares `TINY` and `ROUNDED_UP`, as for a
//! rounded operation, from decNumber's conversion rounded toward zero.
//!
//! Each conversion also runs with flush-to-zero and with the `DefaultNan`
//! rule. Flush-to-zero gives the zero of the least exponent, with `TINY`,
//! `UNDERFLOW`, and `INEXACT`, for a result whose exact value is below
//! `10^emin`, as `flush.rs` states for the operations. `DefaultNan` gives
//! the default NaN for a NaN operand. Each conversion also runs between the
//! BID formats, through the BID and DPD conversions of the Intel library.

use floaty::env::{NanPropagation, NanRule};
use floaty::format::{Bid, Decimal, Dpd, Standard, Storage, Width};
use floaty::{Env, Flags, Float};
use floaty_verify::decnumber::{
    self, Arithmetic, Double, Format, Outcome, Quad, Single, Status, Widening,
};
use floaty_verify::intel_decimal::Format as _;

use super::operands::{Generator, Shape, power_of_ten};
use super::{
    Answer, DpdFloat, Report, SHARED_ROUNDINGS, Tally, Transcode, compared, describe,
    describe_operands, direction, flags_of, to_bid,
};

/// Returns the behaviors of a conversion in a direction: the plain one,
/// flush-to-zero, and the `DefaultNan` rule with a negative default NaN.
fn behaviors(rounding: decnumber::Rounding) -> [Env; 3] {
    let env = Env::IEEE.with_rounding(direction(rounding));
    let default_nan = NanRule::new(NanPropagation::DefaultNan).with_default_negative(true);
    [env, env.with_flush_to_zero(true), env.with_nan(default_nan)]
}

/// The note of a NaN operand, which floaty's payload rule checks.
const NAN_NOTE: &str = "a NaN, with floaty's payload rule";

/// The note of a case where flush-to-zero replaces a tiny result.
const FLUSHED: &str = "ftz: a tiny result becomes a zero";

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

/// Returns the expected result and flags of converting `x` from format `S`
/// to format `T` in a rounding mode under `env`, from `oracle`, which is
/// decNumber's conversion.
fn expected<S: Format, T: Arithmetic>(
    tally: &mut Tally,
    x: S::Bits,
    rounding: decnumber::Rounding,
    env: Env,
    oracle: &impl Fn(S::Bits, decnumber::Rounding) -> Outcome<T::Bits>,
) -> (T::Bits, Flags) {
    let source = S::to_string(x);
    if source.contains("NaN") {
        tally.note(NAN_NOTE);
        let expected = if env.nan.propagation == NanPropagation::DefaultNan {
            let sign = if env.nan.default_negative { "-" } else { "" };
            T::from_string(&format!("{sign}NaN"), decnumber::Rounding::HalfEven).value
        } else {
            converted_nan::<S, T>(&source)
        };
        let signaling = if source.contains("sNaN") {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        return (expected, signaling);
    }
    let outcome = oracle(x, rounding);
    let toward_zero = oracle(x, decnumber::Rounding::Down);
    let class = T::class(toward_zero.value);
    let tiny = class.ends_with("Subnormal")
        || (class.ends_with("Zero") && toward_zero.status.intersects(Status::INEXACT));
    if env.flush_to_zero && tiny {
        tally.note(FLUSHED);
        let negative = source.starts_with('-');
        let (lowest, _) = Generator::<T>::exponents();
        let zero = format!("{}0E{lowest}", if negative { "-" } else { "" });
        let zero = T::from_string(&zero, decnumber::Rounding::HalfEven).value;
        return (zero, Flags::TINY | Flags::UNDERFLOW | Flags::INEXACT);
    }
    let details = Report::Rounded.flags::<T>(
        (outcome.value, outcome.status),
        (toward_zero.value, toward_zero.status),
    );
    (
        canonical::<T>(outcome.value),
        flags_of(outcome.status) | details,
    )
}

/// Converts `x` from format `S` to format `T` with floaty in a rounding
/// mode, under every behavior of [`behaviors`], in the DPD and the BID
/// formats, and compares the result with `oracle`, which is decNumber's
/// conversion.
fn check<S: Transcode, T: Arithmetic + Transcode, const FROM: usize, const TO: usize>(
    tally: &mut Tally,
    x: S::Bits,
    rounding: decnumber::Rounding,
    oracle: impl Fn(S::Bits, decnumber::Rounding) -> Outcome<T::Bits>,
) where
    Width<FROM>: Storage<Bits = S::Bits>,
    Decimal<Dpd>: Standard<FROM, Bits = S::Bits>,
    Decimal<Bid>: Standard<FROM, Bits = S::Bits>,
    Width<TO>: Storage<Bits = T::Bits>,
    Decimal<Dpd>: Standard<TO, Bits = T::Bits>,
    Decimal<Bid>: Standard<TO, Bits = T::Bits>,
{
    for env in behaviors(rounding) {
        let (want, want_flags) = expected::<S, T>(tally, x, rounding, env, &oracle);
        let dpd = DpdFloat::<FROM>::from_bits(x).convert_with::<DpdFloat<TO>>(env);
        let bid = Float::<Decimal<Bid>, FROM>::from_bits(to_bid::<S, FROM>(x))
            .convert_with::<Float<Decimal<Bid>, TO>>(env);
        let results = [
            ("DPD", dpd.0.to_bits(), dpd.1),
            ("BID", T::Bid::to_dpd(bid.0.to_bits()), bid.1),
        ];
        for (encoding, result, flags) in results {
            let flags = compared(flags);
            if result == want && flags == want_flags {
                tally.passed += 1;
                continue;
            }
            tally.fail(|| {
                format!(
                    "{} to {} {rounding:?} {encoding} ftz {} {:?} [{}]: floaty gives {} \
                     {flags:?}, expected {} {want_flags:?}",
                    S::NAME,
                    T::NAME,
                    env.flush_to_zero,
                    env.nan.propagation,
                    describe_operands::<S>(&[x]),
                    describe::<T>(&Answer::Encoding(result)),
                    describe::<T>(&Answer::Encoding(want)),
                )
            });
        }
    }
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
    tally.assert_counts(7_200_000, &[], &[(NAN_NOTE, 255_504)]);
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
    tally.assert_counts(4_800_000, &[], &[(NAN_NOTE, 170_136), (FLUSHED, 165_680)]);
}
