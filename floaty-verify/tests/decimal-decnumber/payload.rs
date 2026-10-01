//! The NaN payload operations of IEEE 754-2019 section 9.7 on the decimal
//! formats: `payload`, `from_payload`, and `from_payload_signaling`, on DPD
//! and on BID. decNumber and the Intel decimal library have no payload
//! operations, so the test evaluates the documented rule of floaty with
//! decNumber:
//!
//! - `payload`: decNumber writes a NaN with its payload, such as `NaN123`.
//!   The digits read as an integer with exponent 0, and a NaN without digits
//!   gives 0. Every other value gives -1.
//! - `from_payload` and `from_payload_signaling`: decNumber writes the
//!   operand, and the test reads its value. +0 and the positive integers up
//!   to `10^(p - 1) - 1`, in any cohort, give the NaN that decNumber reads
//!   from `NaN` or `sNaN` and the digits. Every other value gives +0 with
//!   exponent 0.
//!
//! The BID runs convert each operand to BID and the result back to DPD with
//! the Intel library, as [`super::run_floaty_bid`] does.

use floaty::Float;
use floaty::format::{Bid, Decimal, Standard, Storage, Width};
use floaty_verify::decnumber::{Double, Format, Quad, Single};
use intel_decimal::Format as _;

use super::operands::{Generator, Number, Shape, power_of_ten, signed};
use super::{DpdFloat, Tally, Transcode, intel_decimal, to_bid};

/// A NaN payload operation.
#[derive(Clone, Copy, Debug)]
enum Payload {
    Get,
    Set,
    SetSignaling,
}

impl Payload {
    const ALL: [Self; 3] = [Self::Get, Self::Set, Self::SetSignaling];
}

/// Returns the value of a number when it is a nonnegative integer that fits
/// a `u128`.
fn integer(number: Number) -> Option<u128> {
    if number.negative {
        return None;
    }
    if number.coefficient == 0 {
        return Some(0);
    }
    let scale = number.exponent.unsigned_abs();
    if number.exponent >= 0 {
        return 10_u128
            .checked_pow(scale)
            .and_then(|power| power.checked_mul(number.coefficient));
    }
    let power = 10_u128.checked_pow(scale)?;
    number
        .coefficient
        .is_multiple_of(power)
        .then(|| number.coefficient / power)
}

/// Returns the expected result of `operation` on `x`, by the rules of the
/// module documentation.
fn expected<F: Format>(operation: Payload, x: F::Bits) -> F::Bits {
    let text = F::to_string(x);
    let from_text = Generator::<F>::from_text;
    match operation {
        Payload::Get => {
            let unsigned = text.strip_prefix('-').unwrap_or(&text);
            let unsigned = unsigned.strip_prefix('s').unwrap_or(unsigned);
            match unsigned.strip_prefix("NaN") {
                Some("") => from_text("0"),
                Some(digits) => from_text(digits),
                None => from_text("-1"),
            }
        }
        Payload::Set | Payload::SetSignaling => {
            let largest = power_of_ten(F::PRECISION - 1) - 1;
            let payload = Number::parse(&text)
                .and_then(integer)
                .filter(|&payload| payload <= largest);
            let kind = match operation {
                Payload::SetSignaling => "sNaN",
                _ => "NaN",
            };
            match payload {
                Some(0) => from_text(kind),
                Some(payload) => from_text(&format!("{kind}{payload}")),
                None => from_text("0"),
            }
        }
    }
}

/// Runs a payload operation of floaty on a value of `Float<Decimal<E>, W>`.
fn apply<E, const W: usize>(operation: Payload, value: Float<Decimal<E>, W>) -> Float<Decimal<E>, W>
where
    E: floaty::format::DecimalEncoding,
    Width<W>: Storage,
    Decimal<E>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    match operation {
        Payload::Get => value.payload(),
        Payload::Set => Float::<Decimal<E>, W>::from_payload(value),
        Payload::SetSignaling => Float::<Decimal<E>, W>::from_payload_signaling(value),
    }
}

/// Returns the operands at the edges of the payload range: zeros, the
/// integers around `10^(p - 1)` in two cohorts, small integers, and values
/// that are not integers.
fn edges<F: Format>() -> Vec<F::Bits> {
    let largest = power_of_ten(F::PRECISION - 1) - 1;
    let number = |negative, coefficient, exponent| Number {
        negative,
        coefficient,
        exponent,
    };
    let mut numbers = vec![
        number(false, 0, 0),
        number(true, 0, 0),
        number(false, 0, 3),
        number(false, 0, -3),
        number(false, 1, 0),
        number(true, 1, 0),
        number(false, 10, -1),
        number(false, 15, -1),
        number(false, 1, 2),
        number(false, 12, 1),
        number(false, largest, 0),
        number(false, largest * 10, -1),
        number(false, largest + 1, 0),
        number(false, 1, signed(F::PRECISION) - 1),
    ];
    numbers.push(number(true, largest, 0));
    numbers.into_iter().map(Generator::<F>::encode).collect()
}

/// Runs the three operations on the edges and on `count` random operands of
/// format `F`, on DPD and on BID, and compares floaty with decNumber.
fn run<F: Transcode, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<floaty::format::Dpd>: Standard<W, Bits = F::Bits>,
    Decimal<Bid>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let mut operands = edges::<F>();
    operands.extend((0..count).map(|index| {
        if index % 4 == 0 {
            generator.special()
        } else {
            generator.operand().0
        }
    }));
    let mut tally = Tally::default();
    for x in operands {
        for operation in Payload::ALL {
            let want = expected::<F>(operation, x);
            let dpd = apply(operation, DpdFloat::<W>::from_bits(x)).to_bits();
            let bid = F::Bid::to_dpd(
                apply(
                    operation,
                    Float::<Decimal<Bid>, W>::from_bits(to_bid::<F, W>(x)),
                )
                .to_bits(),
            );
            for (encoding, ours) in [("DPD", dpd), ("BID", bid)] {
                if ours == want {
                    tally.passed += 1;
                    continue;
                }
                tally.fail(|| {
                    format!(
                        "{operation:?} {encoding} {}: floaty gives {}, expected {}",
                        F::to_string(x),
                        F::to_string(ours),
                        F::to_string(want),
                    )
                });
            }
        }
    }
    tally
}

#[test]
fn decimal32_payloads() {
    let tally = run::<Single, 32>(20_000, 0x0032_0907);
    tally.report("decimal32 payloads");
    tally.assert_counts(120_090, &[], &[]);
}

#[test]
fn decimal64_payloads() {
    let tally = run::<Double, 64>(20_000, 0x0064_0907);
    tally.report("decimal64 payloads");
    tally.assert_counts(120_090, &[], &[]);
}

#[test]
fn decimal128_payloads() {
    let tally = run::<Quad, 128>(20_000, 0x0128_0907);
    tally.report("decimal128 payloads");
    tally.assert_counts(120_090, &[], &[]);
}
