//! The minimum and maximum operations of IEEE 754-2019 section 9.6 on the
//! DPD formats: `minimum`, `maximum`, `minimum_number`, and
//! `maximum_number`. decNumber and the Intel decimal library have only the
//! operations of IEEE 754-2008, `min_num` and `max_num`, so the test
//! evaluates the documented rule of floaty with decNumber:
//!
//! - Two numbers: decNumber's `max` or `min`, which breaks a tie by the
//!   total order, as floaty does, and gives the canonical encoding.
//! - `minimum` and `maximum` with a NaN operand: the NaN that decNumber's
//!   addition gives by the NaN rule of `Env::IEEE`, with its conditions.
//! - `minimum_number` and `maximum_number` with one NaN operand: decNumber's
//!   canonical encoding of the number, and invalid for a signaling NaN. With
//!   two NaN operands, the NaN of the addition.

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Env, Flags};
use floaty_verify::decnumber::{self, Arithmetic, Binary, Double, Outcome, Quad, Single, Unary};
use floaty_verify::dectest::Operation;

use super::operands::{Generator, Shape};
use super::{DpdFloat, Tally, compared, describe_operands, flags_of};

/// An operation of IEEE 754-2019 section 9.6.
#[derive(Clone, Copy, Debug)]
enum MinMax {
    Minimum,
    Maximum,
    MinimumNumber,
    MaximumNumber,
}

impl MinMax {
    const ALL: [Self; 4] = [
        Self::Minimum,
        Self::Maximum,
        Self::MinimumNumber,
        Self::MaximumNumber,
    ];

    /// Returns `true` for an operation that gives the larger operand.
    fn larger(self) -> bool {
        matches!(self, Self::Maximum | Self::MaximumNumber)
    }

    /// Returns `true` for an operation that prefers a number to a NaN.
    fn prefers_numbers(self) -> bool {
        matches!(self, Self::MinimumNumber | Self::MaximumNumber)
    }
}

/// Returns the expected result and flags of `operation` on `x` and `y`, by
/// the rules of the module documentation.
fn expected<F: Arithmetic>(operation: MinMax, x: F::Bits, y: F::Bits) -> (F::Bits, Flags) {
    let nan = |value: F::Bits| F::class(value).ends_with("NaN");
    let rounding = decnumber::Rounding::HalfEven;
    let of = |outcome: Outcome<F::Bits>| (outcome.value, flags_of(outcome.status));
    match (nan(x), nan(y)) {
        (false, false) => {
            let extremum = if operation.larger() {
                Binary::Max
            } else {
                Binary::Min
            };
            of(F::binary(extremum, x, y, rounding))
        }
        (true, false) | (false, true) if operation.prefers_numbers() => {
            let (number, other) = if nan(x) { (y, x) } else { (x, y) };
            let flags = if F::class(other) == "sNaN" {
                Flags::INVALID
            } else {
                Flags::NONE
            };
            (F::unary(Unary::Canonical, number, rounding).value, flags)
        }
        _ => of(F::binary(Binary::Add, x, y, rounding)),
    }
}

/// Runs the four operations on `count` random pairs of format `F`, in both
/// operand orders, and compares floaty with decNumber. A quarter of the
/// pairs are raw encodings, which can be non-canonical.
fn run<F: Arithmetic, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let mut tally = Tally::default();
    for index in 0..count {
        let operands = if index % 4 == 0 {
            vec![generator.raw(), generator.raw()]
        } else {
            generator.operands(&Operation::Binary(Binary::Max))
        };
        let [x, y] = operands[..] else {
            panic!("a minimum takes two operands");
        };
        for (a, b) in [(x, y), (y, x)] {
            let (af, bf) = (DpdFloat::<W>::from_bits(a), DpdFloat::<W>::from_bits(b));
            for operation in MinMax::ALL {
                let (result, flags) = match operation {
                    MinMax::Minimum => af.minimum_with(bf, Env::IEEE),
                    MinMax::Maximum => af.maximum_with(bf, Env::IEEE),
                    MinMax::MinimumNumber => af.minimum_number_with(bf, Env::IEEE),
                    MinMax::MaximumNumber => af.maximum_number_with(bf, Env::IEEE),
                };
                let ours = (result.to_bits(), compared(flags));
                let want = expected::<F>(operation, a, b);
                if ours == want {
                    tally.passed += 1;
                    continue;
                }
                tally.fail(|| {
                    format!(
                        "{operation:?} [{}]: floaty gives {} {:?}, expected {} {:?}",
                        describe_operands::<F>(&[a, b]),
                        F::to_string(ours.0),
                        ours.1,
                        F::to_string(want.0),
                        want.1,
                    )
                });
            }
        }
    }
    tally
}

#[test]
fn decimal32_minimum_and_maximum() {
    let tally = run::<Single, 32>(20_000, 0x0032_3196);
    tally.report("decimal32 minimum and maximum");
    tally.assert_counts(160_000, &[], &[]);
}

#[test]
fn decimal64_minimum_and_maximum() {
    let tally = run::<Double, 64>(20_000, 0x0064_3196);
    tally.report("decimal64 minimum and maximum");
    tally.assert_counts(160_000, &[], &[]);
}

#[test]
fn decimal128_minimum_and_maximum() {
    let tally = run::<Quad, 128>(20_000, 0x0128_3196);
    tally.report("decimal128 minimum and maximum");
    tally.assert_counts(160_000, &[], &[]);
}
