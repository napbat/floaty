//! Seeded random operands through floaty and decNumber, in the six rounding
//! modes that both have: result bits, the five IEEE 754 flags, `TINY`, and
//! `ROUNDED_UP`.
//!
//! decimal64 and decimal128 run against `decDouble` and `decQuad`, and
//! decimal32 against decNumber's arbitrary-precision numbers in the decimal32
//! context. The decimal32 operations leave out the copies, because
//! decNumber's arbitrary-precision copies give a canonical encoding, and
//! floaty's sign operations keep the bits. The `dd` and `dq` vectors and the
//! decimal64 and decimal128 random cases check the copies.

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty_verify::decnumber::{Arithmetic, Binary, Double, Quad, Single, Unary};
use floaty_verify::dectest::Operation;

use super::operands::{Generator, Shape};
use super::{
    Report, SHARED_ROUNDINGS, Tally, compared, describe, describe_operands, details, direction,
    excluded, flags_of, keeps_encoding, noted, run_floaty, run_oracle,
};

/// The arithmetic operations, which round.
pub(super) fn arithmetic() -> Vec<Operation> {
    [
        Binary::Add,
        Binary::Subtract,
        Binary::Multiply,
        Binary::Divide,
    ]
    .into_iter()
    .map(Operation::Binary)
    .chain([Operation::Fma])
    .collect()
}

/// The other operations that floaty and decNumber share.
pub(super) fn others() -> Vec<Operation> {
    let unary = [
        Unary::ToIntegralExact,
        Unary::LogB,
        Unary::NextPlus,
        Unary::NextMinus,
        Unary::Copy,
        Unary::CopyAbs,
        Unary::CopyNegate,
    ];
    let binary = [
        Binary::Quantize,
        Binary::ScaleB,
        Binary::RemainderNear,
        Binary::Compare,
        Binary::CompareSignal,
        Binary::CompareTotal,
        Binary::CompareTotalMag,
        Binary::Max,
        Binary::Min,
        Binary::CopySign,
    ];
    unary
        .into_iter()
        .map(Operation::Unary)
        .chain(binary.into_iter().map(Operation::Binary))
        .chain([Operation::Class, Operation::SameQuantum])
        .collect()
}

/// Runs `count` random cases of each operation in format `F`, in every
/// shared rounding mode, and compares floaty with decNumber.
fn run_random<F: Arithmetic, const W: usize>(
    operations: &[Operation],
    count: usize,
    seed: u64,
) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let mut tally = Tally::default();
    for operation in operations {
        let report = Report::of(operation);
        for _ in 0..count {
            let operands = generator.operands(operation);
            let toward_zero = report.needs_toward_zero().then(|| {
                run_oracle::<F>(
                    operation,
                    &operands,
                    floaty_verify::decnumber::Rounding::Down,
                )
            });
            for rounding in SHARED_ROUNDINGS {
                let (expected, status) = run_oracle::<F>(operation, &operands, rounding);
                if let Some(reason) = excluded::<F>(operation, &operands, status) {
                    tally.skip(reason);
                    continue;
                }
                if let Some(note) = noted::<F>(operation, &operands) {
                    tally.note(note);
                }
                let direction = direction(rounding).expect("a shared mode has a direction");
                let (answer, flags) = run_floaty::<F, W>(operation, &operands, direction);
                let flags = compared(flags);
                let expected_flags =
                    flags_of(status) | details::<F>(operation, (expected, status), toward_zero);
                if answer == expected && flags == expected_flags {
                    tally.passed += 1;
                    continue;
                }
                tally.fail(|| {
                    format!(
                        "{operation:?} {rounding:?} [{}]: floaty gives {} {flags:?}, \
                         decNumber {} {expected_flags:?} ({status})",
                        describe_operands::<F>(&operands),
                        describe::<F>(&answer),
                        describe::<F>(&expected),
                    )
                });
            }
        }
    }
    tally
}

/// The skip reason of a fused multiply-add with a signaling NaN factor and
/// a signaling NaN addend.
pub(super) const SIGNALING_PAIR: &str =
    "fma: signaling NaN factor and addend; floaty takes SoftFloat's NaN order";

/// The note of a fused multiply-add of `0 * inf` and a quiet NaN.
const INVALID_PRODUCT: &str =
    "fma: 0 * inf + quiet NaN, which runs with InvalidProduct::YieldsToNan";

/// Returns the skip counts of the other operations.
fn other_skips(
    impossible: usize,
    nan_scale: usize,
    scale_limit: usize,
    scale_exponent: usize,
) -> [(&'static str, usize); 4] {
    [
        (
            "remaindernear: decNumber's Division_impossible; floaty follows IEEE 754",
            impossible,
        ),
        ("scaleb: a NaN scale operand", nan_scale),
        (
            "scaleb: a scale operand beyond decNumber's limit of 2 * (emax + p)",
            scale_limit,
        ),
        (
            "scaleb: a scale operand that is not an integer with exponent 0",
            scale_exponent,
        ),
    ]
}

#[test]
fn random_decimal64_arithmetic() {
    let tally = run_random::<Double, 64>(&arithmetic(), 30_000, 0x6464_0001);
    tally.report("random decimal64 arithmetic");
    // The generator is seeded, so the counts are exact.
    tally.assert_counts(899_862, &[(SIGNALING_PAIR, 138)], &[(INVALID_PRODUCT, 858)]);
}

#[test]
fn random_decimal64_operations() {
    let tally = run_random::<Double, 64>(&others(), 10_000, 0x6464_0002);
    tally.report("random decimal64 operations");
    tally.assert_counts(1_127_568, &other_skips(3666, 438, 2838, 5490), &[]);
}

#[test]
fn random_decimal128_arithmetic() {
    let tally = run_random::<Quad, 128>(&arithmetic(), 30_000, 0x0128_0001);
    tally.report("random decimal128 arithmetic");
    tally.assert_counts(899_862, &[(SIGNALING_PAIR, 138)], &[(INVALID_PRODUCT, 870)]);
}

#[test]
fn random_decimal128_operations() {
    let tally = run_random::<Quad, 128>(&others(), 10_000, 0x0128_0002);
    tally.report("random decimal128 operations");
    tally.assert_counts(1_127_598, &other_skips(3456, 498, 3060, 5388), &[]);
}

#[test]
fn random_decimal32_arithmetic() {
    let tally = run_random::<Single, 32>(&arithmetic(), 30_000, 0x0032_0002);
    tally.report("random decimal32 arithmetic");
    tally.assert_counts(899_880, &[(SIGNALING_PAIR, 120)], &[(INVALID_PRODUCT, 696)]);
}

#[test]
fn random_decimal32_operations() {
    let operations: Vec<Operation> = others()
        .into_iter()
        .filter(|operation| !keeps_encoding(operation))
        .collect();
    let tally = run_random::<Single, 32>(&operations, 10_000, 0x0032_0003);
    tally.report("random decimal32 operations");
    tally.assert_counts(886_818, &other_skips(4206, 372, 3264, 5340), &[]);
}
