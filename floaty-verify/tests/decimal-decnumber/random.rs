//! Seeded random operands through floaty and decNumber, in the eight rounding
//! modes of decNumber: result bits, the five IEEE 754 flags, `TINY`, and
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
    excluded, flags_of, keeps_encoding, noted, run_floaty, run_floaty_static, run_oracle,
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
    let mut static_checks = 0_usize;
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
                let direction = direction(rounding);
                let (answer, flags) = run_floaty::<F, W>(operation, &operands, direction);
                if let Some((fixed, fixed_flags)) =
                    run_floaty_static::<F, W>(operation, &operands, direction)
                {
                    // The static mode must give the result and the flags of
                    // its Env, which the comparison below checks.
                    assert!(
                        fixed == answer && fixed_flags == flags,
                        "{operation:?} {rounding:?} [{}]: the static mode gives {} {fixed_flags:?}",
                        describe_operands::<F>(&operands),
                        describe::<F>(&fixed),
                    );
                    static_checks += 1;
                }
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
    let has_addition = operations
        .iter()
        .any(|operation| matches!(operation, Operation::Binary(Binary::Add)));
    assert!(
        static_checks > 0 || !has_addition,
        "the static modes checked arithmetic cases"
    );
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
    tally.assert_counts(
        1_199_816,
        &[(SIGNALING_PAIR, 184)],
        &[(INVALID_PRODUCT, 1144)],
    );
}

#[test]
fn random_decimal64_operations() {
    let tally = run_random::<Double, 64>(&others(), 10_000, 0x6464_0002);
    tally.report("random decimal64 operations");
    tally.assert_counts(1_503_424, &other_skips(4888, 584, 3784, 7320), &[]);
}

#[test]
fn random_decimal128_arithmetic() {
    let tally = run_random::<Quad, 128>(&arithmetic(), 30_000, 0x0128_0001);
    tally.report("random decimal128 arithmetic");
    tally.assert_counts(
        1_199_816,
        &[(SIGNALING_PAIR, 184)],
        &[(INVALID_PRODUCT, 1160)],
    );
}

#[test]
fn random_decimal128_operations() {
    let tally = run_random::<Quad, 128>(&others(), 10_000, 0x0128_0002);
    tally.report("random decimal128 operations");
    tally.assert_counts(1_503_464, &other_skips(4608, 664, 4080, 7184), &[]);
}

#[test]
fn random_decimal32_arithmetic() {
    let tally = run_random::<Single, 32>(&arithmetic(), 30_000, 0x0032_0002);
    tally.report("random decimal32 arithmetic");
    tally.assert_counts(
        1_199_840,
        &[(SIGNALING_PAIR, 160)],
        &[(INVALID_PRODUCT, 928)],
    );
}

#[test]
fn random_decimal32_operations() {
    let operations: Vec<Operation> = others()
        .into_iter()
        .filter(|operation| !keeps_encoding(operation))
        .collect();
    let tally = run_random::<Single, 32>(&operations, 10_000, 0x0032_0003);
    tally.report("random decimal32 operations");
    tally.assert_counts(1_182_424, &other_skips(5608, 496, 4352, 7120), &[]);
}
