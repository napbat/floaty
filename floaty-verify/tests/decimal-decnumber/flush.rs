//! Flush-to-zero and denormals-are-zero in the decimal formats. No decimal
//! library has them, so the test applies each definition of floaty to
//! decNumber's results.
//!
//! - DAZ reads a subnormal operand as a zero with its sign and its exponent.
//!   The expected result is decNumber's result for the operands with each
//!   subnormal operand replaced so.
//! - FTZ replaces a tiny result of a rounded operation with a zero of its
//!   sign and the least exponent, and signals underflow and inexact. A
//!   decimal result is tiny when its exact value is below `10^emin` in
//!   magnitude. decNumber's result rounded toward zero tells: the exact
//!   value is below `10^emin` exactly when that result is subnormal, or is a
//!   zero that is inexact. The other operations give exact results or an
//!   operand, so FTZ does not change them.
//! - An operation that reads a behavior reports `DENORMAL_INPUT` for a
//!   subnormal operand, with DAZ or without it.
//! - A conversion between the DPD widths under DAZ converts a subnormal
//!   operand as that zero. The expected result is decNumber's `ToWider` or
//!   `FromWider` of the zero.
//!
//! decimal32 runs without the copies, as for its random cases in
//! `random.rs`.

use floaty::format::{Decimal, Dpd, Standard, Storage, Width};
use floaty::{Env, Flags, Rounding};
use floaty_verify::decnumber::{
    self, Arithmetic, Binary, Double, Outcome, Quad, Single, Status, Unary, Widening,
};
use floaty_verify::dectest::Operation;

use super::operands::{Generator, Shape};
use super::random::{SIGNALING_PAIR, arithmetic, others};
use super::{
    Answer, DpdFloat, SHARED_ROUNDINGS, Tally, describe, describe_operands, direction, excluded,
    flags_of, ieee, keeps_encoding, run_floaty_in, run_oracle,
};

/// Returns whether an operation rounds its result, so that FTZ applies.
fn rounds(operation: &Operation) -> bool {
    matches!(
        operation,
        Operation::Fma
            | Operation::Binary(
                Binary::Add | Binary::Subtract | Binary::Multiply | Binary::Divide | Binary::ScaleB
            )
    )
}

/// Returns whether floaty's operation reads a behavior, so that DAZ applies.
/// The sign operations, the total order, the class, and `same_quantum` read
/// none.
fn reads_behavior(operation: &Operation) -> bool {
    !matches!(
        operation,
        Operation::Class
            | Operation::SameQuantum
            | Operation::Unary(Unary::Copy | Unary::CopyAbs | Unary::CopyNegate)
            | Operation::Binary(Binary::CopySign | Binary::CompareTotal | Binary::CompareTotalMag)
    )
}

/// Returns whether an encoding is subnormal, by decNumber's class.
fn subnormal<F: Arithmetic>(bits: F::Bits) -> bool {
    F::class(bits).ends_with("Subnormal")
}

/// Returns the operands that DAZ reads: each subnormal operand becomes a
/// zero with its sign and exponent. The operand of `scaleb` that gives the
/// scale is an integer, not a floaty operand, so it stays.
fn denormals_are_zero<F: Arithmetic>(operation: &Operation, operands: &[F::Bits]) -> Vec<F::Bits> {
    let scale = matches!(operation, Operation::Binary(Binary::ScaleB));
    operands
        .iter()
        .enumerate()
        .map(|(index, &bits)| {
            if !subnormal::<F>(bits) || (scale && index == 1) {
                return bits;
            }
            daz_zero::<F>(bits)
        })
        .collect()
}

/// Returns the zero that DAZ reads for a subnormal encoding: the zero with
/// its sign and the exponent of its last digit.
fn daz_zero<F: Arithmetic>(bits: F::Bits) -> F::Bits {
    // decNumber writes a subnormal value as `[-]d.dddE-n`.
    let text = F::to_string(bits);
    let (mantissa, exponent) = text
        .split_once('E')
        .expect("a subnormal string has an exponent");
    let negative = mantissa.starts_with('-');
    let fraction = mantissa
        .split_once('.')
        .map_or(0, |(_, digits)| digits.len());
    let adjusted: i64 = exponent.parse().expect("the exponent is an integer");
    let exponent = adjusted - i64::try_from(fraction).expect("a digit count fits an i64");
    let zero = format!("{}0E{exponent}", if negative { "-" } else { "" });
    F::from_string(&zero, decnumber::Rounding::HalfEven).value
}

/// Returns whether the exact result of an operation is tiny, from
/// decNumber's result rounded toward zero.
fn tiny<F: Arithmetic>(operation: &Operation, operands: &[F::Bits]) -> Option<bool> {
    let (answer, status) = run_oracle::<F>(operation, operands, decnumber::Rounding::Down);
    let Answer::Encoding(bits) = answer else {
        return None;
    };
    let class = F::class(bits);
    Some(
        class.ends_with("Subnormal")
            || (class.ends_with("Zero") && status.intersects(Status::INEXACT)),
    )
}

/// Returns the zero of the least exponent with the sign of `bits`.
fn least_zero<F: Arithmetic>(bits: F::Bits) -> F::Bits {
    let negative = F::to_string(bits).starts_with('-');
    let (lowest, _) = Generator::<F>::exponents();
    let zero = format!("{}0E{lowest}", if negative { "-" } else { "" });
    F::from_string(&zero, decnumber::Rounding::HalfEven).value
}

/// The note of a case where FTZ replaces a tiny result.
const FLUSHED: &str = "ftz: a tiny result becomes a zero";

/// The note of a case where DAZ replaces a subnormal operand.
const REPLACED: &str = "daz: a subnormal operand reads as a zero";

/// The expected result and flags of a case, and the note of the rule that
/// changed them.
type Expected<B> = (Answer<B>, Flags, Option<&'static str>);

/// Returns the expected result and flags of an operation under a behavior
/// with FTZ and DAZ as `env` sets them.
fn expected<F: Arithmetic, const W: usize>(
    operation: &Operation,
    operands: &[F::Bits],
    rounding: decnumber::Rounding,
    env: Env,
) -> Result<Expected<F::Bits>, &'static str>
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let reads = reads_behavior(operation);
    let read = if reads && env.denormals_are_zero {
        denormals_are_zero::<F>(operation, operands)
    } else {
        operands.to_vec()
    };
    let (answer, status) = run_oracle::<F>(operation, &read, rounding);
    if let Some(reason) = excluded::<F>(operation, &read, status) {
        return Err(reason);
    }
    let denormal = reads
        && operands.iter().enumerate().any(|(index, &bits)| {
            subnormal::<F>(bits)
                && !(index == 1 && matches!(operation, Operation::Binary(Binary::ScaleB)))
        });
    let input = if denormal {
        Flags::DENORMAL_INPUT
    } else {
        Flags::NONE
    };
    if env.flush_to_zero && rounds(operation) && tiny::<F>(operation, &read) == Some(true) {
        let Answer::Encoding(bits) = answer else {
            unreachable!("a rounded operation gives an encoding");
        };
        let zero = Answer::Encoding(least_zero::<F>(bits));
        return Ok((
            zero,
            input | Flags::UNDERFLOW | Flags::INEXACT,
            Some(FLUSHED),
        ));
    }
    let note = (read != operands).then_some(REPLACED);
    Ok((answer, input | flags_of(status), note))
}

/// The behaviors of the test: FTZ, DAZ, and both.
fn behaviors(rounding: Rounding) -> [Env; 3] {
    let env = Env::IEEE.with_rounding(rounding);
    [
        env.with_flush_to_zero(true),
        env.with_denormals_are_zero(true),
        env.with_flush_to_zero(true).with_denormals_are_zero(true),
    ]
}

/// Runs `count` random cases of every operation in format `F`, in every
/// shared rounding mode and every behavior.
fn run<F: Arithmetic, const W: usize>(count: usize, seed: u64) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    let operations: Vec<Operation> = arithmetic().into_iter().chain(others()).collect();
    run_operations::<F, W>(&operations, count, seed)
}

/// Runs `count` random cases of each operation in format `F`, in every
/// shared rounding mode and every behavior.
fn run_operations<F: Arithmetic, const W: usize>(
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
        for _ in 0..count {
            let operands = generator.operands(operation);
            for rounding in SHARED_ROUNDINGS {
                let direction = direction(rounding);
                for env in behaviors(direction) {
                    let (expected, expected_flags, note) =
                        match expected::<F, W>(operation, &operands, rounding, env) {
                            Ok(expected) => expected,
                            Err(reason) => {
                                tally.skip(reason);
                                continue;
                            }
                        };
                    if let Some(note) = note {
                        tally.note(note);
                    }
                    let (answer, flags) = run_floaty_in::<F, W>(operation, &operands, env);
                    let denormal = if flags.contains(Flags::DENORMAL_INPUT) {
                        Flags::DENORMAL_INPUT
                    } else {
                        Flags::NONE
                    };
                    let flags = ieee(flags) | denormal;
                    if answer == expected && flags == expected_flags {
                        tally.passed += 1;
                        continue;
                    }
                    tally.fail(|| {
                        format!(
                            "{operation:?} {rounding:?} ftz {} daz {} [{}]: floaty gives {} \
                             {flags:?}, expected {} {expected_flags:?}",
                            env.flush_to_zero,
                            env.denormals_are_zero,
                            describe_operands::<F>(&operands),
                            describe::<F>(&answer),
                            describe::<F>(&expected),
                        )
                    });
                }
            }
        }
    }
    tally
}

/// Returns the skip counts: `Division_impossible`, a NaN scale, a scale
/// beyond decNumber's limit, and a scale that is not an integer.
fn skips(
    impossible: usize,
    nan: usize,
    limit: usize,
    exponent: usize,
) -> Vec<(&'static str, usize)> {
    vec![
        (
            "remainder, remaindernear: decNumber's Division_impossible; floaty follows IEEE 754",
            impossible,
        ),
        ("scaleb: a NaN scale operand", nan),
        (
            "scaleb: a scale operand beyond decNumber's limit of 2 * (emax + p)",
            limit,
        ),
        (
            "scaleb: a scale operand that is not an integer with exponent 0",
            exponent,
        ),
    ]
}

#[test]
fn decimal64_flush_to_zero_and_denormals_are_zero() {
    let tally = run::<Double, 64>(1_000, 0x6464_0F70);
    tally.report("decimal64 FTZ and DAZ");
    // The generator is seeded, so the counts are exact.
    let mut skipped = skips(2360, 192, 1440, 2184);
    skipped.push((SIGNALING_PAIR, 24));
    tally.assert_counts(593_800, &skipped, &[(REPLACED, 39_000), (FLUSHED, 10_032)]);
}

#[test]
fn decimal128_flush_to_zero_and_denormals_are_zero() {
    let tally = run::<Quad, 128>(1_000, 0x0128_0F70);
    tally.report("decimal128 FTZ and DAZ");
    let skipped = skips(2824, 168, 1632, 2112);
    tally.assert_counts(593_264, &skipped, &[(REPLACED, 38_248), (FLUSHED, 9_544)]);
}

#[test]
fn decimal32_flush_to_zero_and_denormals_are_zero() {
    let operations: Vec<Operation> = arithmetic()
        .into_iter()
        .chain(
            others()
                .into_iter()
                .filter(|operation| !keeps_encoding(operation)),
        )
        .collect();
    let tally = run_operations::<Single, 32>(&operations, 1_000, 0x0032_0F70);
    tally.report("decimal32 FTZ and DAZ");
    let mut skipped = skips(2896, 96, 1128, 2352);
    skipped.push((SIGNALING_PAIR, 24));
    tally.assert_counts(497_504, &skipped, &[(REPLACED, 37_424), (FLUSHED, 9_968)]);
}

/// The note of a conversion of a subnormal operand under DAZ.
const CONVERTED: &str = "daz: a subnormal operand converts as a zero";

/// Converts the subnormal operands among `count` random operands of format
/// `S` to format `T` under DAZ, in every shared rounding mode. The expected
/// result is `oracle`, decNumber's conversion, of the zero that DAZ reads.
fn convert_subnormals<S: Arithmetic, T: Arithmetic, const FROM: usize, const TO: usize>(
    tally: &mut Tally,
    count: usize,
    seed: u64,
    oracle: impl Fn(S::Bits, decnumber::Rounding) -> Outcome<T::Bits>,
) where
    Width<FROM>: Storage<Bits = S::Bits>,
    Decimal<Dpd>: Standard<FROM, Bits = S::Bits>,
    Width<TO>: Storage<Bits = T::Bits>,
    Decimal<Dpd>: Standard<TO, Bits = T::Bits>,
{
    let mut generator = Generator::<S>::new(seed, Shape::of::<S>());
    let operands: Vec<S::Bits> = (0..count)
        .map(|_| generator.operand().0)
        .filter(|&bits| subnormal::<S>(bits))
        .collect();
    for x in operands {
        let zero = daz_zero::<S>(x);
        for rounding in SHARED_ROUNDINGS {
            let env = Env::IEEE
                .with_rounding(direction(rounding))
                .with_denormals_are_zero(true);
            let (result, flags) = DpdFloat::<FROM>::from_bits(x).convert_with::<DpdFloat<TO>>(env);
            let denormal = if flags.contains(Flags::DENORMAL_INPUT) {
                Flags::DENORMAL_INPUT
            } else {
                Flags::NONE
            };
            let flags = ieee(flags) | denormal;
            let outcome = oracle(zero, rounding);
            // decNumber's string of its result gives the canonical encoding.
            let expected =
                T::from_string(&T::to_string(outcome.value), decnumber::Rounding::HalfEven).value;
            let expected_flags = Flags::DENORMAL_INPUT | flags_of(outcome.status);
            tally.note(CONVERTED);
            if result.to_bits() == expected && flags == expected_flags {
                tally.passed += 1;
                continue;
            }
            tally.fail(|| {
                format!(
                    "{} to {} {rounding:?} daz [{}]: floaty gives {} {flags:?}, expected {} \
                     {expected_flags:?}",
                    S::NAME,
                    T::NAME,
                    describe_operands::<S>(&[x]),
                    describe::<T>(&Answer::Encoding(result.to_bits())),
                    describe::<T>(&Answer::Encoding(expected)),
                )
            });
        }
    }
}

/// Returns the outcome of an exact widening.
fn widen<B>(value: B) -> Outcome<B> {
    Outcome {
        value,
        status: Status::NONE,
    }
}

#[test]
fn subnormal_operands_convert_as_zeros_under_denormals_are_zero() {
    let mut tally = Tally::default();
    convert_subnormals::<Single, Double, 32, 64>(&mut tally, 20_000, 0x0032_0DA2, |x, _| {
        widen(Single::to_wider(x))
    });
    convert_subnormals::<Double, Quad, 64, 128>(&mut tally, 20_000, 0x0064_0DA2, |x, _| {
        widen(Double::to_wider(x))
    });
    convert_subnormals::<Double, Single, 64, 32>(
        &mut tally,
        20_000,
        0x0064_0DA3,
        Single::from_wider,
    );
    convert_subnormals::<Quad, Double, 128, 64>(
        &mut tally,
        20_000,
        0x0128_0DA3,
        Double::from_wider,
    );
    tally.report("DAZ conversions");
    // The generator is seeded, so the counts are exact.
    tally.assert_counts(58_136, &[], &[(CONVERTED, 58_136)]);
}
