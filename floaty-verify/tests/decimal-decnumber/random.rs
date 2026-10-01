//! Seeded random operands through floaty and decNumber, in the eight rounding
//! modes of decNumber: result bits, the five IEEE 754 flags, `TINY`, and
//! `ROUNDED_UP`. Each case also runs on the BID format of the width, through
//! the BID and DPD conversions of the Intel decimal library, except the
//! copies, whose result keeps the encoding of the operand.
//!
//! decimal64 and decimal128 run against `decDouble` and `decQuad`, and
//! decimal32 against decNumber's arbitrary-precision numbers in the decimal32
//! context.
//!
//! Each case whose result is a NaN in the direction to nearest even also runs
//! with each NaN rule of [`NAN_RULES`]. It must give the NaN that the rule
//! selects from the NaN operands, by `floaty_verify::mpfr::select_nan`, and
//! the flags of the case. An invalid operation without a NaN operand gives
//! the default NaN of the rule. A fused multiply-add offers its NaN operands
//! in the fused order of the rule, and `0 * inf` with a NaN addend follows
//! its invalid-product rule, as the documentation of `FusedNanOrder` and
//! `InvalidProduct` states. decNumber encodes the expected NaN from its text.
//! The decimal32 operations leave out the copies, because
//! decNumber's arbitrary-precision copies give a canonical encoding, and
//! floaty's sign operations keep the bits. The `dd` and `dq` vectors and the
//! decimal64 and decimal128 random cases check the copies.

use floaty::env::{FusedNanOrder, InvalidProduct, NanPropagation, NanRule};
use floaty::format::{Bid, Decimal, Dpd, Standard, Storage, Width};
use floaty::{Decoded, Env, Flags};
use floaty_verify::decnumber::{self, Arithmetic, Binary, Double, Quad, Single, Unary};
use floaty_verify::dectest::Operation;
use floaty_verify::mpfr::{Nan, select_nan};
use rug::Integer;
use rug::integer::Order;

use super::operands::{Generator, Shape};
use super::{
    Answer, DpdFloat, Report, SHARED_ROUNDINGS, Tally, Transcode, compared, describe,
    describe_operands, details, direction, excluded, flags_of, keeps_encoding, noted, run_floaty,
    run_floaty_bid, run_floaty_in, run_floaty_static, run_oracle,
};

/// The NaN rules of the NaN cases: the default NaN, negative so that its
/// sign shows the rule; the first operand, with a NaN addend before the
/// invalid product; the larger significand, with the addend first and a
/// signaling invalid product; and the signaling NaN first, with the addend
/// second, as on PowerPC.
const NAN_RULES: [NanRule; 4] = [
    NanRule::new(NanPropagation::DefaultNan).with_default_negative(true),
    NanRule::new(NanPropagation::FirstOperand).with_invalid_product(InvalidProduct::YieldsToNan),
    NanRule::new(NanPropagation::LargerSignificand)
        .with_default_negative(true)
        .with_fused_order(FusedNanOrder::AddendFirst)
        .with_invalid_product(InvalidProduct::Signals),
    NanRule::new(NanPropagation::SignalingFirst)
        .with_fused_order(FusedNanOrder::AddendSecond)
        .with_invalid_product(InvalidProduct::SignalsAndYieldsToNan),
];

/// Returns `true` for a DPD encoding of `W` bits that is a NaN: its five
/// combination bits below the sign are all ones.
fn is_nan<const W: usize>(bits: u128) -> bool {
    let width = u32::try_from(W).expect("a width fits a u32");
    (bits >> (width - 6)) & 0x1F == 0x1F
}

/// Returns an operand as a NaN of the oracle, or `None` for a number.
fn nan_of<const W: usize>(bits: <Width<W> as Storage>::Bits) -> Option<Nan>
where
    Width<W>: Storage,
    Decimal<Dpd>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    match DpdFloat::<W>::from_bits(bits).decode::<2>() {
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => Some(Nan {
            negative,
            signaling,
            payload: Integer::from_digits(&payload, Order::Lsf),
        }),
        _ => None,
    }
}

/// Returns the NaN that `env` gives for an operation on `operands`, and
/// whether the invalid product of a fused multiply-add signals invalid.
fn expected_nan<const W: usize>(
    operation: &Operation,
    operands: &[<Width<W> as Storage>::Bits],
    env: &Env,
) -> (Nan, bool)
where
    Width<W>: Storage,
    Decimal<Dpd>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    let nans: Vec<Option<Nan>> = operands.iter().map(|&bits| nan_of::<W>(bits)).collect();
    let offer = |list: &[&Option<Nan>]| -> Option<Nan> {
        let offered: Vec<Nan> = list.iter().filter_map(|nan| (*nan).clone()).collect();
        (!offered.is_empty()).then(|| select_nan(&offered, env))
    };
    let Operation::Fma = operation else {
        let all: Vec<&Option<Nan>> = nans.iter().collect();
        return (offer(&all).unwrap_or_else(|| Nan::default_of(env)), false);
    };
    let [first, second, addend] = &nans[..] else {
        panic!("a fused multiply-add takes three operands");
    };
    if first.is_some() || second.is_some() {
        let nan = match env.nan.fused_order {
            FusedNanOrder::ProductFirst => {
                let product = offer(&[first, second]);
                offer(&[&product, addend])
            }
            FusedNanOrder::AddendFirst => offer(&[addend, first, second]),
            FusedNanOrder::AddendSecond => offer(&[first, addend, second]),
            other => panic!("the test has no rule for {other:?}"),
        };
        return (nan.expect("a factor is a NaN"), false);
    }
    let Some(addend) = addend else {
        // An invalid operation of numbers, such as `inf * 0 + inf`.
        return (Nan::default_of(env), false);
    };
    let zero_times_infinity = {
        let class = |bits| DpdFloat::<W>::from_bits(bits).classify();
        let (a, b) = (class(operands[0]), class(operands[1]));
        matches!(
            (a, b),
            (floaty::Class::Zero, floaty::Class::Infinite)
                | (floaty::Class::Infinite, floaty::Class::Zero)
        )
    };
    if !zero_times_infinity {
        return (select_nan(core::slice::from_ref(addend), env), false);
    }
    match env.nan.invalid_product {
        InvalidProduct::Signals => (
            select_nan(&[Nan::default_of(env), addend.clone()], env),
            true,
        ),
        InvalidProduct::YieldsToNan => (select_nan(core::slice::from_ref(addend), env), false),
        InvalidProduct::SignalsAndYieldsToNan => {
            (select_nan(core::slice::from_ref(addend), env), true)
        }
        other => panic!("the test has no rule for {other:?}"),
    }
}

/// Runs a case whose decNumber result `expected` is a NaN with each rule of
/// [`NAN_RULES`], and records a failure when floaty does not give the NaN of
/// [`expected_nan`] with `expected_flags`. Returns the number of checks.
fn check_nan_rules<F: Arithmetic, const W: usize>(
    operation: &Operation,
    operands: &[F::Bits],
    (expected, expected_flags): (Answer<F::Bits>, Flags),
    tally: &mut Tally,
) -> usize
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
{
    // A copy keeps the bits of its operand under every NaN rule.
    let nan_result = expected
        .encoding()
        .is_some_and(|bits| is_nan::<W>(bits.into()));
    if keeps_encoding(operation) || !nan_result {
        return 0;
    }
    for rule in NAN_RULES {
        let env = Env::IEEE.with_nan(rule);
        let (answer, flags) = match operation {
            // `run_floaty_in` gives `fma` the invalid-product rule of
            // decNumber, so the rule of the case runs here.
            Operation::Fma => {
                let value = |index: usize| DpdFloat::<W>::from_bits(operands[index]);
                let (result, flags) = value(0).mul_add_with(value(1), value(2), env);
                (Answer::Encoding(result.to_bits()), flags)
            }
            _ => run_floaty_in::<F, W>(operation, operands, env),
        };
        let (nan, signals) = expected_nan::<W>(operation, operands, &env);
        let sign = if nan.negative { "-" } else { "" };
        let payload = if nan.payload == 0 {
            String::new()
        } else {
            nan.payload.to_string()
        };
        let text = format!("{sign}NaN{payload}");
        let want = Answer::Encoding(F::from_string(&text, decnumber::Rounding::HalfEven).value);
        let want_flags = if signals {
            expected_flags | Flags::INVALID
        } else {
            expected_flags
        };
        let flags = compared(flags);
        if answer != want || flags != want_flags {
            tally.fail(|| {
                format!(
                    "{operation:?} {rule:?} [{}]: floaty gives {} {flags:?}, expected {} \
                     {want_flags:?}",
                    describe_operands::<F>(operands),
                    describe::<F>(&answer),
                    describe::<F>(&want),
                )
            });
        }
    }
    NAN_RULES.len()
}

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
        Binary::Remainder,
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
fn run_random<F: Transcode, const W: usize>(
    operations: &[Operation],
    count: usize,
    seed: u64,
) -> Tally
where
    Width<W>: Storage<Bits = F::Bits>,
    Decimal<Dpd>: Standard<W, Bits = F::Bits>,
    Decimal<Bid>: Standard<W, Bits = F::Bits>,
{
    let mut generator = Generator::<F>::new(seed, Shape::of::<F>());
    let mut tally = Tally::default();
    let mut static_checks = 0_usize;
    let mut nan_rule_checks = 0_usize;
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
                if rounding == floaty_verify::decnumber::Rounding::HalfEven {
                    nan_rule_checks += check_nan_rules::<F, W>(
                        operation,
                        &operands,
                        (expected, expected_flags),
                        &mut tally,
                    );
                }
                if !keeps_encoding(operation) {
                    let env = Env::IEEE.with_rounding(direction);
                    let (bid, bid_flags) = run_floaty_bid::<F, W>(operation, &operands, env);
                    let bid_flags = compared(bid_flags);
                    if bid == expected && bid_flags == expected_flags {
                        tally.passed += 1;
                    } else {
                        tally.fail(|| {
                            format!(
                                "{operation:?} {rounding:?} BID [{}]: floaty gives {} \
                                 {bid_flags:?}, decNumber {} {expected_flags:?} ({status})",
                                describe_operands::<F>(&operands),
                                describe::<F>(&bid),
                                describe::<F>(&expected),
                            )
                        });
                    }
                }
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
    assert!(
        nan_rule_checks > 0,
        "the NaN rules checked cases with a NaN result"
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
            "remainder, remaindernear: decNumber's Division_impossible; floaty follows IEEE 754",
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
    // The generator is seeded, so the counts are exact. Each case passes
    // once for DPD and once for BID.
    tally.assert_counts(
        2_399_632,
        &[(SIGNALING_PAIR, 184)],
        &[(INVALID_PRODUCT, 1144)],
    );
}

#[test]
fn random_decimal64_operations() {
    let tally = run_random::<Double, 64>(&others(), 10_000, 0x6464_0002);
    tally.report("random decimal64 operations");
    tally.assert_counts(2_836_400, &other_skips(10_112, 584, 3784, 7320), &[]);
}

#[test]
fn random_decimal128_arithmetic() {
    let tally = run_random::<Quad, 128>(&arithmetic(), 30_000, 0x0128_0001);
    tally.report("random decimal128 arithmetic");
    tally.assert_counts(
        2_399_632,
        &[(SIGNALING_PAIR, 184)],
        &[(INVALID_PRODUCT, 1160)],
    );
}

#[test]
fn random_decimal128_operations() {
    let tally = run_random::<Quad, 128>(&others(), 10_000, 0x0128_0002);
    tally.report("random decimal128 operations");
    tally.assert_counts(2_837_152, &other_skips(9496, 664, 4080, 7184), &[]);
}

#[test]
fn random_decimal32_arithmetic() {
    let tally = run_random::<Single, 32>(&arithmetic(), 30_000, 0x0032_0002);
    tally.report("random decimal32 arithmetic");
    tally.assert_counts(
        2_399_680,
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
    tally.assert_counts(2_512_848, &other_skips(11_608, 496, 4352, 7120), &[]);
}
