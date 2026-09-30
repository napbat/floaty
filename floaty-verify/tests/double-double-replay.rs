//! Replays the binary64 steps of floaty's double-double operations, and
//! checks each step with the MPFR oracles, in behaviors that no reference
//! defines.
//!
//! floaty documents the rule of these operations on `DoubleDouble`: each step
//! of an algorithm is a binary64 operation under the behavior of the call,
//! and the result has the flags of every step. The steps ignore saturation.
//! The block of `fmal` rounds to nearest even, as glibc's
//! `SET_RESTORE_ROUND (FE_TONEAREST)` does. libgcc and QD define the results
//! only in the behaviors of their platforms, and `tests/double-double.rs` and
//! `tests/double-double-functions.rs` compare those, with `Qd` also under the
//! FTZ and DAZ bits of MXCSR. No implementation
//! defines the other behaviors: flush-to-zero and denormals-are-zero for
//! `Gcc`, precision limits, the other directions, the other NaN and tininess
//! rules, saturation, and the flags `TINY`, `ROUNDED_UP`, and
//! `DENORMAL_INPUT`. So this test evaluates the documented rule with MPFR.
//!
//! An `Observed` behavior reports each step. The test checks that observing
//! changes neither the result nor the flags, that the flags are the union of
//! the flags of the steps, and that each step runs under the behavior of the
//! call. Then the MPFR oracles compute each step: the arithmetic oracle, the
//! integer conversion oracles for a truncation, and the comparison oracle.
//!
//! The replay does not check the branches of an algorithm on the values that
//! only these behaviors give. The branches read only the results of steps,
//! and a comparison is a step too. The reference tests check the branches bit
//! for bit.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use std::cell::RefCell;
use std::num::NonZeroU32;

use floaty::env::{
    Behavior, FusedNanOrder, InvalidProduct, NanPropagation, NanRule, Observed, Step,
    StepOperation, StepResult, Tininess,
};
use floaty::{Decoded, DoubleDouble, Env, F64, Flags, Gcc, Qd, Rounding, ToInt};
use floaty_verify::arithmetic::{self, Operation as Arithmetic};
use floaty_verify::double_double::reference::Pair;
use floaty_verify::double_double::{BINARY64, pair};
use floaty_verify::mpfr::{DIRECTIONS, Operand, Value};
use floaty_verify::operations::compare::{compare_quiet, compare_signaling};
use floaty_verify::operations::integral::{from_int, to_int};
use floaty_verify::operations::{Outcome, outcome};
use floaty_verify::random::SplitMix64;

/// The number of random operand lists of each algorithm in each behavior.
const CASES: usize = 300;

/// A double-double operation with binary64 steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Sqrt,
    Remainder,
    TruncatedRemainder,
    MulAdd,
    NextUp,
    NextDown,
}

/// The operations of `Gcc`.
const GCC: [Operation; 10] = [
    Operation::Add,
    Operation::Sub,
    Operation::Mul,
    Operation::Div,
    Operation::Sqrt,
    Operation::Remainder,
    Operation::TruncatedRemainder,
    Operation::MulAdd,
    Operation::NextUp,
    Operation::NextDown,
];

/// The operations of `Qd`. QD has no `fma`, `nextup`, or `nextdown`.
const QD: [Operation; 7] = [
    Operation::Add,
    Operation::Sub,
    Operation::Mul,
    Operation::Div,
    Operation::Sqrt,
    Operation::Remainder,
    Operation::TruncatedRemainder,
];

/// The result halves and the flags of an operation.
type Result = (u64, u64, Flags);

/// A double-double algorithm.
#[derive(Clone, Copy, Debug)]
enum Algorithm {
    Gcc,
    Qd,
}

impl Algorithm {
    /// Runs an operation of the algorithm under `behavior`.
    fn run(self, operation: Operation, operands: [Pair; 3], behavior: impl Behavior) -> Result {
        match self {
            Self::Gcc => gcc(operation, operands, behavior),
            Self::Qd => qd(operation, operands, behavior),
        }
    }
}

/// Every direction, and the other behaviors with nearest-even rounding and
/// with a direction toward zero.
fn behaviors() -> Vec<Env> {
    let powerpc = NanRule::new(NanPropagation::FirstOperand)
        .with_fused_order(FusedNanOrder::AddendSecond)
        .with_invalid_product(InvalidProduct::SignalsAndYieldsToNan);
    let mut envs: Vec<Env> = DIRECTIONS
        .iter()
        .map(|&rounding| Env::IEEE.with_rounding(rounding))
        .collect();
    for rounding in [Rounding::TiesToEven, Rounding::TowardZero] {
        let env = Env::IEEE.with_rounding(rounding);
        envs.extend([
            env.with_tininess(Tininess::BeforeRounding),
            env.with_flush_to_zero(true),
            env.with_flush_to_zero(true)
                .with_tininess(Tininess::BeforeRounding),
            env.with_denormals_are_zero(true),
            Env::X86_SSE
                .with_rounding(rounding)
                .with_flush_to_zero(true)
                .with_denormals_are_zero(true),
            env.with_precision(NonZeroU32::new(24)),
            env.with_precision(NonZeroU32::new(5))
                .with_tininess(Tininess::BeforeRounding),
            env.with_saturate(true),
            env.with_nan(powerpc).with_flush_to_zero(true),
            env.with_nan(Env::X87.nan),
            env.with_nan(
                NanRule::new(NanPropagation::LargerSignificand)
                    .with_default_negative(true)
                    .with_fused_order(FusedNanOrder::AddendFirst),
            ),
            env.with_nan(
                NanRule::new(NanPropagation::SignalingFirst)
                    .with_fused_order(FusedNanOrder::AddendFirst),
            ),
            env.with_nan(NanRule::new(NanPropagation::DefaultNan).with_default_negative(true)),
        ]);
    }
    envs
}

/// Returns the halves of a result and the flags.
fn outcome_of<Alg: floaty::Algorithm>((value, flags): (DoubleDouble<Alg>, Flags)) -> Result {
    (value.hi().to_bits(), value.lo().to_bits(), flags)
}

/// Returns the double-double value of a pair.
fn value<Alg: floaty::Algorithm>(pair: Pair) -> DoubleDouble<Alg> {
    DoubleDouble::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
}

/// Runs an operation of `Gcc` under `behavior`.
fn gcc(operation: Operation, [x, y, z]: [Pair; 3], behavior: impl Behavior) -> Result {
    let [x, y, z] = [x, y, z].map(value::<Gcc>);
    outcome_of(match operation {
        Operation::Add => x.add_with(y, behavior),
        Operation::Sub => x.sub_with(y, behavior),
        Operation::Mul => x.mul_with(y, behavior),
        Operation::Div => x.div_with(y, behavior),
        Operation::Sqrt => x.sqrt_with(behavior),
        Operation::Remainder => x.remainder_with(y, behavior),
        Operation::TruncatedRemainder => x.truncated_remainder_with(y, behavior),
        Operation::MulAdd => x.mul_add_with(y, z, behavior),
        Operation::NextUp => x.next_up_with(behavior),
        Operation::NextDown => x.next_down_with(behavior),
    })
}

/// Runs an operation of `Qd` under `behavior`.
fn qd(operation: Operation, [x, y, _]: [Pair; 3], behavior: impl Behavior) -> Result {
    let [x, y] = [x, y].map(value::<Qd>);
    outcome_of(match operation {
        Operation::Add => x.add_with(y, behavior),
        Operation::Sub => x.sub_with(y, behavior),
        Operation::Mul => x.mul_with(y, behavior),
        Operation::Div => x.div_with(y, behavior),
        Operation::Sqrt => x.sqrt_with(behavior),
        Operation::Remainder => x.remainder_with(y, behavior),
        Operation::TruncatedRemainder => x.truncated_remainder_with(y, behavior),
        Operation::MulAdd | Operation::NextUp | Operation::NextDown => {
            unreachable!("QD has no {operation:?}")
        }
    })
}

/// Returns a binary64 value as an operand of the oracles.
fn operand(bits: u64) -> Operand<1> {
    Operand::of(F64::from_bits(bits))
}

/// Returns a binary64 result as the arithmetic oracle gives it: the value,
/// and the payload of a quiet NaN.
fn arithmetic_result(bits: u64) -> (Value, [u64; 1]) {
    let result = F64::from_bits(bits);
    let decoded = result.decode::<1>();
    let payload = match decoded {
        Decoded::Nan {
            signaling, payload, ..
        } => {
            assert!(!signaling, "{result:?}: a NaN result is quiet");
            payload
        }
        _ => [0],
    };
    (Value::from_decoded(decoded), payload)
}

/// Returns the expected result and flags of a truncation: `cvttsd2si`
/// toward zero without a denormal operand flag, then `cvtsi2sd`.
fn truncation(a: u64, env: &Env) -> (Outcome, Flags) {
    let (integer, flags) = to_int::<i64, 1>(&operand(a), &env.with_rounding(Rounding::TowardZero));
    let ToInt::Value(integer) = integer else {
        panic!("{a:#018x}: a truncated value is below 2^52");
    };
    let (result, conversion) = from_int(&integer, &BINARY64, env);
    (result, flags.difference(Flags::DENORMAL_INPUT) | conversion)
}

/// Checks one step with its oracle under the behavior of the step.
fn check_step(step: &Step) {
    let env = &step.env;
    let arithmetic = |operation: Arithmetic, operands: &[u64]| {
        let operands: Vec<Operand<1>> = operands.iter().map(|&bits| operand(bits)).collect();
        let expected = arithmetic::compute(operation, &operands, &BINARY64, env);
        let StepResult::Value(bits) = step.result else {
            panic!("{step:?}: an arithmetic step gives a value");
        };
        assert_eq!(
            (arithmetic_result(bits), step.flags),
            ((expected.value, expected.payload), expected.flags),
            "{step:?}"
        );
    };
    let order = |(order, flags): (Option<core::cmp::Ordering>, Flags)| {
        assert_eq!(
            (step.result, step.flags),
            (StepResult::Order(order), flags),
            "{step:?}"
        );
    };
    match step.operation {
        StepOperation::Add(a, b) => arithmetic(Arithmetic::Add, &[a, b]),
        StepOperation::Sub(a, b) => arithmetic(Arithmetic::Sub, &[a, b]),
        StepOperation::Mul(a, b) => arithmetic(Arithmetic::Mul, &[a, b]),
        StepOperation::Div(a, b) => arithmetic(Arithmetic::Div, &[a, b]),
        StepOperation::Sqrt(a) => arithmetic(Arithmetic::Sqrt, &[a]),
        StepOperation::MulAdd(a, b, c) => arithmetic(Arithmetic::MulAdd, &[a, b, c]),
        StepOperation::Truncate(a) => {
            let StepResult::Value(bits) = step.result else {
                panic!("{step:?}: a truncation gives a value");
            };
            let ours = (outcome(F64::from_bits(bits)), step.flags);
            assert_eq!(ours, truncation(a, env), "{step:?}");
        }
        StepOperation::CompareQuiet(a, b) => order(compare_quiet(&operand(a), &operand(b), env)),
        StepOperation::CompareSignaling(a, b) => {
            order(compare_signaling(&operand(a), &operand(b), env));
        }
    }
}

/// What the replay of one algorithm saw.
#[derive(Default)]
struct Coverage {
    steps: usize,
    flags: Flags,
    kinds: Vec<&'static str>,
}

impl Coverage {
    fn add(&mut self, step: &Step) {
        self.steps += 1;
        self.flags |= step.flags;
        let kind = match step.operation {
            StepOperation::Add(..) => "add",
            StepOperation::Sub(..) => "sub",
            StepOperation::Mul(..) => "mul",
            StepOperation::Div(..) => "div",
            StepOperation::Sqrt(..) => "sqrt",
            StepOperation::MulAdd(..) => "mul_add",
            StepOperation::Truncate(..) => "truncate",
            StepOperation::CompareQuiet(..) => "compare_quiet",
            StepOperation::CompareSignaling(..) => "compare_signaling",
        };
        if !self.kinds.contains(&kind) {
            self.kinds.push(kind);
        }
    }
}

/// Replays one operation of `algorithm` on one operand list under `env`, and
/// checks every step.
fn replay(
    algorithm: Algorithm,
    operation: Operation,
    operands: [Pair; 3],
    env: Env,
    coverage: &mut Coverage,
) {
    let context = || format!("{operation:?} {operands:?} {env:?}");
    let steps = RefCell::new(Vec::new());
    let observer = |step: &Step| steps.borrow_mut().push(*step);
    let result = algorithm.run(operation, operands, Observed::new(env, &observer));
    assert_eq!(
        result,
        algorithm.run(operation, operands, env),
        "observing changes nothing: {}",
        context()
    );
    let steps = steps.into_inner();
    let union = steps
        .iter()
        .fold(Flags::NONE, |flags, step| flags | step.flags);
    assert_eq!(result.2, union, "the flags of the steps: {}", context());
    let call = env.with_saturate(false);
    let nearest = call.with_rounding(Rounding::TiesToEven);
    for step in &steps {
        let allowed = step.env == call || (operation == Operation::MulAdd && step.env == nearest);
        assert!(
            allowed,
            "{step:?} runs under the behavior of the call: {}",
            context()
        );
        check_step(step);
        coverage.add(step);
    }
}

/// Replays every operation of an algorithm in every behavior.
fn replay_algorithm(algorithm: Algorithm, operations: &[Operation], seed: u64) -> Coverage {
    let mut random = SplitMix64::new(seed);
    let mut coverage = Coverage::default();
    for env in behaviors() {
        for _ in 0..CASES {
            let operands = [pair(&mut random), pair(&mut random), pair(&mut random)];
            for &operation in operations {
                replay(algorithm, operation, operands, env, &mut coverage);
            }
        }
    }
    coverage
}

/// Checks that the replay reached every kind of step of the algorithm and
/// every flag.
fn assert_covers(coverage: &Coverage, kinds: &[&str]) {
    let mut missing: Vec<&&str> = kinds
        .iter()
        .filter(|kind| !coverage.kinds.contains(kind))
        .collect();
    missing.sort();
    assert!(missing.is_empty(), "no step of the kinds {missing:?}");
    let all = Flags::INVALID
        | Flags::DIVIDE_BY_ZERO
        | Flags::OVERFLOW
        | Flags::UNDERFLOW
        | Flags::INEXACT
        | Flags::TINY
        | Flags::ROUNDED_UP
        | Flags::DENORMAL_INPUT;
    assert_eq!(coverage.flags, all, "some step signals each flag");
}

#[test]
fn every_step_of_gcc_follows_its_oracle() {
    let coverage = replay_algorithm(Algorithm::Gcc, &GCC, 0x6363_7265_706c_6179);
    assert_covers(
        &coverage,
        &[
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "mul_add",
            "compare_quiet",
        ],
    );
    println!("Gcc: {} steps", coverage.steps);
}

#[test]
fn every_step_of_qd_follows_its_oracle() {
    let coverage = replay_algorithm(Algorithm::Qd, &QD, 0x7164_7265_706c_6179);
    assert_covers(
        &coverage,
        &[
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "mul_add",
            "truncate",
            "compare_quiet",
            "compare_signaling",
        ],
    );
    println!("Qd: {} steps", coverage.steps);
}
