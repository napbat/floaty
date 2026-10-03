//! Compares the `_with` methods of the elementwise slice operations of
//! `Lanes` with the oracles of `floaty_verify` in every behavior of
//! `BEHAVIORS`: the arithmetic oracle for the sums, differences, products,
//! and quotients, and the oracles of the minimum and maximum, the rounding
//! to an integral value, and the conversion to an integer. Each value must
//! have the value and the NaN payload of the oracle, and the flags must be
//! the union of the flags of every value. A reduction must give the
//! documented order of the oracle steps, and a nested view the oracle of
//! each step in turn.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::cell::Cell;

use floaty::elementwise::{
    Abs, Difference, Maximum, MaximumNumber, Minimum, MinimumNumber, Product, Quotient,
    RoundToIntegral, Splat, Sum,
};
use floaty::{Env, F32, Flags, Lanes, Rounding, ToInt, Vector};
use floaty_verify::arithmetic::{self, Operation};
use floaty_verify::encodings::Layout;
use floaty_verify::kernels::{LENGTHS, Mix, integral_vector, pair};
use floaty_verify::mpfr::{Format, Operand, Specials, Value};
use floaty_verify::operations::compare::{self, MinMax};
use floaty_verify::operations::integral::{self, IntegerValue, to_int_value};
use floaty_verify::operations::{BEHAVIORS, Outcome};
use floaty_verify::random::SplitMix64;

/// The binary32 format of the values.
const SINGLE: Format = Format::of::<F32>(Specials::Ieee);

/// The binary32 lanes of the operations. The lane count does not change an
/// elementwise result.
type Single = Lanes<F32, 8>;

/// The rounding directions of `RoundToIntegral`.
const DIRECTIONS: [Rounding; 8] = [
    Rounding::TiesToEven,
    Rounding::TiesToAway,
    Rounding::TiesTowardZero,
    Rounding::TowardPositive,
    Rounding::TowardNegative,
    Rounding::TowardZero,
    Rounding::AwayFromZero,
    Rounding::ToOdd,
];

/// Returns the binary32 encoding of a result of the arithmetic oracle.
fn value_bits(value: &Value, payload: u64) -> u32 {
    let sign = |negative: bool| if negative { 0x8000_0000 } else { 0 };
    match value {
        Value::Zero { negative } => sign(*negative),
        Value::Infinity { negative } => sign(*negative) | 0x7F80_0000,
        Value::Nan { negative } => {
            sign(*negative) | 0x7FC0_0000 | u32::try_from(payload).expect("a binary32 payload")
        }
        Value::Finite(value) => value.to_f32().to_bits(),
    }
}

/// Returns the binary32 encoding of a result of an operation oracle.
fn outcome_bits(outcome: &Outcome) -> u32 {
    match outcome {
        Outcome::Zero { negative } => value_bits(
            &Value::Zero {
                negative: *negative,
            },
            0,
        ),
        Outcome::Infinity { negative } => value_bits(
            &Value::Infinity {
                negative: *negative,
            },
            0,
        ),
        Outcome::Finite(value) => value.to_f32().to_bits(),
        Outcome::Nan {
            negative,
            signaling,
            payload,
        } => {
            assert!(!signaling, "no expected result is signaling");
            let payload = payload.to_u64().expect("a binary32 payload");
            value_bits(
                &Value::Nan {
                    negative: *negative,
                },
                payload,
            )
        }
    }
}

/// Returns a binary32 encoding as an operand of the oracles.
fn operand(bits: u32) -> Operand<1> {
    Operand::of(F32::from_bits(bits))
}

/// The oracle of one call: the behavior, and the union of the flags of the
/// steps so far.
struct Oracle {
    env: Env,
    flags: Flags,
}

impl Oracle {
    /// Returns the encoding of one arithmetic step on two encodings.
    fn arithmetic(&mut self, operation: Operation, x: u32, y: u32) -> u32 {
        let operands = [operand(x), operand(y)];
        let expected = arithmetic::compute(operation, &operands, &SINGLE, &self.env);
        self.flags |= expected.flags;
        value_bits(&expected.value, expected.payload[0])
    }

    /// Returns the encoding of one minimum or maximum step.
    fn min_max(&mut self, operation: MinMax, x: u32, y: u32) -> u32 {
        let (x, y) = (operand(x), operand(y));
        let (outcome, flags) = compare::min_max(operation, &x, &y, &SINGLE, &self.env);
        self.flags |= flags;
        outcome_bits(&outcome)
    }

    /// Returns the encoding of `x` rounded to an integral value in the
    /// direction `rounding`.
    fn round(&mut self, x: u32, rounding: Rounding) -> u32 {
        let env = self.env.with_rounding(rounding);
        let (outcome, flags) = integral::round_to_integral(&operand(x), &SINGLE, &env);
        self.flags |= flags;
        outcome_bits(&outcome)
    }
}

/// A step of a view of two vectors, as the oracle computes it.
#[derive(Clone, Copy, Debug)]
enum Step {
    Arithmetic(Operation),
    MinMax(MinMax),
}

impl Step {
    /// Every step of a view of two vectors.
    const ALL: [Self; 8] = [
        Self::Arithmetic(Operation::Add),
        Self::Arithmetic(Operation::Sub),
        Self::Arithmetic(Operation::Mul),
        Self::Arithmetic(Operation::Div),
        Self::MinMax(MinMax::Minimum),
        Self::MinMax(MinMax::Maximum),
        Self::MinMax(MinMax::MinimumNumber),
        Self::MinMax(MinMax::MaximumNumber),
    ];

    /// Returns the encoding of the step on `x` and `y`.
    fn expected(self, oracle: &mut Oracle, x: u32, y: u32) -> u32 {
        match self {
            Self::Arithmetic(operation) => oracle.arithmetic(operation, x, y),
            Self::MinMax(operation) => oracle.min_max(operation, x, y),
        }
    }

    /// Stores the view of the step on `x` and `y` with the `_with` method,
    /// and returns the flags.
    fn store(self, x: &[F32], y: &[F32], out: &mut [F32], env: Env) -> Flags {
        match self {
            Self::Arithmetic(Operation::Add) => Single::store_with(Sum(x, y), out, env),
            Self::Arithmetic(Operation::Sub) => Single::store_with(Difference(x, y), out, env),
            Self::Arithmetic(Operation::Mul) => Single::store_with(Product(x, y), out, env),
            Self::Arithmetic(Operation::Div) => Single::store_with(Quotient(x, y), out, env),
            Self::MinMax(MinMax::Minimum) => Single::store_with(Minimum(x, y), out, env),
            Self::MinMax(MinMax::Maximum) => Single::store_with(Maximum(x, y), out, env),
            Self::MinMax(MinMax::MinimumNumber) => {
                Single::store_with(MinimumNumber(x, y), out, env)
            }
            Self::MinMax(MinMax::MaximumNumber) => {
                Single::store_with(MaximumNumber(x, y), out, env)
            }
            Self::Arithmetic(_) | Self::MinMax(_) => unreachable!("Step::ALL lists each step"),
        }
    }
}

/// Returns the encodings of values.
fn bits(values: &[F32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// Returns encodings as binary32 values.
fn singles(encodings: &[u128]) -> Vec<F32> {
    encodings
        .iter()
        .map(|&bits| F32::from_bits(u32::try_from(bits).expect("a binary32 encoding")))
        .collect()
}

/// Asserts that a store gives the expected encodings and flags.
fn assert_store(out: &[F32], flags: Flags, oracle: &Oracle, expected: &[u32], context: &str) {
    assert_eq!(
        (bits(out), flags),
        (expected.to_vec(), oracle.flags),
        "{context} {:?}",
        oracle.env
    );
}

#[test]
fn views_of_two_vectors_follow_the_oracle() {
    let mut random = SplitMix64::new(0x7669_6577);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let (a, b) = pair(Layout::BINARY32, mix, length, &mut random);
            let (x, y) = (singles(&a), singles(&b));
            let mut out = vec![F32::from_bits(0); length];
            for env in BEHAVIORS {
                for step in Step::ALL {
                    let mut oracle = Oracle {
                        env,
                        flags: Flags::NONE,
                    };
                    let expected: Vec<u32> = x
                        .iter()
                        .zip(&y)
                        .map(|(x, y)| step.expected(&mut oracle, x.to_bits(), y.to_bits()))
                        .collect();
                    let flags = step.store(&x, &y, &mut out, env);
                    let context = format!("{step:?} {mix:?} {length}");
                    assert_store(&out, flags, &oracle, &expected, &context);
                }
            }
        }
    }
}

#[test]
fn rounding_follows_the_oracle_in_each_direction() {
    let mut random = SplitMix64::new(0x726F_756E);
    for length in LENGTHS {
        let x: Vec<F32> = integral_vector(length, &mut random)
            .into_iter()
            .map(F32::from_bits)
            .collect();
        let mut out = vec![F32::from_bits(0); length];
        for env in BEHAVIORS {
            for rounding in DIRECTIONS {
                let mut oracle = Oracle {
                    env,
                    flags: Flags::NONE,
                };
                let expected: Vec<u32> = x
                    .iter()
                    .map(|x| oracle.round(x.to_bits(), rounding))
                    .collect();
                let view = RoundToIntegral {
                    values: &x[..],
                    rounding,
                };
                let flags = Single::store_with(view, &mut out[..], env);
                let context = format!("{rounding:?} {length}");
                assert_store(&out, flags, &oracle, &expected, &context);
            }
        }
    }
}

/// Checks `to_int_slice_with` into the integer type `I` on `x` in `env`.
fn check_to_int<I: IntegerValue + Copy>(x: &[F32], env: Env, context: &str) {
    let mut flags = Flags::NONE;
    let expected: Vec<_> = x
        .iter()
        .map(|&x| {
            let (integer, step_flags) = integral::to_int::<I, 1>(&Operand::of(x), &env);
            flags |= step_flags;
            integer
        })
        .collect();
    let mut out = vec![ToInt::<I>::Nan; x.len()];
    let ours_flags = Single::to_int_slice_with(x, &mut out, env);
    let ours: Vec<_> = out.into_iter().map(to_int_value).collect();
    assert_eq!(
        (ours, ours_flags),
        (expected, flags),
        "{} {context} {env:?}",
        core::any::type_name::<I>()
    );
}

#[test]
fn integer_conversion_follows_the_oracle() {
    let mut random = SplitMix64::new(0x696E_7473);
    for length in LENGTHS {
        let x: Vec<F32> = integral_vector(length, &mut random)
            .into_iter()
            .map(F32::from_bits)
            .collect();
        let (a, _) = pair(Layout::BINARY32, Mix::Edges, length, &mut random);
        let edges = singles(&a);
        for env in BEHAVIORS {
            for values in [&x, &edges] {
                let context = format!("{length}");
                check_to_int::<i8>(values, env, &context);
                check_to_int::<u8>(values, env, &context);
                check_to_int::<i32>(values, env, &context);
                check_to_int::<u64>(values, env, &context);
            }
        }
    }
}

/// A reduction of `Lanes`.
#[derive(Clone, Copy, Debug)]
struct Reduction(MinMax);

impl Reduction {
    /// Every reduction.
    const ALL: [Self; 4] = [
        Self(MinMax::Minimum),
        Self(MinMax::Maximum),
        Self(MinMax::MinimumNumber),
        Self(MinMax::MaximumNumber),
    ];

    /// Returns the result of the reduction with `lanes` lanes in the
    /// documented order, and the flags.
    fn expected(self, x: &[F32], lanes: usize, env: Env) -> (u32, Flags) {
        let mut oracle = Oracle {
            env,
            flags: Flags::NONE,
        };
        let start = match self.0 {
            MinMax::Minimum | MinMax::MinimumNumber => 0x7F80_0000,
            _ => 0xFF80_0000,
        };
        let mut sums = vec![start; lanes];
        for (index, value) in x.iter().enumerate() {
            let lane = index % lanes;
            sums[lane] = oracle.min_max(self.0, sums[lane], value.to_bits());
        }
        let mut half = lanes / 2;
        while half > 0 {
            for index in 0..half {
                sums[index] = oracle.min_max(self.0, sums[index], sums[index + half]);
            }
            half /= 2;
        }
        (sums[0], oracle.flags)
    }

    /// Returns the result of the `_with` method with `N` lanes.
    fn ours<const N: usize>(self, x: impl Vector, env: Env) -> (F32, Flags) {
        type Reduce<const N: usize> = Lanes<F32, N>;
        match self.0 {
            MinMax::Minimum => Reduce::<N>::minimum_of_with(x, env),
            MinMax::Maximum => Reduce::<N>::maximum_of_with(x, env),
            MinMax::MinimumNumber => Reduce::<N>::minimum_number_of_with(x, env),
            _ => Reduce::<N>::maximum_number_of_with(x, env),
        }
    }

    /// Checks the reduction with `N` lanes on `x` in `env`.
    fn check<const N: usize>(self, x: &[F32], env: Env, context: &str) {
        let (result, flags) = self.ours::<N>(x, env);
        assert_eq!(
            (result.to_bits(), flags),
            self.expected(x, N, env),
            "{self:?} N = {N} {context} {env:?}"
        );
    }
}

#[test]
fn reductions_follow_the_order() {
    let mut random = SplitMix64::new(0x7265_6475);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let (a, _) = pair(Layout::BINARY32, mix, length, &mut random);
            let x = singles(&a);
            let context = format!("{mix:?} {length}");
            for env in BEHAVIORS {
                for reduction in Reduction::ALL {
                    reduction.check::<1>(&x, env, &context);
                    reduction.check::<2>(&x, env, &context);
                    reduction.check::<8>(&x, env, &context);
                    reduction.check::<32>(&x, env, &context);
                }
            }
        }
    }
}

/// Checks an exact in-place update and a disjoint update in one allocation
/// against the same MPFR steps.
fn check_cell_update(x: &[F32], y: &[F32], keep: F32, eta: F32, oracle: &Oracle, expected: &[u32]) {
    let count = x.len();
    let mut in_place = x.to_vec();
    let cells = Cell::from_mut(&mut in_place[..]).as_slice_of_cells();
    let view = Sum(
        Product(cells, Splat::new(keep, count)),
        Product(y, Splat::new(eta, count)),
    );
    let flags = Single::store_with(view, cells, oracle.env);
    assert_store(
        &in_place,
        flags,
        oracle,
        expected,
        &format!("in-place update {count}"),
    );

    let mut disjoint: Vec<_> = x.iter().chain(y).copied().collect();
    let cells = Cell::from_mut(&mut disjoint[..]).as_slice_of_cells();
    let (source, destination) = cells.split_at(count);
    let view = Sum(
        Product(source, Splat::new(keep, count)),
        Product(destination, Splat::new(eta, count)),
    );
    let flags = Single::store_with(view, destination, oracle.env);
    assert_store(
        &disjoint[count..],
        flags,
        oracle,
        expected,
        &format!("disjoint update {count}"),
    );
}

#[test]
fn nested_views_follow_each_step() {
    let mut random = SplitMix64::new(0x6E65_7374);
    let levels = F32::from_bits(0x437F_0000); // 255
    let (zero, one) = (F32::from_bits(0), F32::from_bits(0x3F80_0000));
    for mix in [Mix::Near, Mix::Edges, Mix::Zeros] {
        for length in LENGTHS {
            let (first, second) = pair(Layout::BINARY32, mix, length, &mut random);
            let (x, y) = (singles(&first), singles(&second));
            let (keep, eta) = (F32::from_bits(0x3F7F_0000), F32::from_bits(0x3B80_0000));
            let mut out = vec![F32::from_bits(0); length];
            for env in BEHAVIORS {
                // A k-means update: x * keep + y * eta, each step rounded.
                let mut oracle = Oracle {
                    env,
                    flags: Flags::NONE,
                };
                let expected: Vec<u32> = x
                    .iter()
                    .zip(&y)
                    .map(|(left, right)| {
                        let kept =
                            oracle.arithmetic(Operation::Mul, left.to_bits(), keep.to_bits());
                        let moved =
                            oracle.arithmetic(Operation::Mul, right.to_bits(), eta.to_bits());
                        oracle.arithmetic(Operation::Add, kept, moved)
                    })
                    .collect();
                let view = Sum(
                    Product(&x[..], Splat::new(keep, length)),
                    Product(&y[..], Splat::new(eta, length)),
                );
                let flags = Single::store_with(view, &mut out[..], env);
                assert_store(&out, flags, &oracle, &expected, &format!("update {length}"));
                check_cell_update(&x, &y, keep, eta, &oracle, &expected);
                // A scalar quantization: |x| / y clamped to [0, 1], times
                // 255, rounded to an integral value, a tie away from zero.
                let mut oracle = Oracle {
                    env,
                    flags: Flags::NONE,
                };
                let expected: Vec<u32> = x
                    .iter()
                    .zip(&y)
                    .map(|(left, right)| {
                        let position = oracle.arithmetic(
                            Operation::Div,
                            left.to_bits() & 0x7FFF_FFFF,
                            right.to_bits(),
                        );
                        let low = oracle.min_max(MinMax::Maximum, position, zero.to_bits());
                        let unit = oracle.min_max(MinMax::MinimumNumber, low, one.to_bits());
                        let scaled = oracle.arithmetic(Operation::Mul, unit, levels.to_bits());
                        oracle.round(scaled, Rounding::TiesToAway)
                    })
                    .collect();
                let view = RoundToIntegral {
                    values: Product(
                        MinimumNumber(
                            Maximum(Quotient(Abs(&x[..]), &y[..]), Splat::new(zero, length)),
                            Splat::new(one, length),
                        ),
                        Splat::new(levels, length),
                    ),
                    rounding: Rounding::TiesToAway,
                };
                let flags = Single::store_with(view, &mut out[..], env);
                assert_store(
                    &out,
                    flags,
                    &oracle,
                    &expected,
                    &format!("quantize {length}"),
                );
            }
        }
    }
}
