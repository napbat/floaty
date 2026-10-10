//! Blocks: `map` and `evaluate` of `F16`, `BF16`, `F32`, and `F64` give the
//! results of the steps of each chain one at a time, through their `_with`
//! methods in the default mode. The chains take every step of `Steps`, each
//! rounding direction, and NaNs that the host can hold with other bits than
//! the engine, and that a later step drops. The operands are every pair of
//! the boundary encodings of the format and random pairs, with boundary and
//! random parameters, in slices of the type and of its host type and of
//! each length. The steps of one value also take every 16-bit encoding, and
//! in the wider formats values next to integers and ties. The report of
//! `map` counts each lane, and in the engine at least each lane with a NaN
//! result, and exactly those where no step needs the engine. On x86-64, a
//! block under each MXCSR control of the host-path tests gives the results
//! of the engine, and no unmasked exception traps: the check of the
//! environment comes before every step.

mod width;

use std::fmt::Debug;

use floaty::block::{Chain, Steps};
use floaty::{BF16, Env, F16, F32, F64, HostPath, Rounding};
use floaty_verify::random::SplitMix64;

use self::width::Width;

/// The steps one at a time take the default mode of `F32` and `F64`.
const IEEE: Env = Env::IEEE;

/// The steps of a chain one at a time, from its values and parameter.
type OneAtATime<'a, T> = &'a dyn Fn(T, T, T) -> T;

/// `(x + y) * x - y / x`: the four operators.
#[derive(Debug)]
struct Arithmetic;

impl Chain<2, 1> for Arithmetic {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [_]: [S; 1]) -> S {
        (x + y) * x - y / x
    }
}

/// `Arithmetic` one step at a time.
fn arithmetic<T: Width>(x: T, y: T, _p: T) -> T {
    let product = x.add_with(y, IEEE).0.mul_with(x, IEEE).0;
    product.sub_with(y.div_with(x, IEEE).0, IEEE).0
}

/// `-sqrt(|x * y + p|)`: the fused multiply-add, the absolute value, the
/// square root, and the negation.
#[derive(Debug)]
struct Fused;

impl Chain<2, 1> for Fused {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [p]: [S; 1]) -> S {
        -x.mul_add(y, p).abs().sqrt()
    }
}

/// `Fused` one step at a time.
fn fused<T: Width>(x: T, y: T, p: T) -> T {
    -x.mul_add_with(y, p, IEEE).0.abs().sqrt_with(IEEE).0
}

/// `maximum(minimum_number(maximum_number(minimum(x, y), p), -x), y)`: the
/// four minimum and maximum operations, with a NaN that each kind keeps or
/// drops.
#[derive(Debug)]
struct Order;

impl Chain<2, 1> for Order {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [p]: [S; 1]) -> S {
        x.minimum(y).maximum_number(p).minimum_number(-x).maximum(y)
    }
}

/// `Order` one step at a time.
fn order<T: Width>(x: T, y: T, p: T) -> T {
    let low = x.minimum_with(y, IEEE).0.maximum_number_with(p, IEEE).0;
    low.minimum_number_with(-x, IEEE).0.maximum_with(y, IEEE).0
}

/// `minimum_number(maximum_number(x / y, x * y + p), p)`: number operations
/// that drop the NaN of a quotient, and the fused result that a build
/// without FMA computes in the engine.
#[derive(Debug)]
struct Dropped;

impl Chain<2, 1> for Dropped {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [p]: [S; 1]) -> S {
        (x / y).maximum_number(x.mul_add(y, p)).minimum_number(p)
    }
}

/// `Dropped` one step at a time.
fn dropped<T: Width>(x: T, y: T, p: T) -> T {
    let quotient = x.div_with(y, IEEE).0;
    let larger = quotient
        .maximum_number_with(x.mul_add_with(y, p, IEEE).0, IEEE)
        .0;
    larger.minimum_number_with(p, IEEE).0
}

/// A minimum or maximum step.
#[derive(Clone, Copy, Debug)]
enum Pick {
    Minimum,
    Maximum,
    MinimumNumber,
    MaximumNumber,
    MinNum,
    MaxNum,
    MinimumMagnitude,
    MaximumMagnitude,
    MinimumMagnitudeNumber,
    MaximumMagnitudeNumber,
}

impl Pick {
    /// Every minimum and maximum step.
    const ALL: [Self; 10] = [
        Self::Minimum,
        Self::Maximum,
        Self::MinimumNumber,
        Self::MaximumNumber,
        Self::MinNum,
        Self::MaxNum,
        Self::MinimumMagnitude,
        Self::MaximumMagnitude,
        Self::MinimumMagnitudeNumber,
        Self::MaximumMagnitudeNumber,
    ];

    /// Returns the step of `x` and `y`.
    fn apply<S: Steps>(self, x: S, y: S) -> S {
        match self {
            Self::Minimum => x.minimum(y),
            Self::Maximum => x.maximum(y),
            Self::MinimumNumber => x.minimum_number(y),
            Self::MaximumNumber => x.maximum_number(y),
            Self::MinNum => x.min_num(y),
            Self::MaxNum => x.max_num(y),
            Self::MinimumMagnitude => x.minimum_magnitude(y),
            Self::MaximumMagnitude => x.maximum_magnitude(y),
            Self::MinimumMagnitudeNumber => x.minimum_magnitude_number(y),
            Self::MaximumMagnitudeNumber => x.maximum_magnitude_number(y),
        }
    }

    /// Returns the step of `x` and `y` through its `_with` method.
    fn with<T: Width>(self, x: T, y: T) -> T {
        let (result, _) = match self {
            Self::Minimum => x.minimum_with(y, IEEE),
            Self::Maximum => x.maximum_with(y, IEEE),
            Self::MinimumNumber => x.minimum_number_with(y, IEEE),
            Self::MaximumNumber => x.maximum_number_with(y, IEEE),
            Self::MinNum => x.min_num_with(y, IEEE),
            Self::MaxNum => x.max_num_with(y, IEEE),
            Self::MinimumMagnitude => x.minimum_magnitude_with(y, IEEE),
            Self::MaximumMagnitude => x.maximum_magnitude_with(y, IEEE),
            Self::MinimumMagnitudeNumber => x.minimum_magnitude_number_with(y, IEEE),
            Self::MaximumMagnitudeNumber => x.maximum_magnitude_number_with(y, IEEE),
        };
        result
    }
}

/// A chain of one step, on the values `x` and `y` and the parameter `p`.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// `copy_sign(x, y / p)`: the sign of a quotient, which the host can
    /// make a NaN of another sign than the engine.
    CopySign,
    NextUp,
    NextDown,
    /// `round_to_integral(x)`.
    Integral,
    /// `round_to_integral_by(x, rounding)`.
    IntegralBy(Rounding),
    /// `scale_b(x, scale)`.
    ScaleB(i32),
    LogB,
    /// `remainder(x, y)`.
    Remainder,
    /// `truncated_remainder(x, y)`.
    TruncatedRemainder,
    Exp,
    ExpM1,
    Exp2,
    Exp2M1,
    Exp10,
    Exp10M1,
    Log,
    Log2,
    Log10,
    LogP1,
    Log2P1,
    Log10P1,
    Sinh,
    Cosh,
    Tanh,
    Asinh,
    Acosh,
    Atanh,
    /// `compound(x, n)`.
    Compound(i64),
    /// `hypot(x, y)`.
    Hypot,
    /// `pown(x, n)`.
    Pown(i64),
    /// `rootn(x, n)`.
    Rootn(i64),
    ReciprocalSqrt,
    /// `pick(x, y)`.
    Pick(Pick),
    /// `maximum_number(pick(minimum_number(x, y), y / p), p)`: the NaN of
    /// `minimum_number` of two NaNs and the NaN of a quotient, which the
    /// host can hold with other bits than the engine, and a step that drops
    /// a NaN result.
    PickNan(Pick),
}

impl Chain<2, 1> for Step {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [p]: [S; 1]) -> S {
        match *self {
            Self::CopySign => x.copy_sign(y / p),
            Self::NextUp => x.next_up(),
            Self::NextDown => x.next_down(),
            Self::Integral => x.round_to_integral(),
            Self::IntegralBy(rounding) => x.round_to_integral_by(rounding),
            Self::ScaleB(scale) => x.scale_b(scale),
            Self::LogB => x.log_b(),
            Self::Remainder => x.remainder(y),
            Self::TruncatedRemainder => x.truncated_remainder(y),
            Self::Exp => x.exp(),
            Self::ExpM1 => x.exp_m1(),
            Self::Exp2 => x.exp2(),
            Self::Exp2M1 => x.exp2_m1(),
            Self::Exp10 => x.exp10(),
            Self::Exp10M1 => x.exp10_m1(),
            Self::Log => x.log(),
            Self::Log2 => x.log2(),
            Self::Log10 => x.log10(),
            Self::LogP1 => x.log_p1(),
            Self::Log2P1 => x.log2_p1(),
            Self::Log10P1 => x.log10_p1(),
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            Self::Asinh => x.asinh(),
            Self::Acosh => x.acosh(),
            Self::Atanh => x.atanh(),
            Self::Compound(n) => x.compound(n),
            Self::Hypot => x.hypot(y),
            Self::Pown(n) => x.pown(n),
            Self::Rootn(n) => x.rootn(n),
            Self::ReciprocalSqrt => x.reciprocal_sqrt(),
            Self::Pick(pick) => pick.apply(x, y),
            Self::PickNan(pick) => pick.apply(x.minimum_number(y), y / p).maximum_number(p),
        }
    }
}

impl Step {
    /// Returns the steps of the chain one at a time.
    fn one_at_a_time<T: Width>(self, x: T, y: T, p: T) -> T {
        match self {
            Self::CopySign => x.copy_sign(y.div_with(p, IEEE).0),
            Self::NextUp => x.next_up_with(IEEE).0,
            Self::NextDown => x.next_down_with(IEEE).0,
            Self::Integral => x.round_to_integral_with(IEEE).0,
            Self::IntegralBy(rounding) => x.round_to_integral_with(IEEE.with_rounding(rounding)).0,
            Self::ScaleB(scale) => x.scale_b_with(scale, IEEE).0,
            Self::LogB => x.log_b_with(IEEE).0,
            Self::Remainder => x.remainder_with(y, IEEE).0,
            Self::TruncatedRemainder => x.truncated_remainder_with(y, IEEE).0,
            Self::Exp => x.exp_with(IEEE).0,
            Self::ExpM1 => x.exp_m1_with(IEEE).0,
            Self::Exp2 => x.exp2_with(IEEE).0,
            Self::Exp2M1 => x.exp2_m1_with(IEEE).0,
            Self::Exp10 => x.exp10_with(IEEE).0,
            Self::Exp10M1 => x.exp10_m1_with(IEEE).0,
            Self::Log => x.log_with(IEEE).0,
            Self::Log2 => x.log2_with(IEEE).0,
            Self::Log10 => x.log10_with(IEEE).0,
            Self::LogP1 => x.log_p1_with(IEEE).0,
            Self::Log2P1 => x.log2_p1_with(IEEE).0,
            Self::Log10P1 => x.log10_p1_with(IEEE).0,
            Self::Sinh => x.sinh_with(IEEE).0,
            Self::Cosh => x.cosh_with(IEEE).0,
            Self::Tanh => x.tanh_with(IEEE).0,
            Self::Asinh => x.asinh_with(IEEE).0,
            Self::Acosh => x.acosh_with(IEEE).0,
            Self::Atanh => x.atanh_with(IEEE).0,
            Self::Compound(n) => x.compound_with(n, IEEE).0,
            Self::Hypot => x.hypot_with(y, IEEE).0,
            Self::Pown(n) => x.pown_with(n, IEEE).0,
            Self::Rootn(n) => x.rootn_with(n, IEEE).0,
            Self::ReciprocalSqrt => x.reciprocal_sqrt_with(IEEE).0,
            Self::Pick(pick) => pick.with(x, y),
            Self::PickNan(pick) => {
                let first = x.minimum_number_with(y, IEEE).0;
                let picked = pick.with(first, y.div_with(p, IEEE).0);
                picked.maximum_number_with(p, IEEE).0
            }
        }
    }

    /// Returns `true` when the chain reads its parameter.
    fn reads_parameter(self) -> bool {
        matches!(self, Self::CopySign | Self::PickNan(_))
    }
}

/// Every rounding direction.
const ROUNDINGS: [Rounding; 8] = [
    Rounding::TiesToEven,
    Rounding::TiesToAway,
    Rounding::TiesTowardZero,
    Rounding::TowardPositive,
    Rounding::TowardNegative,
    Rounding::TowardZero,
    Rounding::AwayFromZero,
    Rounding::ToOdd,
];

/// Returns the chains of one step of `T`: each step, each rounding
/// direction, the scales of `T`, and each minimum and maximum step with
/// numbers and with NaNs.
fn steps<T: Width>() -> Vec<Step> {
    let mut steps = vec![
        Step::CopySign,
        Step::NextUp,
        Step::NextDown,
        Step::Integral,
        Step::LogB,
        Step::Remainder,
        Step::TruncatedRemainder,
        Step::Exp,
        Step::ExpM1,
        Step::Exp2,
        Step::Exp2M1,
        Step::Exp10,
        Step::Exp10M1,
        Step::Log,
        Step::Log2,
        Step::Log10,
        Step::LogP1,
        Step::Log2P1,
        Step::Log10P1,
        Step::Sinh,
        Step::Cosh,
        Step::Tanh,
        Step::Asinh,
        Step::Acosh,
        Step::Atanh,
        Step::Hypot,
        Step::ReciprocalSqrt,
    ];
    steps.extend(ROUNDINGS.map(Step::IntegralBy));
    steps.extend(T::SCALES.map(Step::ScaleB));
    steps.extend(
        [0, 3]
            .into_iter()
            .flat_map(|n| [Step::Compound(n), Step::Pown(n), Step::Rootn(n)]),
    );
    steps.extend(
        Pick::ALL
            .into_iter()
            .flat_map(|pick| [Step::Pick(pick), Step::PickNan(pick)]),
    );
    steps
}

/// Checks the block of `chain` against `steps`, one step at a time, for
/// each pair of `pairs` and the parameter `p`: `map` over slices of `T`
/// and of its host type, of every length up to the pairs, and `evaluate`
/// of each pair.
fn check<T: Width, C: Chain<2, 1> + Debug>(
    chain: &C,
    steps: OneAtATime<'_, T>,
    pairs: &[(T, T)],
    p: T,
) {
    let x: Vec<T> = pairs.iter().map(|&(x, _)| x).collect();
    let y: Vec<T> = pairs.iter().map(|&(_, y)| y).collect();
    let expected: Vec<u64> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| steps(x, y, p).bits())
        .collect();
    let context = |index: usize| {
        format!(
            "{chain:?} {:#x} {:#x} {:#x}",
            x[index].bits(),
            y[index].bits(),
            p.bits()
        )
    };
    for length in [0, 1, 2, 7, 8, 9, 31, 32, 33, 100, pairs.len()] {
        let length = length.min(pairs.len());
        let mut out = vec![p; length];
        let report = T::map(chain, [&x[..length], &y[..length]], [p], &mut out[..]);
        for (index, result) in out.iter().enumerate() {
            assert_eq!(result.bits(), expected[index], "map {}", context(index));
        }
        // The engine computes every lane whose result is a NaN.
        let nan_count = out.iter().filter(|&&result| result.is_nan()).count();
        assert_eq!(report.lane_count(), length, "{chain:?} lanes");
        assert!(
            (nan_count..=length).contains(&report.engine_lane_count()),
            "{chain:?} engine lanes {} of {length}, {nan_count} NaN",
            report.engine_lane_count()
        );
    }
    let hosts =
        |values: &[T]| -> Vec<T::Host> { values.iter().map(|&value| value.into()).collect() };
    let (host_x, host_y) = (hosts(&x), hosts(&y));
    let mut host_out = vec![T::Host::from(p); pairs.len()];
    T::map(chain, [&host_x[..], &host_y[..]], [p], &mut host_out[..]);
    for (index, &result) in host_out.iter().enumerate() {
        assert_eq!(
            T::from(result).bits(),
            expected[index],
            "map host {}",
            context(index)
        );
    }
    for (index, (&x, &y)) in x.iter().zip(&y).enumerate() {
        let result = T::evaluate(chain, [x, y], [p]);
        assert_eq!(
            result.bits(),
            expected[index],
            "evaluate {}",
            context(index)
        );
    }
}

/// Checks the chains of the four kinds of steps in `T`.
fn check_chains<T: Width>(seed: u64) {
    let mut random = SplitMix64::new(seed);
    let pairs = T::pairs(&mut random);
    for p in T::parameters(&mut random) {
        check(&Arithmetic, &arithmetic, &pairs, p);
        check(&Fused, &fused, &pairs, p);
        check(&Order, &order, &pairs, p);
        check(&Dropped, &dropped, &pairs, p);
    }
}

/// Checks the chains of one step in `T`, with the values of one operand of
/// `T` for the steps of one value.
fn check_value_steps<T: Width>(seed: u64) {
    let mut random = SplitMix64::new(seed);
    let mut pairs = T::pairs(&mut random);
    pairs.extend(T::unary_values().into_iter().map(|value| (value, value)));
    let parameters = T::parameters(&mut random);
    for step in steps::<T>() {
        let count = if step.reads_parameter() {
            parameters.len()
        } else {
            1
        };
        for &p in &parameters[..count] {
            check(&step, &|x, y, p| step.one_at_a_time(x, y, p), &pairs, p);
        }
    }
}

#[test]
fn blocks_give_the_steps_one_at_a_time() {
    check_chains::<F16>(0xB10C);
    check_chains::<BF16>(0xB10C);
    check_chains::<F32>(0xB10C);
    check_chains::<F64>(0xB10C);
}

#[test]
fn value_steps_give_the_steps_one_at_a_time() {
    check_value_steps::<F16>(0x57E9);
    check_value_steps::<BF16>(0x57E9);
    check_value_steps::<F32>(0x57E9);
    check_value_steps::<F64>(0x57E9);
}

/// Checks the counts of the report of `map` in `T`. The arithmetic chain has
/// no step that the engine must compute, so where the host paths run the
/// engine computes exactly the lanes whose result is a NaN, and elsewhere
/// every lane. No instruction computes `exp`, so the engine computes every
/// lane of that chain.
fn check_report<T: Width>(seed: u64) {
    let mut random = SplitMix64::new(seed);
    let pairs = T::pairs(&mut random);
    let x: Vec<T> = pairs.iter().map(|&(x, _)| x).collect();
    let y: Vec<T> = pairs.iter().map(|&(_, y)| y).collect();
    for p in T::parameters(&mut random) {
        let mut out = vec![p; pairs.len()];
        let report = T::map(&Arithmetic, [&x[..], &y[..]], [p], &mut out[..]);
        let nan_count = out.iter().filter(|&&result| result.is_nan()).count();
        let expected = if T::host_path() == HostPath::Ready {
            nan_count
        } else {
            pairs.len()
        };
        assert_eq!(report.lane_count(), pairs.len(), "{p:?}");
        assert_eq!(report.engine_lane_count(), expected, "arithmetic {p:?}");
        let report = T::map(&Step::Exp, [&x[..], &y[..]], [p], &mut out[..]);
        assert_eq!(report.engine_lane_count(), pairs.len(), "exp {p:?}");
    }
}

#[test]
fn map_reports_the_lanes_that_the_engine_computed() {
    check_report::<F16>(0x4E90);
    check_report::<BF16>(0x4E90);
    check_report::<F32>(0x4E90);
    check_report::<F64>(0x4E90);
}

#[test]
#[should_panic(expected = "each input holds one value for each output")]
fn a_block_rejects_slices_of_other_lengths() {
    let (x, y) = ([F32::from_bits(0); 3], [F32::from_bits(0); 2]);
    let mut out = [F32::from_bits(0); 3];
    F32::map(
        &Arithmetic,
        [&x[..], &y[..]],
        [F32::from_bits(0)],
        &mut out[..],
    );
}

/// Checks that each chain, under each MXCSR control of the host-path tests,
/// gives the results of the engine on operands that raise every exception.
/// A step before the check of MXCSR would trap under an unmasked exception.
#[cfg(target_arch = "x86_64")]
#[test]
fn blocks_read_mxcsr_before_their_steps() {
    mxcsr_comes_first::<F16>();
    mxcsr_comes_first::<BF16>();
    mxcsr_comes_first::<F32>();
    mxcsr_comes_first::<F64>();
}

/// Checks the chains of `blocks_read_mxcsr_before_their_steps` in `T`.
#[cfg(target_arch = "x86_64")]
fn mxcsr_comes_first<T: Width>() {
    let pairs = T::exception_pairs();
    let x = pairs.map(|(x, _)| x);
    let y = pairs.map(|(_, y)| y);
    let p = T::minus_one();
    under_each_control(&Arithmetic, &arithmetic, &x, &y, p);
    under_each_control(&Fused, &fused, &x, &y, p);
    under_each_control(&Order, &order, &x, &y, p);
    under_each_control(&Dropped, &dropped, &x, &y, p);
    let steps = [
        Step::CopySign,
        Step::NextUp,
        Step::Integral,
        Step::IntegralBy(Rounding::TiesToAway),
        // A scale that makes a subnormal value of the host type underflow.
        Step::ScaleB(-126),
        Step::LogB,
        Step::Exp,
        Step::Pick(Pick::MinimumMagnitude),
        Step::PickNan(Pick::MinNum),
    ];
    for step in steps {
        under_each_control(&step, &|x, y, p| step.one_at_a_time(x, y, p), &x, &y, p);
    }
}

/// Checks `map` of `chain` and `evaluate` of its first lane against
/// `steps`, under each MXCSR control of the host-path tests.
#[cfg(target_arch = "x86_64")]
fn under_each_control<T: Width, C: Chain<2, 1> + Debug>(
    chain: &C,
    steps: OneAtATime<'_, T>,
    x: &[T; 10],
    y: &[T; 10],
    p: T,
) {
    use floaty_verify::x86::{MXCSR_HOST_PATH_CONTROLS, with_mxcsr};

    for control in MXCSR_HOST_PATH_CONTROLS {
        let mut out = [p; 10];
        let first = with_mxcsr(control, || {
            T::map(chain, [&x[..], &y[..]], [p], &mut out[..]);
            T::evaluate(chain, [x[0], y[0]], [p])
        });
        assert_eq!(
            first.bits(),
            steps(x[0], y[0], p).bits(),
            "{chain:?} evaluate under {control:#x}"
        );
        for (index, result) in out.iter().enumerate() {
            let expected = steps(x[index], y[index], p);
            assert_eq!(
                result.bits(),
                expected.bits(),
                "{chain:?} {:#x} {:#x} under {control:#x}",
                x[index].bits(),
                y[index].bits()
            );
        }
    }
}
