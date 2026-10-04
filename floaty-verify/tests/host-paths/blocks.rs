//! Blocks: `F32::map` and `F32::evaluate` give the results of the steps of
//! each chain one at a time, through their `_with` methods in the default
//! mode. The chains take every step of `Steps`, each rounding direction,
//! and NaNs that the host can hold with other bits than the engine, and that
//! a later step drops. The operands are every pair of the boundary encodings
//! of binary32, random pairs, and values next to integers and ties, with
//! boundary and random parameters, in slices of `F32` and of `f32` and of
//! each length. On x86-64, a block under each MXCSR control of the host-path
//! tests gives the results of the engine, and no unmasked exception traps:
//! the check of the environment comes before every step.

use std::fmt::Debug;

use floaty::block::{Chain, Steps};
use floaty::{Env, F32, Rounding};
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;

use super::operands::{boundary_pairs, random_pairs};

/// The steps one at a time take the default mode of `F32`.
const IEEE: Env = Env::IEEE;

/// The steps of a chain one at a time, from its values and parameter.
type OneAtATime<'a> = &'a dyn Fn(F32, F32, F32) -> F32;

/// `(x + y) * x - y / x`: the four operators.
#[derive(Debug)]
struct Arithmetic;

impl Chain<2, 1> for Arithmetic {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [_]: [S; 1]) -> S {
        (x + y) * x - y / x
    }
}

/// `Arithmetic` one step at a time.
fn arithmetic(x: F32, y: F32, _p: F32) -> F32 {
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
fn fused(x: F32, y: F32, p: F32) -> F32 {
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
fn order(x: F32, y: F32, p: F32) -> F32 {
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
fn dropped(x: F32, y: F32, p: F32) -> F32 {
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
    fn with(self, x: F32, y: F32) -> F32 {
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
    Log,
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
            Self::Log => x.log(),
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
    fn one_at_a_time(self, x: F32, y: F32, p: F32) -> F32 {
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
            Self::Log => x.log_with(IEEE).0,
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

/// Returns the chains of one step: each step, each rounding direction,
/// scales at both ends of the binary32 powers of two and past them, and
/// each minimum and maximum step with numbers and with NaNs.
fn steps() -> Vec<Step> {
    let mut steps = vec![
        Step::CopySign,
        Step::NextUp,
        Step::NextDown,
        Step::Integral,
        Step::LogB,
        Step::Remainder,
        Step::TruncatedRemainder,
        Step::Exp,
        Step::Log,
        Step::Hypot,
        Step::ReciprocalSqrt,
    ];
    steps.extend(ROUNDINGS.map(Step::IntegralBy));
    steps.extend(
        [
            i32::MIN,
            -300,
            -150,
            -127,
            -126,
            -1,
            0,
            1,
            127,
            128,
            277,
            i32::MAX,
        ]
        .map(Step::ScaleB),
    );
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

/// Returns pairs of values next to an integer or a tie, of both signs, for
/// the integral steps: quarters, halves and their neighbors, and values
/// next to 2^23, from which every value is integral.
fn integral_pairs() -> Vec<(u32, u32)> {
    let edges: [u32; 14] = [
        0x3E80_0000,
        0x3EFF_FFFF,
        0x3F00_0000,
        0x3F00_0001,
        0x3F40_0000,
        0x3F7F_FFFF,
        0x3FC0_0000,
        0x4020_0000,
        0x4060_0000,
        0x4A80_0001,
        0x4AFF_FFFE,
        0x4AFF_FFFF,
        0x4B00_0000,
        0x4B00_0001,
    ];
    edges
        .into_iter()
        .flat_map(|bits| [bits, bits | 0x8000_0000])
        .map(|x| (x, 0x3F80_0000))
        .collect()
}

/// Checks the block of `chain` against `steps`, one step at a time, for
/// each pair of `pairs` and the parameter `p`: `map` over slices of `F32`
/// and of `f32`, of every length up to the pairs, and `evaluate` of each
/// pair.
fn check<C: Chain<2, 1> + Debug>(chain: &C, steps: OneAtATime<'_>, pairs: &[(u32, u32)], p: F32) {
    let x: Vec<F32> = pairs.iter().map(|&(x, _)| F32::from_bits(x)).collect();
    let y: Vec<F32> = pairs.iter().map(|&(_, y)| F32::from_bits(y)).collect();
    let expected: Vec<u32> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| steps(x, y, p).to_bits())
        .collect();
    let context = |index: usize| {
        format!(
            "{chain:?} {:#x} {:#x} {:#x}",
            x[index].to_bits(),
            y[index].to_bits(),
            p.to_bits()
        )
    };
    for length in [0, 1, 2, 7, 8, 9, 31, 32, 33, 100, pairs.len()] {
        let length = length.min(pairs.len());
        let mut out = vec![F32::from_bits(0); length];
        F32::map(chain, [&x[..length], &y[..length]], [p], &mut out[..]);
        for (index, result) in out.iter().enumerate() {
            assert_eq!(result.to_bits(), expected[index], "map {}", context(index));
        }
    }
    let hosts =
        |values: &[F32]| -> Vec<f32> { values.iter().map(|&value| f32::from(value)).collect() };
    let (host_x, host_y) = (hosts(&x), hosts(&y));
    let mut host_out = vec![0.0_f32; pairs.len()];
    F32::map(chain, [&host_x[..], &host_y[..]], [p], &mut host_out[..]);
    for (index, result) in host_out.iter().enumerate() {
        assert_eq!(
            result.to_bits(),
            expected[index],
            "map f32 {}",
            context(index)
        );
    }
    for (index, (&x, &y)) in x.iter().zip(&y).enumerate() {
        let result = F32::evaluate(chain, [x, y], [p]);
        assert_eq!(
            result.to_bits(),
            expected[index],
            "evaluate {}",
            context(index)
        );
    }
}

/// Returns the operand pairs: every pair of the boundary encodings, and
/// random pairs.
fn pairs(random: &mut SplitMix64) -> Vec<(u32, u32)> {
    let mut pairs = boundary_pairs::<u32>(Layout::BINARY32);
    pairs.extend(random_pairs::<u32>(random, Layout::BINARY32, 4_000));
    pairs
}

/// Returns the parameters: zeros, ones, the extremes, infinities, NaNs, and
/// random encodings.
fn parameters(random: &mut SplitMix64) -> Vec<F32> {
    let mut bits = vec![
        0,
        0x8000_0000,
        0x3F80_0000,
        0xBF80_0000,
        1,
        0x7F7F_FFFF,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0000,
        0x7F80_0001,
    ];
    bits.extend(
        (0..6).map(|_| u32::try_from(random.next_u64() >> 32).expect("the shift keeps 32 bits")),
    );
    bits.into_iter().map(F32::from_bits).collect()
}

#[test]
fn blocks_give_the_steps_one_at_a_time() {
    let mut random = SplitMix64::new(0xB10C);
    let pairs = pairs(&mut random);
    for p in parameters(&mut random) {
        check(&Arithmetic, &arithmetic, &pairs, p);
        check(&Fused, &fused, &pairs, p);
        check(&Order, &order, &pairs, p);
        check(&Dropped, &dropped, &pairs, p);
    }
}

#[test]
fn value_steps_give_the_steps_one_at_a_time() {
    let mut random = SplitMix64::new(0x57E9);
    let mut pairs = pairs(&mut random);
    pairs.extend(integral_pairs());
    let parameters = parameters(&mut random);
    for step in steps() {
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
    // Inexact, overflow, underflow, divide by zero, invalid, and a
    // subnormal operand, in each chain.
    let pairs: [(u32, u32); 10] = [
        (0x3F80_0000, 0x4040_0000),
        (0x7F7F_FFFF, 0x7F7F_FFFF),
        (0x0DA2_4260, 0x0DA2_4260),
        (0x3F80_0000, 0),
        (0, 0),
        (0x7F80_0000, 0xFF80_0000),
        (0x0000_0001, 0x4000_0000),
        (0xBF80_0000, 0x3F80_0000),
        (0x7F80_0001, 0x3F80_0000),
        (0xC040_0000, 0x8000_0000),
    ];
    let x = pairs.map(|(x, _)| F32::from_bits(x));
    let y = pairs.map(|(_, y)| F32::from_bits(y));
    let p = F32::from_bits(0xBF80_0000);
    under_each_control(&Arithmetic, &arithmetic, &x, &y, p);
    under_each_control(&Fused, &fused, &x, &y, p);
    under_each_control(&Order, &order, &x, &y, p);
    under_each_control(&Dropped, &dropped, &x, &y, p);
    let steps = [
        Step::CopySign,
        Step::NextUp,
        Step::Integral,
        Step::IntegralBy(Rounding::TiesToAway),
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
fn under_each_control<C: Chain<2, 1> + Debug>(
    chain: &C,
    steps: OneAtATime<'_>,
    x: &[F32; 10],
    y: &[F32; 10],
    p: F32,
) {
    use floaty_verify::x86::{MXCSR_HOST_PATH_CONTROLS, with_mxcsr};

    for control in MXCSR_HOST_PATH_CONTROLS {
        let mut out = [F32::from_bits(0); 10];
        let first = with_mxcsr(control, || {
            F32::map(chain, [&x[..], &y[..]], [p], &mut out[..]);
            F32::evaluate(chain, [x[0], y[0]], [p])
        });
        assert_eq!(
            first.to_bits(),
            steps(x[0], y[0], p).to_bits(),
            "{chain:?} evaluate under {control:#x}"
        );
        for (index, result) in out.iter().enumerate() {
            let expected = steps(x[index], y[index], p);
            assert_eq!(
                result.to_bits(),
                expected.to_bits(),
                "{chain:?} {:#x} {:#x} under {control:#x}",
                x[index].to_bits(),
                y[index].to_bits()
            );
        }
    }
}
