//! Blocks: `F32::map` and `F32::evaluate` give the results of the steps of
//! each chain one at a time, through their `_with` methods in the default
//! mode. The chains take every step of `Steps`, and NaNs that a later step
//! drops. The operands are every pair of the boundary encodings of
//! binary32, and random pairs, with boundary and random parameters, in
//! slices of `F32` and of `f32` and of each length. On x86-64, a block under
//! each MXCSR control of the host-path tests gives the results of the
//! engine, and no unmasked exception traps: the check of the environment
//! comes before every step.

use floaty::block::{Chain, Steps};
use floaty::{Env, F32};
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;

use super::operands::{boundary_pairs, random_pairs};

/// The steps one at a time take the default mode of `F32`.
const IEEE: Env = Env::IEEE;

/// The steps of a chain one at a time, from its values and parameter.
type OneAtATime = fn(F32, F32, F32) -> F32;

/// A block of one chain under test: it writes `map` into its argument and
/// returns `evaluate` of the first lane.
#[cfg(target_arch = "x86_64")]
type Run<'a> = &'a dyn Fn(&mut [F32; 10]) -> F32;

/// `(x + y) * x - y / x`: the four operators.
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

/// Checks the block of `chain` against `steps`, one step at a time, for
/// each pair of `pairs` and the parameter `p`: `map` over slices of `F32`
/// and of `f32`, of every length up to the pairs, and `evaluate` of each
/// pair.
fn check<C: Chain<2, 1>>(chain: &C, steps: OneAtATime, pairs: &[(u32, u32)], p: F32) {
    let x: Vec<F32> = pairs.iter().map(|&(x, _)| F32::from_bits(x)).collect();
    let y: Vec<F32> = pairs.iter().map(|&(_, y)| F32::from_bits(y)).collect();
    let expected: Vec<u32> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| steps(x, y, p).to_bits())
        .collect();
    let context = |index: usize| {
        format!(
            "{:#x} {:#x} {:#x}",
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
        check(&Arithmetic, arithmetic, &pairs, p);
        check(&Fused, fused, &pairs, p);
        check(&Order, order, &pairs, p);
        check(&Dropped, dropped, &pairs, p);
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
    use floaty_verify::x86::{MXCSR_HOST_PATH_CONTROLS, with_mxcsr};

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
    let chains: [(Run<'_>, OneAtATime); 4] = [
        (&|out| block(&Arithmetic, &x, &y, p, out), arithmetic),
        (&|out| block(&Fused, &x, &y, p, out), fused),
        (&|out| block(&Order, &x, &y, p, out), order),
        (&|out| block(&Dropped, &x, &y, p, out), dropped),
    ];
    for control in MXCSR_HOST_PATH_CONTROLS {
        for (run, steps) in &chains {
            let mut out = [F32::from_bits(0); 10];
            let first = with_mxcsr(control, || run(&mut out));
            assert_eq!(
                first.to_bits(),
                steps(x[0], y[0], p).to_bits(),
                "evaluate under {control:#x}"
            );
            for (index, result) in out.iter().enumerate() {
                let expected = steps(x[index], y[index], p);
                assert_eq!(
                    result.to_bits(),
                    expected.to_bits(),
                    "{:#x} {:#x} under {control:#x}",
                    x[index].to_bits(),
                    y[index].to_bits()
                );
            }
        }
    }
}

/// Runs `map` of `chain` into `out`, and returns `evaluate` of the first
/// lane.
#[cfg(target_arch = "x86_64")]
fn block<C: Chain<2, 1>>(
    chain: &C,
    x: &[F32; 10],
    y: &[F32; 10],
    p: F32,
    out: &mut [F32; 10],
) -> F32 {
    F32::map(chain, [&x[..], &y[..]], [p], &mut out[..]);
    F32::evaluate(chain, [x[0], y[0]], [p])
}
