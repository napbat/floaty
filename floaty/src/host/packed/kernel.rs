//! The slice kernels of `Lanes`: a dot product and a sum of squared
//! differences of two vectors, accumulated in binary32 lanes, and the
//! widenings of their values to binary32.
//!
//! The kernel checks the mode and the environment once for the call. It then
//! keeps `N` binary32 lanes in registers, adds the term of each pair of
//! values into its lane with the packed instructions of `binary` and
//! `mul_add`, and sums the lanes by halves. The order is the order of the
//! engine, which `crate::lanes::kernel` states. Each instruction gives the
//! bits of the engine for operands that are not NaNs, in a mode that
//! `ready_for` accepts. A NaN in a lane stays a NaN through every later
//! step, and the sum of the lanes adds every lane. So a NaN anywhere gives a
//! NaN sum, and one test of the sum sends the call to the engine, which
//! selects the NaN by the rule of the mode.

use super::super::bits::nan_32;
use super::super::environment::{self, packed};
use super::super::narrow::{SUBNORMAL_BIAS, round_to_half_lanes};
use super::super::paths::ready_for;
use super::{chunk, chunk_mut, in_chunks};
use crate::env::Env;
use crate::host::{Host, Load, Operation, Step, Term};

/// Returns the sum of the terms of the pairs of values of `x` and `y` in the
/// order of the engine, from the host unit, as a binary32 encoding. The
/// vectors hold one count of values. Returns `None` when the path does not
/// apply, when a conversion has no instruction, and when the sum is a NaN.
#[inline]
pub fn accumulate<const N: usize>(
    x: impl Load,
    y: impl Load,
    term: Term,
    step: Step,
    env: &Env,
) -> Option<u32> {
    if !ready_for(Host::Single, env, Host::Single.precision()) {
        return None;
    }
    // A sum of values has no product, so its step has no effect.
    match (term, step) {
        (Term::Value, _) => sum_vectors::<N, Sum>(x, y),
        (Term::Product, Step::Separate) => sum_vectors::<N, Product>(x, y),
        (Term::Product, Step::Fused) => sum_vectors::<N, FusedProduct>(x, y),
        (Term::SquareDifference, Step::Separate) => sum_vectors::<N, SquareDifference>(x, y),
        (Term::SquareDifference, Step::Fused) => sum_vectors::<N, FusedSquareDifference>(x, y),
    }
}

/// Computes the sum of the terms of each row of `rows` and `query`, as
/// `accumulate` does, with one check of the environment for every row. Row
/// `r` is the `query.count()` values of `rows` from value
/// `r * query.count()`, and `rows` holds `row_count` rows. Calls `each`
/// with the index of each row and its sum, or `None` for a sum that the
/// path does not give. Returns `None`, and calls `each` for no row, when the
/// path does not apply.
#[inline]
pub fn accumulate_rows<const N: usize>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    term: Term,
    env: &Env,
    each: impl FnMut(usize, Option<u32>),
) -> Option<()> {
    if !ready_for(Host::Single, env, Host::Single.precision()) {
        return None;
    }
    match term {
        Term::Value => sum_rows::<N, Sum>(rows, query, row_count, each),
        Term::Product => sum_rows::<N, Product>(rows, query, row_count, each),
        Term::SquareDifference => sum_rows::<N, SquareDifference>(rows, query, row_count, each),
    }
    Some(())
}

/// A term and a step as a type. Each type has its own copy of the loop of
/// the kernel, so the loop does not test the term or the step.
trait Accumulation {
    /// The term of each pair of values.
    const TERM: Term;
    /// How the term adds into its lane.
    const STEP: Step;
}

/// Defines the type of each pair of a term and a step.
macro_rules! accumulations {
    ($($(#[$doc:meta])* $name:ident = ($term:ident, $step:ident);)+) => {
        $(
            $(#[$doc])*
            struct $name;

            impl Accumulation for $name {
                const TERM: Term = Term::$term;
                const STEP: Step = Step::$step;
            }
        )+
    };
}

accumulations! {
    /// The sum of the values of the first vector.
    Sum = (Value, Separate);
    /// The dot product, with the product and the sum rounded apart.
    Product = (Product, Separate);
    /// The dot product, with a fused multiply-add.
    FusedProduct = (Product, Fused);
    /// The square of the distance, with the product and the sum rounded
    /// apart.
    SquareDifference = (SquareDifference, Separate);
    /// The square of the distance, with a fused multiply-add.
    FusedSquareDifference = (SquareDifference, Fused);
}

/// Calls `each` with the index of each row of `rows` and its sum with
/// `query`, as `accumulate_rows` states. The environment allows the host
/// path. Rows of at most [`SHORT`] values take `short_rows`.
#[inline]
fn sum_rows<const N: usize, A: Accumulation>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    mut each: impl FnMut(usize, Option<u32>),
) {
    let count = query.count();
    if N >= SHORT && (1..=SHORT).contains(&count) {
        short_rows::<A>(rows, query, row_count, each);
        return;
    }
    for index in 0..row_count {
        let row = rows.part(index * count, count);
        each(index, sum_vectors::<N, A>(row, query));
    }
}

/// The most values of a row that `short_rows` computes.
const SHORT: usize = 8;

/// Calls `each` with the index of each row of at most [`SHORT`] values and
/// its sum with `query`, four rows at a time, for at least [`SHORT`] lanes
/// and a separate step. Value `i` of a row then adds into lane `i`, once,
/// and the other lanes stay +0. A lane of +0 plus a term is the term with a
/// -0 made +0, and to make each -0 +0 commutes with a sum rounded to nearest
/// even. A lane that is not -0 plus a lane of +0 is the lane. So the levels
/// of the sum by halves above eight lanes change nothing, and the sum is the
/// sum by halves of the eight terms with a -0 made +0. The terms past the
/// last value are +0, as the lanes past it are.
#[inline]
fn short_rows<A: Accumulation>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    mut each: impl FnMut(usize, Option<u32>),
) {
    let count = query.count();
    let query = load_short(query, count);
    let last = row_count.saturating_sub(1);
    for first in (0..row_count).step_by(4) {
        // A group past the last row repeats the last row.
        let mut indices = [0; 4];
        indices
            .iter_mut()
            .enumerate()
            .for_each(|(offset, index)| *index = (first + offset).min(last));
        let sums = query.and_then(|query| four_rows::<A>(rows, &query, indices, count));
        for offset in 0..(row_count - first).min(4) {
            let sum = sums.map(|sums| sums[offset]).filter(|&sum| !nan_32(sum));
            each(first + offset, sum);
        }
    }
}

/// Returns the values of a vector of `count` values, at most [`SHORT`], as
/// [`SHORT`] binary32 values with +0 past the last value.
#[inline]
fn load_short(vector: impl Load, count: usize) -> Option<[f32; SHORT]> {
    let bits = if count == SHORT {
        vector.load::<SHORT>(0)?
    } else {
        vector.load_rest::<SHORT>(0)?
    };
    Some(singles(bits))
}

/// Returns the sums of four rows of `count` values with `query`, as
/// `short_rows` states. Each row adds the halves of its eight terms in one
/// packed sum. The four results then move so that one packed sum adds lane
/// `j` of each row to its lane `j + 2`, and another adds the two lanes left.
#[inline]
fn four_rows<A: Accumulation>(
    rows: impl Load,
    query: &[f32; SHORT],
    indices: [usize; 4],
    count: usize,
) -> Option<[u32; 4]> {
    let mut quarters = [[0.0; 4]; 4];
    for (quarter, index) in quarters.iter_mut().zip(indices) {
        let terms = short_terms::<A>(&load_short(rows.part(index * count, count), count)?, query)?;
        *quarter = packed::binary_f32x4(*chunk(&terms, 0), *chunk(&terms, 4), Operation::Add);
    }
    let column = |lane: usize| {
        [
            quarters[0][lane],
            quarters[1][lane],
            quarters[2][lane],
            quarters[3][lane],
        ]
    };
    let low = packed::binary_f32x4(column(0), column(2), Operation::Add);
    let high = packed::binary_f32x4(column(1), column(3), Operation::Add);
    let mut sums = encodings(packed::binary_f32x4(low, high, Operation::Add));
    // A -0 sum is +0, as the lanes give it.
    for sum in &mut sums {
        *sum &= 0u32.wrapping_sub(u32::from(*sum != 0x8000_0000));
    }
    Some(sums)
}

/// Returns the term of each pair of values of `x` and `y`, rounded once:
/// eight at a time where the build has wide registers, and four otherwise.
#[inline]
fn short_terms<A: Accumulation>(x: &[f32; SHORT], y: &[f32; SHORT]) -> Option<[f32; SHORT]> {
    let mut terms = [0.0; SHORT];
    in_chunks::<f32, SHORT, 8, 4>(
        &mut terms,
        |start| term_chunk::<8, A>(*chunk(x, start), *chunk(y, start), packed::binary_f32x8),
        |start| {
            term_chunk::<4, A>(*chunk(x, start), *chunk(y, start), |a, b, operation| {
                Some(packed::binary_f32x4(a, b, operation))
            })
        },
        |index| {
            let [term] = term_chunk::<1, A>([x[index]], [y[index]], |[a], [b], operation| {
                Some([environment::binary_f32(a, b, operation)])
            })?;
            Some(term)
        },
    )?;
    Some(terms)
}

/// Returns the term of each pair of `x` and `y`. `binary` is the instruction
/// for the chunk width, and gives `None` where the build has none.
#[inline]
fn term_chunk<const C: usize, A: Accumulation>(
    x: [f32; C],
    y: [f32; C],
    binary: impl Fn([f32; C], [f32; C], Operation) -> Option<[f32; C]>,
) -> Option<[f32; C]> {
    match A::TERM {
        Term::Value => Some(x),
        Term::Product => binary(x, y, Operation::Mul),
        Term::SquareDifference => {
            let difference = binary(x, y, Operation::Sub)?;
            binary(difference, difference, Operation::Mul)
        }
    }
}

/// Returns the sum of the terms of the pairs of values of `x` and `y` in the
/// order of the engine, or `None` when a conversion has no instruction and
/// when the sum is a NaN. The environment allows the host path.
#[inline]
fn sum_vectors<const N: usize, A: Accumulation>(x: impl Load, y: impl Load) -> Option<u32> {
    let mut lanes = [0.0; N];
    if N.is_multiple_of(8) {
        add_vectors::<N, 8, A>(&mut lanes, x, y)?;
    } else if N.is_multiple_of(4) {
        add_vectors::<N, 4, A>(&mut lanes, x, y)?;
    } else {
        add_vectors::<N, 1, A>(&mut lanes, x, y)?;
    }
    let sum = sum_by_halves(lanes).to_bits();
    (!nan_32(sum)).then_some(sum)
}

/// Adds the term of each pair of values of `x` and `y` into its lane, `C`
/// lanes at a time. A load of `C` values at a time keeps the values beside
/// the lanes in registers, where a load of `N` values spilled them.
#[inline]
fn add_vectors<const N: usize, const C: usize, A: Accumulation>(
    lanes: &mut [f32; N],
    x: impl Load,
    y: impl Load,
) -> Option<()> {
    let count = x.count();
    let full = count - count % N;
    for start in (0..full).step_by(N) {
        // The compiler knows the count of a part of `N` values. So one check
        // of the bounds for each part serves every load from the part.
        let (x_part, y_part) = (x.part(start, N), y.part(start, N));
        for offset in (0..N).step_by(C) {
            let (a, b) = (x_part.load::<C>(offset)?, y_part.load::<C>(offset)?);
            add_terms::<C, A>(chunk_mut(lanes, offset), &singles(a), &singles(b))?;
        }
    }
    // Value `full + i` adds into lane `i`.
    for offset in (0..count - full).step_by(C) {
        let start = full + offset;
        let lanes = chunk_mut(lanes, offset);
        if start + C <= count {
            let (a, b) = (x.load::<C>(start)?, y.load::<C>(start)?);
            add_terms::<C, A>(lanes, &singles(a), &singles(b))?;
        } else {
            // The lanes past the last value must keep their values. A
            // padding term of +0 can change a lane: a fused step can make a
            // lane -0, and -0 plus +0 is +0.
            let old = *lanes;
            let (a, b) = (x.load_rest::<C>(start)?, y.load_rest::<C>(start)?);
            add_terms::<C, A>(lanes, &singles(a), &singles(b))?;
            let rest = count - start;
            lanes[rest..].copy_from_slice(&old[rest..]);
        }
    }
    Some(())
}

/// Returns binary32 encodings as host values.
#[inline]
pub(super) fn singles<const N: usize>(encodings: [u32; N]) -> [f32; N] {
    // A loop, not `array::map`, which LLVM calls out of line.
    let mut lanes = [0.0; N];
    lanes
        .iter_mut()
        .zip(encodings)
        .for_each(|(lane, bits)| *lane = f32::from_bits(bits));
    lanes
}

/// Returns host values as binary32 encodings.
#[inline]
pub(super) fn encodings<const N: usize>(lanes: [f32; N]) -> [u32; N] {
    let mut encodings = [0; N];
    encodings
        .iter_mut()
        .zip(lanes)
        .for_each(|(bits, lane)| *bits = lane.to_bits());
    encodings
}

/// Returns `operation` of each lane and `operand`: eight lanes at a time
/// where the build has wide registers, then four, and then one.
#[inline]
fn with_operand<const N: usize>(
    lanes: [f32; N],
    operand: f32,
    operation: Operation,
) -> Option<[f32; N]> {
    let mut results = lanes;
    in_chunks::<f32, N, 8, 4>(
        &mut results,
        |start| packed::binary_f32x8(*chunk(&lanes, start), [operand; 8], operation),
        |start| {
            Some(packed::binary_f32x4(
                *chunk(&lanes, start),
                [operand; 4],
                operation,
            ))
        },
        |index| Some(environment::binary_f32(lanes[index], operand, operation)),
    )?;
    Some(results)
}

/// Returns binary16 encodings widened exactly to binary32: in the packed
/// widening instructions where the build has them, and in integer and
/// binary32 instructions otherwise.
#[inline]
pub fn widen_halves<const N: usize>(halves: &[u16; N]) -> Option<[u32; N]> {
    if packed::HALF {
        let mut lanes = [0.0; N];
        in_chunks::<f32, N, 8, 4>(
            &mut lanes,
            |start| packed::widen_halves_x8(*chunk(halves, start)),
            |start| packed::widen_halves_x4(*chunk(halves, start)),
            |index| Some(environment::widen_half(halves[index])),
        )?;
        return Some(encodings(lanes));
    }
    // The fraction of a subnormal binary16 value times 2^-24 is its
    // magnitude, a normal binary32 value. The conversion of the fraction and
    // the product by a power of two are exact.
    let fractions = from_integers(halves, |bits| i32::from(bits & 0x03FF))?;
    let subnormals = encodings(with_operand(
        fractions,
        f32::from_bits(0x3380_0000),
        Operation::Mul,
    )?);
    let mut widened = [0; N];
    widened
        .iter_mut()
        .zip(halves.iter().zip(subnormals))
        .for_each(|(lane, (&bits, subnormal))| *lane = widen_half_bits(bits, subnormal));
    Some(widened)
}

/// Returns binary32 values rounded to binary16 to nearest even, as
/// encodings: in the packed rounding instructions where the build has them,
/// and otherwise in integer instructions and one binary32 sum of each
/// magnitude and 0.5, as `round_to_half_lanes` states. A NaN gives a NaN.
#[inline]
pub fn narrow_halves<const N: usize>(singles: &[f32; N]) -> Option<[u16; N]> {
    let mut halves = [0; N];
    if packed::HALF {
        in_chunks::<u16, N, 8, 4>(
            &mut halves,
            |start| packed::narrow_halves_x8(*chunk(singles, start)),
            |start| packed::narrow_halves_x4(*chunk(singles, start)),
            |index| Some(environment::narrow_half(singles[index])),
        )?;
        return Some(halves);
    }
    let mut magnitudes = [0.0; N];
    magnitudes
        .iter_mut()
        .zip(singles)
        .for_each(|(magnitude, value)| *magnitude = f32::from_bits(value.to_bits() & 0x7FFF_FFFF));
    let sums = encodings(with_operand(
        magnitudes,
        f32::from_bits(SUBNORMAL_BIAS),
        Operation::Add,
    )?);
    halves
        .iter_mut()
        .zip(singles.iter().zip(sums))
        .for_each(|(half, (value, sum))| *half = round_to_half_lanes(value.to_bits(), sum));
    Some(halves)
}

/// Returns the binary32 encoding of a binary16 encoding, in masks without a
/// branch, which LLVM vectorizes. `subnormal` is the binary32 encoding of
/// the magnitude of the value when its exponent field is zero. Otherwise
/// the exponent field and the fraction move to their binary32 positions,
/// and the exponent field takes the bias of binary32: 127 - 15 = 112 more
/// for a normal value, and 255 - 31 = 224 more for an infinity or a NaN.
#[inline]
fn widen_half_bits(bits: u16, subnormal: u32) -> u32 {
    let bits = u32::from(bits);
    let sign = (bits & 0x8000) << 16;
    let magnitude = (bits & 0x7FFF) << 13;
    let field = bits & 0x7C00;
    let zero_field = 0u32.wrapping_sub(u32::from(field == 0));
    let full_field = 0u32.wrapping_sub(u32::from(field == 0x7C00));
    let other = magnitude + (112 << 23) + (full_field & (112 << 23));
    sign | (zero_field & subnormal) | (!zero_field & other)
}

/// Returns integers converted to binary32: eight at a time where the build
/// has wide registers, then four, and then one. `widen` gives each integer,
/// which is below `2^24` in magnitude, so the conversion is exact and the
/// rounding direction does not matter.
#[inline]
fn from_integers<T: Copy, const N: usize>(
    values: &[T; N],
    widen: impl Fn(T) -> i32,
) -> Option<[f32; N]> {
    let mut integers = [0; N];
    integers
        .iter_mut()
        .zip(values)
        .for_each(|(integer, &value)| *integer = widen(value));
    let mut lanes = [0.0; N];
    in_chunks::<f32, N, 8, 4>(
        &mut lanes,
        |start| packed::from_int_x8(*chunk(&integers, start)),
        |start| Some(packed::from_int_x4(*chunk(&integers, start))),
        |index| environment::from_int_f32(i64::from(integers[index])),
    )?;
    Some(lanes)
}

/// Returns unsigned 8-bit integers converted exactly to binary32.
#[inline]
pub fn widen_codes<const N: usize>(codes: &[u8; N]) -> Option<[u32; N]> {
    Some(encodings(from_integers(codes, i32::from)?))
}

/// Returns signed 8-bit integers converted exactly to binary32, each then
/// multiplied by `scale` and rounded.
#[inline]
pub fn widen_scaled_codes<const N: usize>(codes: &[i8; N], scale: u32) -> Option<[u32; N]> {
    let values = from_integers(codes, i32::from)?;
    let scaled = with_operand(values, f32::from_bits(scale), Operation::Mul)?;
    Some(encodings(scaled))
}

/// Adds the term of each pair of `x` and `y` into its lane of `lanes`: eight
/// lanes at a time where the build has wide registers, then four, and then
/// one.
#[inline]
fn add_terms<const N: usize, A: Accumulation>(
    lanes: &mut [f32; N],
    x: &[f32; N],
    y: &[f32; N],
) -> Option<()> {
    let old = *lanes;
    in_chunks::<f32, N, 8, 4>(
        lanes,
        |start| {
            add_chunk::<8, A>(
                *chunk(&old, start),
                *chunk(x, start),
                *chunk(y, start),
                packed::binary_f32x8,
                packed::mul_add_f32x8,
            )
        },
        |start| {
            add_chunk::<4, A>(
                *chunk(&old, start),
                *chunk(x, start),
                *chunk(y, start),
                |a, b, operation| Some(packed::binary_f32x4(a, b, operation)),
                packed::mul_add_f32x4,
            )
        },
        |index| {
            let [lane] = add_chunk::<1, A>(
                [old[index]],
                [x[index]],
                [y[index]],
                |[a], [b], operation| Some([environment::binary_f32(a, b, operation)]),
                |[a], [b], [c]| environment::mul_add_f32(a, b, c).map(|value| [value]),
            )?;
            Some(lane)
        },
    )
}

/// Returns a chunk of lanes with the term of each pair of `x` and `y` added
/// in. `binary` and `fused` are the instructions for the chunk width, and
/// give `None` where the build has no instruction.
#[inline]
fn add_chunk<const C: usize, A: Accumulation>(
    lanes: [f32; C],
    x: [f32; C],
    y: [f32; C],
    binary: impl Fn([f32; C], [f32; C], Operation) -> Option<[f32; C]>,
    fused: impl Fn([f32; C], [f32; C], [f32; C]) -> Option<[f32; C]>,
) -> Option<[f32; C]> {
    let (left, right) = match A::TERM {
        Term::Value => return binary(lanes, x, Operation::Add),
        Term::Product => (x, y),
        Term::SquareDifference => {
            let difference = binary(x, y, Operation::Sub)?;
            (difference, difference)
        }
    };
    match A::STEP {
        Step::Separate => binary(lanes, binary(left, right, Operation::Mul)?, Operation::Add),
        Step::Fused => fused(left, right, lanes),
    }
}

/// Returns the sum of the lanes by halves: while more than one lane is left,
/// lane `i` of the low half becomes lane `i` plus lane `i` of the high half.
/// `N` is a power of two.
#[inline]
fn sum_by_halves<const N: usize>(mut lanes: [f32; N]) -> f32 {
    let mut half = N / 2;
    while half > 0 {
        let (low, high) = lanes.split_at_mut(half);
        let (low_chunks, low_rest) = low.as_chunks_mut::<4>();
        let (high_chunks, high_rest) = high[..half].as_chunks::<4>();
        low_chunks
            .iter_mut()
            .zip(high_chunks)
            .for_each(|(low, &high)| *low = packed::binary_f32x4(*low, high, Operation::Add));
        low_rest
            .iter_mut()
            .zip(high_rest)
            .for_each(|(low, &high)| *low = environment::binary_f32(*low, high, Operation::Add));
        half /= 2;
    }
    lanes[0]
}
