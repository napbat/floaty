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
    sum_vectors::<N>(x, y, term, step)
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
    mut each: impl FnMut(usize, Option<u32>),
) -> Option<()> {
    if !ready_for(Host::Single, env, Host::Single.precision()) {
        return None;
    }
    let count = query.count();
    for index in 0..row_count {
        let row = rows.part(index * count, count);
        each(index, sum_vectors::<N>(row, query, term, Step::Separate));
    }
    Some(())
}

/// Returns the sum of the terms of the pairs of values of `x` and `y` in the
/// order of the engine, or `None` when a conversion has no instruction and
/// when the sum is a NaN. The environment allows the host path.
#[inline]
fn sum_vectors<const N: usize>(x: impl Load, y: impl Load, term: Term, step: Step) -> Option<u32> {
    let mut lanes = [0.0; N];
    if N.is_multiple_of(8) {
        add_vectors::<N, 8>(&mut lanes, x, y, term, step)?;
    } else if N.is_multiple_of(4) {
        add_vectors::<N, 4>(&mut lanes, x, y, term, step)?;
    } else {
        add_vectors::<N, 1>(&mut lanes, x, y, term, step)?;
    }
    let sum = sum_by_halves(lanes).to_bits();
    (!nan_32(sum)).then_some(sum)
}

/// Adds the term of each pair of values of `x` and `y` into its lane, `C`
/// lanes at a time. A load of `C` values at a time keeps the values beside
/// the lanes in registers, where a load of `N` values spilled them.
#[inline]
fn add_vectors<const N: usize, const C: usize>(
    lanes: &mut [f32; N],
    x: impl Load,
    y: impl Load,
    term: Term,
    step: Step,
) -> Option<()> {
    let count = x.count();
    let full = count - count % N;
    for start in (0..full).step_by(N) {
        for offset in (0..N).step_by(C) {
            let (a, b) = (x.load::<C>(start + offset)?, y.load::<C>(start + offset)?);
            add_terms(
                chunk_mut(lanes, offset),
                &singles(a),
                &singles(b),
                term,
                step,
            )?;
        }
    }
    // Value `full + i` adds into lane `i`. A padding value past the last
    // value adds the term +0 or -0, and a lane plus a zero is the lane: a
    // lane starts at +0 and becomes -0 only from -0 plus -0, so no lane is
    // -0. So a chunk of padding alone changes no lane, and the loop skips it.
    for offset in (0..count - full).step_by(C) {
        let start = full + offset;
        let (a, b) = if start + C <= count {
            (x.load::<C>(start)?, y.load::<C>(start)?)
        } else {
            (x.load_rest::<C>(start)?, y.load_rest::<C>(start)?)
        };
        add_terms(
            chunk_mut(lanes, offset),
            &singles(a),
            &singles(b),
            term,
            step,
        )?;
    }
    Some(())
}

/// Returns binary32 encodings as host values.
#[inline]
fn singles<const N: usize>(encodings: [u32; N]) -> [f32; N] {
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
fn encodings<const N: usize>(lanes: [f32; N]) -> [u32; N] {
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
fn add_terms<const N: usize>(
    lanes: &mut [f32; N],
    x: &[f32; N],
    y: &[f32; N],
    term: Term,
    step: Step,
) -> Option<()> {
    let old = *lanes;
    in_chunks::<f32, N, 8, 4>(
        lanes,
        |start| {
            add_chunk(
                *chunk(&old, start),
                *chunk(x, start),
                *chunk(y, start),
                term,
                step,
                packed::binary_f32x8,
                packed::mul_add_f32x8,
            )
        },
        |start| {
            add_chunk(
                *chunk(&old, start),
                *chunk(x, start),
                *chunk(y, start),
                term,
                step,
                |a, b, operation| Some(packed::binary_f32x4(a, b, operation)),
                packed::mul_add_f32x4,
            )
        },
        |index| {
            let [lane] = add_chunk(
                [old[index]],
                [x[index]],
                [y[index]],
                term,
                step,
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
fn add_chunk<const C: usize>(
    lanes: [f32; C],
    x: [f32; C],
    y: [f32; C],
    term: Term,
    step: Step,
    binary: impl Fn([f32; C], [f32; C], Operation) -> Option<[f32; C]>,
    fused: impl Fn([f32; C], [f32; C], [f32; C]) -> Option<[f32; C]>,
) -> Option<[f32; C]> {
    let (left, right) = match term {
        Term::Value => return binary(lanes, x, Operation::Add),
        Term::Product => (x, y),
        Term::SquareDifference => {
            let difference = binary(x, y, Operation::Sub)?;
            (difference, difference)
        }
    };
    match step {
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
