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
//! `dispatch::direction` accepts. Each step rounds in the direction of the
//! mode, as the module `elementwise` states for its views. A NaN in a lane
//! stays a NaN through every later step, and the sum of the lanes adds every
//! lane. So a NaN anywhere gives a NaN sum, and one test of the sum sends
//! the call to the engine, which selects the NaN by the rule of the mode.
//! The kernels compute in the instruction set that `dispatch` selects.

use super::super::bits::nan_32;
use super::super::environment::packed;
use super::super::narrow::{HALF_QUANTUM, SUBNORMAL_BIAS, round_to_half_lanes, widen_half_lanes};
use super::{Directed, Nearest, any_lane, chunk, chunk_mut, dispatch, isa};
use crate::env::{Env, Rounding};
use crate::host::{Direction, Isa, Load, Operation, Step, Term};

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
    let rounding = dispatch::direction(env)?;
    dispatch::accumulate::<N>(x, y, term, step, rounding)
}

/// Returns the sum of `accumulate` in the instruction set `I`, in an
/// environment that allows the path, with each step rounded in the
/// direction `rounding`. The match comes before `sum_vectors`, whose loop
/// runs with the features of `I` for one term and step.
#[inline]
pub(super) fn accumulate_on<I: Isa, const N: usize>(
    x: impl Load,
    y: impl Load,
    term: Term,
    step: Step,
    rounding: Rounding,
) -> Option<u32> {
    // A sum of values has no product, so its step has no effect.
    match (term, step) {
        (Term::Value, _) => sum_vectors::<I, N, Sum>(x, y, rounding),
        (Term::Product, Step::Separate) => sum_vectors::<I, N, Product>(x, y, rounding),
        (Term::Product, Step::Fused) => sum_vectors::<I, N, FusedProduct>(x, y, rounding),
        (Term::SquareDifference, Step::Separate) => {
            sum_vectors::<I, N, SquareDifference>(x, y, rounding)
        }
        (Term::SquareDifference, Step::Fused) => {
            sum_vectors::<I, N, FusedSquareDifference>(x, y, rounding)
        }
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
    let rounding = dispatch::direction(env)?;
    dispatch::rows::<N>(rows, query, row_count, term, rounding, each);
    Some(())
}

/// Calls `each` for each row as `accumulate_rows` does, in the instruction
/// set `I`, in an environment that allows the path, with each step rounded
/// in the direction `rounding`. The match comes before `sum_rows`, whose
/// loop runs with the features of `I` for one term.
#[inline]
pub(super) fn rows_on<I: Isa, const N: usize>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    term: Term,
    rounding: Rounding,
    each: impl FnMut(usize, Option<u32>),
) {
    match term {
        Term::Value => sum_rows::<I, N, Sum>(rows, query, row_count, rounding, each),
        Term::Product => sum_rows::<I, N, Product>(rows, query, row_count, rounding, each),
        Term::SquareDifference => {
            sum_rows::<I, N, SquareDifference>(rows, query, row_count, rounding, each);
        }
    }
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
/// `query`, as `accumulate_rows` states, with the features of `I` and each
/// step rounded in the direction `rounding`. The environment allows the
/// host path. Rows of at most [`SHORT`] values take `short_rows` to nearest
/// even, the direction of its proof.
#[inline]
fn sum_rows<I: Isa, const N: usize, A: Accumulation>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    rounding: Rounding,
    mut each: impl FnMut(usize, Option<u32>),
) {
    I::run(move || {
        let count = query.count();
        if N >= SHORT && (1..=SHORT).contains(&count) && rounding == Rounding::TiesToEven {
            short_rows::<I, A>(rows, query, row_count, each);
            return;
        }
        for index in 0..row_count {
            let row = rows.part(index * count, count);
            each(index, sum_vectors::<I, N, A>(row, query, rounding));
        }
    });
}

/// The most values of a row that `short_rows` computes.
const SHORT: usize = 8;

/// Calls `each` with the index of each row of at most [`SHORT`] values and
/// its sum with `query`, four rows at a time, for at least [`SHORT`] lanes,
/// a separate step, and the direction to nearest even. Value `i` of a row
/// then adds into lane `i`, once, and the other lanes stay +0. A lane of +0
/// plus a term is the term with a -0 made +0, and to make each -0 +0
/// commutes with a sum rounded to nearest even. A lane that is not -0 plus a
/// lane of +0 is the lane. So the levels of the sum by halves above eight
/// lanes change nothing, and the sum is the sum by halves of the eight terms
/// with a -0 made +0. The terms past the last value are +0, as the lanes
/// past it are.
#[inline]
fn short_rows<I: Isa, A: Accumulation>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    each: impl FnMut(usize, Option<u32>),
) {
    // The test of the count comes before the loop, so the loop of rows of
    // `SHORT` values loads each row whole and calls no function.
    if query.count() == SHORT {
        rows_of_count::<I, A, true>(rows, query, row_count, each);
    } else {
        rows_of_count::<I, A, false>(rows, query, row_count, each);
    }
}

/// Calls `each` as `short_rows` does, for rows of [`SHORT`] values when
/// `WHOLE` holds, and of fewer otherwise. The loop runs with the features of
/// `I` also where LLVM leaves the function out of line.
#[inline]
fn rows_of_count<I: Isa, A: Accumulation, const WHOLE: bool>(
    rows: impl Load,
    query: impl Load,
    row_count: usize,
    mut each: impl FnMut(usize, Option<u32>),
) {
    I::run(move || {
        let count = query.count();
        let query = load_short::<I, WHOLE>(query);
        let last = row_count.saturating_sub(1);
        for first in (0..row_count).step_by(4) {
            // A group past the last row repeats the last row.
            let mut indices = [0; 4];
            indices
                .iter_mut()
                .enumerate()
                .for_each(|(offset, index)| *index = (first + offset).min(last));
            let sums =
                query.and_then(|query| four_rows::<I, A, WHOLE>(rows, &query, indices, count));
            // A whole group without a NaN takes one test for its four sums.
            // `dot_rows` of 256 rows of eight values then took 3.7 cycles a
            // row in x86-64-v3 on a Ryzen AI Max+ 395, against 4.8 with a
            // test of each sum.
            if let Some([a, b, c, d]) = sums
                && row_count - first >= 4
                && !any_lane([a, b, c, d], nan_32)
            {
                each(first, Some(a));
                each(first + 1, Some(b));
                each(first + 2, Some(c));
                each(first + 3, Some(d));
                continue;
            }
            for offset in 0..(row_count - first).min(4) {
                let sum = sums.map(|sums| sums[offset]).filter(|&sum| !nan_32(sum));
                each(first + offset, sum);
            }
        }
    });
}

/// Returns the values of a vector of [`SHORT`] values when `WHOLE` holds,
/// and of fewer otherwise, as [`SHORT`] binary32 values with +0 past the
/// last value.
#[inline]
fn load_short<I: Isa, const WHOLE: bool>(vector: impl Load) -> Option<[f32; SHORT]> {
    let bits = if WHOLE {
        vector.load::<I, SHORT>(0, Nearest)?
    } else {
        vector.load_rest::<I, SHORT>(0, Nearest)?
    };
    Some(singles(bits))
}

/// Returns the sums of four rows of `count` values with `query`, as
/// `short_rows` states. Each row adds the halves of its eight terms in one
/// packed sum. One packed sum for each two rows then adds lanes 0 and 1 of
/// each row to its lanes 2 and 3, and one more adds the two lanes left of
/// each of the four rows. The function runs with the features of `I` also
/// where LLVM leaves it out of line: a copy without them called each form
/// out of line, and `dot_rows` of rows of eight values took 12 ns for each
/// row in x86-64-v3 and x86-64-v4, against 0.9 ns with them.
#[inline]
fn four_rows<I: Isa, A: Accumulation, const WHOLE: bool>(
    rows: impl Load,
    query: &[f32; SHORT],
    indices: [usize; 4],
    count: usize,
) -> Option<[u32; 4]> {
    I::run(move || {
        let mut quarters = [[0.0; 4]; 4];
        for (quarter, index) in quarters.iter_mut().zip(indices) {
            let row = load_short::<I, WHOLE>(rows.part(index * count, count))?;
            let terms = short_terms::<I, A>(&row, query)?;
            *quarter = I::binary_f32x4(*chunk(&terms, 0), *chunk(&terms, 4), Operation::Add)?;
        }
        // Whole quarters move into the operands of each sum, as `halves_f32x4`
        // states.
        let pair = |first, second| {
            let (low, high) = packed::halves_f32x4(first, second);
            I::binary_f32x4(low, high, Operation::Add)
        };
        let front = pair(quarters[0], quarters[1])?;
        let back = pair(quarters[2], quarters[3])?;
        let (evens, odds) = packed::evens_odds_f32x4(front, back);
        let mut sums = encodings(I::binary_f32x4(evens, odds, Operation::Add)?);
        // A -0 sum is +0, as the lanes give it.
        for sum in &mut sums {
            *sum &= 0u32.wrapping_sub(u32::from(*sum != 0x8000_0000));
        }
        Some(sums)
    })
}

/// Returns the term of each pair of values of `x` and `y`, rounded once.
#[inline]
fn short_terms<I: Isa, A: Accumulation>(
    x: &[f32; SHORT],
    y: &[f32; SHORT],
) -> Option<[f32; SHORT]> {
    match A::TERM {
        Term::Value => Some(*x),
        Term::Product => isa::binary::<I, SHORT>(x, y, Operation::Mul),
        Term::SquareDifference => {
            let difference = isa::binary::<I, SHORT>(x, y, Operation::Sub)?;
            isa::binary::<I, SHORT>(&difference, &difference, Operation::Mul)
        }
    }
}

/// Returns the sum of the terms of the pairs of values of `x` and `y` in the
/// order of the engine, with each step rounded in the direction `rounding`,
/// or `None` when a conversion or the direction has no instruction and when
/// the sum is a NaN. The environment allows the host path.
#[inline]
fn sum_vectors<I: Isa, const N: usize, A: Accumulation>(
    x: impl Load,
    y: impl Load,
    rounding: Rounding,
) -> Option<u32> {
    if rounding == Rounding::TiesToEven {
        sum_lanes::<I, N, A, _>(x, y, Nearest)
    } else {
        sum_lanes::<I, N, A, _>(x, y, Directed(rounding))
    }
}

/// Returns the sum of `sum_vectors` in the copy of the loop for the
/// direction type `D`. The loop runs with the features of `I`.
#[inline]
fn sum_lanes<I: Isa, const N: usize, A: Accumulation, D: Direction>(
    x: impl Load,
    y: impl Load,
    direction: D,
) -> Option<u32> {
    I::run(move || {
        let mut lanes = [0.0; N];
        if I::EXTRA_WIDE && N.is_multiple_of(16) {
            add_vectors::<I, N, 16, A>(&mut lanes, x, y, direction)?;
        } else if N.is_multiple_of(8) {
            add_vectors::<I, N, 8, A>(&mut lanes, x, y, direction)?;
        } else if N.is_multiple_of(4) {
            add_vectors::<I, N, 4, A>(&mut lanes, x, y, direction)?;
        } else {
            add_vectors::<I, N, 1, A>(&mut lanes, x, y, direction)?;
        }
        let sum = if D::NEAREST {
            sum_by_halves::<I, N>(lanes)
        } else {
            sum_by_halves_rounded::<I, N>(lanes, direction.rounding())
        };
        let sum = sum?.to_bits();
        (!nan_32(sum)).then_some(sum)
    })
}

/// Adds the term of each pair of values of `x` and `y` into its lane, `C`
/// lanes at a time, rounding in the direction `direction`. A load of `C`
/// values at a time keeps the values beside the lanes in registers, where a
/// load of `N` values spilled them.
#[inline]
fn add_vectors<I: Isa, const N: usize, const C: usize, A: Accumulation>(
    lanes: &mut [f32; N],
    x: impl Load,
    y: impl Load,
    direction: impl Direction,
) -> Option<()> {
    let count = x.count();
    let full = count - count % N;
    for start in (0..full).step_by(N) {
        // The compiler knows the count of a part of `N` values. So one check
        // of the bounds for each part serves every load from the part.
        let (x_part, y_part) = (x.part(start, N), y.part(start, N));
        for offset in (0..N).step_by(C) {
            let a = x_part.load::<I, C>(offset, direction)?;
            let b = y_part.load::<I, C>(offset, direction)?;
            add_terms::<I, C, A, _>(
                chunk_mut(lanes, offset),
                &singles(a),
                &singles(b),
                direction,
            )?;
        }
    }
    // Value `full + i` adds into lane `i`. The values past the last part add
    // in whole chunks of `C`, 8, and 4 values, and then one at a time, so no
    // lane takes a padding term. A padding term of +0 can change a lane: a
    // fused step can make a lane -0, and -0 plus +0 is +0.
    let rest = count - full;
    let mut offset = add_chunks::<I, N, C, A>(lanes, x, y, full, 0, rest, direction)?;
    if C > 8 {
        offset = add_chunks::<I, N, 8, A>(lanes, x, y, full, offset, rest, direction)?;
    }
    if C > 4 {
        offset = add_chunks::<I, N, 4, A>(lanes, x, y, full, offset, rest, direction)?;
    }
    if C > 1 {
        add_chunks::<I, N, 1, A>(lanes, x, y, full, offset, rest, direction)?;
    }
    Some(())
}

/// Adds the term of each pair of values from `full + offset` into its lane
/// from `offset`, `C` lanes at a time, while `C` of the `rest` values past
/// `full` are left, rounding in the direction `direction`. Returns the
/// offset of the first value left.
#[inline]
fn add_chunks<I: Isa, const N: usize, const C: usize, A: Accumulation>(
    lanes: &mut [f32; N],
    x: impl Load,
    y: impl Load,
    full: usize,
    mut offset: usize,
    rest: usize,
    direction: impl Direction,
) -> Option<usize> {
    while offset + C <= rest {
        let start = full + offset;
        let (a, b) = (
            x.load::<I, C>(start, direction)?,
            y.load::<I, C>(start, direction)?,
        );
        add_terms::<I, C, A, _>(
            chunk_mut(lanes, offset),
            &singles(a),
            &singles(b),
            direction,
        )?;
        offset += C;
    }
    Some(offset)
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

/// Returns `operation` of each lane and `operand`.
#[inline]
fn with_operand<I: Isa, const N: usize>(
    lanes: &[f32; N],
    operand: f32,
    operation: Operation,
) -> Option<[f32; N]> {
    isa::binary::<I, N>(lanes, &[operand; N], operation)
}

/// Returns binary16 encodings widened exactly to binary32: in the packed
/// widening instructions where the instruction set has them, and in integer
/// and binary32 instructions otherwise, with the features of `I`.
#[inline]
pub fn widen_halves<I: Isa, const N: usize>(halves: &[u16; N]) -> Option<[u32; N]> {
    I::run(move || {
        if I::HALF {
            return isa::widen_halves::<I, N>(halves);
        }
        // `widen_half_lanes` takes the magnitude of a subnormal value, its
        // fraction times 2^-24. The conversion of the fraction and the
        // product by a power of two are exact.
        let fractions = from_integers::<I, _, N>(halves, |bits| i32::from(bits & 0x03FF))?;
        let subnormals = encodings(with_operand::<I, N>(
            &fractions,
            f32::from_bits(HALF_QUANTUM),
            Operation::Mul,
        )?);
        let mut widened = [0; N];
        widened
            .iter_mut()
            .zip(halves.iter().zip(subnormals))
            .for_each(|(lane, (&bits, subnormal))| *lane = widen_half_lanes(bits, subnormal));
        Some(widened)
    })
}

/// Returns binary32 values rounded to binary16 to nearest even, as
/// encodings: in the packed rounding instructions where the instruction set
/// has them, and otherwise in integer instructions and one binary32 sum of
/// each magnitude and 0.5, as `round_to_half_lanes` states. A NaN gives a
/// NaN.
#[inline]
pub fn narrow_halves<I: Isa, const N: usize>(singles: &[f32; N]) -> Option<[u16; N]> {
    if I::HALF {
        return isa::narrow_halves::<I, N>(singles);
    }
    let mut magnitudes = [0.0; N];
    magnitudes
        .iter_mut()
        .zip(singles)
        .for_each(|(magnitude, value)| *magnitude = f32::from_bits(value.to_bits() & 0x7FFF_FFFF));
    let sums = encodings(with_operand::<I, N>(
        &magnitudes,
        f32::from_bits(SUBNORMAL_BIAS),
        Operation::Add,
    )?);
    let mut halves = [0; N];
    halves
        .iter_mut()
        .zip(singles.iter().zip(sums))
        .for_each(|(half, (value, sum))| *half = round_to_half_lanes(value.to_bits(), sum));
    Some(halves)
}

/// Returns integers converted to binary32. `widen` gives each integer, which
/// is below `2^24` in magnitude, so the conversion is exact and the rounding
/// direction does not matter.
#[inline]
fn from_integers<I: Isa, T: Copy, const N: usize>(
    values: &[T; N],
    widen: impl Fn(T) -> i32,
) -> Option<[f32; N]> {
    let mut integers = [0; N];
    integers
        .iter_mut()
        .zip(values)
        .for_each(|(integer, &value)| *integer = widen(value));
    isa::from_int::<I, N>(&integers)
}

/// Returns unsigned 8-bit integers converted exactly to binary32, with the
/// features of `I`.
#[inline]
pub fn widen_codes<I: Isa, const N: usize>(codes: &[u8; N]) -> Option<[u32; N]> {
    I::run(move || Some(encodings(from_integers::<I, _, N>(codes, i32::from)?)))
}

/// Returns signed 8-bit integers converted exactly to binary32, each then
/// multiplied by `scale` and rounded in the direction `D`, with the
/// features of `I`.
#[inline]
pub fn widen_scaled_codes<I: Isa, const N: usize, D: Direction>(
    codes: &[i8; N],
    scale: u32,
    direction: D,
) -> Option<[u32; N]> {
    I::run(move || {
        let values = from_integers::<I, _, N>(codes, i32::from)?;
        let scale = f32::from_bits(scale);
        let scaled = if D::NEAREST {
            with_operand::<I, N>(&values, scale, Operation::Mul)?
        } else {
            let rounding = direction.rounding();
            isa::binary_rounded::<I, N>(&values, &[scale; N], Operation::Mul, rounding)?
        };
        Some(encodings(scaled))
    })
}

/// Adds the term of each pair of `x` and `y` into its lane of `lanes`, with
/// each step rounded in the direction `D`: in the forms of the environment
/// to nearest even, and in the forms with a rounding control otherwise.
#[inline]
fn add_terms<I: Isa, const N: usize, A: Accumulation, D: Direction>(
    lanes: &mut [f32; N],
    x: &[f32; N],
    y: &[f32; N],
    direction: D,
) -> Option<()> {
    let binary = |x: &[f32; N], y: &[f32; N], operation| {
        if D::NEAREST {
            isa::binary::<I, N>(x, y, operation)
        } else {
            isa::binary_rounded::<I, N>(x, y, operation, direction.rounding())
        }
    };
    let (left, right) = match A::TERM {
        Term::Value => {
            *lanes = binary(lanes, x, Operation::Add)?;
            return Some(());
        }
        Term::Product => (*x, *y),
        Term::SquareDifference => {
            let difference = binary(x, y, Operation::Sub)?;
            (difference, difference)
        }
    };
    *lanes = match (A::STEP, D::NEAREST) {
        (Step::Separate, _) => {
            let product = binary(&left, &right, Operation::Mul)?;
            binary(lanes, &product, Operation::Add)?
        }
        (Step::Fused, true) => isa::mul_add::<I, N>(&left, &right, lanes)?,
        (Step::Fused, false) => {
            isa::mul_add_rounded::<I, N>(&left, &right, lanes, direction.rounding())?
        }
    };
    Some(())
}

/// Returns the sum of the lanes by halves: while more than one lane is left,
/// lane `i` of the low half becomes lane `i` plus lane `i` of the high half.
/// `N` is a power of two.
#[inline]
fn sum_by_halves<I: Isa, const N: usize>(mut lanes: [f32; N]) -> Option<f32> {
    I::run(move || {
        let mut half = N / 2;
        while half > 0 {
            let (low, high) = lanes.split_at_mut(half);
            let (low_chunks, low_rest) = low.as_chunks_mut::<4>();
            let (high_chunks, high_rest) = high[..half].as_chunks::<4>();
            for (low, &high) in low_chunks.iter_mut().zip(high_chunks) {
                *low = I::binary_f32x4(*low, high, Operation::Add)?;
            }
            for (low, &high) in low_rest.iter_mut().zip(high_rest) {
                *low = I::binary_f32x4([*low; 4], [high; 4], Operation::Add)?[0];
            }
            half /= 2;
        }
        Some(lanes[0])
    })
}

/// Returns the sum of the lanes by halves, as `sum_by_halves` does, with
/// each sum rounded in the direction `rounding`: sixteen lanes at a time in
/// the form with a rounding control. The lanes past the last lane of a
/// chunk are +0, and their sums are not used. The caller selects the
/// function by the constant of its copy of the loop, so the loop of
/// `sum_by_halves` keeps no test of the direction.
#[inline]
fn sum_by_halves_rounded<I: Isa, const N: usize>(
    mut lanes: [f32; N],
    rounding: Rounding,
) -> Option<f32> {
    I::run(move || {
        let mut half = N / 2;
        while half > 0 {
            let (low, high) = lanes.split_at_mut(half);
            for (low, high) in low.chunks_mut(16).zip(high[..half].chunks(16)) {
                let (mut left, mut right) = ([0.0; 16], [0.0; 16]);
                left.iter_mut()
                    .zip(&*low)
                    .for_each(|(lane, &value)| *lane = value);
                right
                    .iter_mut()
                    .zip(high)
                    .for_each(|(lane, &value)| *lane = value);
                let sums = I::binary_rounded_f32x16(left, right, Operation::Add, rounding)?;
                low.iter_mut().zip(sums).for_each(|(lane, sum)| *lane = sum);
            }
            half /= 2;
        }
        Some(lanes[0])
    })
}
