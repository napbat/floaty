//! The table of blocks, in nanoseconds per vector of 1,280 values.
//!
//! Each row compares a host `f32` loop of the same steps, the views of
//! `floaty::elementwise` where they express the chain, `F32::map` of the
//! chain, and a loop of the scalar operations of `F32`, one value at a time.
//! A fused multiply-add has no view, and its host loop calls the `fmaf` of
//! the C library in a build without FMA, so those rows have no such cell.

use std::hint::black_box;

use floaty::block::{Chain, Steps};
use floaty::elementwise::{
    Difference, MaximumNumber, MinimumNumber, Product, RoundToIntegral, Splat, Sum,
};
use floaty::{F32, Lanes, Rounding};
use floaty_verify::random::SplitMix64;

use super::{Table, measure};

/// The columns of the table.
const COLUMNS: Table<4> = Table {
    columns: ["host", "views x 32", "block", "scalar F32"],
};

/// The values of a vector.
const DIMENSION: usize = 1_280;

/// The vectors of one measurement.
const VECTORS: usize = 64;

/// The lanes of the views.
type Single = Lanes<F32, 32>;

/// `x * keep + y * eta`, each step rounded: the k-means update.
struct Update;

impl Chain<2, 2> for Update {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [keep, eta]: [S; 2]) -> S {
        x * keep + y * eta
    }
}

/// `x * keep + y * eta`, with the second product and the sum rounded once.
struct FusedUpdate;

impl Chain<2, 2> for FusedUpdate {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [keep, eta]: [S; 2]) -> S {
        y.mul_add(eta, x * keep)
    }
}

/// `(x - mean) * inverse * gamma + beta`, each step rounded.
struct Normalize;

impl Chain<1, 4> for Normalize {
    fn apply<S: Steps>(&self, [x]: [S; 1], [mean, inverse, gamma, beta]: [S; 4]) -> S {
        (x - mean) * inverse * gamma + beta
    }
}

/// `minimum_number(maximum_number(x, low), y)`: a lower bound, then the
/// smaller of the result and `y`.
struct Bounds;

impl Chain<2, 1> for Bounds {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [low]: [S; 1]) -> S {
        x.maximum_number(low).minimum_number(y)
    }
}

/// `round_to_integral(x * inverse) * step`: a value quantized to a
/// multiple of `step`.
struct Quantize;

impl Chain<1, 2> for Quantize {
    fn apply<S: Steps>(&self, [x]: [S; 1], [inverse, step]: [S; 2]) -> S {
        (x * inverse).round_to_integral() * step
    }
}

/// `sqrt(|x * scale + offset|) / y - bias`: a score after a kernel.
struct Score;

impl Chain<2, 3> for Score {
    fn apply<S: Steps>(&self, [x, y]: [S; 2], [scale, offset, bias]: [S; 3]) -> S {
        x.mul_add(scale, offset).abs().sqrt() / y - bias
    }
}

/// Returns `VECTORS` random vectors of values between 1/16 and 16, with
/// random signs.
fn vectors(seed: u64) -> Vec<Vec<f32>> {
    let mut random = SplitMix64::new(seed);
    (0..VECTORS)
        .map(|_| {
            (0..DIMENSION)
                .map(|_| {
                    let bits = random.next_u64();
                    let field = 123 + u32::try_from(bits % 8).expect("below 8");
                    let fraction = u32::try_from((bits >> 8) & 0x7F_FFFF).expect("23 bits");
                    f32::from_bits(u32::from(bits >> 63 == 1) << 31 | field << 23 | fraction)
                })
                .collect()
        })
        .collect()
}

/// Measures one call of `operation` per pair of vectors, cycling through
/// the vectors, in nanoseconds per call.
fn per_vector(vectors: &[Vec<f32>], mut operation: impl FnMut(&[f32], &[f32])) -> f64 {
    measure(|| {
        for (x, y) in vectors
            .iter()
            .zip(vectors.iter().cycle().skip(1))
            .cycle()
            .take(super::COUNT)
        {
            operation(black_box(x), black_box(y));
        }
    })
}

/// One measured operation on two vectors, or `None` for a cell that the row
/// does not have.
type Cell<'a> = Option<&'a mut dyn FnMut(&[f32], &[f32])>;

/// Measures the cells of one row that it has.
fn row(name: &str, vectors: &[Vec<f32>], cells: [Cell<'_>; 4]) {
    COLUMNS.row(
        name,
        cells.map(|cell| cell.map(|operation| per_vector(vectors, operation))),
    );
}

/// Measures the k-means update, each step rounded.
fn update_row(vectors: &[Vec<f32>]) {
    let p = [F32::from_bits(0x3F7F_0000), F32::from_bits(0x3B80_0000)];
    let [keep, eta] = p.map(f32::from);
    let mut out = vec![0.0_f32; DIMENSION];
    let mut host = |x: &[f32], y: &[f32]| {
        for (o, (&x, &y)) in out.iter_mut().zip(x.iter().zip(y)) {
            *o = x * keep + y * eta;
        }
        black_box(&out);
    };
    let mut views_out = vec![0.0_f32; DIMENSION];
    let mut views = |x: &[f32], y: &[f32]| {
        let splat = |value| Splat::new(value, DIMENSION);
        Single::store(
            Sum(Product(x, splat(p[0])), Product(y, splat(p[1]))),
            &mut views_out[..],
        );
    };
    let mut block_out = vec![0.0_f32; DIMENSION];
    let mut block = |x: &[f32], y: &[f32]| F32::map(&Update, [x, y], p, &mut block_out[..]);
    let mut scalar_out = vec![0.0_f32; DIMENSION];
    let mut scalar = |x: &[f32], y: &[f32]| {
        for (o, (&x, &y)) in scalar_out.iter_mut().zip(x.iter().zip(y)) {
            *o = f32::from(F32::from(x) * p[0] + F32::from(y) * p[1]);
        }
        black_box(&scalar_out);
    };
    row(
        "update x * keep + y * eta",
        vectors,
        [
            Some(&mut host),
            Some(&mut views),
            Some(&mut block),
            Some(&mut scalar),
        ],
    );
}

/// Measures the k-means update with a fused step.
fn fused_update_row(vectors: &[Vec<f32>]) {
    let p = [F32::from_bits(0x3F7F_0000), F32::from_bits(0x3B80_0000)];
    let mut block_out = vec![0.0_f32; DIMENSION];
    let mut block = |x: &[f32], y: &[f32]| F32::map(&FusedUpdate, [x, y], p, &mut block_out[..]);
    let mut scalar_out = vec![0.0_f32; DIMENSION];
    let mut scalar = |x: &[f32], y: &[f32]| {
        for (o, (&x, &y)) in scalar_out.iter_mut().zip(x.iter().zip(y)) {
            *o = f32::from(F32::from(y).mul_add(p[1], F32::from(x) * p[0]));
        }
        black_box(&scalar_out);
    };
    row(
        "update fused",
        vectors,
        [None, None, Some(&mut block), Some(&mut scalar)],
    );
}

/// Measures the normalization of a vector.
fn normalize_row(vectors: &[Vec<f32>]) {
    let p = [0x3F00_0000, 0x3FC0_0000, 0x3F40_0000, 0xBE80_0000].map(F32::from_bits);
    let [mean, inverse, gamma, beta] = p.map(f32::from);
    let mut out = vec![0.0_f32; DIMENSION];
    let mut host = |x: &[f32], _: &[f32]| {
        for (o, &x) in out.iter_mut().zip(x) {
            *o = (x - mean) * inverse * gamma + beta;
        }
        black_box(&out);
    };
    let mut views_out = vec![0.0_f32; DIMENSION];
    let mut views = |x: &[f32], _: &[f32]| {
        let splat = |value| Splat::new(value, DIMENSION);
        let view = Sum(
            Product(
                Product(Difference(x, splat(p[0])), splat(p[1])),
                splat(p[2]),
            ),
            splat(p[3]),
        );
        Single::store(view, &mut views_out[..]);
    };
    let mut block_out = vec![0.0_f32; DIMENSION];
    let mut block = |x: &[f32], _: &[f32]| F32::map(&Normalize, [x], p, &mut block_out[..]);
    let mut scalar_out = vec![0.0_f32; DIMENSION];
    let mut scalar = |x: &[f32], _: &[f32]| {
        for (o, &x) in scalar_out.iter_mut().zip(x) {
            *o = f32::from((F32::from(x) - p[0]) * p[1] * p[2] + p[3]);
        }
        black_box(&scalar_out);
    };
    row(
        "normalize (x - mean) * inverse * gamma + beta",
        vectors,
        [
            Some(&mut host),
            Some(&mut views),
            Some(&mut block),
            Some(&mut scalar),
        ],
    );
}

/// Measures a lower bound and then the smaller of the result and a second
/// vector, by the number operations, which take the other operand of a NaN.
fn bounds_row(vectors: &[Vec<f32>]) {
    let low = F32::from_bits(0xBF80_0000);
    let host_low = f32::from(low);
    let mut out = vec![0.0_f32; DIMENSION];
    let mut host = |x: &[f32], y: &[f32]| {
        for (o, (&x, &y)) in out.iter_mut().zip(x.iter().zip(y)) {
            *o = x.max(host_low).min(y);
        }
        black_box(&out);
    };
    let mut views_out = vec![0.0_f32; DIMENSION];
    let mut views = |x: &[f32], y: &[f32]| {
        let view = MinimumNumber(MaximumNumber(x, Splat::new(low, DIMENSION)), y);
        Single::store(view, &mut views_out[..]);
    };
    let mut block_out = vec![0.0_f32; DIMENSION];
    let mut block = |x: &[f32], y: &[f32]| F32::map(&Bounds, [x, y], [low], &mut block_out[..]);
    let mut scalar_out = vec![0.0_f32; DIMENSION];
    let mut scalar = |x: &[f32], y: &[f32]| {
        for (o, (&x, &y)) in scalar_out.iter_mut().zip(x.iter().zip(y)) {
            *o = f32::from(
                F32::from(x)
                    .maximum_number(low)
                    .minimum_number(F32::from(y)),
            );
        }
        black_box(&scalar_out);
    };
    row(
        "bounds minimum_number(maximum_number(x, low), y)",
        vectors,
        [
            Some(&mut host),
            Some(&mut views),
            Some(&mut block),
            Some(&mut scalar),
        ],
    );
}

/// Measures a value quantized to a multiple of a step, through the
/// integral value to nearest even.
fn quantize_row(vectors: &[Vec<f32>]) {
    let (inverse, step) = (F32::from_bits(0x4080_0000), F32::from_bits(0x3E80_0000));
    let (host_inverse, host_step) = (f32::from(inverse), f32::from(step));
    let mut out = vec![0.0_f32; DIMENSION];
    let mut host = |x: &[f32], _: &[f32]| {
        for (o, &x) in out.iter_mut().zip(x) {
            *o = (x * host_inverse).round_ties_even() * host_step;
        }
        black_box(&out);
    };
    let mut views_out = vec![0.0_f32; DIMENSION];
    let mut views = |x: &[f32], _: &[f32]| {
        let integral = RoundToIntegral {
            values: Product(x, Splat::new(inverse, DIMENSION)),
            rounding: Rounding::TiesToEven,
        };
        Single::store(
            Product(integral, Splat::new(step, DIMENSION)),
            &mut views_out[..],
        );
    };
    let mut block_out = vec![0.0_f32; DIMENSION];
    let mut block =
        |x: &[f32], _: &[f32]| F32::map(&Quantize, [x], [inverse, step], &mut block_out[..]);
    let mut scalar_out = vec![0.0_f32; DIMENSION];
    let mut scalar = |x: &[f32], _: &[f32]| {
        for (o, &x) in scalar_out.iter_mut().zip(x) {
            *o = f32::from((F32::from(x) * inverse).round_to_integral() * step);
        }
        black_box(&scalar_out);
    };
    row(
        "quantize round_to_integral(x * inverse) * step",
        vectors,
        [
            Some(&mut host),
            Some(&mut views),
            Some(&mut block),
            Some(&mut scalar),
        ],
    );
}

/// Measures a score of each value, with a fused step, a square root, and a
/// quotient, by `map` and by `evaluate` of each value.
fn score_rows(vectors: &[Vec<f32>]) {
    let p = [0x3F00_0000, 0x3F80_0000, 0x3E80_0000].map(F32::from_bits);
    let mut block_out = vec![0.0_f32; DIMENSION];
    let mut block = |x: &[f32], y: &[f32]| F32::map(&Score, [x, y], p, &mut block_out[..]);
    let mut scalar_out = vec![0.0_f32; DIMENSION];
    let mut scalar = |x: &[f32], y: &[f32]| {
        for (o, (&x, &y)) in scalar_out.iter_mut().zip(x.iter().zip(y)) {
            let root = F32::from(x).mul_add(p[0], p[1]).abs().sqrt();
            *o = f32::from(root / F32::from(y) - p[2]);
        }
        black_box(&scalar_out);
    };
    row(
        "score sqrt(|x * a + b|) / y - c",
        vectors,
        [None, None, Some(&mut block), Some(&mut scalar)],
    );
    let mut evaluate_out = vec![0.0_f32; DIMENSION];
    let mut evaluate = |x: &[f32], y: &[f32]| {
        for (o, (&x, &y)) in evaluate_out.iter_mut().zip(x.iter().zip(y)) {
            *o = f32::from(F32::evaluate(&Score, [F32::from(x), F32::from(y)], p));
        }
        black_box(&evaluate_out);
    };
    row(
        "score, evaluate of each value",
        vectors,
        [None, None, Some(&mut evaluate), None],
    );
}

/// Prints the table.
pub fn table() {
    println!();
    println!("Blocks, nanoseconds per vector of {DIMENSION}.");
    println!();
    COLUMNS.header("Chain");
    let vectors = vectors(83);
    update_row(&vectors);
    fused_update_row(&vectors);
    normalize_row(&vectors);
    bounds_row(&vectors);
    quantize_row(&vectors);
    score_rows(&vectors);
}
