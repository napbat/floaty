//! The table of the elementwise slice operations of `Lanes`, in nanoseconds
//! per vector of 1,280 values.
//!
//! Each row compares a host `f32` loop of the same steps, the operation of
//! `Lanes<F32, 32>` with the default mode, its `_with` method, which runs the
//! engine, and a loop of the scalar operations of `F32`, one value at a time.

use std::cell::Cell;
use std::hint::black_box;

use floaty::elementwise::{
    Abs, Difference, Maximum, MinimumNumber, Minimum, Product, Quotient, RoundToIntegral, Splat,
    Sum,
};
use floaty::{BF16, F32, Lanes, Rounding, ToInt, mode};
use floaty_verify::random::SplitMix64;

use super::{Table, measure};

/// The columns of the table.
const COLUMNS: Table<4> = Table {
    columns: ["host", "lanes x 32", "with x 32", "scalar F32"],
};

/// The values of a vector.
const DIMENSION: usize = 1_280;

/// The vectors of one measurement.
const VECTORS: usize = 64;

/// The lanes of the operations.
type Single = Lanes<F32, 32>;

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

/// Measures one call of `operation` per vector, cycling through the vectors,
/// in nanoseconds per call.
fn per_vector(vectors: &[Vec<f32>], mut operation: impl FnMut(&[f32])) -> f64 {
    measure(|| {
        for vector in vectors.iter().cycle().take(super::COUNT) {
            operation(black_box(vector));
        }
    })
}

/// Measures one row: the host loop, the operation, its `_with` method, and
/// the scalar loop.
fn row(name: &str, vectors: &[Vec<f32>], cells: [&mut dyn FnMut(&[f32]); 4]) {
    let [host, lanes, with, scalar] = cells;
    COLUMNS.row(
        name,
        [
            Some(per_vector(vectors, host)),
            Some(per_vector(vectors, lanes)),
            Some(per_vector(vectors, with)),
            Some(per_vector(vectors, scalar)),
        ],
    );
}

/// Measures the k-means update `c = c * keep + p * eta` into a center.
fn update_row(vectors: &[Vec<f32>]) {
    let (keep, eta) = (F32::from_bits(0x3F7F_0000), F32::from_bits(0x3B80_0000));
    let (host_keep, host_eta) = (f32::from(keep), f32::from(eta));
    let mut center = vectors[0].clone();
    let mut host = |point: &[f32]| {
        for (c, &p) in center.iter_mut().zip(point) {
            *c = *c * host_keep + p * host_eta;
        }
        black_box(&center);
    };
    let mut lanes_center = vectors[0].clone();
    let mut lanes = |point: &[f32]| {
        let cells = Cell::from_mut(&mut lanes_center[..]).as_slice_of_cells();
        let view = Sum(
            Product(cells, Splat::new(keep, DIMENSION)),
            Product(point, Splat::new(eta, DIMENSION)),
        );
        Single::store(view, cells);
    };
    let mut with_center = vectors[0].clone();
    let mut with = |point: &[f32]| {
        let cells = Cell::from_mut(&mut with_center[..]).as_slice_of_cells();
        let view = Sum(
            Product(cells, Splat::new(keep, DIMENSION)),
            Product(point, Splat::new(eta, DIMENSION)),
        );
        Single::store_with(view, cells, mode::Ieee);
    };
    let mut scalar_center = vectors[0].clone();
    let mut scalar = |point: &[f32]| {
        for (c, &p) in scalar_center.iter_mut().zip(point) {
            *c = f32::from(F32::from(*c) * keep + F32::from(p) * eta);
        }
        black_box(&scalar_center);
    };
    row(
        "update x * keep + y * eta",
        vectors,
        [&mut host, &mut lanes, &mut with, &mut scalar],
    );
}

/// Measures the scalar quantization of a vector to 8-bit codes.
fn quantize_row(vectors: &[Vec<f32>]) {
    let lows = vec![-16.0_f32; DIMENSION];
    let ranges = vec![32.0_f32; DIMENSION];
    let (zero, one, levels) = (
        F32::from_bits(0),
        F32::from_bits(0x3F80_0000),
        F32::from_bits(0x437F_0000),
    );
    let mut codes = vec![0.0_f32; DIMENSION];
    let mut host = |x: &[f32]| {
        for ((code, &x), (&low, &range)) in codes.iter_mut().zip(x).zip(lows.iter().zip(&ranges)) {
            let unit = ((x - low) / range).clamp(0.0, 1.0);
            *code = (unit * 255.0).round();
        }
        black_box(&codes);
    };
    let mut scaled = vec![ToInt::Value(0_u8); DIMENSION];
    let view = |x: &[f32]| {
        let position = Quotient(Difference(x, &lows[..]), &ranges[..]);
        let unit = Minimum(
            Maximum(position, Splat::new(zero, DIMENSION)),
            Splat::new(one, DIMENSION),
        );
        RoundToIntegral {
            values: Product(unit, Splat::new(levels, DIMENSION)),
            rounding: Rounding::TiesToAway,
        }
    };
    let mut lanes = |x: &[f32]| {
        Single::to_int_slice(view(x), &mut scaled);
        black_box(&scaled);
    };
    let mut with_scaled = vec![ToInt::Value(0_u8); DIMENSION];
    let mut with = |x: &[f32]| {
        Single::to_int_slice_with(view(x), &mut with_scaled, mode::Ieee);
        black_box(&with_scaled);
    };
    let mut scalar_codes = vec![0_u8; DIMENSION];
    let mut scalar = |x: &[f32]| {
        for ((code, &x), (&low, &range)) in
            scalar_codes.iter_mut().zip(x).zip(lows.iter().zip(&ranges))
        {
            let position = (F32::from(x) - F32::from(low)) / F32::from(range);
            let unit = position.maximum(zero).minimum(one) * levels;
            let (rounded, _) = unit.round_to_integral_with(Rounding::TiesToAway);
            *code = match rounded.to_int::<u8>() {
                ToInt::Value(code) => code,
                _ => u8::MAX,
            };
        }
        black_box(&scalar_codes);
    };
    row(
        "quantize to codes",
        vectors,
        [&mut host, &mut lanes, &mut with, &mut scalar],
    );
}

/// Measures the minimum of a running vector and each vector.
fn minimum_row(vectors: &[Vec<f32>]) {
    let mut host_mins = vec![f32::MAX; DIMENSION];
    let mut host = |x: &[f32]| {
        for (low, &x) in host_mins.iter_mut().zip(x) {
            *low = low.min(x);
        }
        black_box(&host_mins);
    };
    let mut mins = vec![f32::MAX; DIMENSION];
    let mut lanes = |x: &[f32]| {
        let cells = Cell::from_mut(&mut mins[..]).as_slice_of_cells();
        Single::store(MinimumNumber(cells, x), cells);
    };
    let mut with_mins = vec![f32::MAX; DIMENSION];
    let mut with = |x: &[f32]| {
        let cells = Cell::from_mut(&mut with_mins[..]).as_slice_of_cells();
        Single::store_with(MinimumNumber(cells, x), cells, mode::Ieee);
    };
    let mut scalar_mins = vec![f32::MAX; DIMENSION];
    let mut scalar = |x: &[f32]| {
        for (low, &x) in scalar_mins.iter_mut().zip(x) {
            *low = f32::from(F32::from(*low).minimum_number(F32::from(x)));
        }
        black_box(&scalar_mins);
    };
    row(
        "minimum_number into cells",
        vectors,
        [&mut host, &mut lanes, &mut with, &mut scalar],
    );
}

/// Measures the largest magnitude of a vector.
fn magnitude_row(vectors: &[Vec<f32>]) {
    let mut host = |x: &[f32]| {
        black_box(x.iter().fold(0.0_f32, |largest, &x| largest.max(x.abs())));
    };
    let mut lanes = |x: &[f32]| {
        black_box(Single::maximum_of(Abs(x)));
    };
    let mut with = |x: &[f32]| {
        black_box(Single::maximum_of_with(Abs(x), mode::Ieee));
    };
    let mut scalar = |x: &[f32]| {
        let largest = x.iter().fold(F32::from_bits(0), |largest, &x| {
            largest.maximum(F32::from(x).abs())
        });
        black_box(largest);
    };
    row(
        "maximum_of Abs",
        vectors,
        [&mut host, &mut lanes, &mut with, &mut scalar],
    );
}

/// Measures the conversion of host values to bfloat16.
fn conversion_row(vectors: &[Vec<f32>]) {
    let mut out = vec![BF16::from_bits(0); DIMENSION];
    let mut lanes = |x: &[f32]| {
        Single::convert_slice(x, &mut out);
        black_box(&out);
    };
    let mut with_out = vec![BF16::from_bits(0); DIMENSION];
    let mut with = |x: &[f32]| {
        Single::convert_slice_with(x, &mut with_out, mode::Ieee);
        black_box(&with_out);
    };
    let mut scalar_out = vec![BF16::from_bits(0); DIMENSION];
    let mut scalar = |x: &[f32]| {
        for (out, &x) in scalar_out.iter_mut().zip(x) {
            *out = F32::from(x).convert();
        }
        black_box(&scalar_out);
    };
    COLUMNS.row(
        "convert_slice f32 to BF16",
        [
            None,
            Some(per_vector(vectors, &mut lanes)),
            Some(per_vector(vectors, &mut with)),
            Some(per_vector(vectors, &mut scalar)),
        ],
    );
}

/// Prints the table.
pub fn table() {
    println!();
    println!("Elementwise slice operations of Lanes, nanoseconds per vector of {DIMENSION}.");
    println!();
    COLUMNS.header("Operation");
    let vectors = vectors(71);
    update_row(&vectors);
    quantize_row(&vectors);
    minimum_row(&vectors);
    magnitude_row(&vectors);
    conversion_row(&vectors);
}
