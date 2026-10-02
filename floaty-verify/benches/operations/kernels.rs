//! The table of the slice kernels of `Lanes`, in nanoseconds per vector.
//!
//! Each row compares a host `f32` loop with 8 and with 32 accumulators in the
//! order of the kernels, the kernels of `Lanes<F32, 8>` and
//! `Lanes<F32, 32>` with the default mode, and the `_with` method of
//! `Lanes<F32, 32>`, which runs the engine. The host loop of a fused row
//! calls `f32::mul_add`, which is the FMA instruction only in a build with
//! FMA. The host has no binary16 type, so the binary16 row has no host
//! cells.

use std::hint::black_box;

use floaty::{BF16, F16, F32, Lanes, LittleEndian, ScaledCodes, mode};
use floaty_verify::random::SplitMix64;

use super::{COUNT, Table, measure};

/// The columns of the table.
const COLUMNS: Table<5> = Table {
    columns: [
        "host x 8",
        "host x 32",
        "lanes x 8",
        "lanes x 32",
        "with x 32",
    ],
};

/// A kernel of one pair of vectors, as a function pointer.
type Kernel<X, Y, R> = fn(X, Y) -> R;

/// Returns the sum of `term` of each pair of `x` and `y` in `N` host
/// accumulators, in the order of the kernels of `Lanes`: value `i` into
/// accumulator `i % N`, then the sum by halves.
fn host<T: Copy, const N: usize>(x: &[T], y: &[f32], term: impl Fn(f32, T, f32) -> f32) -> f32 {
    let mut lanes = [0.0_f32; N];
    let ((x_chunks, x_rest), (y_chunks, y_rest)) = (x.as_chunks::<N>(), y.as_chunks::<N>());
    for (a, b) in x_chunks.iter().zip(y_chunks) {
        for (lane, (&a, &b)) in lanes.iter_mut().zip(a.iter().zip(b)) {
            *lane = term(*lane, a, b);
        }
    }
    for (lane, (&a, &b)) in lanes.iter_mut().zip(x_rest.iter().zip(y_rest)) {
        *lane = term(*lane, a, b);
    }
    let mut half = N / 2;
    while half > 0 {
        let (low, high) = lanes.split_at_mut(half);
        low.iter_mut()
            .zip(&*high)
            .for_each(|(low, high)| *low += high);
        half /= 2;
    }
    lanes[0]
}

/// Measures one call of `kernel` per pair, cycling through the pairs, in
/// nanoseconds per call.
fn per_call<X: Copy, Y: Copy, R>(pairs: &[(X, Y)], kernel: Kernel<X, Y, R>) -> f64 {
    measure(|| {
        for (x, y) in pairs.iter().cycle().take(COUNT) {
            black_box(kernel(black_box(*x), black_box(*y)));
        }
    })
}

/// Measures one row of the table on `pairs`.
fn row<X: Copy, Y: Copy>(
    name: &str,
    pairs: &[(X, Y)],
    hosts: Option<[Kernel<X, Y, f32>; 2]>,
    lanes: [Kernel<X, Y, F32>; 3],
) {
    let [eight, thirty_two, with] = lanes;
    let times = [
        hosts.map(|[eight, _]| per_call(pairs, eight)),
        hosts.map(|[_, thirty_two]| per_call(pairs, thirty_two)),
        Some(per_call(pairs, eight)),
        Some(per_call(pairs, thirty_two)),
        Some(per_call(pairs, with)),
    ];
    COLUMNS.row(name, times);
}

/// Returns eight pairs of random binary32 vectors of `dimension` values
/// between 1/16 and 16, with random signs.
fn vectors(dimension: usize, seed: u64) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut random = SplitMix64::new(seed);
    let mut vector = || -> Vec<f32> {
        (0..dimension)
            .map(|_| {
                let bits = random.next_u64();
                let field = 123 + u32::try_from(bits % 8).expect("below 8");
                let fraction = u32::try_from((bits >> 8) & 0x7F_FFFF).expect("23 bits");
                f32::from_bits(u32::from(bits >> 63 == 1) << 31 | field << 23 | fraction)
            })
            .collect()
    };
    (0..8).map(|_| (vector(), vector())).collect()
}

/// The term of a dot product.
fn product(lane: f32, a: f32, b: f32) -> f32 {
    lane + a * b
}

/// The term of a square of a distance.
fn distance(lane: f32, a: f32, b: f32) -> f32 {
    let difference = a - b;
    lane + difference * difference
}

/// Measures the binary32 rows at `dimension`.
fn binary32_rows(dimension: usize, seed: u64) {
    let vectors = vectors(dimension, seed);
    let pairs: Vec<(&[f32], &[f32])> = vectors.iter().map(|(x, y)| (&x[..], &y[..])).collect();
    row(
        &format!("dot {dimension}"),
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, product),
            |x, y| host::<_, 32>(x, y, product),
        ]),
        [Lanes::<F32, 8>::dot, Lanes::<F32, 32>::dot, |x, y| {
            Lanes::<F32, 32>::dot_with(x, y, mode::Ieee).0
        }],
    );
    row(
        &format!("dot_fused {dimension}"),
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, |lane, a, b| a.mul_add(b, lane)),
            |x, y| host::<_, 32>(x, y, |lane, a, b| a.mul_add(b, lane)),
        ]),
        [
            Lanes::<F32, 8>::dot_fused,
            Lanes::<F32, 32>::dot_fused,
            |x, y| Lanes::<F32, 32>::dot_fused_with(x, y, mode::Ieee).0,
        ],
    );
    row(
        &format!("distance_square {dimension}"),
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, distance),
            |x, y| host::<_, 32>(x, y, distance),
        ]),
        [
            Lanes::<F32, 8>::distance_square,
            Lanes::<F32, 32>::distance_square,
            |x, y| Lanes::<F32, 32>::distance_square_with(x, y, mode::Ieee).0,
        ],
    );
    row(
        &format!("norm {dimension}"),
        &pairs,
        Some([
            |x, _| host::<_, 8>(x, x, product).sqrt(),
            |x, _| host::<_, 32>(x, x, product).sqrt(),
        ]),
        [
            |x, _| Lanes::<F32, 8>::norm(x),
            |x, _| Lanes::<F32, 32>::norm(x),
            |x, _| Lanes::<F32, 32>::norm_with(x, mode::Ieee).0,
        ],
    );
}

/// The term of a dot product of a bfloat16 value and a binary32 value.
fn bfloat_product(lane: f32, a: BF16, b: f32) -> f32 {
    lane + f32::from_bits(u32::from(a.to_bits()) << 16) * b
}

/// The term of a dot product of the little-endian bytes of a bfloat16 value
/// and a binary32 value.
fn bytes_product(lane: f32, a: [u8; 2], b: f32) -> f32 {
    lane + f32::from_bits(u32::from(u16::from_le_bytes(a)) << 16) * b
}

/// The term of a dot product of a code and a binary32 value.
fn code_product(lane: f32, a: u8, b: f32) -> f32 {
    lane + f32::from(a) * b
}

/// The term of a dot product of a signed code times 2^-6 and a binary32
/// value.
fn scaled_product(lane: f32, a: i8, b: f32) -> f32 {
    lane + (f32::from(a) * f32::from_bits(0x3C80_0000)) * b
}

/// Measures the dot products of stored formats with a binary32 query at
/// `dimension`.
fn format_rows(dimension: usize, seed: u64) {
    let vectors = vectors(dimension, seed);
    let bfloats: Vec<(Vec<BF16>, &[f32])> = vectors
        .iter()
        .map(|(x, y)| {
            (
                x.iter().map(|&value| F32::from(value).convert()).collect(),
                &y[..],
            )
        })
        .collect();
    let pairs: Vec<(&[BF16], &[f32])> = bfloats.iter().map(|(x, y)| (&x[..], *y)).collect();
    row(
        &format!("dot BF16 {dimension}"),
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, bfloat_product),
            |x, y| host::<_, 32>(x, y, bfloat_product),
        ]),
        [Lanes::<F32, 8>::dot, Lanes::<F32, 32>::dot, |x, y| {
            Lanes::<F32, 32>::dot_with(x, y, mode::Ieee).0
        }],
    );
    let rows: Vec<(Vec<[u8; 2]>, &[f32])> = bfloats
        .iter()
        .map(|(x, y)| {
            (
                x.iter()
                    .map(|value| value.to_bits().to_le_bytes())
                    .collect(),
                *y,
            )
        })
        .collect();
    let pairs: Vec<(&[[u8; 2]], &[f32])> = rows.iter().map(|(x, y)| (&x[..], *y)).collect();
    row(
        &format!("dot LittleEndian<BF16> {dimension}"),
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, bytes_product),
            |x, y| host::<_, 32>(x, y, bytes_product),
        ]),
        [
            |x, y| Lanes::<F32, 8>::dot(LittleEndian::<BF16>::new(x.as_flattened()), y),
            |x, y| Lanes::<F32, 32>::dot(LittleEndian::<BF16>::new(x.as_flattened()), y),
            |x, y| {
                Lanes::<F32, 32>::dot_with(
                    LittleEndian::<BF16>::new(x.as_flattened()),
                    y,
                    mode::Ieee,
                )
                .0
            },
        ],
    );
    code_rows(dimension, &vectors);
}

/// Measures the dot products of binary16 values and of codes with a binary32
/// query at `dimension`.
fn code_rows(dimension: usize, vectors: &[(Vec<f32>, Vec<f32>)]) {
    let halves: Vec<(Vec<F16>, &[f32])> = vectors
        .iter()
        .map(|(x, y)| {
            (
                x.iter().map(|&value| F32::from(value).convert()).collect(),
                &y[..],
            )
        })
        .collect();
    let pairs: Vec<(&[F16], &[f32])> = halves.iter().map(|(x, y)| (&x[..], *y)).collect();
    row(
        &format!("dot F16 {dimension}"),
        &pairs,
        None,
        [Lanes::<F32, 8>::dot, Lanes::<F32, 32>::dot, |x, y| {
            Lanes::<F32, 32>::dot_with(x, y, mode::Ieee).0
        }],
    );
    let codes: Vec<(Vec<u8>, &[f32])> = vectors
        .iter()
        .map(|(x, y)| {
            (
                x.iter()
                    .map(|value| value.to_bits().to_le_bytes()[1])
                    .collect(),
                &y[..],
            )
        })
        .collect();
    let pairs: Vec<(&[u8], &[f32])> = codes.iter().map(|(x, y)| (&x[..], *y)).collect();
    row(
        &format!("dot u8 {dimension}"),
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, code_product),
            |x, y| host::<_, 32>(x, y, code_product),
        ]),
        [Lanes::<F32, 8>::dot, Lanes::<F32, 32>::dot, |x, y| {
            Lanes::<F32, 32>::dot_with(x, y, mode::Ieee).0
        }],
    );
    let signed: Vec<(Vec<i8>, &[f32])> = codes
        .iter()
        .map(|(x, y)| (x.iter().map(|&code| code.cast_signed()).collect(), *y))
        .collect();
    let pairs: Vec<(&[i8], &[f32])> = signed.iter().map(|(x, y)| (&x[..], *y)).collect();
    row(
        &format!("dot ScaledCodes {dimension}"),
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, scaled_product),
            |x, y| host::<_, 32>(x, y, scaled_product),
        ]),
        [
            |x, y| {
                Lanes::<F32, 8>::dot(
                    ScaledCodes {
                        codes: x,
                        scale: F32::from_bits(0x3C80_0000),
                    },
                    y,
                )
            },
            |x, y| {
                Lanes::<F32, 32>::dot(
                    ScaledCodes {
                        codes: x,
                        scale: F32::from_bits(0x3C80_0000),
                    },
                    y,
                )
            },
            |x, y| {
                let codes = ScaledCodes {
                    codes: x,
                    scale: F32::from_bits(0x3C80_0000),
                };
                Lanes::<F32, 32>::dot_with(codes, y, mode::Ieee).0
            },
        ],
    );
}

/// Measures dot products of eight values: one call each, and the rows of
/// `dot_rows` in nanoseconds per row, as the tables of product quantization
/// compute them.
fn short_rows(seed: u64) {
    let vectors = vectors(256 * 8, seed);
    let pairs: Vec<(&[f32], &[f32])> = vectors
        .iter()
        .flat_map(|(x, y)| x.as_chunks::<8>().0.iter().map(|x| (&x[..], &y[..8])))
        .collect();
    row(
        "dot 8",
        &pairs,
        Some([
            |x, y| host::<_, 8>(x, y, product),
            |x, y| host::<_, 32>(x, y, product),
        ]),
        [Lanes::<F32, 8>::dot, Lanes::<F32, 32>::dot, |x, y| {
            Lanes::<F32, 32>::dot_with(x, y, mode::Ieee).0
        }],
    );
    let per_row = |rows: fn(&[f32], &[f32], &mut [f32])| {
        let mut out = vec![0.0_f32; 256];
        // Four calls of 256 rows are the 1,024 operations of a batch.
        measure(|| {
            for (x, y) in vectors.iter().take(4) {
                rows(black_box(x), black_box(&y[..8]), &mut out);
                black_box(&out);
            }
        })
    };
    let times = [
        None,
        None,
        Some(per_row(|x, y, out| Lanes::<F32, 8>::dot_rows(x, y, out))),
        Some(per_row(|x, y, out| Lanes::<F32, 32>::dot_rows(x, y, out))),
        Some(per_row(|x, y, out| {
            let _ = Lanes::<F32, 32>::dot_rows_with(x, y, out, mode::Ieee);
        })),
    ];
    COLUMNS.row("dot_rows 256 x 8, per row", times);
}

/// Prints the table.
pub fn table() {
    println!();
    println!("Slice kernels of Lanes, nanoseconds per vector.");
    println!();
    COLUMNS.header("Kernel");
    binary32_rows(1024, 61);
    binary32_rows(2560, 62);
    format_rows(1280, 63);
    short_rows(64);
}
