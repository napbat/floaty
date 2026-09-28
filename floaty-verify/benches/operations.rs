//! Measures the time of floaty operations for each format, and of the host
//! `f32` and `f64` types and `rustc_apfloat` as baselines.
//!
//! Run it with `cargo bench -p floaty-verify --bench operations`. Each case
//! runs a batch of 1,024 operations on seeded random normal operands near
//! 1.0, many times, and reports the median of 11 samples in nanoseconds per
//! operation. The operands avoid special values, so the numbers show the
//! common path of each operation. The operator columns of binary32 and
//! binary64 take the host fast path on x86-64. The `add_with` column always
//! measures the engine, with the flags.

use std::hint::black_box;
use std::time::{Duration, Instant};

use floaty::format::Standard;
use floaty::{BF16, Binary, Env, Exact, F16, F32, F64, F80, F128, Float, Fnuz, NoInf, X87};
use floaty_verify::random::SplitMix64;
use rustc_apfloat::ieee::{BFloat, Double, Half, Quad, Single, X87DoubleExtended};
use rustc_apfloat::{Float as _, FloatConvert as _};

/// The operations of one batch.
const COUNT: usize = 1_024;

/// The samples of one case.
const SAMPLES: usize = 11;

/// The time that one sample runs at least.
const SAMPLE_TIME: Duration = Duration::from_millis(4);

/// The operations that each row measures, in column order.
const COLUMNS: [&str; 11] = [
    "add",
    "add_with",
    "mul",
    "div",
    "div_with",
    "sqrt",
    "mul_add",
    "to_f64",
    "to_i64",
    "round_int",
    "decode",
];

/// Returns the median time of one operation of `batch`, which runs `COUNT`
/// operations, in nanoseconds.
fn measure(mut batch: impl FnMut()) -> f64 {
    let mut repeats = 1_u32;
    loop {
        let start = Instant::now();
        for _ in 0..repeats {
            batch();
        }
        if start.elapsed() >= SAMPLE_TIME {
            break;
        }
        repeats *= 2;
    }
    let mut samples: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..repeats {
                batch();
            }
            start.elapsed().as_secs_f64() * 1e9
        })
        .collect();
    samples.sort_by(f64::total_cmp);
    let count = f64::from(u32::try_from(COUNT).expect("the batch size fits a u32"));
    samples[SAMPLES / 2] / f64::from(repeats) / count
}

/// Returns `COUNT` random normal values near 1.0 of a format, rounded from
/// random 64-bit significands.
fn operands<S: Standard<W>, const W: usize>(random: &mut SplitMix64) -> Vec<Float<S, W>> {
    (0..COUNT)
        .map(|_| {
            let exact = Exact {
                negative: random.next_u64() % 2 == 0,
                exponent: -63
                    + i32::try_from(random.next_u64() % 8).expect("a shift below 8 fits an i32"),
                significand: [random.next_u64() | 1 << 63],
                sticky: false,
            };
            Float::round(exact, Env::IEEE).0
        })
        .collect()
}

/// Returns `COUNT` positive values of a format, for the square root.
fn positive<S: Standard<W>, const W: usize>(values: &[Float<S, W>]) -> Vec<Float<S, W>> {
    values.iter().map(|value| value.abs()).collect()
}

/// Prints one row of the table.
fn row(name: &str, times: &[Option<f64>]) {
    let cells: Vec<String> = times
        .iter()
        .map(|time| time.map_or_else(|| "-".to_owned(), |time| format!("{time:.1}")))
        .collect();
    println!("| {name} | {} |", cells.join(" | "));
}

/// Measures every column for one floaty format.
fn floaty_row<S: Standard<W>, const W: usize>(name: &str, seed: u64) {
    let mut random = SplitMix64::new(seed);
    let (a, b, c) = (
        operands::<S, W>(&mut random),
        operands::<S, W>(&mut random),
        operands::<S, W>(&mut random),
    );
    let roots = positive(&a);
    let binary = |operation: fn(Float<S, W>, Float<S, W>) -> Float<S, W>| {
        measure(|| {
            for (&x, &y) in a.iter().zip(&b) {
                black_box(operation(black_box(x), black_box(y)));
            }
        })
    };
    let times = [
        Some(binary(|x, y| x + y)),
        Some(binary(|x, y| x.add_with(y, Env::IEEE).0)),
        Some(binary(|x, y| x * y)),
        Some(binary(|x, y| x / y)),
        Some(binary(|x, y| x.div_with(y, Env::IEEE).0)),
        Some(measure(|| {
            for &x in &roots {
                black_box(black_box(x).sqrt());
            }
        })),
        Some(measure(|| {
            for ((&x, &y), &z) in a.iter().zip(&b).zip(&c) {
                black_box(black_box(x).mul_add(black_box(y), black_box(z)));
            }
        })),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).convert::<F64>());
            }
        })),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).to_int::<i64>());
            }
        })),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).round_to_integral());
            }
        })),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).decode::<8>());
            }
        })),
    ];
    row(name, &times);
}

/// Measures the host type `$host` for comparison, on the operands of the
/// floaty format `Binary<$exponent>` at width `$width`.
macro_rules! host_row {
    ($name:literal, $host:ty, $exponent:literal, $width:literal, $seed:literal) => {{
        let mut random = SplitMix64::new($seed);
        let values = |random: &mut SplitMix64| -> Vec<$host> {
            operands::<Binary<$exponent>, $width>(random)
                .into_iter()
                .map(|value| <$host>::from_bits(value.to_bits()))
                .collect()
        };
        let (a, b, c) = (
            values(&mut random),
            values(&mut random),
            values(&mut random),
        );
        let roots: Vec<$host> = a.iter().map(|value| value.abs()).collect();
        let times = [
            Some(measure(|| {
                for (&x, &y) in a.iter().zip(&b) {
                    black_box(black_box(x) + black_box(y));
                }
            })),
            None,
            Some(measure(|| {
                for (&x, &y) in a.iter().zip(&b) {
                    black_box(black_box(x) * black_box(y));
                }
            })),
            Some(measure(|| {
                for (&x, &y) in a.iter().zip(&b) {
                    black_box(black_box(x) / black_box(y));
                }
            })),
            None,
            Some(measure(|| {
                for &x in &roots {
                    black_box(black_box(x).sqrt());
                }
            })),
            Some(measure(|| {
                for ((&x, &y), &z) in a.iter().zip(&b).zip(&c) {
                    black_box(black_box(x).mul_add(black_box(y), black_box(z)));
                }
            })),
            None,
            None,
            Some(measure(|| {
                for &x in &a {
                    black_box(black_box(x).round_ties_even());
                }
            })),
            None,
        ];
        row($name, &times);
    }};
}

/// Measures a `rustc_apfloat` type for comparison. `rustc_apfloat` has no
/// square root.
macro_rules! apfloat_row {
    ($name:literal, $apfloat:ty, $floaty:ty, $seed:literal) => {{
        let mut random = SplitMix64::new($seed);
        let values = |random: &mut SplitMix64| -> Vec<$apfloat> {
            let mut values = Vec::with_capacity(COUNT);
            for _ in 0..COUNT {
                let exact = Exact {
                    negative: random.next_u64() % 2 == 0,
                    exponent: -63
                        + i32::try_from(random.next_u64() % 8)
                            .expect("a shift below 8 fits an i32"),
                    significand: [random.next_u64() | 1 << 63],
                    sticky: false,
                };
                let (value, _): ($floaty, _) = <$floaty>::round(exact, Env::IEEE);
                values.push(<$apfloat>::from_bits(u128::from(value.to_bits())));
            }
            values
        };
        let (a, b, c) = (
            values(&mut random),
            values(&mut random),
            values(&mut random),
        );
        let times = [
            Some(measure(|| {
                for (&x, &y) in a.iter().zip(&b) {
                    black_box((black_box(x) + black_box(y)).value.to_bits());
                }
            })),
            None,
            Some(measure(|| {
                for (&x, &y) in a.iter().zip(&b) {
                    black_box((black_box(x) * black_box(y)).value.to_bits());
                }
            })),
            Some(measure(|| {
                for (&x, &y) in a.iter().zip(&b) {
                    black_box((black_box(x) / black_box(y)).value.to_bits());
                }
            })),
            None,
            None,
            Some(measure(|| {
                for ((&x, &y), &z) in a.iter().zip(&b).zip(&c) {
                    black_box(
                        black_box(x)
                            .mul_add(black_box(y), black_box(z))
                            .value
                            .to_bits(),
                    );
                }
            })),
            Some(measure(|| {
                for &x in &a {
                    let mut lost = false;
                    let converted: Double = black_box(x).convert(&mut lost).value;
                    black_box(converted.to_bits());
                }
            })),
            Some(measure(|| {
                for &x in &a {
                    let mut exact = false;
                    black_box(
                        black_box(x)
                            .to_i128_r(64, rustc_apfloat::Round::NearestTiesToEven, &mut exact)
                            .value,
                    );
                }
            })),
            Some(measure(|| {
                for &x in &a {
                    black_box(
                        black_box(x)
                            .round_to_integral(rustc_apfloat::Round::NearestTiesToEven)
                            .value
                            .to_bits(),
                    );
                }
            })),
            None,
        ];
        row($name, &times);
    }};
}

fn main() {
    println!("Nanoseconds per operation, median of {SAMPLES} samples.");
    println!();
    println!("| Format | {} |", COLUMNS.join(" | "));
    println!("|---{}|", "|---".repeat(COLUMNS.len()));
    floaty_row::<Binary<4, NoInf>, 8>("floaty F8E4M3", 1);
    floaty_row::<Binary<5, Fnuz>, 8>("floaty F8E5M2Fnuz", 10);
    floaty_row::<Binary<5>, 16>("floaty F16", 2);
    floaty_row::<Binary<8>, 16>("floaty BF16", 3);
    floaty_row::<Binary<8>, 32>("floaty F32", 4);
    floaty_row::<Binary<11>, 64>("floaty F64", 5);
    floaty_row::<Binary<15, X87>, 80>("floaty F80", 6);
    floaty_row::<Binary<15>, 128>("floaty F128", 7);
    floaty_row::<Binary<19>, 256>("floaty F256", 8);
    floaty_row::<Binary<23>, 512>("floaty F512", 9);
    host_row!("host f32", f32, 8, 32, 4);
    host_row!("host f64", f64, 11, 64, 5);
    apfloat_row!("apfloat Half", Half, F16, 2);
    apfloat_row!("apfloat BFloat", BFloat, BF16, 3);
    apfloat_row!("apfloat Single", Single, F32, 4);
    apfloat_row!("apfloat Double", Double, F64, 5);
    apfloat_row!("apfloat X87", X87DoubleExtended, F80, 6);
    apfloat_row!("apfloat Quad", Quad, F128, 7);
}
