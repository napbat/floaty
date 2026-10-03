//! Measures the time of floaty operations for each binary and decimal format,
//! and of the host `f32` and `f64` types and `rustc_apfloat` as baselines.
//!
//! Run it with `cargo bench -p floaty-verify --bench operations`. Each case
//! runs a batch of 1,024 operations on seeded random normal operands near
//! 1.0, many times, and reports the median of 11 samples in nanoseconds per
//! operation. The operands avoid special values, so the numbers show the
//! common path of each operation. The operator columns of binary32 and
//! binary64 take the host fast path on x86-64.
//!
//! The `_with` columns pass an `Env`, which the engine reads at run time. The
//! `_mode` columns pass the mode of the type, `mode::Ieee`, a behavior fixed
//! at compile time. The other floaty columns use the mode of the type too.
//!
//! A second table measures other operations and operands with the mode of
//! the type: the remainder of close and of distant operands, conversions to
//! binary32, binary16, x87 extended, and decimal64 and from `i64`, additions
//! of subnormal operands and of a zero, the comparison and the minimum of
//! two operands, and `exp` and `log`. A third table measures the
//! double-double types, and a fourth the operations of `Lanes` in
//! nanoseconds per lane. A fifth table, in [`kernels`], measures the slice
//! kernels of `Lanes` in nanoseconds per vector, and a sixth, in
//! [`elementwise`], its elementwise slice operations. On x86-64, a seventh
//! table compares the decimal formats with the Intel decimal library for BID
//! and with decNumber for DPD.

mod elementwise;
mod kernels;

use std::hint::black_box;
use std::time::{Duration, Instant};

use floaty::format::Standard;
use floaty::{
    BF16, Bid, Binary, D64Bid, Decimal, DoubleDouble, Dpd, Env, Exact, F16, F32, F64, F80, F128,
    Float, Fnuz, Gcc, Lanes, NoInf, Qd, X87, mode,
};
#[cfg(target_arch = "x86_64")]
use floaty_verify::decnumber::{self, Arithmetic as _};
#[cfg(target_arch = "x86_64")]
use floaty_verify::intel_decimal::{self, Bid64, Bid128, Format as _};
use floaty_verify::random::SplitMix64;
use rustc_apfloat::ieee::{BFloat, Double, Half, Quad, Single, X87DoubleExtended};
use rustc_apfloat::{Float as _, FloatConvert as _};

/// The operations of one batch.
const COUNT: usize = 1_024;

/// The samples of one case.
const SAMPLES: usize = 11;

/// The time that one sample runs at least.
const SAMPLE_TIME: Duration = Duration::from_millis(4);

/// A table of times: its columns, in order. A row holds one cell for each
/// column, so a row with another number of cells does not compile.
struct Table<const N: usize> {
    columns: [&'static str; N],
}

impl<const N: usize> Table<N> {
    /// Prints the header of the table, with `first` above the row names.
    fn header(&self, first: &str) {
        println!("| {first} | {} |", self.columns.join(" | "));
        println!("|---{}|", "|---".repeat(N));
    }

    /// Prints one row of the table: its name, and the time of each column
    /// in nanoseconds, or `-` for an operation that the row does not have.
    // The table is the receiver so that a row takes the column count of its
    // table. The row does not print the column names.
    #[allow(clippy::unused_self)]
    fn row(&self, name: &str, times: [Option<f64>; N]) {
        let cells: Vec<String> = times
            .iter()
            .map(|time| time.map_or_else(|| "-".to_owned(), |time| format!("{time:.1}")))
            .collect();
        println!("| {name} | {} |", cells.join(" | "));
    }
}

/// The operations that each row measures, in column order.
const COLUMNS: Table<16> = Table {
    columns: [
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
        "add_mode",
        "mul_with",
        "mul_mode",
        "div_mode",
        "mul_add_with",
    ],
};

/// The operations of the second table, in column order.
const OTHER_COLUMNS: Table<13> = Table {
    columns: [
        "rem", "rem_far", "to_f32", "to_f16", "to_f80", "to_d64", "from_i64", "add_tiny",
        "add_zero", "cmp", "min", "exp", "log",
    ],
};

/// The operations of the double-double table, in column order.
const DOUBLE_DOUBLE_COLUMNS: Table<5> = Table {
    columns: ["add", "mul", "div", "sqrt", "add_zero"],
};

/// The operations of the `Lanes` table, in column order. `cmp` is
/// `compare_quiet`, and `min` is `minimum`. `to_int` converts to `i32`.
/// `scalar_add` and `scalar_to_int` run the operation of the type on the
/// lanes one at a time, for comparison.
const LANES_COLUMNS: Table<13> = Table {
    columns: [
        "add",
        "mul",
        "div",
        "sqrt",
        "mul_add",
        "round_int",
        "convert",
        "cmp",
        "min",
        "add_with",
        "scalar_add",
        "to_int",
        "scalar_to_int",
    ],
};

/// The operations of the decimal reference table, in column order.
#[cfg(target_arch = "x86_64")]
const DECIMAL_COLUMNS: Table<5> = Table {
    columns: ["add", "mul", "div", "sqrt", "mul_add"],
};

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
    // A 64-bit significand is near 2^63, or near 10^19 in radix 10.
    let base = if S::RADIX == 10 { -19 } else { -63 };
    (0..COUNT)
        .map(|_| {
            let exact = Exact {
                negative: random.below(2) == 0,
                exponent: base
                    + i32::try_from(random.below(8)).expect("a shift below 8 fits an i32"),
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
        Some(binary(|x, y| x.add_with(y, mode::Ieee).0)),
        Some(binary(|x, y| x.mul_with(y, Env::IEEE).0)),
        Some(binary(|x, y| x.mul_with(y, mode::Ieee).0)),
        Some(binary(|x, y| x.div_with(y, mode::Ieee).0)),
        Some(measure(|| {
            for ((&x, &y), &z) in a.iter().zip(&b).zip(&c) {
                black_box(black_box(x).mul_add_with(black_box(y), black_box(z), Env::IEEE));
            }
        })),
    ];
    COLUMNS.row(name, times);
}

/// Returns the zero of a format.
fn zero<S: Standard<W>, const W: usize>() -> Float<S, W> {
    let exact = Exact {
        negative: false,
        exponent: 0,
        significand: [0_u64],
        sticky: false,
    };
    Float::round(exact, Env::IEEE).0
}

/// Returns `COUNT` seeded random integers of every magnitude.
fn integers(random: &mut SplitMix64) -> Vec<i64> {
    (0..COUNT)
        .map(|_| {
            let bits = random.below(64);
            let magnitude = i64::try_from((random.next_u64() >> 1) >> bits)
                .expect("a value below 2^63 fits an i64");
            if random.below(2) == 0 {
                magnitude
            } else {
                -magnitude
            }
        })
        .collect()
}

/// Measures the second table for one floaty format.
fn other_row<S: Standard<W>, const W: usize>(name: &str, seed: u64) {
    let mut random = SplitMix64::new(seed);
    let (a, b) = (operands::<S, W>(&mut random), operands::<S, W>(&mut random));
    let integers = integers(&mut random);
    // The distant dividends lie half the exponent range above the divisors.
    // The tiny operands lie half the precision below the smallest normal.
    let far: Vec<Float<S, W>> = a.iter().map(|x| x.scale_b(S::EMAX / 2)).collect();
    let half_precision = i32::try_from(S::PRECISION / 2).expect("a precision fits an i32");
    let tiny = |values: &[Float<S, W>]| -> Vec<Float<S, W>> {
        values
            .iter()
            .map(|x| x.scale_b(S::EMIN - half_precision))
            .collect()
    };
    let (tiny_a, tiny_b) = (tiny(&a), tiny(&b));
    let roots = positive(&a);
    let zero = zero::<S, W>();
    let pairs = |left: &[Float<S, W>],
                 right: &[Float<S, W>],
                 operation: fn(Float<S, W>, Float<S, W>) -> Float<S, W>| {
        measure(|| {
            for (&x, &y) in left.iter().zip(right) {
                black_box(operation(black_box(x), black_box(y)));
            }
        })
    };
    let times = [
        Some(pairs(&a, &b, Float::remainder)),
        Some(pairs(&far, &b, Float::remainder)),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).convert::<F32>());
            }
        })),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).convert::<F16>());
            }
        })),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).convert::<F80>());
            }
        })),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).convert::<D64Bid>());
            }
        })),
        Some(measure(|| {
            for &integer in &integers {
                black_box(Float::<S, W>::from_int(black_box(integer)));
            }
        })),
        Some(pairs(&tiny_a, &tiny_b, |x, y| x.add_with(y, mode::Ieee).0)),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).add_with(black_box(zero), mode::Ieee));
            }
        })),
        Some(measure(|| {
            for (&x, &y) in a.iter().zip(&b) {
                black_box(black_box(x).partial_cmp(&black_box(y)));
            }
        })),
        Some(pairs(&a, &b, Float::minimum)),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x).exp());
            }
        })),
        Some(measure(|| {
            for &x in &roots {
                black_box(black_box(x).log());
            }
        })),
    ];
    OTHER_COLUMNS.row(name, times);
}

/// Measures `operation` on each pair of `left` and `right`.
fn each_pair<T: Copy, R>(left: &[T], right: &[T], operation: impl Fn(T, T) -> R) -> f64 {
    measure(|| {
        for (&x, &y) in left.iter().zip(right) {
            black_box(operation(black_box(x), black_box(y)));
        }
    })
}

/// Measures the `Lanes` table for one format and lane count, in nanoseconds
/// per lane. `convert` converts the lanes to another format: binary32 and
/// binary64 to each other, and the other formats to binary32 or binary64.
/// The formats without a packed host path show the cost of the scalar
/// operation of each lane.
fn lanes_row<S: Standard<W>, const W: usize, const N: usize, T: Copy>(
    name: &str,
    seed: u64,
    convert: fn(Lanes<Float<S, W>, N>) -> T,
) {
    let mut random = SplitMix64::new(seed);
    let group = |values: &[Float<S, W>]| -> Vec<Lanes<Float<S, W>, N>> {
        values
            .as_chunks::<N>()
            .0
            .iter()
            .map(|&chunk| Lanes::new(chunk))
            .collect()
    };
    let first = operands::<S, W>(&mut random);
    let second = operands::<S, W>(&mut random);
    let third = operands::<S, W>(&mut random);
    let magnitudes: Vec<Float<S, W>> = first.iter().map(|value| value.abs()).collect();
    let (left, right) = (group(&first), group(&second));
    let (addends, roots) = (group(&third), group(&magnitudes));
    let times = [
        Some(each_pair(&left, &right, |x, y| x + y)),
        Some(each_pair(&left, &right, |x, y| x * y)),
        Some(each_pair(&left, &right, |x, y| x / y)),
        Some(each_pair(&roots, &roots, |x, _| x.sqrt())),
        Some(measure(|| {
            for ((&x, &y), &addend) in left.iter().zip(&right).zip(&addends) {
                black_box(black_box(x).mul_add(black_box(y), black_box(addend)));
            }
        })),
        Some(each_pair(&left, &left, |x, _| x.round_to_integral())),
        Some(each_pair(&left, &left, |x, _| convert(x))),
        Some(each_pair(&left, &right, Lanes::compare_quiet)),
        Some(each_pair(&left, &right, Lanes::minimum)),
        Some(each_pair(&left, &right, |x, y| x.add_with(y, Env::IEEE).0)),
        Some(each_pair(&first, &second, |x, y| x + y)),
        Some(each_pair(&left, &left, |x, _| x.to_int::<i32>())),
        Some(each_pair(&first, &first, |x, _| x.to_int::<i32>())),
    ];
    LANES_COLUMNS.row(name, times);
}

/// Measures the double-double table for one algorithm. Only `Qd` has a
/// square root.
fn double_double_row<Alg: floaty::Algorithm>(
    name: &str,
    seed: u64,
    sqrt: Option<fn(DoubleDouble<Alg>) -> DoubleDouble<Alg>>,
) {
    let mut random = SplitMix64::new(seed);
    let (high_a, high_b, low) = (
        operands::<Binary<11>, 64>(&mut random),
        operands::<Binary<11>, 64>(&mut random),
        operands::<Binary<11>, 64>(&mut random),
    );
    // A low part below half a unit in the last place of a high part of at
    // least 1.
    let pair = |high: &F64, low: &F64| DoubleDouble::<Alg>::from_parts(*high, low.scale_b(-62));
    let a: Vec<DoubleDouble<Alg>> = high_a.iter().zip(&low).map(|(h, l)| pair(h, l)).collect();
    let b: Vec<DoubleDouble<Alg>> = high_b.iter().zip(&low).map(|(h, l)| pair(h, l)).collect();
    let roots: Vec<DoubleDouble<Alg>> = high_a
        .iter()
        .zip(&low)
        .map(|(h, l)| pair(&h.abs(), l))
        .collect();
    let zero = DoubleDouble::<Alg>::from_f64(zero::<Binary<11>, 64>());
    let binary = |operation: fn(DoubleDouble<Alg>, DoubleDouble<Alg>) -> DoubleDouble<Alg>| {
        measure(|| {
            for (&x, &y) in a.iter().zip(&b) {
                black_box(operation(black_box(x), black_box(y)));
            }
        })
    };
    let times = [
        Some(binary(|x, y| x + y)),
        Some(binary(|x, y| x * y)),
        Some(binary(|x, y| x / y)),
        sqrt.map(|sqrt| {
            measure(|| {
                for &x in &roots {
                    black_box(sqrt(black_box(x)));
                }
            })
        }),
        Some(measure(|| {
            for &x in &a {
                black_box(black_box(x) + black_box(zero));
            }
        })),
    ];
    DOUBLE_DOUBLE_COLUMNS.row(name, times);
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
            None,
            None,
            None,
            None,
            None,
        ];
        COLUMNS.row($name, times);
    }};
}

/// Measures a `rustc_apfloat` type for comparison. `rustc_apfloat` has no
/// square root.
macro_rules! apfloat_row {
    ($name:literal, $apfloat:ty, $floaty:ty, $seed:literal) => {{
        let mut random = SplitMix64::new($seed);
        let values = |random: &mut SplitMix64| -> Vec<$apfloat> {
            let floats: Vec<$floaty> = operands(random);
            floats
                .into_iter()
                .map(|value| <$apfloat>::from_bits(u128::from(value.to_bits())))
                .collect()
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
            None,
            None,
            None,
            None,
            None,
        ];
        COLUMNS.row($name, times);
    }};
}

/// Measures one row of the decimal reference table. The operands are the
/// encodings of the random operands of `Float<$format, $width>`, the same
/// for floaty and for the reference. `$sqrt` is `None` for a reference
/// without a square root.
#[cfg(target_arch = "x86_64")]
macro_rules! decimal_row {
    (
        $name:literal, $format:ty, $width:literal, $seed:literal,
        $add:expr, $mul:expr, $div:expr, $sqrt:expr, $fma:expr
    ) => {{
        let mut random = SplitMix64::new($seed);
        let (x, y, z) = (
            operands::<$format, $width>(&mut random),
            operands::<$format, $width>(&mut random),
            operands::<$format, $width>(&mut random),
        );
        let bits = |values: &[Float<$format, $width>]| -> Vec<_> {
            values.iter().map(|value| value.to_bits()).collect()
        };
        let (a, b, c, roots) = (bits(&x), bits(&y), bits(&z), bits(&positive(&x)));
        let fma = $fma;
        let times = [
            Some(each_pair(&a, &b, $add)),
            Some(each_pair(&a, &b, $mul)),
            Some(each_pair(&a, &b, $div)),
            $sqrt.map(|sqrt| each_pair(&roots, &roots, |x, _| sqrt(x))),
            Some(measure(|| {
                for ((&x, &y), &z) in a.iter().zip(&b).zip(&c) {
                    black_box(fma(black_box(x), black_box(y), black_box(z)));
                }
            })),
        ];
        DECIMAL_COLUMNS.row($name, times);
    }};
}

/// Prints the decimal reference table: floaty against the Intel decimal
/// library for BID, and against the `decDouble` and `decQuad` functions of
/// decNumber for DPD. Each decNumber call also sets up a context. decNumber
/// has no square root of its fixed-size formats.
#[cfg(target_arch = "x86_64")]
fn decimal_table() {
    println!();
    println!("Decimal against the references, nanoseconds per operation.");
    println!();
    DECIMAL_COLUMNS.header("Format");
    bid_rows();
    dpd_rows();
}

/// Measures the BID rows of the decimal reference table.
#[cfg(target_arch = "x86_64")]
fn bid_rows() {
    use intel_decimal::Rounding::TiesToEven;
    type BidDecimal = Decimal<Bid>;
    decimal_row!(
        "floaty D64Bid",
        BidDecimal,
        64,
        12,
        |x, y| (Float::<BidDecimal, 64>::from_bits(x) + Float::from_bits(y)).to_bits(),
        |x, y| (Float::<BidDecimal, 64>::from_bits(x) * Float::from_bits(y)).to_bits(),
        |x, y| (Float::<BidDecimal, 64>::from_bits(x) / Float::from_bits(y)).to_bits(),
        Some(|x| Float::<BidDecimal, 64>::from_bits(x).sqrt().to_bits()),
        |x, y, z| Float::<BidDecimal, 64>::from_bits(x)
            .mul_add(Float::from_bits(y), Float::from_bits(z))
            .to_bits()
    );
    decimal_row!(
        "Intel bid64",
        BidDecimal,
        64,
        12,
        |x, y| Bid64::add(x, y, TiesToEven).value,
        |x, y| Bid64::mul(x, y, TiesToEven).value,
        |x, y| Bid64::div(x, y, TiesToEven).value,
        Some(|x| Bid64::sqrt(x, TiesToEven).value),
        |x, y, z| Bid64::fma(x, y, z, TiesToEven).value
    );
    decimal_row!(
        "floaty D128Bid",
        BidDecimal,
        128,
        13,
        |x, y| (Float::<BidDecimal, 128>::from_bits(x) + Float::from_bits(y)).to_bits(),
        |x, y| (Float::<BidDecimal, 128>::from_bits(x) * Float::from_bits(y)).to_bits(),
        |x, y| (Float::<BidDecimal, 128>::from_bits(x) / Float::from_bits(y)).to_bits(),
        Some(|x| Float::<BidDecimal, 128>::from_bits(x).sqrt().to_bits()),
        |x, y, z| Float::<BidDecimal, 128>::from_bits(x)
            .mul_add(Float::from_bits(y), Float::from_bits(z))
            .to_bits()
    );
    decimal_row!(
        "Intel bid128",
        BidDecimal,
        128,
        13,
        |x, y| Bid128::add(x, y, TiesToEven).value,
        |x, y| Bid128::mul(x, y, TiesToEven).value,
        |x, y| Bid128::div(x, y, TiesToEven).value,
        Some(|x| Bid128::sqrt(x, TiesToEven).value),
        |x, y, z| Bid128::fma(x, y, z, TiesToEven).value
    );
}

/// Measures the DPD rows of the decimal reference table.
#[cfg(target_arch = "x86_64")]
fn dpd_rows() {
    type DpdDecimal = Decimal<Dpd>;
    const HALF_EVEN: decnumber::Rounding = decnumber::Rounding::HalfEven;
    decimal_row!(
        "floaty D64Dpd",
        DpdDecimal,
        64,
        14,
        |x, y| (Float::<DpdDecimal, 64>::from_bits(x) + Float::from_bits(y)).to_bits(),
        |x, y| (Float::<DpdDecimal, 64>::from_bits(x) * Float::from_bits(y)).to_bits(),
        |x, y| (Float::<DpdDecimal, 64>::from_bits(x) / Float::from_bits(y)).to_bits(),
        Some(|x| Float::<DpdDecimal, 64>::from_bits(x).sqrt().to_bits()),
        |x, y, z| Float::<DpdDecimal, 64>::from_bits(x)
            .mul_add(Float::from_bits(y), Float::from_bits(z))
            .to_bits()
    );
    decimal_row!(
        "decNumber decDouble",
        DpdDecimal,
        64,
        14,
        |x, y| decnumber::Double::binary(decnumber::Binary::Add, x, y, HALF_EVEN).value,
        |x, y| decnumber::Double::binary(decnumber::Binary::Multiply, x, y, HALF_EVEN).value,
        |x, y| decnumber::Double::binary(decnumber::Binary::Divide, x, y, HALF_EVEN).value,
        None::<fn(u64) -> u64>,
        |x, y, z| decnumber::Double::fma(x, y, z, HALF_EVEN).value
    );
    decimal_row!(
        "floaty D128Dpd",
        DpdDecimal,
        128,
        15,
        |x, y| (Float::<DpdDecimal, 128>::from_bits(x) + Float::from_bits(y)).to_bits(),
        |x, y| (Float::<DpdDecimal, 128>::from_bits(x) * Float::from_bits(y)).to_bits(),
        |x, y| (Float::<DpdDecimal, 128>::from_bits(x) / Float::from_bits(y)).to_bits(),
        Some(|x| Float::<DpdDecimal, 128>::from_bits(x).sqrt().to_bits()),
        |x, y, z| Float::<DpdDecimal, 128>::from_bits(x)
            .mul_add(Float::from_bits(y), Float::from_bits(z))
            .to_bits()
    );
    decimal_row!(
        "decNumber decQuad",
        DpdDecimal,
        128,
        15,
        |x, y| decnumber::Quad::binary(decnumber::Binary::Add, x, y, HALF_EVEN).value,
        |x, y| decnumber::Quad::binary(decnumber::Binary::Multiply, x, y, HALF_EVEN).value,
        |x, y| decnumber::Quad::binary(decnumber::Binary::Divide, x, y, HALF_EVEN).value,
        None::<fn(u128) -> u128>,
        |x, y, z| decnumber::Quad::fma(x, y, z, HALF_EVEN).value
    );
}

fn main() {
    println!("Nanoseconds per operation, median of {SAMPLES} samples.");
    println!();
    COLUMNS.header("Format");
    floaty_row::<Binary<4, NoInf>, 8>("floaty F8E4M3Fn", 1);
    floaty_row::<Binary<5, Fnuz>, 8>("floaty F8E5M2Fnuz", 10);
    floaty_row::<Binary<5>, 16>("floaty F16", 2);
    floaty_row::<Binary<8>, 16>("floaty BF16", 3);
    floaty_row::<Binary<8>, 32>("floaty F32", 4);
    floaty_row::<Binary<11>, 64>("floaty F64", 5);
    floaty_row::<Binary<15, X87>, 80>("floaty F80", 6);
    floaty_row::<Binary<15>, 128>("floaty F128", 7);
    floaty_row::<Binary<19>, 256>("floaty F256", 8);
    floaty_row::<Binary<23>, 512>("floaty F512", 9);
    floaty_row::<Decimal<Bid>, 32>("floaty D32Bid", 11);
    floaty_row::<Decimal<Bid>, 64>("floaty D64Bid", 12);
    floaty_row::<Decimal<Bid>, 128>("floaty D128Bid", 13);
    floaty_row::<Decimal<Dpd>, 64>("floaty D64Dpd", 14);
    floaty_row::<Decimal<Dpd>, 128>("floaty D128Dpd", 15);
    host_row!("host f32", f32, 8, 32, 4);
    host_row!("host f64", f64, 11, 64, 5);
    apfloat_row!("apfloat Half", Half, F16, 2);
    apfloat_row!("apfloat BFloat", BFloat, BF16, 3);
    apfloat_row!("apfloat Single", Single, F32, 4);
    apfloat_row!("apfloat Double", Double, F64, 5);
    apfloat_row!("apfloat X87", X87DoubleExtended, F80, 6);
    apfloat_row!("apfloat Quad", Quad, F128, 7);
    println!();
    println!("Other operations in the engine, nanoseconds per operation.");
    println!();
    OTHER_COLUMNS.header("Format");
    other_row::<Binary<4, NoInf>, 8>("floaty F8E4M3Fn", 21);
    other_row::<Binary<5, Fnuz>, 8>("floaty F8E5M2Fnuz", 22);
    other_row::<Binary<5>, 16>("floaty F16", 23);
    other_row::<Binary<8>, 16>("floaty BF16", 24);
    other_row::<Binary<8>, 32>("floaty F32", 25);
    other_row::<Binary<11>, 64>("floaty F64", 26);
    other_row::<Binary<15, X87>, 80>("floaty F80", 27);
    other_row::<Binary<15>, 128>("floaty F128", 28);
    other_row::<Binary<19>, 256>("floaty F256", 29);
    other_row::<Binary<23>, 512>("floaty F512", 30);
    other_row::<Decimal<Bid>, 32>("floaty D32Bid", 31);
    other_row::<Decimal<Bid>, 64>("floaty D64Bid", 32);
    other_row::<Decimal<Bid>, 128>("floaty D128Bid", 33);
    other_row::<Decimal<Dpd>, 64>("floaty D64Dpd", 34);
    other_row::<Decimal<Dpd>, 128>("floaty D128Dpd", 35);
    println!();
    println!("Double-double, nanoseconds per operation.");
    println!();
    DOUBLE_DOUBLE_COLUMNS.header("Algorithm");
    double_double_row::<Qd>("floaty Qd", 41, Some(DoubleDouble::sqrt));
    double_double_row::<Gcc>("floaty Gcc", 42, None);
    println!();
    println!("Lanes, nanoseconds per lane.");
    println!();
    LANES_COLUMNS.header("Lanes");
    lanes_row::<Binary<8>, 32, 4, _>("F32 x 4", 51, Lanes::convert::<F64>);
    lanes_row::<Binary<8>, 32, 8, _>("F32 x 8", 52, Lanes::convert::<F64>);
    lanes_row::<Binary<11>, 64, 2, _>("F64 x 2", 53, Lanes::convert::<F32>);
    lanes_row::<Binary<11>, 64, 4, _>("F64 x 4", 54, Lanes::convert::<F32>);
    lanes_row::<Binary<5>, 16, 8, _>("F16 x 8", 55, Lanes::convert::<F32>);
    lanes_row::<Binary<8>, 16, 8, _>("BF16 x 8", 56, Lanes::convert::<F32>);
    lanes_row::<Binary<15, X87>, 80, 2, _>("F80 x 2", 57, Lanes::convert::<F64>);
    kernels::table();
    elementwise::table();
    #[cfg(target_arch = "x86_64")]
    decimal_table();
}
