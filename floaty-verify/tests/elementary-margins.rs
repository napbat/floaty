//! Checks that the balls of the elementary functions decide the rounding,
//! through the margins that the test feature `record-margins` of floaty
//! records.
//!
//! An elementary function evaluates a ball at the first working precision,
//! and at the second when the first does not decide the truncation. When
//! the second does not decide either, the result rounds from the center of
//! the ball. The margin of a ball is the greatest `j` such that the ball with
//! its radius times `2^j` still decides. So an evaluation is decided when
//! one of its balls has a margin of at least 0.
//!
//! The normal run checks the hard cases: the published worst cases, the
//! arguments with a fraction nearest a multiple of `pi / 2`, and powers of
//! the bases next to 1. Two ignored sweeps print the least margins of
//! every function: every binary32 input at the first ball, and random
//! samples of binary16 to binary512 and decimal32 to decimal128 at both.

// The multiples of `pi / 2` come from MPFR, which builds only for x86-64.
#![cfg(target_arch = "x86_64")]

use std::fmt::Debug;
use std::hint::black_box;
use std::sync::atomic::{AtomicU64, Ordering};

use floaty::format::Standard;
use floaty::margins::{self, Record, Recording};
use floaty::{BF16, D32Bid, D64Bid, D128Bid, Env, F16, F32, F64, F80, F128, F256, F512, Float};
use floaty_verify::encodings::{IntegerBit, Layout, to_limbs};
use floaty_verify::mpfr::decimal::DecimalFormat;
use floaty_verify::operations::elementary::{Bivariate, Function, half_pi_multiples, worst_cases};
use floaty_verify::random::SplitMix64;
use rug::float::Round;
use rug::ops::Pow;
use rug::{Float as BigFloat, Integer};

/// Records the exact margin of both balls.
const BOTH: Recording = Recording {
    both: true,
    cap: i64::MAX,
};

/// The least margins of a set of evaluations, with the input of each, and
/// the inputs that no ball decided.
#[derive(Default)]
struct Least {
    /// The evaluations that took a ball.
    balls: u64,
    /// The least margin of a first ball.
    first: Option<(i64, String)>,
    /// The least margin of a second ball.
    second: Option<(i64, String)>,
    /// The inputs whose first ball did not decide.
    first_undecided: Vec<String>,
    /// The inputs that no ball decided.
    undecided: Vec<String>,
}

impl Least {
    /// Adds the record of one evaluation of `input`.
    fn observe(&mut self, record: Option<Record>, input: &dyn Fn() -> String) {
        let Some(record) = record else { return };
        self.balls += 1;
        let keep = |slot: &mut Option<(i64, String)>, margin: i64| {
            if slot.as_ref().is_none_or(|(least, _)| margin < *least) {
                *slot = Some((margin, input()));
            }
        };
        if let Some(margin) = record.first {
            keep(&mut self.first, margin);
        }
        if let Some(margin) = record.second {
            keep(&mut self.second, margin);
        }
        let decides = |margin: Option<i64>| margin.is_some_and(|margin| margin >= 0);
        if !decides(record.first) && self.first_undecided.len() < 8 {
            self.first_undecided.push(input());
        }
        if !decides(record.first) && !decides(record.second) && self.undecided.len() < 8 {
            self.undecided.push(input());
        }
    }

    /// Adds the evaluations of `other`.
    fn merge(&mut self, other: Self) {
        self.balls += other.balls;
        for (slot, theirs) in [
            (&mut self.first, other.first),
            (&mut self.second, other.second),
        ] {
            if let Some((margin, input)) = theirs
                && slot.as_ref().is_none_or(|(least, _)| margin < *least)
            {
                *slot = Some((margin, input));
            }
        }
        self.first_undecided.extend(other.first_undecided);
        self.undecided.extend(other.undecided);
    }

    /// Prints the least margins under `label`.
    fn print(&self, label: &str) {
        let text = |slot: &Option<(i64, String)>| {
            slot.as_ref().map_or_else(
                || "-".to_owned(),
                |(margin, input)| format!("{margin} at {input}"),
            )
        };
        println!(
            "{label}: {} balls, least first margin {}, least second margin {}",
            self.balls,
            text(&self.first),
            text(&self.second)
        );
    }
}

/// Evaluates `evaluate` and returns the margins that it recorded.
fn margins_of(evaluate: impl FnOnce()) -> Option<Record> {
    let _ = margins::take();
    evaluate();
    margins::take()
}

/// Evaluates `evaluate` on `count` inputs from `input` on every thread, with
/// `recording`, and returns the least margins.
fn sweep<T: Copy + Debug>(
    count: u64,
    recording: Recording,
    seed: u64,
    input: &(dyn Fn(u64, &mut SplitMix64) -> Option<T> + Sync),
    evaluate: &(dyn Fn(T) + Sync),
) -> Least {
    const CHUNK: u64 = 4096;
    let threads = std::thread::available_parallelism().map_or(1, usize::from);
    let next = AtomicU64::new(0);
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|thread| {
                let next = &next;
                scope.spawn(move || {
                    margins::record(Some(recording));
                    let thread = u64::try_from(thread).expect("a thread index fits a u64");
                    let mut random = SplitMix64::new(seed ^ (thread << 32));
                    let mut least = Least::default();
                    loop {
                        let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if start >= count {
                            break;
                        }
                        for index in start..(start + CHUNK).min(count) {
                            if let Some(x) = input(index, &mut random) {
                                let record = margins_of(|| evaluate(black_box(x)));
                                least.observe(record, &|| format!("{x:?}"));
                            }
                        }
                    }
                    margins::record(None);
                    least
                })
            })
            .collect();
        workers
            .into_iter()
            .fold(Least::default(), |mut least, worker| {
                least.merge(worker.join().expect("a sweep thread finished"));
                least
            })
    })
}

/// Returns the binary64 value nearest `value`, with `steps` neighbors on
/// each side.
fn binary64_around(value: &BigFloat, steps: usize) -> Vec<F64> {
    let center = F64::from_bits(value.to_f64().to_bits());
    let mut values = vec![center];
    let (mut up, mut down) = (center, center);
    for _ in 0..steps {
        (up, down) = (up.next_up(), down.next_down());
        values.extend([up, down]);
    }
    values
}

/// Returns the decimal64 value nearest `value`, with `steps` coefficients
/// on each side.
fn decimal64_around(value: &BigFloat, steps: u64) -> Vec<D64Bid> {
    let (negative, digits, exponent) = value.to_sign_string_exp_round(10, Some(16), Round::Nearest);
    let coefficient: u64 = digits.parse().expect("MPFR writes 16 digits");
    let exponent = exponent.expect("the value is finite") - 16;
    (coefficient - steps..=coefficient + steps)
        .map(|coefficient| {
            let value = D64Bid::from_int(coefficient).scale_b(exponent);
            if negative { -value } else { value }
        })
        .collect()
}

/// Returns the multiples of `pi / 2` nearest an argument with a fraction in
/// the binades or decades `RADIX^d` of `exponents` of a format of
/// `precision` digits in `radix`.
fn near_multiples(radix: u32, precision: u32, exponents: &[i32]) -> Vec<BigFloat> {
    let working = 16 * precision + 128;
    let power = |exponent: i32| BigFloat::with_val(working, radix).pow(exponent);
    let digits = i32::try_from(precision).expect("a precision fits an i32");
    exponents
        .iter()
        .flat_map(|&exponent| {
            let unit = power(exponent - digits + 1);
            half_pi_multiples(&unit, &power(exponent), &power(exponent + 1), working)
        })
        .collect()
}

/// The published worst cases, the binary64 and decimal64 arguments with a
/// fraction nearest a multiple of `pi / 2`, and binary64 powers of the bases
/// one to 64 steps from 1 decide at the second precision, or earlier.
#[test]
fn hard_cases_decide() {
    margins::record(Some(BOTH));
    let mut least = Least::default();
    let mut binary64 = |label: &str, x: F64, evaluate: &dyn Fn(F64) -> F64| {
        let record = margins_of(|| {
            black_box(evaluate(x));
        });
        least.observe(record, &|| format!("{label} {:#018x}", x.to_bits()));
    };
    for bits in worst_cases::EXP {
        binary64("exp", F64::from_bits(bits), &|x| x.exp_with(Env::IEEE).0);
    }
    for bits in worst_cases::LOG {
        binary64("log", F64::from_bits(bits), &|x| x.log_with(Env::IEEE).0);
    }
    let mut arguments = vec![F64::from_bits(worst_cases::REDUCTION)];
    for multiple in near_multiples(2, 53, &[0, 1, 25, 50, 51]) {
        arguments.extend(binary64_around(&multiple, 1));
    }
    for &x in &arguments {
        binary64("sin", x, &|x| x.sin_with(Env::IEEE).0);
        binary64("cos", x, &|x| x.cos_with(Env::IEEE).0);
        binary64("tan", x, &|x| x.tan_with(Env::IEEE).0);
    }
    let exponents = [
        2.0,
        3.0,
        5.0,
        -1.0,
        -3.0,
        0.5,
        -0.5,
        1.5,
        2.0 / 3.0,
        1.0 / 3.0,
    ];
    for y in exponents.map(|y: f64| F64::from_bits(y.to_bits())) {
        for step in 1..=64 {
            for bits in [0x3FF0_0000_0000_0000 + step, 0x3FF0_0000_0000_0000 - step] {
                binary64("pow", F64::from_bits(bits), &|x| x.pow_with(y, Env::IEEE).0);
            }
        }
    }
    let mut decimal64 = |label: &str, x: D64Bid, evaluate: &dyn Fn(D64Bid) -> D64Bid| {
        let record = margins_of(|| {
            black_box(evaluate(x));
        });
        least.observe(record, &|| format!("{label} {x:?}"));
    };
    for (coefficient, exponent) in worst_cases::DECIMAL64_EXP {
        let exponent = i32::try_from(exponent).expect("an exponent fits an i32");
        let x = D64Bid::from_int(coefficient).scale_b(exponent);
        decimal64("exp", x, &|x| x.exp_with(Env::IEEE).0);
    }
    for multiple in near_multiples(10, 16, &[0, 1, 7, 13, 14]) {
        for x in decimal64_around(&multiple, 1) {
            decimal64("sin", x, &|x| x.sin_with(Env::IEEE).0);
            decimal64("cos", x, &|x| x.cos_with(Env::IEEE).0);
            decimal64("tan", x, &|x| x.tan_with(Env::IEEE).0);
        }
    }
    margins::record(None);
    least.print("hard cases");
    assert!(least.balls > 1000, "the hard cases take balls");
    assert!(
        least.undecided.is_empty(),
        "no ball decides {:?}",
        least.undecided
    );
}

/// Every binary32 input of each function of one argument decides at the
/// first precision.
// Takes about 65 minutes on 32 threads.
#[test]
#[ignore = "takes about 65 minutes on 32 threads"]
fn every_binary32_input_decides_at_the_first_precision() {
    // The search for the exact margin stops at 48 bits.
    let recording = Recording {
        both: false,
        cap: 48,
    };
    let input = |index: u64, _: &mut SplitMix64| {
        let x = F32::from_bits(u32::try_from(index).expect("the index is a binary32 encoding"));
        (!x.is_nan()).then_some(x)
    };
    let mut undecided = Vec::new();
    for function in Function::ALL {
        let least = sweep(1 << 32, recording, 0, &input, &|x| {
            black_box(function.floaty(x, Env::IEEE));
        });
        least.print(&format!("binary32 {function:?}"));
        undecided.extend(least.first_undecided);
    }
    assert!(
        undecided.is_empty(),
        "the first ball does not decide {undecided:?}"
    );
}

/// Returns a random encoding of `layout`: any encoding, or with `in_range`
/// one whose exponent lies from `2^-(p + 8)` to `2^(E + 2)`, where the
/// functions take a ball.
fn binary_encoding(layout: Layout, in_range: bool, random: &mut SplitMix64) -> Integer {
    let fraction_bits = layout.fraction_bits();
    let largest = u64::from(layout.largest_field());
    let field = if in_range {
        let bias = u64::from(layout.ieee_bias().unsigned_abs());
        let precision = u64::from(layout.precision());
        let low = bias.saturating_sub(precision + 8).max(1);
        let high = (bias + u64::from(layout.exponent_bits) + 2).min(largest - 1);
        low + random.below(high - low + 1)
    } else {
        random.below(largest + 1)
    };
    let mut fraction = Integer::from(random.next_u64());
    while fraction.significant_bits() < fraction_bits + 64 {
        fraction = (fraction << 64) | random.next_u64();
    }
    fraction.keep_bits_mut(fraction_bits);
    if layout.integer_bit == IntegerBit::Explicit {
        fraction.set_bit(fraction_bits - 1, field != 0);
    }
    let sign = Integer::from(random.coin_flip()) << (layout.width - 1);
    sign | (Integer::from(field) << fraction_bits) | fraction
}

/// Returns a random value of a decimal format: a coefficient of up to `p`
/// digits with any exponent, or with `in_range` an exponent that puts the
/// value from `10^-(p + 8)` to `10^(d + 2)`, with `d` the digits of the
/// largest exponent.
fn decimal_value<S: Standard<W>, const W: usize>(
    format: DecimalFormat,
    in_range: bool,
    random: &mut SplitMix64,
) -> Float<S, W> {
    let precision = i64::from(format.precision);
    let coefficient = random.below_u128(10_u128.pow(format.precision));
    let (low, high) = if in_range {
        let digits = i64::try_from(format.emax.to_string().len()).expect("a digit count fits");
        (-(2 * precision + 8), digits + 2 - precision)
    } else {
        format.exponents()
    };
    let span = u64::try_from(high - low + 1).expect("the span is positive");
    let exponent = low + i64::try_from(random.below(span)).expect("an exponent fits");
    let value = Float::<S, W>::from_int(coefficient)
        .scale_b(i32::try_from(exponent).expect("an exponent fits an i32"));
    if random.coin_flip() { -value } else { value }
}

/// Checks every function of one and of two arguments on `count` random
/// inputs from `input`, any encodings and in-range ones, and prints the
/// least margins. Returns the inputs that no ball decided.
fn check_samples<S: Standard<W>, const W: usize>(
    label: &str,
    count: u64,
    input: &(dyn Fn(bool, &mut SplitMix64) -> Float<S, W> + Sync),
) -> Vec<String> {
    let mut undecided = Vec::new();
    for in_range in [false, true] {
        let sampler = if in_range { "in range" } else { "any" };
        let one = |_: u64, random: &mut SplitMix64| {
            let x = input(in_range, random);
            (!x.is_nan()).then_some(x)
        };
        for function in Function::ALL {
            let least = sweep(count, BOTH, 1, &one, &|x| {
                black_box(function.floaty(x, Env::IEEE));
            });
            least.print(&format!("{label} {sampler} {function:?}"));
            undecided.extend(least.undecided);
        }
        let two = |_: u64, random: &mut SplitMix64| {
            let (x, y) = (input(in_range, random), input(in_range, random));
            (!x.is_nan() && !y.is_nan()).then_some((x, y))
        };
        for function in Bivariate::ATAN2.into_iter().chain(Bivariate::POWER) {
            let least = sweep(count, BOTH, 2, &two, &|(x, y)| {
                black_box(function.floaty(x, y, Env::IEEE));
            });
            least.print(&format!("{label} {sampler} {function:?}"));
            undecided.extend(least.undecided);
        }
    }
    undecided
}

/// Random samples of every function decide at the second precision, or
/// earlier, on binary16 to binary512 and decimal32 to decimal128.
// Takes about 7 minutes on 32 threads.
#[test]
#[ignore = "takes about 7 minutes on 32 threads"]
fn samples_decide_at_the_second_precision() {
    let mut undecided = Vec::new();
    macro_rules! binary {
        ($label:literal, $layout:expr, $count:expr, $make:expr) => {{
            let make = $make;
            undecided.extend(check_samples($label, $count, &|in_range, random| {
                make(&binary_encoding($layout, in_range, random))
            }));
        }};
    }
    let narrow = |bits: &Integer| bits.to_u128().expect("the encoding fits a u128");
    binary!("binary16", Layout::BINARY16, 1 << 20, |bits: &Integer| {
        F16::from_bits(u16::try_from(narrow(bits)).expect("16 bits"))
    });
    binary!("bfloat16", Layout::BFLOAT16, 1 << 20, |bits: &Integer| {
        BF16::from_bits(u16::try_from(narrow(bits)).expect("16 bits"))
    });
    binary!("binary32", Layout::BINARY32, 1 << 22, |bits: &Integer| {
        F32::from_bits(u32::try_from(narrow(bits)).expect("32 bits"))
    });
    binary!("binary64", Layout::BINARY64, 1 << 22, |bits: &Integer| {
        F64::from_bits(u64::try_from(narrow(bits)).expect("64 bits"))
    });
    binary!("x87", Layout::X87_EXTENDED, 1 << 20, |bits: &Integer| {
        F80::from_bits(narrow(bits))
    });
    binary!("binary128", Layout::BINARY128, 1 << 20, |bits: &Integer| {
        F128::from_bits(narrow(bits))
    });
    binary!("binary256", Layout::BINARY256, 1 << 16, |bits: &Integer| {
        F256::from_bits(to_limbs::<4>(bits))
    });
    binary!("binary512", Layout::BINARY512, 1 << 12, |bits: &Integer| {
        F512::from_bits(to_limbs::<8>(bits))
    });
    macro_rules! decimal {
        ($label:literal, $type:ty, $count:expr) => {{
            let format = DecimalFormat::of::<$type>();
            undecided.extend(check_samples($label, $count, &|in_range,
                                                             random|
             -> $type {
                decimal_value(format, in_range, random)
            }));
        }};
    }
    decimal!("decimal32", D32Bid, 1 << 22);
    decimal!("decimal64", D64Bid, 1 << 22);
    decimal!("decimal128", D128Bid, 1 << 20);
    assert!(undecided.is_empty(), "no ball decides {undecided:?}");
}
