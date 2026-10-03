//! Compares `DoubleDouble<Qd>::sqr`, `inv`, and `npwr` with QD 2.3.24.
//!
//! Each case checks both halves and the five IEEE flags in every rounding
//! direction. QD also runs with FTZ, DAZ, and both MXCSR bits set.

// The C references build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{DoubleDouble, Env, F64, Flags, Qd};
use floaty_verify::double_double::{operand, pair};
use floaty_verify::qd::{self, Flags as ReferenceFlags, Outcome, Pair, Rounding};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::sse_env;

/// A QD unary operation.
#[derive(Clone, Copy, Debug)]
enum Operation {
    Square,
    Inverse,
}

impl Operation {
    const ALL: [Self; 2] = [Self::Square, Self::Inverse];

    fn floaty(self, value: DoubleDouble<Qd>, env: Env) -> (DoubleDouble<Qd>, Flags) {
        match self {
            Self::Square => value.sqr_with(env),
            Self::Inverse => value.inv_with(env),
        }
    }

    fn default_mode(self, value: DoubleDouble<Qd>) -> DoubleDouble<Qd> {
        match self {
            Self::Square => value.sqr(),
            Self::Inverse => value.inv(),
        }
    }

    fn reference(self, value: Pair, rounding: Rounding) -> Outcome {
        match self {
            Self::Square => qd::sqr(value, rounding),
            Self::Inverse => qd::inv(value, rounding),
        }
    }
}

fn value(pair: Pair) -> DoubleDouble<Qd> {
    DoubleDouble::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
}

fn pair_of(value: DoubleDouble<Qd>) -> Pair {
    Pair::new(value.hi().to_bits(), value.lo().to_bits())
}

fn outcome((result, flags): (DoubleDouble<Qd>, Flags)) -> Outcome {
    Outcome {
        result: pair_of(result),
        flags: ReferenceFlags::from_floaty(flags),
    }
}

fn operands() -> Vec<Pair> {
    let edges = [
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x000F_FFFF_FFFF_FFFF,
        0x0010_0000_0000_0000,
        0x1FF0_0000_0000_0000,
        0x3FE0_0000_0000_0000,
        0x3FF0_0000_0000_0000,
        0xBFF0_0000_0000_0000,
        0x5FE0_0000_0000_0000,
        0x7FEF_FFFF_FFFF_FFFF,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,
        0xFFF4_0000_0000_0002,
    ];
    let mut random = SplitMix64::new(0x0DD5_0123);
    let mut pairs: Vec<Pair> = edges
        .into_iter()
        .flat_map(|hi| edges.map(|lo| Pair::new(hi, lo)))
        .collect();
    pairs.extend((0..4_000).map(|_| operand(&mut random)));
    pairs.extend((0..4_000).map(|_| pair(&mut random)));
    pairs
}

/// Exponents at the edges of QD's binary exponentiation: 0, 1, small
/// powers, powers of two and their neighbors, and the largest magnitudes
/// that QD's `npwr` returns for.
const EXPONENTS: [i32; 27] = [
    0,
    1,
    -1,
    2,
    -2,
    3,
    -3,
    4,
    5,
    -5,
    7,
    8,
    -8,
    15,
    16,
    31,
    -31,
    64,
    100,
    -100,
    1023,
    1024,
    -1075,
    1 << 30,
    -(1 << 30),
    i32::MAX,
    -i32::MAX,
];

/// Returns a pair near 1 in magnitude, whose powers stay finite for longer.
fn near_one(random: &mut SplitMix64) -> Pair {
    let field = 1022 + random.below(2);
    let sign = random.next_u64() & (1 << 63);
    let hi = sign | field << 52 | (random.next_u64() & ((1 << 52) - 1));
    let low_field = field - 54 - random.below(4);
    let lo = (random.next_u64() & 0x800F_FFFF_FFFF_FFFF) | low_field << 52;
    Pair::new(hi, lo)
}

/// Returns the operands of the integer power: the edge pairs, pairs near 1,
/// and random pairs.
fn power_operands() -> Vec<Pair> {
    let mut random = SplitMix64::new(0x0DD5_0456);
    let mut pairs: Vec<Pair> = operands().into_iter().take(256).collect();
    pairs.extend((0..400).map(|_| near_one(&mut random)));
    pairs.extend((0..400).map(|_| operand(&mut random)));
    pairs
}

#[test]
fn qd_integer_power_matches_qd() {
    qd::check_host_fma();
    let mut failures = Vec::new();
    let mut count = 0;
    for pair in power_operands() {
        let x = value(pair);
        for n in EXPONENTS {
            for rounding in Rounding::ALL {
                let flush = [(false, false)].into_iter().chain(qd::FLUSH_SETTINGS);
                for (ftz, daz) in flush {
                    let ours = outcome(x.npwr_with(n, sse_env(rounding.into(), ftz, daz)));
                    let theirs = qd::with_flush(ftz, daz, || qd::npwr(pair, n, rounding));
                    count += 1;
                    if ours != theirs && failures.len() < 40 {
                        failures.push(format!(
                            "npwr {pair:?} {n} {rounding:?} FTZ {ftz} DAZ {daz}: floaty \
                             {ours:?}, QD {theirs:?}"
                        ));
                    }
                }
            }
            let theirs = qd::npwr(pair, n, Rounding::TiesToEven).result;
            if pair_of(x.npwr(n)) != theirs && failures.len() < 40 {
                failures.push(format!("npwr {pair:?} {n} default mode"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(count, 1_056 * 27 * 4 * 4);
}

// Conflict: QD 2.3.24 `src/dd_real.cpp`, line 138, computes
// `int N = std::abs(n)`, which is undefined for INT_MIN. g++ 15.2.0 compiles
// the halving as `sar` (`npwr` at 0x934), so the count stays at -1 and the
// call never returns. Resolution: floaty takes the magnitude 2^31, as the
// source intends. The oracle is the source loop with an unsigned magnitude,
// compiled in the shim with the inline operators of QD. The test first
// checks that loop against the library for every other nonzero exponent.
#[test]
fn qd_integer_power_of_int_min_follows_the_qd_source_loop() {
    qd::check_host_fma();
    let pairs = power_operands();
    let mut failures = Vec::new();
    for &pair in &pairs {
        for n in EXPONENTS.into_iter().filter(|&n| n != 0) {
            for rounding in Rounding::ALL {
                let library = qd::npwr(pair, n, rounding);
                let source_loop = qd::npwr_magnitude(pair, n, rounding);
                if library != source_loop && failures.len() < 40 {
                    failures.push(format!(
                        "source loop {pair:?} {n} {rounding:?}: {source_loop:?}, library \
                         {library:?}"
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    for &pair in &pairs {
        let x = value(pair);
        for rounding in Rounding::ALL {
            let env = Env::X86_SSE.with_rounding(rounding.into());
            let ours = outcome(x.npwr_with(i32::MIN, env));
            let theirs = qd::npwr_magnitude(pair, i32::MIN, rounding);
            if ours != theirs && failures.len() < 40 {
                failures.push(format!(
                    "npwr {pair:?} i32::MIN {rounding:?}: floaty {ours:?}, QD source loop \
                     {theirs:?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn qd_square_and_inverse_match_qd() {
    qd::check_host_fma();
    let mut failures = Vec::new();
    let mut count = 0;
    for pair in operands() {
        let x = value(pair);
        for operation in Operation::ALL {
            for rounding in Rounding::ALL {
                let env = Env::X86_SSE.with_rounding(rounding.into());
                let ours = outcome(operation.floaty(x, env));
                let theirs = operation.reference(pair, rounding);
                count += 1;
                if ours != theirs && failures.len() < 40 {
                    failures.push(format!(
                        "{operation:?} {pair:?} {rounding:?}: floaty {ours:?}, QD {theirs:?}"
                    ));
                }
            }
            if pair_of(operation.default_mode(x))
                != operation.reference(pair, Rounding::TiesToEven).result
                && failures.len() < 40
            {
                failures.push(format!("{operation:?} {pair:?} default mode"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(count, 2 * 4 * 8_256);
}

#[test]
fn qd_square_and_inverse_match_qd_with_flush_to_zero_and_denormals_are_zero() {
    qd::check_host_fma();
    let mut failures = Vec::new();
    let mut changed = 0;
    let pairs = operands();
    for &pair in &pairs {
        let x = value(pair);
        for operation in Operation::ALL {
            for rounding in Rounding::ALL {
                let default = operation.reference(pair, rounding);
                for (ftz, daz) in qd::FLUSH_SETTINGS {
                    let ours = outcome(operation.floaty(x, sse_env(rounding.into(), ftz, daz)));
                    let theirs = qd::with_flush(ftz, daz, || operation.reference(pair, rounding));
                    changed += usize::from(theirs != default);
                    if ours != theirs && failures.len() < 40 {
                        failures.push(format!(
                            "{operation:?} {pair:?} {rounding:?} FTZ {ftz} DAZ {daz}: floaty \
                             {ours:?}, QD {theirs:?}"
                        ));
                    }
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert_ne!(changed, 0, "the flush settings must change some QD results");
}
