//! Compares `DoubleDouble<Qd>::sqr`, `inv`, `npwr`, and `nroot` with QD
//! 2.3.24.
//!
//! Each case checks both halves and the five IEEE flags in every rounding
//! direction. QD also runs with FTZ, DAZ, and both MXCSR bits set.
//!
//! `nroot` runs against a copy of QD's source in the shim that takes its
//! seed as an operand. MPFR gives the seed, `exp(-log(|hi|) / n)` with each
//! step correctly rounded in the behavior, and its flags.

// The C references build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{DoubleDouble, Env, F64, Flags, Qd};
use floaty_verify::arithmetic::{self, Operation as Arithmetic};
use floaty_verify::double_double::{operand, pair};
use floaty_verify::mpfr::{Format, Operand, Specials, Value};
use floaty_verify::operations::Outcome as Expected;
use floaty_verify::operations::elementary::{self, Function};
use floaty_verify::qd::{self, Flags as ReferenceFlags, Outcome, Pair, Rounding};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::sse_env;
use rug::Integer;
use rug::integer::Order;

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

/// The exponents of the root: the invalid ones, 1, the square root, small
/// odd and even roots, and large ones.
const ROOTS: [i32; 17] = [
    i32::MIN,
    -3,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    9,
    16,
    33,
    100,
    1023,
    1 << 20,
    i32::MAX,
];

/// Returns the operands of the root: the edge pairs, pairs near 1, and
/// random pairs.
fn root_operands() -> Vec<Pair> {
    let mut random = SplitMix64::new(0x0DD5_0789);
    let mut pairs: Vec<Pair> = operands().into_iter().take(256).collect();
    pairs.extend((0..100).map(|_| near_one(&mut random)));
    pairs.extend((0..300).map(|_| operand(&mut random)));
    pairs
}

/// Returns `true` when QD's `nroot` computes its seed: for an exponent of
/// 3 or more, a high half that does not compare equal to zero, and no even
/// exponent with a negative high half. Under DAZ a subnormal compares equal
/// to zero.
fn takes_seed(hi: u64, n: i32, daz: bool) -> bool {
    let magnitude = hi & !(1 << 63);
    let zero = magnitude == 0 || (daz && magnitude < 1 << 52);
    let negative = hi >> 63 == 1 && magnitude <= 0x7FF0_0000_0000_0000;
    n >= 3 && !zero && !(n % 2 == 0 && negative)
}

/// Returns the binary64 encoding of an oracle value with a NaN payload.
fn encode(value: &Value, payload: &Integer) -> u64 {
    let sign = |negative: bool| u64::from(negative) << 63;
    match value {
        Value::Zero { negative } => sign(*negative),
        Value::Infinity { negative } => sign(*negative) | 0x7FF0_0000_0000_0000,
        Value::Nan { negative } => {
            let payload = payload.to_u64().expect("a binary64 payload fits a u64");
            sign(*negative) | 0x7FF8_0000_0000_0000 | payload
        }
        Value::Finite(value) => value.to_f64().to_bits(),
    }
}

/// Returns the binary64 encoding of an outcome of the elementary oracle.
fn encode_outcome(outcome: &Expected) -> u64 {
    match outcome {
        Expected::Zero { negative } => encode(
            &Value::Zero {
                negative: *negative,
            },
            &Integer::ZERO,
        ),
        Expected::Finite(value) => encode(&Value::Finite(value.clone()), &Integer::ZERO),
        Expected::Infinity { negative } => encode(
            &Value::Infinity {
                negative: *negative,
            },
            &Integer::ZERO,
        ),
        Expected::Nan {
            negative, payload, ..
        } => encode(
            &Value::Nan {
                negative: *negative,
            },
            payload,
        ),
    }
}

/// Returns the seed `exp(-log(r0) / n)` of QD's `nroot` with each step
/// correctly rounded in `env`, and the flags of the three steps. `r0` is the
/// high half, negated when it is below zero.
fn seed(hi: u64, n: i32, env: &Env) -> (u64, Flags) {
    let format = Format::of::<F64>(Specials::Ieee);
    let below_zero = hi >> 63 == 1 && hi & !(1 << 63) <= 0x7FF0_0000_0000_0000;
    let r0 = if below_zero { hi ^ 1 << 63 } else { hi };
    let operand = |bits: u64| Operand::<8>::of(F64::from_bits(bits));
    let (log, log_flags) = elementary::expected(Function::Log, &operand(r0), &format, env);
    let negated = encode_outcome(&log) ^ 1 << 63;
    let count = F64::from_int(n).to_bits();
    let quotient = arithmetic::compute(
        Arithmetic::Div,
        &[operand(negated), operand(count)],
        &format,
        env,
    );
    let payload = Integer::from_digits(&quotient.payload, Order::Lsf);
    let quotient_bits = encode(&quotient.value, &payload);
    let (exp, exp_flags) =
        elementary::expected(Function::Exp, &operand(quotient_bits), &format, env);
    (encode_outcome(&exp), log_flags | quotient.flags | exp_flags)
}

/// Returns QD's `nroot` with the seed of [`seed`], and the flags of the seed
/// when QD computes it.
fn reference_root(pair: Pair, n: i32, rounding: Rounding, ftz: bool, daz: bool) -> Outcome {
    let env = sse_env(rounding.into(), ftz, daz);
    let (start, flags) = seed(pair.hi, n, &env);
    let root = qd::with_flush(ftz, daz, || qd::nroot(pair, n, start, rounding));
    if !takes_seed(pair.hi, n, daz) {
        return root;
    }
    Outcome {
        result: root.result,
        flags: root.flags | ReferenceFlags::from_floaty(flags),
    }
}

#[test]
fn qd_root_matches_qd_with_a_correctly_rounded_seed() {
    qd::check_host_fma();
    let mut failures = Vec::new();
    let mut count = 0;
    for pair in root_operands() {
        let x = value(pair);
        for n in ROOTS {
            for rounding in Rounding::ALL {
                let flush = [(false, false)].into_iter().chain(qd::FLUSH_SETTINGS);
                for (ftz, daz) in flush {
                    let ours = outcome(x.nroot_with(n, sse_env(rounding.into(), ftz, daz)));
                    let theirs = reference_root(pair, n, rounding, ftz, daz);
                    count += 1;
                    if ours != theirs && failures.len() < 40 {
                        failures.push(format!(
                            "nroot {pair:?} {n} {rounding:?} FTZ {ftz} DAZ {daz}: floaty \
                             {ours:?}, QD {theirs:?}"
                        ));
                    }
                }
            }
            let theirs = reference_root(pair, n, Rounding::TiesToEven, false, false).result;
            if pair_of(x.nroot(n)) != theirs && failures.len() < 40 {
                failures.push(format!("nroot {pair:?} {n} default mode"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
    assert_eq!(count, 656 * 17 * 4 * 4);
}

/// Returns the seed of QD's `nroot` as the library computes it, with the
/// C library `log` and `exp` of this host, in the default environment.
fn library_seed(hi: u64, n: i32) -> u64 {
    let below_zero = hi >> 63 == 1 && hi & !(1 << 63) <= 0x7FF0_0000_0000_0000;
    let r0 = f64::from_bits(if below_zero { hi ^ 1 << 63 } else { hi });
    (-r0.ln() / f64::from(n)).exp().to_bits()
}

/// The seeded copy in the shim must give the result of the library's
/// `nroot` when it takes the seed of the C library. The flags of the copy
/// leave out the seed, so the library raises each of them too.
#[test]
fn the_seeded_copy_matches_the_qd_library() {
    qd::check_host_fma();
    let mut failures = Vec::new();
    for pair in root_operands() {
        for n in ROOTS {
            let library = qd::nroot_library(pair, n, Rounding::TiesToEven);
            let copy = qd::nroot(pair, n, library_seed(pair.hi, n), Rounding::TiesToEven);
            let same = copy.result == library.result && library.flags.contains(copy.flags);
            if !same && failures.len() < 40 {
                failures.push(format!("{pair:?} {n}: copy {copy:?}, library {library:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

// Conflict: QD 2.3.24 `src/dd_real.cpp`, line 115, seeds `nroot` with
// `std::exp(-std::log(r.x[0]) / n)` of the C library. glibc 2.43 `exp`
// does not always round correctly: `sysdeps/ieee754/dbl-64/e_exp.c`, line
// 149, bounds its error by 0.5 + 1.11/N ulp plus the error of its
// polynomial. For the pair below and n = 3, the quotient is
// 0xBFCA15308122C7CD. Its exponential lies 0.498 ulp above
// 0x3FEA19CB18EFFD95, but glibc returns 0x3FEA19CB18EFFD96, and the low half
// of the root changes. In 1.4 million pairs and exponents, most of them
// random, 852 seeds differ, and so do 790 roots. In the five cases
// inspected, glibc `log` rounds correctly and `exp` does not. Resolution:
// floaty takes its correctly rounded `exp` and `log`, so the root does not
// depend on the C library. This test checks that floaty gives the root of
// the seeded copy, not that of the library.
#[test]
fn qd_root_takes_a_correctly_rounded_seed_where_glibc_does_not() {
    qd::check_host_fma();
    let pair = Pair::new(0x3FFD_7C53_810A_8E72, 0x3C67_3420_B1D3_9224);
    let (correct, _) = seed(pair.hi, 3, &Env::X86_SSE);
    assert_eq!(
        (library_seed(pair.hi, 3), correct),
        (0x3FEA_19CB_18EF_FD96, 0x3FEA_19CB_18EF_FD95)
    );
    let library = qd::nroot_library(pair, 3, Rounding::TiesToEven).result;
    let copy = qd::nroot(pair, 3, correct, Rounding::TiesToEven).result;
    let ours = pair_of(value(pair).nroot(3));
    assert_eq!(ours, copy);
    assert_ne!(ours, library);
}
