//! Compares floaty's double-double functions with their references: the IBM
//! `long double` functions of the libm of glibc 2.43 under QEMU for `Gcc`
//! (`sqrtl`, `nextupl`, `nextdownl`, `fmodl`, `remainderl`, `fmal`, and
//! `iscanonicall`), and QD's `drem` and `fmod` for `Qd`.
//!
//! Each case compares both halves of the result bit for bit, with the sign
//! and payload of a NaN half, and the five IEEE flags, in each of the four
//! rounding directions. floaty runs `Gcc` under the behavior of PowerPC,
//! `ibm_ldouble::behavior`, and `Qd` under `Env::X86_SSE`. `Qd` also runs
//! with flush-to-zero and denormals-are-zero, and QD with the FTZ and DAZ
//! bits of MXCSR. In the direction to nearest, the operations of the default
//! mode, the reference mode of each algorithm, give the results of the
//! reference.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{DoubleDouble, Env, F64, Flags, Gcc, Qd};
use floaty_verify::double_double::pair;
use floaty_verify::encodings::Layout;
use floaty_verify::ibm_ldouble::{
    self, Flags as ReferenceFlags, Function, FunctionCase, Outcome, Pair, Rounding,
};
use floaty_verify::qd;
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::sse_env;

/// The layout of the halves.
const BINARY64: Layout = Layout::BINARY64;

/// Returns a random pair: a canonical pair of any magnitude, a pair near
/// the edges of the range, or a pair of the generator of the exact-value
/// tests, which holds special and random halves.
fn operand(random: &mut SplitMix64) -> Pair {
    let negative = random.coin_flip();
    match random.below(4) {
        0 | 1 => {
            let field = match random.below(4) {
                0 => random.below(120),
                1 => 2046 - random.below(60),
                _ => 1 + random.below(2046),
            };
            let hi = BINARY64.encode_u64(negative, field, random.next_u64());
            if field < 56 || random.below(6) == 0 {
                return Pair::new(hi, 0);
            }
            let low_field = field - 54 - random.below((field - 54).min(70));
            let lo = BINARY64.encode_u64(random.coin_flip(), low_field, random.next_u64());
            Pair::new(hi, lo)
        }
        _ => pair(random),
    }
}

fn value(pair: Pair) -> DoubleDouble<Gcc> {
    DoubleDouble::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
}

/// Returns floaty's outcome of a result and its flags.
fn outcome((result, flags): (DoubleDouble<Gcc>, Flags)) -> Outcome {
    Outcome {
        result: Pair::new(result.hi().to_bits(), result.lo().to_bits()),
        flags: ReferenceFlags::from_floaty(flags),
    }
}

/// Returns a divisor for a dividend: random, a scaled or truncated copy of
/// the dividend, far below or above it, or a divisor that divides a
/// multiple of it.
fn divisor(random: &mut SplitMix64, dividend: Pair) -> Pair {
    let x = value(dividend);
    let scaled = |scale: i32| {
        let result = x.scale_b(scale);
        Pair::new(result.hi().to_bits(), result.lo().to_bits())
    };
    match random.below(8) {
        0 => scaled(-i32::try_from(random.below(120)).expect("below 120")),
        1 => scaled(-i32::try_from(random.below(2100)).expect("below 2100")),
        2 => Pair::new(dividend.hi, 0),
        3 => Pair::new(dividend.hi ^ 1 << 63, dividend.lo),
        4 => {
            // A dividend that is an integer times the divisor, near a tie.
            let quotient = F64::from_bits(BINARY64.encode_u64(
                false,
                1023 + random.below(60),
                random.next_u64(),
            ));
            let quotient = DoubleDouble::<Gcc>::from_f64(quotient.round_to_integral());
            let result = x / quotient;
            Pair::new(result.hi().to_bits(), result.lo().to_bits())
        }
        _ => operand(random),
    }
}

/// Compares every case with glibc, and returns the number of cases.
fn compare(cases: &[FunctionCase]) -> usize {
    let outcomes = ibm_ldouble::run_functions(cases);
    let mut failures = Vec::new();
    for (case, theirs) in cases.iter().zip(outcomes) {
        let env = ibm_ldouble::behavior(case.rounding);
        let [a, b, _] = case.operands.map(value);
        let ours = match case.function {
            Function::Sqrt => outcome(a.sqrt_with(env)),
            Function::NextUp => outcome(a.next_up_with(env)),
            Function::NextDown => outcome(a.next_down_with(env)),
            Function::Fmod => outcome(a.truncated_remainder_with(b, env)),
            Function::Remainder => outcome(a.remainder_with(b, env)),
            Function::IsCanonical => {
                let one = if a.is_canonical() {
                    0x3FF0_0000_0000_0000
                } else {
                    0
                };
                Outcome {
                    result: Pair::new(one, 0),
                    flags: ReferenceFlags::NONE,
                }
            }
            Function::MulAdd => {
                let c = value(case.operands[2]);
                outcome(a.mul_add_with(b, c, env))
            }
        };
        if ours != theirs && failures.len() < 40 {
            failures.push(format!(
                "{case:?}: floaty {:?} {:?}, glibc {:?} {:?}",
                ours.result, ours.flags, theirs.result, theirs.flags
            ));
        }
        // The functions of the default mode, the reference mode of `Gcc`,
        // return no flags.
        let default = match case.function {
            _ if case.rounding != Rounding::TiesToEven => None,
            Function::Sqrt => Some(a.sqrt()),
            Function::NextUp => Some(a.next_up()),
            Function::NextDown => Some(a.next_down()),
            Function::Fmod => Some(a % b),
            Function::Remainder => Some(a.remainder(b)),
            Function::MulAdd => Some(a.mul_add(b, value(case.operands[2]))),
            Function::IsCanonical => None,
        };
        if let Some(result) = default {
            let ours = Pair::new(result.hi().to_bits(), result.lo().to_bits());
            if ours != theirs.result && failures.len() < 40 {
                failures.push(format!(
                    "{case:?} in the default mode: floaty {ours:?}, glibc {:?}",
                    theirs.result
                ));
            }
        }
    }
    for failure in &failures {
        println!("FAILED {failure}");
    }
    assert!(failures.is_empty(), "floaty matches glibc");
    cases.len()
}

/// Returns the cases of a function of one operand for every operand and
/// rounding direction.
fn unary(function: Function, operands: &[Pair]) -> Vec<FunctionCase> {
    operands
        .iter()
        .flat_map(|&a| {
            Rounding::ALL.map(|rounding| FunctionCase {
                function,
                rounding,
                operands: [a, Pair::default(), Pair::default()],
            })
        })
        .collect()
}

#[test]
fn sqrt_next_up_and_next_down_match_glibc() {
    let mut random = SplitMix64::new(0x6_11BC);
    let edges = [
        Pair::new(0, 0),
        Pair::new(1 << 63, 0),
        Pair::new(0, 0x8000_0000_0000_0001),
        Pair::new(0x7FEF_FFFF_FFFF_FFFF, 0x7C8F_FFFF_FFFF_FFFE),
        Pair::new(0xFFF0_0000_0000_0000, 0),
        Pair::new(0x0360_0000_0000_0000, 0x8000_0000_0000_0001),
        Pair::new(0x3FF0_0000_0000_0000, 0xBC90_0000_0000_0000),
        Pair::new(0x0000_0000_0000_0001, 0),
    ];
    let mut operands: Vec<Pair> = edges
        .into_iter()
        .chain((0..20_000).map(|_| operand(&mut random)))
        .collect();
    operands.extend((0..2_000).map(|_| far_low(&mut random)));
    let mut count = 0;
    for function in [Function::Sqrt, Function::NextUp, Function::NextDown] {
        count += compare(&unary(function, &operands));
    }
    // Positive operands reach the Newton steps of the square root.
    let positive: Vec<Pair> = operands
        .iter()
        .map(|pair| Pair::new(pair.hi & !(1 << 63), pair.lo))
        .collect();
    count += compare(&unary(Function::Sqrt, &positive));
    assert_eq!(count, 4 * 4 * 22_008);
}

/// Returns a pair whose low half is about 1,075 exponents below its high
/// half, so that `sqrtl` scales the low half to the subnormal range of
/// binary64, or below it.
fn far_low(random: &mut SplitMix64) -> Pair {
    let field = 1100 + random.below(940);
    let low_field = field - 1016 - random.below(70);
    let sign = |random: &mut SplitMix64| random.coin_flip();
    Pair::new(
        BINARY64.encode_u64(false, field, random.next_u64()),
        BINARY64.encode_u64(sign(random), low_field, random.next_u64()),
    )
}

#[test]
fn fmod_and_remainder_match_glibc() {
    let mut random = SplitMix64::new(0x6_F30D);
    let mut cases = Vec::new();
    for _ in 0..20_000 {
        let x = operand(&mut random);
        let y = divisor(&mut random, x);
        for function in [Function::Fmod, Function::Remainder] {
            for rounding in Rounding::ALL {
                cases.push(FunctionCase {
                    function,
                    rounding,
                    operands: [x, y, Pair::default()],
                });
            }
        }
    }
    let special = ties(&mut random);
    let count = special.len();
    cases.extend(special.into_iter().flat_map(|(x, y)| {
        [Function::Fmod, Function::Remainder]
            .into_iter()
            .flat_map(move |function| {
                Rounding::ALL.map(|rounding| FunctionCase {
                    function,
                    rounding,
                    operands: [x, y, Pair::default()],
                })
            })
    }));
    assert_eq!(compare(&cases), 2 * 4 * (20_000 + count));
}

/// Returns dividend and divisor pairs that reach the ties of the remainders:
/// a dividend `(2k + 1.5)` times the divisor, with a normal and with a
/// subnormal divisor, and a dividend with a `-0` high half.
fn ties(random: &mut SplitMix64) -> Vec<(Pair, Pair)> {
    let mut pairs = Vec::new();
    for _ in 0..500 {
        let subnormal = random.below(4) == 0;
        let field = if subnormal { 0 } else { 1 + random.below(2000) };
        // A divisor with at most 30 significant bits, and an even one when
        // subnormal, so that the dividend is exact.
        let fraction = (random.next_u64() >> 34) << 22;
        let divisor = F64::from_bits(BINARY64.encode_u64(random.coin_flip(), field, fraction));
        let multiple = F64::from_int(4 * random.below(256) + 3);
        let (dividend, flags) = divisor
            .mul_with(multiple, floaty::Env::IEEE)
            .0
            .mul_with(F64::from_bits(0x3FE0_0000_0000_0000), floaty::Env::IEEE);
        if flags.is_empty() && dividend.is_finite() {
            pairs.push((
                Pair::new(dividend.to_bits(), 0),
                Pair::new(divisor.to_bits(), 0),
            ));
        }
        let tiny = BINARY64.encode_u64(random.coin_flip(), random.below(3), random.next_u64());
        pairs.push((Pair::new(1 << 63, tiny), Pair::new(divisor.to_bits(), 0)));
    }
    pairs
}

/// Returns an addend for `x * y`: random, zero, the negated product, which
/// cancels all but its error, or a value far from the product.
fn addend(random: &mut SplitMix64, x: Pair, y: Pair) -> Pair {
    match random.below(5) {
        0 => Pair::new(u64::from(random.coin_flip()) << 63, 0),
        1 | 2 => {
            let product = value(x) * value(y);
            let scale = i32::try_from(random.below(8)).expect("below 8") - 4;
            let negated = (-product).scale_b(if random.coin_flip() { scale } else { 0 });
            Pair::new(negated.hi().to_bits(), negated.lo().to_bits())
        }
        _ => operand(random),
    }
}

#[test]
fn mul_add_matches_glibc() {
    let mut random = SplitMix64::new(0x6_F3A1);
    let mut cases = Vec::new();
    for _ in 0..20_000 {
        let x = operand(&mut random);
        let y = if random.below(4) == 0 {
            divisor(&mut random, x)
        } else {
            operand(&mut random)
        };
        let z = addend(&mut random, x, y);
        for rounding in Rounding::ALL {
            cases.push(FunctionCase {
                function: Function::MulAdd,
                rounding,
                operands: [x, y, z],
            });
        }
    }
    assert_eq!(compare(&cases), 4 * 20_000);
}

/// Returns a malformed pair: a finite high half with an infinite low half,
/// or a NaN low half with a payload of its own.
fn malformed(random: &mut SplitMix64) -> Pair {
    let hi = BINARY64.encode_u64(
        random.coin_flip(),
        1 + random.below(2046),
        random.next_u64(),
    );
    let sign = u64::from(random.coin_flip()) << 63;
    let lo = if random.below(3) == 0 {
        0x7FF0_0000_0000_0000
    } else {
        0x7FF8_0000_0000_0000 | (random.next_u64() & 0xFFFF)
    };
    Pair::new(hi, lo | sign)
}

#[test]
fn mul_add_of_malformed_operands_matches_glibc() {
    // The checks of `fmal` read only the high halves, so NaNs of the low
    // halves reach its sort and its sums, which decide which NaN survives.
    let mut random = SplitMix64::new(0x6_BAD1);
    let mut cases = Vec::new();
    for _ in 0..10_000 {
        let [x, y, z] = [0; 3].map(|_| {
            if random.coin_flip() {
                operand(&mut random)
            } else {
                malformed(&mut random)
            }
        });
        for rounding in [Rounding::TiesToEven, Rounding::TowardNegative] {
            cases.push(FunctionCase {
                function: Function::MulAdd,
                rounding,
                operands: [x, y, z],
            });
        }
    }
    // Small values sort below the NaNs, whose exponent is 0, so two NaNs of
    // other payloads reach the top of the sort and meet in one sum.
    let small = |random: &mut SplitMix64| {
        let hi = BINARY64.encode_u64(
            random.coin_flip(),
            900 + random.below(120),
            random.next_u64(),
        );
        Pair::new(hi, 0x7FF8_0000_0000_0000 | (random.next_u64() & 0xFFFF))
    };
    for _ in 0..5_000 {
        let [x, y, z] = [0; 3].map(|_| small(&mut random));
        for rounding in [Rounding::TiesToEven, Rounding::TowardPositive] {
            cases.push(FunctionCase {
                function: Function::MulAdd,
                rounding,
                operands: [x, y, z],
            });
        }
    }
    assert_eq!(compare(&cases), 2 * 15_000);
}

#[test]
fn is_canonical_matches_glibc() {
    let mut random = SplitMix64::new(0x6_CA70);
    let mut operands: Vec<Pair> = (0..50_000).map(|_| operand(&mut random)).collect();
    // Low halves at half an ulp of an even and of an odd high half, just
    // below and above it, and with the other sign, normal and subnormal.
    for hi in [
        0x3FF0_0000_0000_0000_u64,
        0x3FF0_0000_0000_0001,
        0x0370_0000_0000_0000,
        0x0350_0000_0000_0001,
        0x7FEF_FFFF_FFFF_FFFF,
    ] {
        let field = hi >> 52;
        let half = if field <= 53 {
            1 << (field - 2)
        } else {
            (field - 53) << 52
        };
        operands.extend([half, half - 1, half + 1, half | 1 << 63].map(|lo| Pair::new(hi, lo)));
    }
    let cases: Vec<FunctionCase> = unary(Function::IsCanonical, &operands)
        .into_iter()
        .filter(|case| case.rounding == Rounding::TiesToEven)
        .collect();
    assert_eq!(compare(&cases), 50_020);
}

/// Returns the operand pairs of the `Qd` remainder tests: fixed pairs with a
/// tiny dividend, and seeded random pairs.
fn remainder_pairs() -> Vec<(Pair, Pair)> {
    let mut random = SplitMix64::new(0x0D_F30D);
    // A tiny dividend and a larger divisor round the quotient to a zero,
    // whose sign in the directed roundings reaches the low half.
    let tiny = [
        0x0000_0000_0000_0001_u64,
        0x0360_0000_0000_0000,
        0x3FD0_0000_0000_0000,
    ];
    let fixed = tiny.into_iter().flat_map(|a| {
        [
            0x3FF0_0000_0000_0000_u64,
            0xC340_0000_0000_0000,
            0x7FEF_FFFF_FFFF_FFFF,
        ]
        .map(|b| (Pair::new(a, 0), Pair::new(b, 0)))
    });
    let random_pairs = (0..20_000).map(|_| {
        let a = operand(&mut random);
        (a, divisor(&mut random, a))
    });
    fixed.chain(random_pairs).collect()
}

/// Returns the `Qd` value of a pair.
fn qd_value(pair: Pair) -> DoubleDouble<Qd> {
    DoubleDouble::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
}

/// Returns floaty's outcome of a result and its flags.
fn qd_outcome((result, flags): (DoubleDouble<Qd>, Flags)) -> Outcome {
    Outcome {
        result: Pair::new(result.hi().to_bits(), result.lo().to_bits()),
        flags: ReferenceFlags::from_floaty(flags),
    }
}

#[test]
fn qd_remainders_match_qd() {
    qd::check_host_fma();
    let mut failures = Vec::new();
    let mut count = 0;
    for (a, b) in remainder_pairs() {
        let (x, y) = (qd_value(a), qd_value(b));
        for rounding in Rounding::ALL {
            let env = Env::X86_SSE.with_rounding(rounding.into());
            let checks = [
                ("drem", x.remainder_with(y, env), qd::drem(a, b, rounding)),
                (
                    "fmod",
                    x.truncated_remainder_with(y, env),
                    qd::fmod(a, b, rounding),
                ),
            ];
            // The remainders of the default mode, the reference mode of
            // `Qd`, return no flags.
            if rounding == Rounding::TiesToEven {
                for (name, result, (_, _, theirs)) in [("drem", x.remainder(y)), ("%", x % y)]
                    .into_iter()
                    .zip(&checks)
                    .map(|((name, result), check)| (name, result, check))
                {
                    let ours = Pair::new(result.hi().to_bits(), result.lo().to_bits());
                    if ours != theirs.result && failures.len() < 40 {
                        failures.push(format!(
                            "{name} {a:?} {b:?} in the default mode: floaty {ours:?}, QD {:?}",
                            theirs.result
                        ));
                    }
                }
            }
            for (name, result, theirs) in checks {
                let ours = qd_outcome(result);
                count += 1;
                if ours != theirs && failures.len() < 40 {
                    failures.push(format!(
                        "{name} {a:?} {b:?} {rounding:?}: floaty {:?} {:?}, QD {:?} {:?}",
                        ours.result, ours.flags, theirs.result, theirs.flags
                    ));
                }
            }
        }
    }
    for failure in &failures {
        println!("FAILED {failure}");
    }
    assert!(failures.is_empty(), "floaty matches QD");
    assert_eq!(count, 2 * 4 * 20_009);
}

#[test]
fn qd_remainders_match_qd_with_flush_to_zero_and_denormals_are_zero() {
    qd::check_host_fma();
    let mut failures = Vec::new();
    // The cases in which FTZ or DAZ changes the result or the flags of QD.
    let (mut count, mut changed) = (0, 0);
    for (a, b) in remainder_pairs() {
        let (x, y) = (qd_value(a), qd_value(b));
        for rounding in Rounding::ALL {
            let reference = || [qd::drem(a, b, rounding), qd::fmod(a, b, rounding)];
            let default = reference();
            for (ftz, daz) in qd::FLUSH_SETTINGS {
                let env = sse_env(rounding.into(), ftz, daz);
                let ours =
                    [x.remainder_with(y, env), x.truncated_remainder_with(y, env)].map(qd_outcome);
                let theirs = qd::with_flush(ftz, daz, reference);
                for (index, (ours, theirs)) in ours.into_iter().zip(theirs).enumerate() {
                    count += 1;
                    changed += usize::from(theirs != default[index]);
                    if ours != theirs && failures.len() < 40 {
                        let name = ["drem", "fmod"][index];
                        failures.push(format!(
                            "{name} {a:?} {b:?} {rounding:?} FTZ {ftz} DAZ {daz}: floaty {:?} \
                             {:?}, QD {:?} {:?}",
                            ours.result, ours.flags, theirs.result, theirs.flags
                        ));
                    }
                }
            }
        }
    }
    for failure in &failures {
        println!("FAILED {failure}");
    }
    println!("noted {changed}: results that FTZ or DAZ changes");
    assert!(failures.is_empty(), "floaty matches QD");
    // The generator is seeded, so the counts are exact. The changed results
    // show that QD runs with the MXCSR bits set.
    assert_eq!((count, changed), (3 * 2 * 4 * 20_009, 31_270));
}
