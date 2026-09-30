//! Compares floaty's double-double arithmetic with its references: libgcc's
//! IBM `long double` routines under QEMU for `Gcc`, and QD's `dd_real` for
//! `Qd`.
//!
//! Each case compares both halves of the result bit for bit, with the sign
//! and payload of a NaN half, and the five IEEE flags, in each of the four
//! rounding directions. floaty runs `Gcc` under the behavior of PowerPC,
//! `ibm_ldouble::behavior`, and `Qd` under `Env::X86_SSE`. The test counts
//! the cases with a malformed operand, a finite high half and a NaN low half,
//! whose NaN the fused NaN order decides.
//!
//! Each case runs again with saturation, and must give the result of the
//! reference. Double-double arithmetic ignores saturation, because neither
//! reference saturates. The tests count the cases in which a step overflows.
//! In a direction that rounds an overflow to an infinity, a saturating step
//! would change the result of such a case.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use std::collections::BTreeMap;

use floaty::env::Tininess;
use floaty::{DoubleDouble, Env, F64, Flags, Gcc, Qd};
use floaty_verify::ibm_ldouble::{
    self, Case, Flags as ReferenceFlags, Operation, Outcome, Pair, Rounding,
};
use floaty_verify::qd;
use floaty_verify::random::SplitMix64;

/// The number of random operand pairs of each test.
const PAIRS: usize = 24_000;

/// The mask of the exponent field of a binary64 encoding.
const EXPONENT_MASK: u64 = 0x7FF0_0000_0000_0000;

/// Returns a binary64 encoding with this sign, exponent field, and fraction.
fn encode(negative: bool, field: u64, fraction: u64) -> u64 {
    (u64::from(negative) << 63) | (field << 52) | (fraction & ((1 << 52) - 1))
}

/// Seeded random operands, biased toward the edges of double-double
/// arithmetic.
struct Operands {
    random: SplitMix64,
}

impl Operands {
    fn below(&mut self, bound: u64) -> u64 {
        self.random.next_u64() % bound
    }

    fn sign(&mut self) -> bool {
        self.random.next_u64() & 1 == 1
    }

    /// Returns an exponent field, often at the ends of the range, at the
    /// scaling threshold of `__gcc_qdiv`, or near 1.
    fn field(&mut self) -> u64 {
        match self.below(8) {
            0 => self.below(60),
            1 => 2046 - self.below(60),
            2 => 54 + self.below(3),
            3 => 1023 - 30 + self.below(60),
            _ => 1 + self.below(2046),
        }
    }

    /// Returns a well-formed pair: a finite high half, and a low half below
    /// half a unit in the last place of the high half.
    fn well_formed(&mut self) -> Pair {
        let field = self.field();
        let hi = encode(self.sign(), field, self.random.next_u64());
        if field < 55 || self.below(8) == 0 {
            return Pair::new(hi, encode(self.sign(), 0, 0));
        }
        let low_field = field - 54 - self.below((field - 54).min(70));
        let lo = encode(self.sign(), low_field, self.random.next_u64());
        Pair::new(hi, lo)
    }

    /// Returns a special value or a value at an edge of binary64.
    fn edge(&mut self) -> u64 {
        const EDGES: [u64; 10] = [
            0,
            0x0000_0000_0000_0001,
            0x000F_FFFF_FFFF_FFFF,
            0x0010_0000_0000_0000,
            0x0360_0000_0000_0000,
            0x3FF0_0000_0000_0000,
            0x7FEF_FFFF_FFFF_FFFF,
            0x7FF0_0000_0000_0000,
            0x7FF8_0000_0000_0001,
            0x7FF0_0000_0000_0002,
        ];
        let edge = EDGES[usize::try_from(self.below(10)).expect("an index fits a usize")];
        edge | (u64::from(self.sign()) << 63)
    }

    /// Returns an operand.
    fn operand(&mut self) -> Pair {
        match self.below(12) {
            0..=6 => self.well_formed(),
            7 | 8 => Pair::new(self.edge(), encode(self.sign(), 0, 0)),
            9 => Pair::new(self.edge(), self.edge()),
            10 => Pair::new(self.random.next_u64(), self.random.next_u64()),
            _ => {
                let pair = self.well_formed();
                Pair::new(pair.hi, self.edge())
            }
        }
    }

    /// Returns a second operand: often one near the first, for cancellation
    /// and exact quotients.
    fn second(&mut self, first: Pair) -> Pair {
        match self.below(9) {
            0 => Pair::new(first.hi ^ (1 << 63), first.lo ^ (1 << 63)),
            1 => Pair::new(first.hi ^ (1 << 63), self.well_formed().lo),
            2 => first,
            3 => self.near_tiny(first.hi, true),
            4 => self.near_tiny(first.hi, false),
            5 => self.near_tiny(first.lo, true),
            _ => self.operand(),
        }
    }

    /// Returns an operand whose high half times `half`, or `half` divided by
    /// the high half, lies just below 2^-1022, where the tininess rule decides
    /// the underflow flag. With the low half of the first operand, the cross
    /// product of `__gcc_qmul` lands there while the main product stays
    /// normal. The host `f64` only picks the operand.
    fn near_tiny(&mut self, half: u64, product: bool) -> Pair {
        let a = f64::from_bits(half);
        if !(a.is_normal() || a.is_subnormal()) {
            return self.operand();
        }
        let steps = f64::from(u8::try_from(self.below(4)).expect("a step fits a u8"));
        let target = f64::MIN_POSITIVE * (1.0 - steps * f64::EPSILON / 2.0);
        let b = if product { target / a } else { a / target };
        let sign = u64::from(self.sign()) << 63;
        Pair::new(b.abs().to_bits() | sign, encode(self.sign(), 0, 0))
    }
}

/// Operand pairs that reach rare paths of the libgcc routines.
const FIXED: [(Pair, Pair); 5] = [
    // `w = c * b` of `__gcc_qmul` rounds up to 2^-1022: underflow only with
    // tininess before rounding.
    (
        Pair::new(0x3FF8_0000_0000_0000, 0x000F_FFFF_FFFF_FFFF),
        Pair::new(0x3FF0_0000_0000_0001, 0),
    ),
    // `a + c` overflows, and the sum of every term is finite again: the
    // recomputation from 0x80 of `__gcc_qadd`, with each operand larger.
    (
        Pair::new(0x7FEF_FFFF_FFFF_FFFF, 0xFC90_0000_0000_0000),
        Pair::new(0x7C90_0000_0000_0000, 0),
    ),
    (
        Pair::new(0x7C90_0000_0000_0000, 0),
        Pair::new(0x7FEF_FFFF_FFFF_FFFF, 0xFC90_0000_0000_0000),
    ),
    // The same for `__gcc_qsub`, from 0x180.
    (
        Pair::new(0x7FEF_FFFF_FFFF_FFFF, 0xFC90_0000_0000_0000),
        Pair::new(0xFC90_0000_0000_0000, 0),
    ),
    (
        Pair::new(0xFC90_0000_0000_0000, 0),
        Pair::new(0x7FEF_FFFF_FFFF_FFFF, 0xFC90_0000_0000_0000),
    ),
];

/// Returns `true` when a pair has a finite high half and a NaN low half.
fn malformed(pair: Pair) -> bool {
    let nan = |bits: u64| bits & EXPONENT_MASK == EXPONENT_MASK && bits & ((1 << 52) - 1) != 0;
    bits_finite(pair.hi) && nan(pair.lo)
}

fn bits_finite(bits: u64) -> bool {
    bits & EXPONENT_MASK != EXPONENT_MASK
}

/// Counts of one test.
#[derive(Default)]
struct Tally {
    passed: usize,
    failures: Vec<String>,
    failed: usize,
    skipped: BTreeMap<&'static str, usize>,
}

impl Tally {
    fn check(&mut self, context: impl FnOnce() -> String, ours: Outcome, theirs: Outcome) {
        if ours == theirs {
            self.passed += 1;
            return;
        }
        self.failed += 1;
        if self.failures.len() < 40 {
            self.failures.push(format!(
                "{}: floaty {:?} {:?}, reference {:?} {:?}",
                context(),
                ours.result,
                ours.flags,
                theirs.result,
                theirs.flags
            ));
        }
    }

    fn report(&self, title: &str) {
        println!("{title}: {} passed, {} failed", self.passed, self.failed);
        for (reason, count) in &self.skipped {
            println!("  skipped {count:7}: {reason}");
        }
        for failure in &self.failures {
            println!("  FAILED {failure}");
        }
    }
}

/// Returns floaty's outcome of a pair of halves and its flags.
fn outcome(hi: F64, lo: F64, flags: Flags) -> Outcome {
    Outcome {
        result: Pair::new(hi.to_bits(), lo.to_bits()),
        flags: ReferenceFlags::from_floaty(flags),
    }
}

fn value<Alg: floaty::Algorithm>(pair: Pair) -> DoubleDouble<Alg> {
    DoubleDouble::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
}

fn value_in<Alg: floaty::Algorithm, M: floaty::env::Mode>(pair: Pair) -> DoubleDouble<Alg, M> {
    let half = |bits| F64::from_bits(bits).with_mode::<M>();
    DoubleDouble::from_parts(half(pair.hi), half(pair.lo))
}

/// The note of a case with a malformed operand: a finite high half and a NaN
/// low half. The fused NaN order of PowerPC decides its NaN.
const MALFORMED: &str = "an operand with a finite high half and a NaN low half";

/// The note of a case whose flags depend on the tininess rule.
const TININESS: &str = "flags that tininess after rounding changes";

/// The note of a case in which a step overflows.
const OVERFLOW: &str = "a step that overflows";

#[test]
fn gcc_matches_libgcc_under_qemu() {
    let mut operands = Operands {
        random: SplitMix64::new(0x1B3_0128),
    };
    let mut tally = Tally::default();
    let mut cases = Vec::new();
    let random = (0..PAIRS).map(|_| {
        let a = operands.operand();
        (a, operands.second(a))
    });
    let pairs: Vec<(Pair, Pair)> = FIXED.into_iter().chain(random).collect();
    let mut malformed_cases = 0;
    for (a, b) in pairs {
        for operation in Operation::ALL {
            if malformed(a) || malformed(b) {
                malformed_cases += 4;
            }
            for rounding in Rounding::ALL {
                cases.push(Case {
                    operation,
                    rounding,
                    a,
                    b,
                });
            }
        }
    }
    let outcomes = ibm_ldouble::run(&cases);
    let (mut tininess, mut overflow) = (0, 0);
    for (case, theirs) in cases.iter().zip(outcomes) {
        let (x, y) = (value::<Gcc>(case.a), value::<Gcc>(case.b));
        let run = |env: Env| {
            let (result, flags) = match case.operation {
                Operation::Add => x.add_with(y, env),
                Operation::Sub => x.sub_with(y, env),
                Operation::Mul => x.mul_with(y, env),
                Operation::Div => x.div_with(y, env),
            };
            outcome(result.hi(), result.lo(), flags)
        };
        let behavior = ibm_ldouble::behavior(case.rounding);
        let ours = run(behavior);
        let after = run(behavior.with_tininess(Tininess::AfterRounding));
        if after.flags != ours.flags {
            tininess += 1;
        }
        if theirs.flags.contains(ReferenceFlags::OVERFLOW) {
            overflow += 1;
        }
        tally.check(|| format!("{case:?}"), ours, theirs);
        let saturated = run(behavior.with_saturate(true));
        tally.check(|| format!("{case:?} with saturation"), saturated, theirs);
    }
    tally.report("Gcc against libgcc");
    println!("  noted   {malformed_cases:7}: {MALFORMED}");
    println!("  noted   {tininess:7}: {TININESS}");
    println!("  noted   {overflow:7}: {OVERFLOW}");
    assert_eq!(tally.failed, 0, "floaty matches libgcc");
    // The generator is seeded, so the counts are exact. Each case runs with
    // and without saturation. The notes show that the tests reach the fused
    // NaN order, the tininess rule of PowerPC, and overflowing steps.
    assert_eq!(
        (tally.passed, malformed_cases, tininess, overflow),
        (2 * 384_080, 14_880, 333, 21_999)
    );
}

/// Stops the test when the C library `fma(x, y, z)` that QD calls is not
/// `y.mul_add(x, z)` under `Env::X86_SSE`, which floaty's `Qd` follows. glibc
/// selects its `fma` at run time. On a processor with FMA3 it is
/// `vfmadd213sd`, which computes `y * x + z` and takes the first NaN in that
/// order. Another `fma` makes many `Qd` cases differ for one reason.
fn check_host_fma() {
    // 0, 1, infinity, two quiet NaNs, and a signaling NaN.
    let values = [
        0_u64,
        0x3FF0_0000_0000_0000,
        0x7FF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,
        0xFFF8_0000_0000_0002,
        0x7FF0_0000_0000_0003,
    ];
    for x in values {
        for y in values {
            for z in values {
                let (ours, _) = F64::from_bits(y).mul_add_with(
                    F64::from_bits(x),
                    F64::from_bits(z),
                    Env::X86_SSE,
                );
                let theirs = qd::c_fma(x, y, z);
                assert_eq!(
                    theirs,
                    ours.to_bits(),
                    "the C library fma({x:#x}, {y:#x}, {z:#x}) of this host is not y * x + z under \
                     Env::X86_SSE. The Qd tests need a processor on which glibc's fma is the FMA3 \
                     instruction"
                );
            }
        }
    }
}

#[test]
fn qd_matches_the_qd_library() {
    check_host_fma();
    let mut operands = Operands {
        random: SplitMix64::new(0x0D_0128),
    };
    let mut tally = Tally::default();
    let random = (0..PAIRS).map(|_| {
        let a = operands.operand();
        (a, operands.second(a))
    });
    let pairs: Vec<(Pair, Pair)> = FIXED.into_iter().chain(random).collect();
    let (mut operator_checks, mut overflow) = (0_usize, 0_usize);
    for (a, b) in pairs {
        let (x, y) = (value::<Qd>(a), value::<Qd>(b));
        for rounding in Rounding::ALL {
            let env = Env::X86_SSE.with_rounding(rounding.into());
            let binary = [
                ("add", x.add_with(y, env), qd::add(a, b, rounding)),
                ("sub", x.sub_with(y, env), qd::sub(a, b, rounding)),
                ("mul", x.mul_with(y, env), qd::mul(a, b, rounding)),
                ("div", x.div_with(y, env), qd::div(a, b, rounding)),
                ("sqrt", x.sqrt_with(env), qd::sqrt(a, rounding)),
            ];
            let saturating = env.with_saturate(true);
            let saturated = [
                x.add_with(y, saturating),
                x.sub_with(y, saturating),
                x.mul_with(y, saturating),
                x.div_with(y, saturating),
                x.sqrt_with(saturating),
            ]
            .map(|(result, flags)| outcome(result.hi(), result.lo(), flags));
            // The same operations with the static mode of the direction.
            let fixed = floaty_verify::with_rounding_mode!(
                floaty::Rounding::from(rounding),
                floaty::mode::X86Sse,
                Mode => {
                    let (x, y) = (value_in::<Qd, Mode>(a), value_in::<Qd, Mode>(b));
                    [
                        x.add_with(y, Mode::default()),
                        x.sub_with(y, Mode::default()),
                        x.mul_with(y, Mode::default()),
                        x.div_with(y, Mode::default()),
                        x.sqrt_with(Mode::default()),
                    ]
                    .map(|(result, flags)| {
                        let (hi, lo) = (result.hi().with_mode(), result.lo().with_mode());
                        outcome(hi, lo, flags)
                    })
                }
            );
            // The operators and `sqrt` of the SSE mode, which rounds to
            // nearest even, return no flags and take the host paths of
            // binary64 where the build has them.
            let operators = (rounding == Rounding::TiesToEven).then(|| {
                let (x, y) = (
                    value_in::<Qd, floaty::mode::X86Sse>(a),
                    value_in::<Qd, floaty::mode::X86Sse>(b),
                );
                [x + y, x - y, x * y, x / y, x.sqrt()]
                    .map(|result| (result.hi().to_bits(), result.lo().to_bits()))
            });
            for (index, ((name, (result, flags), theirs), fixed)) in
                binary.into_iter().zip(fixed).enumerate()
            {
                let ours = outcome(result.hi(), result.lo(), flags);
                tally.check(|| format!("{name} {a:?} {b:?} {rounding:?}"), ours, theirs);
                let context = || format!("{name} {a:?} {b:?} {rounding:?} static mode");
                tally.check(context, fixed, theirs);
                let context = || format!("{name} {a:?} {b:?} {rounding:?} with saturation");
                tally.check(context, saturated[index], theirs);
                if theirs.flags.contains(ReferenceFlags::OVERFLOW) {
                    overflow += 1;
                }
                if let Some(operators) = operators {
                    assert_eq!(
                        operators[index],
                        (theirs.result.hi, theirs.result.lo),
                        "{name} {a:?} {b:?}: the operator of the SSE mode"
                    );
                    operator_checks += 1;
                }
            }
        }
    }
    tally.report("Qd against QD");
    println!("  noted   {overflow:7}: {OVERFLOW}");
    assert_eq!(tally.failed, 0, "floaty matches QD");
    // The generator is seeded, so the counts are exact: each of the 480,100
    // cases runs with an Env, with that Env and saturation, and with a static
    // mode. Each case that rounds to nearest even also runs through the
    // operators of the SSE mode.
    assert_eq!(tally.passed, 3 * 480_100);
    assert_eq!(operator_checks, 480_100 / 4);
    assert_eq!(overflow, 26_065);
}
