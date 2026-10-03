//! Compares `DoubleDouble<Qd>::sqr` and `inv` with QD 2.3.24.
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
