//! The operators, `sqrt`, and `mul_add`.

use floaty::{BF16, F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F80, TF32};
use floaty_verify::encodings::Layout;
use floaty_verify::entry_points::{arithmetic, arithmetic_with};
use floaty_verify::random::SplitMix64;

use super::operands::{every_16_bit_pair, random_pairs, triples};

/// Checks that each arithmetic entry point of [`arithmetic`] gives the result
/// of its `_with` method under the default mode, for each triple `x`, `y`,
/// and the addend `z`. An entry point takes a host path where the build has
/// one, and the engine otherwise.
macro_rules! arithmetic_matches {
    ($alias:ty, $triples:expr) => {{
        for (a, b, c) in $triples {
            let [x, y, z] = [a, b, c].map(<$alias>::from_bits);
            assert_eq!(
                arithmetic(x, y, z),
                arithmetic_with(x, y, z),
                "{} {x:?} {y:?} {z:?}",
                stringify!($alias)
            );
        }
    }};
}

/// Returns each pair `x` and `y` as a triple with `x` as the addend.
fn with_first_addend<T: Copy>(pairs: impl IntoIterator<Item = (T, T)>) -> Vec<(T, T, T)> {
    pairs.into_iter().map(|(a, b)| (a, b, a)).collect()
}

/// The binary16 fused multiply-add of a host path without `FEAT_FP16`
/// rounds twice: to binary64, and then to binary16. The binary64 sum is
/// inexact only for a product far below the addend, or for a product far
/// above the range of binary16. Each addend meets products far below it, of
/// both signs, and the largest products meet each addend.
#[test]
fn binary16_fused_products_round_once() {
    use floaty::{Env, F16};
    // 2^-24, 3 * 2^-24, the largest subnormal, the smallest normal, and 1.
    let tiny = [0x0001_u16, 0x0003, 0x03FF, 0x0400, 0x3C00];
    // 65504, 32768, and 255.875.
    let large = [0x7BFF_u16, 0x7800, 0x5BFF];
    let factors: Vec<(u16, u16)> = tiny
        .iter()
        .flat_map(|&a| tiny.iter().map(move |&b| (a, b)))
        .chain(
            large
                .iter()
                .flat_map(|&a| large.iter().map(move |&b| (a, b))),
        )
        .collect();
    for addend in (0..=u16::MAX).map(F16::from_bits) {
        for &(a, b) in &factors {
            for sign in [0, 0x8000] {
                let (x, y) = (F16::from_bits(a | sign), F16::from_bits(b));
                assert_eq!(
                    x.mul_add(y, addend).to_bits(),
                    x.mul_add_with(y, addend, Env::IEEE).0.to_bits(),
                    "{x:?} {y:?} {addend:?}"
                );
            }
        }
    }
}

/// The pairs are random and boundary encodings, and every pair of an FP8
/// format, with the first operand as the addend.
#[test]
fn operators_give_the_default_mode_results() {
    let every_pair = || (0..=u8::MAX).flat_map(|a| (0..=u8::MAX).map(move |b| (a, b)));
    arithmetic_matches!(F8E4M3Fn, with_first_addend(every_pair()));
    arithmetic_matches!(F8E5M2, with_first_addend(every_pair()));
    arithmetic_matches!(F8E4M3Fnuz, with_first_addend(every_pair()));
    arithmetic_matches!(F8E5M2Fnuz, with_first_addend(every_pair()));
    let mut random = SplitMix64::new(0x0B0B);
    arithmetic_matches!(
        floaty::F16,
        with_first_addend(random_pairs::<u16>(&mut random, Layout::BINARY16, 20_000))
    );
    arithmetic_matches!(
        BF16,
        with_first_addend(random_pairs::<u16>(&mut random, Layout::BFLOAT16, 20_000))
    );
    arithmetic_matches!(
        TF32,
        with_first_addend(random_pairs::<u32>(&mut random, Layout::TF32, 20_000))
    );
    arithmetic_matches!(
        floaty::F32,
        with_first_addend(random_pairs::<u32>(&mut random, Layout::BINARY32, 100_000))
    );
    arithmetic_matches!(
        floaty::F64,
        with_first_addend(random_pairs::<u64>(&mut random, Layout::BINARY64, 100_000))
    );
    arithmetic_matches!(
        F80,
        with_first_addend(random_pairs::<u128>(
            &mut random,
            Layout::X87_EXTENDED,
            20_000
        ))
    );
    arithmetic_matches!(
        floaty::F128,
        with_first_addend(random_pairs::<u128>(&mut random, Layout::BINARY128, 20_000))
    );
}

#[test]
fn square_roots_and_fused_products_give_the_default_mode_results() {
    let every_pair: Vec<(u8, u8)> = (0..=u8::MAX)
        .flat_map(|a| (0..=u8::MAX).map(move |b| (a, b)))
        .collect();
    arithmetic_matches!(F8E4M3Fn, triples(&every_pair));
    arithmetic_matches!(F8E5M2, triples(&every_pair));
    arithmetic_matches!(F8E4M3Fnuz, triples(&every_pair));
    arithmetic_matches!(F8E5M2Fnuz, triples(&every_pair));
    let mut random = SplitMix64::new(0x5A5A);
    arithmetic_matches!(
        floaty::F16,
        triples(&random_pairs::<u16>(&mut random, Layout::BINARY16, 20_000))
    );
    arithmetic_matches!(
        BF16,
        triples(&random_pairs::<u16>(&mut random, Layout::BFLOAT16, 20_000))
    );
    arithmetic_matches!(
        TF32,
        triples(&random_pairs::<u32>(&mut random, Layout::TF32, 20_000))
    );
    arithmetic_matches!(
        floaty::F32,
        triples(&random_pairs::<u32>(&mut random, Layout::BINARY32, 100_000))
    );
    arithmetic_matches!(
        floaty::F64,
        triples(&random_pairs::<u64>(&mut random, Layout::BINARY64, 100_000))
    );
    arithmetic_matches!(
        F80,
        triples(&random_pairs::<u128>(
            &mut random,
            Layout::X87_EXTENDED,
            20_000
        ))
    );
    arithmetic_matches!(
        floaty::F128,
        triples(&random_pairs::<u128>(
            &mut random,
            Layout::BINARY128,
            20_000
        ))
    );
}

/// Every binary16 square root. The binary16 path computes in binary32 and
/// rounds twice, so this and the sweep below check every result.
#[test]
fn every_binary16_square_root_gives_the_default_mode_result() {
    for bits in 0..=u16::MAX {
        let value = floaty::F16::from_bits(bits);
        assert_eq!(
            value.sqrt().to_bits(),
            value.sqrt_with(floaty::F16::ENV).0.to_bits(),
            "sqrt({bits:#06x})"
        );
    }
}

/// Every bfloat16 square root. The bfloat16 path computes in binary32 and
/// rounds twice, so this and the sweep below check every result.
#[test]
fn every_bfloat16_square_root_gives_the_default_mode_result() {
    for bits in 0..=u16::MAX {
        let value = BF16::from_bits(bits);
        assert_eq!(
            value.sqrt().to_bits(),
            value.sqrt_with(BF16::ENV).0.to_bits(),
            "sqrt({bits:#06x})"
        );
    }
}

/// Checks that the four operators of the 16-bit format `$alias` give the
/// results of their `_with` methods under the default mode, for every pair of
/// operands. The sweep checks the operators alone, not every entry point of
/// [`arithmetic`], which keeps its run time. The other tests of this module
/// check every entry point.
macro_rules! every_pair_matches {
    ($alias:ty) => {
        every_16_bit_pair(|high, low| {
            let (x, y) = (<$alias>::from_bits(high), <$alias>::from_bits(low));
            let env = <$alias>::ENV;
            let ours = [x + y, x - y, x * y, x / y].map(<$alias>::to_bits);
            let engine = [
                x.add_with(y, env).0,
                x.sub_with(y, env).0,
                x.mul_with(y, env).0,
                x.div_with(y, env).0,
            ]
            .map(<$alias>::to_bits);
            assert_eq!(ours, engine, "{high:#06x} {low:#06x}");
        })
    };
}

/// Every pair of bfloat16 operands through the four operators, in 16 threads.
/// Run time in a release build: about one minute on x86-64, and about six
/// minutes under `qemu-aarch64`.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn every_bfloat16_pair_gives_the_default_mode_results() {
    every_pair_matches!(BF16);
}

/// Every pair of binary16 operands through the four operators, in 16 threads.
/// Run time in a release build: about one minute on x86-64, and about ten
/// minutes under `qemu-aarch64`.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn every_binary16_pair_gives_the_default_mode_results() {
    every_pair_matches!(floaty::F16);
}
