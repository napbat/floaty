//! The operators, `sqrt`, and `mul_add`.

use floaty::{BF16, F8E4M3Fn, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F80, TF32};
use floaty_verify::encodings::Layout;
use floaty_verify::entry_points::{arithmetic, arithmetic_with};
use floaty_verify::random::SplitMix64;

use super::operands::{every_16_bit_pair, random_pairs, triples};

/// Checks that each arithmetic entry point of [`arithmetic`] gives the result
/// of its `_with` method under the default mode, for each pair `x` and `y`,
/// with `x` as the addend. The operands are random and boundary encodings,
/// and every pair of an FP8 format. An entry point takes a host path where the
/// build has one, and the engine otherwise.
macro_rules! operators_match {
    ($alias:ty, $bits:ty, $pairs:expr) => {{
        for (a, b) in $pairs {
            let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
            assert_eq!(
                arithmetic(x, y, x),
                arithmetic_with(x, y, x),
                "{} {x:?} {y:?}",
                stringify!($alias)
            );
        }
    }};
}

#[test]
fn operators_give_the_default_mode_results() {
    let every_pair = || (0..=u8::MAX).flat_map(|a| (0..=u8::MAX).map(move |b| (a, b)));
    operators_match!(F8E4M3Fn, u8, every_pair());
    operators_match!(F8E5M2, u8, every_pair());
    operators_match!(F8E4M3Fnuz, u8, every_pair());
    operators_match!(F8E5M2Fnuz, u8, every_pair());
    let mut random = SplitMix64::new(0x0B0B);
    operators_match!(
        floaty::F16,
        u16,
        random_pairs::<u16>(&mut random, Layout::BINARY16, 20_000)
    );
    operators_match!(
        BF16,
        u16,
        random_pairs::<u16>(&mut random, Layout::BFLOAT16, 20_000)
    );
    operators_match!(
        TF32,
        u32,
        random_pairs::<u32>(&mut random, Layout::TF32, 20_000)
    );
    operators_match!(
        floaty::F32,
        u32,
        random_pairs::<u32>(&mut random, Layout::BINARY32, 100_000)
    );
    operators_match!(
        floaty::F64,
        u64,
        random_pairs::<u64>(&mut random, Layout::BINARY64, 100_000)
    );
    operators_match!(
        F80,
        u128,
        random_pairs::<u128>(&mut random, Layout::X87_EXTENDED, 20_000)
    );
    operators_match!(
        floaty::F128,
        u128,
        random_pairs::<u128>(&mut random, Layout::BINARY128, 20_000)
    );
}

/// Checks that each arithmetic entry point of [`arithmetic`] gives the result
/// of its `_with` method under the default mode, for each triple `x`, `y`,
/// and the addend `z`. An entry point takes a host path where the build has
/// one, and the engine otherwise.
macro_rules! methods_match {
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

#[test]
fn square_roots_and_fused_products_give_the_default_mode_results() {
    let every_pair: Vec<(u8, u8)> = (0..=u8::MAX)
        .flat_map(|a| (0..=u8::MAX).map(move |b| (a, b)))
        .collect();
    methods_match!(F8E4M3Fn, triples(&every_pair));
    methods_match!(F8E5M2, triples(&every_pair));
    methods_match!(F8E4M3Fnuz, triples(&every_pair));
    methods_match!(F8E5M2Fnuz, triples(&every_pair));
    let mut random = SplitMix64::new(0x5A5A);
    methods_match!(
        floaty::F16,
        triples(&random_pairs::<u16>(&mut random, Layout::BINARY16, 20_000))
    );
    methods_match!(
        BF16,
        triples(&random_pairs::<u16>(&mut random, Layout::BFLOAT16, 20_000))
    );
    methods_match!(
        TF32,
        triples(&random_pairs::<u32>(&mut random, Layout::TF32, 20_000))
    );
    methods_match!(
        floaty::F32,
        triples(&random_pairs::<u32>(&mut random, Layout::BINARY32, 100_000))
    );
    methods_match!(
        floaty::F64,
        triples(&random_pairs::<u64>(&mut random, Layout::BINARY64, 100_000))
    );
    methods_match!(
        F80,
        triples(&random_pairs::<u128>(
            &mut random,
            Layout::X87_EXTENDED,
            20_000
        ))
    );
    methods_match!(
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
