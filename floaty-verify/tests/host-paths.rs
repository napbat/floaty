//! Checks that every entry point of a host path gives the result of its
//! `_with` method in the default mode, on every target.
//!
//! The other tests compare the `_with` methods, which always run the engine,
//! with the oracles. The engine gives the same bits on every host. So on
//! AArch64, where the C references do not build, this comparison checks each
//! host path against those oracles. In a build without a host path, the
//! entry points take the engine.

use floaty::{BF16, DoubleDouble, F8E4M3, F8E4M3Fnuz, F8E5M2, F8E5M2Fnuz, F64, F80, Gcc, Qd, TF32};
use floaty_verify::encodings::{IntegerBit, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;

/// Checks that each operator gives the result of its `_with` method under the
/// default mode. The operands are random and boundary encodings, and every
/// pair of an FP8 format. An operator takes a host path where the build has
/// one, and the engine otherwise.
macro_rules! operators_match {
    ($alias:ty, $bits:ty, $pairs:expr) => {{
        for (a, b) in $pairs {
            let (x, y) = (<$alias>::from_bits(a), <$alias>::from_bits(b));
            let env = <$alias>::ENV;
            let context = format!("{} {x:?} {y:?}", stringify!($alias));
            assert_eq!(
                (x + y).to_bits(),
                x.add_with(y, env).0.to_bits(),
                "{context} +"
            );
            assert_eq!(
                (x - y).to_bits(),
                x.sub_with(y, env).0.to_bits(),
                "{context} -"
            );
            assert_eq!(
                (x * y).to_bits(),
                x.mul_with(y, env).0.to_bits(),
                "{context} *"
            );
            assert_eq!(
                (x / y).to_bits(),
                x.div_with(y, env).0.to_bits(),
                "{context} /"
            );
        }
    }};
}

/// Returns random encodings of a storage type, and the encodings at the
/// boundaries of a layout, in pairs.
fn random_pairs<T: TryFrom<u128>>(
    random: &mut SplitMix64,
    width: u32,
    exponent_bits: u32,
    count: usize,
) -> Vec<(T, T)>
where
    T::Error: core::fmt::Debug,
{
    let integer_bit = if width == 80 {
        IntegerBit::Explicit
    } else {
        IntegerBit::Implicit
    };
    let mut encodings = boundary_encodings_u128(width, exponent_bits, integer_bit);
    encodings.extend((0..count).map(|_| random.next_u128() >> (128 - width)));
    let convert = |bits: u128| T::try_from(bits).expect("the encoding fits the storage");
    encodings
        .iter()
        .zip(encodings.iter().rev())
        .map(|(&a, &b)| (convert(a), convert(b)))
        .collect()
}

#[test]
fn operators_give_the_default_mode_results() {
    let every_pair = || (0..=u8::MAX).flat_map(|a| (0..=u8::MAX).map(move |b| (a, b)));
    operators_match!(F8E4M3, u8, every_pair());
    operators_match!(F8E5M2, u8, every_pair());
    operators_match!(F8E4M3Fnuz, u8, every_pair());
    operators_match!(F8E5M2Fnuz, u8, every_pair());
    let mut random = SplitMix64::new(0x0B0B);
    operators_match!(
        floaty::F16,
        u16,
        random_pairs::<u16>(&mut random, 16, 5, 20_000)
    );
    operators_match!(BF16, u16, random_pairs::<u16>(&mut random, 16, 8, 20_000));
    operators_match!(TF32, u32, random_pairs::<u32>(&mut random, 19, 8, 20_000));
    operators_match!(
        floaty::F32,
        u32,
        random_pairs::<u32>(&mut random, 32, 8, 100_000)
    );
    operators_match!(
        floaty::F64,
        u64,
        random_pairs::<u64>(&mut random, 64, 11, 100_000)
    );
    operators_match!(F80, u128, random_pairs::<u128>(&mut random, 80, 15, 20_000));
    operators_match!(
        floaty::F128,
        u128,
        random_pairs::<u128>(&mut random, 128, 15, 20_000)
    );
}

/// Checks that `sqrt` and `mul_add` give the results of `sqrt_with` and
/// `mul_add_with` under the default mode. Each case takes the square root of
/// its first operand. `sqrt` and `mul_add` take a host path where the build
/// has one, and the engine otherwise.
macro_rules! methods_match {
    ($alias:ty, $triples:expr) => {{
        for (a, b, c) in $triples {
            let [x, y, z] = [a, b, c].map(<$alias>::from_bits);
            let env = <$alias>::ENV;
            let context = format!("{} {x:?} {y:?} {z:?}", stringify!($alias));
            assert_eq!(
                x.sqrt().to_bits(),
                x.sqrt_with(env).0.to_bits(),
                "{context} sqrt"
            );
            assert_eq!(
                x.mul_add(y, z).to_bits(),
                x.mul_add_with(y, z, env).0.to_bits(),
                "{context} mul_add"
            );
        }
    }};
}

/// Returns triples of the operands of `pairs`: each pair with the first
/// operand of a later pair as the addend.
fn triples<T: Copy>(pairs: &[(T, T)]) -> Vec<(T, T, T)> {
    let addends = pairs.iter().cycle().skip(pairs.len() / 3 + 1);
    pairs
        .iter()
        .zip(addends)
        .map(|(&(a, b), &(c, _))| (a, b, c))
        .collect()
}

#[test]
fn square_roots_and_fused_products_give_the_default_mode_results() {
    let every_pair: Vec<(u8, u8)> = (0..=u8::MAX)
        .flat_map(|a| (0..=u8::MAX).map(move |b| (a, b)))
        .collect();
    methods_match!(F8E4M3, triples(&every_pair));
    methods_match!(F8E5M2, triples(&every_pair));
    methods_match!(F8E4M3Fnuz, triples(&every_pair));
    methods_match!(F8E5M2Fnuz, triples(&every_pair));
    let mut random = SplitMix64::new(0x5A5A);
    methods_match!(
        floaty::F16,
        triples(&random_pairs::<u16>(&mut random, 16, 5, 20_000))
    );
    methods_match!(
        BF16,
        triples(&random_pairs::<u16>(&mut random, 16, 8, 20_000))
    );
    methods_match!(
        TF32,
        triples(&random_pairs::<u32>(&mut random, 19, 8, 20_000))
    );
    methods_match!(
        floaty::F32,
        triples(&random_pairs::<u32>(&mut random, 32, 8, 100_000))
    );
    methods_match!(
        floaty::F64,
        triples(&random_pairs::<u64>(&mut random, 64, 11, 100_000))
    );
    methods_match!(
        F80,
        triples(&random_pairs::<u128>(&mut random, 80, 15, 20_000))
    );
    methods_match!(
        floaty::F128,
        triples(&random_pairs::<u128>(&mut random, 128, 15, 20_000))
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

/// Every pair of binary16 operands through the four operators, in 16 threads.
/// Run time in a release build: about one minute on x86-64, and about ten
/// minutes under `qemu-aarch64`.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn every_binary16_pair_gives_the_default_mode_results() {
    std::thread::scope(|scope| {
        for part in 0..16_u16 {
            scope.spawn(move || {
                for high in part * 0x1000..(part + 1) * 0x1000 {
                    for low in 0..=u16::MAX {
                        let (x, y) = (floaty::F16::from_bits(high), floaty::F16::from_bits(low));
                        let env = floaty::F16::ENV;
                        let ours = [x + y, x - y, x * y, x / y].map(floaty::F16::to_bits);
                        let engine = [
                            x.add_with(y, env).0,
                            x.sub_with(y, env).0,
                            x.mul_with(y, env).0,
                            x.div_with(y, env).0,
                        ]
                        .map(floaty::F16::to_bits);
                        assert_eq!(ours, engine, "{high:#06x} {low:#06x}");
                    }
                }
            });
        }
    });
}

/// Returns double-double operands: halves from the boundary encodings and
/// random bits of binary64, which give NaN halves, infinities, subnormal
/// halves, and malformed pairs, and well-formed pairs with a small low half.
fn double_double_pairs(random: &mut SplitMix64) -> Vec<(F64, F64)> {
    let boundaries = boundary_encodings_u128(64, 11, IntegerBit::Implicit);
    let half = |bits: u128| F64::from_bits(u64::try_from(bits).expect("a binary64 encoding"));
    let mut halves: Vec<(F64, F64)> = boundaries
        .iter()
        .zip(boundaries.iter().rev())
        .map(|(&hi, &lo)| (half(hi), half(lo)))
        .collect();
    halves.extend((0..20_000).map(|_| {
        let hi = F64::from_bits(random.next_u64());
        // A low half near 2^-60 of the high half, or random bits.
        let lo = if random.next_u64() % 4 == 0 {
            F64::from_bits(random.next_u64())
        } else {
            hi.scale_b(-60)
                .mul_with(
                    F64::from_bits(random.next_u64() >> 12 | 0x3FF0_0000_0000_0000),
                    floaty::Env::IEEE,
                )
                .0
        };
        (hi, lo)
    }));
    halves
}

/// Checks that the operators of a double-double algorithm give the results of
/// its `_with` methods under the default mode.
fn double_double_operators_match<Alg: floaty::Algorithm>(pairs: &[(F64, F64)]) {
    let value = |&(hi, lo): &(F64, F64)| DoubleDouble::<Alg>::from_parts(hi, lo);
    let bits = |value: DoubleDouble<Alg>| (value.hi().to_bits(), value.lo().to_bits());
    for (first, second) in pairs.iter().zip(pairs.iter().rev()) {
        let (x, y) = (value(first), value(second));
        let env = floaty::Env::IEEE;
        let context = format!("{x:?} {y:?}");
        assert_eq!(bits(x + y), bits(x.add_with(y, env).0), "{context} +");
        assert_eq!(bits(x - y), bits(x.sub_with(y, env).0), "{context} -");
        assert_eq!(bits(x * y), bits(x.mul_with(y, env).0), "{context} *");
        assert_eq!(bits(x / y), bits(x.div_with(y, env).0), "{context} /");
    }
}

#[test]
fn double_double_operators_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0xDD00);
    let pairs = double_double_pairs(&mut random);
    double_double_operators_match::<Qd>(&pairs);
    double_double_operators_match::<Gcc>(&pairs);
    for pair in &pairs {
        let value = DoubleDouble::<Qd>::from_parts(pair.0, pair.1);
        let (root, expected) = (value.sqrt(), value.sqrt_with(floaty::Env::IEEE).0);
        assert_eq!(
            (root.hi().to_bits(), root.lo().to_bits()),
            (expected.hi().to_bits(), expected.lo().to_bits()),
            "sqrt {value:?}"
        );
    }
}
