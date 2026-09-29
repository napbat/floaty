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

/// Every pair of bfloat16 operands through the four operators, in 16 threads.
/// Run time in a release build: about one minute on x86-64, and about six
/// minutes under `qemu-aarch64`.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn every_bfloat16_pair_gives_the_default_mode_results() {
    std::thread::scope(|scope| {
        for part in 0..16_u16 {
            scope.spawn(move || {
                for high in part * 0x1000..(part + 1) * 0x1000 {
                    for low in 0..=u16::MAX {
                        let (x, y) = (BF16::from_bits(high), BF16::from_bits(low));
                        let env = BF16::ENV;
                        let ours = [x + y, x - y, x * y, x / y].map(BF16::to_bits);
                        let engine = [
                            x.add_with(y, env).0,
                            x.sub_with(y, env).0,
                            x.mul_with(y, env).0,
                            x.div_with(y, env).0,
                        ]
                        .map(BF16::to_bits);
                        assert_eq!(ours, engine, "{high:#06x} {low:#06x}");
                    }
                }
            });
        }
    });
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

/// Returns binary64 operands for the conversions: boundary and random
/// encodings, values near the halfway points of binary32 and binary16, and
/// values near the bounds of the integer types.
fn conversion_operands(random: &mut SplitMix64) -> Vec<F64> {
    let mut bits: Vec<u64> = boundary_encodings_u128(64, 11, IntegerBit::Implicit)
        .into_iter()
        .map(|bits| u64::try_from(bits).expect("a binary64 encoding"))
        .collect();
    bits.extend((0..20_000).map(|_| random.next_u64()));
    // A binary64 value holds 29 bits below binary32 and 42 below binary16:
    // the halfway point, and one unit on each side of it.
    for shift in [28, 41] {
        for _ in 0..5_000 {
            let base = (random.next_u64() >> 1) & !((1 << (shift + 1)) - 1);
            let half = base | (1 << shift);
            bits.extend([half - 1, half, half + 1]);
        }
    }
    let mut values: Vec<F64> = bits.into_iter().map(F64::from_bits).collect();
    // Integers and halves near the bounds of 32- and 64-bit integers.
    for exponent in [31, 32, 53, 63, 64] {
        let bound = F64::from_bits(0x3FF0_0000_0000_0000).scale_b(exponent);
        let half = F64::from_bits(0x3FE0_0000_0000_0000);
        for value in [
            bound,
            -bound,
            bound.next_up(),
            bound.next_down(),
            -bound.next_down(),
        ] {
            values.push(value);
            values.push(value.add_with(half, floaty::Env::IEEE).0);
            values.push(value.sub_with(half, floaty::Env::IEEE).0);
        }
    }
    values
}

/// Returns x87 extended operands: boundary and random encodings, with
/// unsupported encodings and pseudo-denormals among them, values near the
/// halfway points of binary64 and binary32, and integers and halves near the
/// bounds of 32- and 64-bit integers.
fn extended_operands(random: &mut SplitMix64) -> Vec<F80> {
    let mut bits = boundary_encodings_u128(80, 15, IntegerBit::Explicit);
    bits.extend((0..20_000).map(|_| random.next_u128() >> 48));
    // An x87 significand holds 11 bits below binary64 and 40 below binary32:
    // the halfway point, and one unit on each side of it, at exponents near
    // 1.0 and near the bounds of 64-bit integers.
    for shift in [10, 39] {
        for _ in 0..5_000 {
            let exponent = u128::from(0x3FFF - 70 + random.next_u64() % 140);
            let high = (random.next_u64() | 1 << 63) & !((1 << (shift + 1)) - 1);
            let half = u128::from(high | 1 << shift);
            bits.extend([half - 1, half, half + 1].map(|significand| exponent << 64 | significand));
        }
    }
    let mut values: Vec<F80> = bits.into_iter().map(F80::from_bits).collect();
    let one = F80::from_bits(0x3FFF_8000_0000_0000_0000);
    let half = F80::from_bits(0x3FFE_8000_0000_0000_0000);
    for exponent in [31, 32, 63, 64] {
        let bound = one.scale_b(exponent);
        for value in [
            bound,
            -bound,
            bound.next_up(),
            bound.next_down(),
            -bound.next_down(),
        ] {
            values.push(value);
            values.push(value.add_with(half, floaty::Env::IEEE).0);
            values.push(value.sub_with(half, floaty::Env::IEEE).0);
        }
    }
    values
}

/// Checks that `convert` gives the result of `convert_with` under the mode of
/// the destination.
macro_rules! convert_matches {
    ($destination:ty, $values:expr) => {{
        for value in $values {
            let converted: $destination = value.convert();
            let (expected, _): ($destination, _) = value.convert_with(<$destination>::ENV);
            assert_eq!(
                converted.to_bits(),
                expected.to_bits(),
                "{value:?} to {}",
                stringify!($destination)
            );
        }
    }};
}

#[test]
fn conversions_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0xC0C0);
    let doubles = conversion_operands(&mut random);
    let singles: Vec<floaty::F32> = doubles.iter().map(|value| value.convert()).collect();
    let halves: Vec<floaty::F16> = (0..=u16::MAX).map(floaty::F16::from_bits).collect();
    convert_matches!(floaty::F32, doubles.iter().copied());
    convert_matches!(floaty::F16, doubles.iter().copied());
    convert_matches!(BF16, doubles.iter().copied());
    convert_matches!(floaty::F64, singles.iter().copied());
    convert_matches!(floaty::F16, singles.iter().copied());
    convert_matches!(BF16, singles.iter().copied());
    // The halfway points of binary16 in binary32.
    let near_half: Vec<floaty::F32> = (0..20_000_u32)
        .flat_map(|index| {
            let base = (index.wrapping_mul(0x9E37_79B9) >> 1) & !0x1FFF;
            [base | 0xFFF, base | 0x1000, base | 0x1001].map(floaty::F32::from_bits)
        })
        .collect();
    convert_matches!(floaty::F16, near_half.iter().copied());
    // The halfway points of bfloat16 in binary32.
    let near_bfloat: Vec<floaty::F32> = (0..20_000_u32)
        .flat_map(|index| {
            let base = index.wrapping_mul(0x9E37_79B9) & !0xFFFF;
            [base | 0x7FFF, base | 0x8000, base | 0x8001].map(floaty::F32::from_bits)
        })
        .collect();
    convert_matches!(BF16, near_bfloat.iter().copied());
    convert_matches!(floaty::F32, halves.iter().copied());
    convert_matches!(floaty::F64, halves.iter().copied());
    convert_matches!(floaty::F128, halves.iter().copied());
    let extended = extended_operands(&mut random);
    convert_matches!(floaty::F64, extended.iter().copied());
    convert_matches!(floaty::F32, extended.iter().copied());
    convert_matches!(floaty::F16, extended.iter().copied());
    let bfloats: Vec<BF16> = (0..=u16::MAX).map(BF16::from_bits).collect();
    convert_matches!(floaty::F32, bfloats.iter().copied());
    convert_matches!(floaty::F64, bfloats.iter().copied());
    convert_matches!(F80, doubles.iter().copied());
    convert_matches!(F80, singles.iter().copied());
    convert_matches!(F80, halves.iter().copied());
}

/// Checks that `to_int` gives the result of `to_int_with` under the default
/// mode, `Env::IEEE`, for integer types of several widths.
macro_rules! to_int_matches {
    ($values:expr, $($integer:ty),+) => {{
        for value in $values {
            let env = floaty::Env::IEEE;
            $(
                assert_eq!(
                    value.to_int::<$integer>(),
                    value.to_int_with::<$integer>(env).0,
                    "{value:?} to {}",
                    stringify!($integer)
                );
            )+
        }
    }};
}

#[test]
fn conversions_to_integers_give_the_default_mode_results() {
    use floaty::{Int, UInt};
    let mut random = SplitMix64::new(0x1A1A);
    let doubles = conversion_operands(&mut random);
    let singles: Vec<floaty::F32> = doubles.iter().map(|value| value.convert()).collect();
    let halves: Vec<floaty::F16> = (0..=u16::MAX).map(floaty::F16::from_bits).collect();
    to_int_matches!(
        doubles.iter().copied(),
        i8,
        i16,
        i32,
        i64,
        i128,
        u8,
        u16,
        u32,
        u64,
        u128,
        Int<24>,
        UInt<40>
    );
    to_int_matches!(singles.iter().copied(), i8, i32, i64, u8, u32, u64, i128);
    to_int_matches!(halves.iter().copied(), i8, i16, i32, i64, u8, u16, u64);
    to_int_matches!(
        (0..=u16::MAX).map(BF16::from_bits),
        i8,
        i32,
        i64,
        i128,
        u32,
        u64
    );
    to_int_matches!(
        extended_operands(&mut random).into_iter(),
        i8,
        i32,
        i64,
        i128,
        u32,
        u64,
        u128,
        Int<24>
    );
}

#[test]
fn conversions_from_integers_give_the_default_mode_results() {
    let mut random = SplitMix64::new(0x1B1B);
    // Integers of every magnitude, the bounds of each type, and the halfway
    // points and the overflow threshold of binary16.
    let mut integers: Vec<i128> = (0..20_000)
        .map(|_| {
            let bits = random.next_u64() % 127;
            i128::from_le_bytes(random.next_u128().to_le_bytes()) >> bits
        })
        .collect();
    for bound in [
        i128::from(i32::MIN),
        i128::from(i32::MAX),
        i128::from(i64::MIN),
        i128::from(i64::MAX),
        i128::from(u64::MAX),
        1 << 24,
        (1 << 24) + 1,
        1 << 53,
        (1 << 53) + 1,
        2049,
        4097,
        65_504,
        65_520,
        1 << 16,
    ] {
        integers.extend([bound - 1, bound, bound + 1, -bound]);
    }
    let env = floaty::Env::IEEE;
    for integer in integers {
        let check = |ours: u128, expected: u128, format: &str| {
            assert_eq!(ours, expected, "{integer} to {format}");
        };
        check(
            floaty::F64::from_int(integer).to_bits().into(),
            floaty::F64::from_int_with(integer, env).0.to_bits().into(),
            "binary64",
        );
        check(
            floaty::F32::from_int(integer).to_bits().into(),
            floaty::F32::from_int_with(integer, env).0.to_bits().into(),
            "binary32",
        );
        check(
            floaty::F16::from_int(integer).to_bits().into(),
            floaty::F16::from_int_with(integer, env).0.to_bits().into(),
            "binary16",
        );
        check(
            F80::from_int(integer).to_bits(),
            F80::from_int_with(integer, env).0.to_bits(),
            "x87 extended",
        );
        if let Ok(narrow) = i64::try_from(integer) {
            check(
                floaty::F64::from_int(narrow).to_bits().into(),
                floaty::F64::from_int_with(narrow, env).0.to_bits().into(),
                "binary64 from i64",
            );
        }
        if let Ok(narrow) = u64::try_from(integer) {
            check(
                floaty::F32::from_int(narrow).to_bits().into(),
                floaty::F32::from_int_with(narrow, env).0.to_bits().into(),
                "binary32 from u64",
            );
        }
    }
}

#[test]
fn rounding_to_integral_gives_the_default_mode_result() {
    let mut random = SplitMix64::new(0x2B2B);
    for bits in 0..=u16::MAX {
        let value = floaty::F16::from_bits(bits);
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(floaty::F16::ENV).0.to_bits(),
            "{value:?}"
        );
        let value = BF16::from_bits(bits);
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(BF16::ENV).0.to_bits(),
            "{value:?}"
        );
    }
    let doubles = conversion_operands(&mut random);
    // Exact halves, which a tie rounds to even.
    let halves: Vec<F64> = (0..10_000)
        .map(|_| {
            let integer = F64::from_int(random.next_u64() >> 12);
            integer
                .add_with(F64::from_bits(0x3FE0_0000_0000_0000), floaty::Env::IEEE)
                .0
        })
        .collect();
    for value in doubles.iter().chain(&halves).copied() {
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(F64::ENV).0.to_bits(),
            "{value:?}"
        );
        let single: floaty::F32 = value.convert();
        assert_eq!(
            single.round_to_integral().to_bits(),
            single.round_to_integral_with(floaty::F32::ENV).0.to_bits(),
            "{single:?}"
        );
    }
    for value in extended_operands(&mut random) {
        assert_eq!(
            value.round_to_integral().to_bits(),
            value.round_to_integral_with(F80::ENV).0.to_bits(),
            "{value:?}"
        );
    }
}
