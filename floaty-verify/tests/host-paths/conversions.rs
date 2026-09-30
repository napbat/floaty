//! The conversions between formats, to integers, and from integers.

use floaty::{BF16, F80};
use floaty_verify::random::SplitMix64;

use super::operands::{conversion_operands, extended_operands};

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
            let bits = random.below(127);
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
