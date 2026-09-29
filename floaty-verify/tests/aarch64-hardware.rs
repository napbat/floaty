//! Checks the host paths of AArch64 under every value of FPCR that changes a
//! result of the host unit or enables a trap. The test runs only on AArch64,
//! under `qemu-aarch64`, which stands in for AArch64 hardware.

#![cfg(target_arch = "aarch64")]

use core::hint::black_box;

use floaty::{BF16, F16, F32, F64};
use floaty_verify::aarch64::{FPCR_SETTINGS, with_fpcr};
use floaty_verify::encodings::{IntegerBit, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;

/// Returns boundary and random encodings of a binary format, in pairs.
fn pairs(random: &mut SplitMix64, width: u32, exponent_bits: u32) -> Vec<(u128, u128)> {
    let mut encodings = boundary_encodings_u128(width, exponent_bits, IntegerBit::Implicit);
    encodings.extend((0..2_000).map(|_| random.next_u128() >> (128 - width)));
    encodings
        .iter()
        .zip(encodings.iter().rev())
        .map(|(&a, &b)| (a, b))
        .collect()
}

/// Returns the first operands of `pairs` in the storage type `T`.
fn encodings<T: TryFrom<u128, Error: core::fmt::Debug>>(pairs: &[(u128, u128)]) -> Vec<T> {
    pairs
        .iter()
        .map(|&(bits, _)| T::try_from(bits).expect("the encoding fits"))
        .collect()
}

/// Checks the four operators, `sqrt`, and `mul_add` of one type under one
/// FPCR value against the engine results of the default mode.
macro_rules! operators_under {
    ($alias:ty, $bits:ty, $control:expr, $pairs:expr) => {
        for &(a, b) in $pairs {
            let convert = |bits: u128| <$bits>::try_from(bits).expect("the encoding fits");
            let (x, y) = (
                <$alias>::from_bits(convert(a)),
                <$alias>::from_bits(convert(b)),
            );
            let env = <$alias>::ENV;
            let expected = [
                x.add_with(y, env).0.to_bits(),
                x.sub_with(y, env).0.to_bits(),
                x.mul_with(y, env).0.to_bits(),
                x.div_with(y, env).0.to_bits(),
                x.sqrt_with(env).0.to_bits(),
                x.mul_add_with(y, x, env).0.to_bits(),
            ];
            let ours = with_fpcr($control, || {
                let (x, y) = (black_box(x), black_box(y));
                [
                    (x + y).to_bits(),
                    (x - y).to_bits(),
                    (x * y).to_bits(),
                    (x / y).to_bits(),
                    x.sqrt().to_bits(),
                    x.mul_add(y, x).to_bits(),
                ]
            });
            assert_eq!(ours, expected, "{a:#x} {b:#x} under FPCR {:#x}", $control);
        }
    };
}

/// Checks `round_to_integral`, `to_int`, `from_int`, and the conversions to
/// binary16, binary32, and binary64 of one type under one control value
/// against the engine results of the default mode.
macro_rules! conversions_under {
    ($alias:ty, $control:expr, $values:expr) => {
        for &bits in $values {
            let x = <$alias>::from_bits(bits);
            let integer =
                i64::from_ne_bytes(u64::from(bits).to_ne_bytes()) >> (u64::from(bits) % 64);
            let env = <$alias>::ENV;
            let expected = (
                u64::from(x.round_to_integral_with(env).0.to_bits()),
                x.to_int_with::<i64>(env).0,
                x.to_int_with::<u32>(env).0,
                u64::from(<$alias>::from_int_with(integer, env).0.to_bits()),
                x.convert_with::<F16>(F16::ENV).0.to_bits(),
                x.convert_with::<F32>(F32::ENV).0.to_bits(),
                x.convert_with::<F64>(F64::ENV).0.to_bits(),
            );
            let ours = with_fpcr($control, || {
                let x = black_box(x);
                (
                    u64::from(x.round_to_integral().to_bits()),
                    x.to_int::<i64>(),
                    x.to_int::<u32>(),
                    u64::from(<$alias>::from_int(black_box(integer)).to_bits()),
                    x.convert::<F16>().to_bits(),
                    x.convert::<F32>().to_bits(),
                    x.convert::<F64>().to_bits(),
                )
            });
            assert_eq!(ours, expected, "{bits:#x} {integer} under {:#x}", $control);
        }
    };
}

#[test]
fn operators_read_fpcr_before_the_host_unit() {
    // Each setting changes a host result or traps. Under each, and under the
    // default FPCR, the operators give the engine results of the default mode.
    let mut random = SplitMix64::new(0x00A6_4000);
    let half = pairs(&mut random, 16, 5);
    let single = pairs(&mut random, 32, 8);
    let double = pairs(&mut random, 64, 11);
    let bfloat = pairs(&mut random, 16, 8);
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        operators_under!(F16, u16, control, &half);
        operators_under!(BF16, u16, control, &bfloat);
        operators_under!(F32, u32, control, &single);
        operators_under!(F64, u64, control, &double);
        conversions_under!(F16, control, &encodings::<u16>(&half));
        conversions_under!(BF16, control, &encodings::<u16>(&bfloat));
        conversions_under!(F32, control, &encodings::<u32>(&single));
        conversions_under!(F64, control, &encodings::<u64>(&double));
        for x in encodings::<u32>(&single).into_iter().map(F32::from_bits) {
            let expected = x.convert_with::<BF16>(BF16::ENV).0.to_bits();
            let ours = with_fpcr(control, || black_box(x).convert::<BF16>().to_bits());
            assert_eq!(ours, expected, "{x:?} to BF16 under FPCR {control:#x}");
        }
    }
}
