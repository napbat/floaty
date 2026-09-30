//! Checks the host paths of AArch64 under every value of FPCR that changes a
//! result of the host unit or enables a trap. The test runs only on AArch64,
//! under `qemu-aarch64`, which stands in for AArch64 hardware.

#![cfg(target_arch = "aarch64")]

use core::hint::black_box;

use floaty::{BF16, F16, F32, F64, Lanes};
use floaty_verify::aarch64::{FPCR_SETTINGS, under_fpcr, with_fpcr};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::entry_points::{
    LaneArithmetic, LaneComparisons, assert_arithmetic_under, assert_bfloat16_lanes_under,
    assert_comparisons_under, assert_conversions_under, assert_directed_rounding_under,
    lane_arithmetic, lane_arithmetic_with, lane_comparisons, lane_comparisons_with,
};
use floaty_verify::random::SplitMix64;

/// Returns boundary and random encodings of a binary format, in pairs.
fn pairs(random: &mut SplitMix64, layout: Layout) -> Vec<(u128, u128)> {
    let mut encodings = boundary_encodings_u128(layout);
    encodings.extend((0..2_000).map(|_| random.next_u128() >> (128 - layout.width)));
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

/// Checks the arithmetic entry points of one type on each pair under one
/// FPCR value, with the first operand as the addend.
macro_rules! arithmetic_under {
    ($alias:ty, $bits:ty, $control:expr, $pairs:expr) => {{
        let setting = format!("FPCR {:#x}", $control);
        for &(a, b) in $pairs {
            let convert = |bits: u128| <$bits>::try_from(bits).expect("the encoding fits");
            let (x, y) = (
                <$alias>::from_bits(convert(a)),
                <$alias>::from_bits(convert(b)),
            );
            assert_arithmetic_under([x, y, x], &setting, under_fpcr($control));
        }
    }};
}

/// Checks the comparison entry points of one type on each pair under one
/// FPCR value.
macro_rules! comparisons_under {
    ($alias:ty, $bits:ty, $control:expr, $pairs:expr) => {{
        let setting = format!("FPCR {:#x}", $control);
        for &(a, b) in $pairs {
            let convert = |bits: u128| <$bits>::try_from(bits).expect("the encoding fits");
            let (x, y) = (
                <$alias>::from_bits(convert(a)),
                <$alias>::from_bits(convert(b)),
            );
            assert_comparisons_under([x, y], &setting, under_fpcr($control));
        }
    }};
}

/// Checks the conversion entry points of one type on each value under one
/// control value, with an integer made from the bits of the value.
macro_rules! conversions_under {
    ($alias:ty, $control:expr, $values:expr) => {{
        let setting = format!("FPCR {:#x}", $control);
        for &bits in $values {
            let x = <$alias>::from_bits(bits);
            let integer =
                i64::from_ne_bytes(u64::from(bits).to_ne_bytes()) >> (u64::from(bits) % 64);
            assert_conversions_under(x, integer, &setting, under_fpcr($control));
        }
    }};
}

#[test]
fn operators_read_fpcr_before_the_host_unit() {
    // Each setting changes a host result or traps. Under each, and under the
    // default FPCR, the operators give the engine results of the default mode.
    let mut random = SplitMix64::new(0x00A6_4000);
    let half = pairs(&mut random, Layout::BINARY16);
    let single = pairs(&mut random, Layout::BINARY32);
    let double = pairs(&mut random, Layout::BINARY64);
    let bfloat = pairs(&mut random, Layout::BFLOAT16);
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        // The first operand is also the addend.
        arithmetic_under!(F16, u16, control, &half);
        arithmetic_under!(BF16, u16, control, &bfloat);
        arithmetic_under!(F32, u32, control, &single);
        arithmetic_under!(F64, u64, control, &double);
        conversions_under!(F16, control, &encodings::<u16>(&half));
        conversions_under!(BF16, control, &encodings::<u16>(&bfloat));
        // The conversions include binary32 to bfloat16, which rounds in
        // `BFCVT` with `+bf16`.
        conversions_under!(F32, control, &encodings::<u32>(&single));
        conversions_under!(F64, control, &encodings::<u64>(&double));
    }
}

/// The results of the operations of `Lanes` on five binary32 lanes, three
/// binary64 lanes, and five binary16 lanes, which the packed host paths
/// compute in chunks and single lanes: the arithmetic of each format, and the
/// conversions between the formats.
#[derive(Debug, PartialEq, Eq)]
struct LaneResults {
    singles: LaneArithmetic<u32, 5>,
    doubles: LaneArithmetic<u64, 3>,
    halves: LaneArithmetic<u16, 5>,
    /// The encodings of the conversions: binary32 to binary64, binary64 to
    /// binary32, binary32 to binary16, and binary16 to binary32 and to
    /// binary64.
    conversions: Vec<u64>,
}

/// Returns the encodings of the conversions of [`LaneResults`] in one list.
fn conversion_bits(
    widened: Lanes<F64, 5>,
    narrowed: Lanes<F32, 3>,
    halved: Lanes<F16, 5>,
    half_single: Lanes<F32, 5>,
    half_double: Lanes<F64, 5>,
) -> Vec<u64> {
    let mut bits: Vec<u64> = widened.to_bits().to_vec();
    bits.extend(narrowed.to_bits().map(u64::from));
    bits.extend(halved.to_bits().map(u64::from));
    bits.extend(half_single.to_bits().map(u64::from));
    bits.extend(half_double.to_bits());
    bits
}

/// Returns the results of the operations of `Lanes` without flags, with the
/// first lanes of each format as the addends.
fn lanes_without_flags(
    x: Lanes<F32, 5>,
    y: Lanes<F32, 5>,
    u: Lanes<F64, 3>,
    v: Lanes<F64, 3>,
    half_left: Lanes<F16, 5>,
    half_right: Lanes<F16, 5>,
) -> LaneResults {
    LaneResults {
        singles: lane_arithmetic(x, y, x),
        doubles: lane_arithmetic(u, v, u),
        halves: lane_arithmetic(half_left, half_right, half_left),
        conversions: conversion_bits(
            x.convert(),
            u.convert(),
            x.convert(),
            half_left.convert(),
            half_left.convert(),
        ),
    }
}

/// Returns the results of `lanes_without_flags` from the `_with` methods of
/// `Lanes` in the default mode, which run the engine on each lane.
fn lanes_in_engine(
    x: Lanes<F32, 5>,
    y: Lanes<F32, 5>,
    u: Lanes<F64, 3>,
    v: Lanes<F64, 3>,
    half_left: Lanes<F16, 5>,
    half_right: Lanes<F16, 5>,
) -> LaneResults {
    let (single, double, half) = (F32::ENV, F64::ENV, F16::ENV);
    LaneResults {
        singles: lane_arithmetic_with(x, y, x),
        doubles: lane_arithmetic_with(u, v, u),
        halves: lane_arithmetic_with(half_left, half_right, half_left),
        conversions: conversion_bits(
            x.convert_with(double).0,
            u.convert_with(single).0,
            x.convert_with(half).0,
            half_left.convert_with(single).0,
            half_left.convert_with(double).0,
        ),
    }
}

#[test]
fn lanes_read_fpcr_before_the_vector_unit() {
    // Under each setting of FPCR, and under the default, the operations of
    // `Lanes` without flags give the engine results of the default mode.
    let mut random = SplitMix64::new(0x00A6_1A4E);
    let singles = encodings::<u32>(&pairs(&mut random, Layout::BINARY32));
    let doubles = encodings::<u64>(&pairs(&mut random, Layout::BINARY64));
    let halves = encodings::<u16>(&pairs(&mut random, Layout::BINARY16));
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        for start in (0..singles.len()).step_by(3) {
            let lane = |offset: usize| singles[(start + offset) % singles.len()];
            let double = |offset: usize| doubles[(start + offset) % doubles.len()];
            let half_lane = |offset: usize| halves[(start + offset) % halves.len()];
            let x = Lanes::<F32, 5>::from_bits(core::array::from_fn(lane));
            let y = Lanes::<F32, 5>::from_bits(core::array::from_fn(|offset| lane(offset + 7)));
            let u = Lanes::<F64, 3>::from_bits(core::array::from_fn(double));
            let v = Lanes::<F64, 3>::from_bits(core::array::from_fn(|offset| double(offset + 5)));
            let half_left = Lanes::<F16, 5>::from_bits(core::array::from_fn(half_lane));
            let half_right =
                Lanes::<F16, 5>::from_bits(core::array::from_fn(|offset| half_lane(offset + 11)));
            let ours = with_fpcr(control, || {
                lanes_without_flags(black_box(x), black_box(y), u, v, half_left, half_right)
            });
            assert_eq!(
                ours,
                lanes_in_engine(x, y, u, v, half_left, half_right),
                "lanes at {start} under FPCR {control:#x}"
            );
        }
    }
}

#[test]
fn bfloat16_lanes_read_fpcr_before_the_vector_unit() {
    // Under each setting of FPCR, and under the default, the rounding and the
    // conversions of bfloat16 lanes give the engine results of the default
    // mode.
    let mut random = SplitMix64::new(0x00A6_1ABF);
    let encodings = encodings::<u16>(&pairs(&mut random, Layout::BFLOAT16));
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        let setting = format!("FPCR {control:#x}");
        for start in (0..encodings.len()).step_by(5) {
            let bits: [u16; 5] =
                core::array::from_fn(|offset| encodings[(start + offset) % encodings.len()]);
            assert_bfloat16_lanes_under(bits, &setting, under_fpcr(control));
        }
    }
}

#[test]
fn directed_rounding_reads_fpcr_before_the_vector_unit() {
    // The rounding to integral values in a directed mode takes the direction
    // from the instruction, `FRINTP` here, but the other fields of FPCR still
    // apply. Under each setting, the lanes and the scalar values give the
    // engine results of the mode.
    let mut random = SplitMix64::new(0x00A6_D1E0);
    let encodings = encodings::<u32>(&pairs(&mut random, Layout::BINARY32));
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        let setting = format!("FPCR {control:#x}");
        for start in (0..encodings.len()).step_by(7) {
            let bits: [u32; 5] =
                core::array::from_fn(|offset| encodings[(start + offset) % encodings.len()]);
            assert_directed_rounding_under(bits, &setting, under_fpcr(control));
        }
    }
}

#[test]
fn comparisons_read_fpcr_before_the_host_unit() {
    // Under each setting of FPCR, and under the default, the comparison and
    // the minimum and maximum operations give the engine results of the
    // default mode.
    let mut random = SplitMix64::new(0x00A6_C0AA);
    let single_pairs = pairs(&mut random, Layout::BINARY32);
    let double_pairs = pairs(&mut random, Layout::BINARY64);
    let half_pairs = pairs(&mut random, Layout::BINARY16);
    let bfloat_pairs = pairs(&mut random, Layout::BFLOAT16);
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        comparisons_under!(F32, u32, control, &single_pairs);
        comparisons_under!(F64, u64, control, &double_pairs);
        comparisons_under!(F16, u16, control, &half_pairs);
        comparisons_under!(BF16, u16, control, &bfloat_pairs);
    }
}

#[test]
fn lane_orders_read_fpcr_before_the_vector_unit() {
    // Under each setting of FPCR, and under the default, the packed
    // comparison and the minimum and maximum operations of lanes give the
    // engine results of the default mode.
    let mut random = SplitMix64::new(0x00A6_0AD5);
    let singles = encodings::<u32>(&pairs(&mut random, Layout::BINARY32));
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        for start in (0..singles.len()).step_by(5) {
            let lane = |offset: usize| singles[(start + offset) % singles.len()];
            let x = Lanes::<F32, 5>::from_bits(core::array::from_fn(lane));
            let y = Lanes::<F32, 5>::from_bits(core::array::from_fn(|offset| lane(offset + 3)));
            let ours = with_fpcr(control, || lane_comparisons(black_box(x), black_box(y)));
            let engine = lane_comparisons_with(x, y);
            // The test checks the order, `minimum`, `maximumNumber`, and
            // `minNum` of the shared list.
            let checked = |results: LaneComparisons<u32, 5>| {
                (
                    results.order,
                    [results.minimum, results.maximum_number, results.min_num],
                )
            };
            assert_eq!(
                checked(ours),
                checked(engine),
                "lanes at {start} under FPCR {control:#x}"
            );
        }
    }
}
