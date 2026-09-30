//! Checks the host paths of AArch64 under every value of FPCR that changes a
//! result of the host unit or enables a trap. The test runs only on AArch64,
//! under `qemu-aarch64`, which stands in for AArch64 hardware.

#![cfg(target_arch = "aarch64")]

use core::hint::black_box;

use floaty::{BF16, F16, F32, F64, Lanes};
use floaty_verify::aarch64::{FPCR_SETTINGS, with_fpcr};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::entry_points::{
    LaneArithmetic, arithmetic, arithmetic_with, comparisons, comparisons_with, conversions,
    conversions_with, lane_arithmetic, lane_arithmetic_with,
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

/// Checks the arithmetic entry points of [`arithmetic`] of one type under one
/// FPCR value against the engine results of the default mode, with the first
/// operand as the addend.
macro_rules! operators_under {
    ($alias:ty, $bits:ty, $control:expr, $pairs:expr) => {
        for &(a, b) in $pairs {
            let convert = |bits: u128| <$bits>::try_from(bits).expect("the encoding fits");
            let (x, y) = (
                <$alias>::from_bits(convert(a)),
                <$alias>::from_bits(convert(b)),
            );
            let expected = arithmetic_with(x, y, x);
            let ours = with_fpcr($control, || {
                let (x, y) = (black_box(x), black_box(y));
                arithmetic(x, y, x)
            });
            assert_eq!(ours, expected, "{a:#x} {b:#x} under FPCR {:#x}", $control);
        }
    };
}

/// Checks the conversion entry points of [`conversions`] of one type under
/// one control value against the engine results of the default mode.
macro_rules! conversions_under {
    ($alias:ty, $control:expr, $values:expr) => {
        for &bits in $values {
            let x = <$alias>::from_bits(bits);
            let integer =
                i64::from_ne_bytes(u64::from(bits).to_ne_bytes()) >> (u64::from(bits) % 64);
            let expected = conversions_with(x, integer);
            let ours = with_fpcr($control, || conversions(black_box(x), black_box(integer)));
            assert_eq!(ours, expected, "{bits:#x} {integer} under {:#x}", $control);
        }
    };
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

/// Checks the comparison entry points of [`comparisons`] of one type under
/// one FPCR value against the engine results of the default mode.
macro_rules! comparisons_under {
    ($alias:ty, $bits:ty, $control:expr, $pairs:expr) => {
        for &(a, b) in $pairs {
            let convert = |bits: u128| <$bits>::try_from(bits).expect("the encoding fits");
            let (x, y) = (
                <$alias>::from_bits(convert(a)),
                <$alias>::from_bits(convert(b)),
            );
            let expected = comparisons_with(x, y);
            let ours = with_fpcr($control, || comparisons(black_box(x), black_box(y)));
            assert_eq!(ours, expected, "{a:#x} {b:#x} under FPCR {:#x}", $control);
        }
    };
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
        for start in (0..encodings.len()).step_by(5) {
            let bits: [u16; 5] =
                core::array::from_fn(|offset| encodings[(start + offset) % encodings.len()]);
            let lanes = Lanes::<BF16, 5>::from_bits(bits);
            let ours = with_fpcr(control, || {
                let lanes = black_box(lanes);
                let single: Lanes<F32, 5> = lanes.convert();
                let double: Lanes<F64, 5> = lanes.convert();
                (
                    lanes.round_to_integral().to_bits(),
                    single.to_bits(),
                    double.to_bits(),
                )
            });
            let values = bits.map(BF16::from_bits);
            let engine = (
                values.map(|value| value.round_to_integral_with(BF16::ENV).0.to_bits()),
                values.map(|value| value.convert_with::<F32>(F32::ENV).0.to_bits()),
                values.map(|value| value.convert_with::<F64>(F64::ENV).0.to_bits()),
            );
            assert_eq!(ours, engine, "{bits:x?} under FPCR {control:#x}");
        }
    }
}

#[test]
fn directed_rounding_reads_fpcr_before_the_vector_unit() {
    // The rounding to integral values in a directed mode takes the direction
    // from the instruction, `FRINTP` here, but the other fields of FPCR still
    // apply. Under each setting, the lanes and the scalar values give the
    // engine results of the mode.
    type Up = floaty::Float<
        floaty::Binary<8>,
        32,
        floaty::mode::Rounded<floaty::mode::Ieee, floaty::mode::direction::TowardPositive>,
    >;
    let mut random = SplitMix64::new(0x00A6_D1E0);
    let encodings = encodings::<u32>(&pairs(&mut random, Layout::BINARY32));
    for control in core::iter::once(0).chain(FPCR_SETTINGS) {
        for start in (0..encodings.len()).step_by(7) {
            let bits: [u32; 5] =
                core::array::from_fn(|offset| encodings[(start + offset) % encodings.len()]);
            let lanes = Lanes::<Up, 5>::from_bits(bits);
            let ours = with_fpcr(control, || {
                let lanes = black_box(lanes);
                (
                    lanes.round_to_integral().to_bits(),
                    lanes
                        .into_array()
                        .map(|value| value.round_to_integral().to_bits()),
                )
            });
            let engine = bits.map(|bits| {
                Up::from_bits(bits)
                    .round_to_integral_with(Up::ENV)
                    .0
                    .to_bits()
            });
            assert_eq!(ours, (engine, engine), "{bits:x?} under FPCR {control:#x}");
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
            let ours = with_fpcr(control, || {
                let (x, y) = (black_box(x), black_box(y));
                (
                    x.compare_quiet(y),
                    [
                        x.minimum(y).to_bits(),
                        x.maximum_number(y).to_bits(),
                        x.min_num(y).to_bits(),
                    ],
                )
            });
            let (a, b) = (x.into_array(), y.into_array());
            let env = F32::ENV;
            let engine = (
                core::array::from_fn::<_, 5, _>(|index| {
                    a[index].compare_quiet_with(b[index], env).0
                }),
                [
                    core::array::from_fn::<_, 5, _>(|index| {
                        a[index].minimum_with(b[index], env).0.to_bits()
                    }),
                    core::array::from_fn::<_, 5, _>(|index| {
                        a[index].maximum_number_with(b[index], env).0.to_bits()
                    }),
                    core::array::from_fn::<_, 5, _>(|index| {
                        a[index].min_num_with(b[index], env).0.to_bits()
                    }),
                ],
            );
            assert_eq!(ours, engine, "lanes at {start} under FPCR {control:#x}");
        }
    }
}
