//! Checks the host paths of s390x under every value of the FPC register that
//! changes a result of the host unit or enables a trap. The test runs only on
//! s390x, under `qemu-s390x`, which stands in for z/Architecture hardware.

#![cfg(target_arch = "s390x")]

use floaty::format::Standard;
use floaty::{BF16, Binary, F16, F32, F64};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::entry_points::{
    assert_arithmetic_under, assert_comparisons_under, assert_conversions_under,
    assert_directed_rounding_under, assert_lane_arithmetic_under, assert_lane_comparisons_under,
    assert_lane_conversions_under, lane_sets,
};
use floaty_verify::random::SplitMix64;
use floaty_verify::s390x::{FPC_SETTINGS, under_fpc};

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

/// Checks the arithmetic, the comparison, and the conversion entry points of
/// one type on each pair under one FPC value, with the first operand as the
/// addend, and an integer made from the bits of the first operand.
macro_rules! scalars_under {
    ($alias:ty, $bits:ty, $control:expr, $pairs:expr) => {{
        let setting = format!("FPC {:#x}", $control);
        for &(a, b) in $pairs {
            let convert = |bits: u128| <$bits>::try_from(bits).expect("the encoding fits");
            let (x, y) = (
                <$alias>::from_bits(convert(a)),
                <$alias>::from_bits(convert(b)),
            );
            assert_arithmetic_under([x, y, x], &setting, under_fpc($control));
            assert_comparisons_under([x, y], &setting, under_fpc($control));
            let bits = u64::try_from(a).expect("the encoding fits");
            let integer = i64::from_ne_bytes(bits.to_ne_bytes()) >> (bits % 64);
            assert_conversions_under(x, integer, &setting, under_fpc($control));
        }
    }};
}

#[test]
fn operators_read_fpc_before_the_host_unit() {
    // Each setting changes a host result or traps. Under each, and under the
    // default FPC, the scalar entry points give the engine results of the
    // default mode.
    let mut random = SplitMix64::new(0x0539_4000);
    let half = pairs(&mut random, Layout::BINARY16);
    let single = pairs(&mut random, Layout::BINARY32);
    let double = pairs(&mut random, Layout::BINARY64);
    let bfloat = pairs(&mut random, Layout::BFLOAT16);
    for control in core::iter::once(0).chain(FPC_SETTINGS) {
        scalars_under!(F16, u16, control, &half);
        scalars_under!(BF16, u16, control, &bfloat);
        scalars_under!(F32, u32, control, &single);
        scalars_under!(F64, u64, control, &double);
    }
}

#[test]
fn directed_rounding_reads_fpc_before_the_host_unit() {
    // The rounding to integral values in a directed mode takes the direction
    // from the M3 field of `FIEBR`, but the other fields of the FPC register
    // still apply. Under each setting, the lanes and the scalar values give
    // the engine results of the mode.
    let mut random = SplitMix64::new(0x0539_D1E0);
    let encodings: Vec<u32> = pairs(&mut random, Layout::BINARY32)
        .iter()
        .map(|&(bits, _)| u32::try_from(bits).expect("the encoding fits"))
        .collect();
    for control in core::iter::once(0).chain(FPC_SETTINGS) {
        let setting = format!("FPC {control:#x}");
        for start in (0..encodings.len()).step_by(7) {
            let bits: [u32; 5] =
                core::array::from_fn(|offset| encodings[(start + offset) % encodings.len()]);
            assert_directed_rounding_under(bits, &setting, under_fpc(control));
        }
    }
}

/// Checks the arithmetic, the comparisons, and the conversions of `Lanes`
/// of one format and lane count under the default FPC and every setting of
/// `FPC_SETTINGS`, on the lane sets of [`lane_sets`].
fn every_lane_group<S: Standard<W>, const W: usize, const N: usize>(layout: Layout, seed: u64)
where
    S::Bits: TryFrom<u128>,
{
    let mut random = SplitMix64::new(seed);
    let sets = lane_sets::<S, W, N>(layout, &mut random, 1_000);
    for control in core::iter::once(0).chain(FPC_SETTINGS) {
        let setting = format!("FPC {control:#x}");
        for window in sets.windows(3).step_by(5) {
            let [x, y, z] = [window[0], window[1], window[2]];
            assert_lane_arithmetic_under([x, y, z], &setting, under_fpc(control));
            assert_lane_comparisons_under([x, y], &setting, under_fpc(control));
            assert_lane_conversions_under(x, &setting, under_fpc(control));
        }
    }
}

#[test]
fn every_lane_type_reads_fpc_before_the_lane_paths() {
    // The lane counts fill the chunks of four binary32 and two binary64
    // lanes, and leave single lanes.
    every_lane_group::<Binary<8>, 32, 5>(Layout::BINARY32, 0x0539_1A01);
    every_lane_group::<Binary<11>, 64, 3>(Layout::BINARY64, 0x0539_1A02);
    every_lane_group::<Binary<5>, 16, 5>(Layout::BINARY16, 0x0539_1A03);
    every_lane_group::<Binary<8>, 16, 5>(Layout::BFLOAT16, 0x0539_1A04);
}
