//! Compares `Lanes` with the packed instructions of the SSE unit of the host
//! processor. Each lane gives the bits of the packed instruction, and the
//! flags of the instruction are the union of the flags of the lanes. The
//! operations of `Lanes` without flags also give the engine results under
//! every MXCSR setting. The test runs only on x86-64 hosts.

#![cfg(target_arch = "x86_64")]

use std::hint::black_box;

use floaty::format::Standard;
use floaty::{Binary, F16, F32, F64, Flags, Lanes, X87};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::entry_points::{
    LaneArithmetic, assert_bfloat16_lanes_under, assert_directed_rounding_under,
    assert_lane_arithmetic_under, assert_lane_comparisons_under, assert_lane_conversions_under,
    lane_arithmetic, lane_arithmetic_with, lane_sets,
};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, MXCSR_HOST_PATH_CONTROLS, ROUND_USE_MXCSR, mxcsr_flags, mxcsr_round_flags,
};

/// Returns boundary and random encodings of a binary format in chunks of
/// `C` lanes. Consecutive chunks shift by one encoding, so each special
/// encoding reaches every lane.
fn chunks<const C: usize>(random: &mut SplitMix64, layout: Layout) -> Vec<[u64; C]> {
    let mut encodings: Vec<u64> = boundary_encodings_u128(layout)
        .into_iter()
        .map(|bits| u64::try_from(bits).expect("the encoding has at most 64 bits"))
        .collect();
    encodings.extend((0..4_000).map(|_| random.next_u64() >> (64 - layout.width)));
    (0..encodings.len())
        .map(|start| core::array::from_fn(|lane| encodings[(start + lane) % encodings.len()]))
        .collect()
}

/// Returns chunks of four binary32 encodings.
fn single_chunks(random: &mut SplitMix64) -> Vec<[u32; 4]> {
    chunks::<4>(random, Layout::BINARY32)
        .into_iter()
        .map(|chunk| chunk.map(|bits| u32::try_from(bits).expect("a binary32 encoding")))
        .collect()
}

/// Returns chunks of two binary64 encodings.
fn double_chunks(random: &mut SplitMix64) -> Vec<[u64; 2]> {
    chunks::<2>(random, Layout::BINARY64)
}

/// Compares a packed instruction of two operands with the `_with` method of
/// `Lanes` in every setting: the lanes, the union of the flags, and the MXCSR
/// flags of the lanes.
macro_rules! packed_binary {
    ($pairs:expr, $alias:ty, $lanes:literal, $instruction:path, $method:ident) => {
        for setting in x86::sse_settings() {
            let (control, env, daz) = (setting.control, setting.env(), setting.daz);
            for &(a, b) in $pairs {
                let (expected, expected_flags) = $instruction(a, b, control);
                let (x, y) = (
                    Lanes::<$alias, $lanes>::from_bits(a),
                    Lanes::<$alias, $lanes>::from_bits(b),
                );
                let (ours, union) = x.$method(y, env);
                let context = format!("{} {a:x?} {b:x?} {env:?}", stringify!($instruction));
                assert_eq!(ours.to_bits(), expected, "{context}: lanes");
                let (mut lane_union, mut lane_mxcsr) = (Flags::NONE, 0);
                for (&x, &y) in x.into_array().iter().zip(&y.into_array()) {
                    let (_, flags) = x.$method(y, env);
                    lane_union |= flags;
                    lane_mxcsr |= mxcsr_flags(flags, daz, x.is_nan() || y.is_nan());
                }
                assert_eq!(union, lane_union, "{context}: union");
                assert_eq!(lane_mxcsr, expected_flags, "{context}: flags");
            }
        }
    };
}

/// Compares a packed instruction of one operand with a `_with` method of
/// `Lanes` in every setting. `$flags` maps the flags of one lane, its
/// operand, and DAZ to MXCSR flags.
macro_rules! packed_unary {
    ($values:expr, $alias:ty, $lanes:literal, $instruction:path, $method:ident, $flags:expr) => {
        for setting in x86::sse_settings() {
            let (control, env, daz) = (setting.control, setting.env(), setting.daz);
            for &a in $values {
                let (expected, expected_flags) = $instruction(a, a, control);
                let x = Lanes::<$alias, $lanes>::from_bits(a);
                let (ours, union) = x.$method(env);
                let context = format!("{} {a:x?} {env:?}", stringify!($instruction));
                assert_eq!(ours.to_bits(), expected, "{context}: lanes");
                let (mut lane_union, mut lane_mxcsr) = (Flags::NONE, 0);
                for x in x.into_array() {
                    let (_, flags) = x.$method(env);
                    lane_union |= flags;
                    lane_mxcsr |= $flags(flags, x, daz);
                }
                assert_eq!(union, lane_union, "{context}: union");
                assert_eq!(lane_mxcsr, expected_flags, "{context}: flags");
            }
        }
    };
}

#[test]
fn packed_arithmetic_gives_the_lanes_and_the_union_of_their_flags() {
    let mut random = SplitMix64::new(0x00C5_9AC4);
    let singles = single_chunks(&mut random);
    let single_pairs: Vec<([u32; 4], [u32; 4])> = singles
        .iter()
        .zip(singles.iter().rev())
        .map(|(&a, &b)| (a, b))
        .collect();
    packed_binary!(&single_pairs, F32, 4, x86::addps, add_with);
    packed_binary!(&single_pairs, F32, 4, x86::subps, sub_with);
    packed_binary!(&single_pairs, F32, 4, x86::mulps, mul_with);
    packed_binary!(&single_pairs, F32, 4, x86::divps, div_with);
    packed_unary!(
        &singles,
        F32,
        4,
        x86::sqrtps,
        sqrt_with,
        |flags, x: F32, daz| { mxcsr_flags(flags, daz, x.is_nan()) }
    );
    let doubles = double_chunks(&mut random);
    let double_pairs: Vec<([u64; 2], [u64; 2])> = doubles
        .iter()
        .zip(doubles.iter().rev())
        .map(|(&a, &b)| (a, b))
        .collect();
    packed_binary!(&double_pairs, F64, 2, x86::addpd, add_with);
    packed_binary!(&double_pairs, F64, 2, x86::subpd, sub_with);
    packed_binary!(&double_pairs, F64, 2, x86::mulpd, mul_with);
    packed_binary!(&double_pairs, F64, 2, x86::divpd, div_with);
    packed_unary!(
        &doubles,
        F64,
        2,
        x86::sqrtpd,
        sqrt_with,
        |flags, x: F64, daz| { mxcsr_flags(flags, daz, x.is_nan()) }
    );
}

#[test]
fn packed_rounding_and_fused_multiply_add_give_the_lanes_and_their_flags() {
    assert!(
        std::arch::is_x86_feature_detected!("sse4.1") && std::arch::is_x86_feature_detected!("fma"),
        "the test needs a host with SSE4.1 and FMA"
    );
    let mut random = SplitMix64::new(0x00C5_F3A0);
    let singles = single_chunks(&mut random);
    packed_unary!(
        &singles,
        F32,
        4,
        x86::roundps,
        round_to_integral_with,
        |flags, _: F32, daz| { mxcsr_round_flags(flags, ROUND_USE_MXCSR, daz) }
    );
    let doubles = double_chunks(&mut random);
    packed_unary!(
        &doubles,
        F64,
        2,
        x86::roundpd,
        round_to_integral_with,
        |flags, _: F64, daz| { mxcsr_round_flags(flags, ROUND_USE_MXCSR, daz) }
    );
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for ((&a, &b), &c) in singles
            .iter()
            .zip(singles.iter().rev())
            .zip(singles.iter().skip(7))
        {
            let (expected, expected_flags) = x86::vfmadd213ps(a, b, c, control);
            let [x, y, z] = [a, b, c].map(Lanes::<F32, 4>::from_bits);
            let (ours, union) = x.mul_add_with(y, z, env);
            let context = format!("vfmadd213ps {a:x?} {b:x?} {c:x?} {env:?}");
            assert_eq!(ours.to_bits(), expected, "{context}: lanes");
            let (mut lane_union, mut lane_mxcsr) = (Flags::NONE, 0);
            for ((x, y), z) in x
                .into_array()
                .into_iter()
                .zip(y.into_array())
                .zip(z.into_array())
            {
                let (_, flags) = x.mul_add_with(y, z, env);
                lane_union |= flags;
                lane_mxcsr |= mxcsr_flags(flags, daz, x.is_nan() || y.is_nan() || z.is_nan());
            }
            assert_eq!(union, lane_union, "{context}: union");
            assert_eq!(lane_mxcsr, expected_flags, "{context}: flags");
        }
    }
}

#[test]
fn packed_conversions_give_the_lanes_and_their_flags() {
    let mut random = SplitMix64::new(0x00C5_C0F7);
    for setting in x86::sse_settings() {
        let (control, env, daz) = (setting.control, setting.env(), setting.daz);
        for chunk in single_chunks(&mut random).into_iter().step_by(3) {
            let (expected, expected_flags) = x86::cvtps2pd(chunk, chunk, control);
            let expected = [0, 1].map(|lane| {
                u64::from(expected[2 * lane]) | u64::from(expected[2 * lane + 1]) << 32
            });
            let x = Lanes::<F32, 2>::from_bits([chunk[0], chunk[1]]);
            let (ours, union): (Lanes<F64, 2>, _) = x.convert_with(env);
            let context = format!("cvtps2pd {chunk:x?} {env:?}");
            assert_eq!(ours.to_bits(), expected, "{context}: lanes");
            let lane_flags = x.into_array().map(|x| x.convert_with::<F64>(env).1);
            assert_eq!(union, lane_flags[0] | lane_flags[1], "{context}: union");
            let lane_mxcsr = lane_flags.map(|flags| mxcsr_flags(flags, daz, false));
            assert_eq!(
                lane_mxcsr[0] | lane_mxcsr[1],
                expected_flags,
                "{context}: flags"
            );
        }
        for chunk in double_chunks(&mut random).into_iter().step_by(3) {
            let halves = [chunk[0], chunk[0] >> 32, chunk[1], chunk[1] >> 32]
                .map(|half| u32::try_from(half & 0xFFFF_FFFF).expect("the mask keeps 32 bits"));
            let (expected, expected_flags) = x86::cvtpd2ps(halves, halves, control);
            let x = Lanes::<F64, 2>::from_bits(chunk);
            let (ours, union): (Lanes<F32, 2>, _) = x.convert_with(env);
            let context = format!("cvtpd2ps {chunk:x?} {env:?}");
            assert_eq!(
                ours.to_bits(),
                [expected[0], expected[1]],
                "{context}: lanes"
            );
            let lane_flags = x.into_array().map(|x| x.convert_with::<F32>(env).1);
            assert_eq!(union, lane_flags[0] | lane_flags[1], "{context}: union");
            let lane_mxcsr = lane_flags.map(|flags| mxcsr_flags(flags, daz, false));
            assert_eq!(
                lane_mxcsr[0] | lane_mxcsr[1],
                expected_flags,
                "{context}: flags"
            );
        }
    }
}

/// The results of the operations of `Lanes` on eight binary32 lanes and on
/// five binary64 lanes, which the packed host paths compute in chunks and
/// single lanes: the arithmetic of each format, and the conversions between
/// them.
#[derive(Debug, PartialEq, Eq)]
struct LaneResults {
    singles: LaneArithmetic<u32, 8>,
    doubles: LaneArithmetic<u64, 5>,
    /// The binary32 lanes converted to binary64.
    widened: [u64; 8],
    /// The binary64 lanes converted to binary32.
    narrowed: [u32; 5],
}

/// Returns the results of the operations of `Lanes` without flags on eight
/// binary32 lanes and on five binary64 lanes, with the first lanes of each
/// format as the addends.
fn lanes_without_flags(singles: [[u32; 8]; 2], doubles: [[u64; 5]; 2]) -> LaneResults {
    let [x, y] = singles.map(Lanes::<F32, 8>::from_bits);
    let [u, v] = doubles.map(Lanes::<F64, 5>::from_bits);
    let widened: Lanes<F64, 8> = x.convert();
    let narrowed: Lanes<F32, 5> = u.convert();
    LaneResults {
        singles: lane_arithmetic(x, y, x),
        doubles: lane_arithmetic(u, v, u),
        widened: widened.to_bits(),
        narrowed: narrowed.to_bits(),
    }
}

/// Returns the results of `lanes_without_flags` from the `_with` methods of
/// `Lanes` in the default mode, which always run the engine on each lane.
fn engine_results(singles: [[u32; 8]; 2], doubles: [[u64; 5]; 2]) -> LaneResults {
    let [x, y] = singles.map(Lanes::<F32, 8>::from_bits);
    let [u, v] = doubles.map(Lanes::<F64, 5>::from_bits);
    let widened: Lanes<F64, 8> = x.convert_with(F64::ENV).0;
    let narrowed: Lanes<F32, 5> = u.convert_with(F32::ENV).0;
    LaneResults {
        singles: lane_arithmetic_with(x, y, x),
        doubles: lane_arithmetic_with(u, v, u),
        widened: widened.to_bits(),
        narrowed: narrowed.to_bits(),
    }
}

#[test]
fn lanes_read_mxcsr_before_the_packed_unit() {
    // Under each control of `MXCSR_HOST_PATH_CONTROLS`, the operations of `Lanes`
    // without flags give the engine results of the default mode.
    let mut random = SplitMix64::new(0x00C5_4EAD);
    let singles = single_chunks(&mut random);
    let doubles = chunks::<5>(&mut random, Layout::BINARY64);
    for control in MXCSR_HOST_PATH_CONTROLS {
        for (index, pair) in singles.windows(2).enumerate().step_by(5) {
            let lanes: [[u32; 8]; 2] = [0, 1]
                .map(|first| core::array::from_fn(|lane| pair[(first + lane / 4) % 2][lane % 4]));
            let wide = [index, index + 3].map(|start| doubles[start % doubles.len()]);
            let ours = x86::with_mxcsr(control, || lanes_without_flags(black_box(lanes), wide));
            assert_eq!(
                ours,
                engine_results(lanes, wide),
                "{lanes:x?} {wide:x?} under {control:#x}"
            );
        }
    }
}

/// The results of the operations of `Lanes` on 13 binary16 lanes, which the
/// packed paths compute in binary32 chunks and single lanes: the arithmetic,
/// and the conversions of the lanes.
#[derive(Debug, PartialEq, Eq)]
struct HalfLaneResults {
    halves: LaneArithmetic<u16, 13>,
    /// 13 binary32 lanes converted to binary16.
    narrowed: [u16; 13],
    /// The binary16 lanes converted to binary32.
    widened: [u32; 13],
    /// The binary16 lanes converted to binary64.
    doubled: [u64; 13],
}

/// Returns the results of the operations of `Lanes` without flags on 13
/// binary16 lanes, with the first lanes as the addends, and of the
/// conversions to and from the lanes.
fn binary16_lanes_without_flags(halves: [[u16; 13]; 2], singles: [u32; 13]) -> HalfLaneResults {
    let [x, y] = halves.map(Lanes::<F16, 13>::from_bits);
    let narrowed: Lanes<F16, 13> = Lanes::<F32, 13>::from_bits(singles).convert();
    let (widened, doubled): (Lanes<F32, 13>, Lanes<F64, 13>) = (x.convert(), x.convert());
    HalfLaneResults {
        halves: lane_arithmetic(x, y, x),
        narrowed: narrowed.to_bits(),
        widened: widened.to_bits(),
        doubled: doubled.to_bits(),
    }
}

/// Returns the results of `binary16_lanes_without_flags` from the `_with`
/// methods of `Lanes` in the default mode, which always run the engine on
/// each lane.
fn binary16_engine_results(halves: [[u16; 13]; 2], singles: [u32; 13]) -> HalfLaneResults {
    let [x, y] = halves.map(Lanes::<F16, 13>::from_bits);
    let narrowed: Lanes<F16, 13> = Lanes::<F32, 13>::from_bits(singles)
        .convert_with(F16::ENV)
        .0;
    let widened: Lanes<F32, 13> = x.convert_with(F32::ENV).0;
    let doubled: Lanes<F64, 13> = x.convert_with(F64::ENV).0;
    HalfLaneResults {
        halves: lane_arithmetic_with(x, y, x),
        narrowed: narrowed.to_bits(),
        widened: widened.to_bits(),
        doubled: doubled.to_bits(),
    }
}

#[test]
fn binary16_lanes_read_mxcsr_before_the_packed_unit() {
    // Under each control, the binary16 operations of `Lanes` without flags
    // give the engine results of the default mode.
    let mut random = SplitMix64::new(0x00C5_4E16);
    let halves = chunks::<13>(&mut random, Layout::BINARY16);
    let singles = chunks::<13>(&mut random, Layout::BINARY32);
    for control in MXCSR_HOST_PATH_CONTROLS {
        for (index, pair) in halves.windows(2).enumerate().step_by(3) {
            let lanes: [[u16; 13]; 2] = [0, 1].map(|first| {
                pair[first].map(|bits| u16::try_from(bits).expect("a binary16 encoding"))
            });
            let narrow = singles[index % singles.len()]
                .map(|bits| u32::try_from(bits).expect("a binary32 encoding"));
            let ours = x86::with_mxcsr(control, || {
                binary16_lanes_without_flags(black_box(lanes), narrow)
            });
            assert_eq!(
                ours,
                binary16_engine_results(lanes, narrow),
                "{lanes:x?} {narrow:x?} under {control:#x}"
            );
        }
    }
}

#[test]
fn bfloat16_lanes_read_mxcsr_before_the_packed_unit() {
    // Under each control, the rounding and the conversions of bfloat16 lanes
    // give the engine results of the default mode.
    let mut random = SplitMix64::new(0x00C5_4EBF);
    let chunks = chunks::<13>(&mut random, Layout::BFLOAT16);
    for control in MXCSR_HOST_PATH_CONTROLS {
        let setting = format!("{control:#x}");
        for chunk in chunks.iter().step_by(3) {
            let bits = chunk.map(|bits| u16::try_from(bits).expect("a bfloat16 encoding"));
            assert_bfloat16_lanes_under(bits, &setting, x86::under_mxcsr(control));
        }
    }
}

#[test]
fn directed_rounding_reads_mxcsr_before_the_unit() {
    // The rounding to integral values in a directed mode takes the direction
    // from the immediate of `ROUNDPS` and `ROUNDSS`, but FTZ, DAZ, and the
    // exception masks of MXCSR still apply. Under each control, the lanes and
    // the scalar values give the engine results of the mode.
    let mut random = SplitMix64::new(0x00C5_D1E0);
    let chunks = chunks::<5>(&mut random, Layout::BINARY32);
    for control in MXCSR_HOST_PATH_CONTROLS {
        let setting = format!("{control:#x}");
        for chunk in chunks.iter().step_by(7) {
            let bits = chunk.map(|bits| u32::try_from(bits).expect("a binary32 encoding"));
            assert_directed_rounding_under(bits, &setting, x86::under_mxcsr(control));
        }
    }
}

/// Checks the comparison and the minimum and maximum operations of lanes of
/// one type under one MXCSR value against the engine results of the default
/// mode.
macro_rules! orders_under {
    ($alias:ty, $control:expr, $lanes:expr) => {
        let setting = format!("{:#x}", $control);
        for pair in $lanes.windows(2) {
            let lanes = [pair[0], pair[1]].map(Lanes::<$alias, 5>::from_bits);
            assert_lane_comparisons_under(lanes, &setting, x86::under_mxcsr($control));
        }
    };
}

#[test]
fn lane_orders_read_mxcsr_before_the_packed_unit() {
    // DAZ changes the order of a subnormal lane, and an unmasked invalid or
    // denormal exception traps. Under each control, the packed comparison and
    // the minimum and maximum operations give the engine results.
    let mut random = SplitMix64::new(0x00C5_0AD5);
    let singles: Vec<[u32; 5]> = chunks::<5>(&mut random, Layout::BINARY32)
        .into_iter()
        .map(|chunk| chunk.map(|bits| u32::try_from(bits).expect("a binary32 encoding")))
        .collect();
    let doubles = chunks::<5>(&mut random, Layout::BINARY64);
    for control in MXCSR_HOST_PATH_CONTROLS {
        orders_under!(
            F32,
            control,
            singles.iter().step_by(5).copied().collect::<Vec<_>>()
        );
        orders_under!(
            F64,
            control,
            doubles.iter().step_by(5).copied().collect::<Vec<_>>()
        );
    }
}

/// Checks the arithmetic, the comparisons, and the conversions of `Lanes`
/// of one format and lane count under every MXCSR value of
/// `MXCSR_HOST_PATH_CONTROLS`, on the lane sets of [`lane_sets`].
fn every_lane_group<S: Standard<W>, const W: usize, const N: usize>(layout: Layout, seed: u64)
where
    S::Bits: TryFrom<u128>,
{
    let mut random = SplitMix64::new(seed);
    let sets = lane_sets::<S, W, N>(layout, &mut random, 2_000);
    for control in MXCSR_HOST_PATH_CONTROLS {
        let setting = format!("MXCSR {control:#x}");
        for window in sets.windows(3).step_by(5) {
            let [x, y, z] = [window[0], window[1], window[2]];
            assert_lane_arithmetic_under([x, y, z], &setting, x86::under_mxcsr(control));
            assert_lane_comparisons_under([x, y], &setting, x86::under_mxcsr(control));
            assert_lane_conversions_under(x, &setting, x86::under_mxcsr(control));
        }
    }
}

#[test]
fn every_lane_type_reads_mxcsr_before_the_packed_unit() {
    // The lane counts fill the wide chunks, then the narrow chunks, and leave
    // single lanes. The binary32 lanes also convert to bfloat16, and every
    // lane type converts to `i32`. x87 extended lanes read the x87 control
    // word, which x87-hardware checks, and MXCSR must not change them.
    every_lane_group::<Binary<8>, 32, 13>(Layout::BINARY32, 0x00C5_1A01);
    every_lane_group::<Binary<11>, 64, 7>(Layout::BINARY64, 0x00C5_1A02);
    every_lane_group::<Binary<5>, 16, 13>(Layout::BINARY16, 0x00C5_1A03);
    every_lane_group::<Binary<8>, 16, 13>(Layout::BFLOAT16, 0x00C5_1A04);
    every_lane_group::<Binary<15, X87>, 80, 3>(Layout::X87_EXTENDED, 0x00C5_1A05);
}
