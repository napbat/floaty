//! Compares `Lanes` with the packed instructions of the SSE unit of the host
//! processor. Each lane gives the bits of the packed instruction, and the
//! flags of the instruction are the union of the flags of the lanes. The
//! operations of `Lanes` without flags also give the engine results under
//! every MXCSR setting. The test runs only on x86-64 hosts.

#![cfg(target_arch = "x86_64")]

use std::hint::black_box;

use floaty::{Env, F32, F64, Flags, Lanes};
use floaty_verify::encodings::{IntegerBit, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, MXCSR_DAZ, MXCSR_FTZ, MXCSR_MASKED, MXCSR_ROUNDINGS, mxcsr_flags, sse_env,
};

/// Every MXCSR setting of the test: each rounding direction with FTZ and DAZ
/// on and off. Returns the control value, the matching behavior, and DAZ.
fn settings() -> Vec<(u32, Env, bool)> {
    let mut settings = Vec::new();
    for (rounding, field) in MXCSR_ROUNDINGS {
        for (ftz, daz) in [(false, false), (true, false), (false, true), (true, true)] {
            let control = MXCSR_MASKED
                | field
                | if ftz { MXCSR_FTZ } else { 0 }
                | if daz { MXCSR_DAZ } else { 0 };
            settings.push((control, sse_env(rounding, ftz, daz), daz));
        }
    }
    settings
}

/// Returns boundary and random encodings of a binary format in chunks of
/// `C` lanes. Consecutive chunks shift by one encoding, so each special
/// encoding reaches every lane.
fn chunks<const C: usize>(
    random: &mut SplitMix64,
    width: u32,
    exponent_bits: u32,
) -> Vec<[u64; C]> {
    let mut encodings: Vec<u64> =
        boundary_encodings_u128(width, exponent_bits, IntegerBit::Implicit)
            .into_iter()
            .map(|bits| u64::try_from(bits).expect("the encoding has at most 64 bits"))
            .collect();
    encodings.extend((0..4_000).map(|_| random.next_u64() >> (64 - width)));
    (0..encodings.len())
        .map(|start| core::array::from_fn(|lane| encodings[(start + lane) % encodings.len()]))
        .collect()
}

/// Returns chunks of four binary32 encodings.
fn single_chunks(random: &mut SplitMix64) -> Vec<[u32; 4]> {
    chunks::<4>(random, 32, 8)
        .into_iter()
        .map(|chunk| chunk.map(|bits| u32::try_from(bits).expect("a binary32 encoding")))
        .collect()
}

/// Returns chunks of two binary64 encodings.
fn double_chunks(random: &mut SplitMix64) -> Vec<[u64; 2]> {
    chunks::<2>(random, 64, 11)
}

/// Compares a packed instruction of two operands with the `_with` method of
/// `Lanes` in every setting: the lanes, the union of the flags, and the MXCSR
/// flags of the lanes.
macro_rules! packed_binary {
    ($pairs:expr, $alias:ty, $lanes:literal, $instruction:path, $method:ident) => {
        for (control, env, daz) in settings() {
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
        for (control, env, daz) in settings() {
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

/// Returns the MXCSR flags of a rounding lane: `ROUNDPS` and `ROUNDPD` signal
/// only IE and PE, as their scalar forms do.
fn round_flags(flags: Flags, daz: bool) -> u32 {
    mxcsr_flags(flags.difference(Flags::DENORMAL_INPUT), daz, false)
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
        |flags, _: F32, daz| { round_flags(flags, daz) }
    );
    let doubles = double_chunks(&mut random);
    packed_unary!(
        &doubles,
        F64,
        2,
        x86::roundpd,
        round_to_integral_with,
        |flags, _: F64, daz| { round_flags(flags, daz) }
    );
    for (control, env, daz) in settings() {
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
    for (control, env, daz) in settings() {
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

/// Returns the results of the operations of `Lanes` without flags on eight
/// binary32 lanes and on five binary64 lanes, which the packed host paths
/// compute in chunks and single lanes.
fn lanes_without_flags(singles: [[u32; 8]; 2], doubles: [[u64; 5]; 2]) -> Vec<u64> {
    let [x, y] = singles.map(Lanes::<F32, 8>::from_bits);
    let [u, v] = doubles.map(Lanes::<F64, 5>::from_bits);
    let singles = [
        x + y,
        x - y,
        x * y,
        x / y,
        x.sqrt(),
        x.mul_add(y, x),
        x.round_to_integral(),
    ];
    let doubles = [
        u + v,
        u - v,
        u * v,
        u / v,
        u.sqrt(),
        u.mul_add(v, u),
        u.round_to_integral(),
    ];
    let widened: Lanes<F64, 8> = x.convert();
    let narrowed: Lanes<F32, 5> = u.convert();
    let mut bits: Vec<u64> = singles
        .iter()
        .flat_map(|lanes| lanes.to_bits().map(u64::from))
        .collect();
    bits.extend(doubles.iter().flat_map(|lanes| lanes.to_bits()));
    bits.extend(widened.to_bits());
    bits.extend(narrowed.to_bits().map(u64::from));
    bits
}

/// Returns the results of `lanes_without_flags` from the `_with` methods of
/// each lane in the default mode, which always run the engine.
fn engine_results(singles: [[u32; 8]; 2], doubles: [[u64; 5]; 2]) -> Vec<u64> {
    let [x, y] = singles.map(|lanes| lanes.map(F32::from_bits));
    let [u, v] = doubles.map(|lanes| lanes.map(F64::from_bits));
    let mut bits = Vec::new();
    let single_operations: [fn(F32, F32) -> F32; 7] = [
        |x, y| x.add_with(y, F32::ENV).0,
        |x, y| x.sub_with(y, F32::ENV).0,
        |x, y| x.mul_with(y, F32::ENV).0,
        |x, y| x.div_with(y, F32::ENV).0,
        |x, _| x.sqrt_with(F32::ENV).0,
        |x, y| x.mul_add_with(y, x, F32::ENV).0,
        |x, _| x.round_to_integral_with(F32::ENV).0,
    ];
    for operation in single_operations {
        bits.extend(
            x.iter()
                .zip(&y)
                .map(|(&x, &y)| u64::from(operation(x, y).to_bits())),
        );
    }
    let double_operations: [fn(F64, F64) -> F64; 7] = [
        |x, y| x.add_with(y, F64::ENV).0,
        |x, y| x.sub_with(y, F64::ENV).0,
        |x, y| x.mul_with(y, F64::ENV).0,
        |x, y| x.div_with(y, F64::ENV).0,
        |x, _| x.sqrt_with(F64::ENV).0,
        |x, y| x.mul_add_with(y, x, F64::ENV).0,
        |x, _| x.round_to_integral_with(F64::ENV).0,
    ];
    for operation in double_operations {
        bits.extend(u.iter().zip(&v).map(|(&x, &y)| operation(x, y).to_bits()));
    }
    bits.extend(
        x.iter()
            .map(|&x| x.convert_with::<F64>(F64::ENV).0.to_bits()),
    );
    bits.extend(
        u.iter()
            .map(|&x| u64::from(x.convert_with::<F32>(F32::ENV).0.to_bits())),
    );
    bits
}

#[test]
fn lanes_read_mxcsr_before_the_packed_unit() {
    // FTZ, DAZ, and each directed rounding change the packed results, and an
    // unmasked exception traps. Under each, the operations of `Lanes` without
    // flags still give the engine results of the default mode. The masks IM,
    // DM, ZM, OM, UM, and PM are bits 7 to 12.
    let directions = [
        MXCSR_MASKED,
        MXCSR_MASKED | MXCSR_FTZ,
        MXCSR_MASKED | MXCSR_DAZ,
        MXCSR_MASKED | (1 << 13),
        MXCSR_MASKED | (2 << 13),
        MXCSR_MASKED | (3 << 13),
    ];
    let unmasked = (7..=12).map(|mask| MXCSR_MASKED & !(1 << mask));
    let controls: Vec<u32> = directions.into_iter().chain(unmasked).collect();
    let mut random = SplitMix64::new(0x00C5_4EAD);
    let singles = single_chunks(&mut random);
    let doubles = chunks::<5>(&mut random, 64, 11);
    for control in controls {
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
