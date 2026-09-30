//! Compares `Lanes` with the packed instructions of the SSE unit of the host
//! processor. Each lane gives the bits of the packed instruction, and the
//! flags of the instruction are the union of the flags of the lanes. The
//! operations of `Lanes` without flags also give the engine results under
//! every MXCSR setting. The test runs only on x86-64 hosts.

#![cfg(target_arch = "x86_64")]

use std::hint::black_box;

use floaty::{BF16, F16, F32, F64, Flags, Lanes};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;
use floaty_verify::x86::{
    self, MXCSR_DAZ, MXCSR_EXCEPTION_MASKS, MXCSR_FTZ, MXCSR_MASKED, MXCSR_TOWARD_NEGATIVE,
    MXCSR_TOWARD_POSITIVE, MXCSR_TOWARD_ZERO, ROUND_USE_MXCSR, mxcsr_flags, mxcsr_round_flags,
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

/// Returns the MXCSR values under which the paths of `Lanes` must give the
/// engine results: the default, FTZ, DAZ, each directed rounding, and each
/// unmasked exception. FTZ, DAZ, and each directed rounding change the packed
/// results, and an unmasked exception traps.
fn controls() -> Vec<u32> {
    let directions = [
        MXCSR_MASKED,
        MXCSR_MASKED | MXCSR_FTZ,
        MXCSR_MASKED | MXCSR_DAZ,
        MXCSR_MASKED | MXCSR_TOWARD_NEGATIVE,
        MXCSR_MASKED | MXCSR_TOWARD_POSITIVE,
        MXCSR_MASKED | MXCSR_TOWARD_ZERO,
    ];
    let unmasked = MXCSR_EXCEPTION_MASKS.map(|mask| MXCSR_MASKED & !mask);
    directions.into_iter().chain(unmasked).collect()
}

#[test]
fn lanes_read_mxcsr_before_the_packed_unit() {
    // Under each control, the operations of `Lanes` without flags give the
    // engine results of the default mode.
    let controls = controls();
    let mut random = SplitMix64::new(0x00C5_4EAD);
    let singles = single_chunks(&mut random);
    let doubles = chunks::<5>(&mut random, Layout::BINARY64);
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

/// Returns the results of the operations of `Lanes` without flags on 13
/// binary16 lanes, which the packed paths compute in binary32 chunks and
/// single lanes, and of the conversions of the lanes.
fn binary16_lanes_without_flags(halves: [[u16; 13]; 2], singles: [u32; 13]) -> Vec<u64> {
    let [x, y] = halves.map(Lanes::<F16, 13>::from_bits);
    let lanes = [
        x + y,
        x - y,
        x * y,
        x / y,
        x.sqrt(),
        x.round_to_integral(),
        Lanes::<F32, 13>::from_bits(singles).convert(),
    ];
    let (widened, doubled): (Lanes<F32, 13>, Lanes<F64, 13>) = (x.convert(), x.convert());
    let mut bits: Vec<u64> = lanes
        .iter()
        .flat_map(|lanes| lanes.to_bits().map(u64::from))
        .collect();
    bits.extend(widened.to_bits().map(u64::from));
    bits.extend(doubled.to_bits());
    bits
}

/// Returns the results of `binary16_lanes_without_flags` from the `_with`
/// methods of each lane in the default mode, which always run the engine.
fn binary16_engine_results(halves: [[u16; 13]; 2], singles: [u32; 13]) -> Vec<u64> {
    let [x, y] = halves.map(|lanes| lanes.map(F16::from_bits));
    let operations: [fn(F16, F16) -> F16; 6] = [
        |x, y| x.add_with(y, F16::ENV).0,
        |x, y| x.sub_with(y, F16::ENV).0,
        |x, y| x.mul_with(y, F16::ENV).0,
        |x, y| x.div_with(y, F16::ENV).0,
        |x, _| x.sqrt_with(F16::ENV).0,
        |x, _| x.round_to_integral_with(F16::ENV).0,
    ];
    let mut bits = Vec::new();
    for operation in operations {
        bits.extend(
            x.iter()
                .zip(&y)
                .map(|(&x, &y)| u64::from(operation(x, y).to_bits())),
        );
    }
    bits.extend(singles.iter().map(|&bits| {
        u64::from(
            F32::from_bits(bits)
                .convert_with::<F16>(F16::ENV)
                .0
                .to_bits(),
        )
    }));
    bits.extend(
        x.iter()
            .map(|&x| u64::from(x.convert_with::<F32>(F32::ENV).0.to_bits())),
    );
    bits.extend(
        x.iter()
            .map(|&x| x.convert_with::<F64>(F64::ENV).0.to_bits()),
    );
    bits
}

#[test]
fn binary16_lanes_read_mxcsr_before_the_packed_unit() {
    // Under each control, the binary16 operations of `Lanes` without flags
    // give the engine results of the default mode.
    let mut random = SplitMix64::new(0x00C5_4E16);
    let halves = chunks::<13>(&mut random, Layout::BINARY16);
    let singles = chunks::<13>(&mut random, Layout::BINARY32);
    for control in controls() {
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
    for control in controls() {
        for chunk in chunks.iter().step_by(3) {
            let bits = chunk.map(|bits| u16::try_from(bits).expect("a bfloat16 encoding"));
            let lanes = Lanes::<BF16, 13>::from_bits(bits);
            let ours = x86::with_mxcsr(control, || {
                let lanes = black_box(lanes);
                let single: Lanes<F32, 13> = lanes.convert();
                let double: Lanes<F64, 13> = lanes.convert();
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
            assert_eq!(ours, engine, "{bits:x?} under {control:#x}");
        }
    }
}

#[test]
fn directed_rounding_reads_mxcsr_before_the_unit() {
    // The rounding to integral values in a directed mode takes the direction
    // from the immediate of `ROUNDPS` and `ROUNDSS`, but FTZ, DAZ, and the
    // exception masks of MXCSR still apply. Under each control, the lanes and
    // the scalar values give the engine results of the mode.
    type Up = floaty::Float<
        floaty::Binary<8>,
        32,
        floaty::mode::Rounded<floaty::mode::Ieee, floaty::mode::direction::TowardPositive>,
    >;
    let mut random = SplitMix64::new(0x00C5_D1E0);
    let chunks = chunks::<5>(&mut random, Layout::BINARY32);
    for control in controls() {
        for chunk in chunks.iter().step_by(7) {
            let bits = chunk.map(|bits| u32::try_from(bits).expect("a binary32 encoding"));
            let lanes = Lanes::<Up, 5>::from_bits(bits);
            let ours = x86::with_mxcsr(control, || {
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
            assert_eq!(ours, (engine, engine), "{bits:x?} under {control:#x}");
        }
    }
}

/// Checks the comparison and the minimum and maximum operations of lanes of
/// one type under one MXCSR value against the scalar results of the engine.
macro_rules! orders_under {
    ($alias:ty, $control:expr, $lanes:expr) => {
        for pair in $lanes.windows(2) {
            let (x, y) = (
                Lanes::<$alias, 5>::from_bits(pair[0]),
                Lanes::<$alias, 5>::from_bits(pair[1]),
            );
            let ours = x86::with_mxcsr($control, || {
                let (x, y) = (black_box(x), black_box(y));
                (
                    x.compare_quiet(y),
                    [
                        x.minimum(y).to_bits(),
                        x.maximum(y).to_bits(),
                        x.minimum_number(y).to_bits(),
                        x.maximum_number(y).to_bits(),
                        x.min_num(y).to_bits(),
                        x.max_num(y).to_bits(),
                    ],
                )
            });
            let (a, b) = (x.into_array(), y.into_array());
            let env = <$alias>::ENV;
            let each = |operation: &dyn Fn($alias, $alias) -> $alias| {
                core::array::from_fn::<_, 5, _>(|lane| operation(a[lane], b[lane]).to_bits())
            };
            let engine = (
                core::array::from_fn::<_, 5, _>(|lane| a[lane].compare_quiet_with(b[lane], env).0),
                [
                    each(&|x, y| x.minimum_with(y, env).0),
                    each(&|x, y| x.maximum_with(y, env).0),
                    each(&|x, y| x.minimum_number_with(y, env).0),
                    each(&|x, y| x.maximum_number_with(y, env).0),
                    each(&|x, y| x.min_num_with(y, env).0),
                    each(&|x, y| x.max_num_with(y, env).0),
                ],
            );
            assert_eq!(ours, engine, "{:x?} under {:#x}", pair, $control);
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
    for control in controls() {
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
