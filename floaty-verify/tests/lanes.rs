//! Checks that each lane of `Lanes` gives the bits of the scalar operation,
//! and that a `_with` method returns the union of the flags of the lanes, on
//! every target. The scalar operations pass the oracle tests.
//!
//! The lane counts take every path of the packed host paths: wide chunks,
//! narrow chunks, and single lanes. binary32, binary64, binary16, and the
//! rounding of bfloat16 take the packed paths where the build has them; the
//! other formats and operations take the scalar operation of each lane.

use floaty::env::Rounding;
use floaty::mode::{Rounded, X86Sse, direction::TowardZero};
use floaty::{BF16, Env, F16, F32, F64, F80, F128, Flags, Float, Lanes};
use floaty_verify::encodings::{IntegerBit, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;

/// Returns boundary and random encodings of a layout, NaNs among them.
fn encodings(random: &mut SplitMix64, width: u32, exponent_bits: u32) -> Vec<u128> {
    let integer_bit = if width == 80 {
        IntegerBit::Explicit
    } else {
        IntegerBit::Implicit
    };
    let mut encodings = boundary_encodings_u128(width, exponent_bits, integer_bit);
    encodings.extend((0..600).map(|_| random.next_u128() >> (128 - width)));
    encodings
}

/// Returns the lanes that start at `start` in a cycle of `encodings`. Each
/// encoding reaches every lane as `start` moves.
fn window<T: TryFrom<u128, Error: core::fmt::Debug>, const N: usize>(
    encodings: &[u128],
    start: usize,
) -> [T; N] {
    core::array::from_fn(|lane| {
        T::try_from(encodings[(start + lane) % encodings.len()])
            .expect("the encoding fits the storage")
    })
}

/// Checks the operations of `Lanes` of one type and lane count against the
/// scalar operations of each lane: the methods without flags against the
/// `_with` methods in the default mode, and each `_with` method under
/// `behaviors` with the union of the flags.
macro_rules! lanes_match {
    ($alias:ty, $bits:ty, $lanes:literal, $encodings:expr, $behaviors:expr) => {{
        type Value = $alias;
        let encodings: &[u128] = $encodings;
        for start in 0..encodings.len() {
            let a = Lanes::<Value, $lanes>::from_bits(window::<$bits, $lanes>(encodings, start));
            let b = Lanes::<Value, $lanes>::from_bits(window::<$bits, $lanes>(encodings, start * 7 + 3));
            let c = Lanes::<Value, $lanes>::from_bits(window::<$bits, $lanes>(encodings, start * 13 + 5));
            let (x, y, z) = (a.into_array(), b.into_array(), c.into_array());
            let context = format!("{} x {} at {start}", stringify!($alias), $lanes);
            let default = <Value as floaty::FloatType>::Mode::default();
            let each = |operation: &dyn Fn(Value, Value, Value) -> Value| -> Vec<$bits> {
                x.iter().zip(&y).zip(&z).map(|((&x, &y), &z)| operation(x, y, z).to_bits()).collect()
            };
            assert_eq!((a + b).to_bits().to_vec(), each(&|x, y, _| x.add_with(y, default).0), "{context}: +");
            assert_eq!((a - b).to_bits().to_vec(), each(&|x, y, _| x.sub_with(y, default).0), "{context}: -");
            assert_eq!((a * b).to_bits().to_vec(), each(&|x, y, _| x.mul_with(y, default).0), "{context}: *");
            assert_eq!((a / b).to_bits().to_vec(), each(&|x, y, _| x.div_with(y, default).0), "{context}: /");
            assert_eq!(a.sqrt().to_bits().to_vec(), each(&|x, _, _| x.sqrt_with(default).0), "{context}: sqrt");
            assert_eq!(
                a.mul_add(b, c).to_bits().to_vec(),
                each(&|x, y, z| x.mul_add_with(y, z, default).0),
                "{context}: mul_add"
            );
            assert_eq!(
                a.round_to_integral().to_bits().to_vec(),
                each(&|x, _, _| x.round_to_integral_with(default).0),
                "{context}: round_to_integral"
            );
            for behavior in $behaviors {
                let union = |operation: &dyn Fn(Value, Value, Value) -> (Value, Flags)| -> (Vec<$bits>, Flags) {
                    let mut flags = Flags::NONE;
                    let bits = x.iter().zip(&y).zip(&z).map(|((&x, &y), &z)| {
                        let (value, lane_flags) = operation(x, y, z);
                        flags |= lane_flags;
                        value.to_bits()
                    }).collect();
                    (bits, flags)
                };
                let lanes = |(value, flags): (Lanes<Value, $lanes>, Flags)| (value.to_bits().to_vec(), flags);
                let context = format!("{context} {behavior:?}");
                assert_eq!(lanes(a.add_with(b, behavior)), union(&|x, y, _| x.add_with(y, behavior)), "{context}: add");
                assert_eq!(lanes(a.sub_with(b, behavior)), union(&|x, y, _| x.sub_with(y, behavior)), "{context}: sub");
                assert_eq!(lanes(a.mul_with(b, behavior)), union(&|x, y, _| x.mul_with(y, behavior)), "{context}: mul");
                assert_eq!(lanes(a.div_with(b, behavior)), union(&|x, y, _| x.div_with(y, behavior)), "{context}: div");
                assert_eq!(lanes(a.sqrt_with(behavior)), union(&|x, _, _| x.sqrt_with(behavior)), "{context}: sqrt");
                assert_eq!(
                    lanes(a.mul_add_with(b, c, behavior)),
                    union(&|x, y, z| x.mul_add_with(y, z, behavior)),
                    "{context}: mul_add"
                );
                assert_eq!(
                    lanes(a.round_to_integral_with(behavior)),
                    union(&|x, _, _| x.round_to_integral_with(behavior)),
                    "{context}: round_to_integral"
                );
            }
        }
    }};
}

/// The behaviors of the `_with` checks: the default mode, the x86 SSE
/// preset, a directed rounding, and flush-to-zero with denormals-are-zero.
fn behaviors() -> [Env; 4] {
    [
        Env::IEEE,
        Env::X86_SSE,
        Env::IEEE.with_rounding(Rounding::TowardZero),
        Env::IEEE
            .with_flush_to_zero(true)
            .with_denormals_are_zero(true),
    ]
}

#[test]
fn binary32_lanes_give_the_scalar_results() {
    let mut random = SplitMix64::new(0x1A4E_0032);
    let encodings = encodings(&mut random, 32, 8);
    lanes_match!(F32, u32, 1, &encodings, behaviors());
    lanes_match!(F32, u32, 3, &encodings, behaviors());
    lanes_match!(F32, u32, 4, &encodings, behaviors());
    lanes_match!(F32, u32, 5, &encodings, behaviors());
    lanes_match!(F32, u32, 8, &encodings, behaviors());
    lanes_match!(F32, u32, 13, &encodings, behaviors());
    lanes_match!(F32, u32, 16, &encodings, behaviors());
}

#[test]
fn binary64_lanes_give_the_scalar_results() {
    let mut random = SplitMix64::new(0x1A4E_0064);
    let encodings = encodings(&mut random, 64, 11);
    lanes_match!(F64, u64, 1, &encodings, behaviors());
    lanes_match!(F64, u64, 2, &encodings, behaviors());
    lanes_match!(F64, u64, 3, &encodings, behaviors());
    lanes_match!(F64, u64, 4, &encodings, behaviors());
    lanes_match!(F64, u64, 7, &encodings, behaviors());
    lanes_match!(F64, u64, 8, &encodings, behaviors());
}

#[test]
fn binary16_lanes_give_the_scalar_results() {
    let mut random = SplitMix64::new(0x1A4E_0016);
    let encodings = encodings(&mut random, 16, 5);
    lanes_match!(F16, u16, 1, &encodings, behaviors());
    lanes_match!(F16, u16, 3, &encodings, behaviors());
    lanes_match!(F16, u16, 4, &encodings, behaviors());
    lanes_match!(F16, u16, 5, &encodings, behaviors());
    lanes_match!(F16, u16, 8, &encodings, behaviors());
    lanes_match!(F16, u16, 13, &encodings, behaviors());
    lanes_match!(F16, u16, 16, &encodings, behaviors());
}

#[test]
fn bfloat16_lanes_give_the_scalar_results() {
    let mut random = SplitMix64::new(0x1A4E_00BF);
    let encodings = encodings(&mut random, 16, 8);
    lanes_match!(BF16, u16, 1, &encodings, behaviors());
    lanes_match!(BF16, u16, 3, &encodings, behaviors());
    lanes_match!(BF16, u16, 4, &encodings, behaviors());
    lanes_match!(BF16, u16, 5, &encodings, behaviors());
    lanes_match!(BF16, u16, 8, &encodings, behaviors());
    lanes_match!(BF16, u16, 13, &encodings, behaviors());
    lanes_match!(BF16, u16, 16, &encodings, behaviors());
}

#[test]
fn lanes_of_the_other_formats_give_the_scalar_results() {
    let mut random = SplitMix64::new(0x1A4E_0000);
    lanes_match!(F80, u128, 2, &encodings(&mut random, 80, 15), behaviors());
    lanes_match!(F128, u128, 3, &encodings(&mut random, 128, 15), behaviors());
}

#[test]
fn lanes_of_other_modes_give_the_scalar_results() {
    // The SSE mode takes the packed paths and selects NaNs by its own rule.
    // A directed mode never takes them.
    type Sse = Float<floaty::Binary<8>, 32, X86Sse>;
    type Truncating = Float<floaty::Binary<11>, 64, Rounded<floaty::mode::Ieee, TowardZero>>;
    let mut random = SplitMix64::new(0x1A4E_00DE);
    lanes_match!(Sse, u32, 8, &encodings(&mut random, 32, 8), [Env::X86_SSE]);
    lanes_match!(
        Truncating,
        u64,
        4,
        &encodings(&mut random, 64, 11),
        [Env::IEEE]
    );
}

#[test]
fn lanes_round_in_the_direction_of_their_mode() {
    // `round_to_integral` takes the direction of the mode in the packed
    // paths. Each direction and format below checks every operation of
    // `lanes_match!`: the arithmetic of a directed mode runs in the engine,
    // and the rounding to integral values takes the paths.
    use floaty::mode::direction::{NearestAway, ToOdd, TowardNegative, TowardPositive};
    use floaty::mode::{Ieee, Rounded};
    type Up = Float<floaty::Binary<8>, 32, Rounded<Ieee, TowardPositive>>;
    type Down = Float<floaty::Binary<5>, 16, Rounded<Ieee, TowardNegative>>;
    type Away = Float<floaty::Binary<11>, 64, Rounded<Ieee, NearestAway>>;
    type Chopped = Float<floaty::Binary<8>, 16, Rounded<Ieee, TowardZero>>;
    type Odd = Float<floaty::Binary<8>, 32, Rounded<Ieee, ToOdd>>;
    let mut random = SplitMix64::new(0x1A4E_D1E0);
    // Values an exact half above an integer, of both signs, are the ties of
    // the rounding to integral values.
    let ties: Vec<F64> = (-200_i64..200)
        .map(|integer| {
            let half = F64::from_bits(0x3FE0_0000_0000_0000);
            F64::from_int(integer).add_with(half, Env::IEEE).0
        })
        .collect();
    let with_ties = |mut encodings: Vec<u128>, tie: fn(F64) -> u128| {
        encodings.extend(ties.iter().map(|&value| tie(value)));
        encodings
    };
    let singles = with_ties(encodings(&mut random, 32, 8), |value| {
        u128::from(value.convert::<F32>().to_bits())
    });
    let halves = with_ties(encodings(&mut random, 16, 5), |value| {
        u128::from(value.convert::<F16>().to_bits())
    });
    let doubles = with_ties(encodings(&mut random, 64, 11), |value| {
        u128::from(value.to_bits())
    });
    let bfloats = with_ties(encodings(&mut random, 16, 8), |value| {
        u128::from(value.convert::<BF16>().to_bits())
    });
    lanes_match!(Up, u32, 13, &singles, [Env::IEEE]);
    lanes_match!(Odd, u32, 5, &singles, [Env::IEEE]);
    lanes_match!(Down, u16, 13, &halves, [Env::IEEE]);
    lanes_match!(Away, u64, 7, &doubles, [Env::IEEE]);
    lanes_match!(Chopped, u16, 13, &bfloats, [Env::IEEE]);
}

#[test]
fn lanes_convert_as_their_lanes_do() {
    let mut random = SplitMix64::new(0x1A4E_C0F7);
    let singles = encodings(&mut random, 32, 8);
    let doubles = encodings(&mut random, 64, 11);
    let binary16 = encodings(&mut random, 16, 5);
    let bfloat16 = encodings(&mut random, 16, 8);
    for start in 0..singles.len().max(doubles.len()) {
        let x = Lanes::<F32, 9>::from_bits(window::<u32, 9>(&singles, start));
        let widened: Lanes<F64, 9> = x.convert();
        let halved: Lanes<F16, 9> = x.convert();
        let each_double = x
            .into_array()
            .map(|x| x.convert_with::<F64>(F64::ENV).0.to_bits());
        let each_half = x
            .into_array()
            .map(|x| x.convert_with::<F16>(F16::ENV).0.to_bits());
        assert_eq!(
            widened.to_bits(),
            each_double,
            "binary32 at {start} to binary64"
        );
        assert_eq!(
            halved.to_bits(),
            each_half,
            "binary32 at {start} to binary16"
        );
        let h = Lanes::<F16, 13>::from_bits(window::<u16, 13>(&binary16, start));
        let (h_single, h_double): (Lanes<F32, 13>, Lanes<F64, 13>) = (h.convert(), h.convert());
        assert_eq!(
            h_single.to_bits(),
            h.into_array()
                .map(|h| h.convert_with::<F32>(F32::ENV).0.to_bits()),
            "binary16 at {start} to binary32"
        );
        assert_eq!(
            h_double.to_bits(),
            h.into_array()
                .map(|h| h.convert_with::<F64>(F64::ENV).0.to_bits()),
            "binary16 at {start} to binary64"
        );
        let b = Lanes::<BF16, 13>::from_bits(window::<u16, 13>(&bfloat16, start));
        let (b_single, b_double): (Lanes<F32, 13>, Lanes<F64, 13>) = (b.convert(), b.convert());
        assert_eq!(
            b_single.to_bits(),
            b.into_array()
                .map(|b| b.convert_with::<F32>(F32::ENV).0.to_bits()),
            "bfloat16 at {start} to binary32"
        );
        assert_eq!(
            b_double.to_bits(),
            b.into_array()
                .map(|b| b.convert_with::<F64>(F64::ENV).0.to_bits()),
            "bfloat16 at {start} to binary64"
        );
        let u = Lanes::<F64, 7>::from_bits(window::<u64, 7>(&doubles, start));
        let narrowed: Lanes<F32, 7> = u.convert();
        let each_single = u
            .into_array()
            .map(|u| u.convert_with::<F32>(F32::ENV).0.to_bits());
        assert_eq!(
            narrowed.to_bits(),
            each_single,
            "binary64 at {start} to binary32"
        );
        let env = Env::X86_SSE.with_rounding(Rounding::TowardPositive);
        let (converted, union) = u.convert_with::<F32>(env);
        let lanes = u.into_array().map(|u| u.convert_with::<F32>(env));
        assert_eq!(
            converted.to_bits(),
            lanes.map(|(value, _)| value.to_bits()),
            "binary64 at {start} with flags"
        );
        let flags = lanes
            .iter()
            .fold(Flags::NONE, |flags, &(_, lane)| flags | lane);
        assert_eq!(union, flags, "binary64 at {start}: union");
    }
}

/// Checks the other methods of `Lanes` of one type and lane count against
/// the scalar methods of each lane: the sign operations, the predicates,
/// the integer conversions, the steps, `scale_b`, the remainder, the
/// comparisons, the total order, and the minimum and maximum operations. A
/// `_with` method must return the union of the flags of the lanes.
macro_rules! methods_match {
    ($alias:ty, $bits:ty, $lanes:literal, $encodings:expr, $behavior:expr) => {{
        type Value = $alias;
        let encodings: &[u128] = $encodings;
        let behavior = $behavior;
        for start in 0..encodings.len() {
            let a = Lanes::<Value, $lanes>::from_bits(window::<$bits, $lanes>(encodings, start));
            let b = Lanes::<Value, $lanes>::from_bits(window::<$bits, $lanes>(
                encodings,
                start * 7 + 3,
            ));
            let (x, y) = (a.into_array(), b.into_array());
            let context = format!("{} x {} at {start}", stringify!($alias), $lanes);
            let each = |operation: &dyn Fn(Value, Value) -> Value| -> [$bits; $lanes] {
                core::array::from_fn(|lane| operation(x[lane], y[lane]).to_bits())
            };
            let union =
                |operation: &dyn Fn(Value, Value) -> (Value, Flags)| -> ([$bits; $lanes], Flags) {
                    let mut flags = Flags::NONE;
                    let bits = core::array::from_fn(|lane| {
                        let (value, lane_flags) = operation(x[lane], y[lane]);
                        flags |= lane_flags;
                        value.to_bits()
                    });
                    (bits, flags)
                };
            let lanes = |(value, flags): (Lanes<Value, $lanes>, Flags)| (value.to_bits(), flags);
            assert_eq!(a.abs().to_bits(), each(&|x, _| x.abs()), "{context}: abs");
            assert_eq!((-a).to_bits(), each(&|x, _| -x), "{context}: neg");
            assert_eq!(
                a.copy_sign(b).to_bits(),
                each(&|x, y| x.copy_sign(y)),
                "{context}: copy_sign"
            );
            assert_eq!(a.classify(), x.map(Value::classify), "{context}: classify");
            assert_eq!(a.is_nan(), x.map(Value::is_nan), "{context}: is_nan");
            assert_eq!(
                a.is_signaling_nan(),
                x.map(Value::is_signaling_nan),
                "{context}"
            );
            assert_eq!(a.is_infinite(), x.map(Value::is_infinite), "{context}");
            assert_eq!(a.is_finite(), x.map(Value::is_finite), "{context}");
            assert_eq!(a.is_zero(), x.map(Value::is_zero), "{context}");
            assert_eq!(a.is_subnormal(), x.map(Value::is_subnormal), "{context}");
            assert_eq!(a.is_normal(), x.map(Value::is_normal), "{context}");
            assert_eq!(
                a.is_sign_negative(),
                x.map(Value::is_sign_negative),
                "{context}"
            );
            assert_eq!(
                a.is_sign_positive(),
                x.map(Value::is_sign_positive),
                "{context}"
            );
            assert_eq!(a.is_canonical(), x.map(Value::is_canonical), "{context}");
            assert_eq!(
                a.to_int::<i64>(),
                x.map(Value::to_int::<i64>),
                "{context}: to_int"
            );
            assert_eq!(
                a.to_int::<u8>(),
                x.map(Value::to_int::<u8>),
                "{context}: to_int u8"
            );
            let mut flags = Flags::NONE;
            let expected = x.map(|x| {
                let (result, lane_flags) = x.to_int_with::<i32>(behavior);
                flags |= lane_flags;
                result
            });
            assert_eq!(
                a.to_int_with::<i32>(behavior),
                (expected, flags),
                "{context}: to_int_with"
            );
            let integers: [i64; $lanes] = core::array::from_fn(|lane| {
                let bits = encodings[(start * 3 + lane) % encodings.len()];
                i64::from_ne_bytes(
                    u64::try_from(bits & u128::from(u64::MAX))
                        .expect("64 bits")
                        .to_ne_bytes(),
                ) >> (bits % 64)
            });
            assert_eq!(
                Lanes::<Value, $lanes>::from_int(integers).to_bits(),
                integers.map(|integer| Value::from_int(integer).to_bits()),
                "{context}: from_int"
            );
            let mut flags = Flags::NONE;
            let expected = integers.map(|integer| {
                let (value, lane_flags) = Value::from_int_with(integer, behavior);
                flags |= lane_flags;
                value.to_bits()
            });
            assert_eq!(
                lanes(Lanes::<Value, $lanes>::from_int_with(integers, behavior)),
                (expected, flags),
                "{context}: from_int_with"
            );
            assert_eq!(
                a.next_up().to_bits(),
                each(&|x, _| x.next_up()),
                "{context}: next_up"
            );
            assert_eq!(
                a.next_down().to_bits(),
                each(&|x, _| x.next_down()),
                "{context}: next_down"
            );
            assert_eq!(
                lanes(a.next_up_with(behavior)),
                union(&|x, _| x.next_up_with(behavior)),
                "{context}"
            );
            assert_eq!(
                lanes(a.next_down_with(behavior)),
                union(&|x, _| x.next_down_with(behavior)),
                "{context}"
            );
            let scales: [i32; $lanes] = core::array::from_fn(|lane| {
                i32::try_from(encodings[(start + lane * 5) % encodings.len()] % 97).expect("small")
                    - 48
            });
            let scaled: [$bits; $lanes] =
                core::array::from_fn(|lane| x[lane].scale_b(scales[lane]).to_bits());
            assert_eq!(a.scale_b(scales).to_bits(), scaled, "{context}: scale_b");
            let mut flags = Flags::NONE;
            let scaled: [$bits; $lanes] = core::array::from_fn(|lane| {
                let (value, lane_flags) = x[lane].scale_b_with(scales[lane], behavior);
                flags |= lane_flags;
                value.to_bits()
            });
            assert_eq!(
                lanes(a.scale_b_with(scales, behavior)),
                (scaled, flags),
                "{context}: scale_b_with"
            );
            assert_eq!(
                a.remainder(b).to_bits(),
                each(&|x, y| x.remainder(y)),
                "{context}: remainder"
            );
            assert_eq!(
                lanes(a.remainder_with(b, behavior)),
                union(&|x, y| x.remainder_with(y, behavior)),
                "{context}"
            );
            let orders: [Option<core::cmp::Ordering>; $lanes] =
                core::array::from_fn(|lane| x[lane].partial_cmp(&y[lane]));
            assert_eq!(a.compare_quiet(b), orders, "{context}: compare_quiet");
            for signaling in [false, true] {
                let mut flags = Flags::NONE;
                let orders: [Option<core::cmp::Ordering>; $lanes] = core::array::from_fn(|lane| {
                    let (order, lane_flags) = if signaling {
                        x[lane].compare_signaling_with(y[lane], behavior)
                    } else {
                        x[lane].compare_quiet_with(y[lane], behavior)
                    };
                    flags |= lane_flags;
                    order
                });
                let ours = if signaling {
                    a.compare_signaling_with(b, behavior)
                } else {
                    a.compare_quiet_with(b, behavior)
                };
                assert_eq!(
                    ours,
                    (orders, flags),
                    "{context}: compare_with, signaling {signaling}"
                );
            }
            let total: [core::cmp::Ordering; $lanes] =
                core::array::from_fn(|lane| x[lane].total_cmp(y[lane]));
            assert_eq!(a.total_cmp(b), total, "{context}: total_cmp");
            let total: [core::cmp::Ordering; $lanes] = core::array::from_fn(|lane| {
                x[lane].total_cmp_with(y[lane], floaty::TotalOrder::Encoding)
            });
            assert_eq!(
                a.total_cmp_with(b, floaty::TotalOrder::Encoding),
                total,
                "{context}: total_cmp_with"
            );
            assert_eq!(
                a.minimum(b).to_bits(),
                each(&|x, y| x.minimum(y)),
                "{context}: minimum"
            );
            assert_eq!(
                a.maximum(b).to_bits(),
                each(&|x, y| x.maximum(y)),
                "{context}: maximum"
            );
            assert_eq!(
                a.minimum_number(b).to_bits(),
                each(&|x, y| x.minimum_number(y)),
                "{context}"
            );
            assert_eq!(
                a.maximum_number(b).to_bits(),
                each(&|x, y| x.maximum_number(y)),
                "{context}"
            );
            assert_eq!(
                a.min_num(b).to_bits(),
                each(&|x, y| x.min_num(y)),
                "{context}: min_num"
            );
            assert_eq!(
                a.max_num(b).to_bits(),
                each(&|x, y| x.max_num(y)),
                "{context}: max_num"
            );
            assert_eq!(
                lanes(a.minimum_with(b, behavior)),
                union(&|x, y| x.minimum_with(y, behavior)),
                "{context}"
            );
            assert_eq!(
                lanes(a.maximum_with(b, behavior)),
                union(&|x, y| x.maximum_with(y, behavior)),
                "{context}"
            );
            assert_eq!(
                lanes(a.minimum_number_with(b, behavior)),
                union(&|x, y| x.minimum_number_with(y, behavior)),
                "{context}"
            );
            assert_eq!(
                lanes(a.maximum_number_with(b, behavior)),
                union(&|x, y| x.maximum_number_with(y, behavior)),
                "{context}"
            );
            assert_eq!(
                lanes(a.min_num_with(b, behavior)),
                union(&|x, y| x.min_num_with(y, behavior)),
                "{context}"
            );
            assert_eq!(
                lanes(a.max_num_with(b, behavior)),
                union(&|x, y| x.max_num_with(y, behavior)),
                "{context}"
            );
        }
    }};
}

#[test]
fn the_other_methods_of_lanes_give_the_scalar_results() {
    // The pairs of the windows meet zeros of both signs, NaNs, and equal
    // values, which the packed minimum and maximum send to the scalar paths.
    let mut random = SplitMix64::new(0x1A4E_3E7D);
    let singles = encodings(&mut random, 32, 8);
    let behavior = Env::IEEE.with_denormals_are_zero(true);
    methods_match!(F32, u32, 1, &singles, Env::IEEE);
    methods_match!(F32, u32, 4, &singles, behavior);
    methods_match!(F32, u32, 5, &singles, Env::X86_SSE);
    methods_match!(F32, u32, 13, &singles, Env::IEEE);
    let doubles = encodings(&mut random, 64, 11);
    methods_match!(F64, u64, 2, &doubles, Env::IEEE);
    methods_match!(F64, u64, 7, &doubles, behavior);
    methods_match!(F16, u16, 8, &encodings(&mut random, 16, 5), Env::IEEE);
    methods_match!(BF16, u16, 8, &encodings(&mut random, 16, 8), Env::IEEE);
    methods_match!(F80, u128, 2, &encodings(&mut random, 80, 15), Env::IEEE);
    methods_match!(F128, u128, 3, &encodings(&mut random, 128, 15), Env::IEEE);
    let decimals: Vec<u128> = (0..400).map(|_| u128::from(random.next_u64())).collect();
    methods_match!(floaty::D64Bid, u64, 2, &decimals, floaty::D64Bid::ENV);
}

#[test]
fn every_pair_of_boundary_values_orders_as_the_scalar_operations() {
    // Each pair of boundary encodings, such as `+0` and `-0`, a NaN and a
    // number, or two equal values, fills the lanes, so every chunk of the
    // packed comparison and minimum and maximum meets it.
    let singles = boundary_encodings_u128(32, 8, IntegerBit::Implicit);
    let pairs: Vec<(u32, u32)> = singles
        .iter()
        .flat_map(|&a| singles.iter().map(move |&b| (a, b)))
        .map(|(a, b)| {
            (
                u32::try_from(a).expect("32 bits"),
                u32::try_from(b).expect("32 bits"),
            )
        })
        .collect();
    for chunk in pairs.chunks_exact(8) {
        let x = Lanes::<F32, 8>::from_bits(core::array::from_fn(|lane| chunk[lane].0));
        let y = Lanes::<F32, 8>::from_bits(core::array::from_fn(|lane| chunk[lane].1));
        let (a, b) = (x.into_array(), y.into_array());
        let each = |operation: fn(F32, F32) -> F32| -> [u32; 8] {
            core::array::from_fn(|lane| operation(a[lane], b[lane]).to_bits())
        };
        assert_eq!(
            x.minimum(y).to_bits(),
            each(F32::minimum),
            "{chunk:x?}: minimum"
        );
        assert_eq!(
            x.maximum(y).to_bits(),
            each(F32::maximum),
            "{chunk:x?}: maximum"
        );
        assert_eq!(
            x.minimum_number(y).to_bits(),
            each(F32::minimum_number),
            "{chunk:x?}"
        );
        assert_eq!(
            x.maximum_number(y).to_bits(),
            each(F32::maximum_number),
            "{chunk:x?}"
        );
        assert_eq!(
            x.min_num(y).to_bits(),
            each(F32::min_num),
            "{chunk:x?}: min_num"
        );
        assert_eq!(
            x.max_num(y).to_bits(),
            each(F32::max_num),
            "{chunk:x?}: max_num"
        );
        let orders: [Option<core::cmp::Ordering>; 8] =
            core::array::from_fn(|lane| a[lane].partial_cmp(&b[lane]));
        assert_eq!(x.compare_quiet(y), orders, "{chunk:x?}: compare_quiet");
    }
    let doubles = boundary_encodings_u128(64, 11, IntegerBit::Implicit);
    for &a in &doubles {
        for pair in doubles.chunks_exact(4) {
            let x = Lanes::<F64, 4>::from_bits([u64::try_from(a).expect("64 bits"); 4]);
            let y = Lanes::<F64, 4>::from_bits(core::array::from_fn(|lane| {
                u64::try_from(pair[lane]).expect("64 bits")
            }));
            let (left, right) = (x.into_array(), y.into_array());
            let each = |operation: fn(F64, F64) -> F64| -> [u64; 4] {
                core::array::from_fn(|lane| operation(left[lane], right[lane]).to_bits())
            };
            assert_eq!(
                x.minimum(y).to_bits(),
                each(F64::minimum),
                "{a:#x} {pair:x?}"
            );
            assert_eq!(
                x.max_num(y).to_bits(),
                each(F64::max_num),
                "{a:#x} {pair:x?}"
            );
            let orders: [Option<core::cmp::Ordering>; 4] =
                core::array::from_fn(|lane| left[lane].partial_cmp(&right[lane]));
            assert_eq!(x.compare_quiet(y), orders, "{a:#x} {pair:x?}");
        }
    }
}
