//! Checks that each lane of `Lanes` gives the bits of the scalar operation,
//! and that a `_with` method returns the union of the flags of the lanes, on
//! every target. The scalar operations pass the oracle tests.
//!
//! The lane counts take every path of the packed host paths: wide chunks,
//! narrow chunks, and single lanes. binary32, binary64, binary16, and the
//! rounding of bfloat16 take the packed paths where the build has them; the
//! other formats and operations take the scalar operation of each lane. The
//! other formats are x87 extended, binary128, the FP8 and MX formats, TF32,
//! the wide formats, and the decimal formats.

use floaty::env::Rounding;
use floaty::mode::{Rounded, X86Sse, direction::TowardZero};
use floaty::{BF16, Env, F16, F32, F64, F80, F128, Flags, Float, Lanes, TF32};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;

/// Returns boundary and random encodings of a layout, NaNs among them.
fn encodings(random: &mut SplitMix64, layout: Layout) -> Vec<u128> {
    let mut encodings = boundary_encodings_u128(layout);
    encodings.extend((0..600).map(|_| random.next_u128() >> (128 - layout.width)));
    encodings
}

/// Returns `count` random encodings of `width` bits. The random bits reach
/// the special values of the decimal formats too.
fn random_encodings(random: &mut SplitMix64, width: u32, count: usize) -> Vec<u128> {
    (0..count)
        .map(|_| random.next_u128() >> (128 - width))
        .collect()
}

/// Sets bit `bit` of `limbs` to `value`.
fn put_bit<const N: usize>(limbs: &mut [u64; N], bit: u32, value: bool) {
    let limb = &mut limbs[usize::try_from(bit / 64).expect("a limb index fits a usize")];
    let mask = 1 << (bit % 64);
    *limb = if value { *limb | mask } else { *limb & !mask };
}

/// Returns encodings of a wide IEEE format of `width` bits in `N` limbs:
/// both zeros, the smallest subnormals, both infinities, quiet and
/// signaling NaNs, and random encodings. One random encoding in eight has
/// the largest exponent field, and one in eight the smallest.
fn wide_encodings<const N: usize>(
    random: &mut SplitMix64,
    width: u32,
    exponent_bits: u32,
) -> Vec<[u64; N]> {
    let fraction_bits = width - 1 - exponent_bits;
    let storage_bits = u32::try_from(64 * N).expect("the storage has at most 512 bits");
    let field = |limbs: &mut [u64; N], ones: bool| {
        (fraction_bits..width - 1).for_each(|bit| put_bit(limbs, bit, ones));
    };
    let mut encodings = Vec::new();
    for negative in [false, true] {
        for fraction_bit in [None, Some(0), Some(fraction_bits - 1)] {
            for ones in [false, true] {
                let mut limbs = [0; N];
                put_bit(&mut limbs, width - 1, negative);
                field(&mut limbs, ones);
                if let Some(bit) = fraction_bit {
                    put_bit(&mut limbs, bit, true);
                }
                encodings.push(limbs);
            }
        }
    }
    encodings.extend((0..200).map(|index| {
        let mut limbs: [u64; N] = core::array::from_fn(|_| random.next_u64());
        (width..storage_bits).for_each(|bit| put_bit(&mut limbs, bit, false));
        match index % 8 {
            0 => field(&mut limbs, true),
            1 => field(&mut limbs, false),
            _ => {}
        }
        limbs
    }));
    encodings
}

/// Returns the lanes that start at `start` in a cycle of `encodings`. Each
/// encoding reaches every lane as `start` moves.
fn window<T, const N: usize>(
    encodings: &[impl Copy + TryInto<T, Error: core::fmt::Debug>],
    start: usize,
) -> [T; N] {
    core::array::from_fn(|lane| {
        encodings[(start + lane) % encodings.len()]
            .try_into()
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
        let encodings: &[_] = $encodings;
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
    let encodings = encodings(&mut random, Layout::BINARY32);
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
    let encodings = encodings(&mut random, Layout::BINARY64);
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
    let encodings = encodings(&mut random, Layout::BINARY16);
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
    let encodings = encodings(&mut random, Layout::BFLOAT16);
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
    lanes_match!(
        F80,
        u128,
        2,
        &encodings(&mut random, Layout::X87_EXTENDED),
        behaviors()
    );
    lanes_match!(
        F128,
        u128,
        3,
        &encodings(&mut random, Layout::BINARY128),
        behaviors()
    );
}

#[test]
fn lanes_of_other_modes_give_the_scalar_results() {
    // The SSE mode takes the packed paths and selects NaNs by its own rule.
    // A directed mode takes them only for the rounding to integral values.
    type Sse = Float<floaty::Binary<8>, 32, X86Sse>;
    type Truncating = Float<floaty::Binary<11>, 64, Rounded<floaty::mode::Ieee, TowardZero>>;
    let mut random = SplitMix64::new(0x1A4E_00DE);
    lanes_match!(
        Sse,
        u32,
        8,
        &encodings(&mut random, Layout::BINARY32),
        [Env::X86_SSE]
    );
    lanes_match!(
        Truncating,
        u64,
        4,
        &encodings(&mut random, Layout::BINARY64),
        [Env::IEEE]
    );
}

#[test]
fn lanes_round_in_the_direction_of_their_mode() {
    // `round_to_integral` takes the direction of the mode in the packed
    // paths. Each direction and format below checks every operation of
    // `lanes_match!`: the arithmetic of a directed mode runs in the engine,
    // and the rounding to integral values takes the paths.
    use floaty::mode::direction::{
        AwayFromZero, TiesToAway, TiesTowardZero, ToOdd, TowardNegative, TowardPositive,
    };
    use floaty::mode::{Ieee, Rounded};
    type Double<R> = Float<floaty::Binary<11>, 64, Rounded<Ieee, R>>;
    type Up = Float<floaty::Binary<8>, 32, Rounded<Ieee, TowardPositive>>;
    type Down = Float<floaty::Binary<5>, 16, Rounded<Ieee, TowardNegative>>;
    type Away = Float<floaty::Binary<11>, 64, Rounded<Ieee, TiesToAway>>;
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
    let singles = with_ties(encodings(&mut random, Layout::BINARY32), |value| {
        u128::from(value.convert::<F32>().to_bits())
    });
    let halves = with_ties(encodings(&mut random, Layout::BINARY16), |value| {
        u128::from(value.convert::<F16>().to_bits())
    });
    let doubles = with_ties(encodings(&mut random, Layout::BINARY64), |value| {
        u128::from(value.to_bits())
    });
    let bfloats = with_ties(encodings(&mut random, Layout::BFLOAT16), |value| {
        u128::from(value.convert::<BF16>().to_bits())
    });
    lanes_match!(Up, u32, 13, &singles, [Env::IEEE]);
    lanes_match!(Odd, u32, 5, &singles, [Env::IEEE]);
    lanes_match!(Down, u16, 13, &halves, [Env::IEEE]);
    lanes_match!(Away, u64, 7, &doubles, [Env::IEEE]);
    // Seven binary64 lanes take a 256-bit chunk, a 128-bit chunk, and one
    // single lane in the x86-64-v3 build, and each form of `ROUNDPD`.
    lanes_match!(Double<TowardNegative>, u64, 7, &doubles, [Env::IEEE]);
    lanes_match!(Double<TowardPositive>, u64, 7, &doubles, [Env::IEEE]);
    lanes_match!(Double<TowardZero>, u64, 7, &doubles, [Env::IEEE]);
    lanes_match!(Double<TowardZero>, u64, 3, &doubles, [Env::IEEE]);
    // No unit rounds to an integral value in these two directions, so the
    // packed paths decline them. Two lanes take only a 128-bit chunk, and
    // four lanes only a 256-bit chunk in the x86-64-v3 build. A scalar lane
    // after a chunk would decline too and hide a wrong chunk.
    lanes_match!(Double<TiesTowardZero>, u64, 2, &doubles, [Env::IEEE]);
    lanes_match!(Double<TiesTowardZero>, u64, 4, &doubles, [Env::IEEE]);
    lanes_match!(Double<AwayFromZero>, u64, 2, &doubles, [Env::IEEE]);
    lanes_match!(Double<AwayFromZero>, u64, 4, &doubles, [Env::IEEE]);
    lanes_match!(Chopped, u16, 13, &bfloats, [Env::IEEE]);
}

#[test]
fn lanes_convert_as_their_lanes_do() {
    let mut random = SplitMix64::new(0x1A4E_C0F7);
    let singles = encodings(&mut random, Layout::BINARY32);
    let doubles = encodings(&mut random, Layout::BINARY64);
    let binary16 = encodings(&mut random, Layout::BINARY16);
    let bfloat16 = encodings(&mut random, Layout::BFLOAT16);
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
/// the integer conversions, the steps, `scale_b`, the remainders, the
/// operators `%`, `+=`, `-=`, `*=`, `/=`, and `%=`, the comparisons, the
/// total order, and the minimum and maximum operations. A `_with` method
/// must return the union of the flags of the lanes.
macro_rules! methods_match {
    ($alias:ty, $bits:ty, $lanes:literal, $encodings:expr, $behavior:expr) => {{
        type Value = $alias;
        let encodings: &[_] = $encodings;
        let behavior = $behavior;
        for start in 0..encodings.len() {
            // The integers and the scales come from a generator seeded by the
            // window, so that they do not depend on the type of the encodings.
            let mut random = SplitMix64::new(u64::try_from(start).expect("an index fits a u64"));
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
            let integers: [i64; $lanes] = core::array::from_fn(|_| {
                i64::from_ne_bytes(random.next_u64().to_ne_bytes()) >> random.below(64)
            });
            assert_eq!(
                Lanes::<Value, $lanes>::from_int(integers).to_bits(),
                integers.map(|integer| Value::from_int(integer).to_bits()),
                "{context}: from_int"
            );
            assert_eq!(
                a.to_int::<i32>(),
                x.map(Value::to_int::<i32>),
                "{context}: to_int i32"
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
            let scales: [i32; $lanes] = core::array::from_fn(|_| {
                i32::try_from(random.below(97)).expect("a scale below 97 fits an i32") - 48
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
            assert_eq!(
                (a % b).to_bits(),
                each(&|x, y| x.truncated_remainder(y)),
                "{context}: %"
            );
            assert_eq!(
                lanes(a.truncated_remainder_with(b, behavior)),
                union(&|x, y| x.truncated_remainder_with(y, behavior)),
                "{context}"
            );
            let assigned = |operation: &dyn Fn(&mut Lanes<Value, $lanes>)| {
                let mut value = a;
                operation(&mut value);
                value.to_bits()
            };
            assert_eq!(
                assigned(&|value| *value += b),
                (a + b).to_bits(),
                "{context}: +="
            );
            assert_eq!(
                assigned(&|value| *value -= b),
                (a - b).to_bits(),
                "{context}: -="
            );
            assert_eq!(
                assigned(&|value| *value *= b),
                (a * b).to_bits(),
                "{context}: *="
            );
            assert_eq!(
                assigned(&|value| *value /= b),
                (a / b).to_bits(),
                "{context}: /="
            );
            assert_eq!(
                assigned(&|value| *value %= b),
                (a % b).to_bits(),
                "{context}: %="
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
            // The magnitude operations, which no packed path takes.
            assert_eq!(
                a.minimum_magnitude(b).to_bits(),
                each(&|x, y| x.minimum_magnitude(y)),
                "{context}: minimum_magnitude"
            );
            assert_eq!(
                a.maximum_magnitude(b).to_bits(),
                each(&|x, y| x.maximum_magnitude(y)),
                "{context}: maximum_magnitude"
            );
            assert_eq!(
                lanes(a.minimum_magnitude_number_with(b, behavior)),
                union(&|x, y| x.minimum_magnitude_number_with(y, behavior)),
                "{context}"
            );
            assert_eq!(
                lanes(a.maximum_magnitude_number_with(b, behavior)),
                union(&|x, y| x.maximum_magnitude_number_with(y, behavior)),
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
    let singles = encodings(&mut random, Layout::BINARY32);
    let behavior = Env::IEEE.with_denormals_are_zero(true);
    methods_match!(F32, u32, 1, &singles, Env::IEEE);
    methods_match!(F32, u32, 4, &singles, behavior);
    methods_match!(F32, u32, 5, &singles, Env::X86_SSE);
    methods_match!(F32, u32, 13, &singles, Env::IEEE);
    let doubles = encodings(&mut random, Layout::BINARY64);
    methods_match!(F64, u64, 2, &doubles, Env::IEEE);
    methods_match!(F64, u64, 7, &doubles, behavior);
    methods_match!(
        F16,
        u16,
        8,
        &encodings(&mut random, Layout::BINARY16),
        Env::IEEE
    );
    methods_match!(
        BF16,
        u16,
        8,
        &encodings(&mut random, Layout::BFLOAT16),
        Env::IEEE
    );
    methods_match!(
        F80,
        u128,
        2,
        &encodings(&mut random, Layout::X87_EXTENDED),
        Env::IEEE
    );
    methods_match!(
        F128,
        u128,
        3,
        &encodings(&mut random, Layout::BINARY128),
        Env::IEEE
    );
    let decimals: Vec<u128> = (0..400).map(|_| u128::from(random.next_u64())).collect();
    methods_match!(floaty::D64Bid, u64, 2, &decimals, floaty::D64Bid::ENV);
}

/// Every encoding of the FP8 and MX formats, and TF32, fill the lanes. These
/// formats take the scalar operation of each lane.
#[test]
fn lanes_of_the_small_formats_and_tf32_give_the_scalar_results() {
    macro_rules! small {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
            let every: Vec<u128> = (0..1_u128 << $width).collect();
            lanes_match!(floaty::$alias, u8, 5, &every, behaviors());
            methods_match!(floaty::$alias, u8, 3, &every, Env::IEEE);
        }};
    }
    floaty_verify::for_each_small_format!(small);
    let mut random = SplitMix64::new(0x1A4E_0019);
    let tf32 = encodings(&mut random, Layout::TF32);
    lanes_match!(TF32, u32, 5, &tf32, behaviors());
    methods_match!(TF32, u32, 4, &tf32, Env::IEEE);
}

#[test]
fn lanes_of_the_wide_formats_give_the_scalar_results() {
    let mut random = SplitMix64::new(0x1A4E_0160);
    macro_rules! wide {
        ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {{
            let encodings = wide_encodings::<$limbs>(&mut random, $width, $exponent_bits);
            lanes_match!(floaty::$alias, [u64; $limbs], 2, &encodings, behaviors());
            methods_match!(floaty::$alias, [u64; $limbs], 3, &encodings, Env::IEEE);
        }};
    }
    floaty_verify::for_each_wide_format!(wide);
}

#[test]
fn lanes_of_the_decimal_formats_give_the_scalar_results() {
    use floaty::{D32Bid, D32Dpd, D64Bid, D64Dpd, D128Bid, D128Dpd};
    let mut random = SplitMix64::new(0x1A4E_DEC0);
    let (d32, d64, d128) = (
        random_encodings(&mut random, 32, 400),
        random_encodings(&mut random, 64, 400),
        random_encodings(&mut random, 128, 400),
    );
    lanes_match!(D32Bid, u32, 3, &d32, behaviors());
    methods_match!(D32Bid, u32, 3, &d32, D32Bid::ENV);
    lanes_match!(D32Dpd, u32, 4, &d32, behaviors());
    methods_match!(D32Dpd, u32, 4, &d32, D32Dpd::ENV);
    lanes_match!(D64Bid, u64, 2, &d64, behaviors());
    lanes_match!(D64Dpd, u64, 3, &d64, behaviors());
    methods_match!(D64Dpd, u64, 3, &d64, D64Dpd::ENV);
    lanes_match!(D128Bid, u128, 2, &d128, behaviors());
    methods_match!(D128Bid, u128, 2, &d128, D128Bid::ENV);
    lanes_match!(D128Dpd, u128, 3, &d128, behaviors());
    methods_match!(D128Dpd, u128, 3, &d128, D128Dpd::ENV);
}

#[test]
fn every_pair_of_boundary_values_orders_as_the_scalar_operations() {
    // Each pair of boundary encodings, such as `+0` and `-0`, a NaN and a
    // number, or two equal values, fills the lanes, so every chunk of the
    // packed comparison and minimum and maximum meets it.
    let singles = boundary_encodings_u128(Layout::BINARY32);
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
    for chunk in pairs.as_chunks::<8>().0 {
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
    let doubles = boundary_encodings_u128(Layout::BINARY64);
    for &a in &doubles {
        for pair in doubles.as_chunks::<4>().0 {
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
