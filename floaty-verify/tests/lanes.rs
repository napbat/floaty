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
