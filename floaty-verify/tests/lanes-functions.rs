//! Checks that each lane of the functions of `Lanes` gives the bits of the
//! scalar operation, and that a `_with` method returns the union of the
//! flags of the lanes, on every target. The scalar operations pass the
//! oracle tests.
//!
//! The functions are the elementary functions of every format, with `atan2`
//! and `atan2Pi`; `log_b`, `hypot`, `reciprocal_sqrt`, `pown`, `rootn`,
//! `compound`, the augmented operations, and the NaN payload operations of
//! the binary formats; and the quantum operations, `log_b`, and the NaN
//! payload operations of the decimal formats. Each lane of `pown`, `rootn`,
//! and `compound` takes its own exponent.

use floaty::env::Rounding;
use floaty::format::Standard;
use floaty::{Augmented, BF16, Env, F16, F32, F64, F80, F128, Flags, Float, Lanes};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::random::SplitMix64;

/// The exponents of `pown`, `rootn`, and `compound`, which the lanes take in
/// turn: zero, small ones of both signs, and ones past the limit of 64.
const COUNTS: [i64; 9] = [-3, -2, -1, 0, 1, 2, 3, 65, 1000];

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

/// Returns boundary and random encodings of a layout, NaNs among them.
fn encodings(random: &mut SplitMix64, layout: Layout) -> Vec<u128> {
    let mut encodings = boundary_encodings_u128(layout);
    encodings.extend((0..300).map(|_| random.next_u128() >> (128 - layout.width)));
    encodings
}

/// Returns the values that start at `start` in a cycle of `encodings`. Each
/// encoding reaches every lane as `start` moves.
fn window<S: Standard<W>, const W: usize, const N: usize>(
    encodings: &[u128],
    start: usize,
) -> [Float<S, W>; N]
where
    S::Bits: TryFrom<u128, Error: core::fmt::Debug>,
{
    core::array::from_fn(|lane| {
        let bits = encodings[(start + lane) % encodings.len()];
        Float::from_bits(bits.try_into().expect("the encoding fits the storage"))
    })
}

/// The method of `Lanes` without flags, and the method with flags.
type LaneMethods<'a, S, const W: usize, const N: usize> = (
    &'a dyn Fn(Lanes<Float<S, W>, N>) -> Lanes<Float<S, W>, N>,
    &'a dyn Fn(Lanes<Float<S, W>, N>, Env) -> (Lanes<Float<S, W>, N>, Flags),
);

/// The scalar method of lane `i` without flags, and the method with flags.
type ScalarMethods<'a, S, const W: usize> = (
    &'a dyn Fn(Float<S, W>, usize) -> Float<S, W>,
    &'a dyn Fn(Float<S, W>, usize, Env) -> (Float<S, W>, Flags),
);

/// Checks a method of `Lanes` on the lanes `x` against the scalar method of
/// each lane, without flags and with the behavior `env`.
fn check<S: Standard<W>, const W: usize, const N: usize>(
    context: &dyn Fn() -> String,
    x: [Float<S, W>; N],
    env: Env,
    (lanes, lanes_with): LaneMethods<'_, S, W, N>,
    (scalar, scalar_with): ScalarMethods<'_, S, W>,
) {
    let expected: [S::Bits; N] = core::array::from_fn(|lane| scalar(x[lane], lane).to_bits());
    assert_eq!(lanes(Lanes::new(x)).to_bits(), expected, "{}", context());
    let mut flags = Flags::NONE;
    let expected: [S::Bits; N] = core::array::from_fn(|lane| {
        let (value, lane_flags) = scalar_with(x[lane], lane, env);
        flags |= lane_flags;
        value.to_bits()
    });
    let (result, result_flags) = lanes_with(Lanes::new(x), env);
    assert_eq!(
        (result.to_bits(), result_flags),
        (expected, flags),
        "{} with",
        context()
    );
}

/// Checks an augmented operation of `Lanes` on the lanes `x` and `y` against
/// the scalar operation of each pair of lanes, with the behavior `env`.
#[allow(clippy::type_complexity)] // Two method types, used once each.
fn check_augmented<S: Standard<W>, const W: usize, const N: usize>(
    context: &dyn Fn() -> String,
    (x, y): ([Float<S, W>; N], [Float<S, W>; N]),
    env: Env,
    lanes_with: &dyn Fn(
        Lanes<Float<S, W>, N>,
        Lanes<Float<S, W>, N>,
        Env,
    ) -> (Augmented<Lanes<Float<S, W>, N>>, Flags),
    scalar_with: &dyn Fn(Float<S, W>, Float<S, W>, Env) -> (Augmented<Float<S, W>>, Flags),
) {
    let mut flags = Flags::NONE;
    let pairs: [Augmented<Float<S, W>>; N] = core::array::from_fn(|lane| {
        let (pair, lane_flags) = scalar_with(x[lane], y[lane], env);
        flags |= lane_flags;
        pair
    });
    let (result, result_flags) = lanes_with(Lanes::new(x), Lanes::new(y), env);
    let expected = (
        pairs.map(|pair| pair.head.to_bits()),
        pairs.map(|pair| pair.tail.to_bits()),
        flags,
    );
    let actual = (result.head.to_bits(), result.tail.to_bits(), result_flags);
    assert_eq!(actual, expected, "{}", context());
}

/// Checks the exponentials and the logarithms of `Lanes` of one type on the
/// lanes `x`.
macro_rules! check_elementary {
    ($context:expr, $x:expr, $env:expr) => {{
        let (context, x, env) = ($context, $x, $env);
        check_elementary!(
            @each context, x, env;
            exp, exp_with;
            exp_m1, exp_m1_with;
            exp2, exp2_with;
            exp2_m1, exp2_m1_with;
            exp10, exp10_with;
            exp10_m1, exp10_m1_with;
            log, log_with;
            log2, log2_with;
            log10, log10_with;
            log_p1, log_p1_with;
            log2_p1, log2_p1_with;
            log10_p1, log10_p1_with;
            sinh, sinh_with;
            cosh, cosh_with;
            tanh, tanh_with;
            asinh, asinh_with;
            acosh, acosh_with;
            atanh, atanh_with;
            sin_pi, sin_pi_with;
            cos_pi, cos_pi_with;
            tan_pi, tan_pi_with;
            asin, asin_with;
            acos, acos_with;
            atan, atan_with;
            asin_pi, asin_pi_with;
            acos_pi, acos_pi_with;
            atan_pi, atan_pi_with
        );
    }};
    (@each $context:ident, $x:ident, $env:ident; $($name:ident, $name_with:ident);*) => {
        $(
            check(
                &|| $context(stringify!($name)),
                $x,
                $env,
                (&|lanes| lanes.$name(), &|lanes, env| lanes.$name_with(env)),
                (&|x, _| x.$name(), &|x, _, env| x.$name_with(env)),
            );
        )*
    };
}

/// Checks `atan2` and `atan2Pi` of `Lanes` of one type on the lanes `y` and
/// `x`, `y` first.
macro_rules! check_atan2 {
    ($context:expr, $y:expr, $x:expr, $env:expr) => {{
        let (context, y, x, env) = ($context, $y, $x, $env);
        check(
            &|| context("atan2"),
            y,
            env,
            (&|lanes| lanes.atan2(Lanes::new(x)), &|lanes, env| {
                lanes.atan2_with(Lanes::new(x), env)
            }),
            (&|y, lane| y.atan2(x[lane]), &|y, lane, env| {
                y.atan2_with(x[lane], env)
            }),
        );
        check(
            &|| context("atan2_pi"),
            y,
            env,
            (&|lanes| lanes.atan2_pi(Lanes::new(x)), &|lanes, env| {
                lanes.atan2_pi_with(Lanes::new(x), env)
            }),
            (&|y, lane| y.atan2_pi(x[lane]), &|y, lane, env| {
                y.atan2_pi_with(x[lane], env)
            }),
        );
    }};
}

/// Checks the functions of `Lanes` of one binary type and lane count
/// against the scalar functions of each lane.
macro_rules! binary_functions_match {
    ($alias:ty, $lanes:literal, $encodings:expr) => {{
        type Value = $alias;
        let encodings: &[u128] = $encodings;
        for start in 0..encodings.len() {
            let x: [Value; $lanes] = window(encodings, start);
            let y: [Value; $lanes] = window(encodings, start * 7 + 3);
            let n: [i64; $lanes] =
                core::array::from_fn(|lane| COUNTS[(start + 4 * lane) % COUNTS.len()]);
            for env in behaviors() {
                let context = |name: &str| {
                    format!(
                        "{} x {} at {start}: {name} {env:?}",
                        stringify!($alias),
                        $lanes
                    )
                };
                check_elementary!(context, x, env);
                check_atan2!(context, x, y, env);
                check(
                    &|| context("log_b"),
                    x,
                    env,
                    (&|lanes| lanes.log_b(), &|lanes, env| lanes.log_b_with(env)),
                    (&|x, _| x.log_b(), &|x, _, env| x.log_b_with(env)),
                );
                check(
                    &|| context("hypot"),
                    x,
                    env,
                    (&|lanes| lanes.hypot(Lanes::new(y)), &|lanes, env| {
                        lanes.hypot_with(Lanes::new(y), env)
                    }),
                    (&|x, lane| x.hypot(y[lane]), &|x, lane, env| {
                        x.hypot_with(y[lane], env)
                    }),
                );
                check(
                    &|| context("reciprocal_sqrt"),
                    x,
                    env,
                    (&|lanes| lanes.reciprocal_sqrt(), &|lanes, env| {
                        lanes.reciprocal_sqrt_with(env)
                    }),
                    (&|x, _| x.reciprocal_sqrt(), &|x, _, env| {
                        x.reciprocal_sqrt_with(env)
                    }),
                );
                check(
                    &|| context("pown"),
                    x,
                    env,
                    (&|lanes| lanes.pown(n), &|lanes, env| {
                        lanes.pown_with(n, env)
                    }),
                    (&|x, lane| x.pown(n[lane]), &|x, lane, env| {
                        x.pown_with(n[lane], env)
                    }),
                );
                check(
                    &|| context("rootn"),
                    x,
                    env,
                    (&|lanes| lanes.rootn(n), &|lanes, env| {
                        lanes.rootn_with(n, env)
                    }),
                    (&|x, lane| x.rootn(n[lane]), &|x, lane, env| {
                        x.rootn_with(n[lane], env)
                    }),
                );
                check(
                    &|| context("compound"),
                    x,
                    env,
                    (&|lanes| lanes.compound(n), &|lanes, env| {
                        lanes.compound_with(n, env)
                    }),
                    (&|x, lane| x.compound(n[lane]), &|x, lane, env| {
                        x.compound_with(n[lane], env)
                    }),
                );
                check_augmented(
                    &|| context("augmented_add"),
                    (x, y),
                    env,
                    &|x, y, env| x.augmented_add_with(y, env),
                    &|x, y, env| x.augmented_add_with(y, env),
                );
                check_augmented(
                    &|| context("augmented_sub"),
                    (x, y),
                    env,
                    &|x, y, env| x.augmented_sub_with(y, env),
                    &|x, y, env| x.augmented_sub_with(y, env),
                );
                check_augmented(
                    &|| context("augmented_mul"),
                    (x, y),
                    env,
                    &|x, y, env| x.augmented_mul_with(y, env),
                    &|x, y, env| x.augmented_mul_with(y, env),
                );
            }
            let (a, b) = (Lanes::new(x), Lanes::new(y));
            let context = format!("{} x {} at {start}", stringify!($alias), $lanes);
            let heads = (
                a.augmented_add(b).head.to_bits(),
                a.augmented_sub(b).tail.to_bits(),
                a.augmented_mul(b).head.to_bits(),
            );
            let expected = (
                core::array::from_fn(|lane| x[lane].augmented_add(y[lane]).head.to_bits()),
                core::array::from_fn(|lane| x[lane].augmented_sub(y[lane]).tail.to_bits()),
                core::array::from_fn(|lane| x[lane].augmented_mul(y[lane]).head.to_bits()),
            );
            assert_eq!(heads, expected, "{context}: augmented, default mode");
            let payloads = (
                a.payload().to_bits(),
                Lanes::<Value, $lanes>::from_payload(a).to_bits(),
                Lanes::<Value, $lanes>::from_payload_signaling(a).to_bits(),
            );
            let expected = (
                x.map(|x| x.payload().to_bits()),
                x.map(|x| Value::from_payload(x).to_bits()),
                x.map(|x| Value::from_payload_signaling(x).to_bits()),
            );
            assert_eq!(payloads, expected, "{context}: payloads");
        }
    }};
}

#[test]
fn binary_lanes_give_the_scalar_functions() {
    let mut random = SplitMix64::new(0x1A4E_F000);
    let single = encodings(&mut random, Layout::BINARY32);
    binary_functions_match!(F32, 4, &single);
    binary_functions_match!(F64, 2, &encodings(&mut random, Layout::BINARY64));
    binary_functions_match!(F16, 8, &encodings(&mut random, Layout::BINARY16));
    binary_functions_match!(BF16, 3, &encodings(&mut random, Layout::BFLOAT16));
    binary_functions_match!(F80, 2, &encodings(&mut random, Layout::X87_EXTENDED));
    binary_functions_match!(F128, 2, &encodings(&mut random, Layout::BINARY128));
    let every: Vec<u128> = (0..1 << 8).collect();
    binary_functions_match!(floaty::F8E4M3Fn, 5, &every);
    binary_functions_match!(floaty::F8E5M2Fnuz, 3, &every);
    let every: Vec<u128> = (0..1 << 4).collect();
    binary_functions_match!(floaty::F4E2M1Fn, 4, &every);
}

/// Checks the functions of `Lanes` of one decimal type and lane count
/// against the scalar functions of each lane.
macro_rules! decimal_functions_match {
    ($alias:ty, $lanes:literal, $encodings:expr) => {{
        type Value = $alias;
        let encodings: &[u128] = $encodings;
        for start in 0..encodings.len() {
            let x: [Value; $lanes] = window(encodings, start);
            let y: [Value; $lanes] = window(encodings, start * 7 + 3);
            for env in behaviors() {
                let context = |name: &str| {
                    format!(
                        "{} x {} at {start}: {name} {env:?}",
                        stringify!($alias),
                        $lanes
                    )
                };
                check_elementary!(context, x, env);
                check_atan2!(context, x, y, env);
                check(
                    &|| context("log_b"),
                    x,
                    env,
                    (&|lanes| lanes.log_b(), &|lanes, env| lanes.log_b_with(env)),
                    (&|x, _| x.log_b(), &|x, _, env| x.log_b_with(env)),
                );
                check(
                    &|| context("quantize"),
                    x,
                    env,
                    (&|lanes| lanes.quantize(Lanes::new(y)), &|lanes, env| {
                        lanes.quantize_with(Lanes::new(y), env)
                    }),
                    (&|x, lane| x.quantize(y[lane]), &|x, lane, env| {
                        x.quantize_with(y[lane], env)
                    }),
                );
                check(
                    &|| context("quantum"),
                    x,
                    env,
                    (&|lanes| lanes.quantum(), &|lanes, env| {
                        lanes.quantum_with(env)
                    }),
                    (&|x, _| x.quantum(), &|x, _, env| x.quantum_with(env)),
                );
            }
            let (a, b) = (Lanes::new(x), Lanes::new(y));
            let context = format!("{} x {} at {start}", stringify!($alias), $lanes);
            let expected: [bool; $lanes] =
                core::array::from_fn(|lane| x[lane].same_quantum(y[lane]));
            assert_eq!(a.same_quantum(b), expected, "{context}: same_quantum");
            let payloads = (
                a.payload().to_bits(),
                Lanes::<Value, $lanes>::from_payload(a).to_bits(),
                Lanes::<Value, $lanes>::from_payload_signaling(a).to_bits(),
            );
            let expected = (
                x.map(|x| x.payload().to_bits()),
                x.map(|x| Value::from_payload(x).to_bits()),
                x.map(|x| Value::from_payload_signaling(x).to_bits()),
            );
            assert_eq!(payloads, expected, "{context}: payloads");
        }
    }};
}

#[test]
fn decimal_lanes_give_the_scalar_functions() {
    use floaty::{D32Bid, D64Dpd, D128Bid};
    let mut random = SplitMix64::new(0x1A4E_FDEC);
    let mut random_encodings = |width: u32| -> Vec<u128> {
        (0..300)
            .map(|_| random.next_u128() >> (128 - width))
            .collect()
    };
    decimal_functions_match!(D32Bid, 4, &random_encodings(32));
    decimal_functions_match!(D64Dpd, 2, &random_encodings(64));
    decimal_functions_match!(D128Bid, 2, &random_encodings(128));
}
