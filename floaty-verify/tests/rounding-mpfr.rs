//! Compares the rounding routine with the MPFR oracle, for every binary
//! format and every behavior: eight rounding directions, both tininess rules,
//! flush-to-zero, saturation, and precision control.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::env::Tininess;
use floaty::{BF16, Env, Exact, F16, F32, F64, F80, F128, TF32};
use floaty_verify::encodings::to_limbs;
use floaty_verify::mpfr::{self, Format, Input, Specials, Value};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// Inputs for each behavior of each format.
const INPUTS: usize = 1000;

/// Returns every behavior to test: each direction, tininess rule,
/// flush-to-zero setting, and saturation setting, at each precision limit.
fn behaviors(precisions: &[u32]) -> Vec<Env> {
    let limits: Vec<Option<NonZeroU32>> = core::iter::once(None)
        .chain(precisions.iter().map(|&bits| NonZeroU32::new(bits)))
        .collect();
    let mut envs = Vec::new();
    for &rounding in &mpfr::DIRECTIONS {
        for tininess in [Tininess::BeforeRounding, Tininess::AfterRounding] {
            for flush_to_zero in [false, true] {
                for saturate in [false, true] {
                    for &precision in &limits {
                        envs.push(
                            Env::IEEE
                                .with_rounding(rounding)
                                .with_tininess(tininess)
                                .with_flush_to_zero(flush_to_zero)
                                .with_saturate(saturate)
                                .with_precision(precision),
                        );
                    }
                }
            }
        }
    }
    envs
}

/// Returns a random integer of exactly `width` bits, with a random edge
/// pattern below the top bit.
fn significand(random: &mut SplitMix64, width: u32) -> Integer {
    let limbs: Vec<u64> = (0..width.div_ceil(64)).map(|_| random.next_u64()).collect();
    let low = Integer::from_digits(&limbs, Order::Lsf) & ((Integer::from(1) << (width - 1)) - 1u32);
    let pattern = match random.below(4) {
        0 => (Integer::from(1) << (width - 1)) - 1u32,
        1 => Integer::ZERO,
        2 if width >= 3 => {
            let position =
                u32::try_from(random.below(u64::from(width - 2))).expect("below the width");
            Integer::from(1) << position
        }
        _ => low,
    };
    (Integer::from(1) << (width - 1)) | pattern
}

/// Returns random inputs near every boundary of `format` at the precision of
/// `env`: overflow, the largest values, the normal range, the smallest normal
/// value, the subnormal range, and underflow to zero.
fn inputs(format: &Format, env: &Env, random: &mut SplitMix64) -> Vec<Input> {
    let precision = format.precision_in(env);
    let p = i32::try_from(precision).expect("a precision fits an i32");
    let tops = [
        format.emax + 2,
        format.emax + 1,
        format.emax,
        format.emax - 1,
        (format.emax + format.emin) / 2,
        format.emin + 1,
        format.emin,
        format.emin - 1,
        format.emin - p + 1,
        format.emin - p,
        format.emin - p - 1,
        format.emin - 2 * p - 3,
    ];
    let widths = [
        1,
        2,
        precision,
        precision + 1,
        precision + 2,
        format.precision + 2,
        format.precision + 3,
    ];
    (0..INPUTS)
        .map(|index| {
            let width = if index % 3 == 0 {
                u32::try_from(random.below(500)).expect("below 500") + 1
            } else {
                widths[index % widths.len()]
            };
            let top = tops[usize::try_from(random.below(12)).expect("below 12")];
            let exponent = top - i32::try_from(width).expect("a width fits an i32") + 1;
            let sticky = width >= precision + 2 && random.below(2) == 0;
            Input {
                negative: random.below(2) == 0,
                exponent,
                significand: significand(random, width),
                sticky,
            }
        })
        .collect()
}

/// Checks one format against the oracle.
macro_rules! check_format {
    ($alias:ty, $specials:expr, $precisions:expr, $seed:expr) => {{
        let format = Format::of::<$alias>($specials);
        let mut random = SplitMix64::new($seed);
        for env in behaviors($precisions) {
            for input in inputs(&format, &env, &mut random) {
                let exact = Exact::<8> {
                    negative: input.negative,
                    exponent: input.exponent,
                    significand: to_limbs(&input.significand),
                    sticky: input.sticky,
                };
                let (ours, flags) = <$alias>::round(exact, env);
                let context = format!("{} {env:?} {input:?}", stringify!($alias));
                assert!(ours.is_canonical(), "{context}: {ours:?} is canonical");
                let ours = (Value::from_decoded(ours.decode::<8>()), flags);
                assert_eq!(ours, mpfr::round(&input, &format, &env), "{context}");
            }
        }
    }};
}

#[test]
fn ieee_formats_up_to_binary128() {
    check_format!(F16, Specials::Ieee, &[4], 16);
    check_format!(BF16, Specials::Ieee, &[], 17);
    check_format!(TF32, Specials::Ieee, &[], 19);
    check_format!(F32, Specials::Ieee, &[11], 32);
    check_format!(F64, Specials::Ieee, &[24], 64);
    check_format!(F128, Specials::Ieee, &[], 128);
}

/// Returns the precision limits and the seed of a small format, by its alias.
///
/// # Panics
///
/// Panics for a format without an entry. A new format of the lists must get
/// an entry.
fn small_plan(alias: &str) -> (&'static [u32], u64) {
    match alias {
        "F8E4M3Fn" => (&[2], 43),
        "F8E5M2" => (&[], 52),
        "F8E4M3Fnuz" => (&[], 431),
        "F8E5M2Fnuz" => (&[], 521),
        "F8E4M3" => (&[2], 434),
        "F8E3M4" => (&[3], 345),
        "F8E4M3B11Fnuz" => (&[], 4311),
        "F4E2M1Fn" => (&[1], 44),
        "F6E2M3Fn" => (&[2], 62),
        "F6E3M2Fn" => (&[2], 63),
        other => panic!("{other} has no entry in small_plan"),
    }
}

/// Checks one format of the small format lists with the plan of
/// [`small_plan`].
macro_rules! small_format {
    ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
        let (precisions, seed) = small_plan(stringify!($alias));
        check_format!(floaty::$alias, $specials, precisions, seed);
    }};
}

#[test]
fn fp8_formats() {
    floaty_verify::for_each_fp8_format!(small_format);
}

#[test]
fn mx_formats() {
    floaty_verify::for_each_mx_format!(small_format);
}

#[test]
fn x87_with_precision_control() {
    check_format!(F80, Specials::Ieee, &[24, 53, 64], 80);
}

/// Returns the precision limits of a wide format, by its width. A format
/// without an entry runs without a precision limit.
fn wide_precisions(width: u32) -> &'static [u32] {
    match width {
        160 => &[64],
        256 => &[113],
        288 => &[200],
        384 => &[237],
        _ => &[],
    }
}

/// Checks one format of the wide format list, with its width as the seed.
macro_rules! wide_format {
    ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
        check_format!(
            floaty::$alias,
            Specials::Ieee,
            wide_precisions($width),
            $width
        )
    };
}

#[test]
fn wide_formats() {
    floaty_verify::for_each_wide_format!(wide_format);
}

/// Rounds every significand below 2^8 to each FP8 format, at every exponent
/// from below the subnormals to above the largest value, with both signs and
/// with and without a sticky bit, in every behavior. Run time: about 10
/// seconds in a release build.
#[test]
#[ignore = "exhaustive sweep; run with --ignored"]
fn every_small_input_to_the_fp8_formats() {
    macro_rules! sweep {
        ($alias:ty, $specials:expr) => {{
            let format = Format::of::<$alias>($specials);
            let p = i32::try_from(format.precision).expect("a precision fits an i32");
            for env in behaviors(&[1, 2, 3]) {
                let limit = format.precision_in(&env);
                for significand in 0_u64..256 {
                    let width = 64 - significand.leading_zeros();
                    for exponent in format.emin - 2 * p - 12..=format.emax + 3 {
                        for (negative, sticky) in
                            [(false, false), (true, false), (false, true), (true, true)]
                        {
                            if sticky && width < limit + 2 {
                                continue;
                            }
                            let input = Input {
                                negative,
                                exponent,
                                significand: Integer::from(significand),
                                sticky,
                            };
                            let exact = Exact::<1> {
                                negative,
                                exponent,
                                significand: [significand],
                                sticky,
                            };
                            let (ours, flags) = <$alias>::round(exact, env);
                            let ours = (Value::from_decoded(ours.decode::<1>()), flags);
                            let expected = mpfr::round(&input, &format, &env);
                            assert_eq!(ours, expected, "{} {env:?} {input:?}", stringify!($alias));
                        }
                    }
                }
            }
        }};
    }
    macro_rules! sweep_format {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            sweep!(floaty::$alias, $specials)
        };
    }
    floaty_verify::for_each_small_format!(sweep_format);
}
