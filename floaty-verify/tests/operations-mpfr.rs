//! Compares the operations beyond arithmetic with the oracle in
//! `floaty_verify::operations`, for every operand pair and every operand of
//! the FP8 formats and the MX formats FP4 and FP6, in every behavior of
//! `operations::BEHAVIORS`.
//!
//! The pair operations are the comparisons, the total order, the sign
//! operations, the six minimum and maximum operations, and the remainder. The
//! operations on one operand are rounding to an integral value, conversion to
//! seven integer types, scaling, and the next value up and down. Three 8-bit
//! layouts with a small exponent range also run, because a rounded integer
//! can overflow them.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::format::Standard;
use floaty::{B11Fnuz, Binary, Finite, Float, Fnuz, Int, NoInf, UInt};
use floaty_verify::mpfr::Specials;
use floaty_verify::operations::BEHAVIORS;
use floaty_verify::operations::check::{self, Case, format};
use rug::Integer;

/// Returns a case for every encoding of a format of at most 8 bits.
fn cases<S: Standard<W, Bits = u8>, const W: usize>() -> Vec<Case<S, W>> {
    (0..1_u16 << W)
        .map(|bits| {
            let bits = u8::try_from(bits).expect("the format has at most 8 bits");
            Case::new(Integer::from(bits), Float::from_bits(bits))
        })
        .collect()
}

/// Returns the value of an encoding of at most 8 bits.
fn make<S: Standard<W, Bits = u8>, const W: usize>(bits: &Integer) -> Float<S, W> {
    Float::from_bits(bits.to_u8().expect("the format has at most 8 bits"))
}

/// Runs a check for each FP8 format and each MX format of 4 and 6 bits.
macro_rules! each_format {
    ($check:ident) => {
        $check::<Binary<4, NoInf>, 8>(Specials::NoInf);
        $check::<Binary<5>, 8>(Specials::Ieee);
        $check::<Binary<4, Fnuz>, 8>(Specials::Fnuz);
        $check::<Binary<5, Fnuz>, 8>(Specials::Fnuz);
        $check::<Binary<4>, 8>(Specials::Ieee);
        $check::<Binary<3>, 8>(Specials::Ieee);
        $check::<Binary<4, B11Fnuz>, 8>(Specials::Fnuz);
        $check::<Binary<2, Finite>, 4>(Specials::Finite);
        $check::<Binary<2, Finite>, 6>(Specials::Finite);
        $check::<Binary<3, Finite>, 6>(Specials::Finite);
    };
}

fn comparisons<S: Standard<W, Bits = u8>, const W: usize>(specials: Specials) {
    let format = format::<S, W>(specials);
    let cases = cases::<S, W>();
    for x in &cases {
        for y in &cases {
            check::check_order_and_signs(x, y, specials, &make::<S, W>);
            check::check_default_mode(x, y, &format);
            for env in &BEHAVIORS {
                check::check_comparisons(x, y, env);
            }
        }
    }
}

#[test]
fn every_small_format_pair_compares_orders_and_copies_signs() {
    each_format!(comparisons);
}

fn min_max<S: Standard<W, Bits = u8>, const W: usize>(specials: Specials) {
    let format = format::<S, W>(specials);
    let cases = cases::<S, W>();
    for env in &BEHAVIORS {
        for x in &cases {
            for y in &cases {
                check::check_min_max(x, y, &format, env);
            }
        }
    }
}

#[test]
fn every_small_format_pair_minimum_and_maximum() {
    each_format!(min_max);
}

fn remainder<S: Standard<W, Bits = u8>, const W: usize>(specials: Specials) {
    let format = format::<S, W>(specials);
    let cases = cases::<S, W>();
    for env in &BEHAVIORS {
        for x in &cases {
            for y in &cases {
                check::check_remainder(x, y, &format, env);
            }
        }
    }
}

#[test]
fn every_small_format_pair_remainder() {
    each_format!(remainder);
}

/// The scales of `scale_b`: every scale that moves an FP8 value across its
/// whole range, and the extremes, where floaty clamps at `2^30`.
fn scales() -> Vec<i32> {
    let mut scales: Vec<i32> = (-40..=40).collect();
    scales.extend([
        i32::MIN,
        -(1 << 30) - 1,
        -(1 << 30),
        1 << 30,
        (1 << 30) + 1,
        i32::MAX,
    ]);
    scales
}

fn one_operand<S: Standard<W, Bits = u8>, const W: usize>(specials: Specials) {
    let format = format::<S, W>(specials);
    let scales = scales();
    for env in &BEHAVIORS {
        for x in &cases::<S, W>() {
            check::check_integral_and_next(x, &format, env);
            check::check_scale_b(x, &scales, &format, env);
            check::check_to_int::<i8, S, W>(x, env);
            check::check_to_int::<u8, S, W>(x, env);
            check::check_to_int::<Int<3>, S, W>(x, env);
            check::check_to_int::<UInt<3>, S, W>(x, env);
            check::check_to_int::<Int<5>, S, W>(x, env);
            check::check_to_int::<Int<1>, S, W>(x, env);
            check::check_to_int::<UInt<1>, S, W>(x, env);
        }
    }
}

#[test]
fn every_small_format_operand_rounds_converts_scales_and_steps() {
    each_format!(one_operand);
}

/// Runs a check for 8-bit layouts whose largest finite value is below
/// `2^(p - 1)`, so that a rounded integer can overflow: `Binary<3>` is the
/// `ml_dtypes` E3M4 layout.
macro_rules! each_small_range_layout {
    ($check:ident) => {
        $check::<Binary<3>, 8>(Specials::Ieee);
        $check::<Binary<2>, 8>(Specials::Ieee);
        $check::<Binary<2, NoInf>, 8>(Specials::NoInf);
    };
}

#[test]
fn small_range_layouts_round_to_integers_that_overflow() {
    each_small_range_layout!(one_operand);
    each_small_range_layout!(remainder);
    each_small_range_layout!(min_max);
}
