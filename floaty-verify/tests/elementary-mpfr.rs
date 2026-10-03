//! Compares `exp`, `log`, and `compound` of the binary formats with the MPFR
//! oracles in `floaty_verify::operations`, in every behavior of
//! `operations::BEHAVIORS`.
//!
//! Every encoding of the FP8 and MX formats, binary16, and bfloat16 runs. The
//! other formats run their boundary encodings, random encodings, arguments
//! of `exp` in every binade from below `2^-(p + 3)` to the overflow
//! threshold, arguments next to the thresholds where `exp` overflows,
//! becomes tiny, and rounds to zero, and arguments next to 1 for `log`.
//! binary64 also runs the worst cases of Lefèvre and Muller.
//!
//! `compound` runs every exponent of [`COUNTS`] on each encoding of the FP8
//! and MX formats, and one of them on each other sample. It also runs every
//! exponent on arguments whose `1 + x` is a short odd multiple of a power of
//! two, which give exact powers, and on arguments next to -1. And it runs
//! arguments next to the thresholds of `(1 + x)^n` for a few exponents.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::format::Standard;
use floaty::{Binary, Env, F32, F64, F80, F128, Flags, Float, TF32};
use floaty_verify::encodings::{IntegerBit, Layout, boundary_encodings, to_limbs};
use floaty_verify::mpfr::{Format, Specials};
use floaty_verify::operations::BEHAVIORS;
use floaty_verify::operations::check;
use floaty_verify::random::SplitMix64;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

/// The exponents of the `compound` checks: the extremes of `i64`, both sides
/// of the limit of `i32`, of 64, and of the exact powers, and small ones.
const COUNTS: [i64; 24] = [
    i64::MIN,
    -(1 << 40) - 1,
    -(1 << 31),
    -1000,
    -65,
    -64,
    -17,
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3,
    5,
    17,
    64,
    65,
    1000,
    123_457,
    (1 << 31) - 1,
    1 << 31,
    (1 << 40) + 1,
    i64::MAX,
];

/// A `compound_with` of a format.
type Compound<S, const W: usize> = dyn Fn(Float<S, W>, i64, Env) -> (Float<S, W>, Flags);

/// Checks every encoding of a format of the small format lists.
macro_rules! every_small_encoding {
    ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
        let format = Format::of::<Float<$standard, $width>>($specials);
        for bits in 0..1_u16 << $width {
            let x = Float::<$standard, $width>::from_bits(
                u8::try_from(bits).expect("the format has at most 8 bits"),
            );
            for env in &BEHAVIORS {
                check::check_elementary(x, &format, env);
                check::check_compound(x, &COUNTS, &format, env, |x, n, env| {
                    x.compound_with(n, env)
                });
            }
        }
    }};
}

#[test]
fn every_small_format_encoding() {
    floaty_verify::for_each_small_format!(every_small_encoding);
}

/// Checks every encoding of a 16-bit format, and `compound` with a few
/// exponents.
fn every_16_bit_encoding<const E: u32>()
where
    Binary<E>: Standard<16, Bits = u16>,
{
    let format = Format::of::<Float<Binary<E>, 16>>(Specials::Ieee);
    for bits in 0..=u16::MAX {
        let x = Float::<Binary<E>, 16>::from_bits(bits);
        for env in &BEHAVIORS {
            check::check_elementary(x, &format, env);
            check::check_compound(x, &[-7, 3, 1000], &format, env, |x, n, env| {
                x.compound_with(n, env)
            });
        }
    }
}

#[test]
fn every_binary16_encoding() {
    every_16_bit_encoding::<5>();
}

#[test]
fn every_bfloat16_encoding() {
    every_16_bit_encoding::<8>();
}

/// Returns the encoding of a sign, an exponent field, and a significand. An
/// implicit integer bit drops the top bit of the significand. An explicit
/// integer bit is set exactly when the field is not zero.
fn assemble(layout: Layout, negative: bool, field: u64, significand: &Integer) -> Integer {
    let mut fraction = significand.clone().keep_bits(layout.fraction_bits());
    if layout.integer_bit == IntegerBit::Explicit {
        fraction.set_bit(layout.fraction_bits() - 1, field != 0);
    }
    let mut bits = (Integer::from(field) << layout.fraction_bits()) | fraction;
    bits.set_bit(layout.width - 1, negative);
    bits
}

/// Returns the encoding of a value of the precision of the layout, or `None`
/// when the value is not a normal number of the format.
fn encode(layout: Layout, value: &BigFloat) -> Option<Integer> {
    let exponent = value.get_exp()? - 1;
    let field = i64::from(exponent) + i64::from(layout.ieee_bias());
    if !(1..i64::from(layout.largest_field())).contains(&field) {
        return None;
    }
    let shift = exponent - i32::try_from(layout.precision() - 1).ok()?;
    let significand = (value.clone().abs() >> shift).to_integer()?;
    Some(assemble(
        layout,
        value.is_sign_negative(),
        u64::try_from(field).ok()?,
        &significand,
    ))
}

/// Returns the encodings of `value` rounded to the precision of the layout
/// and of the `steps` values on each side of it.
fn around(layout: Layout, value: &BigFloat, steps: usize) -> Vec<Integer> {
    let center = BigFloat::with_val(layout.precision(), value);
    let mut values = vec![center.clone()];
    let (mut up, mut down) = (center.clone(), center);
    for _ in 0..steps {
        up.next_up();
        down.next_down();
        values.extend([up.clone(), down.clone()]);
    }
    values
        .iter()
        .filter_map(|value| encode(layout, value))
        .collect()
}

/// Returns `count` random bits.
fn random_bits(random: &mut SplitMix64, count: u32) -> Integer {
    let digits: Vec<u64> = (0..count.div_ceil(64)).map(|_| random.next_u64()).collect();
    Integer::from_digits(&digits, Order::Lsf).keep_bits(count)
}

/// Returns `count` random encodings. One x87 encoding in four keeps its
/// random integer bit, which gives unnormals and pseudo-denormals.
fn random_encodings(layout: Layout, count: usize, random: &mut SplitMix64) -> Vec<Integer> {
    (0..count)
        .map(|index| {
            let bits = random_bits(random, layout.width);
            if layout.integer_bit == IntegerBit::Explicit && index % 4 != 0 {
                let field = (bits.clone() >> layout.fraction_bits())
                    .keep_bits(layout.exponent_bits)
                    .to_u64()
                    .expect("an exponent field has at most 23 bits");
                assemble(layout, bits.get_bit(layout.width - 1), field, &bits)
            } else {
                bits
            }
        })
        .collect()
}

/// Returns arguments of `exp` with four random significands and both signs,
/// in each binade from `2^-(p + 5)` to the binade past the overflow
/// threshold. Between the binades next to 1 and the tiny binades, one binade
/// in 16 runs.
fn exp_binades(layout: Layout, random: &mut SplitMix64) -> Vec<Integer> {
    let precision = i64::from(layout.precision());
    let emax = i64::from(layout.ieee_bias());
    // `ln 2 (emax + p) < 2^last`, past the arguments that underflow to zero.
    let last = i64::from(Integer::from(emax + precision).significant_bits());
    let field_max = i64::from(layout.largest_field());
    (-(precision + 5)..=last)
        .filter(|top| *top >= -8 || *top <= 4 - precision || top % 16 == 0)
        .map(|top| emax + top)
        .filter(|field| (1..field_max).contains(field))
        .flat_map(|field| {
            let field = u64::try_from(field).expect("the field is positive");
            let significands: Vec<Integer> = (0..4)
                .map(|_| random_bits(random, layout.precision()))
                .collect();
            significands.into_iter().flat_map(move |significand| {
                [false, true].map(|negative| assemble(layout, negative, field, &significand))
            })
        })
        .collect()
}

/// Returns arguments next to the thresholds of `exp`: where the result
/// overflows, where it becomes tiny, where it is the smallest subnormal, and
/// where it is half of it. It also returns arguments of `log` next to 1.
fn thresholds(layout: Layout) -> Vec<Integer> {
    let precision = layout.precision();
    let emax = layout.ieee_bias();
    let emin = 1 - emax;
    let working = precision + 64;
    let ulp = BigFloat::with_val(working, 1) >> (precision - 1);
    let largest = (BigFloat::with_val(working, 2) - ulp) << emax;
    let mut values = vec![BigFloat::with_val(working, largest.ln_ref())];
    let precision = i32::try_from(precision).expect("a precision fits an i32");
    for exponent in [emin, emin - precision + 1, emin - precision] {
        let power = BigFloat::with_val(working, 1) << exponent;
        values.push(BigFloat::with_val(working, power.ln_ref()));
    }
    values.push(BigFloat::with_val(working, 1));
    values
        .iter()
        .flat_map(|value| around(layout, value, 6))
        .collect()
}

/// Returns pairs of an argument and an exponent of `compound`: every
/// exponent of [`COUNTS`] with arguments whose `1 + x` is `m 2^-j` for an odd
/// `m` up to 15, and with arguments next to -1, and a few exponents with the
/// arguments next to the thresholds of `(1 + x)^n`: where it overflows,
/// becomes tiny, is the smallest subnormal, and is half of it.
fn compound_samples(layout: Layout) -> Vec<(Integer, i64)> {
    let precision = layout.precision();
    let emax = layout.ieee_bias();
    let emin = 1 - emax;
    let working = 2 * precision + 64;
    let mut arguments: Vec<BigFloat> = (1..=15_u32)
        .step_by(2)
        .flat_map(|m| (0..=4_u32).map(move |j| (BigFloat::with_val(working, m) >> j) - 1u32))
        .filter(|x| *x > -1 && !x.is_zero())
        .collect();
    arguments.extend((1..=3_u32).map(|k| (BigFloat::with_val(working, k) >> precision) - 1u32));
    let mut pairs: Vec<(Integer, i64)> = arguments
        .iter()
        .filter_map(|x| encode(layout, &BigFloat::with_val(precision, x)))
        .flat_map(|bits| COUNTS.map(|n| (bits.clone(), n)))
        .collect();
    let ulp = BigFloat::with_val(working, 1) >> (precision - 1);
    let largest = (BigFloat::with_val(working, 2) - ulp) << emax;
    let precision = i32::try_from(precision).expect("a precision fits an i32");
    let mut thresholds = vec![largest];
    thresholds.extend(
        [emin, emin - precision + 1, emin - precision]
            .map(|exponent| BigFloat::with_val(working, 1) << exponent),
    );
    for n in [3_i64, -3, 65, -65, 1000, -1000] {
        let root = u32::try_from(n.unsigned_abs()).expect("the exponent is small");
        for threshold in &thresholds {
            let root = BigFloat::with_val(working, threshold.root_ref(root));
            let base = if n > 0 { root } else { root.recip() };
            let x = base - 1u32;
            pairs.extend(around(layout, &x, 3).into_iter().map(|bits| (bits, n)));
        }
    }
    pairs
}

/// Checks `exp`, `log`, and `compound` on the samples of a format of the
/// IEEE layout, and on `extra` encodings. Each sample takes one exponent of
/// [`COUNTS`] in turn, and the pairs of [`compound_samples`] run too.
fn check_format<S: Standard<W>, const W: usize>(
    layout: Layout,
    make: &dyn Fn(&Integer) -> Float<S, W>,
    compound: &Compound<S, W>,
    count: usize,
    seed: u64,
    extra: &[Integer],
) {
    let format = Format::of::<Float<S, W>>(Specials::Ieee);
    let mut random = SplitMix64::new(seed);
    let mut encodings = boundary_encodings(layout);
    encodings.extend(random_encodings(layout, count, &mut random));
    encodings.extend(exp_binades(layout, &mut random));
    encodings.extend(thresholds(layout));
    encodings.extend_from_slice(extra);
    let pairs = compound_samples(layout);
    for env in &BEHAVIORS {
        for (index, bits) in encodings.iter().enumerate() {
            let x = make(bits);
            check::check_elementary(x, &format, env);
            let n = COUNTS[index % COUNTS.len()];
            check::check_compound(x, &[n], &format, env, compound);
        }
        for (bits, n) in &pairs {
            check::check_compound(make(bits), &[*n], &format, env, compound);
        }
    }
}

/// The worst cases of `exp` for binary64 of Lefèvre and Muller, "Some Worst
/// Cases for The Table Maker's Dilemma",
/// <https://perso.ens-lyon.fr/jean-michel.muller/TMD.html>, one for each
/// interval of the table.
const EXP_WORST_CASES: [u64; 33] = [
    0xC02E_8BDB_FCD9_144E,
    0xC017_1E0B_869B_5E79,
    0xC000_2393_D597_6769,
    0xBFF2_A9CA_D999_8262,
    0xBFFC_C37E_F7DE_7501,
    0xBFE2_2E24_FA3D_5CF9,
    0xBFED_C2B5_DF1F_7D3D,
    0xBFC2_90EA_09E3_6479,
    0xBF2A_2FEF_EFD5_80DF,
    0xBE4E_D318_EFB6_27EA,
    0xBE04_BD46_601A_E1EF,
    0xBD51_0000_0000_0242,
    0xBD52_0000_0000_0288,
    0xBCF8_0000_0000_0012,
    0xBCC0_0000_0000_0001,
    0x3CAF_FFFF_FFFF_FFFF,
    0x3CFF_FFFF_FFFF_FFE0,
    0x3DF7_FFE7_FFEE_0024,
    0x3DF8_0017_FFED_FFDC,
    0x3E09_E9CB_BFD6_080B,
    0x3E5D_7A7D_8936_09E5,
    0x3F1B_A07D_7325_0DE7,
    0x3F4D_77FD_13D2_7FFF,
    0x3F76_A4D1_AF9C_C989,
    0x3FEA_CCFB_E46B_4EF0,
    0x3FFA_CA7A_E8DA_5A7B,
    0x3FFD_6336_A880_77AA,
    0x4004_2EE3_C7DC_4946,
    0x4018_3D4B_CDEB_B3F4,
    0x4027_6E7E_5D7B_6EAC,
    0x402A_8EAD_058B_C6B8,
    0x4031_D5C2_DAEB_E367,
    0x403C_44CE_0D71_6A1A,
];

/// The worst cases of `log` for binary64 from the same table.
const LOG_WORST_CASES: [u64; 12] = [
    0x3FF0_0000_0000_0001,
    0x4000_0097_6581_CE4E,
    0x4019_1A8D_FF54_0FF7,
    0x401D_E37F_B31F_D5FC,
    0x4022_B119_9E49_7739,
    0x4025_5F0E_AA1B_2FC8,
    0x4028_EDE4_92D9_6072,
    0x4031_1867_637C_BD03,
    0x4033_D9D7_D597_A9DD,
    0x4037_F382_5778_AAAF,
    0x407A_C50B_409C_8AEE,
    0x40FD_E7CD_6751_029A,
];

#[test]
fn tf32_binary32_binary64_x87_and_binary128() {
    check_format(
        Layout::TF32,
        &|bits: &Integer| TF32::from_bits(bits.to_u32().expect("a TF32 encoding has 19 bits")),
        &|x, n, env| x.compound_with(n, env),
        4_000,
        0x0F19,
        &[],
    );
    check_format(
        Layout::BINARY32,
        &|bits: &Integer| F32::from_bits(bits.to_u32().expect("a binary32 encoding fits a u32")),
        &|x, n, env| x.compound_with(n, env),
        4_000,
        0x0F32,
        &[],
    );
    let worst: Vec<Integer> = EXP_WORST_CASES
        .iter()
        .chain(&LOG_WORST_CASES)
        .map(|&bits| Integer::from(bits))
        .collect();
    check_format(
        Layout::BINARY64,
        &|bits: &Integer| F64::from_bits(bits.to_u64().expect("a binary64 encoding fits a u64")),
        &|x, n, env| x.compound_with(n, env),
        3_000,
        0x0F64,
        &worst,
    );
    check_format(
        Layout::X87_EXTENDED,
        &|bits: &Integer| F80::from_bits(bits.to_u128().expect("an x87 encoding fits a u128")),
        &|x, n, env| x.compound_with(n, env),
        2_000,
        0x0F80,
        &[],
    );
    check_format(
        Layout::BINARY128,
        &|bits: &Integer| {
            F128::from_bits(bits.to_u128().expect("a binary128 encoding fits a u128"))
        },
        &|x, n, env| x.compound_with(n, env),
        1_500,
        0x0128,
        &[],
    );
}

/// Layouts whose precision fills the storage up to two bits, the least room
/// that a storage type keeps, and a 72-bit layout whose exponent field
/// crosses a limb boundary.
#[test]
fn layouts_that_fill_or_cross_limbs() {
    check_format(
        Layout::ieee(64, 2),
        &|bits: &Integer| {
            Float::<Binary<2>, 64>::from_bits(bits.to_u64().expect("the encoding fits a u64"))
        },
        &|x, n, env| x.compound_with(n, env),
        1_000,
        0x0264,
        &[],
    );
    check_format(
        Layout::ieee(128, 2),
        &|bits: &Integer| {
            Float::<Binary<2>, 128>::from_bits(bits.to_u128().expect("the encoding fits a u128"))
        },
        &|x, n, env| x.compound_with(n, env),
        1_000,
        0x0228,
        &[],
    );
    check_format(
        Layout::ieee(72, 15),
        &|bits: &Integer| {
            Float::<Binary<15>, 72>::from_bits(bits.to_u128().expect("the encoding fits a u128"))
        },
        &|x, n, env| x.compound_with(n, env),
        1_000,
        0x0072,
        &[],
    );
}

/// Checks one format of the wide format list.
macro_rules! wide_format {
    ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
        check_format(
            Layout::ieee($width, $exponent_bits),
            &|bits: &Integer| floaty::$alias::from_bits(to_limbs::<$limbs>(bits)),
            &|x, n, env| x.compound_with(n, env),
            200,
            $width,
            &[],
        )
    };
}

#[test]
fn every_wide_format() {
    floaty_verify::for_each_wide_format!(wide_format);
}
