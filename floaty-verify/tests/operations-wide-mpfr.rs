//! Compares the operations beyond arithmetic with the oracle in
//! `floaty_verify::operations`, for the wider formats: binary16, bfloat16,
//! TF32, binary32, binary64, a 72-bit layout whose exponent field crosses a
//! limb boundary, x87 extended precision with precision control, binary128,
//! and every wide format from binary160 to binary512. TestFloat and
//! `rustc_apfloat` check some of these operations for the IEEE formats in
//! their own behaviors. This test runs them in every behavior of
//! `BEHAVIORS`. It also converts random integers to every format, the FP8
//! and MX formats included.
//!
//! The operands are the boundary encodings of each format, random encodings,
//! values near the integers that rounding and integer conversion reach, and
//! remainder pairs with close exponents. Every pair of boundary encodings
//! also gives the remainder at the extreme exponent differences.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::format::Standard;
use floaty::{Binary, Decoded, Env, F32, F64, F80, F128, F256, F512, Float, Int, TF32, UInt, X87};
use floaty_verify::encodings::{IntegerBit, Layout, boundary_encodings, to_limbs};
use floaty_verify::mpfr::{DIRECTIONS, Format, Specials};
use floaty_verify::operations::BEHAVIORS;
use floaty_verify::operations::check::{self, Case};
use floaty_verify::operations::integral::IntegerValue;
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// A 72-bit layout whose exponent field crosses a limb boundary.
type Wide72 = Float<Binary<15>, 72>;

/// Returns the exponent bias of a layout.
fn bias(layout: Layout) -> u64 {
    u64::try_from(layout.ieee_bias()).expect("an exponent bias is positive")
}

/// Returns the exponent field of an encoding.
fn field(layout: Layout, encoding: &Integer) -> u64 {
    (encoding.clone() >> layout.fraction_bits())
        .keep_bits(layout.exponent_bits)
        .to_u64()
        .expect("an exponent field has at most 23 bits")
}

/// Returns the encoding of a sign, an exponent field, and a fraction. An
/// explicit integer bit is set exactly when the field is not zero, so the
/// encoding is canonical.
fn assemble(layout: Layout, negative: bool, field: u64, fraction: &Integer) -> Integer {
    let mut fraction = fraction.clone().keep_bits(layout.fraction_bits());
    if layout.integer_bit == IntegerBit::Explicit {
        fraction.set_bit(layout.fraction_bits() - 1, field != 0);
    }
    let mut bits = (Integer::from(field) << layout.fraction_bits()) | fraction;
    bits.set_bit(layout.width - 1, negative);
    bits
}

/// Returns `count` random bits.
fn random_bits(random: &mut SplitMix64, count: u32) -> Integer {
    let digits: Vec<u64> = (0..count.div_ceil(64)).map(|_| random.next_u64()).collect();
    Integer::from_digits(&digits, Order::Lsf).keep_bits(count)
}

/// Returns a fraction with random top bits and zero below them. Two such
/// values often divide to a tie, and round to an integer at a tie.
fn short_fraction(layout: Layout, random: &mut SplitMix64) -> Integer {
    let keep = 3.min(layout.fraction_bits());
    random_bits(random, keep) << (layout.fraction_bits() - keep)
}

/// Returns the encodings of one sign for a list of exponent fields, each with
/// a zero, an all-ones, a random, and a short fraction.
fn with_fractions(
    layout: Layout,
    fields: impl Iterator<Item = u64>,
    random: &mut SplitMix64,
) -> Vec<Integer> {
    let ones = (Integer::from(1) << layout.fraction_bits()) - 1u32;
    let mut encodings = Vec::new();
    for field in fields {
        let fractions = [
            Integer::ZERO,
            ones.clone(),
            random_bits(random, layout.fraction_bits()),
            short_fraction(layout, random),
        ];
        for fraction in &fractions {
            for negative in [false, true] {
                encodings.push(assemble(layout, negative, field, fraction));
            }
        }
    }
    encodings
}

/// Returns the operand encodings of a format: the boundary encodings,
/// `count` random encodings, and values near the integers.
fn samples(layout: Layout, count: usize, random: &mut SplitMix64) -> Vec<Integer> {
    let mut samples = boundary_encodings(layout);
    for index in 0..count {
        let bits = random_bits(random, layout.width);
        // One x87 encoding in four keeps its random integer bit, which gives
        // unnormals and pseudo-denormals.
        if layout.integer_bit == IntegerBit::Explicit && index % 4 == 0 {
            samples.push(bits);
        } else {
            let negative = bits.get_bit(layout.width - 1);
            samples.push(assemble(layout, negative, field(layout, &bits), &bits));
        }
    }
    // Values from below one to past the precision, and past the width of each
    // integer type of the conversions.
    let precision = i64::from(layout.precision());
    let mut tops: Vec<i64> = (-3..=precision + 1)
        .filter(|top| *top <= 8 || *top >= precision - 3 || top % 17 == 0)
        .collect();
    tops.extend([
        22, 23, 24, 98, 99, 100, 126, 127, 128, 198, 199, 200, 510, 511, 512,
    ]);
    let (bias, field_max) = (bias(layout), u64::from(layout.largest_field()));
    let fields = tops
        .into_iter()
        .filter_map(|top| bias.checked_add_signed(top))
        .filter(|field| (1..field_max).contains(field));
    samples.extend(with_fractions(layout, fields, random));
    samples
}

/// Returns `count` remainder pairs. Three in four have exponent fields that
/// differ by less than the precision plus 4, which gives quotients of every
/// size near 1, and one in four has fields anywhere in the range.
fn remainder_pairs(
    layout: Layout,
    count: usize,
    random: &mut SplitMix64,
) -> Vec<(Integer, Integer)> {
    let span = i64::from(layout.precision()) + 4;
    let field_max = i64::from(layout.largest_field());
    let modulus = u64::try_from(field_max).expect("a field is positive");
    (0..count)
        .map(|index| {
            let field = i64::try_from(random.below(modulus))
                .expect("a value below the field range fits an i64");
            let difference = if index % 4 == 0 {
                field
                    - i64::try_from(random.below(modulus))
                        .expect("a value below the field range fits an i64")
            } else {
                i64::try_from(
                    random.next_u64()
                        % u64::try_from(2 * span + 1)
                            .expect("the span of the test exponents is positive"),
                )
                .expect("the span of the test exponents fits an i64")
                    - span
            };
            let other = (field - difference).clamp(0, field_max - 1);
            let short = index % 3 == 0;
            let mut encoding = |field: i64| {
                let fraction = if short {
                    short_fraction(layout, random)
                } else {
                    random_bits(random, layout.fraction_bits())
                };
                let negative = random.below(2) == 0;
                let field = u64::try_from(field).expect("a field is not negative");
                assemble(layout, negative, field, &fraction)
            };
            (encoding(field), encoding(other))
        })
        .collect()
}

/// Returns the scales of `scale_b` for one value: small scales, the
/// extremes, and the scales that move the value to the overflow threshold
/// and through the subnormal range.
fn scales<S: Standard<W>, const W: usize>(x: &Case<S, W>, format: &Format) -> Vec<i32> {
    let mut scales = vec![0, 1, -1, 3, i32::MIN, -(1 << 30) - 1, 1 << 30, i32::MAX];
    if let Decoded::Finite {
        exponent,
        significand,
        ..
    } = x.sample.operand.decoded
    {
        let width = Integer::from_digits(&significand, Order::Lsf).significant_bits();
        let top = exponent + i32::try_from(width).expect("a width fits an i32") - 1;
        let p = i32::try_from(format.precision).expect("a precision fits an i32");
        let (high, low) = (format.emax - top, format.emin - top);
        scales.extend([high - 1, high, high + 1, low + 1, low, low - 1]);
        scales.extend([low - p + 2, low - p + 1, low - p, low - p - 1]);
    }
    scales
}

/// Checks `to_int_with` into every integer type of the wide tests.
fn to_ints<S: Standard<W>, const W: usize>(x: &Case<S, W>, env: &Env) {
    check::check_to_int::<Int<24>, S, W>(x, env);
    check::check_to_int::<UInt<24>, S, W>(x, env);
    check::check_to_int::<Int<100>, S, W>(x, env);
    check::check_to_int::<UInt<200>, S, W>(x, env);
    check::check_to_int::<Int<512>, S, W>(x, env);
    check::check_to_int::<UInt<512>, S, W>(x, env);
    check::check_to_int::<i128, S, W>(x, env);
    check::check_to_int::<u128, S, W>(x, env);
}

/// What one run of a format checks.
struct Plan<'a, S: Standard<W>, const W: usize> {
    layout: Layout,
    /// The value of an encoding.
    make: &'a dyn Fn(&Integer) -> Float<S, W>,
    /// The number of random encodings and of remainder pairs.
    count: usize,
    /// The behaviors of the operations on pairs.
    pair_envs: &'a [Env],
    /// The behaviors of the operations on one operand.
    single_envs: &'a [Env],
    seed: u64,
}

/// Checks every operation beyond arithmetic on the samples of a format.
fn check_format<S: Standard<W>, const W: usize>(plan: &Plan<'_, S, W>) {
    let layout = plan.layout;
    let format = Format::of::<Float<S, W>>(Specials::Ieee);
    let mut random = SplitMix64::new(plan.seed);
    let case = |bits: &Integer| Case::new(bits.clone(), layout, (plan.make)(bits));
    let boundaries: Vec<Case<S, W>> = boundary_encodings(layout).iter().map(case).collect();
    let samples: Vec<Case<S, W>> = samples(layout, plan.count, &mut random)
        .iter()
        .map(case)
        .collect();
    let mut pairs: Vec<(&Case<S, W>, &Case<S, W>)> = boundaries
        .iter()
        .flat_map(|x| boundaries.iter().map(move |y| (x, y)))
        .collect();
    pairs.extend(samples.chunks_exact(2).map(|pair| (&pair[0], &pair[1])));
    let near: Vec<(Case<S, W>, Case<S, W>)> = remainder_pairs(layout, plan.count, &mut random)
        .iter()
        .map(|(x, y)| (case(x), case(y)))
        .collect();
    for &(x, y) in &pairs {
        check::check_order_and_signs(x, y, Specials::Ieee, plan.make);
        check::check_default_mode(x, y, &format);
    }
    for env in plan.pair_envs {
        for &(x, y) in &pairs {
            check::check_comparisons(x, y, env);
            check::check_min_max(x, y, &format, env);
            check::check_remainder(x, y, &format, env);
        }
        for (x, y) in &near {
            check::check_remainder(x, y, &format, env);
        }
    }
    for env in plan.single_envs {
        for x in &samples {
            check::check_integral_and_next(x, &format, env);
            check::check_scale_b(x, &scales(x, &format), &format, env);
            to_ints(x, env);
        }
    }
}

/// Returns the value of an encoding of a format stored in a `u16`.
fn from_u16<S: Standard<W, Bits = u16>, const W: usize>(bits: &Integer) -> Float<S, W> {
    Float::from_bits(bits.to_u16().expect("the encoding fits 16 bits"))
}

#[test]
fn bfloat16_and_tf32() {
    check_format(&Plan {
        layout: Layout::BFLOAT16,
        make: &from_u16::<Binary<8>, 16>,
        count: 4_000,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0B16,
    });
    check_format(&Plan {
        layout: Layout::TF32,
        make: &|bits: &Integer| {
            TF32::from_bits(bits.to_u32().expect("a TF32 encoding has 19 bits"))
        },
        count: 4_000,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0F32,
    });
}

#[test]
fn binary16_binary32_binary64_and_binary128() {
    check_format(&Plan {
        layout: Layout::BINARY16,
        make: &from_u16::<Binary<5>, 16>,
        count: 4_000,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0F16,
    });
    check_format(&Plan {
        layout: Layout::BINARY32,
        make: &|bits: &Integer| {
            F32::from_bits(bits.to_u32().expect("a binary32 encoding fits a u32"))
        },
        count: 4_000,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x1F32,
    });
    check_format(&Plan {
        layout: Layout::BINARY64,
        make: &|bits: &Integer| {
            F64::from_bits(bits.to_u64().expect("a binary64 encoding fits a u64"))
        },
        count: 3_000,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0F64,
    });
    check_format(&Plan {
        layout: Layout::BINARY128,
        make: &|bits: &Integer| {
            F128::from_bits(bits.to_u128().expect("a binary128 encoding fits a u128"))
        },
        count: 2_000,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0128,
    });
}

#[test]
fn a_layout_whose_exponent_crosses_a_limb() {
    check_format(&Plan {
        layout: Layout::ieee(72, 15),
        make: &|bits: &Integer| {
            Wide72::from_bits(bits.to_u128().expect("a 72-bit encoding fits a u128"))
        },
        count: 3_000,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0072,
    });
}

/// The behaviors of x87 precision control: 24 and 53 bits in every
/// direction. The precision limit applies to `scale_b` and `from_int`, and
/// not to the other operations.
fn precision_control() -> Vec<Env> {
    [24, 53]
        .into_iter()
        .flat_map(|bits| {
            DIRECTIONS.map(|rounding| {
                Env::IEEE
                    .with_rounding(rounding)
                    .with_precision(NonZeroU32::new(bits))
            })
        })
        .collect()
}

#[test]
fn x87_extended_with_precision_control() {
    let limited = precision_control();
    let mut pair_envs = BEHAVIORS.to_vec();
    pair_envs.extend([limited[0], limited[10]]);
    let mut single_envs = BEHAVIORS.to_vec();
    single_envs.extend(limited);
    check_format(&Plan {
        layout: Layout::X87_EXTENDED,
        make: &|bits: &Integer| {
            F80::from_bits(bits.to_u128().expect("an x87 encoding fits a u128"))
        },
        count: 3_000,
        pair_envs: &pair_envs,
        single_envs: &single_envs,
        seed: 0x0080,
    });
}

#[test]
fn binary256_and_binary512() {
    check_format(&Plan {
        layout: Layout::BINARY256,
        make: &|bits: &Integer| F256::from_bits(to_limbs::<4>(bits)),
        count: 1_500,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0256,
    });
    check_format(&Plan {
        layout: Layout::BINARY512,
        make: &|bits: &Integer| F512::from_bits(to_limbs::<8>(bits)),
        count: 1_500,
        pair_envs: &BEHAVIORS,
        single_envs: &BEHAVIORS,
        seed: 0x0512,
    });
}

/// Checks one format of the wide format list, but binary256 and binary512,
/// which `binary256_and_binary512` checks with more operands.
macro_rules! other_wide_format {
    ($alias:ident, 256, $exponent_bits:literal, $limbs:literal) => {};
    ($alias:ident, 512, $exponent_bits:literal, $limbs:literal) => {};
    ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
        check_format(&Plan {
            layout: Layout::ieee($width, $exponent_bits),
            make: &|bits: &Integer| floaty::$alias::from_bits(to_limbs::<$limbs>(bits)),
            count: 500,
            pair_envs: &BEHAVIORS,
            single_envs: &BEHAVIORS,
            seed: $width,
        })
    };
}

#[test]
fn the_other_wide_formats() {
    floaty_verify::for_each_wide_format!(other_wide_format);
}

/// Returns edge and random integers of type `I`: zero, one, the limits, the
/// powers of two near each precision, and `count` random values of random
/// widths.
fn integers<I: IntegerValue>(count: usize, random: &mut SplitMix64) -> Vec<I> {
    let bits = I::width();
    let (low, high) = I::range();
    let mut values = vec![
        Integer::ZERO,
        Integer::from(1),
        Integer::from(-1),
        low.clone(),
        low.clone() + 1u32,
        high.clone(),
        high.clone() - 1u32,
    ];
    for power in [2_u32, 3, 4, 8, 11, 24, 53, 57, 64, 113, 237, 489] {
        let two: Integer = Integer::from(1) << power;
        for delta in [-1_i32, 0, 1] {
            let near: Integer = two.clone() + delta;
            values.push(-near.clone());
            values.push(near);
        }
    }
    for _ in 0..count {
        let width = u32::try_from(random.below(u64::from(bits)))
            .expect("a width below the integer width fits a u32")
            + 1;
        let mut value = random_bits(random, width);
        if I::signed() && random.below(2) == 0 {
            value = -value;
        }
        values.push(value);
    }
    values
        .into_iter()
        .filter(|value| *value >= low && *value <= high)
        .map(|value| I::from_integer(&value))
        .collect()
}

/// Converts the integers of every test type to a format in every behavior.
fn from_ints<S: Standard<W>, const W: usize>(specials: Specials, envs: &[Env], seed: u64) {
    let format = Format::of::<Float<S, W>>(specials);
    let mut random = SplitMix64::new(seed);
    let small: Vec<Int<24>> = integers(1_000, &mut random);
    let middle: Vec<Int<200>> = integers(1_000, &mut random);
    let large: Vec<UInt<512>> = integers(1_000, &mut random);
    let signed_large: Vec<Int<512>> = integers(1_000, &mut random);
    let primitive: Vec<i128> = integers(1_000, &mut random);
    for env in envs {
        for &n in &small {
            check::check_from_int::<_, S, W>(n, &format, env);
        }
        for &n in &middle {
            check::check_from_int::<_, S, W>(n, &format, env);
        }
        for &n in &large {
            check::check_from_int::<_, S, W>(n, &format, env);
        }
        for &n in &signed_large {
            check::check_from_int::<_, S, W>(n, &format, env);
        }
        for &n in &primitive {
            check::check_from_int::<_, S, W>(n, &format, env);
        }
    }
}

/// Converts the integers to one format of the small format lists.
macro_rules! small_from_ints {
    ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
        from_ints::<$standard, $width>($specials, &BEHAVIORS, $seed)
    };
}

/// Converts the integers to one format of the wide format list, with its
/// width as the seed.
macro_rules! wide_from_ints {
    ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
        from_ints::<Binary<$exponent_bits>, $width>(Specials::Ieee, &BEHAVIORS, $width)
    };
}

#[test]
fn integers_convert_to_every_format() {
    floaty_verify::for_each_small_format!(small_from_ints);
    from_ints::<Binary<5>, 16>(Specials::Ieee, &BEHAVIORS, 0x0F16);
    from_ints::<Binary<8>, 16>(Specials::Ieee, &BEHAVIORS, 16);
    from_ints::<Binary<8>, 19>(Specials::Ieee, &BEHAVIORS, 19);
    from_ints::<Binary<8>, 32>(Specials::Ieee, &BEHAVIORS, 32);
    from_ints::<Binary<11>, 64>(Specials::Ieee, &BEHAVIORS, 64);
    from_ints::<Binary<15>, 72>(Specials::Ieee, &BEHAVIORS, 72);
    from_ints::<Binary<15>, 128>(Specials::Ieee, &BEHAVIORS, 128);
    let mut x87 = BEHAVIORS.to_vec();
    x87.extend(precision_control());
    from_ints::<Binary<15, X87>, 80>(Specials::Ieee, &x87, 80);
    floaty_verify::for_each_wide_format!(wide_from_ints);
}
