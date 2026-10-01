//! Compares the reduction operations of IEEE 754-2019 section 9.4 with the
//! oracle in `floaty_verify::operations::reduction`: `sum`, `sumAbs`,
//! `sumSquare`, `dot`, and the three scaled products. The formats are FP8
//! E4M3FN and E5M2, binary16, bfloat16, binary32, binary64, x87 extended
//! precision, binary128, binary256, and binary512, in every behavior of
//! `BEHAVIORS`.
//!
//! The vectors are random vectors of the boundary encodings and random
//! encodings, vectors whose large terms cancel, vectors whose sum lies on a
//! rounding boundary that a far smaller term decides, and long vectors.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::format::{Encoding, Standard, Storage, Width};
use floaty::{Binary, Env, F8E4M3Fn, F8E5M2, F16, F32, F64, F80, F128, F256, F512, Float};
use floaty_verify::encodings::{IntegerBit, Layout, boundary_encodings, to_limbs};
use floaty_verify::mpfr::{Format, Operand, Specials};
use floaty_verify::operations::reduction::{self, Product, Summand};
use floaty_verify::operations::{BEHAVIORS, outcome};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// A binary format of the tests.
type Value<const E: u32, Enc, const W: usize> = Float<Binary<E, Enc>, W>;

/// Returns `count` random bits.
fn random_bits(random: &mut SplitMix64, count: u32) -> Integer {
    let digits: Vec<u64> = (0..count.div_ceil(64)).map(|_| random.next_u64()).collect();
    Integer::from_digits(&digits, Order::Lsf).keep_bits(count)
}

/// Returns the encoding of a finite value: a sign, an exponent field, and a
/// fraction. An x87 encoding gets the integer bit of its field.
fn assemble(layout: Layout, negative: bool, field: u64, fraction: &Integer) -> Integer {
    let mut fraction = Integer::from(fraction.keep_bits_ref(layout.fraction_bits()));
    if layout.integer_bit == IntegerBit::Explicit {
        fraction.set_bit(layout.fraction_bits() - 1, field != 0);
    }
    let mut bits = (Integer::from(field) << layout.fraction_bits()) | fraction;
    bits.set_bit(layout.width - 1, negative);
    bits
}

/// Returns the encoding of `2^power`, a normal value of the format.
fn power_of_two(layout: Layout, power: i64) -> Integer {
    let field = i64::from(layout.ieee_bias()) + power;
    let field = u64::try_from(field).expect("the power is normal");
    assemble(layout, false, field, &Integer::ZERO)
}

/// Returns an encoding with the sign bit set.
fn negated(layout: Layout, encoding: &Integer) -> Integer {
    let mut negated = encoding.clone();
    negated.set_bit(layout.width - 1, !encoding.get_bit(layout.width - 1));
    negated
}

/// Returns the vectors of a format, as encodings.
fn vectors(layout: Layout, random: &mut SplitMix64) -> Vec<Vec<Integer>> {
    let mut pool = boundary_encodings(layout);
    let field_max = u64::from(layout.largest_field());
    pool.extend((0..64).map(|_| random_bits(random, layout.width)));
    let finite = |random: &mut SplitMix64| {
        let field = 1 + random.below(field_max - 1);
        let fraction = random_bits(random, layout.fraction_bits());
        assemble(layout, random.below(2) == 0, field, &fraction)
    };
    let mut vectors = vec![Vec::new()];
    for length in 1..=8_u64 {
        for _ in 0..40 {
            let pick = |random: &mut SplitMix64| {
                let index =
                    usize::try_from(random.below(pool.len() as u64)).expect("an index fits");
                pool[index].clone()
            };
            vectors.push((0..length).map(|_| pick(random)).collect());
        }
    }
    // Large terms that cancel, around a term of any size.
    for _ in 0..40 {
        let large = finite(random);
        let middle = finite(random);
        vectors.push(vec![large.clone(), middle, negated(layout, &large)]);
    }
    // 1 + 2^-p is a tie between 1 and its neighbor above. A far smaller
    // term of either sign decides it, and the tie of an odd neighbor too.
    let precision = i64::from(layout.precision());
    let one = power_of_two(layout, 0);
    let half = power_of_two(layout, -precision);
    let tiny = assemble(layout, false, 0, &Integer::from(1));
    let odd = assemble(
        layout,
        false,
        u64::try_from(layout.ieee_bias()).expect("a bias fits"),
        &Integer::from(1),
    );
    for base in [one, odd] {
        for tail in [Integer::ZERO, tiny.clone(), negated(layout, &tiny)] {
            vectors.push(vec![base.clone(), half.clone(), tail]);
        }
    }
    // The first window of a sum near 1 has its low end at 2^-702, and the
    // windows after a rounding boundary at 2^-1403. Terms below a window
    // add up past the visible bits near a tie: back to the tie, below it in
    // the second window, and past a cancellation of large terms.
    let power = |power: i64| {
        let field = i64::from(layout.ieee_bias()) + power;
        (1..i64::try_from(field_max).expect("a field fits"))
            .contains(&field)
            .then(|| power_of_two(layout, power))
    };
    let repeat = |value: &Integer, count: usize| vec![negated(layout, value); count];
    if let (Some(visible), Some(below)) = (power(-700), power(-703)) {
        for count in [8, 9] {
            let mut vector = vec![power_of_two(layout, 0), half.clone(), visible.clone()];
            vector.extend(repeat(&below, count));
            vectors.push(vector);
        }
    }
    if let (Some(visible), Some(below)) = (power(-1400), power(-1404)) {
        let mut vector = vec![power_of_two(layout, 0), half.clone(), visible];
        vector.extend(repeat(&below, 17));
        vectors.push(vector);
    }
    if let (Some(large), Some(below)) = (power(1000), power(297)) {
        let mut vector = vec![large.clone(), negated(layout, &large)];
        vector.extend(vec![below; 4]);
        vectors.push(vector);
    }
    // Random finite values with their negations at random places.
    for _ in 0..40 {
        let length = 2 + random.below(30);
        let mut vector: Vec<Integer> = (0..length).map(|_| finite(random)).collect();
        for index in 0..vector.len() / 2 {
            if random.below(2) == 0 {
                let target =
                    usize::try_from(random.below(vector.len() as u64)).expect("an index fits");
                vector[target] = negated(layout, &vector[index]);
            }
        }
        vectors.push(vector);
    }
    // Long vectors that overflow, and that cancel to a far smaller value.
    let ones = (Integer::from(1) << layout.fraction_bits()) - 1u32;
    let largest = assemble(layout, false, field_max - 1, &ones);
    vectors.push(vec![largest.clone(); 300]);
    let mut alternating: Vec<Integer> = (0..300)
        .map(|index| {
            if index % 2 == 0 {
                largest.clone()
            } else {
                negated(layout, &largest)
            }
        })
        .collect();
    alternating.push(tiny);
    vectors.push(alternating);
    vectors
}

/// Checks every reduction on one vector against the oracle.
fn check_vector<const E: u32, Enc: Encoding, const W: usize>(
    values: &[Value<E, Enc, W>],
    format: &Format,
    env: &Env,
) where
    Width<W>: Storage,
    Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    let operands: Vec<Operand<8>> = values.iter().map(|&value| Operand::of(value)).collect();
    for summand in Summand::ALL {
        let (result, flags) = summand.apply(values, *env);
        assert!(result.is_canonical(), "{summand:?} {values:?} {env:?}");
        assert_eq!(
            (outcome(result), flags),
            reduction::sum(&operands, summand, format, env),
            "{summand:?} {values:?} {env:?}"
        );
    }
    let pairs: Vec<(Value<E, Enc, W>, Value<E, Enc, W>)> = values
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    let operand_pairs: Vec<(Operand<8>, Operand<8>)> = operands
        .chunks_exact(2)
        .map(|pair| (pair[0], pair[1]))
        .collect();
    let (result, flags) = Value::<E, Enc, W>::dot_with(pairs.iter().copied(), *env);
    assert!(result.is_canonical(), "dot {values:?} {env:?}");
    assert_eq!(
        (outcome(result), flags),
        reduction::dot(&operand_pairs, format, env),
        "dot {values:?} {env:?}"
    );
    for product in Product::ALL {
        let length = match product {
            Product::Values => values.len(),
            Product::Sums | Product::Differences => values.len() / 2 * 2,
        };
        let (scaled, flags) = product.apply(&values[..length], *env);
        assert!(
            scaled.value.is_canonical(),
            "{product:?} {values:?} {env:?}"
        );
        assert_eq!(
            (outcome(scaled.value), scaled.scale, flags),
            reduction::scaled_product(&operands[..length], product, format, env),
            "{product:?} {values:?} {env:?}"
        );
    }
}

/// Checks every vector of a format in every behavior.
fn check_format<const E: u32, Enc: Encoding, const W: usize>(
    layout: Layout,
    make: impl Fn(&Integer) -> Value<E, Enc, W>,
    specials: Specials,
    seed: u64,
) where
    Width<W>: Storage,
    Binary<E, Enc>: Standard<W, Bits = <Width<W> as Storage>::Bits>,
{
    let format = Format::of::<Value<E, Enc, W>>(specials);
    let mut random = SplitMix64::new(seed);
    let vectors: Vec<Vec<Value<E, Enc, W>>> = vectors(layout, &mut random)
        .iter()
        .map(|vector| vector.iter().map(&make).collect())
        .collect();
    for env in &BEHAVIORS {
        for vector in &vectors {
            check_vector(vector, &format, env);
        }
    }
}

/// Returns the value of an encoding of at most 128 bits.
fn narrow<T: TryFrom<u128>>(encoding: &Integer) -> T {
    let bits = encoding
        .to_u128()
        .expect("the encoding has at most 128 bits");
    T::try_from(bits)
        .ok()
        .expect("the encoding fits its format")
}

#[test]
fn fp8_formats() {
    check_format(
        Layout::ieee(8, 4),
        |bits| F8E4M3Fn::from_bits(narrow(bits)),
        Specials::NoInf,
        0x0008_0904,
    );
    check_format(
        Layout::ieee(8, 5),
        |bits| F8E5M2::from_bits(narrow(bits)),
        Specials::Ieee,
        0x0008_0905,
    );
}

#[test]
fn binary16_and_binary32() {
    check_format(
        Layout::BINARY16,
        |bits| F16::from_bits(narrow(bits)),
        Specials::Ieee,
        0x0016_0904,
    );
    check_format(
        Layout::BINARY32,
        |bits| F32::from_bits(narrow(bits)),
        Specials::Ieee,
        0x0032_0904,
    );
}

#[test]
fn binary64_and_x87() {
    check_format(
        Layout::BINARY64,
        |bits| F64::from_bits(narrow(bits)),
        Specials::Ieee,
        0x0064_0904,
    );
    check_format(
        Layout::X87_EXTENDED,
        |bits| F80::from_bits(narrow(bits)),
        Specials::Ieee,
        0x0080_0904,
    );
}

#[test]
fn binary128_to_binary512() {
    check_format(
        Layout::BINARY128,
        |bits| F128::from_bits(narrow(bits)),
        Specials::Ieee,
        0x0128_0904,
    );
    check_format(
        Layout::BINARY256,
        |bits| F256::from_bits(to_limbs(bits)),
        Specials::Ieee,
        0x0256_0904,
    );
    check_format(
        Layout::BINARY512,
        |bits| F512::from_bits(to_limbs(bits)),
        Specials::Ieee,
        0x0512_0904,
    );
}
