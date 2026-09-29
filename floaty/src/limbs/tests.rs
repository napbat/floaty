use super::{Divisor, Limbs, Widen};

#[test]
fn field_reads_across_a_limb_boundary() {
    let value = [0xF000_0000_0000_0000, 0x0000_0000_0000_000A];
    assert_eq!(value.field(60, 8), 0xAF);
    assert_eq!(value.field(0, 64), 0xF000_0000_0000_0000);
    assert_eq!(value.field(64, 4), 0xA);
}

#[test]
fn with_field_writes_across_a_limb_boundary() {
    let value = [u64::MAX, u64::MAX].with_field(60, 8, 0x5A);
    assert_eq!(value, [0xAFFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFF5]);
    assert_eq!(value.field(60, 8), 0x5A);
    assert_eq!([0u64; 1].with_field(0, 64, u64::MAX), [u64::MAX]);
}

#[test]
fn low_bits_and_ones_agree() {
    for count in 0..=192 {
        let all: [u64; 3] = [u64::MAX; 3];
        assert_eq!(
            all.low_bits(count),
            <[u64; 3]>::ones(count),
            "count {count}"
        );
    }
    assert_eq!(<[u64; 2]>::ones(64), [u64::MAX, 0]);
    assert_eq!([u64::MAX].low_bits(0), [0]);
}

#[test]
fn resize_keeps_the_value() {
    let narrow: [u64; 2] = [7, 9];
    let wide: [u64; 4] = narrow.resize();
    assert_eq!(wide, [7, 9, 0, 0]);
    assert_eq!(wide.resize::<[u64; 2]>(), narrow);
}

#[test]
fn shifts_move_bits_across_limbs() {
    let value: [u64; 3] = [0x8000_0000_0000_0001, 0x1, 0];
    assert_eq!(value.shl(1), [0x2, 0x3, 0]);
    assert_eq!(value.shl(64), [0, 0x8000_0000_0000_0001, 0x1]);
    assert_eq!(value.shl(127), [0, 1 << 63, 0xC000_0000_0000_0000]);
    assert_eq!(value.shr(1), [0xC000_0000_0000_0000, 0, 0]);
    assert_eq!(value.shr(63), [0x3, 0, 0]);
    assert_eq!(value.shr(64), [0x1, 0, 0]);
    assert_eq!(value.shr(192), [0; 3]);
    assert_eq!(value.shr(1000), [0; 3]);
    assert_eq!(value.shl(192), [0; 3]);
}

#[test]
fn one_limb_operations_take_every_count() {
    let value: [u64; 1] = [0x8000_0000_0000_0001];
    assert_eq!(value.shl(0), value);
    assert_eq!(value.shl(63), [1 << 63]);
    assert_eq!(value.shl(64), [0]);
    assert_eq!(value.shl(1000), [0]);
    assert_eq!(value.shr(63), [1]);
    assert_eq!(value.shr(64), [0]);
    assert_eq!(value.shr(1000), [0]);
    assert!(!value.any_below(0) && value.any_below(1) && value.any_below(64));
    assert!(!<[u64; 1]>::ZERO.any_below(1000) && value.any_below(1000));
    assert_eq!(value.low_bits(1), [1]);
    assert_eq!(value.low_bits(64), value);
    assert_eq!(value.low_bits(100), value);
    assert_eq!((value.bit_length(), [0_u64].bit_length()), (64, 0));
    assert_eq!([u64::MAX].increment(), [0]);
    assert_eq!([3_u64].add([4]), [7]);
    assert_eq!([7_u64].sub([4]), [3]);
    assert_eq!([3_u64].compare(&[4]), core::cmp::Ordering::Less);
    // Division and the square root of values below 2^64 on a wider
    // array.
    assert_eq!(
        super::divide([100_u64, 0], [1 << 40, 0]),
        ([0, 0], [100, 0])
    );
    assert_eq!(
        super::divide([u64::MAX, 0], [0, 1]),
        ([0, 0], [u64::MAX, 0])
    );
    assert_eq!(
        super::square_root([u64::MAX, 0]),
        ([u64::from(u32::MAX), 0], true)
    );
}

#[test]
fn a_product_that_fits_needs_no_wider_type() {
    assert_eq!(super::multiply_fit([6_u64], [7]), [42]);
    assert_eq!(
        super::multiply_fit([u64::MAX, 0], [u64::MAX, 0]),
        [1, u64::MAX - 1]
    );
    // (2^128 - 1) * (2^64 + 3) across four limbs.
    let product = super::multiply_fit([u64::MAX, u64::MAX, 0, 0], [3, 1, 0, 0]);
    assert_eq!(product, [u64::MAX - 2, u64::MAX - 1, 2, 1]);
    assert_eq!(
        product,
        [u64::MAX, u64::MAX, 0, 0]
            .widening_mul([3, 1, 0, 0])
            .resize::<[u64; 4]>()
    );
}

#[test]
fn bit_length_and_any_below() {
    assert_eq!([0_u64; 2].bit_length(), 0);
    assert_eq!([1_u64, 0].bit_length(), 1);
    assert_eq!([0, 1_u64 << 63].bit_length(), 128);
    let value: [u64; 2] = [0x10, 0];
    assert!(!value.any_below(4) && value.any_below(5) && value.any_below(500));
    assert!(!value.bit(128) && !value.bit(4000));
    assert_eq!(<[u64; 5]>::BITS, 320);
}

#[test]
fn add_sub_and_compare_carry_across_limbs() {
    let a: [u64; 2] = [u64::MAX, 1];
    let b: [u64; 2] = [1, 0];
    assert_eq!(a.add(b), [0, 2]);
    assert_eq!([0, 2].sub(b), a);
    assert_eq!(a.compare(&b), core::cmp::Ordering::Greater);
    assert_eq!(b.compare(&a), core::cmp::Ordering::Less);
    assert_eq!(a.compare(&a), core::cmp::Ordering::Equal);
    assert_eq!([0b1011_u64].shr_jam(2), [0b11]);
    assert_eq!([0b1000_u64].shr_jam(2), [0b10]);
    assert_eq!([1_u64, 0].shr_jam(200), [1, 0]);
}

#[test]
fn widening_multiply_divide_and_square_root() {
    let product = [u64::MAX, u64::MAX].widening_mul([u64::MAX, u64::MAX]);
    // (2^128 - 1)^2 = 2^256 - 2^129 + 1.
    assert_eq!(product, [1, 0, u64::MAX - 1, u64::MAX]);
    let (quotient, remainder) = super::divide([100_u64, 0], [7, 0]);
    assert_eq!((quotient, remainder), ([14, 0], [2, 0]));
    let (quotient, remainder) = super::divide(product, [u64::MAX, u64::MAX, 0, 0]);
    assert_eq!((quotient, remainder), ([u64::MAX, u64::MAX, 0, 0], [0; 4]));
    assert_eq!(super::square_root([144_u64]), ([12], false));
    assert_eq!(super::square_root([145_u64]), ([12], true));
    assert_eq!(super::square_root([0_u64, 1]), ([1 << 32, 0], false));
    assert_eq!(
        super::square_root(product),
        ([u64::MAX, u64::MAX, 0, 0], false)
    );
}

#[test]
fn long_division_matches_the_definition() {
    // (2^128 - 1)^2 + 5 divided by 2^128 - 1. The top limb of the divisor
    // needs no normalization.
    let product = [u64::MAX, u64::MAX].widening_mul([u64::MAX, u64::MAX]);
    let with_rest = product.add([5, 0, 0, 0]);
    assert_eq!(
        super::divide(with_rest, [u64::MAX, u64::MAX, 0, 0]),
        ([u64::MAX, u64::MAX, 0, 0], [5, 0, 0, 0])
    );
    // A quotient limb whose estimate is too large exercises the add-back
    // step: 2^192 / (2^128 + 1).
    let (quotient, remainder) = super::divide([0, 0, 0, 1], [1, 0, 1, 0]);
    // 2^192 = (2^64 - 1)(2^128 + 1) + (2^128 - 2^64 + 1).
    assert_eq!(quotient, [u64::MAX, 0, 0, 0]);
    assert_eq!(remainder, [1, u64::MAX, 0, 0]);
    // One-limb divisor of a wide numerator.
    assert_eq!(
        super::divide([7, 0, 0, 1], [2, 0, 0, 0]),
        ([3, 0, 1 << 63, 0], [1, 0, 0, 0])
    );
    // A wide square root by Newton's iteration.
    let (root, inexact) = super::square_root(product);
    assert_eq!((root, inexact), ([u64::MAX, u64::MAX, 0, 0], false));
    let (root, inexact) = super::square_root(with_rest);
    assert_eq!((root, inexact), ([u64::MAX, u64::MAX, 0, 0], true));
}

#[test]
fn increment_carries() {
    assert_eq!([u64::MAX, 0].increment(), [0, 1]);
    assert_eq!([u64::MAX, u64::MAX].increment(), [0, 0]);
    assert_eq!([5_u64].increment(), [6]);
}

#[test]
fn bits_set_and_read() {
    let value = <[u64; 2]>::ZERO.with_bit(0).with_bit(127);
    assert!(value.bit(0) && value.bit(127) && !value.bit(64));
    assert_eq!(value, [1, 1 << 63]);
    assert!(!value.is_zero());
    assert!(<[u64; 2]>::ZERO.is_zero());
}

#[test]
fn a_divisor_divides_as_the_native_division_does() {
    // A seeded xorshift generator gives the random divisors and limbs.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let check = |divisor: u64, rest: u64, limb: u64| {
        let value = (u128::from(rest) << 64) | u128::from(limb);
        let by = u128::from(divisor);
        let (quotient, remainder) = Divisor::new(divisor).divide_limb(rest, limb);
        assert_eq!(
            (u128::from(quotient), u128::from(remainder)),
            (value / by, value % by),
            "({rest} * 2^64 + {limb}) / {divisor}"
        );
    };
    let powers = (0..20).map(|exponent| 10_u64.pow(exponent));
    // The last two divisors have exact multiples whose first estimate is one
    // too small, so the second correction meets a remainder of zero.
    let edges = [
        2,
        3,
        7,
        (1 << 63) - 1,
        1 << 63,
        (1 << 63) + 1,
        u64::MAX - 1,
        u64::MAX,
        11_839_030_606_303_048_833,
        10_388_559_192_939_298_486,
    ];
    // Random divisors of every bit length, at least 1.
    let random: [u64; 200] = core::array::from_fn(|_| (next() >> (next() % 64)).max(1));
    for divisor in powers.chain(edges).chain(random) {
        // The remainder from the limb above is below the divisor.
        let rests = [0, 1, divisor - 1, next() % divisor];
        for rest in rests.into_iter().filter(|&rest| rest < divisor) {
            for limb in [0, 1, u64::MAX, next()] {
                check(divisor, rest, limb);
            }
        }
        for quotient in [u64::MAX, u64::MAX - 1, 16_424_021_389_205_780_283, next()] {
            let [limb, rest] = super::split_u128(u128::from(quotient) * u128::from(divisor));
            check(divisor, rest, limb);
        }
    }
    // A value of several limbs divides as the long division does.
    let value = [next(), next(), next(), next()];
    let (quotient, remainder) = super::divide_small(value, Divisor::new(1_000_000_007));
    let (expected, rest) = super::divide(value, [1_000_000_007, 0, 0, 0]);
    assert_eq!((quotient, [remainder, 0, 0, 0]), (expected, rest));
}
