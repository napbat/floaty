//! Compares the elementary functions of decimal32, decimal64, and
//! decimal128, in BID and in DPD, with the MPFR oracle in
//! `floaty_verify::operations::elementary`, in every behavior of
//! `TO_DECIMAL_BEHAVIORS`.
//!
//! The arguments are zeros, infinities, and NaNs, random encodings, arguments
//! of `exp` in every decade from below `10^-(p + 2)` to past the overflow
//! threshold, and arguments next to the thresholds where `exp` overflows,
//! becomes tiny, and rounds to zero. For `log`, they are arguments next to 1,
//! 1 in each cohort, powers of ten, subnormals, and random values in the
//! whole range. [`variant_arguments`] adds the exact powers and logarithms
//! in base 2 and 10, and the arguments next to each bound of the other
//! functions. The Intel decimal library converts each BID argument to DPD.
//! `atan2` and `atan2Pi` run the pairs of [`atan2_pairs`].

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::{D32Bid, D32Dpd, D64Bid, D64Dpd, D128Bid, D128Dpd};
use floaty_verify::intel_decimal::{Bid32, Bid64, Bid128, Format as IntelFormat};
use floaty_verify::mpfr::decimal::{DecimalFormat, TO_DECIMAL_BEHAVIORS};
use floaty_verify::operations::check;
use floaty_verify::random::SplitMix64;
use rug::float::Round;
use rug::{Float as BigFloat, Integer};

/// A decimal value `±coefficient * 10^exponent`.
struct Number {
    negative: bool,
    coefficient: Integer,
    exponent: i64,
}

impl Number {
    fn new(negative: bool, coefficient: impl Into<Integer>, exponent: i64) -> Self {
        Self {
            negative,
            coefficient: coefficient.into(),
            exponent,
        }
    }
}

/// Returns `10^digits`.
fn power(digits: u32) -> Integer {
    Integer::from(Integer::u_pow_u(10, digits))
}

/// Returns a format width in bits from its trailing significand field.
fn width(format: DecimalFormat) -> u32 {
    16 * (format.trailing_bits() + 10) / 15
}

/// Returns the BID encoding of a number that the format holds.
fn bid(format: DecimalFormat, number: &Number) -> u128 {
    let (width, trailing) = (width(format), format.trailing_bits());
    let biased =
        u128::try_from(number.exponent - format.exponents().0).expect("the exponent is in range");
    let coefficient = number
        .coefficient
        .to_u128()
        .expect("a coefficient fits a u128");
    let bits = if coefficient >> (trailing + 3) == 0 {
        (biased << (trailing + 3)) | coefficient
    } else {
        // The coefficient starts with the bits 100, which the form with the
        // bits 11 in front of the exponent leaves out.
        (0b11 << (width - 3))
            | (biased << (trailing + 1))
            | (coefficient & ((1 << (trailing + 1)) - 1))
    };
    bits | (u128::from(number.negative) << (width - 1))
}

/// Returns a random coefficient of `digits` digits.
fn coefficient(random: &mut SplitMix64, digits: u32) -> Integer {
    let low = power(digits - 1);
    let span = power(digits) - &low;
    let raw = (Integer::from(random.next_u64()) << 64) | random.next_u64();
    low + raw % span
}

/// Returns a random number of a random digit count with the adjusted
/// exponent `top`, the exponent of its leading digit.
fn with_top(format: DecimalFormat, random: &mut SplitMix64, negative: bool, top: i64) -> Number {
    let digits =
        u32::try_from(random.below(u64::from(format.precision))).expect("a digit count") + 1;
    let exponent = top - i64::from(digits) + 1;
    Number::new(negative, coefficient(random, digits), exponent)
}

/// Returns the zeros, the infinities, and the NaNs, with payloads up to one
/// past the largest canonical payload.
fn specials(format: DecimalFormat) -> Vec<u128> {
    let width = width(format);
    let (lowest, highest) = format.exponents();
    let mut encodings: Vec<u128> = [lowest, 0, highest]
        .iter()
        .map(|&exponent| bid(format, &Number::new(false, 0, exponent)))
        .collect();
    encodings.push(0b11110 << (width - 5));
    let largest = format
        .largest_payload()
        .to_u128()
        .expect("a payload fits a u128");
    for payload in [0, 1, largest, largest + 1] {
        encodings.push((0b11111 << (width - 6)) | payload);
        encodings.push((0b11_1111 << (width - 7)) | payload);
    }
    let sign = 1 << (width - 1);
    let negated: Vec<u128> = encodings.iter().map(|bits| bits | sign).collect();
    encodings.extend(negated);
    encodings
}

/// Returns the numbers next to `value` rounded to `p` digits: that number
/// and `steps` coefficients on each side of it.
fn around(format: DecimalFormat, value: &BigFloat, steps: u32) -> Vec<Number> {
    let precision = format.precision;
    let digits = usize::try_from(precision).expect("a precision fits a usize");
    let (negative, text, exponent) =
        value.to_sign_string_exp_round(10, Some(digits), Round::Nearest);
    let center = Integer::from_str_radix(&text, 10).expect("MPFR writes decimal digits");
    let exponent = i64::from(exponent.expect("the value is finite")) - i64::from(precision);
    (0..=2 * steps)
        .map(|step| center.clone() + step - steps)
        .filter(|coefficient| coefficient.to_string().len() == digits)
        .map(|coefficient| Number::new(negative, coefficient, exponent))
        .collect()
}

/// Returns the arguments next to the thresholds of `exp`: where the result
/// overflows, where it becomes tiny, where it is the smallest subnormal, and
/// where it is half of it.
fn thresholds(format: DecimalFormat) -> Vec<Number> {
    let working = 8 * format.precision + 64;
    let lowest = format.exponents().0;
    let ln_10 = BigFloat::with_val(working, 10).ln();
    let ten = |exponent: i64| BigFloat::with_val(working, &ln_10 * Integer::from(exponent));
    let half = BigFloat::with_val(working, 2).ln();
    [
        ten(i64::from(format.emax) + 1),
        ten(i64::from(1 - format.emax)),
        ten(lowest),
        ten(lowest) - half,
    ]
    .iter()
    .flat_map(|value| around(format, value, 6))
    .collect()
}

/// Returns the arguments of `log` next to 1, 1 in each cohort, the powers
/// of ten, and subnormals.
fn near_one(format: DecimalFormat, random: &mut SplitMix64) -> Vec<Number> {
    let precision = format.precision;
    let digits = i64::from(precision);
    let (lowest, highest) = format.exponents();
    let mut numbers = Vec::new();
    for step in 1..=6_u32 {
        numbers.push(Number::new(false, power(precision - 1) + step, 1 - digits));
        numbers.push(Number::new(false, power(precision) - step, -digits));
    }
    numbers.extend((0..precision).map(|zeros| Number::new(false, power(zeros), -i64::from(zeros))));
    numbers.extend(
        [lowest, i64::from(1 - format.emax), -1, 1, highest]
            .map(|exponent| Number::new(false, 1, exponent)),
    );
    numbers.push(Number::new(false, power(precision - 1) - 1, lowest));
    numbers.extend((0..8).map(|_| {
        let digits = u32::try_from(random.below(u64::from(precision - 1))).expect("a digit count");
        Number::new(false, coefficient(random, digits + 1), lowest)
    }));
    numbers
}

/// Returns arguments of the functions in base 2 and 10 and of the shifted
/// forms: the integers up to 40 in magnitude, two of them in other members
/// of their cohorts, and the integers next to where `10^x` and `2^x`
/// overflow and underflow, which give exact powers; powers of 10 and 2, and
/// the values one below them, which give exact logarithms; and the arguments
/// next to -1, to the tiny bound `10^-(p + 3)`, to where `b^x - 1` comes
/// within one unit of -1, and to the large bound `10^(p + 3)` of
/// `log_b(1 + x)`.
fn variant_arguments(format: DecimalFormat) -> Vec<Number> {
    let precision = format.precision;
    let digits = i64::from(precision);
    let (lowest, highest) = format.exponents();
    let emax = i64::from(format.emax);
    let emin = 1 - emax;
    let working = 8 * precision + 64;
    let integer = |n: i64| Number::new(n < 0, n.unsigned_abs(), 0);
    let mut numbers: Vec<Number> = (-40..=40).filter(|n| *n != 0).map(integer).collect();
    numbers.extend([Number::new(false, 3, 1), Number::new(true, 200, -2)]);
    numbers.extend(
        [
            emax - 1,
            emax,
            emax + 1,
            emin - digits,
            emin - digits + 1,
            emin,
        ]
        .map(integer),
    );
    // `2^x` overflows near `(emax + 1) log2 10`, and rounds to the smallest
    // subnormal or to zero near `(emin - p + 1) log2 10`.
    let log2_10 = BigFloat::with_val(working, 10).log2();
    for exponent in [emax + 1, emin - digits + 1] {
        let threshold = BigFloat::with_val(working, &log2_10 * Integer::from(exponent));
        let floor = threshold
            .floor()
            .to_integer()
            .and_then(|floor| floor.to_i64())
            .expect("a threshold fits an i64");
        numbers.extend((floor - 1..=floor + 2).map(integer));
    }
    numbers.extend(
        (-30..=30)
            .chain([lowest, emin, highest])
            .map(|k| Number::new(false, 1, k)),
    );
    numbers.extend((digits + 2..=digits + 4).map(|k| Number::new(false, 1, k)));
    for k in 1..=precision {
        let nines = power(k) - 1u32;
        numbers.push(Number::new(true, nines.clone(), -i64::from(k)));
        numbers.push(Number::new(false, nines, 0));
    }
    for k in 1..=40_u32 {
        let two = Integer::from(Integer::u_pow_u(2, k));
        let five = Integer::from(Integer::u_pow_u(5, k));
        let one_below = power(k) - &five;
        let k = i64::from(k);
        numbers.push(Number::new(true, one_below, -k));
        numbers.push(Number::new(false, two.clone() - 1u32, 0));
        numbers.push(Number::new(false, five, -k));
        numbers.push(Number::new(false, two, 0));
    }
    numbers.extend([
        Number::new(true, 1, 0),
        Number::new(true, power(precision) - 1u32, -digits),
        Number::new(true, power(precision - 1) + 1u32, 1 - digits),
    ]);
    for top in [-(digits + 4), -(digits + 3), -(digits + 2)] {
        for negative in [false, true] {
            numbers.push(Number::new(negative, 1, top));
            numbers.push(Number::new(
                negative,
                power(precision) - 1u32,
                top - digits + 1,
            ));
        }
    }
    // `b^x - 1` comes within one unit of -1 at `x = -(p + 2) log_b 10`.
    let units = BigFloat::with_val(working, -(digits + 2));
    let ln_10 = BigFloat::with_val(working, 10).ln();
    for factor in [ln_10.clone(), log2_10, BigFloat::with_val(working, 1)] {
        numbers.extend(around(
            format,
            &BigFloat::with_val(working, &units * &factor),
            2,
        ));
    }
    // The hyperbolic functions take their tiny rule below
    // `10^-ceil((p + 3) / 2)`, `tanh` comes within one unit of ±1 at
    // `(p + 3) ln 10 / 2`, and `sinh` and `cosh` overflow past
    // `(emax + 1) ln 10 + ln 2`.
    let half = i64::midpoint(digits, 4);
    for top in [-(half + 1), -half, 1 - half] {
        for negative in [false, true] {
            numbers.push(Number::new(negative, 1, top));
            numbers.push(Number::new(negative, power(precision) - 1u32, top - digits));
        }
    }
    let tanh_bound = BigFloat::with_val(working, &ln_10 * (digits + 3)) >> 1u32;
    let ln_2 = BigFloat::with_val(working, 2).ln();
    let overflow = BigFloat::with_val(working, &ln_10 * (emax + 1)) + ln_2;
    for bound in [tanh_bound, overflow] {
        numbers.extend(around(format, &bound, 2));
        numbers.extend(around(format, &-bound, 2));
    }
    numbers.retain(|number| {
        number.coefficient.to_string().len() <= usize::try_from(precision).expect("a count")
            && (lowest..=highest).contains(&number.exponent)
    });
    numbers
}

/// Returns arguments of `sinPi`, `cosPi`, and `tanPi`. The functions are
/// exact at the quarters, here up to 6 in magnitude, which come with two
/// neighbors on each side. Below `10^(p - 2)`, an argument of `p` digits can
/// have two digits after the point: its reduction keeps the most digits, and
/// a fraction of .25, .51, or .99 lies at or next to an exact value.
fn pi_scaled_arguments(format: DecimalFormat) -> Vec<Number> {
    let precision = format.precision;
    let mut numbers = Vec::new();
    for n in 1..=24_u32 {
        let quarter = BigFloat::with_val(precision + 8, n) >> 2u32;
        numbers.extend(around(format, &quarter, 2));
        numbers.extend(around(format, &-quarter, 2));
    }
    for (below, exponent) in [(1_u32, -1), (1, -2), (49, -2), (75, -2)] {
        for negative in [false, true] {
            numbers.push(Number::new(negative, power(precision) - below, exponent));
        }
    }
    numbers
}

/// Returns arguments of the inverse trigonometric functions: ±1/2 and
/// ±sqrt(1/2) with two neighbors on each side, where `asinPi` and `acosPi`
/// are multiples of 1/6 and the form of `asin` reflects, and the arguments
/// next to ±10^(p + 3), from which `|atanPi|` comes within one unit of 1/2.
fn inverse_arguments(format: DecimalFormat) -> Vec<Number> {
    let precision = format.precision;
    let working = 4 * precision + 64;
    let half = BigFloat::with_val(working, 1) >> 1u32;
    let root = BigFloat::with_val(working, half.sqrt_ref());
    let mut numbers = Vec::new();
    for value in [half, root] {
        numbers.extend(around(format, &value, 2));
        numbers.extend(around(format, &-value, 2));
    }
    let top = i64::from(precision) + 3;
    for negative in [false, true] {
        numbers.push(Number::new(negative, 1, top));
        numbers.push(Number::new(negative, power(precision) - 1u32, 3));
    }
    numbers
}

/// Bad cases of `exp` for decimal64, as coefficient and exponent, from
/// Lefèvre, Stehlé, and Zimmermann, "Worst Cases for the Exponential
/// Function in the IEEE 754r decimal64 Format", JNAO 2006, slides 6 and 13
/// to 16, <https://www.vinc17.net/research/slides/jnao2006-05.pdf>.
const DECIMAL64_EXP_WORST_CASES: [(u64, i64); 39] = [
    (6_581_539_478_341_669, -24),
    (2_662_858_264_545_929, -23),
    (3_639_588_333_766_983, -23),
    (6_036_998_017_773_271, -23),
    (6_638_670_361_402_304, -22),
    (9_366_572_213_364_879, -22),
    (7_970_613_003_079_781, -21),
    (3_089_765_552_852_523, -20),
    (1_302_531_956_641_873, -19),
    (2_241_856_702_421_245, -19),
    (7_230_293_679_121_590, -19),
    (5_259_640_428_979_129, -18),
    (9_407_822_313_572_878, -17),
    (1_267_914_924_960_933, -16),
    (5_091_077_534_282_133, -16),
    (3_359_104_074_009_002, -15),
    (1_910_511_686_234_796, -14),
    (2_949_551_257_293_143, -13),
    (5_879_131_381_356_093, -13),
    (7_906_867_968_553_504, -16),
    (1_548_443_067_391_468, -18),
    (2_953_379_504_777_270, -16),
    (3_897_940_992_403_028, -24),
    (4_230_932_991_049_603, -24),
    (4_291_382_990_792_016, -24),
    (4_581_289_989_505_891, -24),
    (5_999_879_998_200_072, -25),
    (6_000_119_998_199_928, -25),
    (1_019_999_999_994_798, -26),
    (1_039_999_999_994_592, -26),
    (1_099_999_999_999_395, -27),
    (1_199_999_999_999_280, -27),
    (1_199_999_999_999_928, -28),
    (1_399_999_999_999_902, -28),
    (1_999_999_999_999_980, -29),
    (2_999_999_999_999_955, -29),
    (1_999_999_999_999_998, -30),
    (3_999_999_999_999_992, -30),
    (9_999_999_999_999_995, -31),
];

/// Returns the BID encodings of the constructed arguments of a format, and
/// of the positive `extra` arguments, as coefficient and exponent.
fn constructed(format: DecimalFormat, random: &mut SplitMix64, extra: &[(u64, i64)]) -> Vec<u128> {
    let digits = i64::from(format.precision);
    let (lowest, highest) = format.exponents();
    let mut numbers = Vec::new();
    for top in -(digits + 5)..=5 {
        for negative in [false, true] {
            numbers.extend((0..8).map(|_| with_top(format, random, negative, top)));
        }
    }
    numbers.extend(thresholds(format));
    numbers.extend(near_one(format, random));
    numbers.extend(variant_arguments(format));
    numbers.extend(pi_scaled_arguments(format));
    numbers.extend(inverse_arguments(format));
    numbers.extend(
        extra
            .iter()
            .map(|&(coefficient, exponent)| Number::new(false, coefficient, exponent)),
    );
    let span = u64::try_from(highest - lowest + 1).expect("the exponent range is positive");
    numbers.extend((0..2_000).map(|_| {
        let exponent = lowest + i64::try_from(random.below(span)).expect("an exponent fits");
        let count = u32::try_from(random.below(u64::from(format.precision))).expect("a count") + 1;
        Number::new(false, coefficient(random, count), exponent)
    }));
    let mut encodings = specials(format);
    encodings.extend(numbers.iter().map(|number| bid(format, number)));
    encodings
}

/// Returns `width` random bits.
fn random_bits(random: &mut SplitMix64, width: u32) -> u128 {
    let bits = (u128::from(random.next_u64()) << 64) | u128::from(random.next_u64());
    bits >> (128 - width)
}

/// Checks one decimal width in BID and in DPD.
macro_rules! check_width {
    ($bid:ty, $dpd:ty, $intel:ty, $bits:ty, $count:literal, $seed:literal, $extra:expr) => {{
        let format = DecimalFormat::of::<$bid>();
        let mut random = SplitMix64::new($seed);
        let narrow = |bits: u128| <$bits>::try_from(bits).expect("the encoding has the width");
        let mut values: Vec<($bid, $dpd)> = constructed(format, &mut random, $extra)
            .into_iter()
            .map(|bits| {
                let bits = narrow(bits);
                let dpd = <$intel as IntelFormat>::to_dpd(bits);
                (<$bid>::from_bits(bits), <$dpd>::from_bits(dpd))
            })
            .collect();
        values.extend((0..$count).map(|_| {
            let bits = narrow(random_bits(&mut random, width(format)));
            (<$bid>::from_bits(bits), <$dpd>::from_bits(bits))
        }));
        for env in &TO_DECIMAL_BEHAVIORS {
            for &(bid, dpd) in &values {
                check::check_decimal_elementary(bid, format, env);
                check::check_decimal_elementary(dpd, format, env);
            }
        }
    }};
}

#[test]
fn decimal32() {
    check_width!(D32Bid, D32Dpd, Bid32, u32, 2_000, 0xDEC_E032, &[]);
}

#[test]
fn decimal64() {
    check_width!(
        D64Bid,
        D64Dpd,
        Bid64,
        u64,
        2_000,
        0xDEC_E064,
        &DECIMAL64_EXP_WORST_CASES
    );
}

#[test]
fn decimal128() {
    check_width!(D128Bid, D128Dpd, Bid128, u128, 2_000, 0xDEC_E128, &[]);
}

/// Returns the pairs of `atan2`, `y` first, as BID encodings: the zeros, the
/// infinities, the NaNs, ±1 in two cohorts, ±3, ±1/2, and the extremes of
/// the format against each other, random pairs, and pairs whose quotient
/// `|y| / |x|` lies next to each bound of the shortcuts: `10^-(p + 3)`, its
/// reciprocal, and `2^-(4p + 70)` of the oracle. `|y| = |x|` also runs in
/// other cohorts.
fn atan2_pairs(format: DecimalFormat, random: &mut SplitMix64, count: usize) -> Vec<(u128, u128)> {
    let digits = i64::from(format.precision);
    let (lowest, highest) = format.exponents();
    let largest = power(format.precision) - 1u32;
    let numbers = [
        Number::new(false, 1, 0),
        Number::new(false, 10, -1),
        Number::new(false, 3, 0),
        Number::new(false, 5, -1),
        Number::new(false, 1, lowest),
        Number::new(false, 1, 1 - i64::from(format.emax)),
        Number::new(false, largest, highest),
    ];
    let mut values = specials(format);
    for number in &numbers {
        let encoding = bid(format, number);
        values.extend([encoding, encoding | (1 << (width(format) - 1))]);
    }
    let mut pairs: Vec<(u128, u128)> = values
        .iter()
        .flat_map(|&y| values.iter().map(move |&x| (y, x)))
        .collect();
    pairs.extend((0..count).map(|_| {
        // A leading digit from `10^(lowest + p)` keeps every exponent of the
        // coefficient in the range.
        let number = |random: &mut SplitMix64| {
            let span = u64::try_from(highest - lowest - digits).expect("the range is positive");
            let top = lowest + digits + i64::try_from(random.below(span)).expect("an exponent");
            let negative = random.below(2) == 1;
            bid(format, &with_top(format, random, negative, top))
        };
        (number(random), number(random))
    }));
    let mut quotients = vec![
        (Number::new(false, 25, -1), Number::new(false, 250, -2)),
        (Number::new(false, 1, 0), Number::new(false, 100, -2)),
    ];
    // `2^-(4p + 70)` lies near `10^-k` with `k = (4p + 70) log10(2)`.
    let oracle = (4 * digits + 70) * 30_103 / 100_000;
    let bounds = (digits + 1..=digits + 5).chain(oracle - 2..=oracle + 2);
    for k in bounds {
        for (y, x) in [(1, 1), (3, 3), (1, 3)] {
            quotients.push((Number::new(false, y, -k), Number::new(false, x, 0)));
            quotients.push((Number::new(false, y, k), Number::new(false, x, 0)));
        }
    }
    for (y, x) in quotients {
        let (y, x) = (bid(format, &y), bid(format, &x));
        let sign = 1 << (width(format) - 1);
        pairs.extend([(y, x), (y, x | sign), (y | sign, x), (y | sign, x | sign)]);
    }
    pairs
}

/// Checks `atan2` and `atan2Pi` of one decimal width in BID and in DPD.
macro_rules! check_atan2_width {
    ($bid:ty, $dpd:ty, $intel:ty, $bits:ty, $count:literal, $seed:literal) => {{
        let format = DecimalFormat::of::<$bid>();
        let mut random = SplitMix64::new($seed);
        let narrow = |bits: u128| <$bits>::try_from(bits).expect("the encoding has the width");
        let both = |bits: u128| {
            let bits = narrow(bits);
            let dpd = <$intel as IntelFormat>::to_dpd(bits);
            (<$bid>::from_bits(bits), <$dpd>::from_bits(dpd))
        };
        let pairs: Vec<_> = atan2_pairs(format, &mut random, $count)
            .into_iter()
            .map(|(y, x)| (both(y), both(x)))
            .collect();
        for env in &TO_DECIMAL_BEHAVIORS {
            for &((y_bid, y_dpd), (x_bid, x_dpd)) in &pairs {
                check::check_decimal_atan2(y_bid, x_bid, format, env);
                check::check_decimal_atan2(y_dpd, x_dpd, format, env);
            }
        }
    }};
}

#[test]
fn atan2_of_every_width() {
    check_atan2_width!(D32Bid, D32Dpd, Bid32, u32, 500, 0xA2_D032);
    check_atan2_width!(D64Bid, D64Dpd, Bid64, u64, 500, 0xA2_D064);
    check_atan2_width!(D128Bid, D128Dpd, Bid128, u128, 500, 0xA2_D128);
}
