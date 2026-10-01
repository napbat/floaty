//! Compares the conversions between the decimal formats and the binary
//! formats that the Intel decimal library lacks with MPFR: binary16,
//! bfloat16, TF32, the FP8 formats, binary256, and binary512. binary32,
//! binary64, x87 extended, and binary128 run here too, in the behaviors that
//! the Intel library lacks, and with the unsupported x87 encodings.
//!
//! - To decimal: MPFR gives the leading digits of the exact binary value,
//!   truncated and rounded away from zero, and the digit after them. The
//!   rounding directions follow from those digits by their IEEE 754
//!   definitions, and IBM's round to prepare for shorter precision for
//!   `ToOdd`. The exponent follows the IEEE 754 preferred exponent rules,
//!   with preferred exponent 0.
//! - To binary: GMP reduces `C * 10^q` to an integer of at least `p + 3`
//!   bits and a sticky bit, and the MPFR oracle rounds that input.
//!
//! The special values follow floaty's conversion rules. A NaN result must be
//! a quiet NaN with the sign of the source and the payload of floaty's rule,
//! in both directions: the payload bits align with the trailing significand
//! field of the decimal format and keep their high-order bits, and a decimal
//! payload above `10^(p - 1) - 1` becomes zero. The Intel library confirms
//! that rule for binary32, binary64, x87 extended, and binary128. The test
//! computes the payload with GMP from the payload of the source, which
//! floaty's `decode` gives. The binary decoding tests check that payload for
//! the binary formats. For the decimal formats, every decimal operation
//! reads its NaN operands as `decode` does, and the decTest and Intel tests
//! check the payloads of those results bit for bit.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::marker::PhantomData;

use floaty::env::NanPropagation;
use floaty::format::Standard;
use floaty::{
    BF16, Binary, D32Bid, D32Dpd, D64Bid, D64Dpd, D128Bid, D128Dpd, Decoded, Env, Exact, F16, F32,
    F64, F80, F128, F256, F512, Flags, Float, TF32,
};
use floaty_verify::encodings::{
    Layout, boundary_encodings, sample_encodings_u128, to_limbs, to_u128,
};
use floaty_verify::mpfr::decimal::{
    self, DecimalFormat, DecimalValue, FROM_DECIMAL_BEHAVIORS, TO_DECIMAL_BEHAVIORS, align,
};
use floaty_verify::mpfr::{self, Format, Input, Operand, Specials, Value};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// Returns the payload of a binary NaN converted from a decimal value, by
/// floaty's rule. The payload value of a decimal NaN is a field of
/// `trailing` bits. The binary payload is the fraction below the quiet bit.
/// A format with one NaN encoding has no payload, and neither has a NaN that
/// a finite value or an infinity gives, nor the default NaN of the
/// `DefaultNan` rule.
fn binary_payload(source: &Decoded<2>, trailing: u32, format: &Format, env: &Env) -> Integer {
    let Decoded::Nan { payload, .. } = *source else {
        return Integer::ZERO;
    };
    if env.nan.propagation == NanPropagation::DefaultNan {
        return Integer::ZERO;
    }
    match format.specials {
        Specials::Ieee => align(
            Integer::from_digits(&payload, Order::Lsf),
            trailing,
            format.precision - 2,
        ),
        Specials::NoInf | Specials::Fnuz | Specials::Finite => Integer::ZERO,
    }
}

/// Returns the expected result of converting a decimal operand to a binary
/// format.
fn decimal_to_binary(source: &Operand<2>, format: &Format, env: &Env) -> (Value, Flags) {
    let Decoded::Finite {
        negative,
        exponent,
        significand,
    } = source.decoded
    else {
        return mpfr::convert(source, format, env);
    };
    let subnormal = source.subnormal;
    if subnormal && env.denormals_are_zero {
        return mpfr::convert(source, format, env);
    }
    let coefficient = Integer::from_digits(&significand, Order::Lsf);
    let input = if exponent >= 0 {
        Input {
            negative,
            exponent,
            significand: coefficient * Integer::from(Integer::u_pow_u(5, exponent.unsigned_abs())),
            sticky: false,
        }
    } else {
        // C * 10^-k = (C * 2^t / 5^k) * 2^(-k - t), with a quotient of at
        // least p + 3 bits.
        let divisor = Integer::from(Integer::u_pow_u(5, exponent.unsigned_abs()));
        let room = format.precision + 3 + divisor.significant_bits();
        let shift = room.saturating_sub(coefficient.significant_bits());
        let (quotient, remainder) = (coefficient << shift).div_rem(divisor);
        Input {
            negative,
            exponent: exponent - i32::try_from(shift).expect("a shift fits an i32"),
            significand: quotient,
            sticky: remainder != 0,
        }
    };
    let (value, flags) = mpfr::round(&input, format, env);
    let flags = if subnormal {
        flags | Flags::DENORMAL_INPUT
    } else {
        flags
    };
    (value, flags)
}

/// Converts every binary source value to every decimal destination in every
/// behavior, by [`check_to_decimal_format`].
macro_rules! check_to_decimal {
    ($source:ty, $values:expr => $($destination:ty),+) => {{
        let values: Vec<$source> = $values.iter().map(|&bits| <$source>::from_bits(bits)).collect();
        $(
            check_to_decimal_format(
                &values,
                PhantomData::<$destination>,
                (stringify!($source), stringify!($destination)),
            );
        )+
    }};
}

/// Converts every decimal source value to every binary destination in every
/// behavior, by [`check_to_binary_format`].
macro_rules! check_to_binary {
    ($source:ty, $values:expr => $($destination:ty: $specials:expr),+) => {{
        $(
            check_to_binary_format(
                &$values,
                PhantomData::<$destination>,
                $specials,
                (stringify!($source), stringify!($destination)),
            );
        )+
    }};
}

/// Converts every binary source value to the decimal format `D` in every
/// behavior, and compares the result and the flags with the oracle.
/// `names` names the source and the destination in a failure.
fn check_to_decimal_format<S: Standard<SW>, const SW: usize, D: Standard<DW>, const DW: usize>(
    values: &[Float<S, SW>],
    _destination: PhantomData<Float<D, DW>>,
    names: (&str, &str),
) {
    let format = DecimalFormat::of::<Float<D, DW>>();
    // The fraction bits below the quiet bit. A format with one NaN encoding
    // decodes a zero payload, so its width does not matter.
    let payload_bits = Float::<S, SW>::PRECISION - 2;
    for env in TO_DECIMAL_BEHAVIORS {
        for &source in values {
            let operand = Operand::<8>::of(source);
            let (ours, flags): (Float<D, DW>, _) = source.convert_with(env);
            let bits = source.to_bits();
            let context = format!("{} {bits:x?} to {} {env:?}", names.0, names.1);
            assert!(ours.is_canonical(), "{context}: {ours:?} is canonical");
            let ours = (DecimalValue::from_decoded(ours.decode::<2>()), flags);
            let expected = decimal::convert(&operand, payload_bits, format, &env);
            assert_eq!(ours, expected, "{context}");
        }
    }
}

/// Converts every decimal source value to the binary format `D`, which has
/// the special values `specials`, in every behavior. A NaN result must be
/// quiet, with the payload of [`binary_payload`]. `names` names the source
/// and the destination in a failure.
fn check_to_binary_format<S: Standard<SW>, const SW: usize, D: Standard<DW>, const DW: usize>(
    values: &[Float<S, SW>],
    _destination: PhantomData<Float<D, DW>>,
    specials: Specials,
    names: (&str, &str),
) {
    let trailing = DecimalFormat::of::<Float<S, SW>>().trailing_bits();
    let format = Format::of::<Float<D, DW>>(specials);
    for env in FROM_DECIMAL_BEHAVIORS {
        for &source in values {
            let operand = Operand::<2>::of(source);
            let (ours, flags): (Float<D, DW>, _) = source.convert_with(env);
            let bits = source.to_bits();
            let context = format!("{} {bits:x?} to {} {env:?}", names.0, names.1);
            assert!(ours.is_canonical(), "{context}: {ours:?} is canonical");
            let result = ours.decode::<8>();
            let ours = (Value::from_decoded(result), flags);
            assert_eq!(
                ours,
                decimal_to_binary(&operand, &format, &env),
                "{context}"
            );
            if let Decoded::Nan {
                signaling, payload, ..
            } = result
            {
                assert!(!signaling, "{context}: a conversion quiets a NaN");
                assert_eq!(
                    Integer::from_digits(&payload, Order::Lsf),
                    binary_payload(&operand.decoded, trailing, &format, &env),
                    "{context}: the NaN payload"
                );
            }
        }
    }
}

/// Converts every binary source value, of the format named `name`, to every
/// decimal interchange format in both encodings.
fn to_every_decimal_format<S: Standard<SW>, const SW: usize>(values: &[Float<S, SW>], name: &str) {
    check_to_decimal_format(values, PhantomData::<D32Bid>, (name, "D32Bid"));
    check_to_decimal_format(values, PhantomData::<D32Dpd>, (name, "D32Dpd"));
    check_to_decimal_format(values, PhantomData::<D64Bid>, (name, "D64Bid"));
    check_to_decimal_format(values, PhantomData::<D64Dpd>, (name, "D64Dpd"));
    check_to_decimal_format(values, PhantomData::<D128Bid>, (name, "D128Bid"));
    check_to_decimal_format(values, PhantomData::<D128Dpd>, (name, "D128Dpd"));
}

/// Converts every decimal source value, of the format named `name`, to
/// every format of `for_each_small_format!` and of `for_each_wide_format!`.
fn to_every_listed_format<S: Standard<SW>, const SW: usize>(values: &[Float<S, SW>], name: &str) {
    macro_rules! to_small {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
            check_to_binary_format(
                values,
                PhantomData::<Float<$standard, $width>>,
                $specials,
                (name, stringify!($alias)),
            )
        };
    }
    floaty_verify::for_each_small_format!(to_small);
    macro_rules! to_wide {
        ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
            check_to_binary_format(
                values,
                PhantomData::<Float<Binary<$exponent_bits>, $width>>,
                Specials::Ieee,
                (name, stringify!($alias)),
            )
        };
    }
    floaty_verify::for_each_wide_format!(to_wide);
}

/// Returns decimal values at the edges of the coefficient range, at every
/// `stride`-th exponent and at both ends of the exponent range, random
/// values, and the special encodings.
macro_rules! decimal_values {
    ($alias:ty, $stride:expr, $random:expr, $seed:expr, $specials:expr) => {{
        let precision = <$alias>::PRECISION;
        let emax = i64::from(<$alias>::EMAX);
        let digits = i64::from(precision);
        let (lowest, highest) = (2 - emax - digits, emax - digits + 1);
        let largest = Integer::from(Integer::u_pow_u(10, precision)) - 1u32;
        let mut random = SplitMix64::new($seed);
        let mut coefficients: Vec<Integer> = vec![
            Integer::from(1),
            Integer::from(5),
            Integer::from(Integer::u_pow_u(10, precision - 1)),
            largest.clone(),
            largest.clone() / 2u32,
        ];
        coefficients.extend((0..3).map(|_| Integer::from(random.next_u128()) % &largest));
        let mut exponents: Vec<i64> = (lowest..=highest).step_by($stride).collect();
        exponents.extend([lowest, lowest + 1, highest - 1, highest, -1, 0, 1]);
        let make = |negative: bool, coefficient: &Integer, exponent: i64| {
            let mut significand = [0_u64; 2];
            for (slot, limb) in significand
                .iter_mut()
                .zip(coefficient.to_digits::<u64>(Order::Lsf))
            {
                *slot = limb;
            }
            let exact = Exact {
                negative,
                exponent: i32::try_from(exponent).expect("an exponent of the format fits an i32"),
                significand,
                sticky: false,
            };
            <$alias>::round(exact, Env::IEEE).0
        };
        let mut values = Vec::new();
        for &exponent in &exponents {
            for coefficient in &coefficients {
                values.push(make(false, coefficient, exponent));
                values.push(make(true, coefficient, exponent));
            }
        }
        for _ in 0..$random {
            let coefficient = Integer::from(random.next_u128()) % &largest;
            let span = u64::try_from(highest - lowest + 1).expect("the range is positive");
            let exponent = lowest + i64::try_from(random.below(span)).expect("the offset fits");
            values.push(make(random.coin_flip(), &coefficient, exponent));
        }
        values.extend($specials.map(<$alias>::from_bits));
        values
    }};
}

/// The infinities, and quiet and signaling NaNs of both signs. A NaN has a
/// zero or small payload, the top bit of the trailing field, the largest
/// canonical BID payload `10^(p - 1) - 1`, the non-canonical BID payload
/// `10^(p - 1)`, alternate bits, or the bits between the signaling bit and
/// the trailing field. A DPD encoding reads the trailing field as declets,
/// so the same bits have another payload value there.
const DECIMAL32_SPECIALS: [u32; 9] = [
    0x7800_0000,
    0xF800_0000,
    0x7C00_0000,
    0xFE00_0005,
    0x7C08_0000,
    0xFE0F_423F,
    0x7C0F_4240,
    0x7DF0_0001,
    0xFE05_5555,
];
const DECIMAL64_SPECIALS: [u64; 9] = [
    0x7800_0000_0000_0000,
    0xF800_0000_0000_0000,
    0x7C00_0000_0000_0000,
    0xFE00_0000_0000_0005,
    0x7C02_0000_0000_0000,
    0xFE03_8D7E_A4C6_7FFF,
    0x7C03_8D7E_A4C6_8000,
    0x7DFC_0000_0000_0001,
    0xFE01_5555_5555_5555,
];
const DECIMAL128_SPECIALS: [u128; 9] = [
    0x7800 << 112,
    0xF800 << 112,
    0x7C00 << 112,
    (0xFE00 << 112) | 5,
    (0x7C00 << 112) | (1 << 109),
    (0xFE00 << 112) | (10_u128.pow(33) - 1),
    (0x7C00 << 112) | 10_u128.pow(33),
    (0x7C00 << 112) | (0x7FF << 110) | 1,
    (0xFE00 << 112) | ((u128::MAX >> 18) / 3),
];

/// Every encoding of each format of `for_each_small_format!` converts to
/// every decimal format.
#[test]
fn from_the_small_formats_to_decimal() {
    macro_rules! from_listed {
        ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {{
            let values: Vec<Float<$standard, $width>> = (0..1_u16 << $width)
                .map(|bits| Float::from_bits(u8::try_from(bits).expect("at most 8 bits")))
                .collect();
            to_every_decimal_format(&values, stringify!($alias));
        }};
    }
    floaty_verify::for_each_small_format!(from_listed);
}

#[test]
fn from_binary16_and_bfloat16_to_decimal() {
    let every: Vec<u16> = (0..=u16::MAX).collect();
    check_to_decimal!(F16, every => D32Bid, D64Dpd);
    check_to_decimal!(BF16, every => D32Dpd, D64Bid);
}

#[test]
fn from_the_wide_formats_to_decimal() {
    let mut random = SplitMix64::new(0xDEC_0512);
    let tf32: Vec<u32> = boundary_encodings(Layout::TF32)
        .iter()
        .map(|encoding| u32::try_from(to_u128(encoding)).expect("19 bits"))
        .chain((0..5_000).map(|_| u32::try_from(random.next_u64() >> 45).expect("19 bits")))
        .collect();
    check_to_decimal!(TF32, tf32 => D32Bid, D128Dpd);
    let mut wide: Vec<[u64; 8]> = boundary_encodings(Layout::BINARY512)
        .iter()
        .map(to_limbs::<8>)
        .collect();
    wide.extend((0..1_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check_to_decimal!(F512, wide => D32Dpd, D64Bid, D128Bid);
    let mut narrow: Vec<[u64; 4]> = boundary_encodings(Layout::BINARY256)
        .iter()
        .map(to_limbs::<4>)
        .collect();
    narrow.extend((0..1_000).map(|_| core::array::from_fn(|_| random.next_u64())));
    check_to_decimal!(F256, narrow => D32Bid, D64Dpd, D128Dpd);
}

/// The boundary and random encodings of each format of
/// `for_each_wide_format!` convert to every decimal format.
#[test]
fn from_every_wide_format_to_decimal() {
    let mut random = SplitMix64::new(0xDEC_0160);
    macro_rules! from_listed {
        ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {{
            let layout = Layout::ieee($width, $exponent_bits);
            let values: Vec<Float<Binary<$exponent_bits>, $width>> = boundary_encodings(layout)
                .iter()
                .map(to_limbs::<$limbs>)
                .chain((0..300).map(|_| {
                    let digits: [u64; $limbs] = core::array::from_fn(|_| random.next_u64());
                    let mask = (Integer::from(1) << layout.width) - 1u32;
                    to_limbs::<$limbs>(&(Integer::from_digits(&digits, Order::Lsf) & mask))
                }))
                .map(Float::from_bits)
                .collect();
            to_every_decimal_format(&values, stringify!($alias));
        }};
    }
    floaty_verify::for_each_wide_format!(from_listed);
}

/// Returns `count` random encodings of a binary interchange format with the
/// layout `layout`, in `N` limbs. Each has a random sign and random bits in
/// the whole fraction. Its exponent is in the range of `format`, from the
/// smallest subnormal value to the largest value, or up to 16 binades beyond
/// either end.
fn in_decimal_range<const N: usize>(
    random: &mut SplitMix64,
    layout: Layout,
    format: DecimalFormat,
    count: usize,
) -> Vec<[u64; N]> {
    // 10^q is about 2^(q * log2(10)), and log2(10) is about 3.3219281.
    let binary = |q: i64| (q * 33_219_281).div_euclid(10_000_000);
    let (lowest, _) = format.exponents();
    let low = binary(lowest) - 16;
    let span = binary(i64::from(format.emax) + 1) + 16 - low + 1;
    let fraction_bits = layout.fraction_bits();
    let bias = i64::from(layout.ieee_bias());
    (0..count)
        .map(|_| {
            let offset = random.below(span.unsigned_abs());
            let exponent = low + i64::try_from(offset).expect("the offset fits an i64");
            let limbs: [u64; N] = core::array::from_fn(|_| random.next_u64());
            let fraction = Integer::from_digits(&limbs, Order::Lsf).keep_bits(fraction_bits);
            let sign = Integer::from(random.coin_flip()) << (layout.width - 1);
            let field = Integer::from(bias + exponent) << fraction_bits;
            to_limbs::<N>(&(sign | field | fraction))
        })
        .collect()
}

/// The number of random binary256 and binary512 values in the range of each
/// decimal format.
const IN_RANGE: usize = 3_000;

#[test]
fn from_the_wide_formats_in_the_decimal_ranges_to_decimal() {
    let mut random = SplitMix64::new(0xDEC_0256);
    let (wide, narrow) = (
        in_decimal_range::<8>(
            &mut random,
            Layout::BINARY512,
            DecimalFormat::of::<D32Bid>(),
            IN_RANGE,
        ),
        in_decimal_range::<4>(
            &mut random,
            Layout::BINARY256,
            DecimalFormat::of::<D32Bid>(),
            IN_RANGE,
        ),
    );
    check_to_decimal!(F512, wide => D32Bid, D32Dpd);
    check_to_decimal!(F256, narrow => D32Dpd, D32Bid);
    let (wide, narrow) = (
        in_decimal_range::<8>(
            &mut random,
            Layout::BINARY512,
            DecimalFormat::of::<D64Bid>(),
            IN_RANGE,
        ),
        in_decimal_range::<4>(
            &mut random,
            Layout::BINARY256,
            DecimalFormat::of::<D64Bid>(),
            IN_RANGE,
        ),
    );
    check_to_decimal!(F512, wide => D64Dpd, D64Bid);
    check_to_decimal!(F256, narrow => D64Bid, D64Dpd);
    let (wide, narrow) = (
        in_decimal_range::<8>(
            &mut random,
            Layout::BINARY512,
            DecimalFormat::of::<D128Bid>(),
            IN_RANGE,
        ),
        in_decimal_range::<4>(
            &mut random,
            Layout::BINARY256,
            DecimalFormat::of::<D128Bid>(),
            IN_RANGE,
        ),
    );
    check_to_decimal!(F512, wide => D128Bid, D128Dpd);
    check_to_decimal!(F256, narrow => D128Dpd, D128Bid);
}

/// Returns the decimal values of format `$alias` nearest the random
/// binary256 and binary512 values in the range of that format, as floaty
/// converts them. The oracle checks the conversions back to binary, so floaty
/// only picks the inputs here.
macro_rules! near_wide_values {
    ($alias:ty, $random:expr) => {{
        let format = DecimalFormat::of::<$alias>();
        let wide = in_decimal_range::<8>($random, Layout::BINARY512, format, IN_RANGE);
        let narrow = in_decimal_range::<4>($random, Layout::BINARY256, format, IN_RANGE);
        let from_wide = wide
            .into_iter()
            .map(|bits| F512::from_bits(bits).convert::<$alias>());
        let from_narrow = narrow
            .into_iter()
            .map(|bits| F256::from_bits(bits).convert::<$alias>());
        from_wide.chain(from_narrow).collect::<Vec<$alias>>()
    }};
}

#[test]
fn from_decimal_values_near_wide_binary_values() {
    let mut random = SplitMix64::new(0xDEC_0257);
    let values = near_wide_values!(D32Dpd, &mut random);
    check_to_binary!(D32Dpd, values => F256: Specials::Ieee, F512: Specials::Ieee);
    let values = near_wide_values!(D64Bid, &mut random);
    check_to_binary!(D64Bid, values => F256: Specials::Ieee, F512: Specials::Ieee);
    let values = near_wide_values!(D128Dpd, &mut random);
    check_to_binary!(D128Dpd, values => F256: Specials::Ieee, F512: Specials::Ieee);
}

#[test]
fn from_decimal32_to_binary() {
    let values: Vec<D32Bid> = decimal_values!(D32Bid, 1, 2_000, 0xD32, DECIMAL32_SPECIALS);
    check_to_binary!(D32Bid, values =>
        F16: Specials::Ieee, BF16: Specials::Ieee, TF32: Specials::Ieee);
    to_every_listed_format(&values, "D32Bid");
}

#[test]
fn from_decimal64_to_binary() {
    let values: Vec<D64Dpd> = decimal_values!(D64Dpd, 5, 2_000, 0xD64, DECIMAL64_SPECIALS);
    check_to_binary!(D64Dpd, values => F16: Specials::Ieee, BF16: Specials::Ieee);
    to_every_listed_format(&values, "D64Dpd");
}

#[test]
fn from_decimal128_to_binary() {
    let values: Vec<D128Bid> = decimal_values!(D128Bid, 97, 500, 0xD128, DECIMAL128_SPECIALS);
    check_to_binary!(D128Bid, values => F16: Specials::Ieee, BF16: Specials::Ieee);
    to_every_listed_format(&values, "D128Bid");
    let dpd: Vec<D128Dpd> = decimal_values!(D128Dpd, 389, 100, 0xD128D, DECIMAL128_SPECIALS);
    check_to_binary!(D128Dpd, dpd => TF32: Specials::Ieee);
    to_every_listed_format(&dpd, "D128Dpd");
}

/// binary32, binary64, x87 extended, and binary128 convert to every decimal
/// format in every behavior. The Intel library tests run these conversions
/// in five directions with its NaN rule only.
#[test]
fn from_the_standard_formats_to_decimal() {
    let mut random = SplitMix64::new(0xDEC_0032);
    let mut samples = |layout| sample_encodings_u128(layout, 1_500, &mut random);
    let singles: Vec<F32> = samples(Layout::BINARY32)
        .into_iter()
        .map(|bits| F32::from_bits(u32::try_from(bits).expect("the format has 32 bits")))
        .collect();
    to_every_decimal_format(&singles, "F32");
    let doubles: Vec<F64> = samples(Layout::BINARY64)
        .into_iter()
        .map(|bits| F64::from_bits(u64::try_from(bits).expect("the format has 64 bits")))
        .collect();
    to_every_decimal_format(&doubles, "F64");
    let extended: Vec<F80> = samples(Layout::X87_EXTENDED)
        .into_iter()
        .map(F80::from_bits)
        .collect();
    to_every_decimal_format(&extended, "F80");
    let quads: Vec<F128> = samples(Layout::BINARY128)
        .into_iter()
        .map(F128::from_bits)
        .collect();
    to_every_decimal_format(&quads, "F128");
}

/// Every decimal format converts to binary32, binary64, x87 extended, and
/// binary128 in every behavior.
#[test]
fn from_decimal_to_the_standard_formats() {
    let ieee = Specials::Ieee;
    let d32: Vec<D32Dpd> = decimal_values!(D32Dpd, 3, 1_000, 0xD32F, DECIMAL32_SPECIALS);
    check_to_binary!(D32Dpd, d32 => F32: ieee, F64: ieee, F80: ieee, F128: ieee);
    let d64: Vec<D64Bid> = decimal_values!(D64Bid, 11, 1_000, 0xD64F, DECIMAL64_SPECIALS);
    check_to_binary!(D64Bid, d64 => F32: ieee, F64: ieee, F80: ieee, F128: ieee);
    let d128: Vec<D128Dpd> = decimal_values!(D128Dpd, 389, 300, 0xD128F, DECIMAL128_SPECIALS);
    check_to_binary!(D128Dpd, d128 => F32: ieee, F64: ieee, F80: ieee, F128: ieee);
}
