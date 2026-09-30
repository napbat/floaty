//! Compares add, subtract, multiply, divide, square root, and fused
//! multiply-add with the MPFR arithmetic oracle, for the formats that
//! TestFloat lacks: the FP8 formats, bfloat16, TF32, binary256, binary512,
//! and a layout whose exponent field crosses a limb boundary.
//!
//! Every FP8 operand pair runs in every rounding direction.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::env::{InvalidProduct, NanPropagation, NanRule, Tininess};
use floaty::{
    BF16, Binary, Class, Decoded, Env, F4E2M1Fn, F6E2M3Fn, F6E3M2Fn, F8E4M3Fn, F8E4M3Fnuz, F8E5M2,
    F8E5M2Fnuz, F80, F256, F512, Float, Rounding, TF32,
};
use floaty_verify::arithmetic::{self, Operand, Operation};
use floaty_verify::encodings::{IntegerBit, boundary_encodings, to_limbs};
use floaty_verify::mpfr::{DIRECTIONS, Format, Specials, Value};
use floaty_verify::random::SplitMix64;
use rug::Integer;
use rug::integer::Order;

/// A 72-bit layout whose exponent field crosses a limb boundary.
type Wide72 = Float<Binary<15>, 72>;

/// Layouts at the width where addition leaves the limb width of its storage.
/// `Binary<5>` at 64 and 128 bits has exactly the precision plus 5 bits in its
/// limbs, so it adds at the limb width. `Binary<4>` at 64 bits has one bit
/// less, so it adds at twice the width.
type Edge64 = Float<Binary<5>, 64>;
type Short64 = Float<Binary<4>, 64>;
type Edge128 = Float<Binary<5>, 128>;

/// Every direction, and the other behaviors with nearest-even rounding.
fn behaviors() -> Vec<Env> {
    let mut envs: Vec<Env> = DIRECTIONS
        .iter()
        .map(|&rounding| Env::IEEE.with_rounding(rounding))
        .collect();
    envs.extend([
        Env::IEEE.with_tininess(Tininess::BeforeRounding),
        Env::IEEE.with_flush_to_zero(true),
        Env::IEEE
            .with_flush_to_zero(true)
            .with_tininess(Tininess::BeforeRounding),
        // DAZ with the x86 NaN rule, whose NaN addend takes precedence over
        // the invalid product 0 * inf.
        Env::IEEE.with_denormals_are_zero(true).with_nan(
            NanRule::new(NanPropagation::FirstOperand)
                .with_default_negative(true)
                .with_invalid_product(InvalidProduct::YieldsToNan),
        ),
        Env::IEEE
            .with_saturate(true)
            .with_rounding(Rounding::TowardPositive),
        Env::IEEE
            .with_precision(NonZeroU32::new(2))
            .with_tininess(Tininess::BeforeRounding),
    ]);
    envs
}

/// Normalizes a result for comparison: a NaN only has to be a NaN.
fn normalize(value: Value) -> Value {
    match value {
        Value::Nan { .. } => Value::Nan { negative: false },
        other => other,
    }
}

/// Returns the oracle parameters of a format.
macro_rules! format_of {
    ($alias:ty, $specials:expr) => {
        Format {
            precision: <$alias>::PRECISION,
            emin: <$alias>::EMIN,
            emax: <$alias>::EMAX,
            specials: $specials,
        }
    };
}

/// Returns an operand for the oracle.
macro_rules! operand {
    ($value:expr) => {
        Operand::<8> {
            decoded: $value.decode::<8>(),
            subnormal: $value.classify() == Class::Subnormal,
        }
    };
}

/// Compares one operation on one operand list.
macro_rules! compare {
    ($format:expr, $env:expr, $operation:expr, $ours:expr, [$($operand:expr),+]) => {{
        let (result, flags) = $ours;
        assert!(result.is_canonical(), "{result:?} is canonical");
        let ours = (normalize(Value::from_decoded(result.decode::<8>())), flags);
        let operands = [$(operand!($operand)),+];
        let (expected, expected_flags) = arithmetic::compute($operation, &operands, &$format, &$env);
        assert_eq!(ours, (normalize(expected), expected_flags), "{:?} {:?} {:?}", $operation, $env, [$($operand),+]);
        check_nan(&result.decode::<8>(), &operands, flags, &$env);
    }};
}

/// Checks the NaN that an operation with a NaN operand returns: one of the
/// NaN operands made quiet, with its sign and payload, or the default NaN of
/// an invalid operation. Every rule of these tests propagates a NaN operand.
fn check_nan(result: &Decoded<8>, operands: &[Operand<8>], flags: floaty::Flags, env: &Env) {
    let Decoded::Nan {
        negative,
        signaling,
        payload,
    } = *result
    else {
        return;
    };
    let inputs: Vec<(bool, [u64; 8])> = operands
        .iter()
        .filter_map(|operand| match operand.decoded {
            Decoded::Nan {
                negative, payload, ..
            } => Some((negative, payload)),
            _ => None,
        })
        .collect();
    if inputs.is_empty() {
        return;
    }
    assert!(!signaling, "a NaN result is quiet");
    let default = negative == env.nan.default_negative
        && payload == [0; 8]
        && flags.contains(floaty::Flags::INVALID);
    assert!(
        default || inputs.contains(&(negative, payload)),
        "the NaN result {result:?} is an operand NaN or the default NaN"
    );
}

/// Checks every operand pair and every operand of a format of `$width`
/// bits, at most 8, and random triples for the fused multiply-add.
macro_rules! small {
    ($alias:ty, $width:literal, $specials:expr, $seed:literal) => {{
        let format = format_of!($alias, $specials);
        let mut random = SplitMix64::new($seed);
        let encodings = || (0..1_u16 << $width).map(|bits| u8::try_from(bits).expect("8 bits"));
        for env in behaviors() {
            for a in encodings() {
                let x = <$alias>::from_bits(a);
                compare!(format, env, Operation::Sqrt, x.sqrt_with(env), [x]);
                for b in encodings() {
                    let y = <$alias>::from_bits(b);
                    compare!(format, env, Operation::Add, x.add_with(y, env), [x, y]);
                    compare!(format, env, Operation::Sub, x.sub_with(y, env), [x, y]);
                    compare!(format, env, Operation::Mul, x.mul_with(y, env), [x, y]);
                    compare!(format, env, Operation::Div, x.div_with(y, env), [x, y]);
                }
            }
            for _ in 0..20_000 {
                let [a, b, c] = [0, 1, 2].map(|_| {
                    let [byte, ..] = random.next_u64().to_le_bytes();
                    <$alias>::from_bits(byte)
                });
                compare!(
                    format,
                    env,
                    Operation::MulAdd,
                    a.mul_add_with(b, c, env),
                    [a, b, c]
                );
                let cancel = cancelling_addend(a, b);
                compare!(
                    format,
                    env,
                    Operation::MulAdd,
                    a.mul_add_with(b, cancel, env),
                    [a, b, cancel]
                );
            }
        }
    }};
}

#[test]
fn every_fp8_operand_pair() {
    small!(F8E4M3Fn, 8, Specials::NoInf, 1);
    small!(F8E5M2, 8, Specials::Ieee, 2);
    small!(F8E4M3Fnuz, 8, Specials::Fnuz, 3);
    small!(F8E5M2Fnuz, 8, Specials::Fnuz, 4);
}

#[test]
fn every_mx_operand_pair() {
    small!(F4E2M1Fn, 4, Specials::Finite, 5);
    small!(F6E2M3Fn, 6, Specials::Finite, 6);
    small!(F6E3M2Fn, 6, Specials::Finite, 7);
}

/// Returns the negated product `a * b`, rounded to nearest. As an addend, it
/// cancels all but the rounding error of the product, which exercises the
/// widest cancellation of a fused multiply-add.
fn cancelling_addend<S: floaty::format::Standard<W>, const W: usize>(
    a: Float<S, W>,
    b: Float<S, W>,
) -> Float<S, W> {
    let (product, _) = a.mul_with(b, Env::IEEE);
    // The zero from `product - product`, minus the product, negates it.
    let (zero, _) = product.sub_with(product, Env::IEEE);
    zero.sub_with(product, Env::IEEE).0
}

/// Returns the special encodings of an IEEE format `width` bits wide: both
/// zeros, the smallest subnormals, both infinities, and quiet and signaling
/// NaNs of both signs.
fn special_encodings(width: u32, exponent_bits: u32) -> Vec<Integer> {
    let fraction_bits = width - 1 - exponent_bits;
    let field: Integer = ((Integer::from(1) << exponent_bits) - 1u32) << fraction_bits;
    let quiet = Integer::from(1) << (fraction_bits - 1);
    let magnitudes = [
        Integer::ZERO,
        Integer::from(1),
        field.clone(),
        field.clone() | &quiet,
        field | 1u32,
    ];
    let sign = Integer::from(1) << (width - 1);
    magnitudes
        .iter()
        .flat_map(|magnitude| [magnitude.clone(), magnitude.clone() | &sign])
        .collect()
}

/// Returns operand encodings of a format `width` bits wide: the boundary
/// encodings, random encodings, and, after each random encoding, a nearby
/// encoding with the other sign, which cancels in a sum.
fn encodings(
    width: u32,
    exponent_bits: u32,
    count: usize,
    random: &mut SplitMix64,
) -> Vec<Integer> {
    let mask: Integer = (Integer::from(1) << width) - 1u32;
    let mut values = boundary_encodings(width, exponent_bits, IntegerBit::Implicit);
    let limbs = width.div_ceil(64);
    for _ in 0..count {
        let digits: Vec<u64> = (0..limbs).map(|_| random.next_u64()).collect();
        let value: Integer = Integer::from_digits(&digits, Order::Lsf) & &mask;
        let flip = Integer::from(random.next_u64() & 0xFF);
        let near: Integer = (value.clone() ^ flip) ^ (Integer::from(1) << (width - 1));
        values.push(value);
        values.push(near);
    }
    values
}

/// Checks random operand pairs and triples of a wider format.
macro_rules! wide {
    ($alias:ty, $width:literal, $exponent_bits:literal, $bits:expr, $count:literal, $seed:literal) => {{
        let format = format_of!($alias, Specials::Ieee);
        let mut random = SplitMix64::new($seed);
        let values: Vec<$alias> = encodings($width, $exponent_bits, $count, &mut random)
            .iter()
            .map(|encoding| <$alias>::from_bits($bits(encoding)))
            .collect();
        let specials: Vec<$alias> = special_encodings($width, $exponent_bits)
            .iter()
            .map(|encoding| <$alias>::from_bits($bits(encoding)))
            .collect();
        for env in behaviors() {
            for &x in &specials {
                for &y in &specials {
                    compare!(format, env, Operation::Add, x.add_with(y, env), [x, y]);
                    compare!(format, env, Operation::Sub, x.sub_with(y, env), [x, y]);
                    compare!(format, env, Operation::Mul, x.mul_with(y, env), [x, y]);
                    compare!(format, env, Operation::Div, x.div_with(y, env), [x, y]);
                    for &z in &specials {
                        compare!(
                            format,
                            env,
                            Operation::MulAdd,
                            x.mul_add_with(y, z, env),
                            [x, y, z]
                        );
                    }
                }
            }
            for pair in values.chunks_exact(2) {
                let [x, y] = [pair[0], pair[1]];
                compare!(format, env, Operation::Add, x.add_with(y, env), [x, y]);
                compare!(format, env, Operation::Sub, x.sub_with(y, env), [x, y]);
                compare!(format, env, Operation::Mul, x.mul_with(y, env), [x, y]);
                compare!(format, env, Operation::Div, x.div_with(y, env), [x, y]);
                compare!(format, env, Operation::Sqrt, x.sqrt_with(env), [x]);
            }
            for triple in values.chunks_exact(3) {
                let [a, b, c] = [triple[0], triple[1], triple[2]];
                compare!(
                    format,
                    env,
                    Operation::MulAdd,
                    a.mul_add_with(b, c, env),
                    [a, b, c]
                );
                let cancel = cancelling_addend(a, b);
                compare!(
                    format,
                    env,
                    Operation::MulAdd,
                    a.mul_add_with(b, cancel, env),
                    [a, b, cancel]
                );
            }
        }
    }};
}

fn to_u16(encoding: &Integer) -> u16 {
    encoding.to_u16().expect("16 bits")
}

fn to_u64(encoding: &Integer) -> u64 {
    encoding.to_u64().expect("a 64-bit encoding fits a u64")
}

fn to_u32(encoding: &Integer) -> u32 {
    encoding.to_u32().expect("32 bits")
}

fn to_u128(encoding: &Integer) -> u128 {
    encoding.to_u128().expect("128 bits")
}

#[test]
fn bfloat16_tf32_and_a_crossing_layout() {
    wide!(BF16, 16, 8, to_u16, 20_000, 16);
    wide!(TF32, 19, 8, to_u32, 20_000, 19);
    wide!(Wide72, 72, 15, to_u128, 10_000, 72);
}

#[test]
fn layouts_at_the_width_of_the_addition() {
    wide!(Edge64, 64, 5, to_u64, 20_000, 64);
    wide!(Short64, 64, 4, to_u64, 20_000, 65);
    wide!(Edge128, 128, 5, to_u128, 10_000, 128);
}

#[test]
fn binary256_and_binary512() {
    wide!(F256, 256, 19, to_limbs::<4>, 3_000, 256);
    wide!(F512, 512, 23, to_limbs::<8>, 3_000, 512);
}

/// Checks x87 arithmetic and fused multiply-add with precision control at 24,
/// 53, and 64 bits, with canonical operands and pseudo-denormals. The x87 unit
/// has no fused multiply-add, so MPFR is its only reference.
#[test]
fn x87_arithmetic_with_precision_control() {
    let format = format_of!(F80, Specials::Ieee);
    let mut random = SplitMix64::new(0x0087_F3A0);
    let mask = (1_u128 << 80) - 1;
    // Canonical encodings: the integer bit is set exactly when the exponent
    // field is not zero.
    let canonical = |bits: u128| {
        let field = (bits >> 64) & 0x7FFF;
        let integer = 1_u128 << 63;
        if field == 0 {
            bits & !integer
        } else {
            bits | integer
        }
    };
    let values: Vec<F80> = (0..6_000)
        .map(|index| {
            let bits = random.next_u128() & mask;
            match index % 6 {
                // Bias a third of the exponents toward the subnormal range.
                0 | 3 => F80::from_bits(canonical(bits & !(0x7FC0_u128 << 64))),
                // A pseudo-denormal: exponent field 0 with the integer bit set.
                5 => F80::from_bits((bits & !(0x7FFF_u128 << 64)) | (1 << 63)),
                _ => F80::from_bits(canonical(bits)),
            }
        })
        .collect();
    for precision in [None, NonZeroU32::new(24), NonZeroU32::new(53)] {
        for rounding in DIRECTIONS {
            let env = Env::IEEE.with_rounding(rounding).with_precision(precision);
            for pair in values.chunks_exact(2) {
                let [x, y] = [pair[0], pair[1]];
                compare!(format, env, Operation::Add, x.add_with(y, env), [x, y]);
                compare!(format, env, Operation::Sub, x.sub_with(y, env), [x, y]);
                compare!(format, env, Operation::Mul, x.mul_with(y, env), [x, y]);
                compare!(format, env, Operation::Div, x.div_with(y, env), [x, y]);
                compare!(format, env, Operation::Sqrt, x.sqrt_with(env), [x]);
            }
            for triple in values.chunks_exact(3) {
                let [a, b, c] = [triple[0], triple[1], triple[2]];
                compare!(
                    format,
                    env,
                    Operation::MulAdd,
                    a.mul_add_with(b, c, env),
                    [a, b, c]
                );
                let cancel = cancelling_addend(a, b);
                compare!(
                    format,
                    env,
                    Operation::MulAdd,
                    a.mul_add_with(b, cancel, env),
                    [a, b, cancel]
                );
            }
        }
    }
}
