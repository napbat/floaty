//! Compares add, subtract, multiply, divide, square root, and fused
//! multiply-add with the MPFR arithmetic oracle. The formats are those that
//! TestFloat lacks: the FP8 and MX formats, bfloat16, TF32, the wide formats
//! from binary160 to binary512, and a layout whose exponent field crosses a
//! limb boundary. binary16, binary32, binary64, and binary128 run here too,
//! for the behaviors that TestFloat lacks. These are two rounding directions,
//! saturation, precision limits, flush-to-zero with tininess before rounding,
//! and the NaN rules of x87, PowerPC, and Arm.
//!
//! Every FP8, FP6, and FP4 operand pair runs in every behavior of
//! `behaviors`.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::num::NonZeroU32;

use floaty::env::{FusedNanOrder, InvalidProduct, NanPropagation, NanRule, Tininess};
use floaty::{
    BF16, Binary, Decoded, Env, F16, F32, F64, F80, F128, F256, F512, Float, Rounding, TF32,
};
use floaty_verify::arithmetic::{self, Operation};
use floaty_verify::encodings::{Layout, boundary_encodings, to_limbs};
use floaty_verify::mpfr::{DIRECTIONS, Format, Operand, Specials, Value};
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
        // The other NaN rules: the x87 rule, the larger significand with the
        // addend first, the PowerPC rule, the rule of the Arm `FPMulAdd`
        // pseudocode, and the default NaN.
        Env::IEEE.with_nan(Env::X87.nan),
        Env::IEEE.with_nan(
            NanRule::new(NanPropagation::LargerSignificand)
                .with_default_negative(true)
                .with_fused_order(FusedNanOrder::AddendFirst),
        ),
        Env::IEEE.with_nan(
            NanRule::new(NanPropagation::FirstOperand)
                .with_fused_order(FusedNanOrder::AddendSecond)
                .with_invalid_product(InvalidProduct::SignalsAndYieldsToNan),
        ),
        Env::IEEE.with_nan(
            NanRule::new(NanPropagation::SignalingFirst)
                .with_fused_order(FusedNanOrder::AddendFirst),
        ),
        Env::IEEE.with_nan(NanRule::new(NanPropagation::DefaultNan).with_default_negative(true)),
    ]);
    envs
}

/// Compares one operation on one operand list. A NaN result must be the NaN
/// that the NaN rule gives, with its sign and payload.
macro_rules! compare {
    ($format:expr, $env:expr, $operation:expr, $ours:expr, [$($operand:expr),+]) => {{
        let (result, flags) = $ours;
        assert!(result.is_canonical(), "{result:?} is canonical");
        let decoded = result.decode::<8>();
        let payload = match decoded {
            Decoded::Nan { signaling, payload, .. } => {
                assert!(!signaling, "{result:?}: a NaN result is quiet");
                payload
            }
            _ => [0; 8],
        };
        let ours = (Value::from_decoded(decoded), payload, flags);
        let operands = [$(Operand::of($operand)),+];
        let expected = arithmetic::compute($operation, &operands, &$format, &$env);
        assert_eq!(
            ours,
            (expected.value, expected.payload, expected.flags),
            "{:?} {:?} {:?}",
            $operation,
            $env,
            [$($operand),+]
        );
    }};
}

/// Checks every operand pair and every operand of a format of `$width`
/// bits, at most 8, and random triples for the fused multiply-add.
macro_rules! small {
    ($alias:ty, $width:literal, $specials:expr, $seed:literal) => {{
        let format = Format::of::<$alias>($specials);
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

/// Checks one format of the small format lists with `small!`.
macro_rules! small_format {
    ($alias:ident, $standard:ty, $width:literal, $specials:expr, $seed:literal, $block:literal) => {
        small!(floaty::$alias, $width, $specials, $seed)
    };
}

#[test]
fn every_fp8_operand_pair() {
    floaty_verify::for_each_fp8_format!(small_format);
}

#[test]
fn every_mx_operand_pair() {
    floaty_verify::for_each_mx_format!(small_format);
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
fn special_encodings(layout: Layout) -> Vec<Integer> {
    let Layout {
        width,
        exponent_bits,
        ..
    } = layout;
    let fraction_bits = layout.fraction_bits();
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
fn encodings(layout: Layout, count: usize, random: &mut SplitMix64) -> Vec<Integer> {
    let width = layout.width;
    let mask: Integer = (Integer::from(1) << width) - 1u32;
    let nan_field: Integer =
        ((Integer::from(1) << layout.exponent_bits) - 1u32) << layout.fraction_bits();
    let mut values = boundary_encodings(layout);
    let limbs = width.div_ceil(64);
    for index in 0..count {
        let digits: Vec<u64> = (0..limbs).map(|_| random.next_u64()).collect();
        let mut value: Integer = Integer::from_digits(&digits, Order::Lsf) & &mask;
        // One encoding in 16 gets the exponent field of the NaNs, so that NaNs
        // with random payloads occur in every format.
        if index % 16 == 0 {
            value |= &nan_field;
        }
        let flip = Integer::from(random.next_u64() & 0xFF);
        let near: Integer = (value.clone() ^ flip) ^ (Integer::from(1) << (width - 1));
        values.push(value);
        values.push(near);
    }
    values
}

/// Checks random operand pairs and triples of a wider format.
macro_rules! wide {
    ($alias:ty, $layout:expr, $bits:expr, $count:literal, $seed:literal) => {{
        let format = Format::of::<$alias>(Specials::Ieee);
        let mut random = SplitMix64::new($seed);
        let values: Vec<$alias> = encodings($layout, $count, &mut random)
            .iter()
            .map(|encoding| <$alias>::from_bits($bits(encoding)))
            .collect();
        let specials: Vec<$alias> = special_encodings($layout)
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
    wide!(BF16, Layout::BFLOAT16, to_u16, 20_000, 16);
    wide!(TF32, Layout::TF32, to_u32, 20_000, 19);
    wide!(Wide72, Layout::ieee(72, 15), to_u128, 10_000, 72);
}

#[test]
fn layouts_at_the_width_of_the_addition() {
    wide!(Edge64, Layout::ieee(64, 5), to_u64, 20_000, 64);
    wide!(Short64, Layout::ieee(64, 4), to_u64, 20_000, 65);
    wide!(Edge128, Layout::ieee(128, 5), to_u128, 10_000, 128);
}

#[test]
fn binary16_to_binary128() {
    wide!(F16, Layout::BINARY16, to_u16, 20_000, 0x16);
    wide!(F32, Layout::BINARY32, to_u32, 20_000, 0x32);
    wide!(F64, Layout::BINARY64, to_u64, 20_000, 0x64);
    wide!(F128, Layout::BINARY128, to_u128, 10_000, 0x128);
}

#[test]
fn binary256_and_binary512() {
    wide!(F256, Layout::BINARY256, to_limbs::<4>, 3_000, 256);
    wide!(F512, Layout::BINARY512, to_limbs::<8>, 3_000, 512);
}

/// Checks one format of the wide format list with `wide!`, but binary256
/// and binary512, which `binary256_and_binary512` checks with more operands.
macro_rules! other_wide_format {
    ($alias:ident, 256, $exponent_bits:literal, $limbs:literal) => {};
    ($alias:ident, 512, $exponent_bits:literal, $limbs:literal) => {};
    ($alias:ident, $width:literal, $exponent_bits:literal, $limbs:literal) => {
        wide!(
            floaty::$alias,
            Layout::ieee($width, $exponent_bits),
            to_limbs::<$limbs>,
            1_000,
            $width
        )
    };
}

#[test]
fn the_other_wide_formats() {
    floaty_verify::for_each_wide_format!(other_wide_format);
}

/// Checks the arithmetic of the values of one format in the static mode
/// `$mode`, which must name the behavior `$env`.
macro_rules! in_mode {
    ($values:expr, $alias:ty, $mode:ty, $env:expr) => {{
        let env: Env = $env;
        assert_eq!(
            <$mode as floaty::env::Mode>::ENV,
            env,
            "the mode names the behavior"
        );
        let format = Format::of::<$alias>(Specials::Ieee);
        let mode = <$mode>::default();
        let values: Vec<_> = $values
            .iter()
            .map(|value| value.with_mode::<$mode>())
            .collect();
        for pair in values.chunks_exact(2) {
            let [x, y] = [pair[0], pair[1]];
            compare!(format, env, Operation::Add, x.add_with(y, mode), [x, y]);
            compare!(format, env, Operation::Sub, x.sub_with(y, mode), [x, y]);
            compare!(format, env, Operation::Mul, x.mul_with(y, mode), [x, y]);
            compare!(format, env, Operation::Div, x.div_with(y, mode), [x, y]);
            compare!(format, env, Operation::Sqrt, x.sqrt_with(mode), [x]);
        }
        for triple in values.chunks_exact(3) {
            let [a, b, c] = [triple[0], triple[1], triple[2]];
            compare!(
                format,
                env,
                Operation::MulAdd,
                a.mul_add_with(b, c, mode),
                [a, b, c]
            );
        }
    }};
}

/// Checks static modes against the oracle. `Rounded<Ieee, R>` runs in every
/// direction, with the two directions that TestFloat and the processor lack.
/// `mode::X87` limits binary128 to 64 bits, and `FullPrecision<X87>` keeps the
/// NaN rule of x87 without the limit.
#[test]
fn static_modes() {
    let mut random = SplitMix64::new(0x0057_471C);
    let doubles: Vec<F64> = encodings(Layout::BINARY64, 2_000, &mut random)
        .iter()
        .map(|encoding| F64::from_bits(to_u64(encoding)))
        .collect();
    let quads: Vec<F128> = encodings(Layout::BINARY128, 1_000, &mut random)
        .iter()
        .map(|encoding| F128::from_bits(to_u128(encoding)))
        .collect();
    for rounding in DIRECTIONS {
        let env = Env::IEEE.with_rounding(rounding);
        floaty_verify::with_rounding_mode!(rounding, floaty::mode::Ieee, Mode => {
            in_mode!(doubles, F64, Mode, env);
            in_mode!(quads, F128, Mode, env);
        });
    }
    in_mode!(quads, F128, floaty::mode::X87, Env::X87);
    in_mode!(
        quads,
        F128,
        floaty::mode::FullPrecision<floaty::mode::X87>,
        Env::X87.with_precision(None)
    );
}

/// Checks x87 arithmetic and fused multiply-add with precision control at 24,
/// 53, and 64 bits, with canonical operands and pseudo-denormals. The x87 unit
/// has no fused multiply-add, so MPFR is its only reference.
#[test]
fn x87_arithmetic_with_precision_control() {
    let format = Format::of::<F80>(Specials::Ieee);
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
