//! The entry points of the host paths, the same on every architecture that
//! has them. The module of the architecture reads its floating-point
//! environment.
//!
//! Every floating-point instruction of a path runs in inline assembly, and
//! the NaN tests use integer instructions. LLVM assumes the default
//! floating-point environment, so it can move a Rust float operation, even a
//! comparison, above the check of the environment. There an unmasked
//! exception traps.
//!
//! The binary16 and bfloat16 paths compute in binary32 and round twice.
//! binary32 holds `2p + 2` bits of each format, so the two roundings of add,
//! subtract, multiply, divide, and square root give the correctly rounded
//! result: Figueroa, "When is double rounding innocuous?", ACM SIGNUM
//! Newsletter 30(3), 1995. Every binary32 result of binary16 operands is
//! normal. A binary32 result of bfloat16 operands can be subnormal, but it
//! still holds 16 bits more than a bfloat16 subnormal, which has at most 7
//! bits. The theorem does not hold for the fused multiply-add, or for a
//! conversion from binary64 to binary16 through binary32: `1 + 2^-11 +
//! 2^-40` rounds to `1 + 2^-11` in binary32, and then to 1, not to
//! `1 + 2^-10`. So those operations take no path through binary32.

use core::cmp::Ordering;

use super::Operation;
use super::environment::{self, default_environment};
use crate::env::{Env, Rounding};
use crate::format::Standard;
use crate::format::internal::{Host, LimbConversion, MinMax};
use crate::limbs::Limbs;

/// Returns `true` when the host unit gives the results of `env` for a format
/// of `precision` bits.
#[inline]
const fn compatible(env: &Env, precision: u32) -> bool {
    // Every field is named, so a new field of `Env` needs a decision here.
    // Tininess and saturation change only flags and formats without an
    // infinity, the NaN rule applies only to a NaN result, and the total
    // order applies to no arithmetic.
    let Env {
        rounding,
        flush_to_zero,
        denormals_are_zero,
        tininess: _,
        nan: _,
        precision: limit,
        saturate: _,
        total_order: _,
    } = *env;
    let full_precision = match limit {
        None => true,
        Some(limit) => limit.get() >= precision,
    };
    matches!(rounding, Rounding::TiesToEven)
        && !flush_to_zero
        && !denormals_are_zero
        && full_precision
}

/// Returns `true` for the bits of a binary32 NaN.
#[inline]
pub(super) const fn nan_32(bits: u32) -> bool {
    bits & 0x7FFF_FFFF > 0x7F80_0000
}

/// Returns `true` for the bits of a binary16 NaN.
#[inline]
pub(super) const fn nan_16(bits: u16) -> bool {
    bits & 0x7FFF > 0x7C00
}

/// Returns `true` for the bits of a bfloat16 NaN.
#[inline]
pub(super) const fn nan_bfloat(bits: u16) -> bool {
    bits & 0x7FFF > 0x7F80
}

/// Returns `true` for the bits of a binary64 NaN.
#[inline]
pub(super) const fn nan_64(bits: u64) -> bool {
    bits & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000
}

/// Returns the host value of a binary32 encoding.
#[inline]
pub(super) fn single<S: Standard<W>, const W: usize>(bits: S::Bits) -> f32 {
    let low = bits.to_limbs().limb(0);
    f32::from_bits(u32::try_from(low).expect("a binary32 encoding has 32 bits"))
}

/// Returns the host value of a binary16 encoding, widened exactly to binary32
/// in a host instruction, or `None` in a build without the instruction.
#[inline]
fn half<S: Standard<W>, const W: usize>(bits: S::Bits) -> Option<f32> {
    let low = bits.to_limbs().limb(0);
    environment::widen_half(u16::try_from(low).expect("a binary16 encoding has 16 bits"))
}

/// Returns the encoding of a binary32 result rounded to binary16 in a host
/// instruction, or `None` for a NaN, which goes back to the engine.
#[inline]
fn half_encoding<S: Standard<W>, const W: usize>(result: f32) -> Option<S::Bits> {
    if nan_32(result.to_bits()) {
        return None;
    }
    encoding::<S, W>(u64::from(environment::narrow_half(result)?), false)
}

/// Returns the host value of a bfloat16 encoding, widened exactly to
/// binary32: a bfloat16 encoding is the high half of a binary32 encoding.
#[inline]
fn bfloat<S: Standard<W>, const W: usize>(bits: S::Bits) -> f32 {
    let low = bits.to_limbs().limb(0);
    f32::from_bits(u32::try_from(low << 16).expect("a bfloat16 encoding has 16 bits"))
}

/// Returns the encoding of a binary32 result rounded to bfloat16 in a host
/// instruction, or `None` for a NaN or in a build without the instruction.
#[inline]
fn bfloat_encoding<S: Standard<W>, const W: usize>(result: f32) -> Option<S::Bits> {
    if nan_32(result.to_bits()) {
        return None;
    }
    encoding::<S, W>(u64::from(environment::narrow_bfloat(result)?), false)
}

/// Returns the host value of a binary64 encoding.
#[inline]
pub(super) fn double<S: Standard<W>, const W: usize>(bits: S::Bits) -> f64 {
    f64::from_bits(bits.to_limbs().limb(0))
}

/// Returns the encoding of a host result, or `None` for a NaN, which goes
/// back to the engine.
#[inline]
fn encoding<S: Standard<W>, const W: usize>(result: u64, nan: bool) -> Option<S::Bits> {
    if nan {
        return None;
    }
    let limbs = <S::Bits as LimbConversion>::Limbs::ZERO.with_limb(0, result);
    Some(S::Bits::from_limbs(limbs))
}

/// Returns the encoding of a binary32 result, or `None` for a NaN.
#[inline]
pub(super) fn single_encoding<S: Standard<W>, const W: usize>(result: f32) -> Option<S::Bits> {
    let bits = result.to_bits();
    encoding::<S, W>(u64::from(bits), nan_32(bits))
}

/// Returns the encoding of a binary64 result, or `None` for a NaN.
#[inline]
pub(super) fn double_encoding<S: Standard<W>, const W: usize>(result: f64) -> Option<S::Bits> {
    let bits = result.to_bits();
    encoding::<S, W>(bits, nan_64(bits))
}

/// Returns the limbs of an x87 extended encoding.
#[inline]
fn extended<S: Standard<W>, const W: usize>(bits: S::Bits) -> [u64; 2] {
    bits.to_limbs().resize()
}

/// Returns the encoding of an x87 extended result.
#[inline]
fn extended_encoding<S: Standard<W>, const W: usize>(result: [u64; 2]) -> S::Bits {
    S::Bits::from_limbs(result.resize())
}

/// Returns `true` when the mode and the environment of the host unit that
/// computes the host kind `host` allow a host path for a format of
/// `precision` bits. The x87 unit computes x87 extended precision, and the
/// SSE unit or the AArch64 unit computes the other kinds.
#[inline]
pub(super) fn ready_for(host: Host, env: &Env, precision: u32) -> bool {
    let unit = match host {
        Host::Extended => environment::x87_environment(),
        Host::None | Host::Half | Host::BFloat | Host::Single | Host::Double => {
            default_environment()
        }
    };
    compatible(env, precision) && unit
}

/// Returns the result of `operation` from the host unit, or `None` when the
/// path does not apply or the result is a NaN.
#[inline]
pub fn binary<S: Standard<W>, const W: usize>(
    left: S::Bits,
    right: S::Bits,
    operation: Operation,
    env: &Env,
) -> Option<S::Bits> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::None => None,
        Host::Single => {
            let (left, right) = (single::<S, W>(left), single::<S, W>(right));
            single_encoding::<S, W>(environment::binary_f32(left, right, operation))
        }
        Host::Double => {
            let (left, right) = (double::<S, W>(left), double::<S, W>(right));
            double_encoding::<S, W>(environment::binary_f64(left, right, operation))
        }
        Host::Half => {
            let (left, right) = (half::<S, W>(left)?, half::<S, W>(right)?);
            half_encoding::<S, W>(environment::binary_f32(left, right, operation))
        }
        Host::BFloat => {
            let (left, right) = (bfloat::<S, W>(left), bfloat::<S, W>(right));
            bfloat_encoding::<S, W>(environment::binary_f32(left, right, operation))
        }
        Host::Extended => {
            let (left, right) = (extended::<S, W>(left), extended::<S, W>(right));
            environment::x87_binary(&left, &right, operation).map(extended_encoding::<S, W>)
        }
    }
}

/// Returns the square root from the host unit, or `None` when the path does
/// not apply or the result is a NaN.
#[inline]
pub fn sqrt<S: Standard<W>, const W: usize>(value: S::Bits, env: &Env) -> Option<S::Bits> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::None => None,
        Host::Single => single_encoding::<S, W>(environment::sqrt_f32(single::<S, W>(value))),
        Host::Double => double_encoding::<S, W>(environment::sqrt_f64(double::<S, W>(value))),
        Host::Half => half_encoding::<S, W>(environment::sqrt_f32(half::<S, W>(value)?)),
        Host::BFloat => bfloat_encoding::<S, W>(environment::sqrt_f32(bfloat::<S, W>(value))),
        Host::Extended => {
            environment::x87_sqrt(&extended::<S, W>(value)).map(extended_encoding::<S, W>)
        }
    }
}

/// Returns `left * right + addend`, rounded once, from the host unit, or
/// `None` when the path does not apply or the result is a NaN.
#[inline]
pub fn mul_add<S: Standard<W>, const W: usize>(
    left: S::Bits,
    right: S::Bits,
    addend: S::Bits,
    env: &Env,
) -> Option<S::Bits> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        // Two roundings of a fused multiply-add can differ from one, and the
        // x87 unit has no fused multiply-add.
        Host::None | Host::BFloat | Host::Extended => None,
        // `FEAT_FP16` rounds the binary16 result once.
        Host::Half => {
            let [a, b, c] = [left, right, addend].map(|bits| {
                u16::try_from(bits.to_limbs().limb(0)).expect("a binary16 encoding has 16 bits")
            });
            let result = environment::mul_add_f16(a, b, c)?;
            encoding::<S, W>(u64::from(result), nan_16(result))
        }
        Host::Single => {
            let (a, b) = (single::<S, W>(left), single::<S, W>(right));
            single_encoding::<S, W>(environment::mul_add_f32(a, b, single::<S, W>(addend))?)
        }
        Host::Double => {
            let (a, b) = (double::<S, W>(left), double::<S, W>(right));
            double_encoding::<S, W>(environment::mul_add_f64(a, b, double::<S, W>(addend))?)
        }
    }
}

/// Returns `true` when the mode and the environment of the host unit allow a
/// rounding to an integral value in the direction of `env`. The instructions
/// take the direction in their encoding, not from the environment, so every
/// other field must allow a host path, and the environment must round to
/// nearest even.
#[inline]
pub(super) fn ready_for_integral(host: Host, env: &Env, precision: u32) -> bool {
    ready_for(host, &env.with_rounding(Rounding::TiesToEven), precision)
}

/// Returns the value rounded to an integral value in the rounding direction
/// of `env` from the host unit, or `None` when the path does not apply, the
/// unit has no instruction for the direction, or the result is a NaN.
#[inline]
pub fn round_to_integral<S: Standard<W>, const W: usize>(
    value: S::Bits,
    env: &Env,
) -> Option<S::Bits> {
    if !ready_for_integral(S::HOST, env, S::PRECISION) {
        return None;
    }
    let rounding = env.rounding;
    match S::HOST {
        Host::None => None,
        Host::Single => {
            single_encoding::<S, W>(environment::round_f32(single::<S, W>(value), rounding)?)
        }
        Host::Double => {
            double_encoding::<S, W>(environment::round_f64(double::<S, W>(value), rounding)?)
        }
        // The integral value of a binary16 value in each direction is a
        // binary16 value, so the narrowing is exact.
        Host::Half => {
            half_encoding::<S, W>(environment::round_f32(half::<S, W>(value)?, rounding)?)
        }
        // The integral value of a bfloat16 value in each direction is a
        // bfloat16 value, so the low 16 bits of the binary32 result are zero,
        // and a shift narrows it.
        Host::BFloat => {
            let bits = environment::round_f32(bfloat::<S, W>(value), rounding)?.to_bits();
            encoding::<S, W>(u64::from(bits >> 16), nan_32(bits))
        }
        // `FRNDINT` rounds in the direction of the control word, which the
        // path requires to be to nearest. The precision control does not
        // apply to `FRNDINT`, Intel SDM Volume 1, section 8.1.5.2, so the
        // integral value rounds once.
        Host::Extended => {
            if !matches!(rounding, Rounding::TiesToEven) {
                return None;
            }
            environment::x87_round(&extended::<S, W>(value)).map(extended_encoding::<S, W>)
        }
    }
}

/// Returns the value rounded to a 64-bit integer to nearest even from the
/// host unit, or `None` when the path does not apply, for a NaN, and for a
/// result that the host cannot tell from a value out of range.
#[inline]
pub fn to_int<S: Standard<W>, const W: usize>(value: S::Bits, env: &Env) -> Option<i64> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::None => None,
        Host::Single => environment::to_int_f32(single::<S, W>(value)),
        Host::Double => environment::to_int_f64(double::<S, W>(value)),
        Host::Half => environment::to_int_f32(half::<S, W>(value)?),
        Host::BFloat => environment::to_int_f32(bfloat::<S, W>(value)),
        Host::Extended => environment::x87_to_int(&extended::<S, W>(value)),
    }
}

/// Returns a 64-bit integer rounded to the format by the host unit, or
/// `None` when the path does not apply.
#[inline]
pub fn from_int<S: Standard<W>, const W: usize>(value: i64, env: &Env) -> Option<S::Bits> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        // An integer through binary32 to bfloat16 rounds twice.
        Host::None | Host::BFloat => None,
        Host::Single => {
            let result = environment::from_int_f32(value);
            encoding::<S, W>(u64::from(result.to_bits()), false)
        }
        Host::Double => encoding::<S, W>(environment::from_int_f64(value).to_bits(), false),
        // An integer below 2^16 in magnitude converts to binary32 exactly. A
        // larger one overflows binary16 after either rounding, so the result
        // rounds once in effect.
        Host::Half => half_encoding::<S, W>(environment::from_int_f32(value)),
        // x87 extended precision holds every 64-bit integer exactly.
        Host::Extended => environment::x87_from_int(value).map(extended_encoding::<S, W>),
    }
}

/// Returns the order of two values from the host unit, as the quiet
/// predicates give it, or `None` when the path does not apply or the values
/// are unordered. An unordered pair holds a NaN, whose flags the engine
/// computes.
#[inline]
pub fn compare<S: Standard<W>, const W: usize>(
    left: S::Bits,
    right: S::Bits,
    env: &Env,
) -> Option<Ordering> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::None | Host::Extended => None,
        Host::Single => environment::compare_f32(single::<S, W>(left), single::<S, W>(right)),
        Host::Double => environment::compare_f64(double::<S, W>(left), double::<S, W>(right)),
        // The widenings are exact, so the order is the order of the values.
        Host::Half => environment::compare_f32(half::<S, W>(left)?, half::<S, W>(right)?),
        Host::BFloat => environment::compare_f32(bfloat::<S, W>(left), bfloat::<S, W>(right)),
    }
}

/// Returns `true` for two binary32 encodings where the minimum and maximum
/// instructions of the host differ from the operations: a NaN operand, or two
/// zeros. `MINSS` then gives the second operand, and every operation orders
/// `-0` below `+0`. The test has no branch, so LLVM tests the lanes of a
/// packed path at once.
#[inline]
pub(super) fn min_max_differs(left: u32, right: u32) -> bool {
    let nan = (left & 0x7FFF_FFFF).max(right & 0x7FFF_FFFF) > 0x7F80_0000;
    // The shift drops the sign bits, so the result is zero for two zeros.
    let zeros = (left | right) << 1 == 0;
    nan | zeros
}

/// Returns `true` for two binary64 encodings where the minimum and maximum
/// instructions of the host differ from the operations, as
/// `min_max_differs` does for binary32.
#[inline]
pub(super) fn min_max_differs_64(left: u64, right: u64) -> bool {
    let magnitude = 0x7FFF_FFFF_FFFF_FFFF;
    let nan = (left & magnitude).max(right & magnitude) > 0x7FF0_0000_0000_0000;
    let zeros = (left | right) << 1 == 0;
    nan | zeros
}

/// Returns the smaller or the larger of two binary32 values from the host
/// unit, as `operation` selects.
#[inline]
pub(super) fn min_max_f32(left: f32, right: f32, operation: MinMax) -> f32 {
    if operation.is_minimum() {
        environment::min_f32(left, right)
    } else {
        environment::max_f32(left, right)
    }
}

/// Returns the smaller or the larger of two binary64 values from the host
/// unit, as `operation` selects.
#[inline]
pub(super) fn min_max_f64(left: f64, right: f64, operation: MinMax) -> f64 {
    if operation.is_minimum() {
        environment::min_f64(left, right)
    } else {
        environment::max_f64(left, right)
    }
}

/// Returns the result of the minimum or maximum operation `operation` from
/// the host unit, or `None` when the path does not apply, an operand is a
/// NaN, or both operands are zeros. For every other pair, each operation of
/// the families selects the smaller or the larger operand, as the host
/// instructions do. The result is an operand, so no rounding occurs.
#[inline]
pub fn min_max<S: Standard<W>, const W: usize>(
    left: S::Bits,
    right: S::Bits,
    operation: MinMax,
    env: &Env,
) -> Option<S::Bits> {
    if !ready_for(S::HOST, env, S::PRECISION) {
        return None;
    }
    // The widened values of binary16 and bfloat16 are exact, and the
    // instruction returns one of them, so the result is the operand that the
    // instruction returns.
    let select = |a: f32, b: f32| -> Option<S::Bits> {
        if min_max_differs(a.to_bits(), b.to_bits()) {
            return None;
        }
        let result = min_max_f32(a, b, operation);
        Some(if result.to_bits() == a.to_bits() {
            left
        } else {
            right
        })
    };
    match S::HOST {
        Host::None | Host::Extended => None,
        Host::Single => select(single::<S, W>(left), single::<S, W>(right)),
        Host::Double => {
            let (a, b) = (double::<S, W>(left), double::<S, W>(right));
            if min_max_differs_64(a.to_bits(), b.to_bits()) {
                return None;
            }
            encoding::<S, W>(min_max_f64(a, b, operation).to_bits(), false)
        }
        Host::Half => select(half::<S, W>(left)?, half::<S, W>(right)?),
        Host::BFloat => select(bfloat::<S, W>(left), bfloat::<S, W>(right)),
    }
}

/// The largest difference of the exponent fields of a dividend and a divisor
/// that the remainder path takes: ten steps of `FPREM1`, each of which
/// reduces the difference by up to 63. Each step takes about 12 nanoseconds,
/// and the engine grows more slowly. At a difference of 700, binary64 took 175
/// nanoseconds on the path and 181 in the engine, and at 1,000, 252 and 182.
const REMAINDER_REACH: u64 = 630;

/// Returns `true` when the remainder path takes a dividend and a divisor
/// with the exponent fields `dividend` and `divisor`, in a format of
/// `precision` bits. A subnormal operand or result makes the x87 unit take a
/// microcode assist that is slower than the engine, so a dividend with
/// exponent field 0, a zero or a subnormal value, goes to the engine. When
/// the quotient is not zero, the dividend lies at most one binade below the
/// divisor, so the remainder is a multiple of half the unit in the last place
/// of the divisor. When the quotient is zero, the remainder is the dividend.
/// So the remainder can be subnormal only when the exponent field of the
/// divisor is at most the precision.
#[inline]
fn remainder_fits(dividend: u64, divisor: u64, precision: u32) -> bool {
    dividend != 0
        && divisor > u64::from(precision)
        && dividend.saturating_sub(divisor) <= REMAINDER_REACH
}

/// Returns the IEEE remainder of two values from the x87 unit, or `None` when
/// the path does not apply, the exponents of the operands lie farther apart
/// than the path reaches, or the result is a NaN. The x87 unit loads binary32
/// and binary64 values exactly, and the remainder is exact, so it stores
/// exactly in the format of the operands.
#[inline]
pub fn remainder<S: Standard<W>, const W: usize>(
    dividend: S::Bits,
    divisor: S::Bits,
    env: &Env,
) -> Option<S::Bits> {
    if !ready_for(Host::Extended, env, S::PRECISION) {
        return None;
    }
    let (x, y) = (dividend.to_limbs().limb(0), divisor.to_limbs().limb(0));
    match S::HOST {
        Host::None | Host::Half | Host::BFloat => None,
        Host::Single => {
            if !remainder_fits((x >> 23) & 0xFF, (y >> 23) & 0xFF, S::PRECISION) {
                return None;
            }
            let bits = |value: u64| u32::try_from(value).expect("a binary32 encoding has 32 bits");
            let result = environment::x87_remainder_single(bits(x), bits(y))?;
            encoding::<S, W>(u64::from(result), false)
        }
        Host::Double => {
            if !remainder_fits((x >> 52) & 0x7FF, (y >> 52) & 0x7FF, S::PRECISION) {
                return None;
            }
            encoding::<S, W>(environment::x87_remainder_double(x, y)?, false)
        }
        Host::Extended => {
            let (x, y) = (extended::<S, W>(dividend), extended::<S, W>(divisor));
            if !remainder_fits(x[1] & 0x7FFF, y[1] & 0x7FFF, S::PRECISION) {
                return None;
            }
            environment::x87_remainder(&x, &y).map(extended_encoding::<S, W>)
        }
    }
}

/// Returns the precision of a host kind.
pub(super) const fn precision_of(host: Host) -> u32 {
    match host {
        Host::None => 0,
        Host::BFloat => 8,
        Host::Half => 11,
        Host::Single => 24,
        Host::Double => 53,
        Host::Extended => 64,
    }
}

/// Returns an encoding of the host kind `from` converted to the host kind
/// `to` by the host unit, or `None` when the path does not apply or the
/// value is a NaN. A widening is exact, and a narrowing rounds once. The
/// limbs hold the encoding from the low bits up.
#[inline]
pub fn convert(from: Host, to: Host, bits: [u64; 2], env: &Env) -> Option<[u64; 2]> {
    // The x87 unit converts to and from x87 extended precision.
    let unit = if matches!(from, Host::Extended) {
        from
    } else {
        to
    };
    if !ready_for(unit, env, precision_of(to)) {
        return None;
    }
    let low = bits[0];
    let single = |bits: u64| f32::from_bits(u32::try_from(bits).expect("a binary32 encoding"));
    let half =
        |bits: u64| environment::widen_half(u16::try_from(bits).expect("a binary16 encoding"));
    let result = match (from, to) {
        (Host::Single, Host::Double) => {
            let value = single(low);
            (!nan_32(value.to_bits())).then(|| environment::widen_single(value).to_bits())
        }
        (Host::Half, Host::Single) => {
            let bits = half(low)?.to_bits();
            (!nan_32(bits)).then_some(u64::from(bits))
        }
        (Host::Half, Host::Double) => {
            let value = half(low)?;
            (!nan_32(value.to_bits())).then(|| environment::widen_single(value).to_bits())
        }
        (Host::Double, Host::Single) => {
            let value = f64::from_bits(low);
            (!nan_64(low)).then(|| u64::from(environment::narrow_double(value).to_bits()))
        }
        (Host::Single, Host::Half) => {
            let value = single(low);
            if nan_32(value.to_bits()) {
                return None;
            }
            environment::narrow_half(value).map(u64::from)
        }
        (Host::Double, Host::Half) => {
            if nan_64(low) {
                return None;
            }
            environment::narrow_double_to_half(f64::from_bits(low)).map(u64::from)
        }
        (Host::BFloat, Host::Single) => {
            let bits = u32::try_from(low << 16).expect("a bfloat16 encoding has 16 bits");
            (!nan_32(bits)).then_some(u64::from(bits))
        }
        (Host::BFloat, Host::Double) => {
            let bits = u32::try_from(low << 16).expect("a bfloat16 encoding has 16 bits");
            let value = f32::from_bits(bits);
            (!nan_32(bits)).then(|| environment::widen_single(value).to_bits())
        }
        (Host::Single, Host::BFloat) => {
            let value = single(low);
            if nan_32(value.to_bits()) {
                return None;
            }
            environment::narrow_bfloat(value).map(u64::from)
        }
        (Host::Single, Host::Extended) => {
            let encoding = u32::try_from(low).expect("a binary32 encoding");
            return environment::x87_from_single(encoding);
        }
        (Host::Double, Host::Extended) => return environment::x87_from_double(low),
        (Host::Extended, Host::Single) => environment::x87_to_single(&bits).map(u64::from),
        (Host::Extended, Host::Double) => environment::x87_to_double(&bits),
        _ => None,
    };
    result.map(|low| [low, 0])
}

/// Proof that the mode and the floating-point environment of the host allow
/// the host paths of binary64. [`ready`] checks both once for the steps of one
/// double-double operation, and no other code makes a `Ready`.
#[derive(Clone, Copy, Debug)]
pub struct Ready(());

/// Returns the proof that the host paths of binary64 apply under `env`, or
/// `None`.
#[inline]
pub fn ready(env: &Env) -> Option<Ready> {
    ready_for(Host::Double, env, 53).then_some(Ready(()))
}

// `self` is the proof that the host paths apply. The methods need the proof,
// not its value, so `self` is unused by design.
#[allow(clippy::unused_self)]
impl Ready {
    /// Returns `operation` of two binary64 encodings, or `None` for a NaN.
    #[inline]
    pub fn binary(self, left: u64, right: u64, operation: Operation) -> Option<u64> {
        let result =
            environment::binary_f64(f64::from_bits(left), f64::from_bits(right), operation);
        let bits = result.to_bits();
        (!nan_64(bits)).then_some(bits)
    }

    /// Returns the square root of a binary64 encoding, or `None` for a NaN.
    #[inline]
    pub fn sqrt(self, value: u64) -> Option<u64> {
        let bits = environment::sqrt_f64(f64::from_bits(value)).to_bits();
        (!nan_64(bits)).then_some(bits)
    }

    /// Returns `left * right + addend` of binary64 encodings, rounded once,
    /// or `None` for a NaN or in a build without a fused multiply-add path.
    #[inline]
    pub fn mul_add(self, left: u64, right: u64, addend: u64) -> Option<u64> {
        let [a, b, c] = [left, right, addend].map(f64::from_bits);
        let bits = environment::mul_add_f64(a, b, c)?.to_bits();
        (!nan_64(bits)).then_some(bits)
    }
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    use super::compatible;
    use crate::env::{Env, Rounding};

    #[test]
    fn only_a_behavior_that_the_host_matches_is_compatible() {
        assert!(compatible(&Env::IEEE, 53));
        assert!(compatible(&Env::X86_SSE, 24));
        assert!(compatible(&Env::X87, 53));
        assert!(!compatible(&Env::X87, 113));
        assert!(!compatible(
            &Env::IEEE.with_rounding(Rounding::TowardZero),
            24
        ));
        assert!(!compatible(&Env::IEEE.with_flush_to_zero(true), 24));
        assert!(!compatible(&Env::IEEE.with_denormals_are_zero(true), 24));
        assert!(!compatible(
            &Env::IEEE.with_precision(NonZeroU32::new(52)),
            53
        ));
    }
}
