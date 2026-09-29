//! The entry points of the host paths, the same on every architecture that
//! has them. The module of the architecture reads its floating-point
//! environment.

use super::Operation;
use super::environment::{self, default_environment};
use crate::env::{Env, Rounding};
use crate::format::Standard;
use crate::format::internal::{Host, LimbConversion};
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
    matches!(rounding, Rounding::NearestEven)
        && !flush_to_zero
        && !denormals_are_zero
        && full_precision
}

/// Applies an operation to two host values.
macro_rules! apply {
    ($operation:expr, $left:expr, $right:expr) => {
        match $operation {
            Operation::Add => $left + $right,
            Operation::Sub => $left - $right,
            Operation::Mul => $left * $right,
            Operation::Div => $left / $right,
        }
    };
}

/// Returns the host value of a binary32 encoding.
#[inline]
fn single<S: Standard<W>, const W: usize>(bits: S::Bits) -> f32 {
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
    if result.is_nan() {
        return None;
    }
    encoding::<S, W>(u64::from(environment::narrow_half(result)?), false)
}

/// Returns the host value of a binary64 encoding.
#[inline]
fn double<S: Standard<W>, const W: usize>(bits: S::Bits) -> f64 {
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

/// Returns `true` when the mode and the environment of the host allow a host
/// path for a format of `precision` bits.
#[inline]
fn ready_for(env: &Env, precision: u32) -> bool {
    compatible(env, precision) && default_environment()
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
    if !ready_for(env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::None => None,
        Host::Single => {
            let result = apply!(operation, single::<S, W>(left), single::<S, W>(right));
            encoding::<S, W>(u64::from(result.to_bits()), result.is_nan())
        }
        Host::Double => {
            let result = apply!(operation, double::<S, W>(left), double::<S, W>(right));
            encoding::<S, W>(result.to_bits(), result.is_nan())
        }
        Host::Half => {
            let result = apply!(operation, half::<S, W>(left)?, half::<S, W>(right)?);
            half_encoding::<S, W>(result)
        }
    }
}

/// Returns the square root from the host unit, or `None` when the path does
/// not apply or the result is a NaN.
#[inline]
pub fn sqrt<S: Standard<W>, const W: usize>(value: S::Bits, env: &Env) -> Option<S::Bits> {
    if !ready_for(env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        Host::None => None,
        Host::Single => {
            let result = environment::sqrt_f32(single::<S, W>(value));
            encoding::<S, W>(u64::from(result.to_bits()), result.is_nan())
        }
        Host::Double => {
            let result = environment::sqrt_f64(double::<S, W>(value));
            encoding::<S, W>(result.to_bits(), result.is_nan())
        }
        Host::Half => half_encoding::<S, W>(environment::sqrt_f32(half::<S, W>(value)?)),
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
    if !ready_for(env, S::PRECISION) {
        return None;
    }
    match S::HOST {
        // Two roundings of a fused multiply-add can differ from one.
        Host::None | Host::Half => None,
        Host::Single => {
            let (a, b) = (single::<S, W>(left), single::<S, W>(right));
            let result = environment::mul_add_f32(a, b, single::<S, W>(addend))?;
            encoding::<S, W>(u64::from(result.to_bits()), result.is_nan())
        }
        Host::Double => {
            let (a, b) = (double::<S, W>(left), double::<S, W>(right));
            let result = environment::mul_add_f64(a, b, double::<S, W>(addend))?;
            encoding::<S, W>(result.to_bits(), result.is_nan())
        }
    }
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
    ready_for(env, 53).then_some(Ready(()))
}

// `self` is the proof that the host paths apply. The methods need the proof,
// not its value, so `self` is unused by design.
#[allow(clippy::unused_self)]
impl Ready {
    /// Returns `operation` of two binary64 encodings, or `None` for a NaN.
    #[inline]
    pub fn binary(self, left: u64, right: u64, operation: Operation) -> Option<u64> {
        let result = apply!(operation, f64::from_bits(left), f64::from_bits(right));
        (!result.is_nan()).then(|| result.to_bits())
    }

    /// Returns the square root of a binary64 encoding, or `None` for a NaN.
    #[inline]
    pub fn sqrt(self, value: u64) -> Option<u64> {
        let result = environment::sqrt_f64(f64::from_bits(value));
        (!result.is_nan()).then(|| result.to_bits())
    }

    /// Returns `left * right + addend` of binary64 encodings, rounded once,
    /// or `None` for a NaN or in a build without a fused multiply-add path.
    #[inline]
    pub fn mul_add(self, left: u64, right: u64, addend: u64) -> Option<u64> {
        let [a, b, c] = [left, right, addend].map(f64::from_bits);
        let result = environment::mul_add_f64(a, b, c)?;
        (!result.is_nan()).then(|| result.to_bits())
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
