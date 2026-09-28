//! The host fast path: binary32 and binary64 arithmetic of the operators on
//! the host floating-point unit.
//!
//! The path runs only where the host gives the bits of the engine. The host
//! must be x86-64 with SSE2: its SSE unit follows IEEE 754 for these
//! operations, and the oracle tests run there. MXCSR must round to nearest
//! even without FTZ or DAZ, which each call reads. The mode must round to
//! nearest even without FTZ, DAZ, or a precision limit below the format
//! precision. The operators return no flags, so the host flags do not
//! matter. A NaN result goes back to the engine, which selects the NaN by
//! the rule of the mode.

use crate::env::{Env, Rounding};
use crate::format::Standard;
use crate::format::internal::{Host, LimbConversion};
use crate::limbs::Limbs;

/// An operation of the host path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
}

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

/// Returns `true` when the SSE unit rounds to nearest even without FTZ or DAZ.
///
/// Rust code runs with that MXCSR value, but an emulator can load another
/// value, and a library built with `-ffast-math` can set FTZ and DAZ when it
/// loads. The path then goes back to the engine.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn default_environment() -> bool {
    // The MXCSR fields that change a result: the rounding control, FTZ, and
    // DAZ. Each is zero in the default floating-point environment.
    const RESULT_FIELDS: u32 = (3 << 13) | (1 << 15) | (1 << 6);
    let mut mxcsr = 0_u32;
    // SAFETY: `STMXCSR` writes the four bytes of `mxcsr` and changes no other
    // state.
    unsafe {
        core::arch::asm!(
            "stmxcsr [{mxcsr}]",
            mxcsr = in(reg) &raw mut mxcsr,
            options(nostack, preserves_flags),
        );
    }
    mxcsr & RESULT_FIELDS == 0
}

/// Returns `false`: the host path runs only on x86-64 with SSE2, where the
/// oracle tests run.
#[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
#[inline]
fn default_environment() -> bool {
    false
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

/// Returns the result of `operation` from the host unit, or `None` when the
/// path does not apply or the result is a NaN.
#[inline]
pub fn binary<S: Standard<W>, const W: usize>(
    left: S::Bits,
    right: S::Bits,
    operation: Operation,
    env: &Env,
) -> Option<S::Bits> {
    if S::HOST == Host::None || !compatible(env, S::PRECISION) || !default_environment() {
        return None;
    }
    let low = |bits: S::Bits| bits.to_limbs().limb(0);
    let result = match S::HOST {
        Host::None => return None,
        Host::Single => {
            let single = |bits| {
                f32::from_bits(u32::try_from(low(bits)).expect("a binary32 encoding has 32 bits"))
            };
            let result = apply!(operation, single(left), single(right));
            if result.is_nan() {
                return None;
            }
            u64::from(result.to_bits())
        }
        Host::Double => {
            let double = |bits| f64::from_bits(low(bits));
            let result = apply!(operation, double(left), double(right));
            if result.is_nan() {
                return None;
            }
            result.to_bits()
        }
    };
    let limbs = <S::Bits as LimbConversion>::Limbs::ZERO.with_limb(0, result);
    Some(S::Bits::from_limbs(limbs))
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
