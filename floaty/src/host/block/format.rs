//! The formats of the lanes of a block: binary32 and binary64 in their host
//! types, and binary16 and bfloat16 in `f32`, which round each step to the
//! format as their host paths do; and the binary types of floaty that a
//! block computes in each format.
//!
//! binary32 holds `2p + 2` bits of binary16 and of bfloat16, so a step that
//! computes in `f32` and rounds to the format gives the correctly rounded
//! result of add, subtract, multiply, divide, and square root, as the module
//! `paths` states. The other steps give a value of the format exactly: a
//! sign, a selection, an integral value, and the exponent of `log_b`, which
//! is at most 133 in magnitude. A step that rounds the product of a power of
//! two rounds a binary32 product that is exact, or so large or so small that
//! both roundings give an infinity or a zero.

use core::ops::{Add, BitAnd, Not, Sub};

use super::native::Native;
use crate::env::{Env, Mode};
use crate::float::Float;
use crate::format::Binary;
use crate::host::Host;
use crate::host::narrow::{
    HALF_QUANTUM, SUBNORMAL_BIAS, double_half_precision_bias, half_precision_bias,
    round_double_to_half_precision, round_to_bfloat, round_to_half_lanes, round_to_half_precision,
    widen_half_lanes,
};

/// The format of the lanes of a block: the host type in which a lane
/// computes, and the steps whose result the format rounds otherwise than
/// the host type.
pub trait Format {
    /// The host type of the lanes.
    type Native: Native;

    /// The host kind of the format, whose unit the block checks.
    const HOST: Host;

    /// Returns a result in the host type rounded to the format, in the host
    /// type: the identity where the host type is the format.
    fn round(value: Self::Native) -> Self::Native;

    /// Returns the least value of the format above `value`, a value of the
    /// format that is not a NaN.
    fn next_up(value: Self::Native) -> Self::Native;

    /// Returns `left * right + addend` of values of the format, rounded
    /// once to the format, or `None` where the lane has no exact
    /// computation. `fused` is `true` when the instruction set has the
    /// fused multiply-add.
    ///
    /// # Safety
    ///
    /// When `fused` holds, the instruction set of the caller must have the
    /// fused multiply-add.
    unsafe fn mul_add(
        left: Self::Native,
        right: Self::Native,
        addend: Self::Native,
        fused: bool,
    ) -> Option<Self::Native>;
}

/// Returns the encoding of the least value above the value of the encoding
/// `bits`, which is not a NaN, in a format with the sign bit `sign` and the
/// infinity `infinity`.
#[inline]
fn step_up<B>(bits: B, sign: B, infinity: B) -> B
where
    B: Copy
        + Eq
        + From<u16>
        + Not<Output = B>
        + BitAnd<Output = B>
        + Add<Output = B>
        + Sub<Output = B>,
{
    let (zero, one) = (B::from(0), B::from(1));
    if bits & !sign == zero {
        // Both zeros step to the least positive subnormal value.
        one
    } else if bits == infinity {
        bits
    } else if bits & sign == zero {
        bits + one
    } else {
        // A negative value steps toward zero, -∞ to the most negative
        // finite value, and the least negative subnormal value to -0.
        bits - one
    }
}

/// Implements `Format` for a host type, which is its own format.
macro_rules! host_format {
    ($type:ty, $host:ident) => {
        impl Format for $type {
            type Native = Self;

            const HOST: Host = Host::$host;

            #[inline]
            fn round(value: Self) -> Self {
                value
            }

            #[inline]
            fn next_up(value: Self) -> Self {
                let bits = step_up(value.to_bits(), Self::SIGN, Self::INFINITY.to_bits());
                Self::from_bits(bits)
            }

            #[inline]
            unsafe fn mul_add(left: Self, right: Self, addend: Self, fused: bool) -> Option<Self> {
                // SAFETY: the caller guarantees the fused multiply-add when
                // `fused` holds.
                fused.then(|| unsafe { left.block_mul_add(right, addend) })
            }
        }
    };
}

host_format!(f32, Single);
host_format!(f64, Double);

/// binary16 in `f32`, which rounds each step in the binary32 sum and
/// difference with a power of two and the few integer instructions of
/// `round_to_half_precision`. A short chain then stays small enough that
/// LLVM inlines it into the loop of a block and vectorizes the loop: the
/// k-means update `x * keep + y * eta` of 1,280 binary16 values took
/// 1,451 ns in x86-64-v3 on a Ryzen AI Max+ 395. Through
/// `round_to_half_lanes` and a widening of each step, it took 40,391 ns, as
/// a call for each lane.
#[derive(Clone, Copy, Debug)]
pub struct Half;

impl Half {
    /// Returns the binary16 encoding of a binary32 value rounded to nearest
    /// even, or of a quiet NaN for a NaN.
    #[inline]
    fn narrow(value: f32) -> u16 {
        let magnitude = f32::from_bits(value.to_bits() & 0x7FFF_FFFF);
        let sum = magnitude + f32::from_bits(SUBNORMAL_BIAS);
        round_to_half_lanes(value.to_bits(), sum.to_bits())
    }

    /// Returns the binary32 value of a binary16 encoding, exactly.
    #[inline]
    fn widen(bits: u16) -> f32 {
        let subnormal = f32::from(bits & 0x03FF) * f32::from_bits(HALF_QUANTUM);
        f32::from_bits(widen_half_lanes(bits, subnormal.to_bits()))
    }
}

impl Format for Half {
    type Native = f32;

    const HOST: Host = Host::Half;

    #[inline]
    fn round(value: f32) -> f32 {
        let magnitude = f32::from_bits(value.to_bits() & 0x7FFF_FFFF);
        let bias = f32::from_bits(half_precision_bias(value.to_bits()));
        let rounded = (magnitude + bias) - bias;
        f32::from_bits(round_to_half_precision(value.to_bits(), rounded.to_bits()))
    }

    #[inline]
    fn next_up(value: f32) -> f32 {
        Self::widen(step_up(Self::narrow(value), 0x8000, 0x7C00))
    }

    #[inline]
    unsafe fn mul_add(left: f32, right: f32, addend: f32, _fused: bool) -> Option<f32> {
        // The product of two binary16 values is exact in binary64, and the
        // binary64 sum then rounds to binary16 as one rounding of the fused
        // result, as `Host::Half` states. The steps need no fused
        // instruction, and the rounding to binary16 precision is the one of
        // `round` in binary64.
        let sum = f64::from(left) * f64::from(right) + f64::from(addend);
        let magnitude = f64::from_bits(sum.to_bits() & 0x7FFF_FFFF_FFFF_FFFF);
        let bias = f64::from_bits(double_half_precision_bias(sum.to_bits()));
        let rounded = (magnitude + bias) - bias;
        Some(f32::from_bits(round_double_to_half_precision(
            sum.to_bits(),
            rounded.to_bits(),
        )))
    }
}

/// bfloat16 in `f32`, the high half of the binary32 encoding, which rounds
/// in the integer instructions of `round_to_bfloat`.
#[derive(Clone, Copy, Debug)]
pub struct BFloat;

impl BFloat {
    /// Returns the bfloat16 encoding of a binary32 value rounded to nearest
    /// even, or of a quiet NaN for a NaN.
    #[inline]
    fn narrow(value: f32) -> u16 {
        round_to_bfloat(value.to_bits())
    }

    /// Returns the binary32 value of a bfloat16 encoding, exactly.
    #[inline]
    fn widen(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }
}

impl Format for BFloat {
    type Native = f32;

    const HOST: Host = Host::BFloat;

    #[inline]
    fn round(value: f32) -> f32 {
        Self::widen(Self::narrow(value))
    }

    #[inline]
    fn next_up(value: f32) -> f32 {
        Self::widen(step_up(Self::narrow(value), 0x8000, 0x7F80))
    }

    #[inline]
    unsafe fn mul_add(_left: f32, _right: f32, _addend: f32, _fused: bool) -> Option<f32> {
        // The rounding of a binary32 or binary64 fused result to bfloat16 is
        // a second rounding, which can differ from one, as the bfloat16 host
        // paths state.
        None
    }
}

/// A binary type of floaty that a block computes on the host unit: the
/// binary16, bfloat16, binary32, and binary64 types, in each mode.
pub trait Value: Copy {
    /// The format of the lanes.
    type Format: Format;

    /// The environment of the mode.
    const ENV: Env;

    /// Returns the value in the host type of the lanes, exactly.
    fn native(self) -> <Self::Format as Format>::Native;

    /// Returns the value of a result of a lane: a value of the format, or a
    /// NaN, which gives a NaN.
    fn of_native(value: <Self::Format as Format>::Native) -> Self;
}

impl<M: Mode> Value for Float<Binary<8>, 32, M> {
    type Format = f32;

    const ENV: Env = M::ENV;

    #[inline]
    fn native(self) -> f32 {
        f32::from_bits(self.to_bits())
    }

    #[inline]
    fn of_native(value: f32) -> Self {
        Self::from_bits(value.to_bits())
    }
}

impl<M: Mode> Value for Float<Binary<11>, 64, M> {
    type Format = f64;

    const ENV: Env = M::ENV;

    #[inline]
    fn native(self) -> f64 {
        f64::from_bits(self.to_bits())
    }

    #[inline]
    fn of_native(value: f64) -> Self {
        Self::from_bits(value.to_bits())
    }
}

impl<M: Mode> Value for Float<Binary<5>, 16, M> {
    type Format = Half;

    const ENV: Env = M::ENV;

    #[inline]
    fn native(self) -> f32 {
        Half::widen(self.to_bits())
    }

    #[inline]
    fn of_native(value: f32) -> Self {
        Self::from_bits(Half::narrow(value))
    }
}

impl<M: Mode> Value for Float<Binary<8>, 16, M> {
    type Format = BFloat;

    const ENV: Env = M::ENV;

    #[inline]
    fn native(self) -> f32 {
        BFloat::widen(self.to_bits())
    }

    #[inline]
    fn of_native(value: f32) -> Self {
        Self::from_bits(BFloat::narrow(value))
    }
}
