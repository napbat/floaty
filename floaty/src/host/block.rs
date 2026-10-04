//! The host path of blocks: the steps of a chain as Rust operations on
//! `f32`, after one check of the environment, in the instruction set that
//! the processor selects.
//!
//! LLVM assumes the default floating-point environment. Once the check finds
//! that environment, each Rust operation on `f32` gives the bits of the
//! engine for every result but a NaN, and LLVM sees the whole chain, so it
//! computes many lanes in vector instructions. LLVM can also compute a float
//! operation before the check, where an unmasked exception would trap: a
//! test of the environment followed by `Some(a * b + c)` multiplied and added
//! before the test. So every value enters the block through `opaque`, an
//! empty assembly block with side effects after the check. LLVM keeps the
//! block after the check, and each step after the block.
//!
//! A lane whose result is a NaN goes back to the engine, which selects the
//! NaN by the rule of the mode. No step lets the bits of a NaN decide a
//! result that is not a NaN. A NaN operand of the arithmetic gives a NaN,
//! and so does a NaN operand of `minimum`, `maximum`, and their magnitude
//! steps. A `number` step takes the other operand of a NaN, whatever its
//! payload.
//!
//! A step that the instruction set cannot compute exactly taints its lane,
//! and each later step carries the taint to the result: `mul_add` without
//! FMA, `round_to_integral_by` to odd, `scale_b` with a scale outside -126
//! to 127, the remainders, and the steps that have no instruction: `exp`,
//! `log`, `compound`, `hypot`, `pown`, `rootn`, and `reciprocal_sqrt`. A
//! step that reads a bit of a NaN that the lane can hold otherwise than the
//! engine also taints its lane: `copy_sign` from a NaN `sign`, and `min_num`
//! and `max_num` of a signaling NaN.

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::bits::nan_32;
use super::environment::{block_mul_add, block_sqrt};
use super::packed::{Task, run_task};
use super::paths::ready_for;
use super::{Host, Isa};
use crate::block::{Chain, Steps};
use crate::env::{Mode, Rounding};
use crate::float::Float;
use crate::format::Binary;
use crate::format::internal::MinMax;
use crate::sealed::Sealed;

/// The binary32 type of the mode `M`.
type Single<M> = Float<Binary<8>, 32, M>;

/// The result of a tainted lane: a NaN, so that the replay of the NaN lanes
/// runs it in the engine.
const TAINTED: u32 = 0x7FC0_0000;

/// One binary32 lane of a block in the instruction set `I`: its value, and
/// `taint`, `true` once a step had no exact instruction.
#[derive(Debug)]
pub struct Lane<I> {
    value: f32,
    taint: bool,
    isa: PhantomData<fn() -> I>,
}

impl<I> Clone for Lane<I> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I> Copy for Lane<I> {}

impl<I> Sealed for Lane<I> {}

impl<I> Lane<I> {
    /// Returns a lane of `value` with `taint`.
    #[inline]
    fn new(value: f32, taint: bool) -> Self {
        Self {
            value,
            taint,
            isa: PhantomData,
        }
    }

    /// Returns a lane of the binary32 value `value`, which keeps its bits.
    #[inline]
    fn of<M: Mode>(value: Single<M>) -> Self {
        Self::new(f32::from_bits(value.to_bits()), false)
    }

    /// Returns the smaller of the values of two lanes when `minimum` holds,
    /// and the larger otherwise, for two values that are not NaNs. Of two
    /// zeros, -0 is the smaller.
    ///
    /// The minimum and maximum steps follow the rules of `settle` of the
    /// packed paths in float compares and selects, which only a block can
    /// run. `settle` works on integer masks after the packed instruction:
    /// through it, `maximum_number` then `minimum_number` of 1,280 values
    /// took 397 ns in x86-64-v3 on a Ryzen AI Max+ 395, against 170 ns.
    #[inline]
    fn order(self, other: Self, minimum: bool) -> Self {
        let (left, right) = (self.value, other.value);
        let (less, greater) = (left < right, left > right);
        let bits = if !less && !greater {
            // Equal values have equal bits, except -0 and +0.
            if minimum {
                left.to_bits() | right.to_bits()
            } else {
                left.to_bits() & right.to_bits()
            }
        } else if less == minimum {
            left.to_bits()
        } else {
            right.to_bits()
        };
        Self::new(f32::from_bits(bits), self.taint | other.taint)
    }

    /// Returns the lane with its taint set: the lane runs again in the
    /// engine.
    #[inline]
    fn tainted(self) -> Self {
        Self::new(self.value, true)
    }

    /// Returns a lane of the encoding `bits`, with the taint of `self`.
    #[inline]
    fn with_bits(self, bits: u32) -> Self {
        Self::new(f32::from_bits(bits), self.taint)
    }

    /// Returns the minimum or the maximum `operation` of the lanes, as the
    /// engine gives it for every result but a NaN.
    ///
    /// Where the engine holds a signaling NaN, the lane holds the same
    /// NaN: every operation gives a quiet NaN, and only `abs`, `-`, and
    /// `copy_sign` copy a signaling NaN. So `min_num` and `max_num` take the
    /// other operand of a quiet NaN of the lane, as the engine does, and
    /// taint the lane of a signaling NaN, which the engine can hold quiet.
    #[inline]
    fn min_max(self, other: Self, operation: MinMax) -> Self {
        let signaling = |value: f32| value.is_nan() && value.to_bits() & 0x0040_0000 == 0;
        match operation {
            MinMax::Minimum
            | MinMax::Maximum
            | MinMax::MinimumMagnitude
            | MinMax::MaximumMagnitude => self.propagating(other, operation),
            MinMax::MinNum | MinMax::MaxNum if signaling(self.value) || signaling(other.value) => {
                self.tainted()
            }
            _ => self.number(other, operation),
        }
    }

    /// Returns `select` of the lanes, or a NaN for a NaN operand.
    #[inline]
    fn propagating(self, other: Self, operation: MinMax) -> Self {
        if self.value.is_nan() || other.value.is_nan() {
            return Self::new(f32::NAN, self.taint | other.taint);
        }
        self.select(other, operation)
    }

    /// Returns `select` of the lanes, or the other operand of a NaN.
    ///
    /// The two tests of a NaN keep the shape that LLVM vectorizes best: with
    /// one test of either NaN first, `maximum_number` then `minimum_number`
    /// of 1,280 values took 188 ns in x86-64-v3 on a Ryzen AI Max+ 395,
    /// against 170 ns.
    #[inline]
    fn number(self, other: Self, operation: MinMax) -> Self {
        let taint = self.taint | other.taint;
        if self.value.is_nan() {
            return Self::new(other.value, taint);
        }
        if other.value.is_nan() {
            return Self::new(self.value, taint);
        }
        self.select(other, operation)
    }

    /// Returns the lane that `operation` selects of two lanes whose values
    /// are not NaNs: of the smaller or the larger magnitude first for a
    /// magnitude operation, and then `order` of the values.
    #[inline]
    fn select(self, other: Self, operation: MinMax) -> Self {
        let minimum = operation.is_minimum();
        if operation.is_magnitude() {
            // The magnitudes of numbers order as their encodings do.
            let magnitude = |lane: Self| lane.value.to_bits() & 0x7FFF_FFFF;
            let (left, right) = (magnitude(self), magnitude(other));
            if left != right {
                let left_wins = (left < right) == minimum;
                let value = if left_wins { self.value } else { other.value };
                return Self::new(value, self.taint | other.taint);
            }
        }
        self.order(other, minimum)
    }

    /// Returns the value rounded to an integral value in the direction
    /// `rounding`, as the packed path `round_to_integral` rounds, or a
    /// tainted lane for `ToOdd`, which that path sends to the engine.
    ///
    /// The magnitude rounds to nearest even by the sum and difference with
    /// 2^23, in the default environment that the check found. A step of one
    /// then moves it in the direction. The result takes the sign of the
    /// value, as IEEE 754 requires of every direction. LLVM vectorizes these
    /// steps. On x86 the intrinsic `_mm_round_ss` stays one scalar `ROUNDSS`
    /// for each lane, which LLVM does not vectorize.
    #[inline]
    fn integral(self, rounding: Rounding) -> Self {
        // From 2^23 up, every binary32 value is integral.
        const LIMIT: f32 = 8_388_608.0;
        let bits = self.value.to_bits();
        let negative = bits >> 31 != 0;
        let magnitude = f32::from_bits(bits & 0x7FFF_FFFF);
        let even = if magnitude < LIMIT {
            (magnitude + LIMIT) - LIMIT
        } else {
            magnitude
        };
        // Each integer below 2^23, and each step of one from it, is exact,
        // and so is the fraction. An infinity gives a NaN fraction, and a
        // NaN gives NaNs, so neither moves.
        let below = if even > magnitude { even - 1.0 } else { even };
        let above = if even < magnitude { even + 1.0 } else { even };
        let fraction = magnitude - below;
        let rounded = match rounding {
            Rounding::TiesToEven => even,
            Rounding::TiesToAway if fraction >= 0.5 => below + 1.0,
            Rounding::TiesTowardZero if fraction > 0.5 => below + 1.0,
            Rounding::TiesToAway | Rounding::TiesTowardZero | Rounding::TowardZero => below,
            Rounding::TowardPositive if negative => below,
            Rounding::TowardNegative if !negative => below,
            Rounding::AwayFromZero | Rounding::TowardPositive | Rounding::TowardNegative => above,
            Rounding::ToOdd => return self.tainted(),
        };
        self.with_bits(rounded.to_bits() | (bits & 0x8000_0000))
    }
}

/// Implements an operator of `Lane` by the Rust operation on `f32`.
macro_rules! lane_operator {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl<I> $trait for Lane<I> {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                Self::new(self.value $operator right.value, self.taint | right.taint)
            }
        }
    };
}

lane_operator!(Add, add, +);
lane_operator!(Sub, sub, -);
lane_operator!(Mul, mul, *);
lane_operator!(Div, div, /);

impl<I> Neg for Lane<I> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(-self.value, self.taint)
    }
}

/// Implements each step of [`Steps`] that has no exact instruction by a
/// tainted lane.
macro_rules! engine_steps {
    ($($name:ident($($argument:ident: $type:ty),*);)*) => {
        $(
            #[inline]
            fn $name(self, $($argument: $type),*) -> Self {
                self.tainted()
            }
        )*
    };
}

/// Implements each minimum or maximum step of [`Steps`] by `min_max`.
macro_rules! min_max_steps {
    ($($name:ident => $operation:ident;)*) => {
        $(
            #[inline]
            fn $name(self, other: Self) -> Self {
                self.min_max(other, MinMax::$operation)
            }
        )*
    };
}

impl<I: Isa> Steps for Lane<I> {
    #[inline]
    fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        if !I::FUSED {
            return self.tainted();
        }
        // SAFETY: `I` has FMA, and the step runs in `I::run`, from a check
        // of the processor that found the features of `I`.
        let value = unsafe { block_mul_add(self.value, multiplier.value, addend.value) };
        Self::new(value, self.taint | multiplier.taint | addend.taint)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::new(block_sqrt(self.value), self.taint)
    }

    #[inline]
    fn abs(self) -> Self {
        self.with_bits(self.value.to_bits() & 0x7FFF_FFFF)
    }

    #[inline]
    fn copy_sign(self, sign: Self) -> Self {
        // A NaN that the host unit creates takes the sign of the unit, which
        // can differ from the sign of the NaN of the engine.
        if sign.value.is_nan() {
            return self.tainted();
        }
        let bits = (self.value.to_bits() & 0x7FFF_FFFF) | (sign.value.to_bits() & 0x8000_0000);
        Self::new(f32::from_bits(bits), self.taint | sign.taint)
    }

    #[inline]
    fn next_up(self) -> Self {
        let bits = self.value.to_bits();
        let next = if nan_32(bits) {
            f32::NAN.to_bits()
        } else if bits << 1 == 0 {
            // Both zeros step to the least positive subnormal.
            1
        } else if bits == f32::INFINITY.to_bits() {
            bits
        } else if bits >> 31 == 0 {
            bits + 1
        } else {
            // A negative value steps toward zero, -∞ to the most negative
            // finite value, and the least negative subnormal to -0.
            bits - 1
        };
        self.with_bits(next)
    }

    #[inline]
    fn next_down(self) -> Self {
        let up = (-self).next_up();
        -up
    }

    #[inline]
    fn round_to_integral(self) -> Self {
        // The check found the mode rounded to nearest even.
        self.integral(Rounding::TiesToEven)
    }

    #[inline]
    fn round_to_integral_by(self, rounding: Rounding) -> Self {
        self.integral(rounding)
    }

    #[inline]
    fn scale_b(self, scale: i32) -> Self {
        // From -126 to 127, 2^scale is a normal value, so one product gives
        // `value * 2^scale` rounded once, as `scaleB` does.
        match scale.checked_add(127).map(u32::try_from) {
            Some(Ok(biased @ 1..=254)) => {
                Self::new(self.value * f32::from_bits(biased << 23), self.taint)
            }
            _ => self.tainted(),
        }
    }

    #[inline]
    fn log_b(self) -> Self {
        // A subnormal times 2^23 is normal, and exact.
        const NORMALIZE: f32 = 8_388_608.0;
        let magnitude = self.value.to_bits() & 0x7FFF_FFFF;
        let bits = if magnitude >= f32::INFINITY.to_bits() {
            // An infinity gives +∞, and a NaN gives a NaN.
            magnitude
        } else if magnitude == 0 {
            // A zero gives what -1 / 0 gives.
            f32::NEG_INFINITY.to_bits()
        } else {
            let subnormal = magnitude < 0x0080_0000;
            let normal = if subnormal {
                (f32::from_bits(magnitude) * NORMALIZE).to_bits()
            } else {
                magnitude
            };
            let field = i16::try_from(normal >> 23).expect("a binary32 exponent field has 8 bits");
            let bias = if subnormal { 150 } else { 127 };
            f32::from(field - bias).to_bits()
        };
        self.with_bits(bits)
    }

    engine_steps! {
        remainder(_divisor: Self);
        truncated_remainder(_divisor: Self);
        exp();
        log();
        compound(_n: i64);
        hypot(_other: Self);
        pown(_n: i64);
        rootn(_n: i64);
        reciprocal_sqrt();
    }

    min_max_steps! {
        minimum => Minimum;
        maximum => Maximum;
        minimum_number => MinimumNumber;
        maximum_number => MaximumNumber;
        min_num => MinNum;
        max_num => MaxNum;
        minimum_magnitude => MinimumMagnitude;
        maximum_magnitude => MaximumMagnitude;
        minimum_magnitude_number => MinimumMagnitudeNumber;
        maximum_magnitude_number => MaximumMagnitudeNumber;
    }
}

/// Returns `values` through an empty assembly block with side effects, as
/// the module states. The block takes the address of the values, so every
/// load from the result follows the block.
#[inline]
fn opaque<T>(values: &[T]) -> &[T] {
    let mut address = values.as_ptr();
    // SAFETY: the template is a comment, so the block runs no instruction,
    // reads and writes no memory, and keeps the address.
    unsafe {
        core::arch::asm!(
            "/* {address} */",
            address = inout(reg) address,
            options(nostack, preserves_flags),
        );
    }
    // SAFETY: the address is the address of `values`, with its length and
    // lifetime.
    unsafe { core::slice::from_raw_parts(address, values.len()) }
}

/// Writes the result of `chain` for each lane of `x` and of `p` into `out`
/// on the host unit, after one check of the environment, and returns `true`
/// when a lane must run again in the engine: each such lane holds a NaN.
/// Returns `None`, and writes nothing, when the mode or the environment
/// does not allow the path.
///
/// # Panics
///
/// Panics when a slice of `x` and `out` differ in length.
pub fn map<C: Chain<IN, P>, M: Mode, const IN: usize, const P: usize>(
    chain: &C,
    x: [&[Single<M>]; IN],
    p: [Single<M>; P],
    out: &mut [Single<M>],
) -> Option<bool> {
    assert!(
        x.iter().all(|values| values.len() == out.len()),
        "each input holds one value for each output"
    );
    if !ready_for(Host::Single, &M::ENV, Host::Single.precision()) {
        return None;
    }
    Some(run_task(Map { chain, x, p, out }))
}

/// Returns the result of `chain` for the values `x` and `p` on the host
/// unit, after one check of the environment, or `None` when the mode or the
/// environment does not allow the path, or the lane must run in the engine.
pub fn evaluate<C: Chain<IN, P>, M: Mode, const IN: usize, const P: usize>(
    chain: &C,
    x: [Single<M>; IN],
    p: [Single<M>; P],
) -> Option<Single<M>> {
    if !ready_for(Host::Single, &M::ENV, Host::Single.precision()) {
        return None;
    }
    run_task(Evaluate { chain, x, p })
}

/// The arguments of `map` for its instruction set.
struct Map<'a, C, M: Mode, const IN: usize, const P: usize> {
    chain: &'a C,
    x: [&'a [Single<M>]; IN],
    p: [Single<M>; P],
    out: &'a mut [Single<M>],
}

impl<C: Chain<IN, P>, M: Mode, const IN: usize, const P: usize> Task for Map<'_, C, M, IN, P> {
    type Output = bool;

    #[inline]
    fn run<I: Isa>(self) -> bool {
        let Self { chain, x, p, out } = self;
        I::run(move || map_in::<I, C, M, IN, P>(chain, x, p, out))
    }
}

/// Runs `map` in the instruction set `I`, in one counted loop over the
/// lanes. LLVM vectorizes the loop in each instruction set. A loop of chunks
/// of 32 lanes unrolled whole, and in SSE2 LLVM then gathered each lane
/// with scalar loads: the square of the difference of two slices of 1,280
/// values took 0.30 ns per value in the x86-64 build on a Ryzen AI Max+
/// 395, against 0.11 ns with the counted loop.
#[inline]
fn map_in<I: Isa, C: Chain<IN, P>, M: Mode, const IN: usize, const P: usize>(
    chain: &C,
    x: [&[Single<M>]; IN],
    p: [Single<M>; P],
    out: &mut [Single<M>],
) -> bool {
    let x = x.map(opaque);
    let p = opaque(&p);
    let p: [Lane<I>; P] = core::array::from_fn(|index| Lane::of(p[index]));
    // The top bit of `magnitude + 0x007F_FFFF` is set exactly for a NaN, a
    // magnitude above 0x7F80_0000, so the loop ORs the sums and tests the
    // top bit once. A `bool` of each lane made LLVM narrow the mask of each
    // vector to bytes: the update `x * keep + y * eta` of 1,280 values took
    // 87 ns in x86-64-v3 on a Ryzen AI Max+ 395, against 69 ns.
    let mut sums = 0_u32;
    for (index, result) in out.iter_mut().enumerate() {
        // SAFETY: `map` checked that each slice holds a value for each lane.
        let values = x.map(|values| Lane::of(*unsafe { values.get_unchecked(index) }));
        let lane = chain.apply::<Lane<I>>(values, p);
        let bits = if lane.taint {
            TAINTED
        } else {
            lane.value.to_bits()
        };
        sums |= (bits & 0x7FFF_FFFF) + 0x007F_FFFF;
        *result = Single::<M>::from_bits(bits);
    }
    sums >> 31 != 0
}

/// The arguments of `evaluate` for its instruction set.
struct Evaluate<'a, C, M: Mode, const IN: usize, const P: usize> {
    chain: &'a C,
    x: [Single<M>; IN],
    p: [Single<M>; P],
}

impl<C: Chain<IN, P>, M: Mode, const IN: usize, const P: usize> Task for Evaluate<'_, C, M, IN, P> {
    type Output = Option<Single<M>>;

    #[inline]
    fn run<I: Isa>(self) -> Option<Single<M>> {
        let Self { chain, x, p } = self;
        I::run(move || {
            let (x, p) = (opaque(&x), opaque(&p));
            let x: [Lane<I>; IN] = core::array::from_fn(|index| Lane::of(x[index]));
            let p: [Lane<I>; P] = core::array::from_fn(|index| Lane::of(p[index]));
            let lane = chain.apply::<Lane<I>>(x, p);
            let bits = lane.value.to_bits();
            (!lane.taint && !nan_32(bits)).then(|| Single::<M>::from_bits(bits))
        })
    }
}
