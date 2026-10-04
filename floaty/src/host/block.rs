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
//! NaN by the rule of the mode. No step lets the payload of a NaN decide a
//! result that is not a NaN: a NaN operand of the arithmetic, `minimum`, or
//! `maximum` gives a NaN, and a `minimum_number` or `maximum_number` takes
//! the other operand of a NaN, whatever its payload. A step that the
//! instruction set cannot compute exactly, `mul_add` without FMA, taints
//! its lane, and each later step carries the taint to the result.

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

use super::bits::nan_32;
use super::environment::{block_mul_add, block_sqrt};
use super::packed::{Task, run_task};
use super::paths::ready_for;
use super::{Host, Isa};
use crate::block::{Chain, Steps};
use crate::env::Mode;
use crate::float::Float;
use crate::format::Binary;
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

    /// Returns `order` of the lanes, or a NaN for a NaN operand, as
    /// `minimum` and `maximum` give it.
    #[inline]
    fn propagating(self, other: Self, minimum: bool) -> Self {
        if self.value.is_nan() || other.value.is_nan() {
            return Self::new(f32::NAN, self.taint | other.taint);
        }
        self.order(other, minimum)
    }

    /// Returns `order` of the lanes, or the other operand of a NaN, as
    /// `minimum_number` and `maximum_number` give it.
    #[inline]
    fn number(self, other: Self, minimum: bool) -> Self {
        let taint = self.taint | other.taint;
        if self.value.is_nan() {
            return Self::new(other.value, taint);
        }
        if other.value.is_nan() {
            return Self::new(self.value, taint);
        }
        self.order(other, minimum)
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

impl<I: Isa> Steps for Lane<I> {
    #[inline]
    fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        let taint = self.taint | multiplier.taint | addend.taint;
        if !I::FUSED {
            // No exact instruction: the lane runs again in the engine.
            return Self::new(self.value, true);
        }
        // SAFETY: `I` has FMA, and the step runs in `I::run`, from a check
        // of the processor that found the features of `I`.
        let value = unsafe { block_mul_add(self.value, multiplier.value, addend.value) };
        Self::new(value, taint)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::new(block_sqrt(self.value), self.taint)
    }

    #[inline]
    fn abs(self) -> Self {
        Self::new(
            f32::from_bits(self.value.to_bits() & 0x7FFF_FFFF),
            self.taint,
        )
    }

    #[inline]
    fn minimum(self, other: Self) -> Self {
        self.propagating(other, true)
    }

    #[inline]
    fn maximum(self, other: Self) -> Self {
        self.propagating(other, false)
    }

    #[inline]
    fn minimum_number(self, other: Self) -> Self {
        self.number(other, true)
    }

    #[inline]
    fn maximum_number(self, other: Self) -> Self {
        self.number(other, false)
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
