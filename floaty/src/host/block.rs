//! The host path of blocks: the steps of a chain as Rust operations on `f32`
//! for binary32 and on `f64` for binary64, and on `f32` for binary16 and
//! bfloat16, which round each step to their format, after one check of the
//! environment, in the instruction set that the processor selects.
//!
//! LLVM assumes the default floating-point environment. Once the check finds
//! that environment, each Rust operation on `f32` or `f64` gives the bits of
//! the engine for every result but a NaN, and LLVM sees the whole chain, so
//! it computes many lanes in vector instructions. LLVM can also compute a
//! float operation before the check, where an unmasked exception would trap:
//! a test of the environment followed by `Some(a * b + c)` multiplied and
//! added before the test. So every value enters the block through `opaque`,
//! an empty assembly block with side effects after the check. LLVM keeps the
//! block after the check, and each step after the block. The module `format`
//! states the rounding of each format.
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
//! FMA, `round_to_integral_by` to odd, `scale_b` with a scale that has no
//! normal power of two, the remainders, and the steps that have no
//! instruction: the exponentials, the logarithms, the hyperbolic functions,
//! the trigonometric functions and their forms scaled by pi, the inverse
//! trigonometric functions, `atan2`, `atan2_pi`, `compound`, `hypot`, `pown`,
//! `rootn`, and `reciprocal_sqrt`.
//! A step that reads a bit of a NaN that the lane can hold otherwise than
//! the engine also taints its lane: `copy_sign` from a NaN `sign`, and
//! `min_num` and `max_num` of a signaling NaN.

mod format;
mod native;

use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};

pub use self::format::{Format, Value};
pub use self::native::Native;
use super::Isa;
use super::packed::{Task, run_task};
use super::paths::ready_for;
use crate::block::{Chain, Steps};
use crate::env::Rounding;
use crate::format::internal::MinMax;
use crate::sealed::Sealed;

/// One lane of a block in the instruction set `I`, in the format `F`: its
/// value in the host type of the format, and `taint`, `true` once a step had
/// no exact instruction.
#[derive(Debug)]
pub struct Lane<I, F: Format> {
    value: F::Native,
    taint: bool,
    isa: PhantomData<fn() -> I>,
}

impl<I, F: Format> Clone for Lane<I, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<I, F: Format> Copy for Lane<I, F> {}

impl<I, F: Format> Sealed for Lane<I, F> {}

impl<I, F: Format<Native = N>, N: Native> Lane<I, F> {
    /// Returns a lane of `value` with `taint`.
    #[inline]
    fn new(value: N, taint: bool) -> Self {
        Self {
            value,
            taint,
            isa: PhantomData,
        }
    }

    /// Returns a lane of the value `value`, which keeps its bits.
    #[inline]
    fn of<V: Value<Format = F>>(value: V) -> Self {
        Self::new(value.native(), false)
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
        Self::new(N::from_bits(bits), self.taint | other.taint)
    }

    /// Returns the lane with its taint set: the lane runs again in the
    /// engine.
    #[inline]
    fn tainted(self) -> Self {
        Self::new(self.value, true)
    }

    /// Returns a lane of the encoding `bits`, with the taint of `self`.
    #[inline]
    fn with_bits(self, bits: N::Bits) -> Self {
        Self::new(N::from_bits(bits), self.taint)
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
        let signaling = |value: N| value.is_nan() && value.to_bits() & N::QUIET == N::Bits::from(0);
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
            return Self::new(N::NAN, self.taint | other.taint);
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
            let magnitude = |lane: Self| lane.value.to_bits() & !N::SIGN;
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
    /// `INTEGRAL`, in the default environment that the check found. A step of
    /// one then moves it in the direction. The result takes the sign of the
    /// value, as IEEE 754 requires of every direction. LLVM vectorizes these
    /// steps. On x86 the intrinsic `_mm_round_ss` stays one scalar `ROUNDSS`
    /// for each lane, which LLVM does not vectorize.
    #[inline]
    fn integral(self, rounding: Rounding) -> Self {
        let bits = self.value.to_bits();
        let negative = bits & N::SIGN != N::Bits::from(0);
        let magnitude = N::from_bits(bits & !N::SIGN);
        let even = if magnitude < N::INTEGRAL {
            (magnitude + N::INTEGRAL) - N::INTEGRAL
        } else {
            magnitude
        };
        // Each integer below `INTEGRAL`, and each step of one from it, is
        // exact, and so is the fraction. An infinity gives a NaN fraction,
        // and a NaN gives NaNs, so neither moves.
        let one = N::from(1);
        let below = if even > magnitude { even - one } else { even };
        let above = if even < magnitude { even + one } else { even };
        let fraction = magnitude - below;
        let rounded = match rounding {
            Rounding::TiesToEven => even,
            Rounding::TiesToAway if fraction >= N::HALF => below + one,
            Rounding::TiesTowardZero if fraction > N::HALF => below + one,
            Rounding::TiesToAway | Rounding::TiesTowardZero | Rounding::TowardZero => below,
            Rounding::TowardPositive if negative => below,
            Rounding::TowardNegative if !negative => below,
            Rounding::AwayFromZero | Rounding::TowardPositive | Rounding::TowardNegative => above,
            Rounding::ToOdd => return self.tainted(),
        };
        self.with_bits(rounded.to_bits() | (bits & N::SIGN))
    }
}

/// Implements an operator of `Lane` by the Rust operation on the host type,
/// rounded to the format.
macro_rules! lane_operator {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl<I, F: Format> $trait for Lane<I, F> {
            type Output = Self;

            #[inline]
            fn $method(self, right: Self) -> Self {
                let value = F::round(self.value $operator right.value);
                Self::new(value, self.taint | right.taint)
            }
        }
    };
}

lane_operator!(Add, add, +);
lane_operator!(Sub, sub, -);
lane_operator!(Mul, mul, *);
lane_operator!(Div, div, /);

impl<I, F: Format> Neg for Lane<I, F> {
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

impl<I: Isa, F: Format<Native = N>, N: Native> Steps for Lane<I, F> {
    #[inline]
    fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        // SAFETY: `I::FUSED` holds only for an instruction set with FMA, and
        // the step runs in `I::run`, from a check of the processor that found
        // the features of `I`.
        let value = unsafe { F::mul_add(self.value, multiplier.value, addend.value, I::FUSED) };
        match value {
            Some(value) => Self::new(value, self.taint | multiplier.taint | addend.taint),
            None => self.tainted(),
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self::new(F::round(self.value.block_sqrt()), self.taint)
    }

    #[inline]
    fn abs(self) -> Self {
        self.with_bits(self.value.to_bits() & !N::SIGN)
    }

    #[inline]
    fn copy_sign(self, sign: Self) -> Self {
        // A NaN that the host unit creates takes the sign of the unit, which
        // can differ from the sign of the NaN of the engine.
        if sign.value.is_nan() {
            return self.tainted();
        }
        let bits = (self.value.to_bits() & !N::SIGN) | (sign.value.to_bits() & N::SIGN);
        Self::new(N::from_bits(bits), self.taint | sign.taint)
    }

    #[inline]
    fn next_up(self) -> Self {
        if self.value.is_nan() {
            return Self::new(N::NAN, self.taint);
        }
        Self::new(F::next_up(self.value), self.taint)
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
        // Each scale whose biased exponent field is that of a normal value
        // of the host type has a power of two, so one product gives
        // `value * 2^scale` rounded once, as `scaleB` does, and the format
        // rounds it as its module states.
        let bias = i32::from(N::BIAS);
        match scale.checked_add(bias) {
            Some(biased) if (1..=2 * bias).contains(&biased) => {
                let field =
                    u16::try_from(biased).expect("the field of a normal value fits 16 bits");
                let power = N::from_bits(N::Bits::from(field) << u32::from(N::FRACTION_BITS));
                Self::new(F::round(self.value * power), self.taint)
            }
            _ => self.tainted(),
        }
    }

    #[inline]
    fn log_b(self) -> Self {
        let magnitude = self.value.to_bits() & !N::SIGN;
        let infinity = N::INFINITY.to_bits();
        let bits = if magnitude >= infinity {
            // An infinity gives +∞, and a NaN gives a NaN.
            magnitude
        } else if magnitude == N::Bits::from(0) {
            // A zero gives what -1 / 0 gives.
            (-N::INFINITY).to_bits()
        } else {
            let smallest_normal = N::Bits::from(1) << u32::from(N::FRACTION_BITS);
            let (normal, shift) = if magnitude < smallest_normal {
                // A subnormal value times `INTEGRAL` is normal, and exact.
                let normal = N::from_bits(magnitude) * N::INTEGRAL;
                (normal.to_bits(), i16::from(N::FRACTION_BITS))
            } else {
                (magnitude, 0)
            };
            N::from(N::exponent_field(normal) - N::BIAS - shift).to_bits()
        };
        self.with_bits(bits)
    }

    engine_steps! {
        remainder(_divisor: Self);
        truncated_remainder(_divisor: Self);
        exp();
        exp_m1();
        exp2();
        exp2_m1();
        exp10();
        exp10_m1();
        log();
        log2();
        log10();
        log_p1();
        log2_p1();
        log10_p1();
        sinh();
        cosh();
        tanh();
        asinh();
        acosh();
        atanh();
        sin();
        cos();
        tan();
        sin_pi();
        cos_pi();
        tan_pi();
        asin();
        acos();
        atan();
        asin_pi();
        acos_pi();
        atan_pi();
        atan2(_x: Self);
        atan2_pi(_x: Self);
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
pub fn map<C: Chain<IN, P>, F: Value, const IN: usize, const P: usize>(
    chain: &C,
    x: [&[F]; IN],
    p: [F; P],
    out: &mut [F],
) -> Option<bool> {
    assert!(
        x.iter().all(|values| values.len() == out.len()),
        "each input holds one value for each output"
    );
    let host = F::Format::HOST;
    if !ready_for(host, &F::ENV, host.precision()) {
        return None;
    }
    Some(run_task(Map { chain, x, p, out }))
}

/// Returns the result of `chain` for the values `x` and `p` on the host
/// unit, after one check of the environment, or `None` when the mode or the
/// environment does not allow the path, or the lane must run in the engine.
pub fn evaluate<C: Chain<IN, P>, F: Value, const IN: usize, const P: usize>(
    chain: &C,
    x: [F; IN],
    p: [F; P],
) -> Option<F> {
    let host = F::Format::HOST;
    if !ready_for(host, &F::ENV, host.precision()) {
        return None;
    }
    run_task(Evaluate { chain, x, p })
}

/// The arguments of `map` for its instruction set.
struct Map<'a, C, F, const IN: usize, const P: usize> {
    chain: &'a C,
    x: [&'a [F]; IN],
    p: [F; P],
    out: &'a mut [F],
}

impl<C: Chain<IN, P>, F: Value, const IN: usize, const P: usize> Task for Map<'_, C, F, IN, P> {
    type Output = bool;

    #[inline]
    fn run<I: Isa>(self) -> bool {
        let Self { chain, x, p, out } = self;
        I::run(move || {
            map_in::<I, C, F, F::Format, <F::Format as Format>::Native, IN, P>(chain, x, p, out)
        })
    }
}

/// Runs `map` in the instruction set `I`, in one counted loop over the
/// lanes. LLVM vectorizes the loop in each instruction set. A loop of chunks
/// of 32 lanes unrolled whole, and in SSE2 LLVM then gathered each lane
/// with scalar loads: the square of the difference of two slices of 1,280
/// values took 0.30 ns per value in the x86-64 build on a Ryzen AI Max+
/// 395, against 0.11 ns with the counted loop.
#[inline]
fn map_in<I, C, F, K, N, const IN: usize, const P: usize>(
    chain: &C,
    x: [&[F]; IN],
    p: [F; P],
    out: &mut [F],
) -> bool
where
    I: Isa,
    C: Chain<IN, P>,
    F: Value<Format = K>,
    K: Format<Native = N>,
    N: Native,
{
    let x = x.map(opaque);
    let p = opaque(&p);
    let p: [Lane<I, K>; P] = core::array::from_fn(|index| Lane::of(p[index]));
    // The top bit of `magnitude + (sign - infinity - 1)` is set exactly for
    // a NaN, a magnitude above the encoding of infinity, so the loop ORs the
    // sums and tests the top bit once. A `bool` of each lane made LLVM narrow
    // the mask of each vector to bytes: the update `x * keep + y * eta` of
    // 1,280 binary32 values took 87 ns in x86-64-v3 on a Ryzen AI Max+ 395,
    // against 69 ns.
    let (zero, one) = (N::Bits::from(0), N::Bits::from(1));
    let offset = N::SIGN - N::INFINITY.to_bits() - one;
    let mut sums = zero;
    for (index, result) in out.iter_mut().enumerate() {
        // SAFETY: `map` checked that each slice holds a value for each lane.
        let values = x.map(|values| Lane::of(*unsafe { values.get_unchecked(index) }));
        let lane = chain.apply::<Lane<I, K>>(values, p);
        let value = if lane.taint { N::NAN } else { lane.value };
        sums = sums | ((value.to_bits() & !N::SIGN) + offset);
        *result = F::of_native(value);
    }
    sums & N::SIGN != zero
}

/// The arguments of `evaluate` for its instruction set.
struct Evaluate<'a, C, F, const IN: usize, const P: usize> {
    chain: &'a C,
    x: [F; IN],
    p: [F; P],
}

impl<C: Chain<IN, P>, F: Value, const IN: usize, const P: usize> Task
    for Evaluate<'_, C, F, IN, P>
{
    type Output = Option<F>;

    #[inline]
    fn run<I: Isa>(self) -> Option<F> {
        let Self { chain, x, p } = self;
        I::run(move || {
            evaluate_in::<I, C, F, F::Format, <F::Format as Format>::Native, IN, P>(chain, x, p)
        })
    }
}

/// Runs `evaluate` in the instruction set `I`.
#[inline]
fn evaluate_in<I, C, F, K, N, const IN: usize, const P: usize>(
    chain: &C,
    x: [F; IN],
    p: [F; P],
) -> Option<F>
where
    I: Isa,
    C: Chain<IN, P>,
    F: Value<Format = K>,
    K: Format<Native = N>,
    N: Native,
{
    let (x, p) = (opaque(&x), opaque(&p));
    let x: [Lane<I, K>; IN] = core::array::from_fn(|index| Lane::of(x[index]));
    let p: [Lane<I, K>; P] = core::array::from_fn(|index| Lane::of(p[index]));
    let lane = chain.apply::<Lane<I, K>>(x, p);
    (!lane.taint && !N::is_nan_bits(lane.value.to_bits())).then(|| F::of_native(lane.value))
}

#[cfg(test)]
mod tests {
    use super::{Value, map};
    use crate::block::{Chain, Steps};
    use crate::{BF16, F16, F32, F64};

    /// `x * y`.
    struct Product;

    impl Chain<2, 0> for Product {
        fn apply<S: Steps>(&self, [x, y]: [S; 2], []: [S; 0]) -> S {
            x * y
        }
    }

    /// Returns what `map` reports for `x * one`.
    fn reports<F: Value>(x: F, one: F) -> Option<bool> {
        let mut out = [one];
        map(&Product, [&[x][..], &[one][..]], [], &mut out[..])
    }

    #[test]
    fn map_reports_exactly_a_lane_that_holds_a_nan() {
        // The largest finite value, both infinities, and the NaNs next to
        // them.
        let one = F32::from_bits(0x3F80_0000);
        let singles = [
            (0x7F7F_FFFF, false),
            (0x7F80_0000, false),
            (0xFF80_0000, false),
            (0x7F80_0001, true),
            (0xFFC0_0000, true),
        ];
        for (bits, nan) in singles {
            assert_eq!(reports(F32::from_bits(bits), one), Some(nan), "{bits:#x}");
        }
        let one = F64::from_bits(0x3FF0_0000_0000_0000);
        let doubles = [
            (0x7FEF_FFFF_FFFF_FFFF, false),
            (0x7FF0_0000_0000_0000, false),
            (0xFFF0_0000_0000_0000, false),
            (0x7FF0_0000_0000_0001, true),
            (0xFFF8_0000_0000_0000, true),
        ];
        for (bits, nan) in doubles {
            assert_eq!(reports(F64::from_bits(bits), one), Some(nan), "{bits:#x}");
        }
        let one = F16::from_bits(0x3C00);
        let halves = [
            (0x7BFF, false),
            (0x7C00, false),
            (0xFC00, false),
            (0x7C01, true),
            (0xFE00, true),
        ];
        for (bits, nan) in halves {
            assert_eq!(reports(F16::from_bits(bits), one), Some(nan), "{bits:#x}");
        }
        let one = BF16::from_bits(0x3F80);
        let bfloats = [
            (0x7F7F, false),
            (0x7F80, false),
            (0xFF80, false),
            (0x7F81, true),
            (0xFFC0, true),
        ];
        for (bits, nan) in bfloats {
            assert_eq!(reports(BF16::from_bits(bits), one), Some(nan), "{bits:#x}");
        }
    }
}
