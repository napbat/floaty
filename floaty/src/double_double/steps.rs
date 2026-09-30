//! The binary64 steps of a double-double algorithm, each run under one
//! behavior, with the flags of every step collected.

use core::cmp::Ordering;

use crate::env::{Behavior, Flags, Rounding, StepOperation, StepResult};
use crate::float::F64;
use crate::host::{self, Operation, Ready};
use crate::integer::ToInt;

/// Runs binary64 operations under one behavior and collects their flags, as
/// the status register of a processor does.
///
/// The steps ignore saturation, because neither reference saturates.
/// Saturated steps also give wrong pairs: the sum of the largest binary64
/// value and itself would have that value in both halves.
///
/// The steps of an entry point that returns no flags can take the host paths
/// of binary64. A host step gives the bits of the engine, and a step that the
/// host path declines runs in the engine.
///
/// Each step in the engine reports itself to [`Behavior::observe`]. A host
/// step reports nothing, because only a mode takes the host paths, and a
/// mode observes nothing.
pub struct Steps<B> {
    behavior: B,
    flags: Flags,
    host: Option<Ready>,
}

// The steps are `#[inline]`. The report of each step makes them larger, and
// without the attribute the `+` operator of `DoubleDouble<Gcc>` left its
// comparisons out of line: it took 25.2 ns instead of 15.3 ns.
impl<B: Behavior> Steps<B> {
    /// Starts with no flags.
    pub fn new(behavior: B) -> Self {
        Self {
            behavior: behavior.without_saturation(),
            flags: Flags::NONE,
            host: None,
        }
    }

    /// Starts the steps of an entry point that returns no flags. The steps
    /// take the host paths of binary64 when the build has them and the host
    /// allows them, which one check decides for every step.
    pub fn without_flags(behavior: B) -> Self {
        let behavior = behavior.without_saturation();
        let host = host::ready(&behavior.env());
        Self {
            behavior,
            flags: Flags::NONE,
            host,
        }
    }

    /// Returns the result of a host step, or `None` when the step runs in the
    /// engine.
    #[inline]
    fn host_step(&self, step: impl FnOnce(Ready) -> Option<u64>) -> Option<F64> {
        self.host.and_then(step).map(F64::from_bits)
    }

    /// Runs `block` on steps that round to nearest even under the other
    /// fields of the behavior, and keeps the flags of its steps, as glibc's
    /// `SET_RESTORE_ROUND (FE_TONEAREST)` does.
    pub fn rounding_to_nearest<T>(&mut self, block: impl FnOnce(&mut Steps<B::Rounded>) -> T) -> T {
        let mut nearest = Steps::new(self.behavior.rounded(Rounding::TiesToEven));
        let result = block(&mut nearest);
        self.flags |= nearest.flags;
        result
    }

    /// Returns the flags of every step so far.
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Adds the flags of one step and reports the step.
    #[inline]
    fn record(&mut self, operation: StepOperation, result: StepResult, flags: Flags) {
        self.flags |= flags;
        self.behavior.observe(operation, result, flags);
    }

    /// Records one step with a binary64 result, and returns the result.
    #[inline]
    fn record_value(&mut self, operation: StepOperation, (result, flags): (F64, Flags)) -> F64 {
        self.record(operation, StepResult::Value(result.to_bits()), flags);
        result
    }

    /// Records one comparison, and returns its order.
    #[inline]
    fn record_order(
        &mut self,
        operation: StepOperation,
        (order, flags): (Option<Ordering>, Flags),
    ) -> Option<Ordering> {
        self.record(operation, StepResult::Order(order), flags);
        order
    }

    /// Returns an operator of two values, from the host or the engine.
    #[inline]
    fn operator(&mut self, a: F64, b: F64, operation: Operation) -> F64 {
        let host = |ready: Ready| ready.binary(a.to_bits(), b.to_bits(), operation);
        if let Some(result) = self.host_step(host) {
            return result;
        }
        let (x, y) = (a.to_bits(), b.to_bits());
        let (operation, step) = match operation {
            Operation::Add => (StepOperation::Add(x, y), a.add_with(b, self.behavior)),
            Operation::Sub => (StepOperation::Sub(x, y), a.sub_with(b, self.behavior)),
            Operation::Mul => (StepOperation::Mul(x, y), a.mul_with(b, self.behavior)),
            Operation::Div => (StepOperation::Div(x, y), a.div_with(b, self.behavior)),
        };
        self.record_value(operation, step)
    }

    /// Returns `a + b`.
    #[inline]
    pub fn add(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Add)
    }

    /// Returns `a - b`.
    #[inline]
    pub fn sub(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Sub)
    }

    /// Returns `a * b`.
    #[inline]
    pub fn mul(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Mul)
    }

    /// Returns `a / b`.
    #[inline]
    pub fn div(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Div)
    }

    /// Returns the square root of `a`.
    #[inline]
    pub fn sqrt(&mut self, a: F64) -> F64 {
        if let Some(result) = self.host_step(|ready| ready.sqrt(a.to_bits())) {
            return result;
        }
        let step = a.sqrt_with(self.behavior);
        self.record_value(StepOperation::Sqrt(a.to_bits()), step)
    }

    /// Returns `a` truncated toward zero to an integer, as `cvttsd2si` and
    /// `cvtsi2sd` compute it. The truncation signals inexact for a fraction.
    /// `cvttsd2si` reports no denormal operand (Intel SDM Volume 2A,
    /// `CVTTSD2SI`, "SIMD Floating-Point Exceptions": Invalid and Precision),
    /// so the step reports none. `a` must be below 2^52 in magnitude.
    #[inline]
    pub fn truncate(&mut self, a: F64) -> F64 {
        let toward_zero = self.behavior.env().with_rounding(Rounding::TowardZero);
        let (integer, flags) = a.to_int_with::<i64>(toward_zero);
        let ToInt::Value(integer) = integer else {
            unreachable!("a value below 2^52 fits an i64");
        };
        let result = F64::from_int_with(integer, self.behavior).0;
        let flags = flags.difference(Flags::DENORMAL_INPUT);
        self.record_value(StepOperation::Truncate(a.to_bits()), (result, flags))
    }

    /// Returns `a * c + b`, rounded once.
    #[inline]
    pub fn fused_add(&mut self, a: F64, c: F64, b: F64) -> F64 {
        let host = |ready: Ready| ready.mul_add(a.to_bits(), c.to_bits(), b.to_bits());
        if let Some(result) = self.host_step(host) {
            return result;
        }
        let step = a.mul_add_with(c, b, self.behavior);
        let operation = StepOperation::MulAdd(a.to_bits(), c.to_bits(), b.to_bits());
        self.record_value(operation, step)
    }

    /// Returns the order of two values by a quiet comparison, which signals
    /// invalid only for a signaling NaN. `None` means unordered.
    #[inline]
    pub fn compare_quiet(&mut self, a: F64, b: F64) -> Option<Ordering> {
        let step = a.compare_quiet_with(b, self.behavior);
        self.record_order(StepOperation::CompareQuiet(a.to_bits(), b.to_bits()), step)
    }

    /// Returns the order of two values by a signaling comparison, which
    /// signals invalid for every NaN. `None` means unordered.
    #[inline]
    pub fn compare_signaling(&mut self, a: F64, b: F64) -> Option<Ordering> {
        let step = a.compare_signaling_with(b, self.behavior);
        self.record_order(
            StepOperation::CompareSignaling(a.to_bits(), b.to_bits()),
            step,
        )
    }
}

#[cfg(test)]
mod tests {
    use core::cell::Cell;

    use crate::env::{Env, Flags, Observed, Rounding, Step};
    use crate::{DoubleDouble, F64, Gcc};

    #[test]
    fn every_step_reports_itself() {
        let pair = |hi, lo| DoubleDouble::<Gcc>::from_parts(F64::from_bits(hi), F64::from_bits(lo));
        let bits = |value: DoubleDouble<Gcc>| (value.hi().to_bits(), value.lo().to_bits());
        // Finite nonzero operands take the block of `fmal` that rounds to
        // nearest even.
        let one = 0x3FF0_0000_0000_0000;
        let (x, y, z) = (
            pair(one, 0x3C30_0000_0000_0000),
            pair(0x4008_0000_0000_0000, 0),
            pair(one, 0),
        );
        let env = Env::IEEE.with_rounding(Rounding::TowardZero);
        let (union, caller, nearest) = (Cell::new(Flags::NONE), Cell::new(0), Cell::new(0));
        let observer = |step: &Step| {
            union.set(union.get() | step.flags);
            let count = if step.env == env {
                &caller
            } else {
                assert_eq!(step.env, env.with_rounding(Rounding::TiesToEven));
                &nearest
            };
            count.set(count.get() + 1);
        };
        let (result, flags) = x.mul_add_with(y, z, Observed::new(env, &observer));
        let (expected, expected_flags) = x.mul_add_with(y, z, env);
        assert_eq!((bits(result), flags), (bits(expected), expected_flags));
        assert_eq!(flags, union.get());
        // The comparisons before the block run in the direction of the call.
        assert!(caller.get() > 0 && nearest.get() > 0);
    }
}
