//! The binary64 steps of a double-double algorithm, each run under one
//! behavior, with the flags of every step collected.

use core::cmp::Ordering;

use crate::env::{Behavior, Env, Flags, Rounding};
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
pub struct Steps<B> {
    behavior: B,
    flags: Flags,
    host: Option<Ready>,
}

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
    fn host_step(&self, step: impl FnOnce(Ready) -> Option<u64>) -> Option<F64> {
        self.host.and_then(step).map(F64::from_bits)
    }

    /// Runs `block` on steps that round to nearest even under the other
    /// fields of the behavior, and keeps the flags of its steps, as glibc's
    /// `SET_RESTORE_ROUND (FE_TONEAREST)` does.
    pub fn rounding_to_nearest<T>(&mut self, block: impl FnOnce(&mut Steps<Env>) -> T) -> T {
        let mut nearest = Steps::new(self.behavior.env().with_rounding(Rounding::TiesToEven));
        let result = block(&mut nearest);
        self.flags |= nearest.flags;
        result
    }

    /// Returns the flags of every step so far.
    pub fn flags(&self) -> Flags {
        self.flags
    }

    /// Adds the flags of one step and returns its result.
    fn record(&mut self, (result, flags): (F64, Flags)) -> F64 {
        self.flags |= flags;
        result
    }

    /// Returns an operator of two values, from the host or the engine.
    fn operator(&mut self, a: F64, b: F64, operation: Operation) -> F64 {
        let host = |ready: Ready| ready.binary(a.to_bits(), b.to_bits(), operation);
        if let Some(result) = self.host_step(host) {
            return result;
        }
        let step = match operation {
            Operation::Add => a.add_with(b, self.behavior),
            Operation::Sub => a.sub_with(b, self.behavior),
            Operation::Mul => a.mul_with(b, self.behavior),
            Operation::Div => a.div_with(b, self.behavior),
        };
        self.record(step)
    }

    /// Returns `a + b`.
    pub fn add(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Add)
    }

    /// Returns `a - b`.
    pub fn sub(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Sub)
    }

    /// Returns `a * b`.
    pub fn mul(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Mul)
    }

    /// Returns `a / b`.
    pub fn div(&mut self, a: F64, b: F64) -> F64 {
        self.operator(a, b, Operation::Div)
    }

    /// Returns the square root of `a`.
    pub fn sqrt(&mut self, a: F64) -> F64 {
        if let Some(result) = self.host_step(|ready| ready.sqrt(a.to_bits())) {
            return result;
        }
        let step = a.sqrt_with(self.behavior);
        self.record(step)
    }

    /// Returns `a` truncated toward zero to an integer, as `cvttsd2si` and
    /// `cvtsi2sd` compute it. The truncation signals inexact for a fraction.
    /// `cvttsd2si` reports no denormal operand (Intel SDM Volume 2A,
    /// `CVTTSD2SI`, "SIMD Floating-Point Exceptions": Invalid and Precision),
    /// so the step reports none. `a` must be below 2^52 in magnitude.
    pub fn truncate(&mut self, a: F64) -> F64 {
        let toward_zero = self.behavior.env().with_rounding(Rounding::TowardZero);
        let (integer, flags) = a.to_int_with::<i64>(toward_zero);
        self.flags |= flags.difference(Flags::DENORMAL_INPUT);
        let ToInt::Value(integer) = integer else {
            unreachable!("a value below 2^52 fits an i64");
        };
        F64::from_int_with(integer, self.behavior).0
    }

    /// Returns `a * c + b`, rounded once.
    pub fn fused_add(&mut self, a: F64, c: F64, b: F64) -> F64 {
        let host = |ready: Ready| ready.mul_add(a.to_bits(), c.to_bits(), b.to_bits());
        if let Some(result) = self.host_step(host) {
            return result;
        }
        let step = a.mul_add_with(c, b, self.behavior);
        self.record(step)
    }

    /// Returns the order of two values by a quiet comparison, which signals
    /// invalid only for a signaling NaN. `None` means unordered.
    pub fn compare_quiet(&mut self, a: F64, b: F64) -> Option<Ordering> {
        let (order, flags) = a.compare_quiet_with(b, self.behavior);
        self.flags |= flags;
        order
    }

    /// Returns the order of two values by a signaling comparison, which
    /// signals invalid for every NaN. `None` means unordered.
    pub fn compare_signaling(&mut self, a: F64, b: F64) -> Option<Ordering> {
        let (order, flags) = a.compare_signaling_with(b, self.behavior);
        self.flags |= flags;
        order
    }
}
