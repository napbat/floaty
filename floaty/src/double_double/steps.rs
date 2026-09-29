//! The binary64 steps of a double-double algorithm, each run under one
//! behavior, with the flags of every step collected.

use core::cmp::Ordering;

use crate::env::{Behavior, Flags};
use crate::float::F64;
use crate::format::internal::Host;
use crate::host::{self, Kind, Operation, Ready};

/// Runs binary64 operations under one behavior and collects their flags, as
/// the status register of a processor does.
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
            behavior,
            flags: Flags::NONE,
            host: None,
        }
    }

    /// Starts the steps of an entry point that returns no flags. The steps
    /// take the host paths of binary64 when the build has them and the host
    /// allows them, which one check decides for every step.
    pub fn without_flags(behavior: B) -> Self {
        let host = if host::available(Host::Double, Kind::Arithmetic) {
            host::ready(&behavior.env())
        } else {
            None
        };
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

    /// Returns `a * c + b`, rounded once.
    pub fn fused_add(&mut self, a: F64, c: F64, b: F64) -> F64 {
        let host = |ready: Ready| ready.mul_add(a.to_bits(), c.to_bits(), b.to_bits());
        if let Some(result) = self.host_step(host) {
            return result;
        }
        let step = a.mul_add_with(c, b, self.behavior);
        self.record(step)
    }

    /// Returns `a * c - b`, rounded once, as the PowerPC `fmsub` instruction
    /// does: the instruction selects a NaN operand before it negates `b`, so
    /// a NaN `b` keeps its sign. A NaN `b` makes the result a NaN, so the step
    /// then selects from `b` itself.
    pub fn fused_sub(&mut self, a: F64, c: F64, b: F64) -> F64 {
        if b.is_nan() {
            self.fused_add(a, c, b)
        } else {
            self.fused_add(a, c, -b)
        }
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

#[cfg(test)]
mod tests {
    use super::Steps;
    use crate::env::{Env, Flags, FusedNanOrder, NanPropagation, NanRule};
    use crate::float::F64;

    #[test]
    fn a_fused_subtraction_keeps_the_sign_of_a_nan_addend() {
        let env = Env::IEEE.with_nan(
            NanRule::new(NanPropagation::FirstOperand)
                .with_fused_order(FusedNanOrder::AddendSecond),
        );
        let mut steps = Steps::new(env);
        let one = F64::from_bits(0x3FF0_0000_0000_0000);
        let nan = F64::from_bits(0xFFF8_0000_0000_0001);
        assert_eq!(steps.fused_sub(one, one, nan).to_bits(), nan.to_bits());
        // A NaN first factor comes before the NaN addend, and the NaN addend
        // comes before a NaN second factor, with its sign.
        let factor = F64::from_bits(0x7FF8_0000_0000_0002);
        assert_eq!(
            steps.fused_sub(factor, one, nan).to_bits(),
            factor.to_bits()
        );
        assert_eq!(steps.fused_sub(one, factor, nan).to_bits(), nan.to_bits());
        assert_eq!(steps.flags(), Flags::NONE);
        // A signaling addend signals invalid, and the quiet result keeps its
        // sign.
        let signaling = F64::from_bits(0xFFF0_0000_0000_0003);
        assert_eq!(
            steps.fused_sub(one, one, signaling).to_bits(),
            0xFFF8_0000_0000_0003
        );
        assert_eq!(steps.flags(), Flags::INVALID);
        let mut steps = Steps::new(env);
        // A number addend is negated.
        assert_eq!(steps.fused_sub(one, one, one).to_bits(), 0);
        assert_eq!(steps.flags(), Flags::NONE);
    }
}
