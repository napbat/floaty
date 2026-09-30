//! A behavior that reports each binary64 step of a double-double operation.
//! The replay oracle of `floaty-verify` checks each step with MPFR.

use core::cmp::Ordering;
use core::fmt;

use super::{Behavior, Env, Flags, Rounding};
use crate::sealed::Sealed;

/// A binary64 step of a double-double operation. A binary64 value is its
/// bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// The operation and its operands.
    pub operation: StepOperation,
    /// The result.
    pub result: StepResult,
    /// The behavior that the step runs under.
    pub env: Env,
    /// The flags of the step.
    pub flags: Flags,
}

/// The operation of a step, with its operands in operand order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepOperation {
    /// `a + b`.
    Add(u64, u64),
    /// `a - b`.
    Sub(u64, u64),
    /// `a * b`.
    Mul(u64, u64),
    /// `a / b`.
    Div(u64, u64),
    /// The square root of `a`.
    Sqrt(u64),
    /// `a * b + c`, rounded once.
    MulAdd(u64, u64, u64),
    /// `a` truncated toward zero to an integer, as `cvttsd2si` and
    /// `cvtsi2sd` compute it. The step reports no denormal operand.
    Truncate(u64),
    /// A quiet comparison of `a` and `b`.
    CompareQuiet(u64, u64),
    /// A signaling comparison of `a` and `b`.
    CompareSignaling(u64, u64),
}

/// The result of a step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepResult {
    /// A binary64 value.
    Value(u64),
    /// The order of a comparison. `None` means unordered.
    Order(Option<Ordering>),
}

/// A behavior that reports each binary64 step of a double-double operation
/// to an observer. The operations run as under the behavior `B`.
///
/// Only the operations of a double-double value with binary64 steps report
/// them: the arithmetic, the square root, the remainders, `mul_add`,
/// `next_up`, and `next_down`. Every other operation reports nothing.
#[derive(Clone, Copy)]
pub struct Observed<'a, B> {
    behavior: B,
    observer: &'a dyn Fn(&Step),
}

impl<'a, B: Behavior> Observed<'a, B> {
    /// Makes the behavior `behavior` that reports each step to `observer`.
    #[must_use]
    pub fn new(behavior: B, observer: &'a dyn Fn(&Step)) -> Self {
        Self { behavior, observer }
    }
}

impl<B: fmt::Debug> fmt::Debug for Observed<'_, B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Observed")
            .field("behavior", &self.behavior)
            .finish_non_exhaustive()
    }
}

impl<B> Sealed for Observed<'_, B> {}
impl<'a, B: Behavior> Behavior for Observed<'a, B> {
    type Rounded = Observed<'a, B::Rounded>;

    #[inline]
    fn env(self) -> Env {
        self.behavior.env()
    }

    #[inline]
    fn without_saturation(self) -> Self {
        Self {
            behavior: self.behavior.without_saturation(),
            ..self
        }
    }

    #[inline]
    fn rounded(self, rounding: Rounding) -> Self::Rounded {
        Observed {
            behavior: self.behavior.rounded(rounding),
            observer: self.observer,
        }
    }

    #[inline]
    fn observe(self, operation: StepOperation, result: StepResult, flags: Flags) {
        (self.observer)(&Step {
            operation,
            result,
            env: self.env(),
            flags,
        });
    }
}
