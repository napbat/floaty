//! Checks that every entry point of a host path gives the result of its
//! `_with` method in the default mode, on every target.
//!
//! The other tests compare the `_with` methods, which always run the engine,
//! with the oracles. The engine gives the same bits on every host. So on
//! AArch64, where the C references do not build, this comparison checks each
//! host path against those oracles. In a build without a host path, the
//! entry points take the engine.
//!
//! - [`arithmetic`]: the operators, `sqrt`, and `mul_add`.
//! - [`double_double`]: the operators and `sqrt` of the double-double
//!   algorithms.
//! - [`conversions`]: `convert`, `to_int`, and `from_int`.
//! - [`integral`]: `round_to_integral`, in the default mode and in each
//!   rounding direction.
//! - [`comparisons`]: the comparison, the minimum, and the maximum.
//! - [`remainder`]: the remainder.
//!
//! [`operands`] gives the operands of every check.

mod arithmetic;
mod comparisons;
mod conversions;
mod double_double;
mod integral;
mod operands;
mod remainder;
