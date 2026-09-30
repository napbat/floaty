//! The lanes of x87 extended precision. The x87 unit has no packed
//! instructions, so each lane runs the scalar instruction, after one check of
//! the control word for all lanes.

use super::super::Operation;
use super::super::environment;
use super::super::paths::extended;
use crate::env::Mode;
use crate::float::{Float, FloatType};
use crate::format::Standard;

/// Returns `operation` of each pair of lanes, or `None` when a lane is a NaN.
#[inline]
pub(super) fn binary<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: Operation,
) -> Option<[Float<S, W, M>; N]> {
    let mut lanes = *left;
    for (lane, other) in lanes.iter_mut().zip(right) {
        let (x, y) = (
            extended::<S, W>(lane.to_bits()),
            extended::<S, W>(other.to_bits()),
        );
        *lane = Float::from_host(environment::x87_binary(&x, &y, operation)?);
    }
    Some(lanes)
}

/// Returns the square root of each lane, or `None` when a lane is a NaN.
#[inline]
pub(super) fn sqrt<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    value: &[Float<S, W, M>; N],
) -> Option<[Float<S, W, M>; N]> {
    let mut lanes = *value;
    for lane in &mut lanes {
        let x = extended::<S, W>(lane.to_bits());
        *lane = Float::from_host(environment::x87_sqrt(&x)?);
    }
    Some(lanes)
}
