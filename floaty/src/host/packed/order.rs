//! The packed paths of the comparison and of the minimum and maximum
//! operations, for binary32 and binary64 lanes.
//!
//! A comparison gives no NaN, so each lane gives the order of the quiet
//! predicates, and an unordered lane gives `None`. `MINPS` and `MAXPS` give
//! the right lane for a NaN and for two zeros, so a NaN or two zeros in any
//! pair of lanes sends every lane to its scalar operation. For every other
//! pair, each operation of the minimum and maximum families selects the
//! smaller or the larger lane, as the instruction does.

use core::cmp::Ordering;

use super::super::environment::{self, packed};
use super::super::paths::{min_max_differs, min_max_differs_64, min_max_f32, min_max_f64};
use super::{chunk, double_lanes, doubles, in_chunks, single_lanes, singles};
use crate::env::Mode;
use crate::float::Float;
use crate::format::Standard;
use crate::format::internal::{Host, MinMax};

/// Returns the order of each pair of lanes, or `None` for a format without a
/// packed comparison.
#[inline]
pub(super) fn compare<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
) -> Option<[Option<Ordering>; N]> {
    let mut orders = [None; N];
    match S::HOST {
        Host::Single => {
            let (x, y) = (singles(left)?, singles(right)?);
            in_chunks::<Option<Ordering>, N, 8, 4>(
                &mut orders,
                |start| Some(packed::compare_f32x8(*chunk(x, start), *chunk(y, start))?.orders()),
                |start| Some(packed::compare_f32x4(*chunk(x, start), *chunk(y, start)).orders()),
                |index| Some(environment::compare_f32(x[index], y[index])),
            )?;
        }
        Host::Double => {
            let (x, y) = (doubles(left)?, doubles(right)?);
            in_chunks::<Option<Ordering>, N, 4, 2>(
                &mut orders,
                |start| Some(packed::compare_f64x4(*chunk(x, start), *chunk(y, start))?.orders()),
                |start| Some(packed::compare_f64x2(*chunk(x, start), *chunk(y, start)).orders()),
                |index| Some(environment::compare_f64(x[index], y[index])),
            )?;
        }
        Host::None | Host::Half | Host::BFloat | Host::Extended => return None,
    }
    Some(orders)
}

/// Returns the minimum or maximum operation `operation` of each pair of
/// lanes, or `None` when a pair holds a NaN or two zeros, or for a format
/// without a packed path.
#[inline]
pub(super) fn min_max<S: Standard<W>, const W: usize, M: Mode, const N: usize>(
    left: &[Float<S, W, M>; N],
    right: &[Float<S, W, M>; N],
    operation: MinMax,
) -> Option<[Float<S, W, M>; N]> {
    match S::HOST {
        Host::Single => {
            let (x, y) = (singles(left)?, singles(right)?);
            let differs = x.iter().zip(y).fold(false, |differs, (a, b)| {
                differs | min_max_differs(a.to_bits(), b.to_bits())
            });
            if differs {
                return None;
            }
            single_lanes(left, |lanes| {
                in_chunks::<f32, N, 8, 4>(
                    lanes,
                    |start| packed::min_max_f32x8(*chunk(x, start), *chunk(y, start), operation),
                    |start| {
                        Some(packed::min_max_f32x4(
                            *chunk(x, start),
                            *chunk(y, start),
                            operation,
                        ))
                    },
                    |index| Some(min_max_f32(x[index], y[index], operation)),
                )
            })
        }
        Host::Double => {
            let (x, y) = (doubles(left)?, doubles(right)?);
            let differs = x.iter().zip(y).fold(false, |differs, (a, b)| {
                differs | min_max_differs_64(a.to_bits(), b.to_bits())
            });
            if differs {
                return None;
            }
            double_lanes(left, |lanes| {
                in_chunks::<f64, N, 4, 2>(
                    lanes,
                    |start| packed::min_max_f64x4(*chunk(x, start), *chunk(y, start), operation),
                    |start| {
                        Some(packed::min_max_f64x2(
                            *chunk(x, start),
                            *chunk(y, start),
                            operation,
                        ))
                    },
                    |index| Some(min_max_f64(x[index], y[index], operation)),
                )
            })
        }
        Host::None | Host::Half | Host::BFloat | Host::Extended => None,
    }
}
