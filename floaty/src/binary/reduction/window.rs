//! The exact sum of a vector of terms, found in a window of 768 bits that
//! moves down.
//!
//! Each pass adds the bits of every term at or above the low end of the
//! window, modulo `2^768`. A sum of fewer than `2^64` terms then differs from
//! the true sum by less than `2^64` window units, because each term drops
//! less than one unit. Modular addition keeps the pass exact when the true
//! sum fits the window, even when a single term does not. A pass that cannot
//! decide the rounding gives a bound on the sum, and the next pass moves the
//! window down to that bound.

use core::cmp::Ordering;

use super::super::arithmetic::Term;
use crate::exact::Unrounded;
use crate::limbs::Limbs;

/// The limbs of the window. The window holds a result of every format with
/// its round bit, 491 bits at most, the margin of [`ERROR_BITS`] and its
/// guard bits below the result, and the bits of a term count above it.
pub(super) type Window = [u64; 12];

/// The width of [`Window`] in bits.
const WINDOW_BITS: i64 = 768;

/// The limbs that hold a term while it moves into the window: a doubled
/// significand of 1024 bits at most, shifted by less than the window.
type Wide = [u64; 32];

/// A sum has fewer than `2^64` terms, so the bits of the terms below the
/// window add to less than `2^64` window units, and the terms add to less
/// than `2^64` times the largest term.
const ERROR_BITS: i64 = 64;

/// The exact sum of a vector of terms.
pub(super) enum ExactSum {
    /// An exact zero.
    Zero,
    /// A nonzero sum, exact or with a sticky bit.
    Value(Unrounded<Window>),
}

/// One pass over the terms: their sum at or above the low end of the window,
/// modulo `2^768`, and the weight of the highest bit below the window.
struct Pass {
    total: Window,
    lost_top: Option<i64>,
}

/// Returns the exact sum of the terms that `terms` gives on each call, for
/// terms whose highest bit weighs at most `2^top`. The sum is exact, or it has
/// every bit that decides a rounding to `precision` bits or fewer, and a
/// sticky bit.
///
/// A pass whose sum is far above its error rounds as the window sum with a
/// sticky bit, unless the error can move the bits below the round bit across
/// a rounding boundary. Then that boundary becomes an offset, and the next
/// passes find the sign of the sum minus the offset. The next windows end
/// below the lowest bit of the offset, so the offset is a multiple of `2^768`
/// window units: the modular sum of a pass is then the sum minus the offset.
pub(super) fn exact_sum<D: Limbs, I: Iterator<Item = Term<D>>>(
    terms: impl Fn() -> I,
    top: i64,
    precision: u32,
) -> ExactSum {
    let precision = i64::from(precision);
    let margin = Window::ZERO.with_bit(u32::try_from(ERROR_BITS + 1).expect("a margin fits"));
    // The sum is below 2^bound in magnitude.
    let mut bound = top + 1 + ERROR_BITS;
    let mut offset: Option<Term<Window>> = None;
    loop {
        let low = bound + 1 - WINDOW_BITS;
        let pass = pass(terms(), low);
        let negative = pass.total.bit(767);
        let magnitude = if negative {
            negate(pass.total)
        } else {
            pass.total
        };
        let width = i64::from(magnitude.bit_length());
        let exact = pass.lost_top.is_none();
        match offset {
            // The sign of the sum minus the offset decides the rounding.
            Some(boundary) if exact || width > ERROR_BITS + 1 => {
                let sign = match (magnitude.is_zero(), negative == boundary.negative) {
                    (true, _) => Ordering::Equal,
                    (false, true) => Ordering::Greater,
                    (false, false) => Ordering::Less,
                };
                return ExactSum::Value(near_boundary(&boundary, sign));
            }
            None if exact => {
                if magnitude.is_zero() {
                    return ExactSum::Zero;
                }
                return ExactSum::Value(Unrounded {
                    negative,
                    exponent: exponent(low),
                    significand: magnitude,
                    sticky: false,
                });
            }
            None if width >= precision + ERROR_BITS + 6 => {
                let rest = width - precision - 1;
                let rest_bits = u32::try_from(rest).expect("the rest fits the window");
                let below = magnitude.low_bits(rest_bits);
                let ceiling = Window::ZERO.with_bit(rest_bits).sub(margin);
                let near_low = below.compare(&margin) == Ordering::Less;
                let near_high = below.compare(&ceiling) == Ordering::Greater;
                if !near_low && !near_high {
                    // The error leaves the bits below the round bit inside
                    // their interval, so they round as the window sum with a
                    // sticky bit does.
                    return ExactSum::Value(Unrounded {
                        negative,
                        exponent: exponent(low),
                        significand: magnitude,
                        sticky: true,
                    });
                }
                let kept = magnitude.shr(rest_bits);
                let significand = if near_high { kept.increment() } else { kept };
                offset = Some(Term {
                    negative,
                    exponent: low + rest,
                    significand,
                });
                // The sum minus the offset is below the margin plus the
                // error.
                bound = low + ERROR_BITS + 2;
                continue;
            }
            Some(_) | None => {}
        }
        let lost_bound = pass.lost_top.map_or(i64::MIN, |lost| lost + 1 + ERROR_BITS);
        bound = (low + width).max(lost_bound) + 1;
    }
}

/// Returns a value that rounds as a sum does, from a boundary of its
/// rounding and the sign of the sum minus the boundary. A value just above or
/// below the boundary lies between the same two boundaries as the sum.
fn near_boundary(boundary: &Term<Window>, sign: Ordering) -> Unrounded<Window> {
    if sign == Ordering::Equal {
        return Unrounded {
            negative: boundary.negative,
            exponent: exponent(boundary.exponent),
            significand: boundary.significand,
            sticky: false,
        };
    }
    let shifted = boundary.significand.shl(2);
    let significand = if sign == Ordering::Greater {
        shifted
    } else {
        shifted.sub(Window::ZERO.with_bit(0))
    };
    Unrounded {
        negative: boundary.negative,
        exponent: exponent(boundary.exponent - 2),
        significand,
        sticky: true,
    }
}

/// Returns a window position as the exponent of a value to round.
fn exponent(position: i64) -> i32 {
    i32::try_from(position).expect("a window position fits an i32")
}

/// Adds the terms at or above `low`.
fn pass<D: Limbs>(terms: impl Iterator<Item = Term<D>>, low: i64) -> Pass {
    let mut total = Window::ZERO;
    let mut lost_top = None;
    for term in terms {
        let (part, lost) = window_part(&term, low);
        total = if term.negative {
            add(total, &negate(part))
        } else {
            add(total, &part)
        };
        lost_top = lost_top.max(lost);
    }
    Pass { total, lost_top }
}

/// Returns the magnitude of the bits of a term at or above `low`, modulo
/// `2^768`, and the weight of its highest bit below `low`, if it has one. The
/// weight can be an upper bound.
fn window_part<D: Limbs>(term: &Term<D>, low: i64) -> (Window, Option<i64>) {
    let width = i64::from(term.significand.bit_length());
    let shift = term.exponent - low;
    if shift >= WINDOW_BITS {
        // Every bit is a multiple of 2^768 window units.
        return (Window::ZERO, None);
    }
    if shift + width <= 0 {
        return (Window::ZERO, Some(term.top()));
    }
    let window_bits = u32::try_from(WINDOW_BITS).expect("the window width fits");
    if shift >= 0 {
        let shift = u32::try_from(shift).expect("the shift is below the window");
        let part = term.significand.resize::<Wide>().shl(shift);
        return (part.low_bits(window_bits).resize(), None);
    }
    let dropped = u32::try_from(-shift).expect("the term reaches the window");
    let lost = term.significand.any_below(dropped).then_some(low - 1);
    let part = term.significand.shr(dropped).resize::<Wide>();
    (part.low_bits(window_bits).resize(), lost)
}

/// Adds two windows, modulo `2^768`.
fn add(first: Window, second: &Window) -> Window {
    let mut total = Window::ZERO;
    let mut carry = false;
    for ((slot, &a), &b) in total.iter_mut().zip(&first).zip(second) {
        let (partial, first_carry) = a.overflowing_add(b);
        let (value, second_carry) = partial.overflowing_add(u64::from(carry));
        *slot = value;
        carry = first_carry || second_carry;
    }
    total
}

/// Returns the two's complement of a window, modulo `2^768`.
fn negate(value: Window) -> Window {
    add(value.map(|limb| !limb), &Window::ZERO.with_bit(0))
}
