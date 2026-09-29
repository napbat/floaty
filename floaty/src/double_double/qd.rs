//! The double-double arithmetic of QD 2.3.24, `dd_real`, in the
//! configuration of `DESIGN.md`: IEEE-style addition, accurate division, and
//! a `two_prod` with `fma(a, b, -p)`.
//!
//! Each function follows the machine code that g++ 15.2.0 makes of QD with
//! `-O2 -ffp-contract=off` for x86-64: addition, subtraction, and
//! multiplication as the `floaty-verify` shim inlines them in `run`, division
//! in `dd_real::accurate_div`, and the square root in `sqrt(const dd_real&)` of
//! `dd_real.o` in `libqd.a`. The comments give the instruction offsets.
//!
//! The compiler keeps the arithmetic of the source, but it swaps the operands
//! of some additions, and x86 returns the NaN of the first operand. A comment
//! marks each swapped addition. The compiler also drops the error of a
//! `quick_two_sum` whose low half is unused: in the second remainder of the
//! division, and in the square root. So that step signals nothing.
//!
//! QD calls the C library `fma(x, y, z)` for the error of a product. glibc
//! runs it as `vfmadd213sd` on a processor with FMA3, which computes
//! `y * x + z` and takes the first NaN in that order. So each fused step
//! passes `y` as the first factor.

use core::cmp::Ordering;

use super::Pair;
use super::steps::Steps;
use crate::env::Behavior;
use crate::float::F64;

/// The NaN of QD, `std::numeric_limits<double>::quiet_NaN()`.
const NAN: u64 = 0x7FF8_0000_0000_0000;

/// The sum `s` of `two_sum(a, b)` and `bb = s - a`.
fn sum_and_bb<B: Behavior>(steps: &mut Steps<B>, a: F64, b: F64) -> (F64, F64) {
    let s = steps.add(a, b);
    (s, steps.sub(s, a))
}

/// `two_diff(a, b)` as the compiler makes it: the difference and its error,
/// `(a - (s - bb)) - (bb + b)`, with the operands of `b + bb` swapped.
fn two_diff<B: Behavior>(steps: &mut Steps<B>, a: F64, b: F64) -> Pair {
    let s = steps.sub(a, b);
    let bb = steps.sub(s, a);
    let left = steps.sub(s, bb);
    let left = steps.sub(a, left);
    let right = steps.add(bb, b); // swapped: b + bb
    (s, steps.sub(left, right))
}

/// The tail of IEEE-style addition and subtraction from `s2 += t1`: the sum
/// `s1 + s2` with the errors `t1` and `t2`, renormalized twice.
fn renormalize<B: Behavior>(steps: &mut Steps<B>, (s1, s2): Pair, (t1, t2): Pair) -> Pair {
    let s2 = steps.add(s2, t1);
    let s = steps.add(s1, s2);
    let bb = steps.sub(s, s1);
    let s2 = steps.sub(s2, bb);
    let s2 = steps.add(t2, s2); // swapped: s2 += t2
    let high = steps.add(s, s2);
    let bb = steps.sub(high, s);
    (high, steps.sub(s2, bb))
}

/// The high half of [`renormalize`], with the last sum swapped. The division
/// and the square root use only `r.x[0]`, so the compiler drops the error.
fn renormalize_high<B: Behavior>(steps: &mut Steps<B>, (s1, s2): Pair, (t1, t2): Pair) -> F64 {
    let s2 = steps.add(s2, t1);
    let s = steps.add(s1, s2);
    let bb = steps.sub(s, s1);
    let s2 = steps.sub(s2, bb);
    let s2 = steps.add(t2, s2); // swapped: s2 += t2
    steps.add(s2, s) // swapped: s + s2
}

/// The two `two_diff` calls of `dd_real - dd_real` with `QD_IEEE_ADD`: the
/// differences of the high halves and of the low halves, with their errors.
fn differences<B: Behavior>(steps: &mut Steps<B>, (a0, a1): Pair, (b0, b1): Pair) -> (Pair, Pair) {
    (two_diff(steps, a0, b0), two_diff(steps, a1, b1))
}

/// `dd_real + dd_real`, the IEEE-style `ieee_add` (`run` from 0x280).
pub fn add<B: Behavior>(steps: &mut Steps<B>, (a0, a1): Pair, (b0, b1): Pair) -> Pair {
    let (s1, bb) = sum_and_bb(steps, a0, b0); // 0x288, 0x29e
    let (t1, bb_t) = sum_and_bb(steps, a1, b1); // 0x28c, 0x2a3
    let left = steps.sub(s1, bb); // 0x2a7
    let right = steps.sub(b0, bb); // 0x2ac
    let right_t = steps.sub(b1, bb_t); // 0x2b1
    let left = steps.sub(a0, left); // 0x2b5
    let s2 = steps.add(left, right); // 0x2c2
    let left_t = steps.sub(t1, bb_t); // 0x2ca
    let left_t = steps.sub(a1, left_t); // 0x2ce
    let t2 = steps.add(left_t, right_t); // 0x2da
    renormalize(steps, (s1, s2), (t1, t2)) // 0x2c6 and 0x2d2, then 0x2de
}

/// `dd_real - dd_real` with `QD_IEEE_ADD` (`run` from 0x310).
pub fn sub<B: Behavior>(steps: &mut Steps<B>, a: Pair, b: Pair) -> Pair {
    let (highs, lows) = differences(steps, a, b); // 0x318 to 0x36b
    renormalize(steps, highs, lows) // 0x353 and 0x363, then 0x2de
}

/// `dd_real * dd_real` (`run` from 0x119).
pub fn mul<B: Behavior>(steps: &mut Steps<B>, (a0, a1): Pair, (b0, b1): Pair) -> Pair {
    let p = steps.mul(a0, b0); // 0x127
    let e = steps.fused_add(b0, a0, -p); // 0x141 xorpd; 0x14f call fma(a0, b0, -p)
    let first = steps.mul(a0, b1); // 0x160
    let second = steps.mul(a1, b0); // 0x16c
    let cross = steps.add(first, second); // 0x172
    let p2 = steps.add(cross, e); // 0x17a, swapped: p2 += cross
    let high = steps.add(p, p2); // 0x182
    let bb = steps.sub(high, p); // 0x18a
    (high, steps.sub(p2, bb)) // 0x18e
}

/// `dd_real * double` for `b * q` in `accurate_div`: `two_prod(b0, q)` and
/// `p2 += b1 * q`, renormalized.
fn mul_double<B: Behavior>(steps: &mut Steps<B>, (b0, b1): Pair, q: F64) -> Pair {
    let p = steps.mul(b0, q);
    let e = steps.fused_add(q, b0, -p); // fma(b0, q, -p)
    let low = steps.mul(b1, q);
    let p2 = steps.add(low, e); // swapped: p2 += b1 * q
    let high = steps.add(p, p2);
    let bb = steps.sub(high, p);
    (high, steps.sub(p2, bb))
}

/// `dd_real / dd_real`, the accurate `accurate_div`.
pub fn div<B: Behavior>(steps: &mut Steps<B>, a: Pair, b: Pair) -> Pair {
    let q1 = steps.div(a.0, b.0); // 0x27
    let product = mul_double(steps, b, q1); // 0x3b to 0xc2
    let (highs, lows) = differences(steps, a, product); // 0xa0 to 0x10d
    let r = renormalize(steps, highs, lows); // 0xf1 to 0x13a
    let q2 = steps.div(r.0, b.0); // 0x144
    let product = mul_double(steps, b, q2); // 0x157 to 0x1d4
    let (highs, lows) = differences(steps, r, product); // 0x1b6 to 0x227
    let high = renormalize_high(steps, highs, lows); // 0x203 to 0x23c
    let q3 = steps.div(high, b.0); // 0x241
    // `quick_two_sum(q1, q2)`, then `dd_real(q1, q2) + q3`.
    let quotient = steps.add(q2, q1); // 0x21e, swapped: q1 + q2
    let s = steps.add(q3, quotient); // 0x24f, swapped: two_sum(q1, q3)
    let bb = steps.sub(s, quotient); // 0x25c
    let quotient_bb = steps.sub(quotient, q1); // 0x260
    let left = steps.sub(s, bb); // 0x264
    let right = steps.sub(q3, bb); // 0x268
    let q2 = steps.sub(q2, quotient_bb); // 0x26c
    let left = steps.sub(quotient, left); // 0x270
    let s2 = steps.add(left, right); // 0x274
    let s2 = steps.add(q2, s2); // 0x278, swapped: s2 += q2
    let high = steps.add(s, s2); // 0x280
    let bb = steps.sub(high, s); // 0x28d
    (high, steps.sub(s2, bb)) // 0x291
}

/// `sqrt(dd_real)`, by Karp's method (`_Z4sqrtRK7dd_real` of `dd_real.o`).
///
/// One signaling comparison of the high half with 0 makes both tests of the
/// source: a zero gives `+0`, a negative value gives QD's NaN in both halves,
/// and a NaN goes on to the arithmetic. QD also writes an error for a
/// negative value.
pub fn sqrt<B: Behavior>(steps: &mut Steps<B>, a: Pair) -> Pair {
    let zero = F64::from_bits(0);
    match steps.compare_signaling(a.0, zero) {
        // 0x6c0 comisd; 0x6c6 je
        Some(Ordering::Equal) => return (zero, zero),
        // 0x6cc: neither unordered nor greater
        Some(Ordering::Less) => return (F64::from_bits(NAN), F64::from_bits(NAN)),
        _ => {}
    }
    let one = F64::from_bits(0x3FF0_0000_0000_0000);
    let half = F64::from_bits(0x3FE0_0000_0000_0000);
    let root = steps.sqrt(a.0); // 0x705
    let x = steps.div(one, root); // 0x70f
    let ax = steps.mul(a.0, x); // 0x713
    let square = steps.mul(ax, ax); // 0x72b
    let e = steps.fused_add(ax, ax, -square); // 0x735 xorpd; 0x744 call fma
    let scale = steps.mul(x, half); // 0x774
    let (highs, lows) = differences(steps, a, (square, e)); // 0x77c to 0x7cf
    let high = renormalize_high(steps, highs, lows); // 0x7b6 to 0x7df
    let correction = steps.mul(high, scale); // 0x7e3
    let s = steps.add(correction, ax); // 0x7eb, swapped: two_sum(ax, correction)
    let bb = steps.sub(s, ax); // 0x7fb
    let left = steps.sub(s, bb); // 0x7ff
    let right = steps.sub(correction, bb); // 0x803
    let left = steps.sub(ax, left); // 0x807
    (s, steps.add(left, right)) // 0x80b
}
