//! The double-double arithmetic of QD 2.3.24, `dd_real`, configured with
//! `--enable-ieee-add`, `--disable-sloppy-div`, and `--enable-fma=c99`:
//! IEEE-style addition, accurate division, and a `two_prod` with
//! `fma(a, b, -p)`.
//!
//! Each function follows the machine code that g++ 15.2.0 makes of QD with
//! `-O2 -ffp-contract=off` for x86-64: addition, subtraction, and
//! multiplication as the `floaty-verify` shim inlines them in `run`, division
//! in `dd_real::accurate_div`, the remainder `drem` as the shim inlines it in
//! `run_remainder`, and the square root in `sqrt(const dd_real&)` and the
//! truncated remainder in `fmod(const dd_real&, const dd_real&)` of
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

/// `std::floor` as g++ inlines it for SSE2 in `drem` and `fmod`: a value of
/// 2^52 or more, or a NaN, is its own floor (the `ucomisd` with 2^52), and a
/// smaller one truncates by `cvttsd2si`, which signals inexact for a
/// fraction. `cmpnlesd` then subtracts 1 from a truncation above the value,
/// or +0 from another, and `orpd` sets the sign of the value. In a
/// direction toward negative, +0 - +0 is -0, so a fraction below 1 has the
/// floor -0.
fn floor<B: Behavior>(steps: &mut Steps<B>, x: F64) -> F64 {
    let two52 = F64::from_bits(0x4330_0000_0000_0000);
    if steps.compare_quiet(two52, x.abs()) != Some(Ordering::Greater) {
        return x;
    }
    let truncated = steps.truncate(x);
    let step = if steps.compare_signaling(truncated, x) == Some(Ordering::Greater) {
        F64::from_bits(ONE)
    } else {
        F64::from_bits(0)
    };
    with_sign_of(steps.sub(truncated, step), x)
}

/// `std::ceil` as g++ inlines it for SSE2 in `fmod`: as [`floor`], but
/// `cmpnlesd` adds 1 to a truncation below the value. The operands of the
/// addition are finite, so their order does not change the sum.
fn ceil<B: Behavior>(steps: &mut Steps<B>, x: F64) -> F64 {
    let two52 = F64::from_bits(0x4330_0000_0000_0000);
    if steps.compare_quiet(two52, x.abs()) != Some(Ordering::Greater) {
        return x;
    }
    let truncated = steps.truncate(x);
    let step = if steps.compare_signaling(x, truncated) == Some(Ordering::Greater) {
        F64::from_bits(ONE)
    } else {
        F64::from_bits(0)
    };
    with_sign_of(steps.add(step, truncated), x)
}

/// Returns `value` with the sign bit of `sign` set too, as `orpd` does.
fn with_sign_of(value: F64, sign: F64) -> F64 {
    F64::from_bits(value.to_bits() | (sign.to_bits() & (1 << 63)))
}

/// The encoding of 1.
const ONE: u64 = 0x3FF0_0000_0000_0000;

/// The encoding of 0.5.
const HALF: u64 = 0x3FE0_0000_0000_0000;

/// `quick_two_sum(hi, lo)`: the sum and its error.
fn quick_two_sum<B: Behavior>(steps: &mut Steps<B>, hi: F64, lo: F64) -> Pair {
    let s = steps.add(hi, lo);
    let bb = steps.sub(s, hi);
    (s, steps.sub(lo, bb))
}

/// `qd::nint(d)` of `inline.h`: `d` when it equals its floor, and
/// `floor(d + 0.5)` otherwise. Returns the result and whether `d` equals its
/// floor.
fn nearest_integer<B: Behavior>(steps: &mut Steps<B>, d: F64) -> (F64, bool) {
    let floor_d = floor(steps, d);
    if steps.compare_quiet(d, floor_d) == Some(Ordering::Equal) {
        return (d, true);
    }
    let sum = steps.add(d, F64::from_bits(HALF));
    (floor(steps, sum), false)
}

/// `nint(const dd_real&)` of `dd_inline.h`, as the shim inlines it in
/// `drem` (`run_remainder` from 0x4d7): QD rounds a tie up, and the low half
/// breaks a tie of the high half.
fn nint<B: Behavior>(steps: &mut Steps<B>, (q0, q1): Pair) -> Pair {
    let (hi, integer) = nearest_integer(steps, q0); // 0x4ff to 0x578
    // 0x541 jumps to the integer path at once for an integer q0, and 0x578
    // compares hi with q0 otherwise.
    if integer || steps.compare_quiet(q0, hi) == Some(Ordering::Equal) {
        let (lo, _) = nearest_integer(steps, q1); // 0x670 to 0x6e1
        return quick_two_sum(steps, hi, lo); // 0x6e6
    }
    let zero = F64::from_bits(0);
    let difference = steps.sub(hi, q0).abs(); // 0x588
    let tie = steps.compare_quiet(difference, F64::from_bits(HALF)) == Some(Ordering::Equal);
    // 0x5a4: `comisd` of 0 and q1 signals invalid for every NaN.
    if tie && steps.compare_signaling(zero, q1) == Some(Ordering::Greater) {
        return (steps.sub(hi, F64::from_bits(ONE)), zero); // 0x8a0
    }
    (hi, zero)
}

/// `aint(const dd_real&)` of `dd_inline.h`, as `fmod` of `dd_real.o` inlines
/// it (from 0x71e5): `floor` for a high half at or above 0, and `ceil`
/// otherwise. The `comisd` of q0 and 0 signals invalid for every NaN, and a
/// NaN takes `ceil`.
fn aint<B: Behavior>(steps: &mut Steps<B>, (q0, q1): Pair) -> Pair {
    let zero = F64::from_bits(0);
    let round: fn(&mut Steps<B>, F64) -> F64 = if matches!(
        steps.compare_signaling(q0, zero),
        Some(Ordering::Less) | None
    ) {
        ceil
    } else {
        floor
    };
    let hi = round(steps, q0);
    if steps.compare_quiet(q0, hi) != Some(Ordering::Equal) {
        return (hi, zero); // 0x7260
    }
    let lo = round(steps, q1);
    quick_two_sum(steps, hi, lo) // 0x723b
}

/// The product of the multiplier `n` and the divisor `b` in `drem`
/// (`run_remainder` from 0x703): `n * b` with the factors of each product
/// swapped.
fn remainder_product<B: Behavior>(steps: &mut Steps<B>, (n0, n1): Pair, (b0, b1): Pair) -> Pair {
    let p = steps.mul(b0, n0); // 0x716
    let e = steps.fused_add(n0, b0, -p); // 0x741: fma(b0, n0, -p)
    let first = steps.mul(b1, n0); // 0x763
    let second = steps.mul(b0, n1); // 0x74c
    let cross = steps.add(first, second); // 0x77e
    let p2 = steps.add(cross, e); // 0x782, swapped: p2 += cross
    let high = steps.add(p, p2); // 0x786
    let bb = steps.sub(high, p); // 0x792
    (high, steps.sub(p2, bb)) // 0x7a4
}

/// The product of the divisor `b` and the multiplier `n` in `fmod` of
/// `dd_real.o` (from 0x7268): `b * n` with the factors of each product
/// swapped.
fn truncated_product<B: Behavior>(steps: &mut Steps<B>, (b0, b1): Pair, (n0, n1): Pair) -> Pair {
    let p = steps.mul(n0, b0); // 0x727b
    let e = steps.fused_add(b0, n0, -p); // 0x7297: fma(n0, b0, -p)
    let first = steps.mul(n0, b1); // 0x72a8
    let second = steps.mul(b0, n1); // 0x72b3
    let cross = steps.add(first, second); // 0x72c0
    let p2 = steps.add(cross, e); // 0x72c4, swapped: p2 += cross
    let high = steps.add(p, p2); // 0x72c8
    let bb = steps.sub(high, p); // 0x72d0
    (high, steps.sub(p2, bb)) // 0x72dc
}

/// `drem(const dd_real&, const dd_real&)` of `dd_inline.h`, as the shim
/// inlines it in `run_remainder`: `a - nint(a / b) * b`. The quotient rounds
/// in the division, so a large quotient gives an approximate remainder, as
/// QD does. The subtraction has the operand order of [`sub`].
pub fn drem<B: Behavior>(steps: &mut Steps<B>, a: Pair, b: Pair) -> Pair {
    let quotient = div(steps, a, b); // 0x4bc: accurate_div
    let n = nint(steps, quotient);
    let product = remainder_product(steps, n, b);
    sub(steps, a, product) // 0x78a to 0x80e
}

/// `fmod(const dd_real&, const dd_real&)` of `dd_real.o`: `a - b *
/// aint(a / b)`. The subtraction has the operand order of [`sub`].
pub fn fmod<B: Behavior>(steps: &mut Steps<B>, a: Pair, b: Pair) -> Pair {
    let quotient = div(steps, a, b); // 0x71e0: accurate_div
    let n = aint(steps, quotient);
    let product = truncated_product(steps, b, n);
    sub(steps, a, product) // 0x72d4 to 0x735b
}
