//! The IBM double-double arithmetic of libgcc, `config/rs6000/ibm-ldouble.c`,
//! as GCC 15.2.0 compiles it for powerpc64le.
//!
//! Each function follows the machine code of `__gcc_qadd`, `__gcc_qsub`,
//! `__gcc_qmul`, or `__gcc_qdiv` in `ibm-ldouble.o`, one instruction at a
//! time. The comments give the instruction offsets. The operand order of each
//! instruction decides which NaN a step returns, and GCC fused `a*d + w` of
//! `__gcc_qmul` into one `fmadd`.
//!
//! `fcmpu` is a quiet comparison. `bge` branches unless the first operand is
//! less, and `ble` unless it is greater, so both branch for an unordered
//! comparison. A function that returns one `double` returns it with a
//! positive zero low half, which `xxlxor` makes.

use core::cmp::Ordering;

use super::Pair;
use super::steps::Steps;
use crate::env::Behavior;
use crate::float::F64;

/// Positive infinity, the constant at `.toc`.
const INFINITY: u64 = 0x7FF0_0000_0000_0000;

/// The largest finite binary64 value, the constant at `.toc+0x8`.
const LARGEST: u64 = 0x7FEF_FFFF_FFFF_FFFF;

/// 2^-969, the constant at `.toc+0x10`: `__gcc_qdiv` scales a dividend at or
/// below it.
const TINY: u64 = 0x0360_0000_0000_0000;

/// 2^106, the constant at `.toc+0x18`: the scale of a tiny dividend.
const SCALE: u64 = 0x4690_0000_0000_0000;

/// Returns `a * c - b`, rounded once, as the PowerPC `fmsub` instruction
/// does: the instruction selects a NaN operand before it negates `b`, so a
/// NaN `b` keeps its sign. A NaN `b` makes the result a NaN, so the step then
/// selects from `b` itself.
pub fn fmsub<B: Behavior>(steps: &mut Steps<B>, a: F64, c: F64, b: F64) -> F64 {
    if b.is_nan() {
        steps.fused_add(a, c, b)
    } else {
        steps.fused_add(a, c, -b)
    }
}

/// Returns one binary64 result with a positive zero low half.
fn single(value: F64) -> Pair {
    (value, F64::from_bits(0))
}

/// Returns `true` when `bge` branches after `fcmpu` of `a` and `b`: unless
/// `a` is less than `b`.
fn not_less<B: Behavior>(steps: &mut Steps<B>, a: F64, b: F64) -> bool {
    steps.compare_quiet(a, b) != Some(Ordering::Less)
}

/// Returns `true` when `ble` branches after `fcmpu` of `a` and `b`: unless
/// `a` is greater than `b`.
fn not_greater<B: Behavior>(steps: &mut Steps<B>, a: F64, b: F64) -> bool {
    steps.compare_quiet(a, b) != Some(Ordering::Greater)
}

/// Returns `true` when `z` is not finite, by `fcmpu` of `|z|` and infinity.
fn nonfinite<B: Behavior>(steps: &mut Steps<B>, z: F64) -> bool {
    not_less(steps, z.abs(), F64::from_bits(INFINITY))
}

/// The common tail of `__gcc_qadd` (0x40 to 0x78) and `__gcc_qsub` (0x140 to
/// 0x178): the high sum `z` and the sum `zz` of the low terms.
fn finish<B: Behavior>(steps: &mut Steps<B>, z: F64, zz: F64) -> Pair {
    // 0x40: a zero `zz` returns `z` alone, which keeps a -0 result.
    if steps.compare_quiet(zz, F64::from_bits(0)) == Some(Ordering::Equal) {
        return single(z);
    }
    let xh = steps.add(z, zz); // fadd f12,f0,f4
    if nonfinite(steps, xh) {
        return single(xh);
    }
    let xl = steps.sub(z, xh); // fsub f0,f0,f12
    let xl = steps.add(xl, zz); // fadd f0,f0,f4
    (xh, xl)
}

/// `__gcc_qadd`: `(a, aa) + (c, cc)`.
pub fn add<B: Behavior>(steps: &mut Steps<B>, (a, aa): Pair, (c, cc): Pair) -> Pair {
    let z = steps.add(a, c); // 0x0c fadd f0,f1,f3
    if nonfinite(steps, z) {
        return add_nonfinite(steps, (a, aa), (c, cc), z);
    }
    let q = steps.sub(a, z); // 0x20 fsub f12,f1,f0
    let zq = steps.add(z, q); // 0x28 fadd f9,f0,f12
    let cq = steps.add(c, q); // 0x2c fadd f3,f3,f12
    let sum = steps.sub(a, zq); // 0x30 fsub f1,f1,f9
    let sum = steps.add(sum, cq); // 0x34 fadd f1,f1,f3
    let sum = steps.add(sum, aa); // 0x38 fadd f2,f1,f2
    let zz = steps.add(sum, cc); // 0x3c fadd f4,f2,f4
    finish(steps, z, zz)
}

/// `__gcc_qadd` from 0x80, where `z = a + c` is not finite.
fn add_nonfinite<B: Behavior>(steps: &mut Steps<B>, (a, aa): Pair, (c, cc): Pair, z: F64) -> Pair {
    // 0x80: a NaN `z` returns; an infinite `z` recomputes from every term.
    if not_greater(steps, z.abs(), F64::from_bits(LARGEST)) {
        return single(z);
    }
    let lows = steps.add(aa, cc); // 0x90 fadd f2,f2,f4
    let sum = steps.add(c, lows); // 0x94 fadd f12,f3,f2
    let sum = steps.add(sum, a); // 0x98 fadd f12,f12,f1
    if nonfinite(steps, sum) {
        return single(sum);
    }
    // 0xa8: the larger of `|a|` and `|c|` starts the low half.
    let low = if not_greater(steps, a.abs(), c.abs()) {
        let low = steps.sub(c, sum); // 0xe0 fsub f3,f3,f12
        let low = steps.add(low, a); // 0xe4 fadd f3,f3,f1
        steps.add(low, lows) // 0xe8 fadd f0,f3,f2
    } else {
        let low = steps.sub(a, sum); // 0xb8 fsub f1,f1,f12
        let low = steps.add(low, c); // 0xbc fadd f1,f1,f3
        steps.add(low, lows) // 0xc0 fadd f0,f1,f2
    };
    (sum, low)
}

/// `__gcc_qsub`: `(a, aa) - (c, cc)`.
pub fn sub<B: Behavior>(steps: &mut Steps<B>, (a, aa): Pair, (c, cc): Pair) -> Pair {
    let z = steps.sub(a, c); // 0x10c fsub f0,f1,f3
    if nonfinite(steps, z) {
        return sub_nonfinite(steps, (a, aa), (c, cc), z);
    }
    let q = steps.sub(a, z); // 0x120 fsub f12,f1,f0
    let qz = steps.add(q, z); // 0x128 fadd f9,f12,f0
    let qc = steps.sub(q, c); // 0x12c fsub f12,f12,f3
    let sum = steps.sub(a, qz); // 0x130 fsub f1,f1,f9
    let sum = steps.add(sum, qc); // 0x134 fadd f3,f1,f12
    let sum = steps.add(sum, aa); // 0x138 fadd f2,f3,f2
    let zz = steps.sub(sum, cc); // 0x13c fsub f4,f2,f4
    finish(steps, z, zz)
}

/// `__gcc_qsub` from 0x180, where `z = a - c` is not finite.
fn sub_nonfinite<B: Behavior>(steps: &mut Steps<B>, (a, aa): Pair, (c, cc): Pair, z: F64) -> Pair {
    if not_greater(steps, z.abs(), F64::from_bits(LARGEST)) {
        return single(z);
    }
    let lows = steps.sub(aa, cc); // 0x190 fsub f2,f2,f4
    let sum = steps.sub(lows, c); // 0x194 fsub f12,f2,f3
    let sum = steps.add(sum, a); // 0x198 fadd f12,f12,f1
    if nonfinite(steps, sum) {
        return single(sum);
    }
    let low = if not_greater(steps, a.abs(), c.abs()) {
        let low = steps.sub(-c, sum); // 0x1e0 fneg f0,f3; 0x1e4 fsub f0,f0,f12
        let low = steps.add(low, a); // 0x1e8 fadd f0,f0,f1
        steps.add(low, lows) // 0x1ec fadd f0,f0,f2
    } else {
        let low = steps.sub(a, sum); // 0x1b8 fsub f1,f1,f12
        let low = steps.sub(low, c); // 0x1bc fsub f1,f1,f3
        steps.add(low, lows) // 0x1c0 fadd f0,f1,f2
    };
    (sum, low)
}

/// `__gcc_qmul`: `(a, b) * (c, d)`.
// The names are those of `ibm-ldouble.c`, so that each step reads against
// the reference.
#[allow(clippy::many_single_char_names)]
pub fn mul<B: Behavior>(steps: &mut Steps<B>, (a, b): Pair, (c, d): Pair) -> Pair {
    let t = steps.mul(a, c); // 0x208 fmul f0,f1,f3
    // 0x210: a zero product returns alone, which keeps -0.
    if steps.compare_quiet(t, F64::from_bits(0)) == Some(Ordering::Equal) || nonfinite(steps, t) {
        return single(t);
    }
    let w = steps.mul(c, b); // 0x22c fmul f2,f3,f2
    let tau = fmsub(steps, a, c, t); // 0x230 fmsub f3,f1,f3,f0
    let cross = steps.fused_add(a, d, w); // 0x234 fmadd f4,f1,f4,f2
    let tau = steps.add(cross, tau); // 0x238 fadd f4,f4,f3
    let u = steps.add(t, tau); // 0x23c fadd f12,f0,f4
    if nonfinite(steps, u) {
        return single(u);
    }
    let low = steps.sub(t, u); // 0x24c fsub f0,f0,f12
    let low = steps.add(low, tau); // 0x254 fadd f0,f0,f4
    (u, low)
}

/// `__gcc_qdiv`: `(a, b) / (c, d)`.
// The names are those of `ibm-ldouble.c`, so that each step reads against
// the reference.
#[allow(clippy::many_single_char_names)]
pub fn div<B: Behavior>(steps: &mut Steps<B>, (a, b): Pair, (c, d): Pair) -> Pair {
    let t = steps.div(a, c); // 0x298 fdiv f0,f1,f3
    if steps.compare_quiet(t, F64::from_bits(0)) == Some(Ordering::Equal) || nonfinite(steps, t) {
        return single(t);
    }
    // 0x2c0: `cror eq,gt,so` and `bne` scale a dividend at or below 2^-969,
    // so that the low part of `c * t` is exact.
    let tiny = matches!(
        steps.compare_quiet(a.abs(), F64::from_bits(TINY)),
        Some(Ordering::Less | Ordering::Equal)
    );
    let (a, b, c, d) = if tiny {
        let scale = F64::from_bits(SCALE);
        (
            steps.mul(a, scale), // 0x328 fmul f1,f1,f12
            steps.mul(b, scale), // 0x32c fmul f2,f2,f12
            steps.mul(c, scale), // 0x330 fmul f3,f3,f12
            steps.mul(d, scale), // 0x334 fmul f4,f4,f12
        )
    } else {
        (a, b, c, d)
    };
    let s = steps.mul(c, t); // 0x2d4 fmul f12,f3,f0
    let w = fmsub(steps, d, t, b); // 0x2d8 fmsub f2,f4,f0,f2
    let sigma = fmsub(steps, c, t, s); // 0x2e0 fmsub f10,f3,f0,f12
    let v = steps.sub(a, s); // 0x2e4 fsub f1,f1,f12
    let v = steps.sub(v, sigma); // 0x2e8 fsub f1,f1,f10
    let v = steps.sub(v, w); // 0x2ec fsub f2,f1,f2
    let tau = steps.div(v, c); // 0x2f0 fdiv f3,f2,f3
    let u = steps.add(t, tau); // 0x2f4 fadd f12,f0,f3
    if nonfinite(steps, u) {
        return single(u);
    }
    let low = steps.sub(t, u); // 0x304 fsub f0,f0,f12
    let low = steps.add(low, tau); // 0x30c fadd f0,f0,f3
    (u, low)
}

#[cfg(test)]
mod tests {
    use super::fmsub;
    use crate::double_double::steps::Steps;
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
        assert_eq!(fmsub(&mut steps, one, one, nan).to_bits(), nan.to_bits());
        // A NaN first factor comes before the NaN addend, and the NaN addend
        // comes before a NaN second factor, with its sign.
        let factor = F64::from_bits(0x7FF8_0000_0000_0002);
        assert_eq!(
            fmsub(&mut steps, factor, one, nan).to_bits(),
            factor.to_bits()
        );
        assert_eq!(fmsub(&mut steps, one, factor, nan).to_bits(), nan.to_bits());
        assert_eq!(steps.flags(), Flags::NONE);
        // A signaling addend signals invalid, and the quiet result keeps its
        // sign.
        let signaling = F64::from_bits(0xFFF0_0000_0000_0003);
        assert_eq!(
            fmsub(&mut steps, one, one, signaling).to_bits(),
            0xFFF8_0000_0000_0003
        );
        assert_eq!(steps.flags(), Flags::INVALID);
        let mut steps = Steps::new(env);
        // A number addend is negated.
        assert_eq!(fmsub(&mut steps, one, one, one).to_bits(), 0);
        assert_eq!(steps.flags(), Flags::NONE);
    }
}
