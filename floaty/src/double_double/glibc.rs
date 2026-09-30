//! The IBM `long double` functions of the libm of glibc 2.43,
//! `sysdeps/ieee754/ldbl-128ibm`, as the `libc6-dev-ppc64el-cross`
//! 2.43-2ubuntu2cross1 package compiles them for powerpc64le.
//!
//! Each function follows the machine code of its objects in `libm.a`: the
//! wrapper, such as `__sqrtl` in `w_sqrtl.o`, and the function that it
//! calls, such as `__ieee754_sqrtl` in `e_sqrtl.o`. The comments give the
//! instruction offsets. The functions compute on pairs with the libgcc
//! routines of [`gcc`], and the compiler swaps the operands of some of those
//! calls. A comment marks each swapped call.
//!
//! GCC compares two pairs with `fcmpu` on the high halves, and on the low
//! halves when the high halves are equal. The wrappers compare only to set
//! `errno`, but a quiet comparison signals invalid for a signaling NaN, so
//! the functions keep those comparisons. The integer code of `fmodl` runs
//! as its C source says, because integer operations give the same bits in
//! every compilation, except for the shifts that the source leaves without
//! a bound. The compiler emits those as the PowerPC `sld`, `srd`, and `srad`
//! instructions, which shift by the low 7 bits of the count.

mod fma;

use core::cmp::Ordering;

pub use self::fma::mul_add;
use super::Pair;
use super::gcc;
use super::steps::Steps;
use crate::env::Behavior;
use crate::float::F64;

/// The sign bit of a binary64 encoding.
const SIGN: u64 = 1 << 63;

/// The fraction field of a binary64 encoding.
const FRACTION: u64 = (1 << 52) - 1;

/// The encoding of positive infinity.
const INFINITY: u64 = 0x7FF0_0000_0000_0000;

/// The largest finite binary64 value, `DBL_MAX`: the constant of the
/// `isinf` test of the wrappers, at `.rodata.cst8` of `w_fmodl.o`.
const LARGEST: u64 = 0x7FEF_FFFF_FFFF_FFFF;

/// Returns the pair of two binary64 encodings.
fn pair(hi: u64, lo: u64) -> Pair {
    (F64::from_bits(hi), F64::from_bits(lo))
}

/// Returns the order of two pairs as GCC compares them: `fcmpu` of the high
/// halves, and of the low halves when those are equal. `None` means
/// unordered.
fn compare<B: Behavior>(steps: &mut Steps<B>, (a, aa): Pair, (b, bb): Pair) -> Option<Ordering> {
    match steps.compare_quiet(a, b) {
        Some(Ordering::Equal) => steps.compare_quiet(aa, bb),
        order => order,
    }
}

/// Returns the pair with both halves negated, as `fneg` does.
fn negate((hi, lo): Pair) -> Pair {
    (-hi, -lo)
}

/// `fabsl` as GCC inlines it: the magnitude of the high half, and the low
/// half negated unless `fcmpu` finds the high half equal to its magnitude.
/// So `-0` keeps its low half.
fn absolute<B: Behavior>(steps: &mut Steps<B>, (hi, lo): Pair) -> Pair {
    let magnitude = hi.abs();
    if steps.compare_quiet(hi, magnitude) == Some(Ordering::Equal) {
        (magnitude, lo)
    } else {
        (magnitude, -lo)
    }
}

/// `iscanonicall` of the IBM `long double` in glibc 2.43: `true` when a
/// pair is canonical.
pub fn is_canonical((hi, lo): Pair) -> bool {
    // `sysdeps/ieee754/ldbl-128ibm/s_iscanonicall.c` of glibc 2.43, on the
    // magnitudes of the halves.
    let high = hi.to_bits() & !(1 << 63);
    let low = lo.to_bits() & !(1 << 63);
    if low == 0 {
        return true;
    }
    let high_field = high >> 52;
    if high_field == 0x7FF {
        return high != 0x7FF0_0000_0000_0000;
    }
    // glibc compares the exponent fields: the high field must exceed the
    // low field by more than 53, or by 53 when the low half is a power of
    // two and the high half is even. A subnormal low half has the field
    // of its leading bit, `leading - 51`, so the limit is `leading + 2`.
    let (limit, low_power) = match low >> 52 {
        0 => {
            let leading = low.ilog2();
            (u64::from(leading) + 2, low == 1 << leading)
        }
        field => (field + 53, low.trailing_zeros() >= 52),
    };
    high_field > limit || (high_field == limit && low_power && high & 1 == 0)
}

/// `sqrtl`: the wrapper `__sqrtl` compares `x` with 0 to set `errno`
/// (0x1c), then calls `__ieee754_sqrtl` (0x2c).
pub fn sqrt<B: Behavior>(steps: &mut Steps<B>, x: Pair) -> Pair {
    let zero = pair(0, 0);
    compare(steps, x, zero);
    ieee754_sqrt(steps, x)
}

/// `__ieee754_sqrtl`: Newton's method from the binary64 square root of the
/// high half, scaled into [0.5, 2). The comments give the names of the
/// source.
fn ieee754_sqrt<B: Behavior>(steps: &mut Steps<B>, value: Pair) -> Pair {
    let zero = pair(0, 0);
    let (hi_bits, lo_bits) = (value.0.to_bits(), value.1.to_bits());
    let magnitude = hi_bits & !SIGN; // k
    // 0x28: outside 0x000FFFFF00000000 < k < 0x7FF0000000000000, as one
    // unsigned comparison. The source says 2^-1022 <= |x|, but the largest
    // subnormal high halves pass the test too, and take the Newton steps.
    if magnitude.wrapping_sub(0x000F_FFFF_0000_0001) > 0x7FE0_0000_FFFF_FFFE {
        return sqrt_outside(steps, value, magnitude);
    }
    // 0x48: x < 0 gives (big1 - big1) / (big - big), which GCC folds to
    // 0 / 0 at 0x21c.
    if compare(steps, value, zero) == Some(Ordering::Less) {
        return gcc::div(steps, zero, zero);
    }
    // l: the high half with the exponent 0x3FE or 0x3FF, and the low half
    // scaled by the same power of two.
    let high = (magnitude & 0x001F_FFFF_FFFF_FFFF) | 0x3FE0_0000_0000_0000;
    let low = if lo_bits & !SIGN == 0 {
        lo_bits
    } else {
        scaled_low(steps, lo_bits, high.wrapping_sub(magnitude))
    };
    let scaled = pair(high, low); // s
    let root = steps.sqrt(scaled.0); // d, 0xd0: `__ieee754_sqrt` is `fsqrt`
    let first = (root, F64::from_bits(0)); // i
    let scale = pair(
        0x2000_0000_0000_0000 + ((magnitude & 0x7FE0_0000_0000_0000) >> 1),
        0,
    ); // c
    let half = pair(0x3FE0_0000_0000_0000, 0);
    let quotient = gcc::div(steps, scaled, first); // 0x124
    let second = gcc::add(steps, quotient, first); // 0x138, swapped: i + s / i
    let second = gcc::mul(steps, second, half); // t, 0x150, swapped: 0.5L * t
    let quotient = gcc::div(steps, scaled, second); // 0x180
    let third = gcc::add(steps, quotient, second); // 0x198, swapped: t + s / t
    let third = gcc::mul(steps, third, half); // i, 0x1b0, swapped: 0.5L * i
    gcc::mul(steps, third, scale) // 0x1c8, swapped: c.x * i
}

/// Returns the encoding of the low half of `__ieee754_sqrtl`, scaled by the
/// power of two that scales the high half. `difference` is `l - k`, the
/// scaled high half minus the magnitude of the high half, as encodings.
fn scaled_low<B: Behavior>(steps: &mut Steps<B>, lo_bits: u64, difference: u64) -> u64 {
    let field = |bits: u64| i64::try_from((bits >> 52) & 0x7FF).expect("11 bits fit an i64");
    // 0x74 to 0x80: n = (int64_t) ((l - k) * 2) >> 53.
    let n = i64::from_ne_bytes((difference << 1).to_ne_bytes()) >> 53;
    let (mut bits, mut m) = (lo_bits, field(lo_bits));
    if m == 0 {
        // 0x90: a.d[1] *= two54 as `fmul` of 2^54 and the low half.
        let two54 = F64::from_bits(0x4350_0000_0000_0000);
        bits = steps.mul(two54, F64::from_bits(lo_bits)).to_bits();
        m = field(bits) - 54;
    }
    m += n;
    let fraction = bits & (SIGN | FRACTION);
    if m > 0 {
        // 0xb4: an exponent above 2047 carries into the sign bit, as the
        // shift of the C source does.
        let exponent = u64::from_ne_bytes(m.to_ne_bytes());
        return fraction | (exponent << 52);
    }
    if m < -53 {
        return bits & SIGN; // 0x2e4
    }
    // 0x2ec to 0x30c: a.d[1] *= twom54 as `fmul` of 2^-54 and the low half
    // at the exponent m + 54.
    let exponent = u64::try_from(m + 54).expect("the exponent is from 1 to 54");
    let twom54 = F64::from_bits(0x3C90_0000_0000_0000);
    steps
        .mul(twom54, F64::from_bits(fraction | (exponent << 52)))
        .to_bits()
}

/// The part of `__ieee754_sqrtl` for a magnitude `k` of the high half at or
/// below 0x000FFFFF00000000, or at or above infinity, from 0x1e0.
fn sqrt_outside<B: Behavior>(steps: &mut Steps<B>, x: Pair, k: u64) -> Pair {
    let zero = pair(0, 0);
    if k > LARGEST {
        // 0x250: x * x + x gives a NaN, or positive infinity.
        let square = gcc::mul(steps, x, x);
        return gcc::add(steps, square, x);
    }
    // 0x200: x == 0 returns x. The branch at 0x218 reads the same
    // comparison: x < 0 gives 0 / 0, and any other order scales x.
    match compare(steps, x, zero) {
        Some(Ordering::Equal) => x,
        Some(Ordering::Less) => gcc::div(steps, zero, zero),
        Some(Ordering::Greater) | None => {
            // 0x2b0 to 0x2cc: tm256 * __ieee754_sqrtl(x * t512), both
            // products swapped.
            let scaled = gcc::mul(steps, x, pair(0x5FF0_0000_0000_0000, 0));
            let root = ieee754_sqrt(steps, scaled);
            gcc::mul(steps, root, pair(0x2FF0_0000_0000_0000, 0))
        }
    }
}

/// `nextupl` of `s_nextupl.o`: the pair plus one unit in the 106th bit of
/// the high half.
pub fn next_up<B: Behavior>(steps: &mut Steps<B>, x: Pair) -> Pair {
    let (hx, lx) = (x.0.to_bits(), x.1.to_bits());
    let ihx = hx & !SIGN;
    if ihx > INFINITY {
        return gcc::add(steps, x, x); // 0xe8: x + x signals a signaling NaN
    }
    if ihx == 0 {
        return pair(1, 0); // 0xb0: LDBL_TRUE_MIN
    }
    if hx == LARGEST {
        if lx == 0x7C8F_FFFF_FFFF_FFFE {
            return pair(INFINITY, 0); // 0x210
        }
        // 0x1cc: the ulp of the largest high half, swapped: x + u.
        let u = pair(0x7FE0_0000_0000_0000 - (105 << 52), 0);
        return gcc::add(steps, u, x);
    }
    if hx == INFINITY | SIGN {
        return pair(LARGEST | SIGN, 0xFC8F_FFFF_FFFF_FFFE); // 0x1f0: -LDBL_MAX
    }
    if ihx <= 0x0360_0000_0000_0000 {
        // 0x118: x + LDBL_TRUE_MIN, and -0 for a zero sum.
        let sum = gcc::add(steps, x, pair(1, 0));
        return if compare(steps, sum, pair(0, 0)) == Some(Ordering::Equal) {
            pair(SIGN, 0)
        } else {
            sum
        };
    }
    // A power of two with a low half of the other sign, or a negative power
    // of two, has an ulp half as large.
    let below = if lx & !SIGN == 0 {
        hx & SIGN != 0
    } else {
        (hx ^ lx) & SIGN != 0
    };
    let ihx = if hx & FRACTION == 0 && below {
        ihx - (1 << 52)
    } else {
        ihx
    };
    let exponent = ihx & INFINITY;
    let u = if ihx < 106 << 52 {
        // 0x18c: an ulp below the normal range, as `fmul` of yhi and 2^-105.
        steps.mul(
            F64::from_bits(exponent),
            F64::from_bits(0x3960_0000_0000_0000),
        )
    } else {
        F64::from_bits(exponent - (105 << 52))
    };
    gcc::add(steps, (u, F64::from_bits(0)), x) // 0x19c, swapped: x + u
}

/// `nextdownl` of `s_nextdownl.o`: `-nextupl(-x)`, with `fneg` on both
/// halves.
pub fn next_down<B: Behavior>(steps: &mut Steps<B>, x: Pair) -> Pair {
    negate(next_up(steps, negate(x)))
}

/// The comparisons of the wrappers `__fmodl` and `__remainderl`, which test
/// `isinf (x) && !isnan (y)` and `y == 0 && !isnan (x)` to set `errno`.
fn wrapper_comparisons<B: Behavior>(steps: &mut Steps<B>, x: Pair, y: Pair) {
    let largest = F64::from_bits(LARGEST);
    if steps.compare_quiet(x.0.abs(), largest) == Some(Ordering::Greater) {
        steps.compare_quiet(y.0, y.0); // 0x64
    } else if compare(steps, y, pair(0, 0)) == Some(Ordering::Equal) {
        steps.compare_quiet(x.0, y.0); // 0xa4
    }
}

/// `fmodl`: the wrapper `__fmodl`, then `__ieee754_fmodl`.
pub fn fmod<B: Behavior>(steps: &mut Steps<B>, x: Pair, y: Pair) -> Pair {
    wrapper_comparisons(steps, x, y);
    ieee754_fmod(steps, x, y)
}

/// `remainderl`: the wrapper `__remainderl`, then `__ieee754_remainderl`.
pub fn remainder<B: Behavior>(steps: &mut Steps<B>, x: Pair, p: Pair) -> Pair {
    wrapper_comparisons(steps, x, p);
    ieee754_remainder(steps, x, p)
}

/// The mantissa of a pair as `ldbl_extract_mantissa` of `math_ldbl.h` gives
/// it: 106 bits, the high 48 in `hi` and the low 64 in `lo`, and the
/// exponent.
#[derive(Clone, Copy, Debug)]
struct Mantissa {
    hi: i64,
    lo: u64,
    exponent: i32,
}

/// Returns `value << count` as the PowerPC `sld` instruction shifts: by the
/// low 7 bits of the count, and to zero for 64 to 127.
fn shift_left_doubleword(value: u64, count: i64) -> u64 {
    let count = u32::try_from(count & 0x7F).expect("7 bits fit a u32");
    value.checked_shl(count).unwrap_or(0)
}

/// Returns `value >> count` as the PowerPC `srd` instruction shifts: by the
/// low 7 bits of the count, and to zero for 64 to 127.
fn shift_right_doubleword(value: u64, count: i64) -> u64 {
    let count = u32::try_from(count & 0x7F).expect("7 bits fit a u32");
    value.checked_shr(count).unwrap_or(0)
}

/// Returns `value >> count` as the PowerPC `srad` instruction shifts: by the
/// low 7 bits of the count, and to the sign for 64 to 127.
fn shift_right_algebraic(value: i64, count: i64) -> i64 {
    let count = u32::try_from(count & 0x7F).expect("7 bits fit a u32");
    value.checked_shr(count).unwrap_or(value >> 63)
}

/// `ldbl_extract_mantissa` of `math_ldbl.h`, inlined in `__ieee754_fmodl`.
///
/// The shift `lo << -ediff` has no guard in the source, and the compiler
/// emits it as `sld` (0x484 and 0x474). So a low half more than 127
/// exponents above its place shifts by the difference modulo 128.
fn extract_mantissa((hi, lo): Pair) -> Mantissa {
    let (hi_bits, lo_bits) = (hi.to_bits(), lo.to_bits());
    let field = |bits: u64| i32::try_from((bits >> 52) & 0x7FF).expect("11 bits fit an i32");
    let (hi_field, lo_field) = (field(hi_bits), field(lo_bits));
    let mut exponent = hi_field - 1023;
    let mut high = hi_bits & FRACTION;
    let mut low = lo_bits & FRACTION;
    if hi_field == 0 {
        // A subnormal high half has a zero low half.
        high <<= 1;
    } else {
        high |= 1 << 52;
        if lo_field == 0 {
            low <<= 1;
        } else {
            low |= 1 << 52;
        }
        low <<= 7;
        let ediff = hi_field - lo_field - 53;
        low = match ediff.cmp(&0) {
            Ordering::Greater if ediff < 64 => low >> ediff,
            Ordering::Greater => 0,
            Ordering::Less => shift_left_doubleword(low, i64::from(-ediff)),
            Ordering::Equal => low,
        };
        if (hi_bits ^ lo_bits) & SIGN != 0 && low != 0 {
            high -= 1;
            low = (1_u64 << 60).wrapping_sub(low);
            if high < 1 << 52 {
                // A borrow from the hidden bit.
                high = (high << 1) | (low >> 59);
                low = ((1 << 60) - 1) & (low << 1);
                exponent -= 1;
            }
        }
    }
    Mantissa {
        hi: i64::try_from(high >> 4).expect("49 bits fit an i64"),
        lo: (high << 60) | low,
        exponent,
    }
}

/// `ldbl_insert_mantissa` of `math_ldbl.h`: the pair of a sign, an exponent,
/// and a mantissa of `hi` and `lo` as [`extract_mantissa`] gives it.
fn insert_mantissa(negative: bool, exponent: i32, hi: i64, lo: u64) -> Pair {
    let hi = u64::from_ne_bytes(hi.to_ne_bytes());
    let sign = if negative { SIGN } else { 0 };
    // The exponent fields are 11-bit bit fields.
    let bias = |exponent: i32| u64::from_ne_bytes(i64::from(exponent + 1023).to_ne_bytes()) & 0x7FF;
    let mut high_field = bias(exponent);
    let mut low_sign = sign;
    let mut low_field = 0;
    let mut low = (lo >> 7) & ((1 << 53) - 1);
    let mut high = (lo >> 60) | (hi << 4);
    if low == 0 {
        low_sign = 0;
    } else {
        // The hidden bit of the low part rounds the high part.
        if low & (1 << 52) != 0 && (high & 1 != 0 || low & ((1 << 52) - 1) != 0) {
            high += 1;
            if high & (1 << 53) != 0 {
                high >>= 1;
                high_field = (high_field + 1) & 0x7FF;
            }
            low_sign = sign ^ SIGN;
            low = (1 << 53) - low;
        }
        let shift = low.leading_zeros() - (64 - 53);
        low <<= shift;
        let low_exponent =
            exponent - 53 + 1023 - i32::try_from(shift).expect("a shift fits an i32");
        if low_exponent >= 1 {
            low_field = u64::try_from(low_exponent).expect("the exponent is positive") & 0x7FF;
        } else if low_exponent > -53 {
            low >>= 1 - low_exponent;
        } else {
            low = 0;
        }
    }
    pair(
        sign | (high_field << 52) | (high & FRACTION),
        low_sign | (low_field << 52) | (low & FRACTION),
    )
}

/// The zeros of `__ieee754_fmodl`, `Zero[]` of `.rodata`: `+0` and `-0`,
/// each with a `+0` low half.
fn signed_zero(negative: bool) -> Pair {
    pair(if negative { SIGN } else { 0 }, 0)
}

/// `__ieee754_fmodl`: the remainder of the truncated quotient, by a long
/// division of the mantissas.
fn ieee754_fmod<B: Behavior>(steps: &mut Steps<B>, x: Pair, y: Pair) -> Pair {
    let (x_hi, x_lo) = (x.0.to_bits(), x.1.to_bits());
    let (y_hi, y_lo) = (y.0.to_bits(), y.1.to_bits());
    let (sx, sy) = (x_hi & SIGN, y_hi & SIGN);
    let (hx, hy) = (x_hi ^ sx, y_hi ^ sy);
    if hy == 0 || hx >= INFINITY || hy > INFINITY {
        // 0x438 and 0x448: (x * y) / (x * y).
        let product = gcc::mul(steps, x, y);
        return gcc::div(steps, product, product);
    }
    let negative = sx != 0;
    if hx <= hy {
        if hx < hy {
            return x;
        }
        if x_lo & !SIGN == 0 && y_lo & !SIGN == 0 {
            return signed_zero(negative);
        }
        let signed = |bits: u64| i64::from_ne_bytes(bits.to_ne_bytes());
        let (x_rest, y_rest) = (signed(x_lo ^ sx), signed(y_lo ^ sy));
        if (y_lo ^ sy) & SIGN == 0 && x_rest < y_rest {
            return x;
        }
        if (x_lo ^ sx) & SIGN != 0 && x_rest > y_rest {
            return x;
        }
        if x_rest == y_rest {
            return signed_zero(negative);
        }
    }
    let (mut dividend, mut divisor) = (extract_mantissa(x), extract_mantissa(y));
    for mantissa in [&mut dividend, &mut divisor] {
        if mantissa.exponent == -1023 {
            // A subnormal value, shifted to normal.
            while mantissa.hi & (1 << 48) == 0 {
                mantissa.hi = (mantissa.hi << 1) | i64::from(u8::from(mantissa.lo >> 63 == 1));
                mantissa.lo <<= 1;
                mantissa.exponent -= 1;
            }
        }
    }
    let Mantissa {
        hi: mut hx,
        lo: mut lx,
        exponent: ix,
    } = dividend;
    let Mantissa {
        hi: hy,
        lo: ly,
        exponent: mut iy,
    } = divisor;
    let carry = |bits: u64| i64::from(u8::from(bits >> 63 == 1));
    let difference = |hx: i64, lx: u64| {
        let hz = hx - hy - i64::from(u8::from(lx < ly));
        (hz, lx.wrapping_sub(ly))
    };
    // Fixed-point fmod: one quotient bit per step.
    for _ in 0..ix - iy {
        let (hz, lz) = difference(hx, lx);
        if hz < 0 {
            hx = hx + hx + carry(lx);
            lx <<= 1;
        } else {
            if hz == 0 && lz == 0 {
                return signed_zero(negative);
            }
            hx = hz + hz + carry(lz);
            lx = lz << 1;
        }
    }
    let (hz, lz) = difference(hx, lx);
    if hz >= 0 {
        hx = hz;
        lx = lz;
    }
    if hx == 0 && lx == 0 {
        return signed_zero(negative);
    }
    while hx < 1 << 48 {
        hx = hx + hx + carry(lx);
        lx <<= 1;
        iy -= 1;
    }
    if iy >= -1022 {
        return insert_mantissa(negative, iy, hx, lx);
    }
    // A subnormal result. The source says that 1 <= n <= 52, but a
    // non-canonical operand can give a larger n, and the compiler emits the
    // shifts as `srd`, `srad`, and `sld` (0x264 to 0x26c). The product with 1
    // of the source has no instruction in the object.
    let n = i64::from(-1022 - iy);
    let high = u64::from_ne_bytes(hx.to_ne_bytes());
    lx = shift_right_doubleword(lx, n) | shift_left_doubleword(high, 64 - n);
    hx = shift_right_algebraic(hx, n);
    insert_mantissa(negative, -1023, hx, lx)
}

/// `__ieee754_remainderl`: the IEEE 754 remainder from `fmodl` by twice the
/// divisor, and at most two subtractions.
fn ieee754_remainder<B: Behavior>(steps: &mut Steps<B>, x: Pair, p: Pair) -> Pair {
    let (x_hi, x_lo) = (x.0.to_bits(), x.1.to_bits());
    let (p_hi, p_lo) = (p.0.to_bits(), p.1.to_bits());
    let sx = x_hi & SIGN;
    // The low halves with the sign of their high halves cleared, and -0 as 0.
    let rest = |low: u64, high: u64| {
        let low = low ^ (high & SIGN);
        if low == SIGN { 0 } else { low }
    };
    let (lx, lp) = (rest(x_lo, x_hi), rest(p_lo, p_hi));
    let (hx, hp) = (x_hi & !SIGN, p_hi & !SIGN);
    if hp == 0 || hx >= INFINITY || hp > INFINITY {
        // 0x1f0 and 0x200: (x * p) / (x * p).
        let product = gcc::mul(steps, x, p);
        return gcc::div(steps, product, product);
    }
    let mut x = x;
    if hp <= 0x7FDF_FFFF_FFFF_FFFF {
        let twice = gcc::add(steps, p, p); // 0x290
        x = ieee754_fmod(steps, x, twice); // 0x2b0
    }
    // The halves of the original operands decide this test.
    if hx == hp && lx == lp {
        return gcc::mul(steps, x, pair(0, 0)); // 0x240, swapped: zero * x
    }
    let mut x = absolute(steps, x);
    let p = absolute(steps, p);
    if hp & 0x7FE0_0000_0000_0000 == 0 {
        // hp < 0x0020000000000000.
        let twice = gcc::add(steps, x, x); // 0x10c
        if compare(steps, twice, p) == Some(Ordering::Greater) {
            x = gcc::sub(steps, x, p); // 0x148
            let twice = gcc::add(steps, x, x); // 0x160
            // 0x188: `cror eq,lt,so` skips a smaller or unordered sum.
            if matches!(
                compare(steps, twice, p),
                Some(Ordering::Greater | Ordering::Equal)
            ) {
                x = gcc::sub(steps, x, p); // 0x1a4
            }
        }
    } else {
        let half = gcc::mul(steps, p, pair(0x3FE0_0000_0000_0000, 0)); // 0x31c, swapped
        if compare(steps, x, half) == Some(Ordering::Greater) {
            x = gcc::sub(steps, x, p); // 0x368
            // 0x384: x >= p_half as `fcmpu` of p_half and x, and `cror
            // eq,gt,so` skips a larger or unordered p_half.
            if matches!(
                compare(steps, half, x),
                Some(Ordering::Less | Ordering::Equal)
            ) {
                x = gcc::sub(steps, x, p); // 0x3a8
            }
        }
    }
    if sx != 0 { negate(x) } else { x } // 0x1c8
}
