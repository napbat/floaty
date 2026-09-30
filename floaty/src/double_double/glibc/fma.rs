//! `fmal` of the IBM `long double` in glibc 2.43, `s_fmal.c`, as `s_fmal.o`
//! of the pinned libm computes it, with `__frexp`, `__scalbn`, and `qsort`
//! of the pinned C library.

use core::cmp::Ordering;

use super::super::Pair;
use super::super::gcc;
use super::super::steps::Steps;
use super::{FRACTION, INFINITY, LARGEST, SIGN, absolute, compare, pair};
use crate::env::{Behavior, Env};
use crate::float::F64;

/// `LDBL_MAX`, the largest finite IBM `long double` of glibc, at
/// `.rodata.cst16+0x10` of `s_fmal.o`.
const LONG_DOUBLE_MAX: (u64, u64) = (LARGEST, 0x7C8F_FFFF_FFFF_FFFE);

/// `__scalbn` of `s_scalbn.o` in the C library: `x * 2^n` by the exponent
/// field, with a product for a subnormal operand or result, for an overflow,
/// and for an underflow. The products take finite operands, so their order
/// does not change the flags.
fn scalbn<B: Behavior>(steps: &mut Steps<B>, x: F64, n: i32) -> F64 {
    let field = |bits: u64| i64::try_from((bits >> 52) & 0x7FF).expect("11 bits fit an i64");
    let mut bits = x.to_bits();
    let mut k = field(bits);
    if k == 0 {
        if bits & FRACTION == 0 {
            return x;
        }
        bits = steps
            .mul(x, F64::from_bits(0x4350_0000_0000_0000))
            .to_bits(); // 0xc0: x * 2^54
        k = field(bits) - 54;
    }
    if k == 0x7FF {
        return steps.add(x, x); // 0xe0
    }
    // 0x7c and 0xa4: copysign(huge, x) * huge, and the same for tiny.
    let signed = |magnitude: u64| F64::from_bits(magnitude | (bits & SIGN));
    let huge = 0x7E37_E43C_8800_759C; // 1e300
    let tiny = 0x01A5_6E1F_C2F8_F359; // 1e-300
    if n < -50_000 {
        return steps.mul(signed(tiny), F64::from_bits(tiny));
    }
    let k = k + i64::from(n);
    if n > 50_000 || k > 0x7FE {
        return steps.mul(signed(huge), F64::from_bits(huge));
    }
    let fraction = bits & (SIGN | FRACTION);
    if k > 0 {
        let k = u64::try_from(k).expect("the exponent is positive");
        return F64::from_bits(fraction | (k << 52));
    }
    if k < -53 {
        return steps.mul(signed(tiny), F64::from_bits(tiny));
    }
    // 0x110: a subnormal result, as a product with 2^-54.
    let k = u64::try_from(k + 54).expect("the exponent is from 1 to 54");
    steps.mul(
        F64::from_bits(fraction | (k << 52)),
        F64::from_bits(0x3C90_0000_0000_0000),
    )
}

/// A value of `s_fmal.c`: a fraction in [0.5, 1), as `frexp` gives it, or
/// zero, and the power of two that it multiplies.
#[derive(Clone, Copy, Debug)]
struct Scaled {
    fraction: F64,
    exponent: i32,
}

impl Scaled {
    /// Returns a zero with the exponent 0, as `s_fmal.c` clears a value.
    fn zero() -> Self {
        Self {
            fraction: F64::from_bits(0),
            exponent: 0,
        }
    }
}

/// `__frexp` of `s_frexp.o` in the C library: the fraction and the exponent
/// of a value, by its encoding. A zero, an infinity, and a NaN give `x + x`
/// and the exponent 0.
fn frexp<B: Behavior>(steps: &mut Steps<B>, x: F64) -> Scaled {
    let bits = x.to_bits();
    let field = (bits >> 52) & 0x7FF;
    if field.wrapping_sub(1) < 0x7FE {
        let exponent = i32::try_from(field).expect("11 bits fit an i32") - 1022;
        let shift = u64::from_ne_bytes(i64::from(exponent).to_ne_bytes()) << 52;
        return Scaled {
            fraction: F64::from_bits(bits.wrapping_sub(shift)),
            exponent,
        };
    }
    if i64::from_ne_bytes((bits << 1).to_ne_bytes()) <= 0 {
        return Scaled {
            fraction: steps.add(x, x),
            exponent: 0,
        };
    }
    // A subnormal value, normalized by its leading zeros.
    let leading = (bits << 11).leading_zeros();
    let shifted = bits << leading;
    Scaled {
        fraction: F64::from_bits((shifted & FRACTION) | (bits & SIGN) | (1022 << 52)),
        exponent: -1021 - i32::try_from(leading).expect("a count fits an i32"),
    }
}

/// `store_ext_val` of `s_fmal.c`.
fn store<B: Behavior>(steps: &mut Steps<B>, value: F64) -> Scaled {
    frexp(steps, value)
}

/// `mul_ext_val` of `s_fmal.c`: the product of two values as a sum of two
/// scaled values, from `mul_split`, which PowerPC computes with a fused
/// multiply-subtract.
fn multiply<B: Behavior>(steps: &mut Steps<B>, x: F64, y: F64) -> (Scaled, Scaled) {
    let (x, y) = (frexp(steps, x), frexp(steps, y));
    let hi = steps.mul(x.fraction, y.fraction);
    let lo = gcc::fmsub(steps, x.fraction, y.fraction, hi);
    let scale = x.exponent + y.exponent;
    let mut high = store(steps, hi);
    if !hi.is_zero() {
        high.exponent += scale;
    }
    let mut low = store(steps, lo);
    if !lo.is_zero() {
        low.exponent += scale;
    }
    (high, low)
}

/// `compare` of `s_fmal.c`, the order of `qsort`: zeros first, then by
/// exponent, then by magnitude.
///
/// A NaN from a malformed operand, a finite high half with a NaN or an
/// infinite low half, compares as the C source says: `fabs (p) < fabs (q)`
/// and `fabs (p) == fabs (q)` are false, so the order is `Greater` both
/// ways. The values hold no signaling NaN, so the comparisons signal
/// nothing.
fn order(p: &Scaled, q: &Scaled) -> Ordering {
    if p.fraction.is_zero() {
        return if q.fraction.is_zero() {
            Ordering::Equal
        } else {
            Ordering::Less
        };
    }
    if q.fraction.is_zero() {
        return Ordering::Greater;
    }
    match p.exponent.cmp(&q.exponent) {
        Ordering::Equal => {}
        order => return order,
    }
    let (magnitude_p, magnitude_q) = (p.fraction.abs(), q.fraction.abs());
    match magnitude_p.compare_quiet_with(magnitude_q, Env::IEEE).0 {
        Some(Ordering::Less) => Ordering::Less,
        Some(Ordering::Equal) => Ordering::Equal,
        Some(Ordering::Greater) | None => Ordering::Greater,
    }
}

/// Sorts scaled values as `qsort` of glibc 2.43 does: `msort_with_tmp` of
/// `stdlib/qsort.c`, a top-down merge sort that splits at `n / 2` and takes
/// the left value unless it compares greater. The order of a NaN is not
/// consistent, so only the same merges give the same order.
fn sort(values: &mut [Scaled]) {
    if values.len() <= 1 {
        return;
    }
    let middle = values.len() / 2;
    sort(&mut values[..middle]);
    sort(&mut values[middle..]);
    let mut merged = [Scaled::zero(); 10];
    let (mut left, mut right) = (0, middle);
    for slot in merged.iter_mut().take(values.len()) {
        let take_left = right == values.len()
            || (left < middle && order(&values[left], &values[right]) != Ordering::Greater);
        if take_left {
            *slot = values[left];
            left += 1;
        } else {
            *slot = values[right];
            right += 1;
        }
    }
    values.copy_from_slice(&merged[..values.len()]);
}

/// The operand order of the sum `*hi = x + y` of `add_split` at one inlined
/// site of `add_split_ext`. A sum of two NaNs gives the first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sum {
    /// `x + y`, as the source writes it.
    Source,
    /// `y + x`.
    Swapped,
}

/// The operand orders of the nine unrolled iterations of the second loop of
/// `__fmal`, for `i` from 1 to 9: the sums at 0x1428, 0x1308, 0x11d8,
/// 0x10a8, 0xf78, 0xe48, 0xd18, 0xbe8, and 0xad8.
const SECOND_LOOP: [Sum; 9] = [
    Sum::Swapped,
    Sum::Source,
    Sum::Swapped,
    Sum::Source,
    Sum::Source,
    Sum::Source,
    Sum::Swapped,
    Sum::Swapped,
    Sum::Swapped,
];

/// `add_split_ext` of `s_fmal.c`: replaces `values[high]` and
/// `values[low]`, which is at most as large, with their sum rounded to
/// nearest and its error. `sum` is the operand order of the sum at the
/// inlined site.
fn add_split<B: Behavior>(
    steps: &mut Steps<B>,
    values: &mut [Scaled; 10],
    (high, low): (usize, usize),
    sum: Sum,
) {
    let (high_exponent, low_exponent) = (values[high].exponent, values[low].exponent);
    if values[low].fraction.is_zero() || high_exponent - low_exponent > 53 {
        return;
    }
    let hi = values[high].fraction;
    let lo = scalbn(steps, values[low].fraction, low_exponent - high_exponent);
    let sum = match sum {
        Sum::Source => steps.add(hi, lo),
        Sum::Swapped => steps.add(lo, hi),
    };
    let difference = steps.sub(hi, sum);
    let rest = steps.add(difference, lo);
    values[high] = store(steps, sum);
    if !sum.is_zero() {
        values[high].exponent += high_exponent;
    }
    values[low] = store(steps, rest);
    if !rest.is_zero() {
        values[low].exponent += high_exponent;
    }
}

/// The end of the block of `__fmal` that rounds to nearest.
enum Fused {
    /// The result.
    Pair(Pair),
    /// An exact zero, which `zero_out` computes in the direction of the
    /// caller.
    Zero,
    /// A result outside the exponent range, which `scale_out` scales in the
    /// direction of the caller.
    Scale(F64, i32),
}

/// `fmal` of `s_fmal.o`: `x * y + z`.
///
/// Special operands take the libgcc routines. Otherwise the ten partial
/// values of the halves and their products are summed in round to nearest,
/// as accurate as the long double arithmetic, and only an overflow, an
/// underflow, or an exact zero reads the direction of the caller.
pub fn mul_add<B: Behavior>(steps: &mut Steps<B>, x: Pair, y: Pair, z: Pair) -> Pair {
    let zero = pair(0, 0);
    let special = |value: Pair| value.0.to_bits() & INFINITY == INFINITY;
    let product_sum = |steps: &mut Steps<B>| {
        let product = gcc::mul(steps, x, y); // 0x688
        gcc::add(steps, product, z) // 0x6a0: (x * y) + z
    };
    if special(z) {
        if !special(x) && !special(y) {
            let sum = gcc::add(steps, x, z); // 0x718, swapped: z + x
            return gcc::add(steps, sum, y); // 0x730
        }
        // 0x758: z == 0 fails for a NaN or an infinity.
        compare(steps, z, zero);
        return product_sum(steps);
    }
    if compare(steps, z, zero) == Some(Ordering::Equal) {
        // 0x660 and 0x7a0: x != 0 && y != 0 gives x * y.
        if compare(steps, x, zero) != Some(Ordering::Equal)
            && compare(steps, y, zero) != Some(Ordering::Equal)
        {
            return gcc::mul(steps, x, y); // 0x7c0
        }
        return product_sum(steps);
    }
    if special(x)
        || special(y)
        || compare(steps, x, zero) == Some(Ordering::Equal)
        || compare(steps, y, zero) == Some(Ordering::Equal)
    {
        return product_sum(steps);
    }
    match steps.rounding_to_nearest(|nearest| fused(nearest, x, y, z)) {
        Fused::Pair(result) => result,
        Fused::Zero => {
            // 0x63c: zero - zero in the direction of the caller.
            let zero = F64::from_bits(0);
            (steps.sub(zero, zero), zero)
        }
        Fused::Scale(fraction, exponent) => {
            let scaled = scalbn(steps, fraction, exponent); // 0xa2c
            let magnitude = scaled.abs();
            if steps.compare_quiet(magnitude, F64::from_bits(LARGEST)) == Some(Ordering::Equal) {
                // 0x15a0: copysignl (LDBL_MAX, scale_val).
                let (hi, lo) = LONG_DOUBLE_MAX;
                let sign = scaled.to_bits() & SIGN;
                return pair(hi | sign, lo | sign);
            }
            // 0xa58: math_check_force_underflow of a binary64 value.
            let smallest = F64::from_bits(0x0010_0000_0000_0000);
            if steps.compare_quiet(magnitude, smallest) == Some(Ordering::Less) {
                steps.mul(scaled, scaled);
            }
            (scaled, F64::from_bits(0))
        }
    }
}

/// The block of `__fmal` under `SET_RESTORE_ROUND (FE_TONEAREST)`, from the
/// C source and the operand orders of its machine code. The checks before
/// the block read only the high halves, so a malformed operand, a finite
/// high half with a NaN or an infinite low half, brings NaNs into the
/// block. So [`order`] and [`sort`] follow the source and `qsort` exactly,
/// and each sum of two values that can both be NaNs has the operand order of
/// its site.
fn fused(steps: &mut Steps<Env>, x: Pair, y: Pair, z: Pair) -> Fused {
    let mut values = [Scaled::zero(); 10];
    values[0] = store(steps, z.0);
    values[1] = store(steps, z.1);
    (values[2], values[3]) = multiply(steps, x.0, y.0);
    (values[4], values[5]) = multiply(steps, x.0, y.1);
    (values[6], values[7]) = multiply(steps, x.1, y.0);
    (values[8], values[9]) = multiply(steps, x.1, y.1);
    sort(&mut values);
    // The first loop, one site at 0x3f4.
    for index in 0..=8 {
        add_split(steps, &mut values, (index + 1, index), Sum::Source);
        sort(&mut values[index + 1..]);
    }
    let mut destination = 9;
    for (index, sum) in (1..=9).zip(SECOND_LOOP) {
        let source = 9 - index;
        if values[destination].fraction.is_zero() {
            values[destination] = values[source];
            values[source] = Scaled::zero();
        } else {
            add_split(steps, &mut values, (destination, source), sum);
            if !values[source].fraction.is_zero() {
                if source < destination - 1 {
                    values[destination - 1] = values[source];
                    values[source] = Scaled::zero();
                }
                destination -= 1;
            }
        }
    }
    if values[9].fraction.is_zero() {
        return Fused::Zero;
    }
    add_split(steps, &mut values, (9, 8), Sum::Source); // 0x81c
    if order(&values[8], &values[7]) == Ordering::Less {
        values.swap(7, 8);
    }
    add_split(steps, &mut values, (8, 7), Sum::Swapped); // 0x8ec
    add_split(steps, &mut values, (9, 8), Sum::Swapped); // 0x988
    if values[9].exponent > 1024 || values[9].exponent < -1021 {
        return Fused::Scale(values[9].fraction, values[9].exponent);
    }
    let hi = scalbn(steps, values[9].fraction, values[9].exponent); // 0x14f8
    let lo = scalbn(steps, values[8].fraction, values[8].exponent); // 0x150c
    // 0x1518 to 0x1520: ldbl_canonicalize.
    let sum = steps.add(hi, lo);
    let difference = steps.sub(hi, sum);
    let result = (sum, steps.add(difference, lo));
    // 0x1530 to 0x1578: math_check_force_underflow of the long double, as
    // `fabsl` and a comparison with LDBL_MIN, 2^-969.
    let magnitude = absolute(steps, result);
    if compare(steps, magnitude, pair(0x0360_0000_0000_0000, 0)) == Some(Ordering::Less) {
        gcc::mul(steps, result, result);
    }
    Fused::Pair(result)
}
