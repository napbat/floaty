//! Double-double values: the sum of two binary64 values, `hi + lo`, with the
//! arithmetic of one reference implementation.
//!
//! No standard defines the correct result of a double-double operation, and
//! the published algorithms give different low halves. So each algorithm
//! follows its references: [`Gcc`] the IBM `long double` of libgcc and of the
//! libm of glibc on PowerPC, and [`Qd`] the `dd_real` of the QD library. Each
//! algorithm follows the machine code that a pinned compiler makes of its
//! references, because the operand order of each step decides which NaN the
//! step returns. Under the behavior of the platform of the reference, the
//! results match it bit for bit, NaN halves included. That behavior is the
//! default mode of the algorithm, [`Algorithm::Reference`]. The operations
//! without a reference follow the exact value by the rule of
//! [`DoubleDouble`].
//!
//! Each step of an algorithm is a binary64 operation that runs under the
//! behavior of the call, and the result has the flags of every step, as a
//! processor that runs the reference sets them. `TINY`, `ROUNDED_UP`, and
//! `DENORMAL_INPUT` then say that some step was tiny, rounded up, or read a
//! subnormal operand.
//!
//! The steps ignore [`Env::saturate`], because neither reference saturates.
//! An overflow gives what the reference gives: an infinity or a NaN. A
//! conversion applies saturation to its binary destination format.

mod convert;
mod gcc;
mod glibc;
mod operations;
mod payload;
mod qd;
mod steps;
mod value;

use core::fmt;
use core::marker::PhantomData;

use self::steps::Steps;
use crate::env::{Behavior, Env, Flags, Mode, Override, mode};
use crate::float::{F64, Float};
use crate::format::Binary;
use crate::sealed::Sealed;

/// The high half and the low half of a value.
type Pair = (F64, F64);

/// An algorithm with one operand, on binary64 steps under the behavior `B`.
type OneOperand<B> = fn(&mut Steps<B>, Pair) -> Pair;

/// An algorithm with two operands, on binary64 steps under the behavior `B`.
type TwoOperands<B> = fn(&mut Steps<B>, Pair, Pair) -> Pair;

/// The algorithm of a double-double type: [`Gcc`] or [`Qd`]. The trait is
/// sealed.
pub trait Algorithm: Sealed + 'static {
    /// The mode under which the algorithm matches its reference. It is the
    /// default mode of the double-double type of the algorithm.
    type Reference: Mode;

    /// Which reference the algorithm follows.
    #[doc(hidden)]
    const KIND: AlgorithmKind;
}

/// The reference of an algorithm.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AlgorithmKind {
    /// libgcc on PowerPC.
    Gcc,
    /// The QD library.
    Qd,
}

/// The IBM `long double` of PowerPC: `__gcc_qadd`, `__gcc_qsub`,
/// `__gcc_qmul`, and `__gcc_qdiv` of libgcc, as GCC 15.2.0 compiles them for
/// powerpc64le, and `sqrtl`, `remainderl`, `fmodl`, `fmal`, `nextupl`, and
/// `nextdownl` of the libm of glibc 2.43, as its powerpc64le build compiles
/// them.
///
/// Its reference mode is [`mode::Libgcc`], the behavior of PowerPC under
/// QEMU. In that mode, the default mode of `DoubleDouble<Gcc>`, the results
/// and the five IEEE flags match libgcc, NaN halves included.
///
/// ```
/// use floaty::{DoubleDouble, F64, Gcc};
///
/// let one = DoubleDouble::<Gcc>::from_f64(F64::from_bits(0x3FF0_0000_0000_0000));
/// let nan = DoubleDouble::<Gcc>::from_f64(F64::from_bits(0x7FF8_0000_0000_0001));
/// // PowerPC returns the first NaN operand, with its payload.
/// assert_eq!((nan + one).hi().to_bits(), 0x7FF8_0000_0000_0001);
/// let (two, flags) = one.add_with(one, DoubleDouble::<Gcc>::ENV);
/// assert_eq!(two.hi().to_bits(), 0x4000_0000_0000_0000);
/// assert!(flags.is_empty());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gcc {}

/// The `dd_real` of QD 2.3.24, configured with IEEE-style addition, accurate
/// division, and a fused multiply-subtract in `two_prod`.
///
/// Its reference mode is [`mode::X86Sse`], the behavior of [`Env::X86_SSE`].
/// In that mode, the default mode of `DoubleDouble<Qd>`, the results and the
/// five IEEE flags match QD on x86-64, NaN halves included. The results
/// also match QD with flush-to-zero and denormals-are-zero, which MXCSR
/// sets. The algorithm follows the machine code that g++ 15.2.0 makes of QD
/// with `-O2 -ffp-contract=off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qd {}

impl Sealed for Gcc {}
impl Algorithm for Gcc {
    type Reference = mode::Libgcc;
    const KIND: AlgorithmKind = AlgorithmKind::Gcc;
}

impl Sealed for Qd {}
impl Algorithm for Qd {
    type Reference = mode::X86Sse;
    const KIND: AlgorithmKind = AlgorithmKind::Qd;
}

/// A double-double value: the sum of two binary64 values, with the arithmetic
/// of the algorithm `Alg` and the default mode `M`. `M` is the reference mode
/// of the algorithm unless the type names another mode.
///
/// The arithmetic ignores [`Env::saturate`], because neither reference
/// saturates. [`convert_with`](Self::convert_with) applies saturation to a
/// binary destination format. A conversion into a pair saturates an overflow
/// and an infinity to the largest finite pair.
///
/// # Rounding to a pair
///
/// The operations without a reference follow the exact value `hi + lo`: the
/// conversions, the integer conversions, `scale_b`, `round_to_integral`, the
/// classification, the total order, and the minimum and maximum operations.
/// They do not depend on the algorithm. They read the halves without
/// denormals-are-zero, and report no [`Flags::DENORMAL_INPUT`] for them.
///
/// An exact value rounds to a pair in two steps. The high half is the value
/// rounded to binary64 to nearest even. The low half is the rest rounded to
/// binary64 in the direction of the behavior. The sum of the halves is the
/// value of the result, and it splits into its canonical pair: the high half
/// is the sum rounded to nearest even, and the low half is the exact rest.
/// So a value that a canonical pair holds gives that pair in every direction.
/// A zero result has the sign of the value and a `+0` low half.
///
/// The result is inexact when the low half rounds. The low half gives the
/// other flags of the rounding: [`Flags::TINY`] when the low half is tiny,
/// and [`Flags::UNDERFLOW`] when the low half is also inexact, as for a
/// binary64 result. So a pair with an exact subnormal low half reports
/// `TINY`, and a subnormal value with a zero low half reports no `TINY`. The
/// result is rounded up when its magnitude is above that of the value.
///
/// An overflow gives an infinity with a `+0` low half, or the largest finite
/// canonical pair, as the direction and saturation select for binary64. That
/// pair is `(0x7FEF_FFFF_FFFF_FFFF, 0x7C8F_FFFF_FFFF_FFFF)`, `2^1024 - 2^970 -
/// 2^917`, one unit of `2^917` above the `LDBL_MAX` of GCC, which counts 106
/// bits. An infinity keeps its sign with a `+0` low half. A NaN converts to
/// binary64 as a conversion does, and takes a `+0` low half, and so does an
/// unsupported x87 encoding, which gives the default NaN and signals
/// invalid.
///
/// ```
/// use floaty::{DoubleDouble, F128, Flags, Qd};
///
/// // 1 + 2^-54 + 2^-112: the low half keeps 2^-54, and 2^-112 is below its
/// // precision.
/// let value = F128::from_bits(0x3FFF_0000_0000_0000_0400_0000_0000_0001);
/// let (pair, flags) = value.convert_with::<DoubleDouble<Qd>>(F128::ENV);
/// assert_eq!(pair.hi().to_bits(), 0x3FF0_0000_0000_0000);
/// assert_eq!(pair.lo().to_bits(), 0x3C90_0000_0000_0000);
/// assert_eq!(flags, Flags::INEXACT);
/// ```
///
/// ```
/// use floaty::{DoubleDouble, F64, Qd};
///
/// let one = DoubleDouble::<Qd>::from_f64(F64::from_bits(0x3FF0_0000_0000_0000));
/// let three = one + one + one;
/// let third = one / three;
/// // 1/3 = 0.333... is the high half plus a low half below its last bit.
/// assert_eq!(third.hi().to_bits(), 0x3FD5_5555_5555_5555);
/// assert_eq!(third.lo().to_bits(), 0x3C75_5555_5555_5555);
/// ```
pub struct DoubleDouble<Alg: Algorithm, M: Mode = <Alg as Algorithm>::Reference> {
    hi: F64,
    lo: F64,
    marker: PhantomData<(Alg, M)>,
}

impl<Alg: Algorithm, M: Mode> Clone for DoubleDouble<Alg, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Alg: Algorithm, M: Mode> Copy for DoubleDouble<Alg, M> {}

impl<Alg: Algorithm, M: Mode> fmt::Debug for DoubleDouble<Alg, M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DoubleDouble")
            .field("hi", &self.hi)
            .field("lo", &self.lo)
            .finish()
    }
}

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// The behavior of the default mode.
    pub const ENV: Env = M::ENV;

    /// The radix of both binary64 halves: 2.
    pub const RADIX: u32 = 2;

    /// The nominal precision of the reference, in binary digits: 106 for
    /// [`Gcc`], and 104 for [`Qd`].
    ///
    /// A double-double has no fixed-width significand. The low half can hold
    /// bits beyond this precision. This constant does not limit the exact
    /// value or the operations on that value.
    pub const PRECISION: u32 = match Alg::KIND {
        // GCC 15.2.0 defines __LDBL_MANT_DIG__ as 106 with -mabi=ibmlongdouble.
        AlgorithmKind::Gcc => 106,
        // QD 2.3.24, include/qd/dd_real.h, line 170: numeric_limits<dd_real>::digits.
        AlgorithmKind::Qd => 104,
    };

    /// Makes a value from its halves. The pair is kept as it is. The halves
    /// can have any mode, and the value has the mode `M`.
    #[must_use]
    pub fn from_parts<N: Mode>(hi: Float<Binary<11>, 64, N>, lo: Float<Binary<11>, 64, N>) -> Self {
        Self::new(hi.with_mode(), lo.with_mode())
    }

    /// Makes the value of a binary64 value: the value and a positive zero.
    /// The binary64 value can have any mode, and the value has the mode `M`.
    #[must_use]
    pub fn from_f64<N: Mode>(value: Float<Binary<11>, 64, N>) -> Self {
        Self::new(value.with_mode(), F64::from_bits(0))
    }

    /// Returns the high half.
    #[must_use]
    pub fn hi(self) -> Float<Binary<11>, 64, M> {
        self.hi.with_mode()
    }

    /// Returns the low half.
    #[must_use]
    pub fn lo(self) -> Float<Binary<11>, 64, M> {
        self.lo.with_mode()
    }

    fn new(hi: F64, lo: F64) -> Self {
        Self {
            hi,
            lo,
            marker: PhantomData,
        }
    }

    /// Runs the algorithm `algorithm` of an operation with one operand in
    /// `steps`.
    fn apply_one<B: Behavior>(
        self,
        mut steps: Steps<B>,
        algorithm: OneOperand<B>,
    ) -> (Self, Flags) {
        let (hi, lo) = algorithm(&mut steps, (self.hi, self.lo));
        (Self::new(hi, lo), steps.flags())
    }

    /// Runs the algorithm of `Alg` among the algorithms of an operation with
    /// one operand in `steps`.
    fn run_one<B: Behavior>(self, steps: Steps<B>, operation: [OneOperand<B>; 2]) -> (Self, Flags) {
        self.apply_one(steps, of_algorithm::<Alg, _>(operation))
    }

    /// Runs the algorithm of `Alg` among the algorithms of an operation with
    /// two operands in `steps`.
    fn run<B: Behavior>(
        self,
        other: Self,
        mut steps: Steps<B>,
        operation: [TwoOperands<B>; 2],
    ) -> (Self, Flags) {
        let algorithm = of_algorithm::<Alg, _>(operation);
        let (hi, lo) = algorithm(&mut steps, (self.hi, self.lo), (other.hi, other.lo));
        (Self::new(hi, lo), steps.flags())
    }

    /// Returns `self + other` and the flags of every step.
    #[must_use]
    pub fn add_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(
            other,
            Steps::new(behavior.apply::<M>()),
            [gcc::add, qd::add],
        )
    }

    /// Returns `self - other` and the flags of every step.
    #[must_use]
    pub fn sub_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(
            other,
            Steps::new(behavior.apply::<M>()),
            [gcc::sub, qd::sub],
        )
    }

    /// Returns `self * other` and the flags of every step.
    #[must_use]
    pub fn mul_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(
            other,
            Steps::new(behavior.apply::<M>()),
            [gcc::mul, qd::mul],
        )
    }

    /// Returns `self / other` and the flags of every step.
    #[must_use]
    pub fn div_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(
            other,
            Steps::new(behavior.apply::<M>()),
            [gcc::div, qd::div],
        )
    }
}

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// Returns the square root with the default mode. The steps take the host
    /// paths of binary64 where the build has them.
    #[must_use]
    pub fn sqrt(self) -> Self {
        self.run_one(Steps::without_flags(M::default()), [glibc::sqrt, qd::sqrt])
            .0
    }

    /// Returns the square root, and the flags of every step.
    ///
    /// [`Gcc`] follows `sqrtl` of the IBM `long double` in glibc 2.43, as
    /// the libm of the pinned powerpc64le C library compiles it: Newton's
    /// method from the binary64 square root. A negative value gives `0 / 0`,
    /// and a zero gives itself. [`Qd`] follows QD's `sqrt`: a zero gives
    /// `+0`, and a negative value gives QD's NaN in both halves.
    #[must_use]
    pub fn sqrt_with(self, behavior: impl Override) -> (Self, Flags) {
        self.run_one(Steps::new(behavior.apply::<M>()), [glibc::sqrt, qd::sqrt])
    }
}

impl<Alg: Algorithm, M: Mode> DoubleDouble<Alg, M> {
    /// Returns the remainder with the default mode.
    #[must_use]
    pub fn remainder(self, divisor: Self) -> Self {
        self.remainder_with(divisor, M::default()).0
    }

    /// Returns the remainder of the reference, and the flags of every step.
    ///
    /// [`Gcc`] follows `remainderl` of the IBM `long double` in glibc 2.43:
    /// the IEEE 754 remainder from `fmodl` by twice the divisor, as exact as
    /// the 106 bits of `fmodl`. [`Qd`] follows QD's `drem`: `a - nint(a / b) *
    /// b`, with the rounded quotient of the division, and a tie of the
    /// quotient rounded up. So a large quotient gives an approximate
    /// remainder.
    #[must_use]
    pub fn remainder_with(self, divisor: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(
            divisor,
            Steps::new(behavior.apply::<M>()),
            [glibc::remainder, qd::drem],
        )
    }

    /// Returns the truncated remainder with the default mode. The `%`
    /// operator calls it.
    #[must_use]
    pub fn truncated_remainder(self, divisor: Self) -> Self {
        self.truncated_remainder_with(divisor, M::default()).0
    }

    /// Returns the truncated remainder of the reference, and the flags of
    /// every step.
    ///
    /// [`Gcc`] follows `fmodl` of the IBM `long double` in glibc 2.43: a long
    /// division of the mantissas of the operands, as `ldbl_extract_mantissa`
    /// reads 106 bits of each. [`Qd`] follows QD's `fmod`: `a - b * aint(a /
    /// b)`, with the rounded quotient of the division.
    #[must_use]
    pub fn truncated_remainder_with(self, divisor: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(
            divisor,
            Steps::new(behavior.apply::<M>()),
            [glibc::fmod, qd::fmod],
        )
    }
}

impl<M: Mode> DoubleDouble<Gcc, M> {
    /// Returns `self * multiplier + addend`, with the default mode.
    #[must_use]
    pub fn mul_add(self, multiplier: Self, addend: Self) -> Self {
        self.mul_add_with(multiplier, addend, M::default()).0
    }

    /// Returns `self * multiplier + addend` as `fmal` of the IBM `long
    /// double` in glibc 2.43 computes it, and the flags of every step.
    ///
    /// A special high half, a zero addend, and a zero factor take
    /// `__gcc_qmul` and `__gcc_qadd` in the rounding direction of the
    /// behavior. Otherwise glibc sums the ten partial values of the halves and
    /// their products in round to nearest, as accurate as the long double
    /// arithmetic, not rounded once. Then only an overflow, an underflow, and
    /// an exact zero read the rounding direction. QD has no such function, so
    /// [`Qd`] has no `mul_add`.
    #[must_use]
    pub fn mul_add_with(
        self,
        multiplier: Self,
        addend: Self,
        behavior: impl Override,
    ) -> (Self, Flags) {
        let mut steps = Steps::new(behavior.apply::<M>());
        let (hi, lo) = glibc::mul_add(
            &mut steps,
            (self.hi, self.lo),
            (multiplier.hi, multiplier.lo),
            (addend.hi, addend.lo),
        );
        (Self::new(hi, lo), steps.flags())
    }

    /// Returns the next pair up, with the default mode.
    #[must_use]
    pub fn next_up(self) -> Self {
        self.next_up_with(M::default()).0
    }

    /// Returns the next pair up, as `nextupl` of the IBM `long double` in
    /// glibc 2.43 computes it, and the flags of every step.
    ///
    /// glibc counts 106 bits in a pair: the step is one unit in the 106th
    /// bit of the high half, added with `__gcc_qadd`. The step above a zero
    /// is `2^-1074`, and a NaN gives `x + x`. QD has no such function, so
    /// [`Qd`] has no `next_up`.
    #[must_use]
    pub fn next_up_with(self, behavior: impl Override) -> (Self, Flags) {
        self.apply_one(Steps::new(behavior.apply::<M>()), glibc::next_up)
    }

    /// Returns the next pair down, with the default mode.
    #[must_use]
    pub fn next_down(self) -> Self {
        self.next_down_with(M::default()).0
    }

    /// Returns the next pair down, as `nextdownl` of glibc 2.43 computes it:
    /// the negation of [`next_up_with`](Self::next_up_with) of the negated
    /// pair.
    #[must_use]
    pub fn next_down_with(self, behavior: impl Override) -> (Self, Flags) {
        self.apply_one(Steps::new(behavior.apply::<M>()), glibc::next_down)
    }
}

/// Returns the algorithm of `Alg` among the algorithms `[gcc, qd]` of an
/// operation.
fn of_algorithm<Alg: Algorithm, T>([gcc, qd]: [T; 2]) -> T {
    match Alg::KIND {
        AlgorithmKind::Gcc => gcc,
        AlgorithmKind::Qd => qd,
    }
}

impl<Alg: Algorithm, M: Mode> core::ops::Neg for DoubleDouble<Alg, M> {
    type Output = Self;

    /// Negates both halves, as GCC negates an `__ibm128` and QD negates a
    /// `dd_real`. The operation signals nothing.
    fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
}

/// Implements an arithmetic operator with the default mode, without flags. The
/// steps take the host paths of binary64 where the build has them.
macro_rules! operator {
    ($trait:ident, $method:ident, $gcc:path, $qd:path) => {
        impl<Alg: Algorithm, M: Mode> core::ops::$trait for DoubleDouble<Alg, M> {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                self.run(other, Steps::without_flags(M::default()), [$gcc, $qd])
                    .0
            }
        }
    };
}

operator!(Add, add, gcc::add, qd::add);
operator!(Sub, sub, gcc::sub, qd::sub);
operator!(Mul, mul, gcc::mul, qd::mul);
operator!(Div, div, gcc::div, qd::div);

/// Implements a compound assignment operator from its binary operator.
macro_rules! assign {
    ($trait:ident, $method:ident, $operator:tt) => {
        impl<Alg: Algorithm, M: Mode> core::ops::$trait for DoubleDouble<Alg, M> {
            fn $method(&mut self, other: Self) {
                *self = *self $operator other;
            }
        }
    };
}

/// The truncated remainder of the reference, as
/// [`DoubleDouble::truncated_remainder`] computes it.
impl<Alg: Algorithm, M: Mode> core::ops::Rem for DoubleDouble<Alg, M> {
    type Output = Self;

    fn rem(self, divisor: Self) -> Self {
        self.truncated_remainder(divisor)
    }
}

assign!(AddAssign, add_assign, +);
assign!(SubAssign, sub_assign, -);
assign!(MulAssign, mul_assign, *);
assign!(DivAssign, div_assign, /);
assign!(RemAssign, rem_assign, %);

#[cfg(test)]
mod tests {
    use super::{DoubleDouble, Gcc, Qd};
    use crate::env::{Env, Flags, mode};
    use crate::float::{F32, F64};

    fn gcc(hi: u64, lo: u64) -> DoubleDouble<Gcc> {
        DoubleDouble::from_parts(F64::from_bits(hi), F64::from_bits(lo))
    }

    fn qd(hi: u64, lo: u64) -> DoubleDouble<Qd> {
        DoubleDouble::from_parts(F64::from_bits(hi), F64::from_bits(lo))
    }

    fn bits<Alg: super::Algorithm>(value: DoubleDouble<Alg>) -> (u64, u64) {
        (value.hi().to_bits(), value.lo().to_bits())
    }

    const ONE: u64 = 0x3FF0_0000_0000_0000;
    const THREE: u64 = 0x4008_0000_0000_0000;

    #[test]
    fn the_halves_can_have_any_mode() {
        let (hi, lo) = (F64::from_bits(ONE), F64::from_bits(0x3C30_0000_0000_0000));
        let pair = DoubleDouble::<Qd, mode::X87>::from_parts(hi, lo);
        assert_eq!(
            (pair.hi().to_bits(), pair.lo().to_bits()),
            (ONE, lo.to_bits())
        );
        let single = DoubleDouble::<Gcc, mode::X86Sse>::from_f64(hi);
        assert_eq!((single.hi().to_bits(), single.lo().to_bits()), (ONE, 0));
    }

    #[test]
    fn a_small_addend_lands_in_the_low_half() {
        // 1 + 2^-60 needs the low half.
        let tiny = 0x3C30_0000_0000_0000;
        assert_eq!(bits(gcc(ONE, 0) + gcc(tiny, 0)), (ONE, tiny));
        assert_eq!(bits(qd(ONE, 0) + qd(tiny, 0)), (ONE, tiny));
        assert_eq!(bits(gcc(ONE, 0) - gcc(ONE, 0)), (0, 0));
    }

    #[test]
    fn a_third_has_a_low_half() {
        let expected = (0x3FD5_5555_5555_5555, 0x3C75_5555_5555_5555);
        assert_eq!(bits(gcc(ONE, 0) / gcc(THREE, 0)), expected);
        assert_eq!(bits(qd(ONE, 0) / qd(THREE, 0)), expected);
        let (_, flags) = gcc(ONE, 0).div_with(gcc(THREE, 0), Env::IEEE);
        // Some step rounded up; the flags are the union of the steps.
        assert_eq!(flags, Flags::INEXACT | Flags::ROUNDED_UP);
        let product = gcc(0x3FD5_5555_5555_5555, 0x3C75_5555_5555_5555) * gcc(THREE, 0);
        assert_eq!(product.hi().to_bits(), ONE);
    }

    #[test]
    fn a_zero_result_keeps_the_sign_of_the_high_part() {
        let negative_zero = 0x8000_0000_0000_0000;
        assert_eq!(
            bits(gcc(negative_zero, 0) * gcc(ONE, 0)),
            (negative_zero, 0)
        );
        assert_eq!(bits(qd(0, 0).sqrt()), (0, 0));
        assert_eq!(bits(qd(negative_zero, 0).sqrt()), (0, 0));
        let nan = 0x7FF8_0000_0000_0000;
        assert_eq!(bits(qd(0xBFF0_0000_0000_0000, 0).sqrt()), (nan, nan));
    }

    #[test]
    fn saturation_leaves_the_arithmetic_unchanged() {
        // Each operation overflows in a step. `Gcc` gives an infinity, and `Qd`
        // gives a NaN in both halves.
        let (max, min) = (0x7FEF_FFFF_FFFF_FFFF, 0x0010_0000_0000_0000);
        let saturating = Env::IEEE.with_saturate(true);
        let outcome = |(value, flags): (DoubleDouble<Gcc>, Flags)| (bits(value), flags);
        let (x, y) = (gcc(max, 0), gcc(min, 0));
        let overflow = Flags::OVERFLOW | Flags::INEXACT | Flags::ROUNDED_UP;
        for result in [
            x.add_with(x, saturating),
            x.sub_with(-x, saturating),
            x.mul_with(x, saturating),
            x.div_with(y, saturating),
        ] {
            assert_eq!(outcome(result), ((0x7FF0_0000_0000_0000, 0), overflow));
        }
        let outcome = |(value, flags): (DoubleDouble<Qd>, Flags)| (bits(value), flags);
        let (x, y) = (qd(max, 0), qd(min, 0));
        let nan = 0x7FF8_0000_0000_0000;
        for result in [
            x.add_with(x, saturating),
            x.sub_with(-x, saturating),
            x.mul_with(x, saturating),
            x.div_with(y, saturating),
        ] {
            assert_eq!(outcome(result), ((nan, nan), overflow | Flags::INVALID));
        }
        // A conversion saturates as its destination format does.
        let (value, flags) = x.convert_with::<F32>(saturating);
        assert_eq!(
            (value.to_bits(), flags),
            (0x7F7F_FFFF, Flags::OVERFLOW | Flags::INEXACT)
        );
    }

    #[test]
    fn compound_assignment_follows_the_operators() {
        let (x, y) = (gcc(ONE, 0), gcc(THREE, 0));
        let mut value = x;
        value += y;
        assert_eq!(bits(value), bits(x + y));
        value -= y;
        assert_eq!(bits(value), bits(x + y - y));
        value *= y;
        assert_eq!(bits(value), bits((x + y - y) * y));
        value /= y;
        assert_eq!(bits(value), bits((x + y - y) * y / y));
        // 7 % 3 truncates the quotient to 2, for both references.
        let seven = 0x401C_0000_0000_0000;
        assert_eq!(bits(gcc(seven, 0) % y), (0x3FF0_0000_0000_0000, 0));
        assert_eq!(
            bits(qd(seven, 0) % qd(THREE, 0)),
            (0x3FF0_0000_0000_0000, 0)
        );
        let mut value = gcc(seven, 0);
        value %= y;
        assert_eq!(bits(value), bits(gcc(seven, 0).truncated_remainder(y)));
    }
}
