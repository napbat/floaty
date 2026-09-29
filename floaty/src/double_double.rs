//! Double-double values: the sum of two binary64 values, `hi + lo`, with the
//! arithmetic of one reference implementation.
//!
//! No standard defines the correct result of a double-double operation, and
//! the published algorithms give different low halves. So each algorithm
//! follows one reference: [`Gcc`] the IBM `long double` of libgcc on PowerPC,
//! and [`Qd`] the `dd_real` of the QD library. Each algorithm follows the
//! machine code that a pinned compiler makes of its reference, because the
//! operand order of each step decides which NaN the step returns. Under the
//! behavior of the platform of the reference, the results match it bit for
//! bit, NaN halves included.
//!
//! Each step of an algorithm is a binary64 operation that runs under the
//! behavior of the call, and the result has the flags of every step, as a
//! processor that runs the reference sets them. `TINY`, `ROUNDED_UP`, and
//! `DENORMAL_INPUT` then say that some step was tiny, rounded up, or read a
//! subnormal operand.

mod gcc;
mod qd;
mod steps;

use core::cmp::Ordering;
use core::fmt;
use core::marker::PhantomData;

use self::steps::Steps;
use crate::env::{Env, Flags, Mode, Override, mode};
use crate::float::{Decoded, F64, Float, FloatType};
use crate::format::internal::Source;
use crate::format::{Binary, Standard};
use crate::limbs::Limbs;
use crate::sealed::Sealed;
use crate::unpacked::Unpacked;

/// The high half and the low half of a value.
type Pair = (F64, F64);

/// An algorithm with two operands, on binary64 steps under the behavior `B`.
type TwoOperands<B> = fn(&mut Steps<B>, Pair, Pair) -> Pair;

/// `|hi + lo| * 2^1074`. Every finite pair is a multiple of 2^-1074 below
/// 2^1025, so 2,099 bits hold the exact value of every finite pair.
type Magnitude = [u64; 33];

/// The weight of the lowest bit of a [`Magnitude`].
const LOWEST: i32 = -1074;

/// The algorithm of a double-double type: [`Gcc`] or [`Qd`]. The trait is
/// sealed.
pub trait Algorithm: Sealed + 'static {
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
/// powerpc64le.
///
/// With the behavior of PowerPC, the results and the five IEEE flags match
/// libgcc, NaN halves included. That behavior has the `FirstOperand` NaN rule,
/// a positive default NaN, the `AddendSecond` fused NaN order, the
/// `SignalsAndYieldsToNan` invalid product rule, and tininess before rounding.
/// The default mode has other NaN rules and detects tininess after rounding.
/// So pass the behavior of PowerPC to match the NaNs and the underflow flag of
/// libgcc.
///
/// ```
/// use floaty::env::{FusedNanOrder, InvalidProduct, NanPropagation, NanRule, Tininess};
/// use floaty::{DoubleDouble, Env, F64, Gcc};
///
/// let powerpc = Env::IEEE
///     .with_nan(
///         NanRule::new(NanPropagation::FirstOperand)
///             .with_fused_order(FusedNanOrder::AddendSecond)
///             .with_invalid_product(InvalidProduct::SignalsAndYieldsToNan),
///     )
///     .with_tininess(Tininess::BeforeRounding);
/// let one = DoubleDouble::<Gcc>::from_f64(F64::from_bits(0x3FF0_0000_0000_0000));
/// let (two, flags) = one.add_with(one, powerpc);
/// assert_eq!(two.hi().to_bits(), 0x4000_0000_0000_0000);
/// assert!(flags.is_empty());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gcc {}

/// The `dd_real` of QD 2.3.24, configured with IEEE-style addition, accurate
/// division, and a fused multiply-subtract in `two_prod`.
///
/// With [`Env::X86_SSE`] the results and the five IEEE flags match QD on
/// x86-64, NaN halves included. The algorithm follows the machine code that g++ 15.2.0
/// makes of QD with `-O2 -ffp-contract=off`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Qd {}

impl Sealed for Gcc {}
impl Algorithm for Gcc {
    const KIND: AlgorithmKind = AlgorithmKind::Gcc;
}

impl Sealed for Qd {}
impl Algorithm for Qd {
    const KIND: AlgorithmKind = AlgorithmKind::Qd;
}

/// A double-double value: the sum of two binary64 values, with the arithmetic
/// of the algorithm `Alg` and the default mode `M`.
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
pub struct DoubleDouble<Alg, M: Mode = mode::Ieee> {
    hi: F64,
    lo: F64,
    marker: PhantomData<(Alg, M)>,
}

impl<Alg, M: Mode> Clone for DoubleDouble<Alg, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Alg, M: Mode> Copy for DoubleDouble<Alg, M> {}

impl<Alg, M: Mode> fmt::Debug for DoubleDouble<Alg, M> {
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

    /// Makes a value from its halves. The pair is kept as it is.
    #[must_use]
    pub fn from_parts(hi: Float<Binary<11>, 64, M>, lo: Float<Binary<11>, 64, M>) -> Self {
        Self::new(hi.with_mode(), lo.with_mode())
    }

    /// Makes the value of a binary64 value: the value and a positive zero.
    #[must_use]
    pub fn from_f64(value: Float<Binary<11>, 64, M>) -> Self {
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

    /// Runs the algorithm of an operation with two operands.
    fn run<O: Override>(
        self,
        other: Self,
        behavior: O,
        operation: [TwoOperands<O::Behavior>; 2],
    ) -> (Self, Flags) {
        let mut steps = Steps::new(behavior.apply::<M>());
        let [gcc, qd] = operation;
        let algorithm = match Alg::KIND {
            AlgorithmKind::Gcc => gcc,
            AlgorithmKind::Qd => qd,
        };
        let (hi, lo) = algorithm(&mut steps, (self.hi, self.lo), (other.hi, other.lo));
        (Self::new(hi, lo), steps.flags())
    }

    /// Returns `self + other` and the flags of every step.
    #[must_use]
    pub fn add_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(other, behavior, [gcc::add, qd::add])
    }

    /// Returns `self - other` and the flags of every step.
    #[must_use]
    pub fn sub_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(other, behavior, [gcc::sub, qd::sub])
    }

    /// Returns `self * other` and the flags of every step.
    #[must_use]
    pub fn mul_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(other, behavior, [gcc::mul, qd::mul])
    }

    /// Returns `self / other` and the flags of every step.
    #[must_use]
    pub fn div_with(self, other: Self, behavior: impl Override) -> (Self, Flags) {
        self.run(other, behavior, [gcc::div, qd::div])
    }

    /// Returns the value with both halves negated when the exact value has a
    /// negative sign: a negative number, `-0`, a negative infinity, or a NaN
    /// with a negative sign. The operation signals nothing.
    #[must_use]
    pub fn abs(self) -> Self {
        let negative = match self.exact() {
            Unpacked::Zero { negative, .. }
            | Unpacked::Finite { negative, .. }
            | Unpacked::Infinity { negative }
            | Unpacked::Nan { negative, .. } => negative,
            Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
        };
        if negative { -self } else { self }
    }

    /// Returns the exact value `hi + lo`.
    ///
    /// A NaN or an infinite high half gives that value. With a finite high
    /// half, a NaN or an infinite low half gives that value. Otherwise a
    /// finite value has an odd significand, and a zero takes the sign of the
    /// high half. 33 limbs hold every significand.
    #[must_use]
    pub fn decode(self) -> Decoded<33> {
        match self.exact() {
            Unpacked::Zero { negative, .. } => Decoded::Zero {
                negative,
                exponent: 0,
            },
            Unpacked::Finite {
                negative,
                significand,
                ..
            } => {
                let (zeros, exponent) = lowest_bit(&significand);
                Decoded::Finite {
                    negative,
                    exponent,
                    significand: significand.shr(zeros),
                }
            }
            Unpacked::Infinity { negative } => Decoded::Infinity { negative },
            Unpacked::Nan {
                negative,
                signaling,
                payload,
            } => Decoded::Nan {
                negative,
                signaling,
                payload,
            },
            Unpacked::Unsupported => unreachable!("a binary64 half has no unsupported encoding"),
        }
    }

    /// Converts the exact value to another format, rounded once with the
    /// default mode of that format.
    #[must_use]
    pub fn convert<T: FloatType>(self) -> T {
        self.convert_with(T::Mode::default()).0
    }

    /// Converts the exact value to another format, rounded once, with an
    /// override of the destination behavior. Returns the result and the
    /// flags. A NaN keeps the high-order bits of its payload.
    #[must_use]
    pub fn convert_with<T: FloatType>(self, behavior: impl Override) -> (T, Flags) {
        let source = Source {
            radix: 2,
            payload_digits: <Binary<11> as Standard<64>>::PAYLOAD_DIGITS,
        };
        T::convert_from(self.exact(), source, behavior.apply::<T::Mode>())
    }

    /// Compares the exact values as the IEEE 754 quiet predicates do. `None`
    /// means unordered. A signaling NaN signals invalid.
    #[must_use]
    pub fn compare_quiet(self, other: Self) -> (Option<Ordering>, Flags) {
        let (order, signaling) = self.compare(other);
        let flags = if signaling {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        (order, flags)
    }

    /// Compares the exact values as the IEEE 754 signaling predicates do.
    /// `None` means unordered. Every NaN signals invalid.
    #[must_use]
    pub fn compare_signaling(self, other: Self) -> (Option<Ordering>, Flags) {
        let (order, _) = self.compare(other);
        let flags = if order.is_none() {
            Flags::INVALID
        } else {
            Flags::NONE
        };
        (order, flags)
    }

    /// Returns the order of the exact values, and `true` when a value is a
    /// signaling NaN.
    fn compare(self, other: Self) -> (Option<Ordering>, bool) {
        let (first, second) = (self.exact(), other.exact());
        if first.is_nan() || second.is_nan() {
            return (None, first.is_signaling() || second.is_signaling());
        }
        let (first_sign, first_magnitude) = signed(&first);
        let (second_sign, second_magnitude) = signed(&second);
        let order = first_sign.cmp(&second_sign).then_with(|| match first_sign {
            Ordering::Greater => first_magnitude.compare(&second_magnitude),
            Ordering::Less => second_magnitude.compare(&first_magnitude),
            Ordering::Equal => Ordering::Equal,
        });
        (Some(order), false)
    }

    /// Returns the exact value of the pair, as `DESIGN.md` defines it.
    fn exact(self) -> Unpacked<Magnitude> {
        let (hi, lo) = (self.hi.decode::<1>(), self.lo.decode::<1>());
        match (hi, lo) {
            (Decoded::Nan { .. } | Decoded::Infinity { .. }, _) => special(hi),
            (_, Decoded::Nan { .. } | Decoded::Infinity { .. }) => special(lo),
            _ => sum(hi, lo),
        }
    }
}

impl<M: Mode> DoubleDouble<Qd, M> {
    /// Returns the square root with the default mode.
    #[must_use]
    pub fn sqrt(self) -> Self {
        self.sqrt_with(M::default()).0
    }

    /// Returns the square root by QD's `sqrt`, and the flags of every step.
    /// A zero gives `+0`, and a negative value gives QD's NaN in both halves,
    /// as QD does.
    #[must_use]
    pub fn sqrt_with(self, behavior: impl Override) -> (Self, Flags) {
        let mut steps = Steps::new(behavior.apply::<M>());
        let (hi, lo) = qd::sqrt(&mut steps, (self.hi, self.lo));
        (Self::new(hi, lo), steps.flags())
    }
}

/// Returns the NaN or the infinity of a half as an exact value.
fn special(half: Decoded<1>) -> Unpacked<Magnitude> {
    match half {
        Decoded::Nan {
            negative,
            signaling,
            payload,
        } => Unpacked::Nan {
            negative,
            signaling,
            payload: payload.resize(),
        },
        Decoded::Infinity { negative } => Unpacked::Infinity { negative },
        _ => unreachable!("the caller passes a NaN or an infinity"),
    }
}

/// Returns the sign and `|half| * 2^1074` of a zero or finite half.
fn scaled(half: Decoded<1>) -> (bool, Magnitude) {
    match half {
        Decoded::Zero { negative, .. } => (negative, Magnitude::ZERO),
        Decoded::Finite {
            negative,
            exponent,
            significand,
        } => {
            let shift = u32::try_from(exponent - LOWEST)
                .expect("a binary64 exponent is at or above the lowest weight");
            (negative, significand.resize::<Magnitude>().shl(shift))
        }
        _ => unreachable!("the caller passes a zero or a finite half"),
    }
}

/// Returns the exact sum of a zero or finite high half and a zero or finite
/// low half. A zero sum takes the sign of the high half.
fn sum(hi: Decoded<1>, lo: Decoded<1>) -> Unpacked<Magnitude> {
    let ((hi_negative, hi_magnitude), (lo_negative, lo_magnitude)) = (scaled(hi), scaled(lo));
    let (negative, magnitude) = if hi_negative == lo_negative {
        (hi_negative, hi_magnitude.add(lo_magnitude))
    } else {
        match hi_magnitude.compare(&lo_magnitude) {
            Ordering::Greater => (hi_negative, hi_magnitude.sub(lo_magnitude)),
            Ordering::Less => (lo_negative, lo_magnitude.sub(hi_magnitude)),
            Ordering::Equal => {
                return Unpacked::Zero {
                    negative: hi_negative,
                    exponent: 0,
                };
            }
        }
    };
    if magnitude.is_zero() {
        return Unpacked::Zero {
            negative: hi_negative,
            exponent: 0,
        };
    }
    Unpacked::Finite {
        negative,
        exponent: LOWEST,
        significand: magnitude,
    }
}

/// Returns the sign and the magnitude of a number that orders it: an infinity
/// has a magnitude above every finite magnitude, and a zero has no sign.
fn signed(value: &Unpacked<Magnitude>) -> (Ordering, Magnitude) {
    let sign = |negative: bool| {
        if negative {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    };
    match *value {
        Unpacked::Zero { .. } => (Ordering::Equal, Magnitude::ZERO),
        Unpacked::Finite {
            negative,
            significand,
            ..
        } => (sign(negative), significand),
        Unpacked::Infinity { negative } => (sign(negative), Magnitude::ones(Magnitude::BITS)),
        Unpacked::Nan { .. } | Unpacked::Unsupported => {
            unreachable!("the caller orders only numbers")
        }
    }
}

/// Returns the number of zero bits below the lowest set bit of a nonzero
/// magnitude, and the weight of that bit.
fn lowest_bit(magnitude: &Magnitude) -> (u32, i32) {
    let (index, limb) = magnitude
        .iter()
        .enumerate()
        .find(|(_, limb)| **limb != 0)
        .expect("the magnitude is not zero");
    let zeros = u32::try_from(index).expect("a limb index fits a u32") * 64 + limb.trailing_zeros();
    let weight = LOWEST + i32::try_from(zeros).expect("a bit index of 33 limbs fits an i32");
    (zeros, weight)
}

impl<Alg: Algorithm, M: Mode> core::ops::Neg for DoubleDouble<Alg, M> {
    type Output = Self;

    /// Negates both halves, as GCC negates an `__ibm128` and QD negates a
    /// `dd_real`. The operation signals nothing.
    fn neg(self) -> Self {
        Self::new(-self.hi, -self.lo)
    }
}

/// Implements an arithmetic operator with the default mode, and drops the
/// flags.
macro_rules! operator {
    ($trait:ident, $method:ident, $with:ident) => {
        impl<Alg: Algorithm, M: Mode> core::ops::$trait for DoubleDouble<Alg, M> {
            type Output = Self;

            fn $method(self, other: Self) -> Self {
                self.$with(other, M::default()).0
            }
        }
    };
}

operator!(Add, add, add_with);
operator!(Sub, sub, sub_with);
operator!(Mul, mul, mul_with);
operator!(Div, div, div_with);

impl<Alg: Algorithm, M: Mode> PartialEq for DoubleDouble<Alg, M> {
    /// Compares the exact values as the quiet equality predicate does.
    fn eq(&self, other: &Self) -> bool {
        self.compare_quiet(*other).0 == Some(Ordering::Equal)
    }
}

impl<Alg: Algorithm, M: Mode> PartialOrd for DoubleDouble<Alg, M> {
    /// Orders the exact values as the quiet predicates do.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.compare_quiet(*other).0
    }
}

#[cfg(test)]
mod tests;
