//! The exponentials and the logarithms of IEEE 754-2019 section 9.2, in base
//! e, 2, and 10, with their forms shifted by one, `b^x - 1` and
//! `log_b(1 + x)`, the hyperbolic functions and their inverses, the
//! trigonometric functions of an argument scaled by pi, and the inverse
//! trigonometric functions with their forms scaled by pi.

use super::Float;
use crate::elementary::Transcendental;
use crate::env::{Flags, Mode, Override};
use crate::format::Standard;

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    /// Returns `function` of `self` and the flags.
    fn elementary_with(self, function: Transcendental, behavior: impl Override) -> (Self, Flags) {
        let (bits, flags) = S::elementary(self.bits, function, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }

    /// Returns `e^self`, with the default mode.
    #[must_use]
    pub fn exp(self) -> Self {
        self.exp_with(M::default()).0
    }

    /// Returns `e^self`, correctly rounded in the direction of the behavior,
    /// as IEEE 754-2019 `exp` does, and the flags.
    ///
    /// Every finite result but `e^0` is inexact. The result rounds once, so
    /// it has every flag of a rounding: overflow, underflow, `TINY`, and
    /// `ROUNDED_UP`. The special cases follow IEEE 754-2019 section 9.2.1:
    /// `e^±0` is 1 and exact, `e^+inf` is +inf, and `e^-inf` is +0. A NaN
    /// gives the NaN of the NaN rule. An exact decimal result takes the
    /// exponent nearest 0, and an inexact decimal result keeps every digit.
    ///
    /// The value evaluates in ball arithmetic, which bounds every error, at
    /// twice and then four times the bits of the storage. No input is known
    /// that needs more. For binary64 the second precision has 256 bits, and
    /// the published hardest cases need fewer than 160.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (e, flags) = F64::from_bits(0x3FF0_0000_0000_0000).exp_with(Env::IEEE);
    /// assert_eq!((e.to_bits(), flags), (0x4005_BF0A_8B14_5769, Flags::INEXACT));
    /// ```
    #[must_use]
    pub fn exp_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Exp, behavior)
    }

    /// Returns `e^self - 1`, with the default mode.
    #[must_use]
    pub fn exp_m1(self) -> Self {
        self.exp_m1_with(M::default()).0
    }

    /// Returns `e^self - 1`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `expm1` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does, also
    /// for a `self` near 0, where `e^self` is near 1. Every finite result but
    /// that of a zero is inexact. The special cases follow IEEE 754-2019
    /// section 9.2.1: ±0 gives ±0, +inf gives +inf, and -inf gives -1.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags, Rounding};
    ///
    /// // e^(2^-60) - 1 is 2^-60 + 2^-121 and a little more.
    /// let x = F64::from_bits(0x3C30_0000_0000_0000);
    /// assert_eq!(x.exp_m1_with(Env::IEEE), (x, Flags::INEXACT));
    /// let (up, _) = x.exp_m1_with(Rounding::TowardPositive);
    /// assert_eq!(up.to_bits(), 0x3C30_0000_0000_0001);
    /// ```
    #[must_use]
    pub fn exp_m1_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::ExpM1, behavior)
    }

    /// Returns `2^self`, with the default mode.
    #[must_use]
    pub fn exp2(self) -> Self {
        self.exp2_with(M::default()).0
    }

    /// Returns `2^self`, correctly rounded in the direction of the behavior,
    /// as IEEE 754-2019 `exp2` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. An
    /// integer `self` gives the exact value `2^self`, which signals inexact
    /// only where the format cannot hold it. Every other finite result is
    /// inexact. The special cases are those of `exp`: ±0 gives 1, +inf gives
    /// +inf, and -inf gives +0.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (eight, flags) = F64::from_bits(0x4008_0000_0000_0000).exp2_with(Env::IEEE);
    /// assert_eq!((eight.to_bits(), flags), (0x4020_0000_0000_0000, Flags::NONE));
    /// ```
    #[must_use]
    pub fn exp2_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Exp2, behavior)
    }

    /// Returns `2^self - 1`, with the default mode.
    #[must_use]
    pub fn exp2_m1(self) -> Self {
        self.exp2_m1_with(M::default()).0
    }

    /// Returns `2^self - 1`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `exp2m1` does, and the flags.
    ///
    /// The result rounds once, as [`exp_m1_with`](Self::exp_m1_with) does.
    /// An integer `self` gives the exact value `2^self - 1`, which signals
    /// inexact only where the format cannot hold it. Every other finite
    /// result is inexact. The special cases are those of `expm1`: ±0 gives
    /// ±0, +inf gives +inf, and -inf gives -1.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (half, flags) = F64::from_bits(0xBFF0_0000_0000_0000).exp2_m1_with(Env::IEEE);
    /// assert_eq!((half.to_bits(), flags), (0xBFE0_0000_0000_0000, Flags::NONE));
    /// ```
    #[must_use]
    pub fn exp2_m1_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Exp2M1, behavior)
    }

    /// Returns `10^self`, with the default mode.
    #[must_use]
    pub fn exp10(self) -> Self {
        self.exp10_with(M::default()).0
    }

    /// Returns `10^self`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `exp10` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. An
    /// integer `self` gives the exact value `10^self`, which signals inexact
    /// only where the format cannot hold it. A binary format holds no
    /// negative power of 10. Every other finite result is inexact. The
    /// special cases are those of `exp`: ±0 gives 1, +inf gives +inf, and
    /// -inf gives +0.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let exp10 = |bits| F64::from_bits(bits).exp10_with(Env::IEEE);
    /// assert_eq!(exp10(0x4000_0000_0000_0000), (F64::from_bits(0x4059_0000_0000_0000), Flags::NONE));
    /// assert_eq!(exp10(0xBFF0_0000_0000_0000).0.to_bits(), 0x3FB9_9999_9999_999A);
    /// ```
    #[must_use]
    pub fn exp10_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Exp10, behavior)
    }

    /// Returns `10^self - 1`, with the default mode.
    #[must_use]
    pub fn exp10_m1(self) -> Self {
        self.exp10_m1_with(M::default()).0
    }

    /// Returns `10^self - 1`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `exp10m1` does, and the flags.
    ///
    /// The result rounds once, as [`exp_m1_with`](Self::exp_m1_with) does.
    /// An integer `self` gives the exact value `10^self - 1`, which signals
    /// inexact only where the format cannot hold it. Every other finite
    /// result is inexact. The special cases are those of `expm1`: ±0 gives
    /// ±0, +inf gives +inf, and -inf gives -1.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (nines, flags) = F64::from_bits(0x4000_0000_0000_0000).exp10_m1_with(Env::IEEE);
    /// assert_eq!((nines.to_bits(), flags), (0x4058_C000_0000_0000, Flags::NONE));
    /// ```
    #[must_use]
    pub fn exp10_m1_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Exp10M1, behavior)
    }

    /// Returns `ln self`, with the default mode.
    #[must_use]
    pub fn log(self) -> Self {
        self.log_with(M::default()).0
    }

    /// Returns the natural logarithm `ln self`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `log` does, and the
    /// flags.
    ///
    /// Every finite result but `ln 1` is inexact, and rounds once as
    /// [`exp_with`](Self::exp_with) does. The special cases follow IEEE
    /// 754-2019 section 9.2.1: `ln 1` is +0 and exact, `ln ±0` is -inf and
    /// signals divide-by-zero, `ln +inf` is +inf, and a negative value or
    /// -inf gives the default NaN and signals invalid. A format without an
    /// infinity gives its NaN or its largest finite value for -inf. A NaN
    /// gives the NaN of the NaN rule. A decimal 0 has the exponent 0.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (ln2, flags) = F64::from_bits(0x4000_0000_0000_0000).log_with(Env::IEEE);
    /// assert_eq!((ln2.to_bits(), flags), (0x3FE6_2E42_FEFA_39EF, Flags::INEXACT));
    /// ```
    #[must_use]
    pub fn log_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Log, behavior)
    }

    /// Returns `log_2 self`, with the default mode.
    #[must_use]
    pub fn log2(self) -> Self {
        self.log2_with(M::default()).0
    }

    /// Returns the base-2 logarithm `log_2 self`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `log2` does, and the
    /// flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. A power
    /// of two `2^k` gives the exact integer `k`, which signals inexact only
    /// where the format cannot hold it. Every other finite result is
    /// inexact. The special cases are those of [`log_with`](Self::log_with):
    /// 1 gives +0, ±0 gives -inf and signals divide-by-zero, +inf gives +inf,
    /// and a negative value or -inf gives the default NaN and signals
    /// invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let log2 = |bits| F64::from_bits(bits).log2_with(Env::IEEE);
    /// assert_eq!(log2(0x4020_0000_0000_0000), (F64::from_bits(0x4008_0000_0000_0000), Flags::NONE));
    /// assert_eq!(log2(0x4024_0000_0000_0000).0.to_bits(), 0x400A_934F_0979_A371);
    /// ```
    #[must_use]
    pub fn log2_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Log2, behavior)
    }

    /// Returns `log_10 self`, with the default mode.
    #[must_use]
    pub fn log10(self) -> Self {
        self.log10_with(M::default()).0
    }

    /// Returns the base-10 logarithm `log_10 self`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `log10` does, and the
    /// flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. A power
    /// of ten `10^k` gives the exact integer `k`, which signals inexact only
    /// where the format cannot hold it. Every other finite result is
    /// inexact. The special cases are those of [`log_with`](Self::log_with).
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (three, flags) = F64::from_bits(0x408F_4000_0000_0000).log10_with(Env::IEEE);
    /// assert_eq!((three.to_bits(), flags), (0x4008_0000_0000_0000, Flags::NONE));
    /// ```
    #[must_use]
    pub fn log10_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Log10, behavior)
    }

    /// Returns `ln(1 + self)`, with the default mode.
    #[must_use]
    pub fn log_p1(self) -> Self {
        self.log_p1_with(M::default()).0
    }

    /// Returns `ln(1 + self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `logp1` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does, also
    /// for a `self` near 0, where `1 + self` is near 1. Every finite result
    /// but that of a zero is inexact. The special cases follow IEEE 754-2019
    /// section 9.2.1: ±0 gives ±0, -1 gives -inf and signals divide-by-zero,
    /// +inf gives +inf, and a value below -1 or -inf gives the default NaN
    /// and signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (ln2, flags) = F64::from_bits(0x3FF0_0000_0000_0000).log_p1_with(Env::IEEE);
    /// assert_eq!((ln2.to_bits(), flags), (0x3FE6_2E42_FEFA_39EF, Flags::INEXACT));
    /// ```
    #[must_use]
    pub fn log_p1_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::LogP1, behavior)
    }

    /// Returns `log_2(1 + self)`, with the default mode.
    #[must_use]
    pub fn log2_p1(self) -> Self {
        self.log2_p1_with(M::default()).0
    }

    /// Returns `log_2(1 + self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `log2p1` does, and the flags.
    ///
    /// The result rounds once, as [`log_p1_with`](Self::log_p1_with) does.
    /// A `self` whose `1 + self` is a power of two `2^k` gives the exact
    /// integer `k`, which signals inexact only where the format cannot hold
    /// it. Every other finite result is inexact. The special cases are those
    /// of `logp1`.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (two, flags) = F64::from_bits(0x4008_0000_0000_0000).log2_p1_with(Env::IEEE);
    /// assert_eq!((two.to_bits(), flags), (0x4000_0000_0000_0000, Flags::NONE));
    /// ```
    #[must_use]
    pub fn log2_p1_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Log2P1, behavior)
    }

    /// Returns `log_10(1 + self)`, with the default mode.
    #[must_use]
    pub fn log10_p1(self) -> Self {
        self.log10_p1_with(M::default()).0
    }

    /// Returns `log_10(1 + self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `log10p1` does, and the flags.
    ///
    /// The result rounds once, as [`log_p1_with`](Self::log_p1_with) does.
    /// A `self` whose `1 + self` is a power of ten `10^k` gives the exact
    /// integer `k`, which signals inexact only where the format cannot hold
    /// it. Every other finite result is inexact. The special cases are those
    /// of `logp1`.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (two, flags) = F64::from_bits(0x4058_C000_0000_0000).log10_p1_with(Env::IEEE);
    /// assert_eq!((two.to_bits(), flags), (0x4000_0000_0000_0000, Flags::NONE));
    /// ```
    #[must_use]
    pub fn log10_p1_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Log10P1, behavior)
    }

    /// Returns `sinh self`, with the default mode.
    #[must_use]
    pub fn sinh(self) -> Self {
        self.sinh_with(M::default()).0
    }

    /// Returns the hyperbolic sine `sinh self`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `sinh` does, and the
    /// flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does, also
    /// for a `self` near 0. Every finite result but that of a zero is
    /// inexact. The special cases follow IEEE 754-2019 section 9.2.1: ±0
    /// gives ±0, and ±inf gives ±inf.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags, Rounding};
    ///
    /// // sinh(2^-60) lies above 2^-60 by about 2^-183.
    /// let x = F64::from_bits(0x3C30_0000_0000_0000);
    /// assert_eq!(x.sinh_with(Env::IEEE), (x, Flags::INEXACT));
    /// assert_eq!(x.sinh_with(Rounding::TowardPositive).0, x.next_up());
    /// ```
    #[must_use]
    pub fn sinh_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Sinh, behavior)
    }

    /// Returns `cosh self`, with the default mode.
    #[must_use]
    pub fn cosh(self) -> Self {
        self.cosh_with(M::default()).0
    }

    /// Returns the hyperbolic cosine `cosh self`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `cosh` does, and the
    /// flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. Every
    /// finite result but that of a zero is inexact. The special cases follow
    /// IEEE 754-2019 section 9.2.1: ±0 gives 1, and ±inf gives +inf.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let one = F64::from_bits(0x3FF0_0000_0000_0000);
    /// assert_eq!(F64::from_bits(0x8000_0000_0000_0000).cosh_with(Env::IEEE), (one, Flags::NONE));
    /// ```
    #[must_use]
    pub fn cosh_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Cosh, behavior)
    }

    /// Returns `tanh self`, with the default mode.
    #[must_use]
    pub fn tanh(self) -> Self {
        self.tanh_with(M::default()).0
    }

    /// Returns the hyperbolic tangent `tanh self`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `tanh` does, and the
    /// flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does, also
    /// for a `self` near 0. Every finite result but that of a zero is
    /// inexact. The special cases follow IEEE 754-2019 section 9.2.1: ±0
    /// gives ±0, and ±inf gives ±1.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// // tanh(20) lies below 1 by less than 2^-56, so it rounds up to 1.
    /// let (one, flags) = F64::from_bits(0x4034_0000_0000_0000).tanh_with(Env::IEEE);
    /// let rounded_up = Flags::INEXACT | Flags::ROUNDED_UP;
    /// assert_eq!((one.to_bits(), flags), (0x3FF0_0000_0000_0000, rounded_up));
    /// ```
    #[must_use]
    pub fn tanh_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Tanh, behavior)
    }

    /// Returns `asinh self`, with the default mode.
    #[must_use]
    pub fn asinh(self) -> Self {
        self.asinh_with(M::default()).0
    }

    /// Returns the inverse hyperbolic sine `asinh self`, correctly rounded
    /// in the direction of the behavior, as IEEE 754-2019 `asinh` does, and
    /// the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does, also
    /// for a `self` near 0. Every finite result but that of a zero is
    /// inexact. The special cases follow IEEE 754-2019 section 9.2.1: ±0
    /// gives ±0, and ±inf gives ±inf.
    ///
    /// ```
    /// use floaty::{F64, Rounding};
    ///
    /// // asinh(2^-60) lies below 2^-60 by about 2^-183.
    /// let x = F64::from_bits(0x3C30_0000_0000_0000);
    /// assert_eq!(x.asinh_with(Rounding::TowardZero).0, x.next_down());
    /// ```
    #[must_use]
    pub fn asinh_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Asinh, behavior)
    }

    /// Returns `acosh self`, with the default mode.
    #[must_use]
    pub fn acosh(self) -> Self {
        self.acosh_with(M::default()).0
    }

    /// Returns the inverse hyperbolic cosine `acosh self`, correctly rounded
    /// in the direction of the behavior, as IEEE 754-2019 `acosh` does, and
    /// the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. Every
    /// finite result but `acosh 1` is inexact. The special cases follow IEEE
    /// 754-2019 section 9.2.1: 1 gives +0, +inf gives +inf, and a value
    /// below 1 or -inf gives the default NaN and signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (zero, flags) = F64::from_bits(0x3FF0_0000_0000_0000).acosh_with(Env::IEEE);
    /// assert_eq!((zero.to_bits(), flags), (0, Flags::NONE));
    /// let (nan, flags) = F64::from_bits(0x3FE0_0000_0000_0000).acosh_with(Env::IEEE);
    /// assert!(nan.is_nan() && flags == Flags::INVALID);
    /// ```
    #[must_use]
    pub fn acosh_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Acosh, behavior)
    }

    /// Returns `atanh self`, with the default mode.
    #[must_use]
    pub fn atanh(self) -> Self {
        self.atanh_with(M::default()).0
    }

    /// Returns the inverse hyperbolic tangent `atanh self`, correctly
    /// rounded in the direction of the behavior, as IEEE 754-2019 `atanh`
    /// does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does, also
    /// for a `self` near 0. Every finite result but that of a zero is
    /// inexact. The special cases follow IEEE 754-2019 section 9.2.1: ±0
    /// gives ±0, ±1 gives ±inf and signals divide-by-zero, and a value
    /// beyond ±1 or an infinity gives the default NaN and signals invalid. A
    /// format without an infinity gives its NaN or its largest finite value
    /// for ±inf.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let (pole, flags) = F64::from_bits(0xBFF0_0000_0000_0000).atanh_with(Env::IEEE);
    /// assert_eq!((pole.to_bits(), flags), (0xFFF0_0000_0000_0000, Flags::DIVIDE_BY_ZERO));
    /// ```
    #[must_use]
    pub fn atanh_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Atanh, behavior)
    }

    /// Returns `sin(pi self)`, with the default mode.
    #[must_use]
    pub fn sin_pi(self) -> Self {
        self.sin_pi_with(M::default()).0
    }

    /// Returns `sin(pi self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `sinPi` does, and the flags.
    ///
    /// The argument reduces exactly, so a large `self` keeps every digit of
    /// its fraction, and the result rounds once, as
    /// [`exp_with`](Self::exp_with) does. An integer gives ±0 with the sign of
    /// `self`, and a half-integer gives ±1, exactly. Every other finite result
    /// is inexact. The special cases follow IEEE 754-2019 section 9.2.1: ±0
    /// gives ±0, and ±inf gives the default NaN and signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let sin_pi = |bits| F64::from_bits(bits).sin_pi_with(Env::IEEE);
    /// assert_eq!(sin_pi(0x3FE0_0000_0000_0000), (F64::from_bits(0x3FF0_0000_0000_0000), Flags::NONE));
    /// assert_eq!(sin_pi(0xBFF0_0000_0000_0000).0.to_bits(), 0x8000_0000_0000_0000);
    /// ```
    #[must_use]
    pub fn sin_pi_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::SinPi, behavior)
    }

    /// Returns `cos(pi self)`, with the default mode.
    #[must_use]
    pub fn cos_pi(self) -> Self {
        self.cos_pi_with(M::default()).0
    }

    /// Returns `cos(pi self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `cosPi` does, and the flags.
    ///
    /// The result rounds once, as [`sin_pi_with`](Self::sin_pi_with) does.
    /// An integer gives ±1, and a half-integer gives +0, exactly. Every other
    /// finite result is inexact. The special cases follow IEEE 754-2019
    /// section 9.2.1: ±0 gives 1, and ±inf gives the default NaN and signals
    /// invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let cos_pi = |bits| F64::from_bits(bits).cos_pi_with(Env::IEEE);
    /// assert_eq!(cos_pi(0xBFE0_0000_0000_0000), (F64::from_bits(0), Flags::NONE));
    /// assert_eq!(cos_pi(0x4008_0000_0000_0000).0.to_bits(), 0xBFF0_0000_0000_0000);
    /// ```
    #[must_use]
    pub fn cos_pi_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::CosPi, behavior)
    }

    /// Returns `tan(pi self)`, with the default mode.
    #[must_use]
    pub fn tan_pi(self) -> Self {
        self.tan_pi_with(M::default()).0
    }

    /// Returns `tan(pi self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `tanPi` does, and the flags.
    ///
    /// The result rounds once, as [`sin_pi_with`](Self::sin_pi_with) does.
    /// An integer gives ±0: with the sign of `self` for an even integer, and
    /// the other sign for an odd one. `n + 1/4` gives 1 and `n - 1/4` gives
    /// -1, exactly. `n + 1/2` gives +inf for an even `n` and -inf for an odd
    /// `n`, and signals divide-by-zero. Every other finite result is inexact.
    /// The special cases follow IEEE 754-2019 section 9.2.1: ±0 gives ±0, and
    /// ±inf gives the default NaN and signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let tan_pi = |bits| F64::from_bits(bits).tan_pi_with(Env::IEEE);
    /// assert_eq!(tan_pi(0x3FD0_0000_0000_0000), (F64::from_bits(0x3FF0_0000_0000_0000), Flags::NONE));
    /// let (pole, flags) = tan_pi(0x3FE0_0000_0000_0000);
    /// assert_eq!((pole.to_bits(), flags), (0x7FF0_0000_0000_0000, Flags::DIVIDE_BY_ZERO));
    /// assert_eq!(tan_pi(0x3FF0_0000_0000_0000).0.to_bits(), 0x8000_0000_0000_0000);
    /// ```
    #[must_use]
    pub fn tan_pi_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::TanPi, behavior)
    }

    /// Returns `asin(self)`, with the default mode.
    #[must_use]
    pub fn asin(self) -> Self {
        self.asin_with(M::default()).0
    }

    /// Returns `asin(self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `asin` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. Every
    /// finite result other than a zero is inexact: ±1 gives ±pi/2 rounded.
    /// The special cases follow IEEE 754-2019 section 9.2.1: ±0 gives ±0, and
    /// an argument above 1 in magnitude, or ±inf, gives the default NaN and
    /// signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let asin = |bits| F64::from_bits(bits).asin_with(Env::IEEE);
    /// assert_eq!(asin(0x3FF0_0000_0000_0000), (F64::from_bits(0x3FF9_21FB_5444_2D18), Flags::INEXACT));
    /// assert_eq!(asin(0x4000_0000_0000_0000).1, Flags::INVALID);
    /// ```
    #[must_use]
    pub fn asin_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Asin, behavior)
    }

    /// Returns `acos(self)`, with the default mode.
    #[must_use]
    pub fn acos(self) -> Self {
        self.acos_with(M::default()).0
    }

    /// Returns `acos(self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `acos` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. 1 gives
    /// +0 exactly. Every other finite result is inexact: -1 gives pi rounded,
    /// and ±0 gives pi/2 rounded. An argument above 1 in magnitude, or ±inf,
    /// gives the default NaN and signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let acos = |bits| F64::from_bits(bits).acos_with(Env::IEEE);
    /// assert_eq!(acos(0x3FF0_0000_0000_0000), (F64::from_bits(0), Flags::NONE));
    /// assert_eq!(acos(0xBFF0_0000_0000_0000), (F64::from_bits(0x4009_21FB_5444_2D18), Flags::INEXACT));
    /// ```
    #[must_use]
    pub fn acos_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Acos, behavior)
    }

    /// Returns `atan(self)`, with the default mode.
    #[must_use]
    pub fn atan(self) -> Self {
        self.atan_with(M::default()).0
    }

    /// Returns `atan(self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `atan` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. Every
    /// result other than a zero is inexact: ±1 gives ±pi/4 rounded, and ±inf
    /// gives ±pi/2 rounded. The special cases follow IEEE 754-2019 section
    /// 9.2.1: ±0 gives ±0.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let atan = |bits| F64::from_bits(bits).atan_with(Env::IEEE);
    /// assert_eq!(atan(0x3FF0_0000_0000_0000), (F64::from_bits(0x3FE9_21FB_5444_2D18), Flags::INEXACT));
    /// assert_eq!(atan(0xFFF0_0000_0000_0000).0.to_bits(), 0xBFF9_21FB_5444_2D18);
    /// ```
    #[must_use]
    pub fn atan_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::Atan, behavior)
    }

    /// Returns `asin(self) / pi`, with the default mode.
    #[must_use]
    pub fn asin_pi(self) -> Self {
        self.asin_pi_with(M::default()).0
    }

    /// Returns `asin(self) / pi`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `asinPi` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. ±1
    /// gives ±1/2 exactly. Every other finite result other than a zero is
    /// inexact. The special cases follow IEEE 754-2019 section 9.2.1: ±0
    /// gives ±0, and an argument above 1 in magnitude, or ±inf, gives the
    /// default NaN and signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let asin_pi = |bits| F64::from_bits(bits).asin_pi_with(Env::IEEE);
    /// assert_eq!(asin_pi(0xBFF0_0000_0000_0000), (F64::from_bits(0xBFE0_0000_0000_0000), Flags::NONE));
    /// ```
    #[must_use]
    pub fn asin_pi_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::AsinPi, behavior)
    }

    /// Returns `acos(self) / pi`, with the default mode.
    #[must_use]
    pub fn acos_pi(self) -> Self {
        self.acos_pi_with(M::default()).0
    }

    /// Returns `acos(self) / pi`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `acosPi` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. 1 gives
    /// +0, ±0 gives 1/2, and -1 gives 1, exactly. Every other finite result
    /// is inexact. An argument above 1 in magnitude, or ±inf, gives the
    /// default NaN and signals invalid.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let acos_pi = |bits| F64::from_bits(bits).acos_pi_with(Env::IEEE);
    /// assert_eq!(acos_pi(0x8000_0000_0000_0000), (F64::from_bits(0x3FE0_0000_0000_0000), Flags::NONE));
    /// assert_eq!(acos_pi(0xBFF0_0000_0000_0000), (F64::from_bits(0x3FF0_0000_0000_0000), Flags::NONE));
    /// ```
    #[must_use]
    pub fn acos_pi_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::AcosPi, behavior)
    }

    /// Returns `atan(self) / pi`, with the default mode.
    #[must_use]
    pub fn atan_pi(self) -> Self {
        self.atan_pi_with(M::default()).0
    }

    /// Returns `atan(self) / pi`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `atanPi` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. ±1
    /// gives ±1/4 exactly. Every other finite result other than a zero is
    /// inexact. The special cases follow IEEE 754-2019 section 9.2.1: ±0
    /// gives ±0, and ±inf gives ±1/2.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let atan_pi = |bits| F64::from_bits(bits).atan_pi_with(Env::IEEE);
    /// assert_eq!(atan_pi(0x3FF0_0000_0000_0000), (F64::from_bits(0x3FD0_0000_0000_0000), Flags::NONE));
    /// assert_eq!(atan_pi(0xFFF0_0000_0000_0000), (F64::from_bits(0xBFE0_0000_0000_0000), Flags::NONE));
    /// ```
    #[must_use]
    pub fn atan_pi_with(self, behavior: impl Override) -> (Self, Flags) {
        self.elementary_with(Transcendental::AtanPi, behavior)
    }
}
