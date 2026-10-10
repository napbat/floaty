//! The functions of two arguments of IEEE 754-2019 section 9.2: `atan2` and
//! `atan2Pi`, and `pow` and `powr`.

use super::Float;
use crate::elementary::Bivariate;
use crate::env::{Flags, Mode, Override};
use crate::format::Standard;

impl<S: Standard<W>, const W: usize, M: Mode> Float<S, W, M> {
    /// Returns `atan2(self, x)`, the angle of the point `(x, self)`, with the
    /// default mode.
    #[must_use]
    pub fn atan2(self, x: Self) -> Self {
        self.atan2_with(x, M::default()).0
    }

    /// Returns `atan2(self, x)`, the angle of the point `(x, self)` in
    /// `[-pi, pi]`, correctly rounded in the direction of the behavior, as
    /// IEEE 754-2019 `atan2` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. Every
    /// result other than a zero is inexact. The special cases follow IEEE
    /// 754-2019 section 9.2.1, with the sign of `self` on each angle:
    ///
    /// - a zero `self` gives ±0 for `x` of the + sign, `+0` and `+inf`
    ///   included, and ±pi for `x` of the - sign;
    /// - a finite `self` other than zero gives ±pi/2 for a zero `x`, ±0 for
    ///   `x = +inf`, and ±pi for `x = -inf`;
    /// - an infinite `self` gives ±pi/2 for a finite `x`, ±pi/4 for
    ///   `x = +inf`, and ±3pi/4 for `x = -inf`.
    ///
    /// A NaN operand gives a NaN by the rule of the behavior, in the operand
    /// order `self`, `x`.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let atan2 = |y, x| F64::from_bits(y).atan2_with(F64::from_bits(x), Env::IEEE);
    /// assert_eq!(atan2(0x3FF0_0000_0000_0000, 0x3FF0_0000_0000_0000), (F64::from_bits(0x3FE9_21FB_5444_2D18), Flags::INEXACT));
    /// assert_eq!(atan2(0, 0x8000_0000_0000_0000).0.to_bits(), 0x4009_21FB_5444_2D18);
    /// ```
    #[must_use]
    pub fn atan2_with(self, x: Self, behavior: impl Override) -> (Self, Flags) {
        self.bivariate_with(x, Bivariate::Atan2, behavior)
    }

    /// Returns `atan2(self, x) / pi`, with the default mode.
    #[must_use]
    pub fn atan2_pi(self, x: Self) -> Self {
        self.atan2_pi_with(x, M::default()).0
    }

    /// Returns `atan2(self, x) / pi`, in `[-1, 1]`, correctly rounded in the
    /// direction of the behavior, as IEEE 754-2019 `atan2Pi` does, and the
    /// flags.
    ///
    /// The special cases are those of [`atan2_with`](Self::atan2_with), with
    /// pi as 1: ±1, ±1/2, ±1/4, and ±3/4 are exact. `|self| = |x|` also gives
    /// ±1/4 or ±3/4 exactly. Every other result other than a zero is
    /// inexact.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let atan2_pi = |y, x| F64::from_bits(y).atan2_pi_with(F64::from_bits(x), Env::IEEE);
    /// assert_eq!(atan2_pi(0x4000_0000_0000_0000, 0xC000_0000_0000_0000), (F64::from_bits(0x3FE8_0000_0000_0000), Flags::NONE));
    /// assert_eq!(atan2_pi(0x8000_0000_0000_0000, 0).0.to_bits(), 0x8000_0000_0000_0000);
    /// ```
    #[must_use]
    pub fn atan2_pi_with(self, x: Self, behavior: impl Override) -> (Self, Flags) {
        self.bivariate_with(x, Bivariate::Atan2Pi, behavior)
    }

    /// Returns `self^y`, with the default mode.
    #[must_use]
    pub fn pow(self, y: Self) -> Self {
        self.pow_with(y, M::default()).0
    }

    /// Returns `self^y`, correctly rounded in the direction of the behavior,
    /// as IEEE 754-2019 `pow` does, and the flags.
    ///
    /// The result rounds once, as [`exp_with`](Self::exp_with) does. A
    /// rational result with at most `p + 3` bits, or `p + 2` digits, is exact.
    /// A negative `self` takes an integer `y`, and gives a negative result for
    /// an odd `y`. The special cases follow IEEE 754-2019 section 9.2.1:
    ///
    /// - `y = ±0` gives 1 for any `self`, a quiet NaN included, and `self =
    ///   +1` gives 1 for any `y`, a quiet NaN included;
    /// - a zero `self` gives an infinity and signals divide-by-zero for a
    ///   finite `y < 0`, with the sign of `self` for an odd integer `y`, a
    ///   zero for `y > 0`, and +inf for `y = -inf`;
    /// - `self = -1` gives 1 for `y = ±inf`, and a finite `self` gives +0 or
    ///   +inf for `y = ±inf` by `|self|` against 1;
    /// - `self = ±inf` gives the reciprocal results of `±0`, without a flag;
    /// - a finite `self < 0` and a finite `y` that is not an integer give the
    ///   default NaN and signal invalid.
    ///
    /// A NaN operand otherwise gives a NaN by the rule of the behavior, in the
    /// operand order `self`, `y`.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let pow = |x, y| F64::from_bits(x).pow_with(F64::from_bits(y), Env::IEEE);
    /// // 4^(1/2) = 2 and (-2)^3 = -8, exactly.
    /// assert_eq!(pow(0x4010_0000_0000_0000, 0x3FE0_0000_0000_0000), (F64::from_bits(0x4000_0000_0000_0000), Flags::NONE));
    /// assert_eq!(pow(0xC000_0000_0000_0000, 0x4008_0000_0000_0000), (F64::from_bits(0xC020_0000_0000_0000), Flags::NONE));
    /// // A quiet NaN to the power 0 is 1, and (-1)^(1/2) is invalid.
    /// assert_eq!(pow(0x7FF8_0000_0000_0000, 0), (F64::from_bits(0x3FF0_0000_0000_0000), Flags::NONE));
    /// assert_eq!(pow(0xBFF0_0000_0000_0000, 0x3FE0_0000_0000_0000).1, Flags::INVALID);
    /// ```
    #[must_use]
    pub fn pow_with(self, y: Self, behavior: impl Override) -> (Self, Flags) {
        self.bivariate_with(y, Bivariate::Pow, behavior)
    }

    /// Returns `e^(y ln self)`, with the default mode.
    #[must_use]
    pub fn powr(self, y: Self) -> Self {
        self.powr_with(y, M::default()).0
    }

    /// Returns `e^(y ln self)`, correctly rounded in the direction of the
    /// behavior, as IEEE 754-2019 `powr` does, and the flags.
    ///
    /// `powr` is [`pow_with`](Self::pow_with) for `self >= 0` alone. The
    /// special cases follow IEEE 754-2019 section 9.2.1: `y = ±0` gives 1 for
    /// a finite `self > 0`, `self = +1` gives 1 for a finite `y`, and a zero
    /// `self` gives +inf and signals divide-by-zero for a finite `y < 0`, +0
    /// for `y > 0`, and +inf for `y = -inf`. A `self < 0`, `±0` to the power
    /// `±0`, `+inf` to the power `±0`, and `+1` to the power `±inf` give the
    /// default NaN and signal invalid. A NaN operand gives a NaN by the rule of
    /// the behavior.
    ///
    /// ```
    /// use floaty::{Env, F64, Flags};
    ///
    /// let powr = |x, y| F64::from_bits(x).powr_with(F64::from_bits(y), Env::IEEE);
    /// assert_eq!(powr(0x4000_0000_0000_0000, 0x4024_0000_0000_0000), (F64::from_bits(0x4090_0000_0000_0000), Flags::NONE));
    /// assert_eq!(powr(0xC000_0000_0000_0000, 0x4008_0000_0000_0000).1, Flags::INVALID);
    /// assert_eq!(powr(0, 0).1, Flags::INVALID);
    /// ```
    #[must_use]
    pub fn powr_with(self, y: Self, behavior: impl Override) -> (Self, Flags) {
        self.bivariate_with(y, Bivariate::Powr, behavior)
    }

    /// Returns `function` of `self` and `other`, in that order, and the
    /// flags.
    fn bivariate_with(
        self,
        other: Self,
        function: Bivariate,
        behavior: impl Override,
    ) -> (Self, Flags) {
        let (bits, flags) = S::bivariate(self.bits, other.bits, function, behavior.apply::<M>());
        (Self::from_masked(bits), flags)
    }
}
