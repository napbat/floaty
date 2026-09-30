//! The rules that map floaty's results to the results of the library, which
//! the tests against the library share.

use floaty::format::{Bid, Decimal, Standard};
use floaty::{Env, Float, ToInt};

use super::{Flags, Format, Integer, Rounding};

/// The flags that a library function reports, as floaty's flags map to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signals {
    /// The five IEEE flags.
    Ieee,
    /// The IEEE flags without inexact, which the function does not signal.
    NoInexact,
}

impl Signals {
    /// Returns the library flags of floaty's flags. The library sets its
    /// denormal flag only in a conversion from a binary format, so the
    /// mapping drops `DENORMAL_INPUT`.
    #[must_use]
    pub fn map(self, flags: floaty::Flags) -> Flags {
        let dropped = match self {
            Self::Ieee => floaty::Flags::DENORMAL_INPUT,
            Self::NoInexact => floaty::Flags::DENORMAL_INPUT | floaty::Flags::INEXACT,
        };
        Flags::from_floaty(flags.difference(dropped))
    }
}

/// A result: an encoding or an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    /// An encoding.
    Bits(u128),
    /// An integer, or a predicate as 0 or 1.
    Integer(i128),
}

/// The rule for `minnum` and `maxnum` of operands that compare equal, which
/// [`equal_operand_choice`] applies.
pub const EQUAL_OPERANDS: &str = "floaty's rule for operands that compare equal: -0 below +0, and \
     the members of a cohort in totalOrder";

/// Returns an encoding in its storage type.
///
/// # Panics
///
/// Panics when the encoding does not fit the storage.
#[must_use]
pub fn narrow<T: TryFrom<u128>>(bits: u128) -> T {
    T::try_from(bits).unwrap_or_else(|_| panic!("the encoding {bits:#x} fits its storage"))
}

/// Returns an encoding of the format `F` from its bits. A caller whose
/// bounds fix `F::Bits` to a storage type takes this function, because the
/// bound `TryFrom<u128>` of [`Format::Bits`] holds only through `F`.
///
/// # Panics
///
/// Panics when the encoding does not fit the storage.
#[must_use]
pub fn encoding<F: Format>(bits: u128) -> F::Bits {
    narrow(bits)
}

/// Converts to an integer of type `I` with floaty, and maps an invalid
/// conversion to the integer indefinite of `integer`, as the library gives
/// it.
pub fn to_integer<I, const W: usize>(
    x: Float<Decimal<Bid>, W>,
    integer: Integer,
    env: Env,
) -> (i128, floaty::Flags)
where
    I: floaty::Integer + Into<i128>,
    Decimal<Bid>: Standard<W>,
{
    let (result, flags) = x.to_int_with::<I>(env);
    let value = match result {
        ToInt::Value(value) => value.into(),
        ToInt::OutOfRange { .. } | ToInt::Nan => integer.indefinite(),
    };
    (value, flags)
}

/// Returns the floaty behavior that follows the library in `rounding`.
///
/// - A two-operand function returns the first NaN operand, made quiet, and
///   gives a signaling NaN no priority, as `bid64_add.c` shows.
/// - `fma` checks its NaN operands before the invalid product `0 * inf`, so
///   a quiet NaN addend gives that NaN without invalid, as `bid64_fma.c`
///   shows. The order of its NaN operands is `y`, `z`, then `x`; floaty
///   has no rule with that order.
/// - A conversion to a binary format detects tininess after rounding, as
///   every conversion line of `readtest.in` expects. A decimal result
///   detects tininess before rounding, as floaty always does.
/// - The default NaN is positive with a zero payload, as floaty's is.
#[must_use]
pub fn env(rounding: Rounding) -> floaty::Env {
    use floaty::env::{InvalidProduct, NanPropagation, NanRule};
    let nan = NanRule::new(NanPropagation::FirstOperand)
        .with_invalid_product(InvalidProduct::YieldsToNan);
    floaty::Env::IEEE
        .with_rounding(rounding.into())
        .with_nan(nan)
}

/// The minimum or the maximum operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extremum {
    /// `minNum`, `minimum`, or `minimumNumber`.
    Minimum,
    /// `maxNum`, `maximum`, or `maximumNumber`.
    Maximum,
}

/// Returns the result that floaty gives the minimum or the maximum of
/// two operands that compare equal.
///
/// IEEE 754-2008 section 5.3.1 lets `minNum` and `maxNum` return either
/// operand when the operands compare equal, and `readtest.c` accepts either
/// one. floaty fixes the choice: `-0` is below `+0`, and the members of
/// a cohort order as `totalOrder` of IEEE 754-2019 section 5.10 orders them.
/// The library's `totalOrder` decides the order here, and the result is the
/// canonical encoding of the operand that it puts first for the minimum, or
/// last for the maximum.
///
/// # Panics
///
/// Panics when [`Layout::canonical`](super::Layout::canonical) does not give
/// a canonical encoding of the same datum, as the library reads it.
#[must_use]
pub fn equal_operand_choice<F: Format>(x: F::Bits, y: F::Bits, extremum: Extremum) -> F::Bits {
    let x_first = F::total_order(x, y);
    let chosen = match (extremum, x_first) {
        (Extremum::Minimum, true) | (Extremum::Maximum, false) => x,
        (Extremum::Minimum, false) | (Extremum::Maximum, true) => y,
    };
    let canonical = F::Bits::try_from(F::LAYOUT.canonical(chosen.into()))
        .ok()
        .expect("a canonical encoding fits the storage of its format");
    assert!(
        F::is_canonical(canonical)
            && F::total_order(canonical, chosen)
            && F::total_order(chosen, canonical),
        "{canonical:#x} is the canonical encoding of {chosen:#x}"
    );
    canonical
}
