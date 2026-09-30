//! The rules that map floaty's results to the results of the library, which
//! the tests against the library share.

use floaty::format::{Bid, Decimal, Standard};
use floaty::{Env, Float, ToInt};

use super::{Flags, Format, Integer};

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
/// [`super::equal_operand_choice`] applies.
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
