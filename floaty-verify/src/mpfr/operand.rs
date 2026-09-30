//! An operand of the oracles, and the one rule by which every oracle reads
//! it.
//!
//! The oracles read their operands with floaty's own `decode` and
//! `classify`, which `tests/classification.rs` checks. They use no other part
//! of floaty to compute an expected result.

use floaty::env::Mode;
use floaty::format::Standard;
use floaty::{Class, Decoded, Env, Flags, Float};
use rug::float::Special;
use rug::integer::Order;
use rug::{Float as BigFloat, Integer};

use super::{Nan, exact};

/// An operand: its decoded value, and whether its encoding is subnormal.
#[derive(Clone, Copy, Debug)]
pub struct Operand<const N: usize> {
    /// The decoded value.
    pub decoded: Decoded<N>,
    /// `true` when the encoding is subnormal.
    pub subnormal: bool,
}

impl<const N: usize> Operand<N> {
    /// Returns a floaty value as an operand, decoded in `N` limbs.
    #[must_use]
    pub fn of<S: Standard<W>, const W: usize, M: Mode>(value: Float<S, W, M>) -> Self {
        Self {
            decoded: value.decode::<N>(),
            subnormal: value.classify() == Class::Subnormal,
        }
    }

    /// Reads the operand as an operation in `env` reads it. A subnormal
    /// operand reports `DENORMAL_INPUT` in `flags`. With denormals-are-zero
    /// set, it reads as a zero with its sign.
    pub fn read(&self, env: &Env, flags: &mut Flags) -> Read {
        if self.subnormal {
            *flags |= Flags::DENORMAL_INPUT;
        }
        match self.decoded {
            Decoded::Zero { negative, .. } => Read::Number(signed_zero(negative)),
            Decoded::Finite { negative, .. } if self.subnormal && env.denormals_are_zero => {
                Read::Number(signed_zero(negative))
            }
            Decoded::Finite {
                negative,
                exponent,
                significand,
            } => Read::Number(exact(negative, exponent, &significand)),
            Decoded::Infinity { negative } => Read::Number(special(if negative {
                Special::NegInfinity
            } else {
                Special::Infinity
            })),
            Decoded::Nan {
                negative,
                signaling,
                payload,
            } => Read::Nan(Nan {
                negative,
                signaling,
                payload: Integer::from_digits(&payload, Order::Lsf),
            }),
            Decoded::Unsupported => Read::Unsupported,
        }
    }
}

/// An operand as an operation reads it, after denormals-are-zero.
#[derive(Clone, Debug)]
pub enum Read {
    /// A zero, a finite value, or an infinity, with its sign.
    Number(BigFloat),
    /// A NaN.
    Nan(Nan),
    /// An unsupported x87 encoding.
    Unsupported,
}

impl Read {
    /// Returns `true` for a signaling NaN.
    #[must_use]
    pub fn is_signaling(&self) -> bool {
        matches!(
            self,
            Self::Nan(Nan {
                signaling: true,
                ..
            })
        )
    }
}

/// Returns a zero or an infinity with a sign, as an MPFR value.
fn special(value: Special) -> BigFloat {
    BigFloat::with_val(2, value)
}

/// Returns a zero with a sign, as an MPFR value.
pub(crate) fn signed_zero(negative: bool) -> BigFloat {
    special(if negative {
        Special::NegZero
    } else {
        Special::Zero
    })
}
