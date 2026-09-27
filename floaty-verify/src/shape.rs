//! Checks of the form that `Decoded` documents, beyond the value itself.

use floaty::{Class, Decoded};
use rug::Integer;
use rug::integer::Order;

/// The payload that a NaN encoding carries by the definition of its format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Payload {
    /// The format has no payload, so the payload is zero.
    None,
    /// The low `PRECISION - 2` bits of the encoding: the trailing significand
    /// field below the quiet bit.
    Trailing(Integer),
}

/// Checks the form of a decoded value.
///
/// A finite significand has 1 to `precision` bits, and its exponent is at
/// least `emin - (precision - 1)`. A significand with fewer than `precision`
/// bits has exactly that exponent. A normal value has `precision` bits. A NaN
/// payload has at most `precision - 2` bits and equals `payload`.
///
/// # Panics
///
/// Panics with `context` when the form is wrong.
pub fn check<const N: usize>(
    decoded: &Decoded<N>,
    class: Class,
    precision: u32,
    emin: i32,
    payload: &Payload,
    context: &str,
) {
    let lowest = emin - i32::try_from(precision - 1).expect("a precision fits an i32");
    match decoded {
        Decoded::Finite {
            exponent,
            significand,
            ..
        } => {
            let bits = Integer::from_digits(significand, Order::Lsf).significant_bits();
            assert!(
                (1..=precision).contains(&bits),
                "{context}: significand width {bits}"
            );
            assert!(
                *exponent >= lowest,
                "{context}: exponent {exponent} below {lowest}"
            );
            if bits < precision {
                assert_eq!(
                    *exponent, lowest,
                    "{context}: a short significand is subnormal"
                );
            }
            if class == Class::Normal {
                assert_eq!(
                    bits, precision,
                    "{context}: a normal significand is full width"
                );
            }
        }
        Decoded::Nan { payload: ours, .. } => {
            let ours = Integer::from_digits(ours, Order::Lsf);
            assert!(
                ours.significant_bits() <= precision - 2,
                "{context}: payload width"
            );
            let expected = match payload {
                Payload::None => Integer::ZERO,
                Payload::Trailing(bits) => bits.clone(),
            };
            assert_eq!(ours, expected, "{context}: payload");
        }
        Decoded::Zero { .. } | Decoded::Infinity { .. } | Decoded::Unsupported => {}
    }
}

/// Returns the low `precision - 2` bits of an encoding, the payload of a NaN
/// in a format whose NaNs carry one.
#[must_use]
pub fn trailing_payload(encoding: &Integer, precision: u32) -> Payload {
    let mask = (Integer::from(1) << (precision - 2)) - 1u32;
    Payload::Trailing(encoding.clone() & mask)
}
