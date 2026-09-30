//! The decoded form of a value that the engines compute with.

/// The largest scale that `scale_b` applies. Every format overflows or
/// underflows at a smaller scale, and the limit keeps every exponent of an
/// [`Unpacked`] value inside an `i32`.
pub const SCALE_LIMIT: i32 = 1 << 30;

/// A decoded value of a binary or a decimal format.
///
/// A finite value is `significand * RADIX^exponent`. `exponent` is the weight
/// of the lowest significand digit. A normal binary significand has its bit
/// `PRECISION - 1` set. A decimal significand is the coefficient of the
/// encoding, and its exponent is the quantum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Unpacked<L> {
    /// A zero.
    Zero {
        /// The sign.
        negative: bool,
        /// The exponent of a decimal zero, its quantum. A binary zero has
        /// exponent 0.
        exponent: i32,
    },
    /// A nonzero finite value.
    Finite {
        /// The sign.
        negative: bool,
        /// The weight of the lowest significand digit, as a power of the
        /// radix.
        exponent: i32,
        /// The significand, below `RADIX^PRECISION`.
        significand: L,
    },
    /// An infinity.
    Infinity {
        /// The sign.
        negative: bool,
    },
    /// A NaN.
    Nan {
        /// The sign.
        negative: bool,
        /// `true` for a signaling NaN.
        signaling: bool,
        /// The payload: the fraction bits below the quiet bit of a binary
        /// format, or the value of the trailing significand field of a
        /// decimal format.
        payload: L,
    },
    /// An encoding that the format does not define, such as an x87 unnormal.
    Unsupported,
}

impl<L> Unpacked<L> {
    /// Returns a zero of the binary engine, whose exponent is 0.
    #[inline]
    pub const fn zero(negative: bool) -> Self {
        Self::Zero {
            negative,
            exponent: 0,
        }
    }

    /// Returns the parts of a nonzero finite value, or `None` for any other
    /// value.
    #[inline]
    pub fn number(self) -> Option<Number<L>> {
        match self {
            Self::Finite {
                negative,
                exponent,
                significand,
            } => Some(Number {
                negative,
                exponent,
                significand,
            }),
            _ => None,
        }
    }
}

/// The parts of a nonzero finite value: `significand * RADIX^exponent`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Number<L> {
    /// The sign.
    pub negative: bool,
    /// The weight of the lowest significand digit, as a power of the radix.
    pub exponent: i32,
    /// The significand.
    pub significand: L,
}
