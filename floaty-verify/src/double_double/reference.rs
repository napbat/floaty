//! The types that the double-double references share: the value, the
//! rounding direction, and the flags of [`crate::ibm_ldouble`] and
//! [`crate::qd`], so one test can compare both references.

use core::fmt;
use core::ops::BitOr;

/// A double-double value as the binary64 encodings of its two halves.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Pair {
    /// The encoding of the high half.
    pub hi: u64,
    /// The encoding of the low half.
    pub lo: u64,
}

impl Pair {
    /// Returns the pair of two binary64 encodings.
    #[must_use]
    pub const fn new(hi: u64, lo: u64) -> Self {
        Self { hi, lo }
    }
}

impl fmt::Debug for Pair {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Pair({:#018x}, {:#018x})", self.hi, self.lo)
    }
}

/// A rounding direction of the C `fesetround` function.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rounding {
    /// `FE_TONEAREST`: to nearest, with a tie to even.
    TiesToEven,
    /// `FE_TOWARDZERO`.
    TowardZero,
    /// `FE_UPWARD`: toward positive infinity.
    TowardPositive,
    /// `FE_DOWNWARD`: toward negative infinity.
    TowardNegative,
}

impl Rounding {
    /// Every rounding direction.
    pub const ALL: [Self; 4] = [
        Self::TiesToEven,
        Self::TowardZero,
        Self::TowardPositive,
        Self::TowardNegative,
    ];
}

impl From<Rounding> for floaty::Rounding {
    fn from(rounding: Rounding) -> Self {
        match rounding {
            Rounding::TiesToEven => Self::TiesToEven,
            Rounding::TowardZero => Self::TowardZero,
            Rounding::TowardPositive => Self::TowardPositive,
            Rounding::TowardNegative => Self::TowardNegative,
        }
    }
}

/// A set of the five IEEE exception flags.
///
/// The bits are the flag bits of both shims: `shim/ibm_ldouble.c` and
/// `shim/qd_shim.cpp`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Flags(u8);

impl Flags {
    /// No flag.
    pub const NONE: Self = Self(0);
    /// `FE_INVALID`.
    pub const INVALID: Self = Self(0x01);
    /// `FE_DIVBYZERO`.
    pub const DIVIDE_BY_ZERO: Self = Self(0x02);
    /// `FE_OVERFLOW`.
    pub const OVERFLOW: Self = Self(0x04);
    /// `FE_UNDERFLOW`.
    pub const UNDERFLOW: Self = Self(0x08);
    /// `FE_INEXACT`.
    pub const INEXACT: Self = Self(0x10);

    /// Every flag, and its name.
    const NAMES: [(Self, &'static str); 5] = [
        (Self::INVALID, "INVALID"),
        (Self::DIVIDE_BY_ZERO, "DIVIDE_BY_ZERO"),
        (Self::OVERFLOW, "OVERFLOW"),
        (Self::UNDERFLOW, "UNDERFLOW"),
        (Self::INEXACT, "INEXACT"),
    ];

    /// The bits of every flag.
    const ALL_BITS: u8 = 0x1F;

    /// Returns the flags with the shim bits, or `None` when a bit is not a
    /// flag.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Option<Self> {
        if bits & !Self::ALL_BITS == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    /// Returns the shim bits.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Returns whether every flag of `other` is set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Returns the five IEEE flags of floaty flags. The references have no
    /// other flag.
    #[must_use]
    pub fn from_floaty(flags: floaty::Flags) -> Self {
        [
            (floaty::Flags::INVALID, Self::INVALID),
            (floaty::Flags::DIVIDE_BY_ZERO, Self::DIVIDE_BY_ZERO),
            (floaty::Flags::OVERFLOW, Self::OVERFLOW),
            (floaty::Flags::UNDERFLOW, Self::UNDERFLOW),
            (floaty::Flags::INEXACT, Self::INEXACT),
        ]
        .into_iter()
        .filter(|&(ours, _)| flags.contains(ours))
        .fold(Self::NONE, |set, (_, theirs)| set | theirs)
    }
}

impl BitOr for Flags {
    type Output = Self;

    fn bitor(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl fmt::Debug for Flags {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = Self::NAMES
            .into_iter()
            .filter(|&(flag, _)| self.contains(flag))
            .map(|(_, name)| name)
            .collect();
        if names.is_empty() {
            formatter.write_str("Flags(NONE)")
        } else {
            write!(formatter, "Flags({})", names.join(" | "))
        }
    }
}

/// The result of an operation and the flags that it raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome {
    /// The result.
    pub result: Pair,
    /// The flags that the operation raised.
    pub flags: Flags,
}
