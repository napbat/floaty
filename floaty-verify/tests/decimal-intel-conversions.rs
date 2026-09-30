//! Compares floaty's conversions between `D32Bid`, `D64Bid`, and `D128Bid`,
//! to and from the DPD formats, and to and from binary32, binary64, x87
//! extended, and binary128 with the Intel Decimal Floating-Point Math Library
//! 2.0 Update 2: its `readtest.in` vectors, and seeded random operands.
//!
//! floaty runs with the behavior of [`intel_decimal::env`]. The test compares
//! the result bits, NaN payloads included, and the five IEEE flags. These
//! rules map floaty to the library:
//!
//! - The library sets its denormal flag in a conversion from a binary format
//!   exactly when the binary operand is subnormal, an x87 pseudo-denormal
//!   included. Those conversions compare it with `DENORMAL_INPUT`. The other
//!   conversions drop `DENORMAL_INPUT`, and the library must not set its
//!   denormal flag.
//! - `bid_to_dpd*` and `bid_dpd_to_bid*` re-encode a NaN as it is: a
//!   signaling NaN stays signaling and signals nothing, and the DPD to BID
//!   direction keeps the bits between the signaling bit and the payload.
//!   floaty's `convert` is IEEE 754 `convertFormat`, which quiets a signaling
//!   NaN and signals invalid, and returns a canonical encoding. For a NaN
//!   result the test expects the result of the library made quiet and
//!   canonical, and invalid for a signaling NaN.
//! - The library converts an x87 unnormal, pseudo-infinity, or pseudo-NaN as
//!   the value of its bits. floaty follows the 387 and later processors,
//!   which treat such an operand as invalid, as the Intel SDM, Volume 1,
//!   section 8.2.2 states: the test expects the default NaN and invalid.
//!
//! The counts show how many cases each rule decides. The archive is pinned
//! and the generators are seeded, so each test asserts the counts exactly.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use std::collections::BTreeMap;

use floaty::format::{Bid, Binary, Decimal, Dpd, Standard, X87};
use floaty::{Env, Flags, Float};
use floaty_verify::intel_decimal::{
    self, Bid32, Bid64, Bid128, Flags as IntelFlags, Format, Layout, Outcome,
    Rounding as IntelRounding, narrow,
};
use floaty_verify::random::SplitMix64;
use floaty_verify::readtest::{self, Field, FieldError};

/// The layout of a binary interchange format, or of x87 extended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BinaryLayout {
    /// The width in bits.
    width: u32,
    /// The width of the exponent field.
    exponent_bits: u32,
}

/// binary32.
const BINARY32: BinaryLayout = BinaryLayout {
    width: 32,
    exponent_bits: 8,
};
/// binary64.
const BINARY64: BinaryLayout = BinaryLayout {
    width: 64,
    exponent_bits: 11,
};
/// x87 extended, with its explicit integer bit.
const BINARY80: BinaryLayout = BinaryLayout {
    width: 80,
    exponent_bits: 15,
};
/// binary128.
const BINARY128: BinaryLayout = BinaryLayout {
    width: 128,
    exponent_bits: 15,
};

impl BinaryLayout {
    /// Returns whether the format stores its integer bit: x87 extended.
    fn explicit_integer(self) -> bool {
        self.width == 80
    }

    /// The bits below the exponent field, with the x87 integer bit.
    fn fraction_bits(self) -> u32 {
        self.width - 1 - self.exponent_bits
    }

    /// The precision in bits.
    fn precision(self) -> u32 {
        if self.explicit_integer() {
            self.fraction_bits()
        } else {
            self.fraction_bits() + 1
        }
    }

    /// The largest exponent field, which infinities and NaNs use.
    fn largest_field(self) -> u32 {
        (1 << self.exponent_bits) - 1
    }

    /// The exponent bias.
    fn bias(self) -> i64 {
        (1 << (self.exponent_bits - 1)) - 1
    }

    fn encode(self, negative: bool, field: u32, fraction: u128) -> u128 {
        (u128::from(negative) << (self.width - 1))
            | (u128::from(field) << self.fraction_bits())
            | fraction
    }

    /// Returns whether an x87 encoding is unsupported: an unnormal, a
    /// pseudo-infinity, or a pseudo-NaN, which has a nonzero exponent field
    /// and a clear integer bit (Intel SDM Volume 1, Table 8-3).
    fn unsupported(self, bits: u128) -> bool {
        self.explicit_integer() && (bits >> 64) & 0x7fff != 0 && (bits >> 63) & 1 == 0
    }

    /// Returns the normal encoding of `magnitude * 2^exponent`, or `None`
    /// when the value is not a normal value of the format.
    fn exact(self, negative: bool, magnitude: u128, exponent: i64) -> Option<u128> {
        let top = magnitude.checked_ilog2()?;
        let precision = self.precision();
        if top >= precision {
            return None;
        }
        let field = exponent + i64::from(top) + self.bias();
        if field < 1 || field >= i64::from(self.largest_field()) {
            return None;
        }
        let significand = magnitude << (precision - 1 - top);
        let fraction = if self.explicit_integer() {
            significand
        } else {
            significand - (1 << (precision - 1))
        };
        Some(self.encode(negative, u32::try_from(field).ok()?, fraction))
    }

    /// Returns a random encoding, biased to zeros, subnormal values, both
    /// ends of the exponent range, infinities, and NaNs with payloads. An x87
    /// encoding has a wrong integer bit at times: a pseudo-denormal, an
    /// unnormal, a pseudo-infinity, or a pseudo-NaN.
    fn random(self, rng: &mut SplitMix64) -> u128 {
        let negative = rng.next_u64() & 1 == 1;
        let largest = self.largest_field();
        let field = match below(rng, 8) {
            0 => 0,
            1 => largest,
            2 => 1 + below_u32(rng, 3),
            3 => largest - 1 - below_u32(rng, 3),
            4 => {
                let bias = u32::try_from(self.bias()).expect("a bias fits a u32");
                bias - 70 + below_u32(rng, 140)
            }
            _ => below_u32(rng, largest + 1),
        };
        let bits = if self.explicit_integer() {
            self.fraction_bits() - 1
        } else {
            self.fraction_bits()
        };
        let mask = (1 << bits) - 1;
        let fraction = match below(rng, 8) {
            0 => 0,
            1 => mask,
            2 => 1,
            3 => 1 << below(rng, u64::from(bits)),
            4 => 1 << (bits - 1),
            5 => (rng.next_u128() & mask) >> below(rng, u64::from(bits)) << below(rng, 8),
            _ => rng.next_u128(),
        } & mask;
        if !self.explicit_integer() {
            return self.encode(negative, field, fraction);
        }
        let integer = (field != 0) != (below(rng, 8) == 0);
        self.encode(negative, field, fraction | (u128::from(integer) << bits))
    }

    /// Returns a value with few significant bits, whose exact decimal value
    /// has few digits: an exact, a halfway, or a near-halfway decimal case.
    fn short(self, rng: &mut SplitMix64) -> u128 {
        let bits = 1 + below_u32(rng, self.precision().min(60));
        let magnitude = (rng.next_u128() & ((1 << bits) - 1)) | (1 << (bits - 1));
        let exponent = i64::from(below_u32(rng, 161)) - 80;
        self.exact(rng.next_u64() & 1 == 1, magnitude, exponent)
            .unwrap_or_else(|| self.random(rng))
    }

    /// Returns an integer halfway between two values of `layout`: `p + 1`
    /// digits that end in 5, or a neighbor of such an integer.
    fn decimal_tie(self, rng: &mut SplitMix64, layout: Layout) -> u128 {
        let unit = Layout::power_of_ten(layout.precision());
        let head = unit + below_u128(rng, 9 * unit);
        let tie = head / 10 * 10 + 5;
        let magnitude = match below(rng, 3) {
            0 => tie - 1,
            1 => tie + 1,
            _ => tie,
        };
        self.exact(rng.next_u64() & 1 == 1, magnitude, 0)
            .unwrap_or_else(|| self.short(rng))
    }
}

/// Returns a random value below `bound`.
fn below(rng: &mut SplitMix64, bound: u64) -> u64 {
    rng.next_u64() % bound
}

/// Returns a random value below `bound`.
fn below_u32(rng: &mut SplitMix64, bound: u32) -> u32 {
    u32::try_from(below(rng, u64::from(bound))).expect("the value is below a u32 bound")
}

/// Returns a random value below `bound`.
fn below_u128(rng: &mut SplitMix64, bound: u128) -> u128 {
    rng.next_u128() % bound
}

/// A decimal value of `layout` inside or just outside the range of
/// `binary`, near its smallest subnormal value, its smallest normal value,
/// its largest value, or anywhere between.
fn near_binary(rng: &mut SplitMix64, layout: Layout, binary: BinaryLayout) -> u128 {
    let precision = i64::from(binary.precision());
    let smallest = 1 - binary.bias() - (precision - 1);
    let normal = 1 - binary.bias();
    let largest = binary.bias() + 1;
    let near = |rng: &mut SplitMix64, at: i64| at + i64::from(below_u32(rng, 9)) - 4;
    let exponent = match below(rng, 4) {
        0 => near(rng, smallest),
        1 => near(rng, normal),
        2 => near(rng, largest),
        _ => {
            smallest
                + i64::try_from(below(rng, (largest - smallest).unsigned_abs() + 1))
                    .expect("an exponent fits an i64")
        }
    };
    // The decimal exponent of the leading digit is about exponent * log10(2).
    let leading = (exponent * 30_103).div_euclid(100_000);
    let digits = 1 + below_u32(rng, layout.precision());
    let field = leading - i64::from(digits) + 1 + i64::from(below_u32(rng, 3)) - 1
        + i64::from(layout.bias());
    let field = u32::try_from(field.clamp(0, i64::from(layout.largest_field())))
        .expect("a clamped field fits a u32");
    let low = Layout::power_of_ten(digits - 1);
    let coefficient = low + below_u128(rng, Layout::power_of_ten(digits) - low);
    layout.number(rng.next_u64() & 1 == 1, field, coefficient)
}

/// A decimal value of `layout` that is exactly halfway between two values of
/// `binary`, exactly a value of `binary`, or next to one of those: an odd
/// integer of `p + 1` or fewer bits times a power of 2, written exactly in
/// decimal. The decimal exponent is 0 for a positive power of 2, negative for
/// a negative power of 2, and positive for an odd integer that holds a power
/// of 5; see [`scaled_tie`].
fn binary_tie(rng: &mut SplitMix64, layout: Layout, binary: BinaryLayout) -> u128 {
    let bits = if below(rng, 2) == 0 {
        binary.precision() + 1
    } else {
        1 + below_u32(rng, binary.precision() + 1)
    };
    if bits > 120 {
        return near_binary(rng, layout, binary);
    }
    let odd = (rng.next_u128() & ((1 << bits) - 1)) | (1 << (bits - 1)) | 1;
    let power = below_u32(rng, 41);
    let (coefficient, exponent) = match below(rng, 3) {
        0 => (odd.checked_mul(1 << power), 0),
        1 => {
            let five = 5_u128.checked_pow(power);
            (
                five.and_then(|five| odd.checked_mul(five)),
                -i64::from(power),
            )
        }
        _ => match scaled_tie(rng, layout, bits) {
            Some((coefficient, exponent)) => (Some(coefficient), exponent),
            None => (None, 0),
        },
    };
    let largest = Layout::power_of_ten(layout.precision()) - 1;
    let Some(coefficient) = coefficient.filter(|&coefficient| coefficient < largest) else {
        return near_binary(rng, layout, binary);
    };
    let coefficient = match below(rng, 4) {
        0 => coefficient + 1,
        1 => coefficient - 1,
        _ => coefficient,
    };
    let field = u32::try_from(exponent + i64::from(layout.bias())).expect("the field fits");
    layout.number(rng.next_u64() & 1 == 1, field, coefficient)
}

/// Returns the coefficient and the exponent of a decimal value that is an
/// odd integer of `bits` bits times a power of 2, with a positive exponent.
///
/// The odd integer is `m * 5^q`, so the value `m * 5^q * 2^(q + s)` is
/// `(m * 2^s) * 10^q`. The coefficient `m * 2^s` stays below `10^p - 1`. The
/// shift `s` is often the largest that fits, so the coefficient has all the
/// digits of the format. Returns `None` when no such odd integer has a small
/// enough factor `m`.
fn scaled_tie(rng: &mut SplitMix64, layout: Layout, bits: u32) -> Option<(u128, i64)> {
    let limit = Layout::power_of_ten(layout.precision()) - 2;
    let (low, high) = (1_u128 << (bits - 1), (1_u128 << bits) - 1);
    // The exponents q with 5^q below 2^(bits - 1), so an integer of `bits`
    // bits can be a multiple of 5^q, and with the largest factor m within
    // the limit.
    let exponents = (1..=55).filter(|&q| {
        let five = 5_u128.pow(q);
        five < low && high / five <= limit
    });
    let (first, last) = (exponents.clone().min()?, exponents.max()?);
    let q = first + below_u32(rng, last - first + 1);
    let five = 5_u128.pow(q);
    let (smallest, largest) = (low.div_ceil(five), high / five);
    let m = (smallest + below_u128(rng, largest - smallest + 1)) | 1;
    let m = if m > largest { m - 2 } else { m };
    if m < smallest {
        return None;
    }
    let room = (limit / m).ilog2();
    let s = if below(rng, 2) == 0 {
        room
    } else {
        below_u32(rng, room + 1)
    };
    Some((m << s, i64::from(q)))
}

/// One side of a conversion: a format and its encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    /// A decimal format in the BID encoding.
    Bid(Layout),
    /// A decimal format in the DPD encoding.
    Dpd(Layout),
    /// A binary format.
    Binary(BinaryLayout),
}

impl Side {
    /// Reads an operand or a result of this side from a `readtest.in`
    /// field. A decimal string converts to BID with `from_string`, as
    /// `readtest.c` does, also for a DPD field.
    fn read(self, field: &Field, rounding: IntelRounding) -> Result<u128, FieldError> {
        match self {
            Self::Bid(layout) | Self::Dpd(layout) => match layout.width {
                32 => field.decimal::<Bid32>(rounding).map(u128::from),
                64 => field.decimal::<Bid64>(rounding).map(u128::from),
                _ => field.decimal::<Bid128>(rounding),
            },
            Self::Binary(binary) => match binary.width {
                32 => field.binary32().map(u128::from),
                64 => field.binary64().map(u128::from),
                80 => field.binary80(),
                _ => field.binary128(),
            },
        }
    }
}

/// A conversion of floaty from and to encodings in a `u128`.
type Floaty = fn(u128, Env) -> (u128, Flags);

/// A conversion of the library from and to encodings in a `u128`.
type Library = fn(u128, IntelRounding) -> Outcome<u128>;

/// A conversion between two formats.
struct Conversion {
    /// The name of the library function.
    name: String,
    source: Side,
    target: Side,
    floaty: Floaty,
    library: Library,
}

/// Converts the encoding `bits` of `Float<S, V>` to `Float<T, W>` with
/// floaty.
fn convert<S: Standard<V>, const V: usize, T: Standard<W>, const W: usize>(
    bits: u128,
    env: Env,
) -> (u128, Flags)
where
    S::Bits: TryFrom<u128>,
    T::Bits: Into<u128>,
{
    let source = Float::<S, V>::from_bits(narrow(bits));
    let (result, flags) = source.convert_with::<Float<T, W>>(env);
    (result.to_bits().into(), flags)
}

/// Returns a result of a library function without flags.
fn exact(value: u128) -> Outcome<u128> {
    Outcome {
        value,
        flags: IntelFlags::NONE,
    }
}

fn to_dpd<F: Format>(x: u128, _: IntelRounding) -> Outcome<u128> {
    exact(F::to_dpd(narrow(x)).into())
}

fn from_dpd<F: Format>(x: u128, _: IntelRounding) -> Outcome<u128> {
    exact(F::from_dpd(narrow(x)).into())
}

fn from_binary32<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::from_binary32(narrow(x), rounding).map(Into::into)
}

fn from_binary64<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::from_binary64(narrow(x), rounding).map(Into::into)
}

fn from_binary80<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::from_binary80(x, rounding).map(Into::into)
}

fn from_binary128<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::from_binary128(x, rounding).map(Into::into)
}

fn to_binary32<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::to_binary32(narrow(x), rounding).map(u128::from)
}

fn to_binary64<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::to_binary64(narrow(x), rounding).map(u128::from)
}

fn to_binary80<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::to_binary80(narrow(x), rounding)
}

fn to_binary128<F: Format>(x: u128, rounding: IntelRounding) -> Outcome<u128> {
    F::to_binary128(narrow(x), rounding)
}

/// binary32 as a floaty standard.
type B32 = Binary<8>;
/// binary64.
type B64 = Binary<11>;
/// x87 extended.
type B80 = Binary<15, X87>;
/// binary128.
type B128 = Binary<15>;
/// A decimal format in BID.
type D = Decimal<Bid>;
/// A decimal format in DPD.
type P = Decimal<Dpd>;

/// Returns the conversions between the BID format `F` of width `W` and its
/// DPD format and the binary formats.
fn conversions_of<F: Format, const W: usize>() -> Vec<Conversion>
where
    D: Standard<W, Bits = F::Bits>,
    P: Standard<W, Bits = F::Bits>,
{
    let bid = Side::Bid(F::LAYOUT);
    let dpd = Side::Dpd(F::LAYOUT);
    let conversion = |name: String, source, target, floaty, library| Conversion {
        name,
        source,
        target,
        floaty,
        library,
    };
    let from: [(Floaty, Library); 4] = [
        (convert::<B32, 32, D, W>, from_binary32::<F>),
        (convert::<B64, 64, D, W>, from_binary64::<F>),
        (convert::<B80, 80, D, W>, from_binary80::<F>),
        (convert::<B128, 128, D, W>, from_binary128::<F>),
    ];
    let to: [(Floaty, Library); 4] = [
        (convert::<D, W, B32, 32>, to_binary32::<F>),
        (convert::<D, W, B64, 64>, to_binary64::<F>),
        (convert::<D, W, B80, 80>, to_binary80::<F>),
        (convert::<D, W, B128, 128>, to_binary128::<F>),
    ];
    let mut conversions = vec![
        conversion(
            format!("bid_to_dpd{W}"),
            bid,
            dpd,
            convert::<D, W, P, W>,
            to_dpd::<F>,
        ),
        conversion(
            format!("bid_dpd_to_bid{W}"),
            dpd,
            bid,
            convert::<P, W, D, W>,
            from_dpd::<F>,
        ),
    ];
    let binary = [BINARY32, BINARY64, BINARY80, BINARY128];
    for ((layout, (floaty_from, library_from)), (floaty_to, library_to)) in
        binary.into_iter().zip(from).zip(to)
    {
        let (side, width) = (Side::Binary(layout), layout.width);
        conversions.push(conversion(
            format!("binary{width}_to_bid{W}"),
            side,
            bid,
            floaty_from,
            library_from,
        ));
        conversions.push(conversion(
            format!("bid{W}_to_binary{width}"),
            bid,
            side,
            floaty_to,
            library_to,
        ));
    }
    conversions
}

/// Returns every conversion.
fn conversions() -> Vec<Conversion> {
    let widths: [(&str, Layout, Layout, Floaty, Library); 6] = [
        (
            "bid32_to_bid64",
            Bid32::LAYOUT,
            Bid64::LAYOUT,
            convert::<D, 32, D, 64>,
            |x, _| intel_decimal::bid32_to_bid64(narrow(x)).map(u128::from),
        ),
        (
            "bid32_to_bid128",
            Bid32::LAYOUT,
            Bid128::LAYOUT,
            convert::<D, 32, D, 128>,
            |x, _| intel_decimal::bid32_to_bid128(narrow(x)),
        ),
        (
            "bid64_to_bid32",
            Bid64::LAYOUT,
            Bid32::LAYOUT,
            convert::<D, 64, D, 32>,
            |x, r| intel_decimal::bid64_to_bid32(narrow(x), r).map(u128::from),
        ),
        (
            "bid64_to_bid128",
            Bid64::LAYOUT,
            Bid128::LAYOUT,
            convert::<D, 64, D, 128>,
            |x, _| intel_decimal::bid64_to_bid128(narrow(x)),
        ),
        (
            "bid128_to_bid32",
            Bid128::LAYOUT,
            Bid32::LAYOUT,
            convert::<D, 128, D, 32>,
            |x, r| intel_decimal::bid128_to_bid32(x, r).map(u128::from),
        ),
        (
            "bid128_to_bid64",
            Bid128::LAYOUT,
            Bid64::LAYOUT,
            convert::<D, 128, D, 64>,
            |x, r| intel_decimal::bid128_to_bid64(x, r).map(u128::from),
        ),
    ];
    let mut all: Vec<Conversion> = widths
        .into_iter()
        .map(|(name, source, target, floaty, library)| Conversion {
            name: name.to_owned(),
            source: Side::Bid(source),
            target: Side::Bid(target),
            floaty,
            library,
        })
        .collect();
    all.extend(conversions_of::<Bid32, 32>());
    all.extend(conversions_of::<Bid64, 64>());
    all.extend(conversions_of::<Bid128, 128>());
    all
}

/// The rule for a NaN in a conversion between BID and DPD.
const REENCODED_NAN: &str = "a quiet canonical NaN, and invalid for a signaling NaN (floaty)";

/// The rule for an unsupported x87 operand.
const UNSUPPORTED_X87: &str = "the default NaN and invalid for an unsupported x87 operand \
     (Intel SDM)";

impl Conversion {
    /// Returns the flags of floaty that the library reports.
    fn map_flags(&self, flags: Flags) -> IntelFlags {
        let flags = match self.source {
            Side::Binary(_) => flags,
            Side::Bid(_) | Side::Dpd(_) => flags.difference(Flags::DENORMAL_INPUT),
        };
        IntelFlags::from_floaty(flags)
    }

    /// Returns the expected outcome for operand `x`, from the outcome of the
    /// library, and the rule of floaty that changes it, if any.
    fn expected(&self, x: u128, library: Outcome<u128>) -> (Outcome<u128>, Option<&'static str>) {
        match (self.source, self.target) {
            (Side::Binary(binary), Side::Bid(layout)) if binary.unsupported(x) => {
                let nan = Outcome {
                    value: layout.nan(false, false, 0, 0),
                    flags: IntelFlags::INVALID,
                };
                (nan, Some(UNSUPPORTED_X87))
            }
            (Side::Bid(layout), Side::Dpd(_)) | (Side::Dpd(layout), Side::Bid(_)) => {
                let nan = 0b1_1111 << (layout.width - 6);
                if library.value & nan != nan {
                    return (library, None);
                }
                let signaling = 0b11_1111 << (layout.width - 7);
                let negative = library.value >> (layout.width - 1) == 1;
                let payload = library.value & ((1 << layout.trailing()) - 1);
                let flags = if x & signaling == signaling {
                    library.flags | IntelFlags::INVALID
                } else {
                    library.flags
                };
                let canonical = Outcome {
                    value: layout.nan(negative, false, payload, 0),
                    flags,
                };
                let rule = (canonical != library).then_some(REENCODED_NAN);
                (canonical, rule)
            }
            _ => (library, None),
        }
    }
}

/// The failures of one group: their count and the first few.
#[derive(Default)]
struct Failures {
    count: usize,
    examples: Vec<String>,
}

/// The counts of a run.
#[derive(Default)]
struct Report {
    passed: BTreeMap<String, usize>,
    failures: BTreeMap<String, Failures>,
    /// How many expected outcomes raise each library flag, by bit.
    flags: [usize; 6],
}

impl Report {
    /// The number of failures that the report prints for each group.
    const EXAMPLES: usize = 6;

    /// Runs one conversion of `x` with floaty, and compares it with the
    /// library outcome `library`.
    fn check(
        &mut self,
        conversion: &Conversion,
        x: u128,
        rounding: IntelRounding,
        library: Outcome<u128>,
        context: &dyn Fn() -> String,
    ) {
        let (value, flags) = (conversion.floaty)(x, intel_decimal::env(rounding));
        let actual = Outcome {
            value,
            flags: conversion.map_flags(flags),
        };
        let (expected, rule) = conversion.expected(x, library);
        count_flags(&mut self.flags, expected.flags);
        let group = match rule {
            Some(rule) => format!("{} by the rule: {rule}", conversion.name),
            None => conversion.name.clone(),
        };
        if actual == expected {
            *self.passed.entry(group).or_default() += 1;
            return;
        }
        let failures = self.failures.entry(group).or_default();
        failures.count += 1;
        if failures.examples.len() < Self::EXAMPLES {
            failures.examples.push(format!(
                "{} {x:#x} {rounding:?} {}: floaty {:#x} {:#04x}, expected {:#x} {:#04x}",
                conversion.name,
                context(),
                actual.value,
                actual.flags.bits(),
                expected.value,
                expected.flags.bits(),
            ));
        }
    }

    /// Prints the counts and the failures. Returns the number of passes and
    /// of failures.
    fn print(&self, title: &str) -> (usize, usize) {
        let passed: usize = self.passed.values().sum();
        let failed: usize = self.failures.values().map(|failures| failures.count).sum();
        println!("{title}: {passed} passed, {failed} failed");
        print_flags(&self.flags);
        for (group, count) in &self.passed {
            println!("  passed {count:7}: {group}");
        }
        for (group, failures) in &self.failures {
            println!("  FAILED {:7}: {group}", failures.count);
            for example in &failures.examples {
                println!("    {example}");
            }
        }
        (passed, failed)
    }

    /// Asserts the number of passes, and the cases that each rule decides,
    /// as `(conversion, rule, count)`.
    fn assert_counts(&self, passed: usize, rules: &[(&str, &str, usize)]) {
        assert_eq!(self.passed.values().sum::<usize>(), passed, "the passes");
        let decided: BTreeMap<String, usize> = self
            .passed
            .iter()
            .filter(|(group, _)| group.contains(" by the rule: "))
            .map(|(group, &count)| (group.clone(), count))
            .collect();
        let expected: BTreeMap<String, usize> = rules
            .iter()
            .map(|&(conversion, rule, count)| (format!("{conversion} by the rule: {rule}"), count))
            .collect();
        assert_eq!(decided, expected, "the cases that each rule decides");
    }
}

/// Counts the flags of an expected outcome.
fn count_flags(counts: &mut [usize; 6], flags: IntelFlags) {
    for (bit, count) in counts.iter_mut().enumerate() {
        *count += usize::from(flags.bits() >> bit & 1 == 1);
    }
}

/// Prints how many expected outcomes raise each flag.
fn print_flags(counts: &[usize; 6]) {
    let counts: Vec<String> = IntelFlags::NAMES
        .iter()
        .zip(counts)
        .map(|(name, count)| format!("{name} {count}"))
        .collect();
    println!("  expected flags: {}", counts.join(", "));
}

#[test]
fn readtest_conversions() {
    let conversions = conversions();
    let mut report = Report::default();
    let mut unreadable = Vec::new();
    for line in readtest::read() {
        let Some(conversion) = conversions.iter().find(|c| c.name == line.function) else {
            continue;
        };
        // `readtest.c` converts a decimal string operand rounding to nearest,
        // and a decimal string result in the direction of the line.
        let operand = line
            .operands
            .first()
            .ok_or_else(|| String::from("the operand is missing"))
            .and_then(|field| {
                let read = conversion.source.read(field, IntelRounding::TiesToEven);
                read.map_err(|error| error.to_string())
            });
        let result = conversion.target.read(&line.result, line.rounding);
        let (x, value) = match (operand, result) {
            (Ok(x), Ok(value)) => (x, value),
            (operand, result) => {
                unreadable.push(format!(
                    "readtest.in:{} {operand:?} {result:?}",
                    line.number
                ));
                continue;
            }
        };
        let library = Outcome {
            value,
            flags: line.flags,
        };
        let context = || format!("readtest.in:{}", line.number);
        report.check(conversion, x, line.rounding, library, &context);
    }
    let (_, failed) = report.print("readtest.in conversions, floaty");
    assert!(unreadable.is_empty(), "unreadable lines: {unreadable:#?}");
    assert_eq!(failed, 0, "floaty differs from readtest.in");
    report.assert_counts(
        41_505,
        &[
            ("bid_to_dpd32", REENCODED_NAN, 1),
            ("bid_to_dpd64", REENCODED_NAN, 2),
            ("bid_to_dpd128", REENCODED_NAN, 1),
            ("bid_dpd_to_bid32", REENCODED_NAN, 6),
            ("bid_dpd_to_bid64", REENCODED_NAN, 7),
            ("bid_dpd_to_bid128", REENCODED_NAN, 7),
            ("binary80_to_bid32", UNSUPPORTED_X87, 10),
            ("binary80_to_bid64", UNSUPPORTED_X87, 10),
            ("binary80_to_bid128", UNSUPPORTED_X87, 11),
        ],
    );
}

/// Returns a random operand for a conversion, biased to the edge cases of
/// both formats.
fn operand(rng: &mut SplitMix64, conversion: &Conversion) -> u128 {
    match (conversion.source, conversion.target) {
        (Side::Binary(binary), Side::Bid(layout)) => match below(rng, 5) {
            0 => binary.short(rng),
            1 => binary.decimal_tie(rng, layout),
            _ => binary.random(rng),
        },
        (Side::Bid(layout), Side::Binary(binary)) => match below(rng, 4) {
            0 => layout.random(rng),
            1 => binary_tie(rng, layout, binary),
            _ => near_binary(rng, layout, binary),
        },
        (Side::Bid(layout), _) => layout.random(rng),
        (Side::Dpd(layout), _) => {
            let bid = layout.random(rng);
            match (below(rng, 2), layout.width) {
                (0, _) => rng.next_u128() & (u128::MAX >> (128 - layout.width)),
                (_, 32) => Bid32::to_dpd(narrow(bid)).into(),
                (_, 64) => Bid64::to_dpd(narrow(bid)).into(),
                _ => Bid128::to_dpd(bid),
            }
        }
        (Side::Binary(_), _) => unreachable!("a binary format converts only to BID"),
    }
}

/// Compares the conversions that `select` picks on random operands, and
/// returns the report of a run without failures.
fn random_conversions(
    title: &str,
    seed: u64,
    cases: usize,
    select: fn(&Conversion) -> bool,
) -> Report {
    let mut report = Report::default();
    let mut rng = SplitMix64::new(seed);
    for conversion in conversions().iter().filter(|conversion| select(conversion)) {
        for _ in 0..cases {
            let x = operand(&mut rng, conversion);
            let index = usize::try_from(below(&mut rng, 5)).expect("an index fits a usize");
            let rounding = IntelRounding::ALL[index];
            let library = (conversion.library)(x, rounding);
            report.check(conversion, x, rounding, library, &String::new);
        }
    }
    let (_, failed) = report.print(title);
    assert_eq!(failed, 0, "{title}: floaty differs from the library");
    report
}

#[test]
fn conversions_between_decimals_match_the_library() {
    let report = random_conversions(
        "conversions between decimal formats",
        0x7e57_1001,
        400_000,
        |conversion| {
            !matches!(conversion.source, Side::Binary(_))
                && !matches!(conversion.target, Side::Binary(_))
        },
    );
    report.assert_counts(
        4_800_000,
        &[
            ("bid_to_dpd32", REENCODED_NAN, 10_441),
            ("bid_to_dpd64", REENCODED_NAN, 10_282),
            ("bid_to_dpd128", REENCODED_NAN, 10_360),
            ("bid_dpd_to_bid32", REENCODED_NAN, 8_218),
            ("bid_dpd_to_bid64", REENCODED_NAN, 11_370),
            ("bid_dpd_to_bid128", REENCODED_NAN, 8_229),
        ],
    );
}

#[test]
fn binary_to_decimal_matches_the_library() {
    let report = random_conversions(
        "conversions from binary formats",
        0x7e57_1002,
        240_000,
        |conversion| matches!(conversion.source, Side::Binary(_)),
    );
    report.assert_counts(
        2_880_000,
        &[
            ("binary80_to_bid32", UNSUPPORTED_X87, 15_691),
            ("binary80_to_bid64", UNSUPPORTED_X87, 15_822),
            ("binary80_to_bid128", UNSUPPORTED_X87, 15_841),
        ],
    );
}

#[test]
fn decimal_to_binary_matches_the_library() {
    let report = random_conversions(
        "conversions to binary formats",
        0x7e57_1003,
        120_000,
        |conversion| matches!(conversion.target, Side::Binary(_)),
    );
    report.assert_counts(1_440_000, &[]);
}
