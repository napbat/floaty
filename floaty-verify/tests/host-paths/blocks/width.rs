//! The binary types of the block tests, `F16`, `BF16`, `F32`, and `F64`:
//! their host elements, their operands, their entry points, and the `_with`
//! methods that give the steps one at a time.

use std::fmt::Debug;
use std::ops::Neg;

use floaty::block::{Chain, Steps};
use floaty::{BF16, Element, Env, F16, F32, F64, Flags};
use floaty_verify::encodings::Layout;
use floaty_verify::random::SplitMix64;

use super::super::operands::{boundary_pairs, random_pairs};

/// Calls `$macro` with the `_with` methods of the steps one at a time,
/// and the arguments of each before its behavior.
macro_rules! steps_with {
    ($macro:ident) => {
        $macro! {
            add_with(other: Self);
            sub_with(other: Self);
            mul_with(other: Self);
            div_with(other: Self);
            mul_add_with(multiplier: Self, addend: Self);
            sqrt_with();
            next_up_with();
            next_down_with();
            round_to_integral_with();
            remainder_with(divisor: Self);
            truncated_remainder_with(divisor: Self);
            scale_b_with(scale: i32);
            log_b_with();
            exp_with();
            log_with();
            compound_with(n: i64);
            hypot_with(other: Self);
            pown_with(n: i64);
            rootn_with(n: i64);
            reciprocal_sqrt_with();
            minimum_with(other: Self);
            maximum_with(other: Self);
            minimum_number_with(other: Self);
            maximum_number_with(other: Self);
            min_num_with(other: Self);
            max_num_with(other: Self);
            minimum_magnitude_with(other: Self);
            maximum_magnitude_with(other: Self);
            minimum_magnitude_number_with(other: Self);
            maximum_magnitude_number_with(other: Self);
        }
    };
}

/// Declares each `_with` method.
macro_rules! declare {
    ($($name:ident($($argument:ident: $type:ty),*);)*) => {
        $(
            fn $name(self, $($argument: $type,)* behavior: Env) -> (Self, Flags);
        )*
    };
}

/// Implements each `_with` method by the method of the type.
macro_rules! forward {
    ($($name:ident($($argument:ident: $type:ty),*);)*) => {
        $(
            fn $name(self, $($argument: $type,)* behavior: Env) -> (Self, Flags) {
                Self::$name(self, $($argument,)* behavior)
            }
        )*
    };
}

/// Implements `map` and `evaluate` of a type by its entry points.
macro_rules! entry_points {
    () => {
        fn map<C, E, O>(chain: &C, x: [&[E]; 2], p: [Self; 1], out: &mut [O])
        where
            C: Chain<2, 1>,
            E: Element<Self>,
            O: Element<Self>,
        {
            Self::map(chain, x, p, out);
        }

        fn evaluate<C: Chain<2, 1>>(chain: &C, x: [Self; 2], p: [Self; 1]) -> Self {
            Self::evaluate(chain, x, p)
        }
    };
}

/// A binary type of the block tests.
pub(super) trait Width:
    Steps + Element<Self> + Debug + Neg<Output = Self> + From<Self::Host>
{
    /// The host type, whose slices `map` also takes, or the type itself
    /// where the host has none.
    type Host: Copy + Element<Self> + From<Self>;

    /// Scales at both ends of the normal powers of two of the host type of
    /// the lanes, past them, and at the ends of the range of the format, up
    /// to the scales that move the least subnormal value past the largest
    /// finite value.
    const SCALES: [i32; 12];

    /// Returns the encoding, widened to 64 bits.
    fn bits(self) -> u64;

    /// Returns -1, the parameter of the MXCSR test.
    #[cfg(target_arch = "x86_64")]
    fn minus_one() -> Self;

    /// Returns every pair of the boundary encodings, and random pairs.
    fn pairs(random: &mut SplitMix64) -> Vec<(Self, Self)>;

    /// Returns the parameters: zeros, ones, the extremes, infinities, NaNs,
    /// and random encodings.
    fn parameters(random: &mut SplitMix64) -> Vec<Self>;

    /// Returns values that each step of one value takes as both operands:
    /// every encoding of a 16-bit format, and otherwise quarters, halves and
    /// their neighbors, and values next to `2^(p - 1)`, from which every
    /// value is integral, of both signs.
    fn unary_values() -> Vec<Self>;

    /// Returns ten pairs that raise in the host type of the lanes the
    /// exceptions that the format can raise: inexact, overflow, underflow,
    /// divide by zero, invalid, and a subnormal operand, in the chains of the
    /// MXCSR test.
    #[cfg(target_arch = "x86_64")]
    fn exception_pairs() -> [(Self, Self); 10];

    /// Returns `map` of the type.
    fn map<C, E, O>(chain: &C, x: [&[E]; 2], p: [Self; 1], out: &mut [O])
    where
        C: Chain<2, 1>,
        E: Element<Self>,
        O: Element<Self>;

    /// Returns `evaluate` of the type.
    fn evaluate<C: Chain<2, 1>>(chain: &C, x: [Self; 2], p: [Self; 1]) -> Self;

    steps_with!(declare);
}

/// Returns the pairs of `layout` as values of a type from their encodings.
fn pairs_of<T: TryFrom<u128>, F>(
    random: &mut SplitMix64,
    layout: Layout,
    value: impl Fn(T) -> F,
) -> Vec<(F, F)>
where
    T::Error: Debug,
{
    let mut pairs = boundary_pairs::<T>(layout);
    pairs.extend(random_pairs::<T>(random, layout, 4_000));
    pairs
        .into_iter()
        .map(|(x, y)| (value(x), value(y)))
        .collect()
}

/// Returns `edges` and their negations as values of a type.
fn signed<T: Copy + std::ops::BitOr<Output = T>, F>(
    edges: &[T],
    sign: T,
    value: impl Fn(T) -> F,
) -> Vec<F> {
    edges
        .iter()
        .flat_map(|&bits| [bits, bits | sign])
        .map(value)
        .collect()
}

/// Returns the encodings of `bits`, then six random encodings of 16 bits.
fn halves_with_random(random: &mut SplitMix64, bits: [u16; 10]) -> Vec<u16> {
    let mut bits = bits.to_vec();
    bits.extend(
        (0..6).map(|_| u16::try_from(random.next_u64() >> 48).expect("the shift keeps 16 bits")),
    );
    bits
}

impl Width for F16 {
    type Host = Self;

    const SCALES: [i32; 12] = [
        i32::MIN,
        -150,
        -127,
        -126,
        -40,
        -25,
        -1,
        0,
        1,
        40,
        127,
        i32::MAX,
    ];

    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }

    #[cfg(target_arch = "x86_64")]
    fn minus_one() -> Self {
        Self::from_bits(0xBC00)
    }

    fn pairs(random: &mut SplitMix64) -> Vec<(Self, Self)> {
        pairs_of(random, Layout::BINARY16, Self::from_bits)
    }

    fn parameters(random: &mut SplitMix64) -> Vec<Self> {
        let bits = [
            0, 0x8000, 0x3C00, 0xBC00, 1, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0x7C01,
        ];
        halves_with_random(random, bits)
            .into_iter()
            .map(Self::from_bits)
            .collect()
    }

    fn unary_values() -> Vec<Self> {
        (0..=u16::MAX).map(Self::from_bits).collect()
    }

    #[cfg(target_arch = "x86_64")]
    fn exception_pairs() -> [(Self, Self); 10] {
        [
            (0x3C00, 0x4200),
            (0x7BFF, 0x7BFF),
            (0x0400, 0x0400),
            (0x3C00, 0),
            (0, 0),
            (0x7C00, 0xFC00),
            (0x0001, 0x4000),
            (0xBC00, 0x3C00),
            (0x7C01, 0x3C00),
            (0xC200, 0x8000),
        ]
        .map(|(x, y)| (Self::from_bits(x), Self::from_bits(y)))
    }

    entry_points!();
    steps_with!(forward);
}

impl Width for BF16 {
    type Host = Self;

    const SCALES: [i32; 12] = [
        i32::MIN,
        -300,
        -150,
        -127,
        -126,
        -1,
        0,
        1,
        127,
        128,
        277,
        i32::MAX,
    ];

    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }

    #[cfg(target_arch = "x86_64")]
    fn minus_one() -> Self {
        Self::from_bits(0xBF80)
    }

    fn pairs(random: &mut SplitMix64) -> Vec<(Self, Self)> {
        pairs_of(random, Layout::BFLOAT16, Self::from_bits)
    }

    fn parameters(random: &mut SplitMix64) -> Vec<Self> {
        let bits = [
            0, 0x8000, 0x3F80, 0xBF80, 1, 0x7F7F, 0x7F80, 0xFF80, 0x7FC0, 0x7F81,
        ];
        halves_with_random(random, bits)
            .into_iter()
            .map(Self::from_bits)
            .collect()
    }

    fn unary_values() -> Vec<Self> {
        (0..=u16::MAX).map(Self::from_bits).collect()
    }

    #[cfg(target_arch = "x86_64")]
    fn exception_pairs() -> [(Self, Self); 10] {
        [
            (0x3F80, 0x4040),
            (0x7F7F, 0x7F7F),
            (0x0DA2, 0x0DA2),
            (0x3F80, 0),
            (0, 0),
            (0x7F80, 0xFF80),
            (0x0001, 0x4000),
            (0xBF80, 0x3F80),
            (0x7F81, 0x3F80),
            (0xC040, 0x8000),
        ]
        .map(|(x, y)| (Self::from_bits(x), Self::from_bits(y)))
    }

    entry_points!();
    steps_with!(forward);
}

impl Width for F32 {
    type Host = f32;

    const SCALES: [i32; 12] = [
        i32::MIN,
        -300,
        -150,
        -127,
        -126,
        -1,
        0,
        1,
        127,
        128,
        277,
        i32::MAX,
    ];

    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }

    #[cfg(target_arch = "x86_64")]
    fn minus_one() -> Self {
        Self::from_bits(0xBF80_0000)
    }

    fn pairs(random: &mut SplitMix64) -> Vec<(Self, Self)> {
        pairs_of(random, Layout::BINARY32, Self::from_bits)
    }

    fn parameters(random: &mut SplitMix64) -> Vec<Self> {
        let mut bits = vec![
            0,
            0x8000_0000,
            0x3F80_0000,
            0xBF80_0000,
            1,
            0x7F7F_FFFF,
            0x7F80_0000,
            0xFF80_0000,
            0x7FC0_0000,
            0x7F80_0001,
        ];
        bits.extend(
            (0..6)
                .map(|_| u32::try_from(random.next_u64() >> 32).expect("the shift keeps 32 bits")),
        );
        bits.into_iter().map(Self::from_bits).collect()
    }

    fn unary_values() -> Vec<Self> {
        let edges: [u32; 14] = [
            0x3E80_0000,
            0x3EFF_FFFF,
            0x3F00_0000,
            0x3F00_0001,
            0x3F40_0000,
            0x3F7F_FFFF,
            0x3FC0_0000,
            0x4020_0000,
            0x4060_0000,
            0x4A80_0001,
            0x4AFF_FFFE,
            0x4AFF_FFFF,
            0x4B00_0000,
            0x4B00_0001,
        ];
        signed(&edges, 0x8000_0000, Self::from_bits)
    }

    #[cfg(target_arch = "x86_64")]
    fn exception_pairs() -> [(Self, Self); 10] {
        [
            (0x3F80_0000, 0x4040_0000),
            (0x7F7F_FFFF, 0x7F7F_FFFF),
            (0x0DA2_4260, 0x0DA2_4260),
            (0x3F80_0000, 0),
            (0, 0),
            (0x7F80_0000, 0xFF80_0000),
            (0x0000_0001, 0x4000_0000),
            (0xBF80_0000, 0x3F80_0000),
            (0x7F80_0001, 0x3F80_0000),
            (0xC040_0000, 0x8000_0000),
        ]
        .map(|(x, y)| (Self::from_bits(x), Self::from_bits(y)))
    }

    entry_points!();
    steps_with!(forward);
}

impl Width for F64 {
    type Host = f64;

    const SCALES: [i32; 12] = [
        i32::MIN,
        -2_200,
        -1_100,
        -1_023,
        -1_022,
        -1,
        0,
        1,
        1_023,
        1_024,
        2_098,
        i32::MAX,
    ];

    fn bits(self) -> u64 {
        self.to_bits()
    }

    #[cfg(target_arch = "x86_64")]
    fn minus_one() -> Self {
        Self::from_bits(0xBFF0_0000_0000_0000)
    }

    fn pairs(random: &mut SplitMix64) -> Vec<(Self, Self)> {
        pairs_of(random, Layout::BINARY64, Self::from_bits)
    }

    fn parameters(random: &mut SplitMix64) -> Vec<Self> {
        let mut bits = vec![
            0,
            0x8000_0000_0000_0000,
            0x3FF0_0000_0000_0000,
            0xBFF0_0000_0000_0000,
            1,
            0x7FEF_FFFF_FFFF_FFFF,
            0x7FF0_0000_0000_0000,
            0xFFF0_0000_0000_0000,
            0x7FF8_0000_0000_0000,
            0x7FF0_0000_0000_0001,
        ];
        bits.extend((0..6).map(|_| random.next_u64()));
        bits.into_iter().map(Self::from_bits).collect()
    }

    fn unary_values() -> Vec<Self> {
        let edges: [u64; 14] = [
            0x3FD0_0000_0000_0000,
            0x3FDF_FFFF_FFFF_FFFF,
            0x3FE0_0000_0000_0000,
            0x3FE0_0000_0000_0001,
            0x3FE8_0000_0000_0000,
            0x3FEF_FFFF_FFFF_FFFF,
            0x3FF8_0000_0000_0000,
            0x4004_0000_0000_0000,
            0x400C_0000_0000_0000,
            0x4320_0000_0000_0001,
            0x432F_FFFF_FFFF_FFFE,
            0x432F_FFFF_FFFF_FFFF,
            0x4330_0000_0000_0000,
            0x4330_0000_0000_0001,
        ];
        signed(&edges, 0x8000_0000_0000_0000, Self::from_bits)
    }

    #[cfg(target_arch = "x86_64")]
    fn exception_pairs() -> [(Self, Self); 10] {
        [
            (0x3FF0_0000_0000_0000, 0x4008_0000_0000_0000),
            (0x7FEF_FFFF_FFFF_FFFF, 0x7FEF_FFFF_FFFF_FFFF),
            (0x1A70_0000_0000_0000, 0x1A70_0000_0000_0000),
            (0x3FF0_0000_0000_0000, 0),
            (0, 0),
            (0x7FF0_0000_0000_0000, 0xFFF0_0000_0000_0000),
            (1, 0x4000_0000_0000_0000),
            (0xBFF0_0000_0000_0000, 0x3FF0_0000_0000_0000),
            (0x7FF0_0000_0000_0001, 0x3FF0_0000_0000_0000),
            (0xC008_0000_0000_0000, 0x8000_0000_0000_0000),
        ]
        .map(|(x, y)| (Self::from_bits(x), Self::from_bits(y)))
    }

    entry_points!();
    steps_with!(forward);
}
