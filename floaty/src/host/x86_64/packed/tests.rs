//! Unit tests of the packed instructions of the SSE unit.

use super::super::{
    binary_f32, binary_f64, narrow_double, narrow_half, sqrt_f32, sqrt_f64, widen_half,
    widen_single,
};
use super::Operation;
use crate::env::Rounding;

/// Storage aligned to 32 bytes, so that the lanes at an offset of one lane
/// have an address that is not a multiple of 16.
#[repr(align(32))]
struct Aligned<T>(T);

/// Returns the `C` lanes at lane 1 of `lanes`.
fn unaligned<T: Copy, const C: usize>(lanes: &[T]) -> [T; C] {
    lanes[1..=C]
        .try_into()
        .expect("the storage holds a chunk after lane 0")
}

#[test]
fn every_chunk_gives_the_lanes_of_the_scalar_instructions() {
    let singles = Aligned(
        [
            0x3FC0_0000_u32,
            0x4010_0000,
            0x4080_0000,
            0x4110_0000,
            0x4180_0000,
        ]
        .map(f32::from_bits),
    );
    let doubles = Aligned(
        [
            0x4000_0000_0000_0000_u64,
            0x3FD0_0000_0000_0000,
            0x4022_0000_0000_0000,
        ]
        .map(f64::from_bits),
    );
    let four: [f32; 4] = unaligned(&singles.0);
    let two: [f64; 2] = unaligned(&doubles.0);
    let pair: [f32; 2] = unaligned(&singles.0);
    let lanes = super::binary_f32x4(four, four, Operation::Mul);
    let each = four.map(|lane| binary_f32(lane, lane, Operation::Mul));
    assert_eq!(lanes.map(f32::to_bits), each.map(f32::to_bits), "MULPS");
    let lanes = super::sqrt_f32x4(four);
    assert_eq!(
        lanes.map(f32::to_bits),
        four.map(sqrt_f32).map(f32::to_bits),
        "SQRTPS"
    );
    let lanes = super::binary_f64x2(two, two, Operation::Div);
    let each = two.map(|lane| binary_f64(lane, lane, Operation::Div));
    assert_eq!(lanes.map(f64::to_bits), each.map(f64::to_bits), "DIVPD");
    let lanes = super::sqrt_f64x2(two);
    assert_eq!(
        lanes.map(f64::to_bits),
        two.map(sqrt_f64).map(f64::to_bits),
        "SQRTPD"
    );
    let widened = super::widen_x2(pair);
    assert_eq!(
        widened.map(f64::to_bits),
        pair.map(widen_single).map(f64::to_bits),
        "CVTPS2PD"
    );
    let narrowed = super::narrow_x2(two);
    assert_eq!(
        narrowed.map(f32::to_bits),
        two.map(narrow_double).map(f32::to_bits),
        "CVTPD2PS"
    );
    if let Some(lanes) = super::round_f32x4(four, Rounding::TiesToEven) {
        let again = super::round_f32x4(lanes, Rounding::TiesToEven).expect("the build has ROUNDPS");
        assert_eq!(lanes.map(f32::to_bits), again.map(f32::to_bits), "ROUNDPS");
    }
    if let Some(lanes) = super::round_f64x2(two, Rounding::TiesToEven) {
        let again = super::round_f64x2(lanes, Rounding::TiesToEven).expect("the build has ROUNDPD");
        assert_eq!(lanes.map(f64::to_bits), again.map(f64::to_bits), "ROUNDPD");
    }
}

#[test]
fn every_binary16_chunk_gives_the_lanes_of_the_scalar_instructions() {
    // 1.5, the smallest subnormal, the largest finite value, -0, 1.0, the
    // largest subnormal, -infinity, and a signaling NaN.
    let halves: [u16; 8] = [
        0x3E00, 0x0001, 0x7BFF, 0x8000, 0x3C00, 0x03FF, 0xFC00, 0x7C01,
    ];
    // A tie below 1.0 + 2^-10, a tie at half the smallest subnormal, a
    // value that overflows, the negative smallest subnormal, the smallest
    // normal binary32 value, a value just above a tie, 65504.0, and a
    // signaling NaN.
    let singles = [
        0x3F80_1000_u32,
        0x3300_0000,
        0x477F_F000,
        0xB380_0000,
        0x0080_0000,
        0x3F80_1001,
        0x477F_E000,
        0x7F80_0001,
    ]
    .map(f32::from_bits);
    let widen = |bits: u16| widen_half(bits).expect("the build has F16C").to_bits();
    let narrow = |value: f32| narrow_half(value).expect("the build has F16C");
    let first: [u16; 4] = halves[..4].try_into().expect("eight lanes");
    let four: [f32; 4] = singles[..4].try_into().expect("eight lanes");
    if let Some(lanes) = super::widen_halves_x4(first) {
        assert_eq!(lanes.map(f32::to_bits), first.map(widen), "VCVTPH2PS");
    }
    if let Some(lanes) = super::narrow_halves_x4(four) {
        assert_eq!(lanes, four.map(narrow), "VCVTPS2PH");
    }
    if let Some(lanes) = super::widen_halves_x8(halves) {
        assert_eq!(
            lanes.map(f32::to_bits),
            halves.map(widen),
            "VCVTPH2PS, 256 bits"
        );
    }
    if let Some(lanes) = super::narrow_halves_x8(singles) {
        assert_eq!(lanes, singles.map(narrow), "VCVTPS2PH, 256 bits");
    }
}
