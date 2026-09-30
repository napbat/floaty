use super::Layout;
use crate::env::{Env, Flags};
use crate::float::Class;
use crate::float::{F4E2M1Fn, F6E3M2Fn, F32};
use crate::format::EncodingKind;
use crate::format::internal::LimbConversion;
use crate::format::{B11Fnuz, Encoding, Finite, Fnuz, Ieee, NoInf, Storage, Width, X87};
use crate::limbs::Limbs;
use crate::unpacked::Unpacked;

type X87Layout = Layout<15, X87, 80>;

/// Counts of each class over every encoding of a small format.
#[derive(Debug, Default, PartialEq, Eq)]
struct Census {
    zero: u64,
    subnormal: u64,
    normal: u64,
    infinite: u64,
    quiet_nan: u64,
    signaling_nan: u64,
}

/// Decodes every encoding below `2^W`, checks the round trip, and counts the
/// classes.
fn sweep<const E: u32, Enc: Encoding, const W: usize>() -> Census
where
    Width<W>: Storage,
{
    let mut census = Census::default();
    for bits in 0..1_u64 << W {
        let limbs = [bits];
        let value = Layout::<E, Enc, W>::decode(limbs);
        assert!(
            Layout::<E, Enc, W>::is_canonical(limbs),
            "{bits:#x} is canonical"
        );
        assert_eq!(
            Layout::<E, Enc, W>::encode(value),
            limbs,
            "{bits:#x} round trip"
        );
        let class = Layout::<E, Enc, W>::classify(limbs);
        check_normal::<E, Enc, W>(limbs, value, class);
        let count = match class {
            Class::Zero => &mut census.zero,
            Class::Subnormal => &mut census.subnormal,
            Class::Normal => &mut census.normal,
            Class::Infinite => &mut census.infinite,
            Class::QuietNan => &mut census.quiet_nan,
            Class::SignalingNan => &mut census.signaling_nan,
            Class::Unsupported => panic!("{bits:#x} is not unsupported"),
        };
        *count += 1;
    }
    census
}

/// Checks that the direct read of a normal operand agrees with the decode:
/// it reads every normal encoding, except a `NoInf` number with the largest
/// exponent field, and no other encoding.
fn check_normal<const E: u32, Enc: Encoding, const W: usize>(
    limbs: [u64; 1],
    value: Unpacked<[u64; 1]>,
    class: Class,
) where
    Width<W>: Storage,
{
    let field = limbs.field(Layout::<E, Enc, W>::FRACTION_BITS, E);
    let largest_field = field == Layout::<E, Enc, W>::FIELD_MAX;
    match Layout::<E, Enc, W>::normal(limbs) {
        Some(number) => {
            assert_eq!(class, Class::Normal, "{limbs:x?}");
            assert_eq!(Some(number), value.number(), "{limbs:x?}");
        }
        None => assert!(
            class != Class::Normal || (Enc::KIND == EncodingKind::NoInf && largest_field),
            "{limbs:x?}"
        ),
    }
}

/// The census of an IEEE format with `e` exponent bits and `f` fraction bits.
fn ieee_census(e: u32, f: u32) -> Census {
    let fractions = 1_u64 << f;
    Census {
        zero: 2,
        subnormal: 2 * (fractions - 1),
        normal: 2 * ((1 << e) - 2) * fractions,
        infinite: 2,
        quiet_nan: 2 * (fractions / 2),
        signaling_nan: 2 * (fractions / 2 - 1),
    }
}

#[test]
fn ieee_formats_round_trip_and_match_their_census() {
    assert_eq!(sweep::<5, Ieee, 8>(), ieee_census(5, 2));
    assert_eq!(sweep::<5, Ieee, 16>(), ieee_census(5, 10));
    assert_eq!(sweep::<8, Ieee, 16>(), ieee_census(8, 7));
    assert_eq!(sweep::<8, Ieee, 19>(), ieee_census(8, 10));
}

#[test]
fn finite_formats_round_trip_and_match_their_census() {
    let census = |e: u32, f: u32| Census {
        zero: 2,
        subnormal: 2 * ((1 << f) - 1),
        normal: 2 * ((1 << e) - 1) * (1 << f),
        ..Census::default()
    };
    assert_eq!(sweep::<2, Finite, 4>(), census(2, 1));
    assert_eq!(sweep::<2, Finite, 6>(), census(2, 3));
    assert_eq!(sweep::<3, Finite, 6>(), census(3, 2));
}

#[test]
fn a_finite_format_saturates_and_gives_zero_for_a_nan() {
    let (zero, six) = (F4E2M1Fn::from_bits(0), F4E2M1Fn::from_bits(0x7));
    let (sum, flags) = six.add_with(six, Env::IEEE);
    assert_eq!(
        (sum.to_bits(), flags),
        (0x7, Flags::OVERFLOW | Flags::INEXACT)
    );
    let (quotient, flags) = six.div_with(zero, Env::IEEE);
    assert_eq!((quotient.to_bits(), flags), (0x7, Flags::DIVIDE_BY_ZERO));
    let (nan, flags) = zero.div_with(zero, Env::IEEE);
    assert_eq!((nan.to_bits(), flags), (0, Flags::INVALID));
    let (root, flags) = F4E2M1Fn::from_bits(0xA).sqrt_with(Env::IEEE);
    assert_eq!((root.to_bits(), flags), (0, Flags::INVALID));
    assert_eq!(six.next_up().to_bits(), 0x7);
    let (converted, flags) = F32::from_bits(0xFFC0_0000).convert_with::<F6E3M2Fn>(Env::IEEE);
    assert_eq!((converted.to_bits(), flags), (0, Flags::INVALID));
    let (converted, flags) = F32::from_bits(0xFF80_0000).convert_with::<F6E3M2Fn>(Env::IEEE);
    assert_eq!((converted.to_bits(), flags), (0x3F, Flags::INVALID));
}

#[test]
fn no_inf_format_round_trips_and_matches_its_census() {
    let census = Census {
        zero: 2,
        subnormal: 2 * 7,
        normal: 2 * (15 * 8 - 1),
        infinite: 0,
        quiet_nan: 2,
        signaling_nan: 0,
    };
    assert_eq!(sweep::<4, NoInf, 8>(), census);
}

#[test]
fn fnuz_formats_round_trip_and_match_their_census() {
    let e4m3 = Census {
        zero: 1,
        subnormal: 2 * 7,
        normal: 2 * 15 * 8,
        infinite: 0,
        quiet_nan: 1,
        signaling_nan: 0,
    };
    assert_eq!(sweep::<4, Fnuz, 8>(), e4m3);
    let e5m2 = Census {
        zero: 1,
        subnormal: 2 * 3,
        normal: 2 * 31 * 4,
        infinite: 0,
        quiet_nan: 1,
        signaling_nan: 0,
    };
    assert_eq!(sweep::<5, Fnuz, 8>(), e5m2);
}

fn finite<L>(negative: bool, exponent: i32, significand: L) -> Unpacked<L> {
    Unpacked::Finite {
        negative,
        exponent,
        significand,
    }
}

#[test]
fn binary32_decodes_known_values() {
    type L = Layout<8, Ieee, 32>;
    assert_eq!(L::decode([0x3F80_0000]), finite(false, -23, [1 << 23]));
    assert_eq!(L::decode([0x0000_0001]), finite(false, -149, [1]));
    assert_eq!(L::decode([0xFF7F_FFFF]), finite(true, 104, [0xFF_FFFF]));
    assert_eq!(L::decode([0x8000_0000]), Unpacked::zero(true));
    assert_eq!(
        L::decode([0xFF80_0001]),
        Unpacked::Nan {
            negative: true,
            signaling: true,
            payload: [1]
        }
    );
    assert_eq!(
        L::decode([0x7FC0_0000]),
        Unpacked::Nan {
            negative: false,
            signaling: false,
            payload: [0]
        }
    );
}

#[test]
fn fp8_formats_decode_their_largest_values() {
    // OCP E4M3: 448 = 14 * 2^5. FNUZ E4M3: 240 = 15 * 2^4. FNUZ E5M2: 57344 = 7 * 2^13.
    assert_eq!(
        Layout::<4, NoInf, 8>::decode([0x7E]),
        finite(false, 5, [14])
    );
    assert_eq!(Layout::<4, Fnuz, 8>::decode([0x7F]), finite(false, 4, [15]));
    assert_eq!(Layout::<5, Fnuz, 8>::decode([0x7F]), finite(false, 13, [7]));
    // E4M3B11FNUZ: 30 = 15 * 2^1, and the smallest subnormal value is 2^-13.
    assert_eq!(
        Layout::<4, B11Fnuz, 8>::decode([0x7F]),
        finite(false, 1, [15])
    );
    assert_eq!(
        Layout::<4, B11Fnuz, 8>::decode([0x01]),
        finite(false, -13, [1])
    );
    assert_eq!(
        Layout::<4, B11Fnuz, 8>::decode([0x80]),
        Layout::<4, Fnuz, 8>::decode([0x80])
    );
    assert_eq!(
        Layout::<4, Fnuz, 8>::decode([0x80]),
        Unpacked::Nan {
            negative: true,
            signaling: false,
            payload: [0]
        }
    );
}

#[test]
fn fnuz_encodes_negative_zero_as_positive_zero() {
    assert_eq!(
        Layout::<4, Fnuz, 8>::encode::<[u64; 1]>(Unpacked::zero(true)),
        [0]
    );
    assert_eq!(
        Layout::<4, NoInf, 8>::encode::<[u64; 1]>(Unpacked::zero(true)),
        [0x80]
    );
}

fn x87(bits: u128) -> [u64; 2] {
    bits.to_limbs()
}

#[test]
fn x87_decodes_its_canonical_encodings() {
    assert_eq!(
        X87Layout::decode(x87(0x3FFF_8000_0000_0000_0000)),
        finite(false, -63, x87(1 << 63))
    );
    assert_eq!(
        X87Layout::decode(x87(0x0000_0000_0000_0000_0001)),
        finite(false, -16445, x87(1))
    );
    assert_eq!(
        X87Layout::decode(x87(0xFFFF_8000_0000_0000_0000)),
        Unpacked::Infinity { negative: true }
    );
    assert_eq!(
        X87Layout::decode(x87(0x7FFF_C000_0000_0000_0000)),
        Unpacked::Nan {
            negative: false,
            signaling: false,
            payload: x87(0)
        }
    );
    assert_eq!(
        X87Layout::decode(x87(0x7FFF_8000_0000_0000_0001)),
        Unpacked::Nan {
            negative: false,
            signaling: true,
            payload: x87(1)
        }
    );
    for bits in [
        0x3FFF_8000_0000_0000_0000,
        0x0000_0000_0000_0000_0001,
        0x8000_0000_0000_0000_0000,
        0xFFFF_8000_0000_0000_0000,
        0x7FFF_C000_0000_0000_1234,
        0x7FFF_8000_0000_0000_0001,
        0x7FFE_FFFF_FFFF_FFFF_FFFF,
        0x0001_8000_0000_0000_0000,
    ] {
        let limbs = x87(bits);
        assert!(X87Layout::is_canonical(limbs), "{bits:#x} is canonical");
        assert_eq!(
            X87Layout::encode(X87Layout::decode(limbs)),
            limbs,
            "{bits:#x} round trip"
        );
    }
}

#[test]
fn x87_pseudo_denormal_is_a_non_canonical_subnormal_encoding() {
    let pseudo = x87(0x0000_8000_0000_0000_0001);
    let normal = x87(0x0001_8000_0000_0000_0001);
    assert_eq!(X87Layout::decode(pseudo), X87Layout::decode(normal));
    assert_eq!(X87Layout::classify(pseudo), Class::Subnormal);
    assert_eq!(X87Layout::classify(normal), Class::Normal);
    assert!(!X87Layout::is_canonical(pseudo));
    assert_eq!(X87Layout::encode(X87Layout::decode(pseudo)), normal);
}

#[test]
fn x87_unnormals_and_pseudo_specials_are_unsupported() {
    for bits in [
        0x3FFF_0000_0000_0000_0001, // unnormal
        0x0001_0000_0000_0000_0000, // unnormal with a zero significand
        0x7FFF_0000_0000_0000_0000, // pseudo-infinity
        0xFFFF_4000_0000_0000_0000, // pseudo-NaN, quiet bit set
        0x7FFF_0000_0000_0000_0001, // pseudo-NaN, quiet bit clear
    ] {
        let limbs = x87(bits);
        assert_eq!(X87Layout::decode(limbs), Unpacked::Unsupported, "{bits:#x}");
        assert_eq!(X87Layout::classify(limbs), Class::Unsupported, "{bits:#x}");
        assert!(!X87Layout::is_canonical(limbs), "{bits:#x}");
        assert_eq!(X87Layout::normal(limbs), None, "{bits:#x}");
    }
}

#[test]
fn x87_normal_operands_read_directly() {
    // 1.0, the largest finite value, and the smallest normal value read
    // directly.
    for bits in [
        0x3FFF_8000_0000_0000_0000_u128,
        0x7FFE_FFFF_FFFF_FFFF_FFFF,
        0x8001_8000_0000_0000_0000,
    ] {
        let limbs = x87(bits);
        assert_eq!(
            X87Layout::normal(limbs),
            X87Layout::decode(limbs).number(),
            "{bits:#x}"
        );
    }
    // A pseudo-denormal, a denormal, the infinity, and a quiet NaN.
    for bits in [
        0x0000_8000_0000_0000_0001_u128,
        0x0000_0000_0000_0000_0001,
        0x7FFF_8000_0000_0000_0000,
        0x7FFF_C000_0000_0000_0000,
    ] {
        assert_eq!(X87Layout::normal(x87(bits)), None, "{bits:#x}");
    }
}

#[test]
fn binary512_decodes_its_boundary_encodings() {
    type L = Layout<23, Ieee, 512>;
    const FIELD_ALL_ONES: u64 = 0x7F_FFFF;
    let zero = <[u64; 8]>::ZERO;
    let infinity = zero.with_field(488, 23, FIELD_ALL_ONES);
    let quiet_nan = infinity.with_bit(487);
    let signaling_nan = infinity.with_bit(0);
    let largest = <[u64; 8]>::ones(488).with_field(488, 23, FIELD_ALL_ONES - 1);
    let smallest_normal = zero.with_bit(511).with_field(488, 23, 1);
    let smallest_subnormal = zero.with_bit(0);
    let cases = [
        (zero, Class::Zero, Unpacked::zero(false)),
        (
            infinity,
            Class::Infinite,
            Unpacked::Infinity { negative: false },
        ),
        (
            quiet_nan,
            Class::QuietNan,
            Unpacked::Nan {
                negative: false,
                signaling: false,
                payload: zero,
            },
        ),
        (
            signaling_nan,
            Class::SignalingNan,
            Unpacked::Nan {
                negative: false,
                signaling: true,
                payload: zero.with_bit(0),
            },
        ),
        (
            largest,
            Class::Normal,
            finite(false, 4_194_303 - 488, <[u64; 8]>::ones(489)),
        ),
        (
            smallest_normal,
            Class::Normal,
            finite(true, -4_194_302 - 488, zero.with_bit(488)),
        ),
        (
            smallest_subnormal,
            Class::Subnormal,
            finite(false, -4_194_302 - 488, zero.with_bit(0)),
        ),
    ];
    for (limbs, class, value) in cases {
        assert_eq!(L::classify(limbs), class, "{limbs:x?}");
        assert_eq!(L::decode(limbs), value, "{limbs:x?}");
        assert_eq!(L::encode(value), limbs, "{limbs:x?}");
    }
}
