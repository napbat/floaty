//! Checks double-double payloads against glibc and the exact value from MPFR.
//! Checks nominal precision against glibc's step and QD's numeric limits.

// The C references build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::env::Mode;
use floaty::{Algorithm, DoubleDouble, F64, Gcc, Qd, mode};
use floaty_verify::double_double::{self, Exact};
use floaty_verify::ibm_ldouble::{self, Function, FunctionCase};
use floaty_verify::libm::{self, Format, Payload};
use floaty_verify::qd;
use floaty_verify::random::SplitMix64;
use ibm_ldouble::{Flags, Pair, Rounding};

const PAYLOAD_MASK: u64 = (1 << 51) - 1;
const ONE: u64 = 0x3FF0_0000_0000_0000;

type Flushing = mode::DenormalsAreZero<
    mode::FlushToZero<
        mode::Rounded<mode::X86Sse, mode::direction::TowardPositive>,
        mode::switch::On,
    >,
    mode::switch::On,
>;

fn apply<Alg: Algorithm, M: Mode>(pair: Pair, operation: Payload) -> Pair {
    let value =
        DoubleDouble::<Alg, M>::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo));
    let result = match operation {
        Payload::Get => value.payload(),
        Payload::Set => DoubleDouble::from_payload(value),
        Payload::SetSignaling => DoubleDouble::from_payload_signaling(value),
    };
    Pair::new(result.hi().to_bits(), result.lo().to_bits())
}

fn scalar(bits: u64, operation: Payload) -> Pair {
    let result = libm::payload(Format::Double, operation, u128::from(bits));
    Pair::new(
        u64::try_from(result).expect("a binary64 encoding fits u64"),
        0,
    )
}

fn expected(pair: Pair, operation: Payload) -> Pair {
    let value = double_double::exact(pair);
    match operation {
        Payload::Get => match value {
            Exact::Nan(bits) => scalar(bits, operation),
            _ => scalar(0, operation),
        },
        Payload::Set | Payload::SetSignaling => match value {
            Exact::Zero { negative: false } => scalar(0, operation),
            Exact::Number(number)
                if number.is_integer() && !number.is_sign_negative() && number <= PAYLOAD_MASK =>
            {
                scalar(number.to_f64().to_bits(), operation)
            }
            _ => Pair::default(),
        },
    }
}

fn operands() -> Vec<Pair> {
    let edges = [
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x000F_FFFF_FFFF_FFFF,
        0x0010_0000_0000_0000,
        0x3FE0_0000_0000_0000,
        ONE,
        0xBFF0_0000_0000_0000,
        0x4000_0000_0000_0000,
        0x431F_FFFF_FFFF_FFFC,
        0x4320_0000_0000_0000,
        0x7FEF_FFFF_FFFF_FFFF,
        0x7FF0_0000_0000_0000,
        0xFFF0_0000_0000_0000,
        0x7FF8_0000_0000_0000,
        0xFFF8_0000_0000_0001,
        0x7FFF_FFFF_FFFF_FFFF,
        0x7FF0_0000_0000_0001,
        0xFFF7_FFFF_FFFF_FFFF,
    ];
    let mut pairs: Vec<_> = edges
        .into_iter()
        .flat_map(|hi| edges.map(|lo| Pair::new(hi, lo)))
        .collect();
    let mut random = SplitMix64::new(0xDD09_0704);
    pairs.extend((0..4_000).map(|_| double_double::pair(&mut random)));
    for _ in 0..1_000 {
        let integer = random.next_u64() & PAYLOAD_MASK;
        let high = F64::from_int(integer).to_bits();
        pairs.extend([
            Pair::new(high, 0),
            Pair::new(high, 1),
            Pair::new(high, 0x8000_0000_0000_0001),
        ]);
    }
    pairs
}

#[test]
fn payloads_read_exact_values_in_every_mode() {
    // Conflict: glibc 2.43's IBM setters reject a nonzero low half, and its
    // getter reads only the high half. See ldbl-128ibm/s_setpayloadl_main.c,
    // lines 40 to 47, and s_getpayloadl.c, lines 29 to 34. Resolution: glibc
    // checks canonical pairs below. MPFR checks floaty's exact-value rule here.
    for pair in operands() {
        for operation in Payload::ALL {
            let reference = expected(pair, operation);
            assert_eq!(
                apply::<Gcc, mode::Libgcc>(pair, operation),
                reference,
                "Gcc {operation:?} {pair:?}"
            );
            assert_eq!(
                apply::<Qd, mode::X86Sse>(pair, operation),
                reference,
                "Qd {operation:?} {pair:?}"
            );
            assert_eq!(
                apply::<Gcc, Flushing>(pair, operation),
                reference,
                "Gcc flushing {operation:?} {pair:?}"
            );
            assert_eq!(
                apply::<Qd, Flushing>(pair, operation),
                reference,
                "Qd flushing {operation:?} {pair:?}"
            );
        }
    }
}

#[test]
fn canonical_payloads_match_ibm_glibc_without_flags() {
    let pairs: Vec<_> = operands()
        .into_iter()
        .filter(|pair| {
            DoubleDouble::<Gcc>::from_parts(F64::from_bits(pair.hi), F64::from_bits(pair.lo))
                .is_canonical()
        })
        .collect();
    let mut cases = Vec::new();
    for pair in pairs {
        for rounding in Rounding::ALL {
            for function in [
                Function::GetPayload,
                Function::SetPayload,
                Function::SetPayloadSignaling,
            ] {
                cases.push(FunctionCase {
                    function,
                    rounding,
                    operands: [pair; 3],
                });
            }
        }
    }
    for (case, reference) in cases.iter().zip(ibm_ldouble::run_functions(&cases)) {
        let operation = match case.function {
            Function::GetPayload => Payload::Get,
            Function::SetPayload => Payload::Set,
            Function::SetPayloadSignaling => Payload::SetSignaling,
            _ => unreachable!("the cases contain only payload functions"),
        };
        assert_eq!(
            apply::<Gcc, mode::Libgcc>(case.operands[0], operation),
            reference.result,
            "{case:?}"
        );
        assert_eq!(reference.flags, Flags::NONE, "{case:?}");
    }
}

#[test]
fn metadata_matches_reference_precision_and_radix() {
    assert_eq!(DoubleDouble::<Qd>::RADIX, qd::radix());
    assert_eq!(DoubleDouble::<Qd>::PRECISION, qd::precision());

    let one = Pair::new(ONE, 0);
    let cases = [
        FunctionCase {
            function: Function::NextUp,
            rounding: Rounding::TiesToEven,
            operands: [one; 3],
        },
        FunctionCase {
            function: Function::ScaleB,
            rounding: Rounding::TiesToEven,
            operands: [one, Pair::new(1, 0), one],
        },
    ];
    let reference = ibm_ldouble::run_functions(&cases);
    let precision = i32::try_from(DoubleDouble::<Gcc>::PRECISION).expect("the precision fits i32");
    let step = F64::from_bits(ONE).scale_b(1 - precision);
    assert_eq!(reference[0].result, Pair::new(ONE, step.to_bits()));
    let radix = F64::from_int(DoubleDouble::<Gcc>::RADIX);
    assert_eq!(reference[1].result, Pair::new(radix.to_bits(), 0));
}
