//! Compares the `_with` methods of the slice kernels of `Lanes` with their
//! documented order, which the MPFR arithmetic oracle computes one step at a
//! time, in every behavior of `BEHAVIORS`: the widening of each value, the
//! term, the addition into lane `i % N`, and the sum of the lanes by halves.
//! The result must have the value, the NaN payload, and the flags of the
//! last step, and the union of the flags of every step.
//!
//! The vectors are those of `floaty_verify::kernels`: binary32, bfloat16,
//! and binary16 values, a mixed pair, and integer codes with and without a
//! scale. The lane counts are 1, 2, 8, and 32.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use floaty::env::NanPropagation;
use floaty::format::Standard;
use floaty::{BF16, Binary, Decoded, Env, F16, F32, Flags, Float, Lanes, ScaledCodes, Vector};
use floaty_verify::arithmetic::{self, Operation};
use floaty_verify::encodings::{Layout, boundary_encodings_u128};
use floaty_verify::kernels::{LENGTHS, Mix, codes, pair, vector};
use floaty_verify::mpfr::{self, Format, Operand, Specials, Value};
use floaty_verify::operations::BEHAVIORS;
use floaty_verify::random::SplitMix64;
use rug::Float as BigFloat;

/// The binary32 format of the lanes.
const SINGLE: Format = Format::of::<F32>(Specials::Ieee);

/// The payload bits of a binary32 NaN, below its quiet bit.
const SINGLE_PAYLOAD_BITS: u32 = 22;

/// A value of a vector, as the oracle reads it.
#[derive(Clone, Debug)]
enum Input {
    /// A binary value of `precision` bits, which widens to binary32.
    Float { operand: Operand<1>, precision: u32 },
    /// An integer code, which converts exactly.
    Code(i32),
    /// A signed code times a binary32 scale.
    Scaled { code: i32, scale: Operand<1> },
}

/// Returns the values of a slice as inputs.
fn inputs<S: Standard<W>, const W: usize>(values: &[Float<S, W>]) -> Vec<Input> {
    values
        .iter()
        .map(|&value| Input::Float {
            operand: Operand::of(value),
            precision: S::PRECISION,
        })
        .collect()
}

/// A rounded result: its value and the payload of a NaN.
#[derive(Clone, Debug, PartialEq)]
struct Rounded {
    value: Value,
    payload: [u64; 1],
}

impl Rounded {
    /// +0, the value of a lane before its first term.
    fn zero() -> Self {
        Self {
            value: Value::Zero { negative: false },
            payload: [0],
        }
    }

    /// Returns the result as an operand of the next step.
    fn operand(&self) -> Operand<1> {
        let decoded = match &self.value {
            Value::Zero { negative } => Decoded::Zero {
                negative: *negative,
                exponent: 0,
            },
            Value::Infinity { negative } => Decoded::Infinity {
                negative: *negative,
            },
            Value::Nan { negative } => Decoded::Nan {
                negative: *negative,
                signaling: false,
                payload: self.payload,
            },
            Value::Finite(value) => {
                let (significand, exponent) = value.to_integer_exp().expect("the value is finite");
                let significand = significand.abs();
                let top =
                    i32::try_from(significand.significant_bits()).expect("24 bits") + exponent - 1;
                return Operand {
                    decoded: Decoded::Finite {
                        negative: value.is_sign_negative(),
                        exponent,
                        significand: [significand.to_u64().expect("a binary32 significand")],
                    },
                    subnormal: top < SINGLE.emin,
                };
            }
        };
        Operand {
            decoded,
            subnormal: false,
        }
    }
}

/// The oracle of one kernel call: the behavior, and the union of the flags
/// of the steps so far.
struct Oracle {
    env: Env,
    flags: Flags,
}

impl Oracle {
    /// Returns the result of one arithmetic step.
    fn run(&mut self, operation: Operation, operands: &[Operand<1>]) -> Rounded {
        let expected = arithmetic::compute(operation, operands, &SINGLE, &self.env);
        self.flags |= expected.flags;
        Rounded {
            value: expected.value,
            payload: expected.payload,
        }
    }

    /// Returns an input as a binary32 operand, after its conversion.
    fn value(&mut self, input: &Input) -> Operand<1> {
        match input {
            Input::Float {
                operand,
                precision: 24,
            } => *operand,
            Input::Float { operand, precision } => {
                let (value, flags) = mpfr::convert(operand, &SINGLE, &self.env);
                self.flags |= flags;
                // A NaN keeps the high-order bits of its payload, as
                // `convert_with` documents, unless the rule gives the
                // default NaN.
                let payload = match operand.decoded {
                    Decoded::Nan { payload, .. }
                        if self.env.nan.propagation != NanPropagation::DefaultNan =>
                    {
                        [payload[0] << (SINGLE_PAYLOAD_BITS + 2 - precision)]
                    }
                    _ => [0],
                };
                Rounded { value, payload }.operand()
            }
            Input::Code(code) => self.code(*code),
            Input::Scaled { code, scale } => {
                let code = self.code(*code);
                self.run(Operation::Mul, &[code, *scale]).operand()
            }
        }
    }

    /// Returns an integer code converted to binary32 in the behavior: exact
    /// at the full precision, and rounded under a precision limit.
    fn code(&mut self, code: i32) -> Operand<1> {
        let (value, flags) = mpfr::convert(&code_operand(code), &SINGLE, &self.env);
        self.flags |= flags;
        Rounded {
            value,
            payload: [0],
        }
        .operand()
    }
}

/// Returns an integer code as an exact operand.
fn code_operand(code: i32) -> Operand<1> {
    let value = if code == 0 {
        Value::Zero { negative: false }
    } else {
        Value::Finite(BigFloat::with_val(32, code))
    };
    Rounded {
        value,
        payload: [0],
    }
    .operand()
}

/// A kernel of `Lanes`.
#[derive(Clone, Copy, Debug)]
enum Kernel {
    Sum,
    Dot { fused: bool },
    Distance { fused: bool },
    Norm { fused: bool },
}

impl Kernel {
    /// Every kernel.
    const ALL: [Self; 7] = [
        Self::Sum,
        Self::Dot { fused: false },
        Self::Dot { fused: true },
        Self::Distance { fused: false },
        Self::Distance { fused: true },
        Self::Norm { fused: false },
        Self::Norm { fused: true },
    ];

    /// Returns the result of the kernel with `lanes` lanes in the documented
    /// order, and the flags.
    fn expected(self, x: &[Input], y: &[Input], lanes: usize, env: &Env) -> (Rounded, Flags) {
        let mut oracle = Oracle {
            env: *env,
            flags: Flags::NONE,
        };
        let mut sums = vec![Rounded::zero(); lanes];
        let y = match self {
            Self::Norm { .. } => x,
            _ => y,
        };
        for (index, (a, b)) in x.iter().zip(y).enumerate() {
            let a = oracle.value(a);
            let lane = sums[index % lanes].operand();
            let (left, right, fused) = match self {
                Self::Sum => {
                    sums[index % lanes] = oracle.run(Operation::Add, &[lane, a]);
                    continue;
                }
                Self::Dot { fused } | Self::Norm { fused } => (a, oracle.value(b), fused),
                Self::Distance { fused } => {
                    let b = oracle.value(b);
                    let difference = oracle.run(Operation::Sub, &[a, b]).operand();
                    (difference, difference, fused)
                }
            };
            sums[index % lanes] = if fused {
                oracle.run(Operation::MulAdd, &[left, right, lane])
            } else {
                let product = oracle.run(Operation::Mul, &[left, right]).operand();
                oracle.run(Operation::Add, &[lane, product])
            };
        }
        let mut half = lanes / 2;
        while half > 0 {
            for index in 0..half {
                let operands = [sums[index].operand(), sums[index + half].operand()];
                sums[index] = oracle.run(Operation::Add, &operands);
            }
            half /= 2;
        }
        let mut sum = sums.swap_remove(0);
        if let Self::Norm { .. } = self {
            sum = oracle.run(Operation::Sqrt, &[sum.operand()]);
        }
        (sum, oracle.flags)
    }

    /// Returns the result of the `_with` method of the kernel with `N` lanes,
    /// and the flags.
    fn ours<const N: usize>(self, x: impl Vector, y: impl Vector, env: Env) -> (F32, Flags) {
        type Single<const N: usize> = Lanes<F32, N>;
        match self {
            Self::Sum => Single::<N>::sum_with(x, env),
            Self::Dot { fused: false } => Single::<N>::dot_with(x, y, env),
            Self::Dot { fused: true } => Single::<N>::dot_fused_with(x, y, env),
            Self::Distance { fused: false } => Single::<N>::distance_square_with(x, y, env),
            Self::Distance { fused: true } => Single::<N>::distance_square_fused_with(x, y, env),
            Self::Norm { fused: false } => Single::<N>::norm_with(x, env),
            Self::Norm { fused: true } => Single::<N>::norm_fused_with(x, env),
        }
    }
}

/// Returns a result of floaty in the form of the oracle. A NaN result must
/// be quiet.
fn rounded(result: F32) -> Rounded {
    let decoded = result.decode::<1>();
    let payload = match decoded {
        Decoded::Nan {
            signaling, payload, ..
        } => {
            assert!(!signaling, "{result:?}: a NaN result is quiet");
            payload
        }
        _ => [0],
    };
    Rounded {
        value: Value::from_decoded(decoded),
        payload,
    }
}

/// Checks every kernel with `N` lanes on `x` and `y` in every behavior.
fn check<const N: usize>(
    x: impl Vector,
    y: impl Vector,
    xi: &[Input],
    yi: &[Input],
    context: &str,
) {
    for env in BEHAVIORS {
        for kernel in Kernel::ALL {
            let (result, flags) = kernel.ours::<N>(x, y, env);
            let expected = kernel.expected(xi, yi, N, &env);
            assert_eq!(
                (rounded(result), flags),
                expected,
                "{kernel:?} N = {N} {context} {env:?}"
            );
        }
    }
}

/// Checks every kernel in each lane count on `x` and `y`.
fn check_lanes(x: impl Vector, y: impl Vector, xi: &[Input], yi: &[Input], context: &str) {
    check::<1>(x, y, xi, yi, context);
    check::<2>(x, y, xi, yi, context);
    check::<8>(x, y, xi, yi, context);
    check::<32>(x, y, xi, yi, context);
}

/// Returns encodings as values of a format with the storage `T`.
fn values<S: Standard<W, Bits = T>, const W: usize, T: TryFrom<u128>>(
    encodings: &[u128],
) -> Vec<Float<S, W>> {
    encodings
        .iter()
        .map(|&bits| Float::from_bits(T::try_from(bits).ok().expect("the encoding fits")))
        .collect()
}

#[test]
fn binary32_vectors_follow_the_order() {
    let mut random = SplitMix64::new(0x6F72_6465);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let (a, b) = pair(Layout::BINARY32, mix, length, &mut random);
            let x: Vec<F32> = values::<Binary<8>, 32, u32>(&a);
            let y: Vec<F32> = values::<Binary<8>, 32, u32>(&b);
            let context = format!("{mix:?} {length}");
            check_lanes(&x[..], &y[..], &inputs(&x), &inputs(&y), &context);
        }
    }
}

#[test]
fn narrow_vectors_follow_the_order() {
    let mut random = SplitMix64::new(0x6E61_7272);
    for mix in Mix::ALL {
        for length in LENGTHS {
            let context = format!("{mix:?} {length}");
            let (a, b) = pair(Layout::BFLOAT16, mix, length, &mut random);
            let x: Vec<BF16> = values::<Binary<8>, 16, u16>(&a);
            let y: Vec<BF16> = values::<Binary<8>, 16, u16>(&b);
            check_lanes(&x[..], &y[..], &inputs(&x), &inputs(&y), &context);
            // A mixed pair: binary16 against binary32.
            let x: Vec<F16> =
                values::<Binary<5>, 16, u16>(&vector(Layout::BINARY16, mix, length, &mut random));
            let y: Vec<F32> =
                values::<Binary<8>, 32, u32>(&vector(Layout::BINARY32, mix, length, &mut random));
            check_lanes(
                &x[..],
                &y[..],
                &inputs(&x),
                &inputs(&y),
                &format!("{context} mixed"),
            );
        }
    }
}

#[test]
fn codes_follow_the_order() {
    let mut random = SplitMix64::new(0x7363_616C);
    let boundary: Vec<F32> =
        values::<Binary<8>, 32, u32>(&boundary_encodings_u128(Layout::BINARY32));
    let mut boundary_scales = boundary.iter().cycle();
    for mix in Mix::ALL {
        for length in LENGTHS {
            let context = format!("{mix:?} {length}");
            let y: Vec<F32> =
                values::<Binary<8>, 32, u32>(&vector(Layout::BINARY32, mix, length, &mut random));
            let unsigned = codes(length, &mut random);
            let unsigned_inputs: Vec<Input> = unsigned
                .iter()
                .map(|&code| Input::Code(i32::from(code)))
                .collect();
            check_lanes(
                &unsigned[..],
                &y[..],
                &unsigned_inputs,
                &inputs(&y),
                &context,
            );
            let signed: Vec<i8> = unsigned.iter().map(|&code| code.cast_signed()).collect();
            let boundary_scale = *boundary_scales.next().expect("the cycle never ends");
            for scale in [boundary_scale, F32::from_bits(0x3C00_0000)] {
                let scaled_inputs: Vec<Input> = signed
                    .iter()
                    .map(|&code| Input::Scaled {
                        code: i32::from(code),
                        scale: Operand::of(scale),
                    })
                    .collect();
                let scaled = ScaledCodes {
                    codes: &signed,
                    scale,
                };
                let context = format!("{context} scale {scale:?}");
                check_lanes(scaled, &y[..], &scaled_inputs, &inputs(&y), &context);
            }
        }
    }
}
