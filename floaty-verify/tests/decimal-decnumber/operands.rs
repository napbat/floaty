//! Seeded random operands of decNumber's fixed-size formats, biased toward
//! the edges of a format.

use std::marker::PhantomData;

use floaty_verify::decnumber::{self, Arithmetic, Binary, Format, Unary};
use floaty_verify::dectest::Operation;
use floaty_verify::random::SplitMix64;

/// A finite number: `(-1)^negative * coefficient * 10^exponent`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Number {
    pub(super) negative: bool,
    pub(super) coefficient: u128,
    pub(super) exponent: i32,
}

impl Number {
    /// Returns the number as a string that decNumber reads.
    fn text(self) -> String {
        let sign = if self.negative { "-" } else { "" };
        format!("{sign}{}E{}", self.coefficient, self.exponent)
    }

    /// Reads a scientific string that decNumber writes, such as `-1.20E+5`
    /// or `0.00`: the coefficient is the digits without the point, and the
    /// exponent is that of the last digit. Returns `None` for an infinity
    /// or a NaN.
    pub(super) fn parse(text: &str) -> Option<Self> {
        let (negative, magnitude) = match text.strip_prefix('-') {
            Some(magnitude) => (true, magnitude),
            None => (false, text),
        };
        if !magnitude.starts_with(|first: char| first.is_ascii_digit()) {
            return None;
        }
        let (mantissa, exponent) = magnitude.split_once('E').unwrap_or((magnitude, "0"));
        let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let exponent: i32 = exponent
            .parse()
            .expect("decNumber writes an integer exponent");
        let fraction_digits = i32::try_from(fraction.len()).expect("a digit count fits an i32");
        Some(Self {
            negative,
            coefficient: format!("{whole}{fraction}")
                .parse()
                .expect("decNumber writes at most 34 digits"),
            exponent: exponent - fraction_digits,
        })
    }
}

/// Returns `10^power`.
pub(super) fn power_of_ten(power: u32) -> u128 {
    10_u128.pow(power)
}

/// Returns the digit count of a coefficient. Zero has one digit.
pub(super) fn digit_count(coefficient: u128) -> u32 {
    coefficient.checked_ilog10().map_or(1, |log| log + 1)
}

/// Returns a value as an `i32`, for the digit counts and exponents here.
pub(super) fn signed(value: u32) -> i32 {
    i32::try_from(value).expect("a digit count fits an i32")
}

/// The coefficient length and the exponent range whose edges a generator
/// probes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    /// The most digits of a coefficient.
    pub(super) digits: u32,
    /// The precision whose subnormal range the exponents probe.
    pub(super) precision: u32,
    /// The smallest adjusted exponent of a normal value.
    pub(super) emin: i32,
    /// The largest adjusted exponent.
    pub(super) emax: i32,
}

impl Shape {
    /// Returns the shape of a format.
    pub(super) fn of<F: Format>() -> Self {
        Self {
            digits: F::PRECISION,
            precision: F::PRECISION,
            emin: F::EMIN,
            emax: F::EMAX,
        }
    }
}

/// Seeded random operands of format `F`, biased toward the edges of a
/// shape: extreme coefficients and exponents, subnormal values, zeros of
/// both signs, infinities, NaNs with payloads, and random encodings, which
/// include non-canonical ones.
pub(super) struct Generator<F: Format> {
    random: SplitMix64,
    shape: Shape,
    format: PhantomData<F>,
}

impl<F: Format> Generator<F> {
    /// Returns a generator with a seed, which probes the edges of a shape.
    pub(super) fn new(seed: u64, shape: Shape) -> Self {
        Self {
            random: SplitMix64::new(seed),
            shape,
            format: PhantomData,
        }
    }

    /// Returns the least and the greatest exponent of an encoding of `F`.
    pub(super) fn exponents() -> (i32, i32) {
        let precision = signed(F::PRECISION);
        (F::EMIN - precision + 1, F::EMAX - precision + 1)
    }

    /// Returns a random value below `bound`.
    pub(super) fn below(&mut self, bound: u64) -> u64 {
        self.random.next_u64() % bound
    }

    /// Returns `true` with a chance of `percent` in 100.
    pub(super) fn chance(&mut self, percent: u64) -> bool {
        self.below(100) < percent
    }

    /// Returns a random value from `low` to `high`, both included.
    pub(super) fn between(&mut self, low: i32, high: i32) -> i32 {
        let span = u64::try_from(i64::from(high) - i64::from(low) + 1)
            .expect("the caller passes low <= high");
        let offset = i64::try_from(self.below(span)).expect("an offset fits an i64");
        i32::try_from(i64::from(low) + offset).expect("the value is between two i32 values")
    }

    /// Returns a random digit count from 1 to the shape's digits.
    fn length(&mut self) -> u32 {
        u32::try_from(self.between(1, signed(self.shape.digits))).expect("a length is positive")
    }

    /// Returns a random coefficient of exactly `length` digits.
    pub(super) fn full(&mut self, length: u32) -> u128 {
        let low = power_of_ten(length - 1);
        low + self.random.next_u128() % (power_of_ten(length) - low)
    }

    /// Returns a coefficient of at most the shape's digits, biased toward
    /// its edges, trailing zeros, and a last digit of 5.
    fn coefficient(&mut self) -> u128 {
        let digits = self.shape.digits;
        let length = self.length();
        match self.below(12) {
            0 => 1,
            1 => 5,
            2 => power_of_ten(digits - 1),
            3 => power_of_ten(digits) - 1,
            4 => 5 * power_of_ten(digits - 1),
            5 => power_of_ten(length - 1),
            6 => power_of_ten(length) - 1,
            7 => self.full(digits),
            8 => self.full(length) * power_of_ten(digits - length),
            9 => self.full(length) / 10 * 10 + 5,
            _ => self.full(length),
        }
    }

    /// Returns an exponent for a coefficient of `length` digits. The
    /// adjusted exponent is biased toward the edges of the shape, and the
    /// exponent is clamped into the encodings of `F`.
    pub(super) fn exponent(&mut self, length: u32) -> i32 {
        let shape = self.shape;
        let precision = signed(shape.precision);
        let tiny = shape.emin - precision + 1;
        let adjusted = match self.below(12) {
            0 => shape.emax,
            1 => shape.emax - self.between(1, 3),
            2 => shape.emin,
            3 => shape.emin - 1,
            4 => shape.emin - self.between(1, precision),
            5 => tiny - self.between(0, 1),
            6 => self.between(-3, 3),
            7 => self.between(-2 * precision, 2 * precision),
            _ => self.between(tiny, shape.emax),
        };
        let (lowest, highest) = Self::exponents();
        (adjusted - signed(length) + 1).clamp(lowest, highest)
    }

    /// Returns the encoding of a number that the format holds exactly.
    pub(super) fn encode(number: Number) -> F::Bits {
        let outcome = F::from_string(&number.text(), decnumber::Rounding::HalfEven);
        assert!(outcome.status.is_empty(), "{number:?} fits {}", F::NAME);
        outcome.value
    }

    /// Returns a nonzero number.
    pub(super) fn number(&mut self) -> Number {
        let coefficient = self.coefficient();
        Number {
            negative: self.chance(50),
            coefficient,
            exponent: self.exponent(digit_count(coefficient)),
        }
    }

    /// Returns a zero of either sign.
    pub(super) fn zero(&mut self) -> Number {
        let length = self.length();
        Number {
            negative: self.chance(50),
            coefficient: 0,
            exponent: self.exponent(length),
        }
    }

    /// Returns an infinity or a NaN of either sign. A NaN has no payload,
    /// the largest payload, or a random one.
    pub(super) fn special(&mut self) -> F::Bits {
        let sign = if self.chance(50) { "-" } else { "" };
        let kind = match self.below(3) {
            0 => return Self::from_text(&format!("{sign}Infinity")),
            1 => "sNaN",
            _ => "NaN",
        };
        let payload_digits = F::PRECISION - 1;
        let payload = match self.below(4) {
            0 => String::new(),
            1 => (power_of_ten(payload_digits) - 1).to_string(),
            _ => (self.random.next_u128()
                % power_of_ten(self.between(1, signed(payload_digits)).unsigned_abs()))
            .to_string(),
        };
        Self::from_text(&format!("{sign}{kind}{payload}"))
    }

    /// Returns the encoding of a string, rounded to nearest even.
    pub(super) fn from_text(text: &str) -> F::Bits {
        F::from_string(text, decnumber::Rounding::HalfEven).value
    }

    /// Returns random bits: any encoding, often a non-canonical one.
    pub(super) fn raw(&mut self) -> F::Bits {
        let width = 8 * u32::try_from(size_of::<F::Bits>()).expect("a width fits a u32");
        let bits = self.random.next_u128() & (u128::MAX >> (128 - width));
        let Ok(bits) = F::Bits::try_from(bits) else {
            unreachable!("the mask keeps the bits of the format");
        };
        bits
    }

    /// Returns an operand, and its value when it is a nonzero number.
    pub(super) fn operand(&mut self) -> (F::Bits, Option<Number>) {
        match self.below(20) {
            0..=12 => {
                let number = self.number();
                (Self::encode(number), Some(number))
            }
            13..=15 => (Self::encode(self.zero()), None),
            16 | 17 => (self.special(), None),
            _ => (self.raw(), None),
        }
    }

    /// Returns an operand related to the number `x`: its negation, `x`
    /// itself, another member of its cohort, a number with a nearby
    /// exponent, a neighbor of its coefficient, or half a unit in the last
    /// place of `x` at the precision of `F`, which makes a sum an exact
    /// tie.
    pub(super) fn related(&mut self, x: Option<Number>) -> F::Bits {
        let Some(x) = x else {
            return self.operand().0;
        };
        let spare = F::PRECISION - digit_count(x.coefficient);
        let precision = signed(F::PRECISION);
        let (lowest, highest) = Self::exponents();
        let related = match self.below(12) {
            0 | 1 => Number {
                negative: !x.negative,
                ..x
            },
            2 => x,
            3 | 4 => {
                let shift = self.between(0, signed(spare));
                let exponent = (x.exponent - shift).max(lowest);
                let shift =
                    u32::try_from(x.exponent - exponent).expect("the shift is not negative");
                Number {
                    coefficient: x.coefficient * power_of_ten(shift),
                    exponent,
                    ..x
                }
            }
            5..=7 => {
                let coefficient = self.coefficient();
                Number {
                    negative: self.chance(50),
                    coefficient,
                    exponent: (x.exponent + self.between(-precision - 2, precision + 2))
                        .clamp(lowest, highest),
                }
            }
            8 => Number {
                coefficient: (x.coefficient + 1).min(power_of_ten(F::PRECISION) - 1),
                ..x
            },
            9 | 10 => Number {
                negative: self.chance(50),
                coefficient: 5,
                exponent: (x.exponent + signed(digit_count(x.coefficient)) - precision - 1)
                    .clamp(lowest, highest),
            },
            _ => return self.operand().0,
        };
        Self::encode(related)
    }
}

impl<F: Arithmetic> Generator<F> {
    /// Returns the operands of a random case of an operation. The second
    /// and third operands relate to the first, as each operation needs.
    pub(super) fn operands(&mut self, operation: &Operation) -> Vec<F::Bits> {
        let (x, number) = self.operand();
        match operation {
            Operation::Unary(Unary::ToIntegralExact) if self.chance(50) => {
                vec![self.near_integer()]
            }
            Operation::Binary(Binary::Divide) if self.chance(30) => self.exact_division(),
            Operation::Binary(Binary::Divide) if self.chance(30) => self.halving(),
            Operation::Binary(Binary::Quantize) => vec![x, self.quantum(number)],
            Operation::Binary(Binary::ScaleB) => vec![x, self.scale()],
            Operation::Binary(Binary::Remainder | Binary::RemainderNear) => {
                vec![x, self.divisor(number)]
            }
            Operation::Fma => {
                let y = if self.chance(25) {
                    self.unit()
                } else {
                    self.related(number)
                };
                vec![x, y, self.addend(x, y, number)]
            }
            Operation::Unary(_) | Operation::Class => vec![x],
            _ => vec![x, self.related(number)],
        }
    }

    /// Returns a number whose digits straddle the units digit, for
    /// rounding to an integral value. A third of them end in the digit 5
    /// right after the point: an exact tie.
    fn near_integer(&mut self) -> F::Bits {
        let precision = signed(F::PRECISION);
        let coefficient = self.coefficient();
        let (coefficient, exponent) = if self.chance(33) {
            (coefficient / 10 * 10 + 5, -1)
        } else {
            (coefficient, self.between(-precision - 1, 1))
        };
        Self::encode(Number {
            negative: self.chance(50),
            coefficient,
            exponent,
        })
    }

    /// Returns a dividend and a divisor whose quotient is exact when the
    /// dividend, the divisor times a small factor, fits the format.
    fn exact_division(&mut self) -> Vec<F::Bits> {
        const FACTORS: [u128; 11] = [1, 2, 3, 4, 5, 7, 8, 16, 25, 125, 1000];
        let (divisor, _) = self.operand();
        let index = self.below(11);
        let factor = Self::encode(Number {
            negative: self.chance(50),
            coefficient: FACTORS[usize::try_from(index).expect("an index fits a usize")],
            exponent: self.between(-3, 3),
        });
        let rounding = decnumber::Rounding::HalfEven;
        let dividend = F::binary(Binary::Multiply, divisor, factor, rounding).value;
        vec![dividend, divisor]
    }

    /// Returns an odd dividend of `p` digits and a divisor that is a power
    /// of 2 or of 5. The quotient then ends in the digit 5, often beyond
    /// the precision: an exact tie.
    fn halving(&mut self) -> Vec<F::Bits> {
        const DIVISORS: [u128; 8] = [2, 4, 8, 16, 32, 64, 25, 125];
        let coefficient = self.full(F::PRECISION) | 1;
        let dividend = Self::encode(Number {
            negative: self.chance(50),
            coefficient,
            exponent: self.exponent(F::PRECISION),
        });
        let index = self.below(8);
        let divisor = Self::encode(Number {
            negative: self.chance(50),
            coefficient: DIVISORS[usize::try_from(index).expect("an index fits a usize")],
            exponent: self.between(-3, 3),
        });
        vec![dividend, divisor]
    }

    /// Returns 1 or -1, as a member of its cohort with up to two trailing
    /// zeros. A fused multiply-add with this factor is a sum.
    fn unit(&mut self) -> F::Bits {
        let shift = self.between(0, 2);
        Self::encode(Number {
            negative: self.chance(50),
            coefficient: power_of_ten(shift.unsigned_abs()),
            exponent: -shift,
        })
    }

    /// Returns a second operand of `quantize`: a zero or a one with an
    /// exponent just above the exponent of `x`, which drops one to three
    /// digits, near it, or anywhere, or another operand.
    fn quantum(&mut self, x: Option<Number>) -> F::Bits {
        let precision = signed(F::PRECISION);
        let (lowest, highest) = Self::exponents();
        let exponent = match x {
            Some(x) if self.chance(25) => x.exponent + self.between(1, 3),
            Some(x) if self.chance(60) => x.exponent + self.between(-precision - 2, precision + 2),
            _ if self.chance(15) => return self.operand().0,
            _ => self.between(lowest, highest),
        };
        Self::encode(Number {
            negative: self.chance(50),
            coefficient: u128::from(self.chance(80)),
            exponent: exponent.clamp(lowest, highest),
        })
    }

    /// Returns a scale operand: mostly an integer with exponent 0, near 0,
    /// anywhere in decNumber's range, or at its limit of `2 * (emax + p)`.
    fn scale(&mut self) -> F::Bits {
        let limit = 2 * (F::EMAX + signed(F::PRECISION));
        let magnitude = match self.below(10) {
            0..=3 => self.between(0, 20),
            4 | 5 => self.between(0, limit),
            6 | 7 => limit + self.between(-2, 1),
            8 => F::EMAX + self.between(-2, 2),
            _ => return self.operand().0,
        };
        Self::encode(Number {
            negative: self.chance(50),
            coefficient: u128::from(magnitude.unsigned_abs()),
            exponent: 0,
        })
    }

    /// Returns a divisor for `remaindernear`. Its leading digit is at most
    /// a few digits above that of `x` and at most `p` digits below, so the
    /// integer quotient mostly fits `p` digits.
    fn divisor(&mut self, x: Option<Number>) -> F::Bits {
        let Some(x) = x.filter(|_| self.chance(80)) else {
            return self.related(x);
        };
        let precision = signed(F::PRECISION);
        let (lowest, highest) = Self::exponents();
        let coefficient = self.coefficient();
        let adjusted =
            x.exponent + signed(digit_count(x.coefficient)) - 1 - self.between(-2, precision);
        Self::encode(Number {
            negative: self.chance(50),
            coefficient,
            exponent: (adjusted - signed(digit_count(coefficient)) + 1).clamp(lowest, highest),
        })
    }

    /// Returns an addend for `x * y + z`: the rounded product negated,
    /// which cancels the product to its rounding error, a neighbor of it,
    /// or an operand related to `x`.
    fn addend(&mut self, x: F::Bits, y: F::Bits, number: Option<Number>) -> F::Bits {
        let rounding = decnumber::Rounding::HalfEven;
        let product = F::binary(Binary::Multiply, x, y, rounding).value;
        let negated = F::unary(Unary::CopyNegate, product, rounding).value;
        match self.below(10) {
            0..=2 => negated,
            3 => F::unary(Unary::NextPlus, negated, rounding).value,
            4 => F::unary(Unary::NextMinus, negated, rounding).value,
            _ => self.related(number),
        }
    }
}
