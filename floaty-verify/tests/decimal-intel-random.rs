//! Compares floaty's operations on `D32Bid`, `D64Bid`, and `D128Bid` with the
//! Intel Decimal Floating-Point Math Library 2.0 Update 2 on seeded random
//! operands, in the five rounding directions of the library.
//!
//! [`Layout::random`] draws each first operand: zeros, one-digit and full
//! coefficients, both ends of the exponent range, subnormal values,
//! non-canonical coefficients, infinities, and NaNs with canonical and
//! non-canonical payloads. [`Layout::related`] draws a second operand near
//! the first one: the same exponent, a close exponent, a member of its
//! cohort, half a unit in its last place, its negation, or itself. The
//! comparisons also draw two operands that compare equal with
//! [`Layout::equal_pair`]. The square roots draw exact squares with
//! [`Layout::square`], and the conversions to integers draw operands near
//! the bounds of each integer type with [`Layout::near_integer`].
//!
//! The test compares the result bits and the flags by the rules of
//! `decimal-intel.rs`, and these:
//!
//! - `fma(x, y, z)` runs as `y.mul_add_with(x, z)`. A case with NaN `x` and
//!   `z` and a number `y` is counted and skipped, because floaty cannot take
//!   the NaN order of the library there.
//! - Where `bid32_fma` or `bid64_fma` differs from the exact result of the
//!   next wider format, the wider result decides; see [`fused`].
//! - Where `bid*_quantum` is wrong, the IEEE 754 definition decides; see
//!   [`quantum`].
//! - Where `minnum` or `maxnum` has operands that compare equal, the library
//!   decides the value and the flags, and floaty's rule decides which
//!   operand the result is; see [`min_max`].
//!
//! The report counts the cases that each rule decides. The generators are
//! seeded, so each test asserts the counts exactly.

// The references of this test build only for x86-64.
#![cfg(target_arch = "x86_64")]

use core::cmp::Ordering;
use std::collections::BTreeMap;

use floaty::format::{Bid, Decimal, Standard, Storage, Width};
use floaty::{Env, Flags, Float, Integer, ToInt};
use floaty_verify::intel_decimal::{
    self, Bid32, Bid64, Bid128, Class as IntelClass, Extremum, Flags as IntelFlags, Format,
    Inexact, Integer as IntelInteger, Layout, Outcome, Predicate, Rounding as IntelRounding,
};
use floaty_verify::random::SplitMix64;

/// The storage of a format of width `W`.
type Bits<const W: usize> = <Width<W> as Storage>::Bits;

/// The BID format of width `W`.
type BidFloat<const W: usize> = Float<Decimal<Bid>, W>;

/// The flags that a library function reports, as floaty's flags map to them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Signals {
    /// The five IEEE flags.
    Ieee,
    /// The IEEE flags without inexact, which the function does not signal.
    NoInexact,
}

impl Signals {
    /// Returns the library flags of floaty's flags. The library sets its
    /// denormal flag only in a conversion from a binary format.
    fn map(self, flags: Flags) -> IntelFlags {
        let dropped = match self {
            Self::Ieee => Flags::DENORMAL_INPUT,
            Self::NoInexact => Flags::DENORMAL_INPUT | Flags::INEXACT,
        };
        IntelFlags::from_floaty(flags.difference(dropped))
    }
}

/// A result: an encoding or an integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Value {
    /// An encoding.
    Bits(u128),
    /// An integer, or a predicate as 0 or 1.
    Integer(i128),
}

/// The failures of one operation: their count and the first few.
#[derive(Default)]
struct Failures {
    count: usize,
    examples: Vec<String>,
}

/// The counts of a run.
#[derive(Default)]
struct Report {
    cases: BTreeMap<String, usize>,
    failures: BTreeMap<String, Failures>,
    skipped: BTreeMap<&'static str, usize>,
    /// How many outcomes of the library raise each flag, by bit.
    flags: [usize; 6],
}

impl Report {
    /// The number of failures that the report prints for each operation.
    const EXAMPLES: usize = 4;

    /// Compares a floaty outcome with the outcome of the library.
    fn check(
        &mut self,
        operation: &str,
        operands: &dyn Fn() -> String,
        ours: Outcome<Value>,
        theirs: Outcome<Value>,
    ) {
        *self.cases.entry(operation.to_owned()).or_default() += 1;
        for (bit, count) in self.flags.iter_mut().enumerate() {
            *count += usize::from(theirs.flags.bits() >> bit & 1 == 1);
        }
        if ours == theirs {
            return;
        }
        let failures = self.failures.entry(operation.to_owned()).or_default();
        failures.count += 1;
        if failures.examples.len() < Self::EXAMPLES {
            failures.examples.push(format!(
                "{operation} {}: floaty {:x?} {:#04x}, library {:x?} {:#04x}",
                operands(),
                ours.value,
                ours.flags.bits(),
                theirs.value,
                theirs.flags.bits(),
            ));
        }
    }

    /// Prints the counts and the failures, and fails on a failure.
    fn finish(&self, title: &str) {
        let cases: usize = self.cases.values().sum();
        let failed: usize = self.failures.values().map(|failures| failures.count).sum();
        println!("{title}: {cases} cases, {failed} failed");
        let names = [
            "invalid",
            "denormal",
            "zero-divide",
            "overflow",
            "underflow",
            "inexact",
        ];
        let flags: Vec<String> = names
            .iter()
            .zip(self.flags)
            .map(|(name, count)| format!("{name} {count}"))
            .collect();
        println!("  library flags: {}", flags.join(", "));
        for (operation, count) in &self.cases {
            println!("  cases   {count:7}: {operation}");
        }
        for (reason, count) in &self.skipped {
            println!("  skipped {count:7}: {reason}");
        }
        for (operation, failures) in &self.failures {
            println!("  FAILED  {:7}: {operation}", failures.count);
            for example in &failures.examples {
                println!("    {example}");
            }
        }
        assert_eq!(failed, 0, "{title}: floaty differs from the library");
    }

    /// Asserts the number of cases, the cases that each rule decides, as
    /// `(operation, rule, count)`, and the skipped cases. The generators are
    /// seeded, so the counts are exact.
    fn assert_counts(
        &self,
        cases: usize,
        rules: &[(&str, &str, usize)],
        skipped: &[(&'static str, usize)],
    ) {
        assert_eq!(self.cases.values().sum::<usize>(), cases, "the cases");
        let decided: BTreeMap<String, usize> = self
            .cases
            .iter()
            .filter(|(operation, _)| operation.contains(" by "))
            .map(|(operation, &count)| (operation.clone(), count))
            .collect();
        let expected: BTreeMap<String, usize> = rules
            .iter()
            .map(|&(operation, rule, count)| (format!("{operation} by {rule}"), count))
            .collect();
        assert_eq!(decided, expected, "the cases that each rule decides");
        let skipped = BTreeMap::from_iter(skipped.iter().copied());
        assert_eq!(self.skipped, skipped, "the skipped cases");
    }
}

/// Returns a random value below `bound`.
fn below(rng: &mut SplitMix64, bound: u64) -> u64 {
    rng.next_u64() % bound
}

/// Returns a random rounding direction of the library.
fn rounding(rng: &mut SplitMix64) -> IntelRounding {
    let index = usize::try_from(below(rng, 5)).expect("an index below 5 fits a usize");
    IntelRounding::ALL[index]
}

/// Converts floaty's result and flags to an outcome of the library.
fn ours<const W: usize>((value, flags): (BidFloat<W>, Flags), signals: Signals) -> Outcome<Value>
where
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    Outcome {
        value: Value::Bits(value.to_bits().into()),
        flags: signals.map(flags),
    }
}

/// Converts a result of the library.
fn theirs<B: Into<u128>>(outcome: Outcome<B>) -> Outcome<Value> {
    outcome.map(|bits| Value::Bits(bits.into()))
}

/// Converts a value and flags that are not an encoding.
fn plain(value: i128, flags: Flags, signals: Signals) -> Outcome<Value> {
    Outcome {
        value: Value::Integer(value),
        flags: signals.map(flags),
    }
}

/// Returns an encoding of format `F` from its bits.
fn narrow<F: Format>(bits: u128) -> F::Bits {
    F::Bits::try_from(bits)
        .ok()
        .expect("an encoding of the format fits its storage")
}

/// Returns whether an encoding is a NaN.
fn is_nan<F: Format>(bits: F::Bits) -> bool {
    matches!(
        F::class(bits),
        IntelClass::QuietNan | IntelClass::SignalingNan
    )
}

/// Returns an addend for `x * y`: the negated product, which cancels, a
/// value near `x`, or a random value.
fn addend<F: Format>(rng: &mut SplitMix64, x: u128, y: u128) -> u128 {
    let layout = F::LAYOUT;
    match below(rng, 3) {
        0 => {
            let product = F::mul(narrow::<F>(x), narrow::<F>(y), rounding(rng)).value;
            F::negate(product).into()
        }
        1 => layout.related(rng, x),
        _ => layout.random(rng),
    }
}

/// Compares the arithmetic of one format: `add`, `sub`, `mul`, `div`,
/// `fma`, and `sqrt`.
fn arithmetic<F, const W: usize>(report: &mut Report, rng: &mut SplitMix64, cases: usize)
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let layout = F::LAYOUT;
    let name = |operation: &str| format!("{}_{operation}", F::NAME);
    for _ in 0..cases {
        let rounding = rounding(rng);
        let env = intel_decimal::env(rounding);
        let x = layout.random(rng);
        let y = layout.related(rng, x);
        let z = addend::<F>(rng, x, y);
        let (xb, yb, zb) = (narrow::<F>(x), narrow::<F>(y), narrow::<F>(z));
        let (xf, yf, zf) = (
            BidFloat::<W>::from_bits(xb),
            BidFloat::<W>::from_bits(yb),
            BidFloat::<W>::from_bits(zb),
        );
        let operands = || format!("{x:#x} {y:#x} {z:#x} {rounding:?}");
        let ieee = Signals::Ieee;
        report.check(
            &name("add"),
            &operands,
            ours(xf.add_with(yf, env), ieee),
            theirs(F::add(xb, yb, rounding)),
        );
        report.check(
            &name("sub"),
            &operands,
            ours(xf.sub_with(yf, env), ieee),
            theirs(F::sub(xb, yb, rounding)),
        );
        report.check(
            &name("mul"),
            &operands,
            ours(xf.mul_with(yf, env), ieee),
            theirs(F::mul(xb, yb, rounding)),
        );
        report.check(
            &name("div"),
            &operands,
            ours(xf.div_with(yf, env), ieee),
            theirs(F::div(xb, yb, rounding)),
        );
        // The library takes the NaN of `y`, then `z`, then `x`, so floaty
        // runs `y * x + z`. The orders differ only for NaN `x` and `z`.
        if is_nan::<F>(xb) && !is_nan::<F>(yb) && is_nan::<F>(zb) {
            *report.skipped.entry(FMA_NAN_ORDER).or_default() += 1;
        } else {
            let (library, rule) = fused::<F>(x, y, z, rounding);
            let operation = match rule {
                Some(rule) => format!("{} by {rule}", name("fma")),
                None => name("fma"),
            };
            report.check(
                &operation,
                &operands,
                ours(yf.mul_add_with(xf, zf, env), ieee),
                library,
            );
        }
        let root = layout.square(rng);
        let rootb = narrow::<F>(root);
        report.check(
            &name("sqrt"),
            &|| format!("{root:#x} {rounding:?}"),
            ours(BidFloat::<W>::from_bits(rootb).sqrt_with(env), ieee),
            theirs(F::sqrt(rootb, rounding)),
        );
    }
}

/// The rule for an `fma` case where the library differs from its own exact
/// result in the next wider format.
const FMA_WIDER: &str = "the exact fma of the next wider format of the library";

/// Returns the expected `fma` outcome of the library, and the rule that
/// gives it when the function of the format is wrong.
///
/// `bid32_fma` and `bid64_fma` differ from one rounding of the exact result
/// in some tiny cases: `bid32_fma` signals underflow and inexact for an
/// exact zero, and `bid64_fma` rounds a tiny negative result to `+0`. The
/// operands convert exactly to the next wider format. When `fma` there is
/// exact, the conversion back rounds the exact result once, which is the
/// correct result. The test uses it when it differs from the function of
/// the format.
fn fused<F: Format>(
    x: u128,
    y: u128,
    z: u128,
    rounding: IntelRounding,
) -> (Outcome<Value>, Option<&'static str>) {
    let direct = theirs(F::fma(
        narrow::<F>(x),
        narrow::<F>(y),
        narrow::<F>(z),
        rounding,
    ));
    let wider = match F::LAYOUT.width {
        32 => {
            let widen = |v: u128| intel_decimal::bid32_to_bid64(narrow::<Bid32>(v));
            let (x, y, z) = (widen(x), widen(y), widen(z));
            let product = Bid64::fma(x.value, y.value, z.value, rounding);
            let back = intel_decimal::bid64_to_bid32(product.value, rounding);
            (x.flags | y.flags | z.flags, product.flags, theirs(back))
        }
        64 => {
            let widen = |v: u128| intel_decimal::bid64_to_bid128(narrow::<Bid64>(v));
            let (x, y, z) = (widen(x), widen(y), widen(z));
            let product = Bid128::fma(x.value, y.value, z.value, rounding);
            let back = intel_decimal::bid128_to_bid64(product.value, rounding);
            (x.flags | y.flags | z.flags, product.flags, theirs(back))
        }
        _ => return (direct, None),
    };
    let (widened, product, back) = wider;
    if product.contains(IntelFlags::INEXACT) {
        return (direct, None);
    }
    let wider = Outcome {
        value: back.value,
        flags: widened | product | back.flags,
    };
    if wider == direct {
        (direct, None)
    } else {
        (wider, Some(FMA_WIDER))
    }
}

/// The reason for an `fma` case whose NaN order floaty cannot follow.
const FMA_NAN_ORDER: &str = "fma with NaN x and z and a number y: the library returns the NaN \
     of z, floaty the NaN of the factors first (FusedNanOrder::ProductFirst, SoftFloat's order)";

/// Returns a scale for `scalbn`: small, across the exponent range, or
/// extreme.
fn scale(rng: &mut SplitMix64, layout: Layout) -> i32 {
    let range = i32::try_from(layout.largest_field() + layout.precision() + 2)
        .expect("the range fits an i32");
    let within = |rng: &mut SplitMix64, bound: i32| {
        let width = u64::from(bound.unsigned_abs()) * 2 + 1;
        i32::try_from(below(rng, width)).expect("the offset fits an i32") - bound
    };
    match below(rng, 6) {
        0 => within(rng, 20),
        1 => [i32::MIN, i32::MAX, 1 << 30, -(1 << 30), range, -range]
            [usize::try_from(below(rng, 6)).expect("an index fits a usize")],
        _ => within(rng, range),
    }
}

/// Compares the rounding and quantum operations of one format: `quantize`,
/// `rem`, rounding to an integral value, `scalbn`, `nextup`, `nextdown`,
/// `logb`, and `quantum`.
fn quantum_operations<F, const W: usize>(report: &mut Report, rng: &mut SplitMix64, cases: usize)
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let layout = F::LAYOUT;
    let name = |operation: &str| format!("{}_{operation}", F::NAME);
    for _ in 0..cases {
        let rounding = rounding(rng);
        let env = intel_decimal::env(rounding);
        let x = layout.random(rng);
        let y = layout.related(rng, x);
        let n = scale(rng, layout);
        let (xb, yb) = (narrow::<F>(x), narrow::<F>(y));
        let (xf, yf) = (BidFloat::<W>::from_bits(xb), BidFloat::<W>::from_bits(yb));
        let operands = || format!("{x:#x} {y:#x} {n} {rounding:?}");
        let (ieee, quiet) = (Signals::Ieee, Signals::NoInexact);
        let fixed = |direction: IntelRounding| env.with_rounding(direction.into());
        let checks = [
            (
                "quantize",
                ours(xf.quantize_with(yf, env), ieee),
                theirs(F::quantize(xb, yb, rounding)),
            ),
            (
                "rem",
                ours(xf.remainder_with(yf, env), ieee),
                theirs(F::rem(xb, yb)),
            ),
            (
                "round_integral_exact",
                ours(xf.round_to_integral_with(env), ieee),
                theirs(F::round_integral_exact(xb, rounding)),
            ),
            (
                "nearbyint",
                ours(xf.round_to_integral_with(env), quiet),
                theirs(F::nearbyint(xb, rounding)),
            ),
            (
                "round_integral",
                ours(xf.round_to_integral_with(fixed(rounding)), quiet),
                theirs(F::round_integral(xb, rounding)),
            ),
            (
                "scalbn",
                ours(xf.scale_b_with(n, env), ieee),
                theirs(F::scalbn(xb, n, rounding)),
            ),
            (
                "nextup",
                ours(xf.next_up_with(env), ieee),
                theirs(F::nextup(xb)),
            ),
            (
                "nextdown",
                ours(xf.next_down_with(env), ieee),
                theirs(F::nextdown(xb)),
            ),
            ("logb", ours(xf.log_b_with(env), ieee), theirs(F::logb(xb))),
        ];
        for (operation, floaty, library) in checks {
            report.check(&name(operation), &operands, floaty, library);
        }
        let (quantum, rule) = quantum::<F>(x);
        let operation = match rule {
            Some(rule) => format!("{} by {rule}", name("quantum")),
            None => name("quantum"),
        };
        report.check(
            &operation,
            &operands,
            ours(xf.quantum_with(env), ieee),
            quantum,
        );
    }
}

/// Returns the expected `quantum` of `x`, and the rule that gives it when
/// the library is wrong.
///
/// IEEE 754-2019 section 5.3.2 defines `quantum(x)` as `1 * 10^q` for the
/// exponent `q` of a finite `x`, and `+inf` for an infinite `x`. A NaN gives
/// a NaN by section 6.2: quiet, and invalid for a signaling NaN. The library
/// source `bid*_quantumd.c` differs:
///
/// - `bid32_quantum` tests the implicit `100` form with the 64-bit mask
///   `MASK_STEERING_BITS`, so it reads the exponent of that form from the
///   wrong bits.
/// - `bid32_quantum` and `bid64_quantum` return `|x|` for a NaN or an
///   infinity: a signaling NaN stays signaling without invalid, and the
///   extra bits of a non-canonical encoding stay.
/// - `bid128_quantum` of a NaN sets only the high 64 bits of its result.
///
/// For those operands the test evaluates the definition. The NaN result is
/// the one `nextup` of the library gives, which follows section 6.2.
fn quantum<F: Format>(x: u128) -> (Outcome<Value>, Option<&'static str>) {
    let layout = F::LAYOUT;
    let xb = narrow::<F>(x);
    let library = theirs(F::quantum(xb));
    let number = |value: u128| Outcome {
        value: Value::Bits(value),
        flags: IntelFlags::NONE,
    };
    let (definition, rule) = match layout.fields(x) {
        None if is_nan::<F>(xb) => (theirs(F::nextup(xb)), QUANTUM_NAN),
        None => (number(layout.infinity(false)), QUANTUM_INFINITY),
        Some((field, _)) if layout.width == 32 && (x >> 29) & 0b11 == 0b11 => {
            (number(layout.number(false, field, 1)), QUANTUM_FORM)
        }
        Some(_) => return (library, None),
    };
    if definition == library {
        (library, None)
    } else {
        (definition, Some(rule))
    }
}

/// The rule for `quantum` of a NaN.
const QUANTUM_NAN: &str = "IEEE 754 for a NaN: the library returns |x| or a partial NaN";
/// The rule for `quantum` of an infinity.
const QUANTUM_INFINITY: &str = "IEEE 754 for an infinity: the library keeps extra bits";
/// The rule for `quantum` of a decimal32 coefficient in the `100` form.
const QUANTUM_FORM: &str = "IEEE 754 for the 100 form: the library reads the wrong exponent bits";

/// Compares the comparisons, the total order, `minnum`, `maxnum`, the
/// class, and the sign operations of one format.
fn comparisons<F, const W: usize>(report: &mut Report, rng: &mut SplitMix64, cases: usize)
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let layout = F::LAYOUT;
    let name = |operation: &str| format!("{}_{operation}", F::NAME);
    let env = intel_decimal::env(IntelRounding::TiesToEven);
    let flag = |holds: bool| plain(i128::from(holds), Flags::NONE, Signals::Ieee);
    let library = |holds: bool| Outcome {
        value: Value::Integer(i128::from(holds)),
        flags: IntelFlags::NONE,
    };
    for _ in 0..cases {
        let (x, y) = if below(rng, 4) == 0 {
            layout.equal_pair(rng)
        } else {
            let x = layout.random(rng);
            (x, layout.related(rng, x))
        };
        let (xb, yb) = (narrow::<F>(x), narrow::<F>(y));
        let (xf, yf) = (BidFloat::<W>::from_bits(xb), BidFloat::<W>::from_bits(yb));
        let operands = || format!("{x:#x} {y:#x}");
        for predicate in Predicate::ALL {
            let (order, flags) = if predicate.is_signaling() {
                xf.compare_signaling_with(yf, env)
            } else {
                xf.compare_quiet_with(yf, env)
            };
            let floaty = plain(i128::from(predicate.holds(order)), flags, Signals::Ieee);
            let theirs = F::compare(xb, yb, predicate).map(|holds| Value::Integer(holds.into()));
            report.check(&name(predicate.name()), &operands, floaty, theirs);
        }
        let total = |a: BidFloat<W>, b: BidFloat<W>| a.total_cmp(b) != Ordering::Greater;
        report.check(
            &name("totalOrder"),
            &operands,
            flag(total(xf, yf)),
            library(F::total_order(xb, yb)),
        );
        report.check(
            &name("totalOrderMag"),
            &operands,
            flag(total(xf.abs(), yf.abs())),
            library(F::total_order_mag(xb, yb)),
        );
        report.check(
            &name("sameQuantum"),
            &operands,
            flag(xf.same_quantum(yf)),
            library(F::same_quantum(xb, yb)),
        );
        let extrema = [
            ("minnum", Extremum::Minimum, xf.min_num_with(yf, env)),
            ("maxnum", Extremum::Maximum, xf.max_num_with(yf, env)),
        ];
        for (operation, extremum, floaty) in extrema {
            let (expected, rule) = min_max::<F>(xb, yb, extremum);
            let operation = match rule {
                Some(rule) => format!("{} by {rule}", name(operation)),
                None => name(operation),
            };
            report.check(&operation, &operands, ours(floaty, Signals::Ieee), expected);
        }
        let class = IntelClass::from_floaty(xf.classify(), xf.is_sign_negative())
            .expect("a decimal value has a class of the library");
        let properties = [
            ("class", class.code(), F::class(xb).code()),
            (
                "isCanonical",
                i32::from(xf.is_canonical()),
                i32::from(F::is_canonical(xb)),
            ),
        ];
        for (operation, floaty, library) in properties {
            report.check(
                &name(operation),
                &operands,
                plain(i128::from(floaty), Flags::NONE, Signals::Ieee),
                plain(i128::from(library), Flags::NONE, Signals::Ieee),
            );
        }
        let sign_operations = [
            ("abs", xf.abs(), F::abs(xb)),
            ("negate", -xf, F::negate(xb)),
            ("copySign", xf.copy_sign(yf), F::copy_sign(xb, yb)),
        ];
        for (operation, floaty, library) in sign_operations {
            report.check(
                &name(operation),
                &operands,
                ours((floaty, Flags::NONE), Signals::Ieee),
                theirs(Outcome {
                    value: library,
                    flags: IntelFlags::NONE,
                }),
            );
        }
    }
}

/// The rule for `minnum` and `maxnum` of operands that compare equal.
const EQUAL_OPERANDS: &str = "floaty's rule for operands that compare equal: -0 below +0, and \
     the members of a cohort in totalOrder";

/// Returns the expected `minnum` or `maxnum` outcome, and the rule that gives
/// it when the choice of the library differs.
///
/// The library is the oracle for the value and the flags. When the operands
/// compare equal and the library returns a value equal to them, IEEE 754
/// lets either operand be the result, and `readtest.c` accepts either. There
/// floaty's rule, with the order of the library's `totalOrder`, decides the
/// encoding; see [`intel_decimal::equal_operand_choice`].
fn min_max<F: Format>(
    x: F::Bits,
    y: F::Bits,
    extremum: Extremum,
) -> (Outcome<Value>, Option<&'static str>) {
    let library = match extremum {
        Extremum::Minimum => F::minnum(x, y),
        Extremum::Maximum => F::maxnum(x, y),
    };
    let equal = |a: F::Bits, b: F::Bits| F::compare(a, b, Predicate::QuietEqual).value;
    if !(equal(x, y) && equal(library.value, x)) {
        return (theirs(library), None);
    }
    let choice = intel_decimal::equal_operand_choice::<F>(x, y, extremum);
    if choice == library.value {
        return (theirs(library), None);
    }
    let decided = Outcome {
        value: choice,
        flags: library.flags,
    };
    (theirs(decided), Some(EQUAL_OPERANDS))
}

/// Converts to an integer of type `I`, and maps an invalid conversion to the
/// integer indefinite of `integer`.
fn to_integer<I, const W: usize>(x: BidFloat<W>, integer: IntelInteger, env: Env) -> (i128, Flags)
where
    I: Integer + Into<i128>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
{
    let (result, flags) = x.to_int_with::<I>(env);
    let value = match result {
        ToInt::Value(value) => value.into(),
        ToInt::OutOfRange { .. } | ToInt::Nan => integer.indefinite(),
    };
    (value, flags)
}

/// Compares the conversions to and from integers of one format.
fn integers<F, const W: usize>(report: &mut Report, rng: &mut SplitMix64, cases: usize)
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let layout = F::LAYOUT;
    for _ in 0..cases {
        for integer in IntelInteger::ALL {
            let x = layout.near_integer(rng, integer);
            let xb = narrow::<F>(x);
            let xf = BidFloat::<W>::from_bits(xb);
            for (direction, inexact) in IntelRounding::ALL.into_iter().flat_map(|direction| {
                [
                    (direction, Inexact::Ignored),
                    (direction, Inexact::Signaled),
                ]
            }) {
                let env = intel_decimal::env(direction);
                let (value, flags) = match integer {
                    IntelInteger::Int8 => to_integer::<i8, W>(xf, integer, env),
                    IntelInteger::Int16 => to_integer::<i16, W>(xf, integer, env),
                    IntelInteger::Int32 => to_integer::<i32, W>(xf, integer, env),
                    IntelInteger::Int64 => to_integer::<i64, W>(xf, integer, env),
                    IntelInteger::UInt8 => to_integer::<u8, W>(xf, integer, env),
                    IntelInteger::UInt16 => to_integer::<u16, W>(xf, integer, env),
                    IntelInteger::UInt32 => to_integer::<u32, W>(xf, integer, env),
                    IntelInteger::UInt64 => to_integer::<u64, W>(xf, integer, env),
                };
                let signals = match inexact {
                    Inexact::Ignored => Signals::NoInexact,
                    Inexact::Signaled => Signals::Ieee,
                };
                let library = F::to_integer(xb, integer, direction, inexact).map(Value::Integer);
                let operation = format!(
                    "{}_to_{}_{direction:?}_{inexact:?}",
                    F::NAME,
                    integer.name()
                );
                report.check(
                    &operation,
                    &|| format!("{x:#x}"),
                    plain(value, flags, signals),
                    library,
                );
            }
        }
        from_integers::<F, W>(report, rng);
    }
}

/// Compares the conversions from `i32`, `u32`, `i64`, and `u64`, with
/// boundary values, powers of 10 and their neighbors, and random values.
fn from_integers<F, const W: usize>(report: &mut Report, rng: &mut SplitMix64)
where
    F: Format<Bits = Bits<W>>,
    Width<W>: Storage,
    Decimal<Bid>: Standard<W, Bits = Bits<W>>,
    Bits<W>: Into<u128>,
{
    let rounding = rounding(rng);
    let env = intel_decimal::env(rounding);
    let power = 10_u64.pow(u32::try_from(below(rng, 20)).expect("an exponent fits a u32"));
    let raw = match below(rng, 5) {
        0 => power,
        1 => power - 1,
        2 => power + 1,
        3 => [
            0,
            1,
            u64::MAX,
            u64::MAX >> 1,
            1 << 63,
            u64::from(u32::MAX),
            1 << 31,
        ][usize::try_from(below(rng, 7)).expect("an index fits a usize")],
        _ => rng.next_u64() >> below(rng, 64),
    };
    let negative = below(rng, 2) == 0;
    let operands = || format!("{raw:#x} negative {negative} {rounding:?}");
    let signed = i64::try_from(raw).unwrap_or(i64::MAX);
    let signed = if negative {
        signed.wrapping_neg()
    } else {
        signed
    };
    let low = u32::try_from(raw & u64::from(u32::MAX)).expect("the mask leaves 32 bits");
    let low_signed = i32::try_from(signed.clamp(i32::MIN.into(), i32::MAX.into()))
        .expect("a clamped value fits an i32");
    let name = |operation: &str| format!("{}_{operation}", F::NAME);
    let checks = [
        (
            "from_int32",
            ours(BidFloat::<W>::from_int_with(low_signed, env), Signals::Ieee),
            theirs(F::from_int32(low_signed, rounding)),
        ),
        (
            "from_uint32",
            ours(BidFloat::<W>::from_int_with(low, env), Signals::Ieee),
            theirs(F::from_uint32(low, rounding)),
        ),
        (
            "from_int64",
            ours(BidFloat::<W>::from_int_with(signed, env), Signals::Ieee),
            theirs(F::from_int64(signed, rounding)),
        ),
        (
            "from_uint64",
            ours(BidFloat::<W>::from_int_with(raw, env), Signals::Ieee),
            theirs(F::from_uint64(raw, rounding)),
        ),
    ];
    for (operation, floaty, library) in checks {
        report.check(&name(operation), &operands, floaty, library);
    }
}

/// Runs `run` for each format with its own seeded generator, and returns the
/// report of a run without failures.
fn each_format(
    title: &str,
    seed: u64,
    cases: [usize; 3],
    run: [fn(&mut Report, &mut SplitMix64, usize); 3],
) -> Report {
    let mut report = Report::default();
    for (index, (run, cases)) in run.into_iter().zip(cases).enumerate() {
        let mut rng = SplitMix64::new(seed + u64::try_from(index).expect("an index fits a u64"));
        run(&mut report, &mut rng, cases);
    }
    report.finish(title);
    report
}

#[test]
fn arithmetic_matches_the_library() {
    let report = each_format(
        "add, sub, mul, div, fma, sqrt",
        0x7e57_0001,
        [120_000, 120_000, 60_000],
        [
            arithmetic::<Bid32, 32>,
            arithmetic::<Bid64, 64>,
            arithmetic::<Bid128, 128>,
        ],
    );
    report.assert_counts(
        1_794_599,
        &[("bid32_fma", FMA_WIDER, 10), ("bid64_fma", FMA_WIDER, 1)],
        &[(FMA_NAN_ORDER, 5_401)],
    );
}

#[test]
fn quantum_operations_match_the_library() {
    let report = each_format(
        "quantize, rem, round_integral, scalbn, nextup, nextdown, logb, quantum",
        0x7e57_0002,
        [100_000, 100_000, 50_000],
        [
            quantum_operations::<Bid32, 32>,
            quantum_operations::<Bid64, 64>,
            quantum_operations::<Bid128, 128>,
        ],
    );
    report.assert_counts(
        2_500_000,
        &[
            ("bid32_quantum", QUANTUM_NAN, 4_258),
            ("bid64_quantum", QUANTUM_NAN, 4_472),
            ("bid128_quantum", QUANTUM_NAN, 2_677),
            ("bid32_quantum", QUANTUM_INFINITY, 2_661),
            ("bid64_quantum", QUANTUM_INFINITY, 2_663),
            ("bid32_quantum", QUANTUM_FORM, 16_992),
        ],
        &[],
    );
}

#[test]
fn comparisons_match_the_library() {
    let report = each_format(
        "comparisons, total order, minnum, maxnum, class, sign operations",
        0x7e57_0003,
        [40_000, 40_000, 40_000],
        [
            comparisons::<Bid32, 32>,
            comparisons::<Bid64, 64>,
            comparisons::<Bid128, 128>,
        ],
    );
    report.assert_counts(
        3_600_000,
        &[
            ("bid32_minnum", EQUAL_OPERANDS, 4_129),
            ("bid64_minnum", EQUAL_OPERANDS, 4_340),
            ("bid128_minnum", EQUAL_OPERANDS, 6_148),
            ("bid32_maxnum", EQUAL_OPERANDS, 4_073),
            ("bid64_maxnum", EQUAL_OPERANDS, 4_387),
            ("bid128_maxnum", EQUAL_OPERANDS, 6_246),
        ],
        &[],
    );
}

#[test]
fn integer_conversions_match_the_library() {
    let report = each_format(
        "conversions to and from integers",
        0x7e57_0004,
        [20_000, 20_000, 20_000],
        [
            integers::<Bid32, 32>,
            integers::<Bid64, 64>,
            integers::<Bid128, 128>,
        ],
    );
    report.assert_counts(5_040_000, &[], &[]);
}
