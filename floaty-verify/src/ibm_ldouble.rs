//! Runs the IBM `long double` arithmetic of libgcc on PowerPC, the IBM
//! `long double` functions of glibc's libm, and the PowerPC fused
//! multiply-add instructions, as an oracle.
//!
//! The build script builds the batch program `shim/ibm_ldouble.c` with the
//! pinned powerpc64le GCC 15.2.0, and links it statically with the libgcc of
//! that compiler and the libm of its glibc 2.43. The program calls
//! `__gcc_qadd`, `__gcc_qsub`, `__gcc_qmul`, and `__gcc_qdiv`, the libm
//! functions of [`Function`], and executes `fmadd` and `fmsub`, under
//! `qemu-ppc64le`. [`run`], [`run_functions`], and [`run_instructions`] send
//! a batch of cases to one QEMU process. A process takes about 10 ms to
//! start, and then runs about 500,000 cases a second.
//!
//! The module works on raw bits: a [`Pair`] holds the binary64 encodings of
//! the high and the low half. The program sets the rounding direction with
//! `fesetround`, clears the flags, calls the routine, and reads the flags
//! with `fetestexcept`. QEMU computes the flags of the FPSCR with the rules
//! of its PowerPC target. PowerPC has no denormal flag.
//!
//! [`Pair`], [`Rounding`], [`Flags`], and [`Outcome`] are the types of
//! [`crate::double_double::reference`], which [`crate::qd`] also uses, so one test can
//! compare both references.

use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};
use std::thread;

use floaty::env::{FusedNanOrder, InvalidProduct, NanPropagation, NanRule, Tininess};

pub use crate::double_double::reference::{Flags, Outcome, Pair, Rounding};

/// The batch program, a static powerpc64le executable.
pub const PROGRAM: &str = env!("FLOATY_IBM_LDOUBLE");

/// The emulator that runs [`PROGRAM`], which the build script checks.
pub const QEMU: &str = env!("FLOATY_QEMU");

/// The pinned QEMU release. The build script checks it when it builds
/// [`PROGRAM`].
const QEMU_VERSION: &str = env!("FLOATY_QEMU_VERSION");

/// An arithmetic operation of libgcc.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    /// `__gcc_qadd`.
    Add,
    /// `__gcc_qsub`.
    Sub,
    /// `__gcc_qmul`.
    Mul,
    /// `__gcc_qdiv`.
    Div,
}

impl Operation {
    /// Every operation.
    pub const ALL: [Self; 4] = [Self::Add, Self::Sub, Self::Mul, Self::Div];

    /// Returns the name of the operation in the input of [`PROGRAM`].
    const fn word(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }
}

/// An IBM `long double` function of glibc's libm.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Function {
    /// `sqrtl`.
    Sqrt,
    /// `nextupl`.
    NextUp,
    /// `nextdownl`.
    NextDown,
    /// `iscanonicall`. The program gives its result, 1 or 0, as the high
    /// half of the result.
    IsCanonical,
    /// `logbl`: the exponent of the value.
    LogB,
    /// `copysignl`: the first operand with the sign of the second.
    CopySign,
    /// `fmodl`: the remainder of the truncated quotient.
    Fmod,
    /// `remainderl`: the IEEE 754 remainder.
    Remainder,
    /// `fmal`: `a * b + c`.
    MulAdd,
    /// `floorl`: the value rounded toward negative infinity to an integral
    /// value.
    Floor,
    /// `ceill`: the value rounded toward positive infinity to an integral
    /// value.
    Ceil,
    /// `truncl`: the value rounded toward zero to an integral value.
    Trunc,
    /// `roundl`: the value rounded to the nearest integral value, ties away
    /// from zero.
    Round,
    /// `roundevenl`: the value rounded to the nearest integral value, ties to
    /// even.
    RoundEven,
    /// `rintl`: the value rounded to an integral value in the rounding
    /// direction, with the inexact flag.
    Rint,
    /// `nearbyintl`: the value rounded to an integral value in the rounding
    /// direction, without the inexact flag.
    NearbyInt,
    /// `scalbnl`: the first operand times 2 to the power of the integer that
    /// the bits of the high half of the second operand hold.
    ScaleB,
    /// `ilogbl`: the exponent of the value as an `int`. The program gives the
    /// bits of the integer as the high half.
    ILogB,
    /// `getpayloadl`: the NaN payload as an integral value, or -1.
    GetPayload,
    /// `setpayloadl`: the quiet NaN with an integral payload, or +0.
    SetPayload,
    /// `setpayloadsigl`: the signaling NaN with an integral payload, or +0.
    SetPayloadSignaling,
    /// `fmaxl`: the IEEE 754-2008 `maxNum`.
    Fmax,
    /// `fminl`: the IEEE 754-2008 `minNum`.
    Fmin,
    /// `fmaximuml`: the IEEE 754-2019 `maximum`.
    Fmaximum,
    /// `fminimuml`: the IEEE 754-2019 `minimum`.
    Fminimum,
    /// `fmaximum_numl`: the IEEE 754-2019 `maximumNumber`.
    FmaximumNumber,
    /// `fminimum_numl`: the IEEE 754-2019 `minimumNumber`.
    FminimumNumber,
    /// `fmaximum_magl`: the IEEE 754-2019 `maximumMagnitude`.
    FmaximumMagnitude,
    /// `fminimum_magl`: the IEEE 754-2019 `minimumMagnitude`.
    FminimumMagnitude,
    /// `fmaximum_mag_numl`: the IEEE 754-2019 `maximumMagnitudeNumber`.
    FmaximumMagnitudeNumber,
    /// `fminimum_mag_numl`: the IEEE 754-2019 `minimumMagnitudeNumber`.
    FminimumMagnitudeNumber,
    /// `totalorderl`: 1 when the first operand orders at or below the second
    /// in the IEEE 754 total order, and 0 otherwise. The program gives the
    /// bits of the integer as the high half.
    TotalOrder,
    /// `totalordermagl`: `totalorderl` of the magnitudes.
    TotalOrderMagnitude,
    /// `llrintl`: the value rounded to a `long long` in the rounding
    /// direction. The program gives the bits of the integer as the high half.
    LlRint,
    /// `lroundl`: the value rounded to a `long`, ties away from zero. The
    /// program gives the bits of the integer as the high half.
    LRound,
}

impl Function {
    /// Returns the name of the function in the input of [`PROGRAM`].
    const fn word(self) -> &'static str {
        match self {
            Self::Sqrt => "sqrtl",
            Self::NextUp => "nextupl",
            Self::NextDown => "nextdownl",
            Self::IsCanonical => "iscanonicall",
            Self::LogB => "logbl",
            Self::CopySign => "copysignl",
            Self::Fmod => "fmodl",
            Self::Remainder => "remainderl",
            Self::MulAdd => "fmal",
            Self::Floor => "floorl",
            Self::Ceil => "ceill",
            Self::Trunc => "truncl",
            Self::Round => "roundl",
            Self::RoundEven => "roundevenl",
            Self::Rint => "rintl",
            Self::NearbyInt => "nearbyintl",
            Self::ScaleB => "scalbnl",
            Self::ILogB => "ilogbl",
            Self::GetPayload => "getpayloadl",
            Self::SetPayload => "setpayloadl",
            Self::SetPayloadSignaling => "setpayloadsigl",
            Self::Fmax => "fmaxl",
            Self::Fmin => "fminl",
            Self::Fmaximum => "fmaximuml",
            Self::Fminimum => "fminimuml",
            Self::FmaximumNumber => "fmaximum_numl",
            Self::FminimumNumber => "fminimum_numl",
            Self::FmaximumMagnitude => "fmaximum_magl",
            Self::FminimumMagnitude => "fminimum_magl",
            Self::FmaximumMagnitudeNumber => "fmaximum_mag_numl",
            Self::FminimumMagnitudeNumber => "fminimum_mag_numl",
            Self::TotalOrder => "totalorderl",
            Self::TotalOrderMagnitude => "totalordermagl",
            Self::LlRint => "llrintl",
            Self::LRound => "lroundl",
        }
    }

    /// Returns the number of operands of the function.
    #[must_use]
    pub const fn operand_count(self) -> usize {
        match self {
            Self::Sqrt
            | Self::NextUp
            | Self::NextDown
            | Self::IsCanonical
            | Self::LogB
            | Self::Floor
            | Self::Ceil
            | Self::Trunc
            | Self::Round
            | Self::RoundEven
            | Self::Rint
            | Self::NearbyInt
            | Self::ILogB
            | Self::GetPayload
            | Self::SetPayload
            | Self::SetPayloadSignaling
            | Self::LlRint
            | Self::LRound => 1,
            Self::Fmod
            | Self::Remainder
            | Self::CopySign
            | Self::ScaleB
            | Self::Fmax
            | Self::Fmin
            | Self::Fmaximum
            | Self::Fminimum
            | Self::FmaximumNumber
            | Self::FminimumNumber
            | Self::FmaximumMagnitude
            | Self::FminimumMagnitude
            | Self::FmaximumMagnitudeNumber
            | Self::FminimumMagnitudeNumber
            | Self::TotalOrder
            | Self::TotalOrderMagnitude => 2,
            Self::MulAdd => 3,
        }
    }
}

/// A fused multiply-add instruction of PowerPC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// `fmadd`: `a * c + b`.
    MultiplyAdd,
    /// `fmsub`: `a * c - b`.
    MultiplySubtract,
}

impl Instruction {
    /// Every instruction.
    pub const ALL: [Self; 2] = [Self::MultiplyAdd, Self::MultiplySubtract];

    /// Returns the name of the instruction in the input of [`PROGRAM`].
    const fn word(self) -> &'static str {
        match self {
            Self::MultiplyAdd => "fmadd",
            Self::MultiplySubtract => "fmsub",
        }
    }
}

/// Returns the name of a rounding direction in the input of [`PROGRAM`].
const fn rounding_word(rounding: Rounding) -> &'static str {
    match rounding {
        Rounding::TiesToEven => "nearest",
        Rounding::TowardZero => "zero",
        Rounding::TowardPositive => "up",
        Rounding::TowardNegative => "down",
    }
}

/// Returns the behavior of QEMU's PowerPC target in a rounding direction.
///
/// - The `FirstOperand` NaN rule, and a positive default NaN.
/// - A fused multiply-add takes its NaN in the order first factor, addend,
///   second factor, and signals invalid for `0 * inf + NaN` but returns the
///   NaN addend.
/// - Tininess before rounding.
#[must_use]
pub fn behavior(rounding: Rounding) -> floaty::Env {
    floaty::Env::IEEE
        .with_rounding(rounding.into())
        .with_tininess(Tininess::BeforeRounding)
        .with_nan(
            NanRule::new(NanPropagation::FirstOperand)
                .with_fused_order(FusedNanOrder::AddendSecond)
                .with_invalid_product(InvalidProduct::SignalsAndYieldsToNan),
        )
}

/// One operation on two operands in one rounding direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Case {
    /// The operation.
    pub operation: Operation,
    /// The rounding direction.
    pub rounding: Rounding,
    /// The first operand.
    pub a: Pair,
    /// The second operand.
    pub b: Pair,
}

/// One libm function on its operands in one rounding direction. The function
/// reads the first [`Function::operand_count`] operands.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionCase {
    /// The function.
    pub function: Function,
    /// The rounding direction.
    pub rounding: Rounding,
    /// The operands.
    pub operands: [Pair; 3],
}

/// One fused multiply-add instruction on three binary64 operands in one
/// rounding direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstructionCase {
    /// The instruction.
    pub instruction: Instruction,
    /// The rounding direction.
    pub rounding: Rounding,
    /// The encoding of the first factor, FRA.
    pub a: u64,
    /// The encoding of the second factor, FRC.
    pub c: u64,
    /// The encoding of the addend, FRB.
    pub b: u64,
}

/// The result of an instruction and the flags that it raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstructionOutcome {
    /// The encoding of the result.
    pub result: u64,
    /// The flags that the instruction raised.
    pub flags: Flags,
}

/// Runs every case in one QEMU process, and returns the outcomes in the
/// order of the cases.
///
/// # Panics
///
/// Panics when QEMU or the program cannot run, when the program fails, or
/// when its output does not match the cases.
#[must_use]
pub fn run(cases: &[Case]) -> Vec<Outcome> {
    batch(cases, |case| {
        format!(
            "{} {} {:016x} {:016x} {:016x} {:016x}",
            case.operation.word(),
            rounding_word(case.rounding),
            case.a.hi,
            case.a.lo,
            case.b.hi,
            case.b.lo,
        )
    })
    .iter()
    .map(|line| {
        let (results, flags) = parse_outcome(line);
        let &[hi, lo] = results.as_slice() else {
            panic!("an operation gives two halves: {line}");
        };
        Outcome {
            result: Pair::new(hi, lo),
            flags,
        }
    })
    .collect()
}

/// Runs every function case in one QEMU process, and returns the outcomes in
/// the order of the cases.
///
/// # Panics
///
/// Panics when QEMU or the program cannot run, when the program fails, or
/// when its output does not match the cases.
#[must_use]
pub fn run_functions(cases: &[FunctionCase]) -> Vec<Outcome> {
    batch(cases, |case| {
        let operands: Vec<String> = case.operands[..case.function.operand_count()]
            .iter()
            .map(|pair| format!("{:016x} {:016x}", pair.hi, pair.lo))
            .collect();
        format!(
            "{} {} {}",
            case.function.word(),
            rounding_word(case.rounding),
            operands.join(" ")
        )
    })
    .iter()
    .map(|line| {
        let (results, flags) = parse_outcome(line);
        let &[hi, lo] = results.as_slice() else {
            panic!("a function gives two halves: {line}");
        };
        Outcome {
            result: Pair::new(hi, lo),
            flags,
        }
    })
    .collect()
}

/// Runs every instruction case in one QEMU process, and returns the outcomes
/// in the order of the cases.
///
/// # Panics
///
/// Panics when QEMU or the program cannot run, when the program fails, or
/// when its output does not match the cases.
#[must_use]
pub fn run_instructions(cases: &[InstructionCase]) -> Vec<InstructionOutcome> {
    batch(cases, |case| {
        format!(
            "{} {} {:016x} {:016x} {:016x}",
            case.instruction.word(),
            rounding_word(case.rounding),
            case.a,
            case.c,
            case.b,
        )
    })
    .iter()
    .map(|line| {
        let (results, flags) = parse_outcome(line);
        let &[result] = results.as_slice() else {
            panic!("an instruction gives one result: {line}");
        };
        InstructionOutcome { result, flags }
    })
    .collect()
}

/// Writes one input line for each case to one QEMU process, and returns its
/// output lines in the order of the cases.
fn batch<T: Sync>(cases: &[T], input_line: impl Fn(&T) -> String + Sync) -> Vec<String> {
    check_qemu();
    let mut child = qemu()
        .arg(PROGRAM)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("{QEMU} cannot run ({error})"));
    let stdin = child.stdin.take().expect("the input is piped");
    let stdout = child.stdout.take().expect("the output is piped");
    // A second thread writes the cases while this thread reads the outcomes.
    // Otherwise both processes can stop on full pipes.
    let (written, outcomes) = thread::scope(|scope| {
        let writer = scope.spawn(|| {
            let mut writer = BufWriter::new(stdin);
            cases
                .iter()
                .try_for_each(|case| writeln!(writer, "{}", input_line(case)))
                .and_then(|()| writer.flush())
        });
        let outcomes: Vec<String> = BufReader::new(stdout)
            .lines()
            .map(|line| line.expect("the program writes text"))
            .collect();
        let written: io::Result<()> = writer.join().expect("the writer thread does not panic");
        (written, outcomes)
    });
    let status = child.wait().expect("the QEMU process can be waited for");
    assert!(status.success(), "{PROGRAM} failed under {QEMU}: {status}");
    written.expect("the program reads every case");
    assert_eq!(
        outcomes.len(),
        cases.len(),
        "the program writes one line for each case"
    );
    outcomes
}

/// Returns a command that runs [`QEMU`]. A `QEMU_*` variable, such as
/// `QEMU_CPU`, changes the emulated processor, so the command runs without
/// them.
fn qemu() -> Command {
    let mut command = Command::new(QEMU);
    for (name, _) in std::env::vars_os() {
        if name.to_string_lossy().starts_with("QEMU_") {
            command.env_remove(name);
        }
    }
    command
}

/// Stops the test when [`QEMU`] is not the pinned release. QEMU can change
/// after the build script checked it, for example in a system update.
fn check_qemu() {
    let output = qemu()
        .arg("--version")
        .output()
        .unwrap_or_else(|error| panic!("{QEMU} cannot run ({error})"));
    let text = String::from_utf8_lossy(&output.stdout);
    let first = text.lines().next().unwrap_or_default();
    assert!(
        first.starts_with(&format!("{QEMU} version {QEMU_VERSION} ")),
        "{first:?} is not the pinned QEMU {QEMU_VERSION}. QEMU executes the libgcc reference and \
         gives its flags"
    );
}

/// Reads one output line: the results and the flags, in hexadecimal.
fn parse_outcome(line: &str) -> (Vec<u64>, Flags) {
    let (results, flags) = line
        .rsplit_once(' ')
        .unwrap_or_else(|| panic!("an output line ends with the flags: {line}"));
    let results = results
        .split(' ')
        .map(|field| u64::from_str_radix(field, 16).expect("a result is hexadecimal"))
        .collect();
    let flags = u8::from_str_radix(flags, 16)
        .ok()
        .and_then(Flags::from_bits)
        .expect("the program writes the five flag bits in hexadecimal");
    (results, flags)
}
