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
//! [`Pair`], [`Rounding`], [`Flags`], and [`Outcome`] are also the types of
//! [`crate::qd`], so one test can compare both references.

use core::fmt;
use core::ops::BitOr;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};
use std::thread;

use floaty::env::{FusedNanOrder, InvalidProduct, NanPropagation, NanRule, Tininess};

/// The batch program, a static powerpc64le executable.
pub const PROGRAM: &str = env!("FLOATY_IBM_LDOUBLE");

/// The emulator that runs [`PROGRAM`], which the build script checks.
pub const QEMU: &str = env!("FLOATY_QEMU");

/// The pinned QEMU release. The build script checks it when it builds
/// [`PROGRAM`].
const QEMU_VERSION: &str = env!("FLOATY_QEMU_VERSION");

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
    /// `fmodl`: the remainder of the truncated quotient.
    Fmod,
    /// `remainderl`: the IEEE 754 remainder.
    Remainder,
    /// `fmal`: `a * b + c`.
    MulAdd,
}

impl Function {
    /// Returns the name of the function in the input of [`PROGRAM`].
    const fn word(self) -> &'static str {
        match self {
            Self::Sqrt => "sqrtl",
            Self::NextUp => "nextupl",
            Self::NextDown => "nextdownl",
            Self::IsCanonical => "iscanonicall",
            Self::Fmod => "fmodl",
            Self::Remainder => "remainderl",
            Self::MulAdd => "fmal",
        }
    }

    /// Returns the number of operands of the function.
    #[must_use]
    pub const fn operand_count(self) -> usize {
        match self {
            Self::Sqrt | Self::NextUp | Self::NextDown | Self::IsCanonical => 1,
            Self::Fmod | Self::Remainder => 2,
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

    /// Returns the name of the direction in the input of [`PROGRAM`].
    const fn word(self) -> &'static str {
        match self {
            Self::TiesToEven => "nearest",
            Self::TowardZero => "zero",
            Self::TowardPositive => "up",
            Self::TowardNegative => "down",
        }
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

/// The result of an operation and the flags that it raised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Outcome {
    /// The result.
    pub result: Pair,
    /// The flags that the operation raised.
    pub flags: Flags,
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
            case.rounding.word(),
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
            case.rounding.word(),
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
            case.rounding.word(),
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
        first.starts_with(&format!("qemu-ppc64le version {QEMU_VERSION} ")),
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
