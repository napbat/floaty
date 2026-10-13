//! Runs SSE and x87 instructions with unmasked exceptions.
//!
//! An unmasked SSE exception raises #XM at once, and Linux delivers it as
//! SIGFPE. The SSE functions install a handler that resumes after the
//! instruction, so the caller reads MXCSR and the destination as the trap
//! left them. An unmasked x87 exception waits for the next waiting
//! instruction. The x87 functions read the status word and clear the flags
//! before any such instruction, so they never trap.

use core::arch::asm;
use core::cell::Cell;
use std::sync::LazyLock;

/// The result of an instruction under unmasked exceptions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unmasked<T> {
    /// The destination after the instruction: unchanged when it trapped.
    pub value: T,
    /// The exception flags after the instruction: MXCSR bits 0 to 5, or the
    /// whole x87 status word.
    pub flags: u32,
    /// Whether the instruction raised a SIMD floating-point exception.
    pub trapped: bool,
}

thread_local! {
    /// The address at which an instruction that traps resumes, or 0 outside
    /// the instructions of this module.
    static RESUME: Cell<usize> = const { Cell::new(0) };
    /// Set by the handler when an instruction traps.
    static TRAPPED: Cell<bool> = const { Cell::new(false) };
}

/// Installs [`resume_after_trap`] as the SIGFPE handler of the process, once.
static HANDLER: LazyLock<()> = LazyLock::new(|| {
    // SAFETY: a zeroed `sigaction` is a valid value, and the call installs a
    // handler with the `SA_SIGINFO` signature for SIGFPE alone.
    unsafe {
        let mut action: libc::sigaction = core::mem::zeroed();
        action.sa_sigaction = (resume_after_trap as *const ()).addr();
        action.sa_flags = libc::SA_SIGINFO;
        libc::sigemptyset(&raw mut action.sa_mask);
        let status = libc::sigaction(libc::SIGFPE, &raw const action, core::ptr::null_mut());
        assert_eq!(status, 0, "the SIGFPE handler installs");
    }
});

/// Handles SIGFPE: resumes at the address in [`RESUME`] and sets [`TRAPPED`].
/// A SIGFPE from elsewhere restores the default action, which ends the
/// process when the instruction runs again.
extern "C" fn resume_after_trap(
    _signal: libc::c_int,
    _information: *mut libc::siginfo_t,
    context: *mut libc::c_void,
) {
    let resume = RESUME.with(Cell::get);
    if resume == 0 {
        // SAFETY: restoring the default action of SIGFPE has no precondition.
        unsafe {
            libc::signal(libc::SIGFPE, libc::SIG_DFL);
        }
        return;
    }
    let index = usize::try_from(libc::REG_RIP).expect("REG_RIP is an index");
    // SAFETY: with `SA_SIGINFO`, the kernel passes the `ucontext_t` of the
    // interrupted thread, and the return from the handler loads its RIP.
    unsafe {
        (*context.cast::<libc::ucontext_t>()).uc_mcontext.gregs[index] =
            i64::from_ne_bytes(resume.to_ne_bytes());
    }
    TRAPPED.with(|trapped| trapped.set(true));
}

/// The exception flags of MXCSR, bits 0 to 5.
const MXCSR_FLAGS: u32 = 0x3F;

/// Defines a function that runs an SSE instruction on `destination` and
/// `source` under the MXCSR value `control`, which can unmask exceptions.
macro_rules! sse {
    ($(#[$doc:meta])* $name:ident, $instruction:literal) => {
        $(#[$doc])*
        ///
        /// Returns the destination register, the MXCSR flags, and whether the
        /// instruction trapped. MXCSR is restored afterward.
        #[must_use]
        pub fn $name(destination: u128, source: u128, control: u32) -> Unmasked<u128> {
            LazyLock::force(&HANDLER);
            TRAPPED.with(|trapped| trapped.set(false));
            let resume = RESUME.with(Cell::as_ptr);
            let mut saved = 0_u32;
            let mut after = 0_u32;
            let value: u128;
            // SAFETY: the code saves MXCSR, loads `control`, and stores in
            // `RESUME` the address of the label after the instruction. A
            // trap of the instruction enters `resume_after_trap`, which
            // resumes at that label with the registers of the trap, so the
            // destination stays unchanged. The code then reads MXCSR and
            // restores the saved value.
            unsafe {
                asm!(
                    "stmxcsr dword ptr [{saved}]",
                    "ldmxcsr dword ptr [{control}]",
                    "lea {scratch}, [rip + 2f]",
                    "mov qword ptr [{resume}], {scratch}",
                    $instruction,
                    "2:",
                    "stmxcsr dword ptr [{after}]",
                    "ldmxcsr dword ptr [{saved}]",
                    "mov qword ptr [{resume}], 0",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    resume = in(reg) resume,
                    scratch = out(reg) _,
                    x = inout(xmm_reg) destination => value,
                    y = in(xmm_reg) source,
                    options(nostack),
                );
            }
            Unmasked {
                value,
                flags: after & MXCSR_FLAGS,
                trapped: TRAPPED.with(Cell::get),
            }
        }
    };
}

sse!(
    /// Runs `ADDSS`.
    addss, "addss {x}, {y}"
);
sse!(
    /// Runs `SUBSS`.
    subss, "subss {x}, {y}"
);
sse!(
    /// Runs `MULSS`.
    mulss, "mulss {x}, {y}"
);
sse!(
    /// Runs `DIVSS`.
    divss, "divss {x}, {y}"
);
sse!(
    /// Runs `SQRTSS` on the low lane of `source`.
    sqrtss, "sqrtss {x}, {y}"
);
sse!(
    /// Runs `ADDSD`.
    addsd, "addsd {x}, {y}"
);
sse!(
    /// Runs `SUBSD`.
    subsd, "subsd {x}, {y}"
);
sse!(
    /// Runs `MULSD`.
    mulsd, "mulsd {x}, {y}"
);
sse!(
    /// Runs `DIVSD`.
    divsd, "divsd {x}, {y}"
);
sse!(
    /// Runs `SQRTSD` on the low lane of `source`.
    sqrtsd, "sqrtsd {x}, {y}"
);
sse!(
    /// Runs `CVTSD2SS` on the low lane of `source`.
    cvtsd2ss, "cvtsd2ss {x}, {y}"
);

/// Defines a function that loads x87 operands, runs an operation under a
/// control word, which can unmask exceptions, and stores ST(0).
macro_rules! x87 {
    ($(#[$doc:meta])* $name:ident, $load:literal, $operation:literal) => {
        $(#[$doc])*
        ///
        /// `a` is pushed first and `b` second, so `b` is ST(0). Returns ST(0)
        /// after the operation and the status word that the operation leaves.
        /// The function never traps.
        #[must_use]
        pub fn $name(a: u128, b: u128, control: u16) -> Unmasked<u128> {
            let operands = [a.to_le_bytes(), b.to_le_bytes()];
            let mut output = [0_u8; 16];
            let mut saved = 0_u16;
            let status: u16;
            // SAFETY: the code saves the control word, clears the flags,
            // loads `control`, loads the operands, and runs the operation.
            // `FNSTSW` and `FNCLEX` do not wait, so an unmasked exception
            // that the operation flags is cleared before `FSTP`, the first
            // waiting instruction. An unmasked exception can leave both
            // operands on the stack, so `FNINIT` empties it before the saved
            // control word returns.
            unsafe {
                asm!(
                    "fnstcw word ptr [{saved}]",
                    "fnclex",
                    "fldcw word ptr [{control}]",
                    $load,
                    $operation,
                    "fnstsw ax",
                    "fnclex",
                    "fstp tbyte ptr [{target}]",
                    "fninit",
                    "fldcw word ptr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    operands = in(reg) operands.as_ptr(),
                    target = in(reg) output.as_mut_ptr(),
                    out("ax") status,
                    out("st(0)") _, out("st(1)") _, out("st(2)") _, out("st(3)") _,
                    out("st(4)") _, out("st(5)") _, out("st(6)") _, out("st(7)") _,
                    options(nostack),
                );
            }
            Unmasked {
                value: u128::from_le_bytes(output),
                flags: u32::from(status),
                trapped: false,
            }
        }
    };
}

x87!(
    /// `a + b` with `FADDP`.
    fadd_unmasked, "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]",
    "faddp st(1), st"
);
x87!(
    /// `a - b` with `FSUBP`.
    fsub_unmasked, "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]",
    "fsubp st(1), st"
);
x87!(
    /// `a * b` with `FMULP`.
    fmul_unmasked, "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]",
    "fmulp st(1), st"
);
x87!(
    /// `a / b` with `FDIVP`.
    fdiv_unmasked, "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]",
    "fdivp st(1), st"
);
x87!(
    /// The square root of `b` with `FSQRT`, with `a` below it.
    fsqrt_unmasked, "fld tbyte ptr [{operands}]\n fld tbyte ptr [{operands} + 16]",
    "fsqrt"
);
