//! Runs SSE, SSE3, SSE4.1, AVX, F16C, AVX-512, and AVX-512 BF16 instructions
//! that an emulator builds from several floaty operations: the minimum and
//! maximum, the compares with a predicate, the conversions to integers, the
//! horizontal sums and dot products, and the AVX-512 operations on the parts
//! of a value. Each function runs one instruction under an MXCSR value, and
//! returns its result and the MXCSR flags.
//!
//! The operands and the result are 128-bit `xmm` values: the low lane of a
//! scalar instruction is the low 32 or 64 bits. An instruction with an
//! immediate takes it as the const parameter `IMM`.

use core::arch::asm;

use super::MXCSR_FLAGS;

/// Defines a function that loads the `xmm` registers `x`, `y`, and `z` from
/// its operands `a`, `b`, and `c`, runs one instruction under an MXCSR value,
/// and returns `x` and the flags.
macro_rules! xmm {
    ($(#[$doc:meta])* $name:ident, $operation:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: u128, b: u128, c: u128, control: u32) -> (u128, u32) {
            xmm!(@body a, b, c, control, $operation, [])
        }
    };
    ($(#[$doc:meta])* $name:ident<IMM>, $operation:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name<const IMM: u8>(a: u128, b: u128, c: u128, control: u32) -> (u128, u32) {
            xmm!(@body a, b, c, control, $operation, [imm = const IMM,])
        }
    };
    (@body $a:ident, $b:ident, $c:ident, $control:ident, $operation:literal, [$($immediate:tt)*]) => {{
        let (mut saved, mut after) = (0_u32, 0_u32);
        let mut result = 0_u128;
        // SAFETY: the code saves MXCSR, loads three `xmm` registers from the
        // local operands, runs one instruction under `control`, stores `x`
        // to `result`, reads the flags, and restores MXCSR. It reads and
        // writes only the local variables. The caller checks that the
        // processor has the instruction.
        unsafe {
            asm!(
                "stmxcsr [{saved}]",
                "ldmxcsr [{control}]",
                "movdqu {x}, xmmword ptr [{a}]",
                "movdqu {y}, xmmword ptr [{b}]",
                "movdqu {z}, xmmword ptr [{c}]",
                $operation,
                "movdqu xmmword ptr [{result}], {x}",
                "stmxcsr [{after}]",
                "ldmxcsr [{saved}]",
                saved = in(reg) &raw mut saved,
                control = in(reg) &raw const $control,
                after = in(reg) &raw mut after,
                a = in(reg) &raw const $a,
                b = in(reg) &raw const $b,
                c = in(reg) &raw const $c,
                result = in(reg) &raw mut result,
                x = out(xmm_reg) _,
                y = out(xmm_reg) _,
                z = out(xmm_reg) _,
                $($immediate)*
                options(nostack),
            );
        }
        (result, after & MXCSR_FLAGS)
    }};
}

/// Defines a function that loads the `xmm` register `x` from its operand,
/// runs one conversion to a general register under an MXCSR value, and
/// returns the integer and the flags.
macro_rules! to_gpr {
    ($(#[$doc:meta])* $name:ident, $integer:ty, $operation:literal) => {
        $(#[$doc])*
        #[must_use]
        pub fn $name(a: u128, control: u32) -> ($integer, u32) {
            let (mut saved, mut after) = (0_u32, 0_u32);
            let result: u64;
            // SAFETY: the code saves MXCSR, loads one `xmm` register from
            // the local operand, runs one conversion under `control`, reads
            // the flags, and restores MXCSR. It reads and writes only the
            // local variables. The caller checks that the processor has the
            // instruction.
            unsafe {
                asm!(
                    "stmxcsr [{saved}]",
                    "ldmxcsr [{control}]",
                    "xor {result:e}, {result:e}",
                    "movdqu {x}, xmmword ptr [{a}]",
                    $operation,
                    "stmxcsr [{after}]",
                    "ldmxcsr [{saved}]",
                    saved = in(reg) &raw mut saved,
                    control = in(reg) &raw const control,
                    after = in(reg) &raw mut after,
                    a = in(reg) &raw const a,
                    result = out(reg) result,
                    x = out(xmm_reg) _,
                    options(nostack),
                );
            }
            let narrow = <$integer>::try_from(result & u64::from(<$integer>::MAX))
                .expect("the mask keeps the width of the integer");
            (narrow, after & MXCSR_FLAGS)
        }
    };
}

xmm!(
    /// `MINSS x, y`.
    minss, "minss {x}, {y}"
);
xmm!(
    /// `MAXSS x, y`.
    maxss, "maxss {x}, {y}"
);
xmm!(
    /// `MINSD x, y`.
    minsd, "minsd {x}, {y}"
);
xmm!(
    /// `MAXSD x, y`.
    maxsd, "maxsd {x}, {y}"
);
xmm!(
    /// `VCMPSS x, x, y, IMM`, with one of the 32 AVX predicates.
    vcmpss<IMM>, "vcmpss {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VCMPSD x, x, y, IMM`, with one of the 32 AVX predicates.
    vcmpsd<IMM>, "vcmpsd {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `HADDPS x, y`.
    haddps, "haddps {x}, {y}"
);
xmm!(
    /// `HSUBPS x, y`.
    hsubps, "hsubps {x}, {y}"
);
xmm!(
    /// `ADDSUBPS x, y`.
    addsubps, "addsubps {x}, {y}"
);
xmm!(
    /// `HADDPD x, y`.
    haddpd, "haddpd {x}, {y}"
);
xmm!(
    /// `ADDSUBPD x, y`.
    addsubpd, "addsubpd {x}, {y}"
);
xmm!(
    /// `DPPS x, y, IMM`.
    dpps<IMM>, "dpps {x}, {y}, {imm}"
);
xmm!(
    /// `DPPD x, y, IMM`.
    dppd<IMM>, "dppd {x}, {y}, {imm}"
);
xmm!(
    /// `VCVTPH2PS x, y`: the four binary16 values of the low 64 bits of `y`
    /// to binary32.
    vcvtph2ps, "vcvtph2ps {x}, {y}"
);
xmm!(
    /// `VCVTPS2PH x, y, IMM`: the four binary32 values of `y` to binary16,
    /// in the low 64 bits of the result.
    vcvtps2ph<IMM>, "vcvtps2ph {x}, {y}, {imm}"
);
xmm!(
    /// `VRNDSCALESS x, x, y, IMM`.
    vrndscaless<IMM>, "vrndscaless {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VRNDSCALESD x, x, y, IMM`.
    vrndscalesd<IMM>, "vrndscalesd {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VREDUCESS x, x, y, IMM`.
    vreducess<IMM>, "vreducess {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VREDUCESD x, x, y, IMM`.
    vreducesd<IMM>, "vreducesd {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VGETEXPSS x, x, y`.
    vgetexpss, "vgetexpss {x}, {x}, {y}"
);
xmm!(
    /// `VGETEXPSD x, x, y`.
    vgetexpsd, "vgetexpsd {x}, {x}, {y}"
);
xmm!(
    /// `VGETMANTSS x, x, y, IMM`.
    vgetmantss<IMM>, "vgetmantss {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VGETMANTSD x, x, y, IMM`.
    vgetmantsd<IMM>, "vgetmantsd {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VSCALEFSS x, x, y`: `x 2^floor(y)`.
    vscalefss, "vscalefss {x}, {x}, {y}"
);
xmm!(
    /// `VSCALEFSD x, x, y`: `x 2^floor(y)`.
    vscalefsd, "vscalefsd {x}, {x}, {y}"
);
xmm!(
    /// `VRANGESS x, x, y, IMM`.
    vrangess<IMM>, "vrangess {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VRANGESD x, x, y, IMM`.
    vrangesd<IMM>, "vrangesd {x}, {x}, {y}, {imm}"
);
xmm!(
    /// `VFIXUPIMMSS x, y, z, IMM`: `x` the destination, `y` the value, and
    /// the low 32 bits of `z` the table.
    vfixupimmss<IMM>, "vfixupimmss {x}, {y}, {z}, {imm}"
);
xmm!(
    /// `VFIXUPIMMSD x, y, z, IMM`: `x` the destination, `y` the value, and
    /// the low 32 bits of `z` the table.
    vfixupimmsd<IMM>, "vfixupimmsd {x}, {y}, {z}, {imm}"
);
xmm!(
    /// `VADDSS x, x, y` with the embedded rounding `{rn-sae}`.
    vaddss_nearest, "vaddss {x}, {x}, {y}, {{rn-sae}}"
);
xmm!(
    /// `VADDSS x, x, y` with the embedded rounding `{rd-sae}`.
    vaddss_down, "vaddss {x}, {x}, {y}, {{rd-sae}}"
);
xmm!(
    /// `VADDSS x, x, y` with the embedded rounding `{ru-sae}`.
    vaddss_up, "vaddss {x}, {x}, {y}, {{ru-sae}}"
);
xmm!(
    /// `VADDSS x, x, y` with the embedded rounding `{rz-sae}`.
    vaddss_zero, "vaddss {x}, {x}, {y}, {{rz-sae}}"
);
xmm!(
    /// `VCVTNEPS2BF16 x, y`: the four binary32 values of `y` to bfloat16, in
    /// the low 64 bits of the result.
    vcvtneps2bf16, "vcvtneps2bf16 {x}, {y}"
);
xmm!(
    /// `VCVTNE2PS2BF16 x, y, z`: the four binary32 values of `z` to the low
    /// four bfloat16 lanes, and those of `y` to the high four.
    vcvtne2ps2bf16, "vcvtne2ps2bf16 {x}, {y}, {z}"
);
xmm!(
    /// `VDPBF16PS x, y, z`: each binary32 lane of `x` plus the dot product of
    /// the two bfloat16 pairs of that lane in `y` and `z`.
    vdpbf16ps, "vdpbf16ps {x}, {y}, {z}"
);

to_gpr!(
    /// `VCVTSS2USI r32, x`.
    vcvtss2usi32, u32, "vcvtss2usi {result:e}, {x}"
);
to_gpr!(
    /// `VCVTTSS2USI r32, x`.
    vcvttss2usi32, u32, "vcvttss2usi {result:e}, {x}"
);
to_gpr!(
    /// `VCVTSD2USI r64, x`.
    vcvtsd2usi64, u64, "vcvtsd2usi {result}, {x}"
);
to_gpr!(
    /// `VCVTTSD2USI r64, x`.
    vcvttsd2usi64, u64, "vcvttsd2usi {result}, {x}"
);

/// Defines a function that runs `VFPCLASSSS` or `VFPCLASSSD` on the low lane
/// of its operand, and returns the low bit of the mask result. The
/// instruction reads MXCSR.DAZ and changes no flag.
macro_rules! fpclass {
    ($(#[$doc:meta])* $name:ident, $operation:literal) => {
        $(#[$doc])*
        ///
        /// # Panics
        ///
        /// Panics when the processor lacks AVX-512F or AVX-512DQ.
        #[must_use]
        pub fn $name<const IMM: u8>(a: u128, control: u32) -> bool {
            /// Runs the instruction with the features of AVX-512, which the
            /// mask register needs.
            #[target_feature(enable = "avx512f,avx512dq")]
            unsafe fn run<const IMM: u8>(a: u128, control: u32) -> bool {
                let mut saved = 0_u32;
                let mask: u32;
                // SAFETY: the code saves MXCSR, loads one `xmm` register
                // from the local operand, classifies it under `control`,
                // moves the mask to a general register, and restores MXCSR.
                // It reads and writes only the local variables.
                unsafe {
                    asm!(
                        "stmxcsr [{saved}]",
                        "ldmxcsr [{control}]",
                        "movdqu {x}, xmmword ptr [{a}]",
                        $operation,
                        "kmovw {mask:e}, {k}",
                        "ldmxcsr [{saved}]",
                        saved = in(reg) &raw mut saved,
                        control = in(reg) &raw const control,
                        a = in(reg) &raw const a,
                        mask = out(reg) mask,
                        x = out(xmm_reg) _,
                        k = out(kreg) _,
                        imm = const IMM,
                        options(nostack),
                    );
                }
                mask & 1 == 1
            }
            assert!(
                std::arch::is_x86_feature_detected!("avx512dq"),
                "VFPCLASS needs a processor with AVX-512DQ"
            );
            // SAFETY: the assertion shows that the processor has AVX-512DQ,
            // which implies AVX-512F.
            unsafe { run::<IMM>(a, control) }
        }
    };
}

fpclass!(
    /// `VFPCLASSSS k, x, IMM`.
    vfpclassss, "vfpclassss {k}, {x}, {imm}"
);
fpclass!(
    /// `VFPCLASSSD k, x, IMM`.
    vfpclasssd, "vfpclasssd {k}, {x}, {imm}"
);
