# x86-64 Acceleration Coverage

This record compares three things for each operation of `Lanes`:

- what floaty computes today on x86-64, in each x86-64 build of the gates,
- which x86-64 instructions can give the bits of the engine, and
- the gap between the two.

It also lists the operations of `Float` that `Lanes` does not have.
`DESIGN.md` lists the host paths that exist. This record lists the host
paths that could exist, and what blocks each one.

The record describes commit `f8edd24` on 2026-09-29. The figures come from
`cargo bench -p floaty-verify --bench operations` on a Core i9-9900K, which
has x86-64-v3 and no AVX-512. Each figure is the lower of two runs. A
`Lanes` figure is in nanoseconds per lane, and a `Float` figure in
nanoseconds per operation. Where a cell gives two figures, the first is the
default build and the second the x86-64-v3 build. For comparison, a packed
path of binary32 takes 0.2 to 0.6 nanoseconds per lane.

## Terms

| Term | Meaning |
| --- | --- |
| Packed path | One check of MXCSR for all lanes, then packed instructions. |
| Per-lane path | `Lanes` calls the scalar operation of each lane. The scalar operation takes its scalar host path, and checks MXCSR once for each lane. |
| Engine | The software implementation, with no host path. |
| Exact | The instruction gives the bits of the engine for each input that the path gives it, in the default environment. The path detects every other input, such as a NaN result, and sends it to the engine. |
| Exact with fallback | Exact after the path sends a known set of lanes to the engine, for example lanes with a subnormal value that the instruction flushes. |
| Not usable | The instruction gives other bits, and a path cannot detect the lanes that differ. An approximation such as `VRCP14PS` is not usable. |

## Instruction Sets

| Level or extension | Contents that matter here | Processors | Build in the gates |
| --- | --- | --- | --- |
| x86-64 baseline | SSE, SSE2 | Every x86-64 processor | Default |
| x86-64-v2 | SSE4.1: `ROUNDPS` | Not listed here | None. The x86-64-v3 build has SSE4.1. |
| x86-64-v3 | AVX, AVX2, FMA, F16C | Not listed here | x86-64-v3 |
| x86-64-v4 | AVX-512F, BW, DQ, VL, CD | Not listed here | None |
| AVX512_BF16 | `VCVTNEPS2BF16`, `VDPBF16PS` | Part of AVX10.1 | None |
| AVX512-FP16 | binary16 arithmetic and conversions | Sapphire Rapids and later. Part of AVX10.1. | None |
| AVX-NE-CONVERT | VEX `VCVTNEPS2BF16`, widening of bfloat16 and binary16 | Sierra Forest, Grand Ridge, Arrow Lake H, Diamond Rapids | None |
| AVX10.1 | The AVX-512 set of Sapphire Rapids, with BF16 and FP16 | Granite Rapids, Nova Lake | None |
| AVX10.2 | bfloat16 arithmetic, FP8 conversions, `VMINMAX`, saturating integer conversions | Diamond Rapids, Nova Lake | None |

The processor lists come from the Intel Architecture Instruction Set
Extensions Programming Reference, 319433-062, March 2026, Table 1-2. AVX10
has 512-bit registers on every processor since revision 4.0 of the AVX10.2
specification. This record does not check AMD processors.

The instruction facts come from these documents:

- the Intel SDM, Volume 1, 253665-087, and the Volume 2 instruction pages,
- the executable specification of the SDM at `intel.github.io/SDM`, from
  SDM revision 092,
- the AVX512-FP16 Architecture Specification, 347407-001, June 2021, and
- the AVX10.2 Architecture Specification, 361050-006, January 2026.

## The Operations of `Lanes`

| Operation | `Float` | `Lanes` | Floating-point instruction |
| --- | --- | --- | --- |
| `+`, `-`, `*`, `/` | Yes | Yes | Needed |
| `sqrt` | Yes | Yes | Needed |
| `mul_add` | Yes | Yes | Needed |
| `round_to_integral` | Yes | Yes | Needed |
| `convert` | Yes | Yes | Needed |
| `to_int`, `from_int` | Yes | No | Needed |
| `remainder` | Yes | No | Needed |
| `scale_b` | Yes | No | Needed, or an exact product by a power of two |
| `minimum`, `maximum`, `minimum_number`, `maximum_number`, `min_num`, `max_num` | Yes | No | Not needed: integer code can select an operand. An instruction is faster. |
| `compare_quiet_with`, `compare_signaling_with`, `PartialEq`, `PartialOrd` | Yes | No. A result for each lane needs a mask type. | Not needed, as for the minimum |
| `total_cmp` | Yes | No | Not needed |
| `abs`, `copy_sign`, negation | Yes | No | Not needed |
| `classify`, `is_nan`, and the other `is_` methods | Yes | No | Not needed |
| `next_up`, `next_down` | Yes | No | Not needed |
| `log_b` | Decimal formats only | No | `VGETEXPPS` computes the binary operation |
| `quantize`, `quantum`, `same_quantum` | Decimal formats only | No | No x86-64 instruction |
| The `_with` methods | Yes | Yes | Never used: the host flags do not give `ROUNDED_UP`, or `TINY` before rounding |

## binary32 and binary64

The lanes are binary32 x 4 and binary64 x 2 in the default build, and
binary32 x 8 and binary64 x 4 in the x86-64-v3 build.

| Operation | Default build | x86-64-v3 build | x86-64 instructions | Gap |
| --- | --- | --- | --- | --- |
| `+`, `-`, `*`, `/` | Packed, SSE2: 0.5, 1.2 | Packed, AVX: 0.2, 0.5 | `ADDPS`, `ADDPD` and the others. 512-bit forms with AVX-512F. | 512-bit chunks, and tail lanes in a mask |
| `sqrt` | Packed: 0.6, 1.3 | Packed: 0.3, 0.7 | `SQRTPS`, `SQRTPD`. 512-bit forms. | As above |
| `mul_add` | Engine: 23.1, 25.6 | Packed, FMA: 0.3, 0.6 | `VFMADD213PS`, `VFMADD213PD` (FMA). SSE2 has none. | None in the default build |
| `round_to_integral` | Engine: 9.0, 8.4 | Packed, SSE4.1: 0.3, 0.7 | `ROUNDPS`, `ROUNDPD` (SSE4.1). `VRNDSCALEPS` (AVX-512F). SSE2 has none. | None in the default build. For directed modes, see Modes. |
| `convert` between binary32 and binary64 | Packed: 0.9, 1.3 | Packed: 0.4, 2.4 | `CVTPS2PD`, `CVTPD2PS`, and their AVX forms | binary64 x 4 to binary32 takes 2.4 in both builds, against 1.3 for binary64 x 2. See Existing Paths. |
| `to_int` | No method. Scalar `CVTSS2SI`: 2.2, 2.2 | No method. Scalar: 2.6, 2.2 | `CVTPS2DQ`, `CVTPD2DQ` to a 32-bit integer (SSE2). `VCVTPS2QQ`, `VCVTPD2QQ` to a 64-bit integer (AVX-512DQ), and the unsigned forms (AVX-512F, DQ). An integer result out of range or from a NaN is the integer indefinite, which goes to the engine. | A method. Packed paths to 32 bits at SSE2, to 64 bits at AVX-512DQ. |
| `from_int` | No method. Scalar `CVTSI2SS`: 2.2, 2.2 | No method. Scalar: 2.0, 2.1 | `CVTDQ2PS` rounds, and `CVTDQ2PD` is exact (SSE2). `VCVTQQ2PS`, `VCVTQQ2PD`, and the unsigned forms (AVX-512DQ, F). | A method. Packed paths from 32 bits at SSE2, from 64 bits at AVX-512DQ. |
| Minimum and maximum | No method. Scalar engine: 3.9, 3.4 | No method. Scalar engine: 4.6, 3.1 | `MINPS` and `MAXPS` return the second operand for a NaN and for two zeros: exact with fallback for those lanes. `VRANGEPS` with imm8 0x04 and 0x05 gives `min_num` and `max_num`, with -0 below +0 (AVX-512DQ). `VMINMAXPS` gives the four IEEE 754-2019 operations (AVX10.2). | Methods. Scalar and packed paths at SSE2 with the fallback. |
| Comparison | No method. Scalar engine: 6.8, 7.2 | No method. Scalar engine: 9.3, 7.0 | `CMPPS` gives an exact mask: 8 predicates at SSE2, 32 at AVX, each quiet or signaling. `UCOMISS` and `COMISS` order two scalars. A comparison gives no NaN, so no lane goes to the engine. | A mask type and methods. A scalar path at SSE2. |
| `scale_b` | No method. Scalar engine. | No method. Scalar engine. | `VSCALEFPS`, `VSCALEFSS` (AVX-512F): `x * 2^floor(y)`, rounded once. An `i32` scale is exact in binary64. For binary32, a clamp to 4096 in magnitude changes no result and makes the scale exact. At SSE2, a product by an exact power of two is a host algorithm that needs a proof. | A method. A path at AVX-512F. |
| `remainder` | No method. Scalar engine: 31.7, 37.8; distant operands 77.5, 174.5 | No method. Scalar engine: 31.9, 36.2; distant operands 76.4, 174.4 | No SSE or AVX instruction. The x87 `FPREM1` gives the IEEE remainder exactly, one lane at a time. | A scalar path through `FPREM1` |
| Classification, `abs`, `copy_sign`, negation, `total_cmp`, `next_up`, `next_down` | No methods. Scalar integer code. | As the default build | Integer instructions: `PAND`, `PXOR`, `PCMPGTD`. `VFPCLASSPS` classifies (AVX-512DQ). | Methods. The integer code needs no host path. |
| `log_b` | No binary method | No binary method | `VGETEXPPS` gives IEEE `logB` as a value of the format (AVX-512F) | An API decision first |
| `convert` to the same format | Engine: 9.4, 9.0 | Engine: 8.2, 8.4 | No instruction needed: the result is the operand, or its quiet NaN | A shortcut in the engine, not a host path |

## binary16

`Float::convert` and the arithmetic of binary16 have scalar host paths
through F16C in the x86-64-v3 build. `Lanes` has no packed path for
binary16, so each lane takes the scalar path.

| Operation | Default build | x86-64-v3 build | x86-64 instructions | Gap |
| --- | --- | --- | --- | --- |
| `+`, `-`, `*`, `/` | Engine: 9.5 to 14.4 | Per-lane F16C: 2.3 | `VCVTPH2PS` widens 8 lanes exactly, and `VCVTPS2PH` with imm8 0 rounds to nearest even (F16C). Packed binary32 arithmetic computes between them. The double-rounding proof of the scalar path holds for each lane. `VADDPH` and the others compute in binary16 (AVX512-FP16). | A packed F16C path, now |
| `sqrt` | Engine: 27.9 | Per-lane F16C: 2.0 | As above, with `SQRTPS`. `VSQRTPH` (AVX512-FP16). | A packed F16C path, now |
| `mul_add` | Engine: 22.8 | Engine: 20.3 | `VFMADD213PH` rounds once (AVX512-FP16). Through binary32, the two roundings of a fused multiply-add can differ from one. | AVX512-FP16 only |
| `round_to_integral` | Engine: 7.4 | Per-lane F16C and SSE4.1: 2.0 | `VCVTPH2PS`, `ROUNDPS`, `VCVTPS2PH`, which is exact as in the scalar path. `VRNDSCALEPH` (AVX512-FP16). | A packed F16C path, now |
| `convert` to binary32 and binary64 | Engine: 10.2 | Per-lane F16C: 1.8 | `VCVTPH2PS` is exact (F16C), then `CVTPS2PD`. `VCVTPH2PD` (AVX512-FP16). | A packed F16C path, now |
| `convert` from binary32 | Engine. Scalar: 13.5 | Per-lane F16C. Scalar: 1.5 | `VCVTPS2PH` with imm8 0 (F16C) ignores FTZ, and reads DAZ only on its binary32 input | A packed F16C path, now |
| `convert` from binary64 | Engine. Scalar: 12.9 | Engine. Scalar: 11.4 | `VCVTPD2PH` and `VCVTSD2SH` round once (AVX512-FP16). F16C rounds twice, through binary32. | AVX512-FP16 only. The AArch64 build has this path today, with `FCVT`. |
| `to_int`, `from_int` | No methods. Scalar engine: 10.3, 13.2 | No methods. Scalar through binary32: 2.5, 2.7 | Through binary32 with F16C. `VCVTPH2DQ`, `VCVTDQ2PH`, and the forms of the other widths (AVX512-FP16). | Methods |
| Minimum, maximum, comparison, classification, `scale_b` | No methods | No methods | `VMINPH` with the legacy rule, `VCMPPH`, `VFPCLASSPH`, `VSCALEFPH` (AVX512-FP16) | Methods |

The AVX512-FP16 specification, chapter 4, says that the instructions never
flush a binary16 operand or result and never read one as zero, whatever
MXCSR holds. The executable specification of the SDM does not agree: see
Manual Conflicts.

## bfloat16

A shift widens bfloat16 to binary32 exactly. x86-64 has no instruction that
rounds binary32 to bfloat16 at every value: `VCVTNEPS2BF16` reads a
subnormal binary32 input as zero, whatever MXCSR holds.

| Operation | Default build | x86-64-v3 build | x86-64 instructions | Gap |
| --- | --- | --- | --- | --- |
| `+`, `-`, `*`, `/`, `sqrt` | Engine: 8.9 to 26.7 | Engine: 8.5 to 26.4 | A shift widens the operands (SSE2), and packed binary32 arithmetic computes. `VCVTNEPS2BF16` rounds to nearest even (AVX512_BF16, AVX-NE-CONVERT). A lane whose binary32 result is subnormal or a NaN goes to the engine: exact with fallback. AVX10.2 `VADDBF16` and the others always read a subnormal input as zero and flush a subnormal result. | A path at AVX512_BF16 with VL, or at AVX-NE-CONVERT, with the subnormal fallback. The AArch64 build has this path today, with `BFCVT`. |
| `mul_add` | Engine: 20.9 | Engine: 22.5 | Through binary32, the two roundings can differ from one. AVX10.2 `VFMADD213BF16` rounds once, and flushes as `VADDBF16` does. A flushed result reads as zero, so the path also needs a rule that finds results that flush. | AVX10.2, after the rule for flushed results |
| `round_to_integral` | Engine: 6.8 | Per-lane shift and `ROUNDSS`: 2.0 | A shift, `ROUNDPS`, and a shift back (SSE4.1), which is exact as in the scalar path | A packed path at SSE4.1, now |
| `convert` to binary32 and binary64 | Per-lane shift: 1.7 | Per-lane shift: 1.8 | A shift (SSE2 integer instructions), then `CVTPS2PD` | A packed path at SSE2, now |
| `convert` from binary32 | Engine | Engine | `VCVTNEPS2BF16`, with the subnormal fallback | AVX512_BF16 or AVX-NE-CONVERT |
| `to_int` | No method. Scalar through binary32: 2.3 | No method. Scalar: 2.3 | A shift, then `CVTPS2DQ` | A method |
| `from_int` | Engine. Scalar: 14.0 | Engine. Scalar: 14.1 | None. Two roundings through binary32 can differ from one. | None |

## x87 Extended

The x87 unit has no packed instructions. `Lanes<F80, N>` takes the scalar
x87 path of each lane: 5.6 and 5.5 per lane for `+`. A loop of the scalar
operator takes 4.0 and 3.9, so `Lanes` adds about 1.5 per lane. `mul_add`
runs in the engine, 36.6 and 37.1 per lane, because the x87 unit has no fused
multiply-add.

| Operation | Gap |
| --- | --- |
| `remainder` | A scalar path through `FPREM1`. Today the engine takes 50.3 and 46.9, and 402 and 389 for distant operands. |
| `Lanes` of x87 extended | The per-lane cost above the scalar operator. `convert` to binary64 takes 4.7 and 4.1 per lane, against 2.1 for the scalar conversion. |

## FP8 and the Other Formats

| Formats | Today | x86-64 instructions | Gap |
| --- | --- | --- | --- |
| `F8E4M3`, `F8E5M2` | Engine for every operation. `+` takes 14.4 and 12.8. | AVX10.2 has conversions only. `VCVTHF82PH` widens E4M3 to binary16 exactly. A shift widens E5M2 to binary16 exactly, at SSE2. `VCVTPH2HF8` and `VCVTPH2BF8` round binary16 to FP8 to nearest even and keep subnormals. An overflow gives a NaN for E4M3 and an infinity for E5M2, and the `S` forms saturate. AVX10.2 has no FP8 arithmetic. AMX-FP8 has only dot products. | Widening of E5M2 by a shift, now. The other conversions at AVX10.2. Arithmetic through binary32 needs a proof for the chain of roundings, and AVX10.2 to round to FP8. |
| `F8E4M3Fnuz`, `F8E5M2Fnuz` | Engine | None. The FNUZ encodings differ from the E4M3 and E5M2 of AVX10.2. | None |
| `TF32` | Engine | None. AMX-TF32 left the instruction set reference in 319433-062. | None |
| binary128 and wider, the decimal formats | Engine | None | None |

## Modes

Every host path needs a mode that rounds to nearest even without FTZ, DAZ,
or a precision limit. A mode is a type, so an operation in another mode
compiles to the engine.

| Mode | x86-64 support | Gap |
| --- | --- | --- |
| Directed rounding, `Rounded<M, R>` | `ROUNDPS` and `ROUNDSS` take the direction in imm8 (SSE4.1), and `VCVTPS2PH` too (F16C). `CVTTPS2DQ` and the other `T` forms round toward zero. Embedded rounding, such as `{rz-sae}`, sets the direction of one instruction and suppresses every exception. It applies to 512-bit register forms and to scalar forms (AVX-512F). Revision 4.0 of the AVX10.2 specification removed embedded rounding on 256-bit registers. | Paths for the directed modes: `round_to_integral` at SSE4.1, the other operations at AVX-512F. Embedded rounding does not override FTZ or DAZ, so the path must still check them in MXCSR. |
| FTZ and DAZ | MXCSR.FTZ and MXCSR.DAZ apply to every SSE and AVX instruction that honors them. No instruction takes them from an operand. `VCVTNEPS2BF16`, `VDPBF16PS`, and the AVX10.2 bfloat16 arithmetic always flush. | A path for an FTZ or DAZ mode must write MXCSR, and this record does not measure that cost. The AVX10.2 bfloat16 arithmetic can match an FTZ and DAZ mode of bfloat16 if its tininess rule matches. This record does not check that. |
| Precision limit | SSE and AVX have no precision control. The x87 precision control applies only to x87 extended. | None |

## Widths, Masks, Verification, and Rust

- 512-bit registers hold 16 binary32 or 8 binary64 lanes (AVX-512F,
  AVX10). A mask register computes the tail lanes without the scalar path.
- The gates run on a Core i9-9900K, which has no AVX-512. QEMU 10.2.1 does
  not emulate AVX-512: `qemu-x86_64 -cpu SapphireRapids` reports that TCG
  does not support `avx512f`, and under `-cpu max` a binary with `VADDPS`
  on a 512-bit register stops with an illegal instruction. An AVX-512 path
  needs Intel SDE or a processor with AVX-512 for its oracle tests. A new
  build in the gates is a design change.
- Stable Rust 1.85, the minimum version of floaty, never sets
  `cfg(target_feature = "avx512f")` or the other AVX-512 features. Rust
  1.89 made them stable. A path at AVX-512F, AVX512_BF16, AVX512-FP16, or
  AVX-NE-CONVERT needs a minimum version of 1.89 or later. `avx10.1` and
  `avx10.2` are unstable in Rust 1.98.1.

## Manual Conflicts

The Intel manuals disagree with themselves in three places that a future
path would depend on. Each conflict needs a record in `docs/anomalies/`,
with evidence from hardware, before a host path depends on it.

1. Chapter 4 of the AVX512-FP16 specification says that binary16 operands
   never flush and never read as zero. The executable specification of the
   SDM reads `MXCSR.DAZ16` and `MXCSR.FTZ16` for every binary16 operation,
   and no manual defines those bits.
2. For an infinite input to the saturating FP8 conversions, Table 3.6 of
   the AVX10.2 specification, revisions 6.0 and 7.0, gives an infinity or a
   NaN. Section 9.1.2 and its pseudocode give the largest finite value.
3. Table 14-13 of the SDM Volume 1, 253665-087, says that a masked overflow
   of `VCVTPS2PH` gives an infinity in every rounding direction. The
   binary32-to-binary16 function of the AVX10.2 specification gives a
   result that depends on the rounding direction.

## Existing Paths

Two existing paths lose time outside the instructions:

- `convert` of `Lanes<F64, 4>` to binary32 takes 2.4 per lane in both
  builds, against 1.3 for `Lanes<F64, 2>`. The packed path returns the
  result lanes as a `[u64; 4]`, and `Lanes::convert` makes the binary32
  lanes from it with `array::map`. For four lanes, LLVM calls that map out
  of line. For two lanes it inlines it.
- `Lanes` of x87 extended, as the x87 section shows.

## Gaps by Priority

The order puts first the gaps that the current gates can test, and then the
gaps with the largest cost today.

| Priority | Gap | Instructions | Feature | Cost today | Blocker |
| --- | --- | --- | --- | --- | --- |
| 1 | Packed binary16: `+`, `-`, `*`, `/`, `sqrt`, `round_to_integral`, and `convert` to and from binary32 and binary64 | `VCVTPH2PS`, `VCVTPS2PH`, and packed binary32 | F16C, in the x86-64-v3 build | 1.8 to 2.3 per lane in the x86-64-v3 build | None. The proof of the scalar path holds for each lane. |
| 2 | Packed bfloat16: `convert` to binary32 and binary64, and `round_to_integral` | Integer shifts, `ROUNDPS` | SSE2, SSE4.1 | 1.7 to 2.0 per lane | None |
| 3 | `convert` of binary64 x 4 to binary32 | No new instruction | AVX | 2.4 per lane | None |
| 4 | `Lanes` methods: `to_int`, `from_int`, minimum and maximum, comparison, classification, the sign operations, `total_cmp`, `next_up`, `next_down`, `scale_b`, `remainder` | The instructions of the tables | SSE2 for most | No method | API design: a mask type for comparisons and classification, and the integer type of each lane |
| 5 | Scalar comparison, minimum, and maximum | `UCOMISS`, `COMISS`, and `MINSS`, `MAXSS` with the fallback | SSE2 | 6.8 to 9.3, and 3.1 to 4.6 | None |
| 6 | Scalar `remainder` | `FPREM1` | x87 | 31.7 to 50.3, and 76.4 to 402 for distant operands | A loop of partial remainders, and the check of the x87 control word |
| 7 | `round_to_integral` in the directed modes | `ROUNDPS`, `ROUNDSS` with the direction in imm8 | SSE4.1 | Engine | The paths accept only a mode that rounds to nearest even |
| 8 | bfloat16 arithmetic, and `convert` from binary32 | `VCVTNEPS2BF16`, with the subnormal fallback | AVX512_BF16 with VL, or AVX-NE-CONVERT | 8.5 to 26.7 per lane | No hardware in the gates. Rust 1.89. |
| 9 | 512-bit chunks, masked tails, directed rounding, `scale_b`, conversions of 64-bit integers, `min_num` and `max_num` | 512-bit forms, masks, embedded rounding, `VSCALEFPS`, `VCVTPS2QQ`, `VRANGEPS` | AVX-512F, DQ, VL | Engine, or 256-bit paths | No hardware in the gates. Rust 1.89. |
| 10 | binary16 `mul_add`, and `convert` from binary64 with one rounding | `VFMADD213PH`, `VCVTPD2PH`, `VCVTSD2SH` | AVX512-FP16 | 20.3 to 22.8 per lane, and 11.4 to 12.9 | No hardware in the gates. Rust 1.89. Conflict 1. |
| 11 | The IEEE 754-2019 minimum and maximum, bfloat16 `mul_add`, FP8 conversions, and saturating integer conversions | `VMINMAXPS`, `VFMADD213BF16`, `VCVTHF82PH`, `VCVTPH2HF8`, `VCVTTPS2DQS` | AVX10.2 | Engine | No hardware in the gates. `avx10.2` is unstable in Rust. Conflict 2. |

## No x86-64 Instruction

These operations have no x86-64 instruction that gives the bits of the
engine:

- `mul_add` of binary32 and binary64 without FMA
- `from_int` of bfloat16
- `mul_add` of x87 extended
- FP8 arithmetic, `TF32`, binary128 and wider, the decimal formats, and the
  FNUZ FP8 formats
