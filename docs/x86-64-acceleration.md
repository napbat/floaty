# x86-64 Acceleration Coverage

This record compares three things for each operation of `Lanes`:

- what floaty computes today on x86-64, in each x86-64 build of the gates,
- which x86-64 instructions can give the bits of the engine, and
- the gap between the two.

It also lists the operations of `Float` that `Lanes` does not have.
The README lists the host paths that exist. This record lists the host
paths that could exist, and what blocks each one.

The record describes the branch `x86-64-gaps` on 2026-09-29, which closed
the first seven gaps of the first version of this record, and the work of
2026-09-30, which closed five more. A row that the later work changed has
figures from that work. The figures come
from `cargo bench -p floaty-verify --bench operations` on a Core i9-9900K,
which has x86-64-v3 and no AVX-512. Each figure is the lower of two runs. A
`Lanes` figure is in nanoseconds per lane, and a `Float` figure in
nanoseconds per operation. Each table has a column for the default build
and one for the x86-64-v3 build. In the table of binary32 and binary64, a
cell gives the figure of binary32 and then that of binary64. For
comparison, a packed path of binary32 takes 0.2 to 0.6 nanoseconds per
lane.

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

`Lanes` has every operation of `Float` that takes one value of the type in
each lane. A result that is not a value of the type is an array with one
element for each lane.

| Operation | `Lanes` | Path of `Lanes` without flags |
| --- | --- | --- |
| `+`, `-`, `*`, `/`, `sqrt`, `mul_add`, `round_to_integral`, `convert` | Yes | Packed where the build has instructions, otherwise the scalar path of each lane |
| `compare_quiet`, and the minimum and maximum families | Yes | Packed for binary32, binary64, and bfloat16, and for binary16 with F16C, otherwise the scalar path of each lane |
| `to_int` | Yes | Packed for binary32 and binary64, otherwise the scalar path of each lane |
| `from_int`, `remainder`, `scale_b` | Yes | The scalar path of each lane |
| `abs`, `copy_sign`, negation, `classify`, the `is_` methods, `total_cmp`, `next_up`, `next_down` | Yes | Integer code for each lane: no floating-point instruction is needed |
| `log_b` | Decimal formats only | `VGETEXPPS` computes the binary operation |
| `quantize`, `quantum`, `same_quantum` | Decimal formats only | No x86-64 instruction |
| The `_with` methods | Yes | Never a host path: the host flags do not give `ROUNDED_UP`, or `TINY` before rounding |

## binary32 and binary64

The lanes are binary32 x 4 and binary64 x 2 in the default build, and
binary32 x 8 and binary64 x 4 in the x86-64-v3 build.

| Operation | Default build | x86-64-v3 build | x86-64 instructions | Gap |
| --- | --- | --- | --- | --- |
| `+`, `-`, `*`, `/` | Packed, SSE2: 0.5, 1.2 | Packed, AVX: 0.3, 0.4 | `ADDPS`, `ADDPD` and the others. 512-bit forms with AVX-512F. | 512-bit chunks, and tail lanes in a mask |
| `sqrt` | Packed: 0.6, 1.3 | Packed: 0.3, 0.7 | `SQRTPS`, `SQRTPD`. 512-bit forms. | As above |
| `mul_add` | Engine: 22.8, 25.2 | Packed, FMA: 0.4, 0.9 | `VFMADD213PS`, `VFMADD213PD` (FMA). SSE2 has none. | None in the default build |
| `round_to_integral` | Engine: 9.0, 8.4 | Packed, SSE4.1, in each IEEE 754 direction but `TiesToAway`: 0.3, 0.6 | `ROUNDPS`, `ROUNDPD` with the direction in the immediate (SSE4.1). `VRNDSCALEPS` (AVX-512F). SSE2 has none. | None in the default build |
| `convert` between binary32 and binary64 | Packed: 0.9, 1.4 | Packed: 0.5, 0.8 | `CVTPS2PD`, `CVTPD2PS`, and their AVX forms | None |
| Comparison | Packed `compare_quiet`: 1.4, 1.8. Scalar `UCOMISS`: 2.4, 2.5 | Packed: 1.7, 1.8. Scalar: 4.1, 1.8 | `CMPLTPS` and `CMPUNORDPS`, and their other forms. `UCOMISS`, `UCOMISD`. | None |
| Minimum and maximum | Packed `minimum`: 0.7, 2.1. A NaN or two zeros in any pair of lanes sends every lane to its scalar path. Scalar `MINSS`: 2.3, 2.4 | Packed: 0.5, 1.1. Scalar: 2.7, 2.2 | `MINPS`, `MAXPS`, and their other forms. `VRANGEPS` gives `min_num` and `max_num` without the fallback (AVX-512DQ). `VMINMAXPS` gives the IEEE 754-2019 operations (AVX10.2). | None below AVX-512DQ |
| `to_int` | Packed `CVTPS2DQ`: 1.5, 2.0. A lane with the integer indefinite takes the scalar path. Scalar `CVTSS2SI` for each lane: 3.2, 3.9 | Packed: 1.7, 1.8. Scalar: 2.8, 3.2 | `CVTPS2DQ`, `CVTPD2DQ` to a 32-bit integer (SSE2). `VCVTPS2QQ`, `VCVTPD2QQ` to a 64-bit integer (AVX-512DQ). | Packed paths to 64 bits at AVX-512DQ |
| `from_int` | Scalar `CVTSI2SS` for each lane: 1.7, 1.5 | Scalar: 1.7, 1.5 | `CVTDQ2PS` rounds, and `CVTDQ2PD` is exact (SSE2). A packed path with them measured no faster than the scalar path, because the lanes of the integer type cost more than the instruction. `VCVTQQ2PS`, `VCVTQQ2PD` (AVX-512DQ). | None at SSE2 |
| `remainder` | Scalar `FPREM1` for each lane: 12.2, 12.7; distant operands 21.4, 131.4 | Scalar: 12.3, 12.5; 21.0, 131.2 | The x87 `FPREM1`, one lane at a time. No SSE or AVX instruction. | None |
| `scale_b` | Engine for each lane | Engine | `VSCALEFPS`, `VSCALEFSS` (AVX-512F). At SSE2, a product by an exact power of two is a host algorithm that needs a proof. | A path at AVX-512F |
| `log_b` | No binary method | No binary method | `VGETEXPPS` gives IEEE `logB` as a value of the format (AVX-512F) | An API decision first |
| `convert` to the same format | Engine: 8.8, 9.2 | Engine: 8.6, 7.9 | No instruction needed: the result is the operand, or its quiet NaN | A shortcut in the engine, not a host path |

## binary16

The packed paths widen binary16 lanes to binary32 with F16C and compute in
packed binary32, as the scalar binary16 path computes one value. Without
F16C, the scalar path widens binary16 and rounds binary32 results in integer
instructions, in `host/narrow.rs`.

| Operation | Default build | x86-64-v3 build | x86-64 instructions | Gap |
| --- | --- | --- | --- | --- |
| `+`, `-`, `*`, `/`, `sqrt`, `round_to_integral` | Scalar through binary32, with integer widening and rounding: 6.0 to 6.8, and 3.0 for `sqrt`. For each lane: 4.1 to 6.4. `round_to_integral`: engine, 8.9 | Packed F16C: 0.3 | `VCVTPH2PS`, packed binary32, and `VCVTPS2PH` with immediate 0 (F16C). `VADDPH` and the others (AVX512-FP16). | None at F16C. At SSE2, a packed form of the integer widening and rounding. |
| `mul_add` | Engine: 22.7 | Engine: 21.0 | `VFMADD213PH` rounds once (AVX512-FP16). Through binary32, the two roundings of a fused multiply-add can differ from one. | AVX512-FP16 only |
| `convert` to binary32 and binary64, and from binary32 | Scalar integer widening: 2.4 to binary32, 2.0 to binary64. Integer rounding from binary32: 2.1. For each lane: 2.1 | Packed F16C: 0.4 | `VCVTPH2PS`, `VCVTPS2PH` (F16C) | None at F16C |
| `convert` from binary64 | Engine. Scalar: 12.3 | Engine. Scalar: 12.4 | `VCVTPD2PH` and `VCVTSD2SH` round once (AVX512-FP16). F16C rounds twice, through binary32. | AVX512-FP16 only. The AArch64 build has this path, with `FCVT`. |
| Comparison, minimum, and maximum | Engine for each lane: 2.9 and 3.2. The engine compares faster than the integer widening. | Packed F16C: 1.8 and 2.4 | Packed `VCVTPH2PS`, then `CMPLTPS`, as for binary32. The minimum and maximum take the operand that the comparison selects. `VCMPPH`, `VMINPH` (AVX512-FP16). | None at F16C |
| `to_int`, `from_int`, `remainder` | Scalar through binary32 for `to_int` and `from_int`: 3.0, 3.1. `remainder` through binary32 by `FPREM1`: 15.4, and 18.0 for distant operands | Scalar: 2.3, 2.6. `remainder`: 13.8, and 16.6 | `to_int` and `from_int` through binary32. A binary16 remainder is a binary16 value, so the x87 `FPREM1` of the widened values gives it exactly. | Packed conversions at F16C |

The AVX512-FP16 specification, chapter 4, says that the instructions never
flush a binary16 operand or result and never read one as zero, whatever
MXCSR holds. The executable specification of the SDM does not agree: see
Manual Conflicts.

## bfloat16

A shift widens bfloat16 to binary32 exactly. x86-64 has no instruction that
rounds binary32 to bfloat16 at every value: `VCVTNEPS2BF16` reads a
subnormal binary32 input as zero, whatever MXCSR holds. So integer
instructions round binary32 results to bfloat16 to nearest even, in
`host/narrow.rs`.

| Operation | Default build | x86-64-v3 build | x86-64 instructions | Gap |
| --- | --- | --- | --- | --- |
| `+`, `-`, `*`, `/`, `sqrt` | Packed, with integer rounding: 1.8 to 2.1. Scalar: 1.7 to 2.5 | Packed: 1.6 to 1.8. Scalar: 1.7 to 2.4 | A shift widens the operands, packed binary32 arithmetic computes, and integer instructions round each lane. `VCVTNEPS2BF16` rounds to nearest even (AVX512_BF16, AVX-NE-CONVERT), but reads a subnormal input as zero. AVX10.2 `VADDBF16` and the others always read a subnormal input as zero and flush a subnormal result. | None |
| `mul_add` | Engine: 21.0 | Engine: 21.1 | Through binary32, the two roundings can differ from one. AVX10.2 `VFMADD213BF16` rounds once, and flushes as `VADDBF16` does. A flushed result reads as zero, so the path also needs a rule that finds results that flush. | AVX10.2, after the rule for flushed results |
| `round_to_integral` | Engine: 6.8 | Packed: 0.4 | A shift, `ROUNDPS` in the direction of the mode, and a shift back (SSE4.1) | None at SSE4.1 |
| `convert` to binary32 and binary64 | Packed shift: 0.4 | Packed shift: 0.4 | A shift, then `CVTPS2PD` | None |
| `convert` from binary32 | Integer rounding, scalar and packed. Not measured. | As the default build | Integer instructions, as for the arithmetic | None |
| Comparison, minimum, and maximum | Packed: 1.3 and 2.0 | Packed: 1.7 and 2.2 | A packed shift, then `CMPLTPS`. The minimum and maximum take the operand that the comparison selects. | None |
| `to_int` | Scalar through binary32 for each lane: 2.4 | Scalar: 2.3 | A shift, then `CVTPS2DQ` | A packed path at SSE2 |
| `remainder` | Scalar through binary32 by `FPREM1`: 13.3, and 23.1 for distant operands | 13.1, and 23.0 | A shift, then the x87 `FPREM1`. The remainder is a bfloat16 value, so it narrows exactly. | None |
| `from_int` | Engine. Scalar: 13.8 | Engine. Scalar: 13.8 | None. Two roundings through binary32 can differ from one. | None |

## x87 Extended

The x87 unit has no packed instructions. `+`, `-`, `*`, `/`, and `sqrt` of
`Lanes<F80, N>` check the control word once, and then run the x87
instruction of each lane: 4.2 and 4.0 per lane for `+`, against 3.8 and 4.0
for a loop of the scalar operator. `mul_add` runs in the engine, 36.5 and
35.6 per lane, because the x87 unit has no fused multiply-add. The
remainder of close operands takes `FPREM1`: 14.7 and 14.8 per operation.

| Operation | Gap |
| --- | --- |
| `convert` of `Lanes` to binary64 | 4.6 and 4.2 per lane, against 1.8 for the scalar conversion. A check of the control word for all lanes, as for the operators. |
| Comparison, minimum, and maximum | No path. `FUCOMIP` compares two x87 values. |

## FP8 and the Other Formats

| Formats | Today | x86-64 instructions | Gap |
| --- | --- | --- | --- |
| `F8E4M3Fn`, `F8E5M2` | Engine for every operation. `+` takes 14.4 and 12.8. | AVX10.2 has conversions only. `VCVTHF82PH` widens E4M3 to binary16 exactly. A shift widens E5M2 to binary16 exactly, at SSE2. `VCVTPH2HF8` and `VCVTPH2BF8` round binary16 to FP8 to nearest even and keep subnormals. An overflow gives a NaN for E4M3 and an infinity for E5M2, and the `S` forms saturate. AVX10.2 has no FP8 arithmetic. AMX-FP8 has only dot products. | Widening of E5M2 by a shift, now. The other conversions at AVX10.2. Arithmetic through binary32 needs a proof for the chain of roundings, and AVX10.2 to round to FP8. |
| `F8E4M3Fnuz`, `F8E5M2Fnuz`, `F8E4M3B11Fnuz` | Engine | None. The FNUZ encodings differ from the E4M3 and E5M2 of AVX10.2. | None |
| `F8E4M3`, `F8E3M4` | Engine | None | None |
| `TF32` | Engine | None. AMX-TF32 left the instruction set reference in 319433-062. | None |
| binary128 and wider, the decimal formats | Engine | None | None |

## Modes

Every host path needs a mode without FTZ, DAZ, saturation, or a precision limit, and
every path but `round_to_integral` needs a mode that rounds to nearest even.
A mode is a type, so an operation in another mode compiles to the engine.

| Mode | x86-64 support | Gap |
| --- | --- | --- |
| Directed rounding, `Rounded<M, R>` | `round_to_integral` takes the direction in the immediate of `ROUNDSS` and `ROUNDPS` (SSE4.1): a path today. `VCVTPS2PH` takes the direction in its immediate too (F16C). `CVTTPS2DQ` and the other `T` forms round toward zero. Embedded rounding, such as `{rz-sae}`, sets the direction of one instruction and suppresses every exception. It applies to 512-bit register forms and to scalar forms (AVX-512F). Revision 4.0 of the AVX10.2 specification removed embedded rounding on 256-bit registers. | Paths for the other operations at AVX-512F. Embedded rounding does not override FTZ or DAZ, so the path must still check them in MXCSR. |
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
  build needs its own gates in `AGENTS.md`.
- Rust 1.89, the minimum version of floaty, sets
  `cfg(target_feature = "avx512f")`, `avx512bf16`, `avx512fp16`, and
  `avxneconvert` on stable Rust, so a path at AVX-512F, AVX512_BF16,
  AVX512-FP16, or AVX-NE-CONVERT needs no newer compiler. Rust 1.89 does not
  set `avx10.1` or `avx10.2`, and both are unstable in Rust 1.98.1.

## Manual Conflicts

The Intel manuals disagree with themselves in three places that a future
path would depend on. Each conflict needs evidence from hardware, recorded
beside the test that checks it, before a host path depends on it.

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

- A scalar comparison sends an unordered pair to the engine, although the
  host order is exact for it: the path returns no result for a NaN, so
  `PartialOrd` cannot tell the unordered pair from a path that declines.
- The packed comparison gains little over the scalar path of each lane:
  1.4 nanoseconds per lane against 2.4 for binary32 x 8. The path makes the
  `Option<Ordering>` of each lane from three masks, lane by lane.
- The packed `to_int` of binary32 lanes takes 1.5 to 1.8 nanoseconds per
  lane, and `CVTPS2DQ` takes a fraction of that. The conversion of each lane
  to the integer type takes the rest.
- The packed bfloat16 paths round in integer instructions on AArch64 too.
  The AArch64 build with `FEAT_BF16` has `BFCVTN`, which rounds packed
  binary32 lanes to bfloat16, and which they do not use. This record lists
  only the gaps of x86-64.

## Gaps by Priority

The table lists the gaps that need a processor that the gates do not have.
The sections above also list smaller gaps that the gates can test, such as
the packed forms of the integer widening of binary16. The first version of
this record listed seven more gaps, which the branch
`x86-64-gaps` closed: packed binary16, packed bfloat16 widening and
rounding, the conversion of binary64 x 4, the other operations of `Lanes`,
the scalar comparison and minimum and maximum, the scalar remainder, and
`round_to_integral` in the directed modes. The work of 2026-09-30 closed
five more:

- bfloat16 arithmetic and `convert` from binary32, by integer rounding and
  without AVX512_BF16,
- the packed comparison and minimum and maximum of binary16 and bfloat16
  lanes,
- the packed `to_int` of binary32 and binary64 lanes,
- the remainder of binary16 and bfloat16, by `FPREM1`, and
- the per-lane cost of `Lanes` of x87 extended.

A packed `from_int` from 32-bit integers measured no faster than the scalar
path, so the record lists no gap for it.

| Priority | Gap | Instructions | Feature | Cost today | Blocker |
| --- | --- | --- | --- | --- | --- |
| 1 | 512-bit chunks, masked tails, directed rounding of the other operations, `scale_b`, conversions of 64-bit integers, `min_num` and `max_num` without the fallback | 512-bit forms, masks, embedded rounding, `VSCALEFPS`, `VCVTPS2QQ`, `VRANGEPS` | AVX-512F, DQ, VL | Engine, or 256-bit paths | No hardware in the gates. |
| 2 | binary16 `mul_add`, and `convert` from binary64 with one rounding | `VFMADD213PH`, `VCVTPD2PH`, `VCVTSD2SH` | AVX512-FP16 | 21.0 to 22.7 per lane, and 12.3 to 12.4 | No hardware in the gates. Conflict 1. |
| 3 | The IEEE 754-2019 minimum and maximum without the fallback, bfloat16 `mul_add`, FP8 conversions, and saturating integer conversions | `VMINMAXPS`, `VFMADD213BF16`, `VCVTHF82PH`, `VCVTPH2HF8`, `VCVTTPS2DQS` | AVX10.2 | Engine | No hardware in the gates. `avx10.2` is unstable in Rust. Conflict 2. |

## No x86-64 Instruction

These operations have no x86-64 instruction that gives the bits of the
engine:

- `mul_add` of binary32 and binary64 without FMA
- `from_int` of bfloat16
- `mul_add` of x87 extended
- FP8 arithmetic, `TF32`, binary128 and wider, the decimal formats, and the
  FNUZ FP8 formats
