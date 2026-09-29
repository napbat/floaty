# floaty

Bit-exact, platform-independent software floating point for Rust.

floaty emulates floating-point formats and the behavior of the hardware that
uses them. Every operation gives the same bits on every host. It is a base
layer for binary lifters, constant folders, and software FPU emulators.

> **Status:** build steps 1 to 8 of 8 are complete: every binary format
> decodes, classifies, rounds, converts, and computes `+ - * /`, square root,
> and fused multiply-add, correctly rounded in every rounding direction, and
> has comparisons and a total order, the minimum and maximum families,
> integer conversions, rounding to an integral value, the IEEE remainder,
> `scale_b`, `next_up` and `next_down`, and the sign operations. The x86
> SSE and x87 presets give the behavior of those units. The binary32 and
> binary64 operators use the host unit on x86-64 where it gives the same
> bits. decimal32, decimal64, and decimal128, in BID and DPD, have the same
> operations with the IEEE 754 preferred exponents, the quantum operations,
> and correctly rounded conversions to and from every binary format.
> Double-double values match libgcc's IBM `long double` under QEMU, or QD's
> `dd_real`, bit for bit under the behavior of each platform, NaN payloads
> included.
> [DESIGN.md](DESIGN.md) holds the approved design and the build order.

## Planned Scope

- **Binary formats** described by parameters: every IEEE 754 binary width up
  to 512 bits, bfloat16, TF32, the FP8 variants (OCP E4M3 and E5M2, and the
  FNUZ variants), and x87 80-bit extended precision.
- **Decimal formats**: decimal32, decimal64, and decimal128, in both the BID
  and the DPD encodings.
- **Double-double**: bit-compatible with GCC's PowerPC `long double`, or with
  the QD library.
- **Correct rounding** for add, subtract, multiply, divide, square root,
  fused multiply-add, and conversions.
- **Platform behavior as data**: rounding direction, flush-to-zero,
  denormals-are-zero, tininess detection, NaN rules, and x87 precision
  control, with presets for x86 SSE and x87.

## Planned API

Each type carries a default mode. A single operation can override it and get
the flags back.

```rust
type F32 = Float<Binary<8>, 32>; // default mode: mode::Ieee

let sum = a + b;                                        // default mode, flags dropped
let (sum, flags) = a.add_with(b, Rounding::TowardZero); // one-operation override
let (sum, flags) = a.add_with(b, Env::X86_SSE);         // behavior chosen at run time
let (sum, flags) = a.add_with(b, mode::X86Sse);         // behavior fixed at compile time
```

## Workspace

| Package | Purpose |
| --- | --- |
| `floaty` | The library. `no_std`, no dependencies. |
| `floaty-verify` | The verification harness against TestFloat, MPFR, the host processor, and other references. Not a default member. |

## Development

Rust 1.85 or later, edition 2024. The verification harness runs on Linux
x86-64 hosts and also needs the reference sources, `make`, `gcc`, `g++`
15.2.0, `objcopy`, `m4`, `curl`, `tar`, `sha256sum`, `python3`,
`powerpc64le-linux-gnu-gcc` 15.2.0, and `qemu-ppc64le` 10.2.1. The QD
tests need a processor with FMA3. The first build downloads the decimal
and QD references, so it needs network access once:

```text
git submodule update --init
```

Run the quality gates from the workspace root:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic
cargo test --workspace --no-fail-fast
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo +1.85 clippy -p floaty --all-targets -- -D warnings -D clippy::pedantic
```

[AGENTS.md](AGENTS.md) holds the full contributor rules.
