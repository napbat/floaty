# Repository Agent Guidelines

These rules apply to the whole repository. A more specific `AGENTS.md` in a
subdirectory overrides them for that subdirectory. `CLAUDE.md` only points
here. Put every rule in this file.

## Project

floaty is bit-exact, platform-independent software floating point. The
crate documentation records each rule of its behavior. The
[README](README.md) lists the formats, the host paths, and the oracles.

- Read the documentation of an item before you change the item.
- When a change alters a rule, update its documentation in the same change.
  Update the README too when the README states the rule.

## Priorities

- Optimize for correctness, clarity, maintainability, and a small review
  surface.
- Make the smallest cohesive change that solves the whole problem.
- Keep public behavior and compatibility unless the task asks for a breaking
  change.
- Write plain code, not clever code. Make invariants visible.
- Do not add speculative abstractions, dependencies, configuration, or
  features.

## Correctness Rules

- Give the same bits on every host. Do not use host floating-point arithmetic
  in `floaty`: no `f32` or `f64` operations and no math library calls. A host
  path is the only exception. The host path table of the README must list
  each one.
- The `From` conversions between `F32` and `f32`, and between `F64` and
  `f64`, are bit casts, not arithmetic. Use only `to_bits` and `from_bits`
  in them.
- Make a host path give the bits of the engine for every input it accepts.
  Send every other input, and every NaN result, to the engine.
- Select a host path at compile time with `cfg(target_arch)` and
  `cfg(target_feature)`. Only the slice kernels, the elementwise slice
  operations, and `convert_slice` of `Lanes`, and blocks, also select an
  instruction set at run time, in `host/packed/dispatch`. Keep that check
  behind the feature `std`, run it once, and keep its answer in an atomic.
  Do not detect the processor anywhere else.
- Give each instruction set of `dispatch` the bits of `Build`. Run the
  generic code of a set inside `Isa::run`, which compiles it with the
  features of the set behind `#[target_feature]`, and tie the `SAFETY`
  comment of each `run` to the check that found the features. The
  host-path tests run in each set that the processor has: `levels.rs` runs
  them again with `FLOATY_HOST_LEVEL`.
- Use a host path only in an entry point that returns no flags. The `_with`
  methods always run the engine.
- Pass the oracle tests in a build that enables each host path. Show its gain
  with `cargo bench -p floaty-verify --bench operations`.
- Put the `unsafe` code of `floaty` only in its host module. Give each
  `unsafe` block a `SAFETY` comment.
- Run every floating-point instruction of a host path in inline assembly.
  Test a NaN with integer instructions on the bits. LLVM can move a Rust
  float operation above the check of the environment.
- Blocks are the one exception: their steps run as Rust operations on
  `f32` and as `core::arch` intrinsics, so LLVM vectorizes the chain. A
  block must check the environment first, and pass every input through the
  empty assembly block of `host/block.rs` after the check. Send each lane
  with a NaN result, and each lane with a step without an exact
  instruction, to the engine. Add a step only where no NaN payload can
  decide a result that is not a NaN, or taint the lane.
- Put code for one architecture, such as inline assembly and `core::arch`
  intrinsics, behind `cfg(target_arch)` in `floaty` and in its tests. Every
  crate must build for each target of the gates.
- Round through the one rounding routine of each radix: `exact` for binary
  formats and `decimal::round` for decimal formats. Do not write another
  rounding path. A host path can also round in a host instruction. A host
  path can also round a binary32 or binary64 result to binary16, or a
  binary32 result to bfloat16, in `host/narrow.rs`.
- Handle special values in the operation. NaNs, infinities, and zeros never
  reach the rounding routine.
- Test every behavior against an established oracle: TestFloat, MPFR,
  `rustc_apfloat`, `ml_dtypes`, the host processor, decTest and decNumber,
  the Intel decimal library and its tests, QD, libgcc, or Mesa. Where no
  implementation exists, evaluate the published definition from IEEE 754 or
  a vendor manual, or the documented rule of floaty, with MPFR or decNumber.
  A behavior without an oracle test is not finished.
- A preset field is a claim about hardware. Cite the vendor manual, volume,
  and section beside the field. Confirm the field on hardware in
  `floaty-verify`.
- Do not model an instruction-level quirk of one instruction set in `floaty`.
  Give consumers the primitive that builds the quirk.
- Keep every trait sealed.

## Quality Gates

Run the targeted tests while you iterate. Before you hand off a Rust change,
run all of these from the workspace root:

```text
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic
cargo nextest run --workspace --no-fail-fast
cargo test --workspace --doc
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo +1.89 clippy -p floaty --all-targets -- -D warnings -D clippy::pedantic
```

Then run the clippy and test gates in the other builds. These are the
x86-64-v3 build, the engine-only build, the two AArch64 builds, the 32-bit
i686 build, and the big-endian s390x build. The last four run under QEMU.
Each build with its own flags keeps its own target directory.

```text
export V3="-C target-cpu=x86-64-v3" ENGINE="--cfg floaty_engine_only" FP16="-C target-feature=+fp16,+bf16"
RUSTFLAGS="$V3" CARGO_TARGET_DIR=target/x86-64-v3 cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic
RUSTFLAGS="$V3" CARGO_TARGET_DIR=target/x86-64-v3 cargo nextest run --workspace --no-fail-fast
RUSTFLAGS="$V3" CARGO_TARGET_DIR=target/x86-64-v3 cargo test --workspace --doc
RUSTFLAGS="$V3" CARGO_TARGET_DIR=target/x86-64-v3 cargo +1.89 clippy -p floaty --all-targets -- -D warnings -D clippy::pedantic
RUSTFLAGS="$ENGINE" CARGO_TARGET_DIR=target/engine-only cargo clippy --workspace --all-targets -- -D warnings -D clippy::pedantic
RUSTFLAGS="$ENGINE" CARGO_TARGET_DIR=target/engine-only cargo nextest run --workspace --no-fail-fast
RUSTFLAGS="$ENGINE" CARGO_TARGET_DIR=target/engine-only cargo test --workspace --doc
cargo clippy --workspace --all-targets --target aarch64-unknown-linux-gnu -- -D warnings -D clippy::pedantic
cargo nextest run --workspace --no-fail-fast --target aarch64-unknown-linux-gnu
cargo test --workspace --doc --target aarch64-unknown-linux-gnu
cargo +1.89 clippy -p floaty --all-targets --target aarch64-unknown-linux-gnu -- -D warnings -D clippy::pedantic
RUSTFLAGS="$FP16" CARGO_TARGET_DIR=target/aarch64-fp16 cargo clippy --workspace --all-targets --target aarch64-unknown-linux-gnu -- -D warnings -D clippy::pedantic
RUSTFLAGS="$FP16" CARGO_TARGET_DIR=target/aarch64-fp16 cargo nextest run --workspace --no-fail-fast --target aarch64-unknown-linux-gnu
RUSTFLAGS="$FP16" CARGO_TARGET_DIR=target/aarch64-fp16 cargo test --workspace --doc --target aarch64-unknown-linux-gnu
RUSTFLAGS="$FP16" CARGO_TARGET_DIR=target/aarch64-fp16 cargo +1.89 clippy -p floaty --all-targets --target aarch64-unknown-linux-gnu -- -D warnings -D clippy::pedantic
cargo clippy --workspace --all-targets --target i686-unknown-linux-gnu -- -D warnings -D clippy::pedantic
cargo nextest run --workspace --no-fail-fast --target i686-unknown-linux-gnu
cargo test --workspace --doc --target i686-unknown-linux-gnu
cargo +1.89 clippy -p floaty --all-targets --target i686-unknown-linux-gnu -- -D warnings -D clippy::pedantic
cargo clippy --workspace --all-targets --target s390x-unknown-linux-gnu -- -D warnings -D clippy::pedantic
cargo nextest run --workspace --no-fail-fast --target s390x-unknown-linux-gnu
cargo test --workspace --doc --target s390x-unknown-linux-gnu
cargo +1.89 clippy -p floaty --all-targets --target s390x-unknown-linux-gnu -- -D warnings -D clippy::pedantic
```

- Install cargo-nextest once:
  `cargo install cargo-nextest --version 0.9.146 --locked`. nextest runs
  each test in its own process, and runs the tests of all binaries at the
  same time. nextest does not run doctests, so each build also runs
  `cargo test --doc`.
- Install the minimum-version toolchain once:
  `rustup toolchain install 1.89 --profile minimal --component clippy`.
- Install the AArch64, i686, and s390x targets once for both toolchains:
  `rustup target add aarch64-unknown-linux-gnu i686-unknown-linux-gnu
  s390x-unknown-linux-gnu`, and the same command with `--toolchain 1.89`.
- The AArch64 gates need `aarch64-linux-gnu-gcc` 15.2.0 with its C library
  (packages gcc-aarch64-linux-gnu and libc6-dev-arm64-cross) and
  `qemu-aarch64` 10.2.1 (package qemu-user). `.cargo/config.toml` names the
  linker and the QEMU runner.
- The i686 and s390x gates need `i686-linux-gnu-gcc` and
  `s390x-linux-gnu-gcc` 15.2.0 with their C libraries (packages
  gcc-i686-linux-gnu, libc6-dev-i386-cross, gcc-s390x-linux-gnu, and
  libc6-dev-s390x-cross), and `qemu-i386` and `qemu-s390x` 10.2.1 (package
  qemu-user). `.cargo/config.toml` names the linkers and the QEMU runners.
- The x86-64-v3 gates need a processor with AVX2 and FMA. The default
  build checks its x86-64-v3 copies on such a processor, and its x86-64-v4
  copies only on a processor with AVX-512F, BW, CD, DQ, and VL. Run the
  gates on one before you hand off a change to those copies.
- `--workspace` includes `floaty-verify`. It builds its C reference
  libraries and its x86 hardware tests only for x86-64. That build runs on
  Linux x86-64 hosts only, and needs the submodules (`git submodule update
  --init`), `make`, `gcc`, `g++` 15.2.0, `objcopy`, `m4`, `curl`, `tar`,
  `sha256sum`, `python3`, a host glibc 2.32 or later, whose libm has the
  C23 NaN payload functions, `powerpc64le-linux-gnu-gcc` 15.2.0 with its
  binutils (package gcc-powerpc64le-linux-gnu) and its glibc 2.43 (package
  libc6-dev-ppc64el-cross 2.43-2ubuntu2cross1), and `qemu-ppc64le` 10.2.1
  (package qemu-user). The build compiles TestFloat, MPFR, decNumber, the Intel
  decimal library, and QD from source. The first build downloads the
  decimal and QD archives and two Mesa headers, so it needs network access
  once. The `Qd` tests need a processor on which glibc's `fma` is the FMA3
  instruction. On another host, run the gates with `-p floaty` instead of
  `--workspace`.
- Keep the submodules under `floaty-verify/reference/`, the pinned archives
  in `floaty-verify/build/`, the pinned compilers, the pinned PowerPC C
  library, the pinned QEMU, and the pinned cargo-nextest at their pinned
  releases. Make a change of release in its own commit, and state the
  reason in the commit message.
- Run the long exhaustive sweeps with
  `cargo test -p floaty-verify --release -- --ignored`.
- Measure performance with `cargo bench -p floaty-verify --bench operations`
  before and after a change to the engine.

A change is not finished until every gate passes.

## Lints

The workspace manifest denies these lint groups for every package:

| Tool   | Groups |
| ------ | ------ |
| rustc  | `warnings`, `unused`, `nonstandard_style`, `rust_2018_idioms`, `future_incompatible`, `missing_docs` |
| clippy | `correctness`, `suspicious`, `style`, `complexity`, `perf`, `pedantic` |
| rustdoc | `all` |

- Treat every compiler, Clippy, and rustdoc warning as an error.
- Fix the cause of a lint. Do not hide the lint, and do not weaken the
  workspace lint configuration. Do not satisfy a lint with a cast, a clone, or
  a rename that hides the problem.
- Do not add crate-, module-, or file-wide `#[allow(...)]`, and do not allow
  `clippy::too_many_lines` on a function that grew. Split the function
  instead.
- A narrow `#[allow(...)]` is acceptable only when the lint makes the code
  less correct or less clear. Put it on the smallest item. Add a short comment
  that says why the lint is wrong here.
- A function on the common path of the engine can allow
  `clippy::inline_always` when LLVM does not inline it for `#[inline]`. Put
  the allowance on that function only. `cargo bench -p floaty-verify --bench
  operations` must show the gain. The comment must state the measured gain.
- `clippy.toml` lists product names for the `doc_markdown` lint. Add a name
  there only when it is a product name, not a code identifier.

## Idiomatic Rust

- Use iterators and combinators instead of index loops. Limb arithmetic that
  carries a value from one index to the next can use an index loop.
- Take `&str`, `&[T]`, and `impl Iterator` in parameters. Return owned data
  only when the caller needs ownership.
- Implement `From` and `TryFrom` for conversions. Implement `Display` for
  user-facing text and `Debug` for everything else.
- Mark pure accessors and constructors `#[must_use]`, and public enums that
  will gain variants `#[non_exhaustive]`.
- Do not bend engine code to fit `const fn`. The engine uses traits.
- Derive `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`, `Hash`, and `Default`
  where the type supports them. Do not derive `Copy` on a type that can grow.
- Use a newtype or a small enum instead of a raw integer, a boolean parameter,
  a magic value, or an ambiguous tuple.
- Do not use lossy `as` casts. Use `From`, `TryFrom`, or an explicit helper
  that names the truncation, for example `low_u64`.
- Use `core` paths. `floaty` is `no_std` without `alloc`.
- Return structured errors for recoverable failures. Error types end in
  `Error` and implement `core::error::Error`. Do not panic in a public API
  unless the panic is a documented contract.
- Use `unwrap` or `expect` only for a proven invariant. Make the `expect`
  message state the violated invariant, not the immediate operation.
- Keep functions focused and control flow shallow. Use guard clauses so the
  happy path stays visible. Design public APIs so correct use is easy and
  invalid states are hard to represent.
- Document every public item. Update examples and crate-level docs when the
  behavior they describe changes.

## File and Module Size

- Keep every hand-written source file at or below 1,000 physical lines,
  including inline tests.
- Split a file before it reaches the cap. Extract modules by responsibility,
  never by line range. A split that leaves two halves of one concept is worse
  than the oversized file.
- Move inline tests to `<parent>/tests.rs` first when a file approaches the
  cap. Split the implementation next.
- Split a long function before you split its file.
- A generated file is exempt only when its header contains `@generated`.
  Never hand-edit a generated file. Change its generator.

## Cargo Project Layout

Follow the [Cargo Book package layout](https://doc.rust-lang.org/cargo/guide/project-layout.html)
in every workspace package.

- Keep the package manifest at the package root and Rust code under `src/`.
- Put integration tests in `tests/`, examples in `examples/`, and benchmarks
  in `benches/` at the package root. Do not put Rust source anywhere else.
- Put the build script of `floaty-verify` and its modules in
  `floaty-verify/build/`, with `main.rs` as the root.
- Name every binary, example, benchmark, and integration-test target in
  `kebab-case`. Name module files and directories in `snake_case`.
- Use the named-file module style: `parent.rs` beside a `parent/` directory.
  `mod.rs` is forbidden.
- Keep every C reference library and its build script in `floaty-verify`.
  `floaty` has no dependencies.

## Naming

- Spell compound words in full with underscores, for example
  `flush_to_zero`. Keep domain-standard acronyms: `nan`, `ieee`, `bid`,
  `dpd`, `ulp`, `fma`, `sse`, `x87`.
- Reserve `*Result` names for types used in `Result` positions.
- Use the `*_count` suffix for counting accessors. Do not introduce `num_*`
  names.
- Name every test module `tests`.
- Use American English spelling in identifiers, documentation, and comments.

## Writing Style

Apply ASD-STE100 Simplified Technical English to documentation, comments,
error messages, commit messages, and handoff notes.

- Write short sentences. Keep an instruction under 20 words and a description
  under 25 words. Put one instruction in one sentence, and one topic in one
  paragraph.
- Use the active voice and the imperative mood for instructions.
- Use one term for one concept.
- Use "make sure", not "ensure". Use "must" for a requirement and "can" for a
  possibility. Do not use "should", "may", or "might" for a rule.
- Name the noun. Do not use "it" or "this" when the referent can be misread.
- Write a comment to explain intent, an invariant, a tradeoff, or safety
  reasoning. Do not write a comment that repeats the code.
- Do not use decorative section-divider comments.
- Record a conflict when a vendor manual, a reference implementation, or an
  oracle disagrees with itself or with silicon. Write the record as a
  comment beside the test or the oracle code that handles the conflict. State the affected
  rule, the conflicting evidence, and the resolution. Cite the manual,
  revision, volume, and page, or the source file, release, and line.

## Generated Files

Never hand-edit a generated file. Change its generator and run it again.

| Generated file | Command |
| --- | --- |
| `floaty-verify/data/fp8-reference.txt`, `fp8-from-f16.bin`, `fp8-arithmetic.bin`, and `fp8-operations.bin` | `floaty-verify/scripts/generate_fp8_reference.py floaty-verify/data`, as its docstring states |
| `floaty-verify/data/mx-reference.txt`, `mx-from-f16.bin`, `mx-arithmetic.bin`, and `mx-e8m0.bin` | `floaty-verify/scripts/generate_mx_reference.py floaty-verify/data`, as its docstring states |

## Tests

- Add a regression test for every bug fix and every observable behavior
  change.
- Test behavior and invariants, not implementation details.
- Cover the edge cases of floating point: zeros of both signs, subnormals,
  the smallest and largest normals, infinities, quiet and signaling NaNs,
  halfway cases, and results that round across a binade.
- Keep unit tests inline in `#[cfg(test)] mod tests` in the file they test.
  Put oracle comparisons in `floaty-verify`.
- Test every FP8, FP6, and FP4 input pair in the normal test run. Mark sweeps that take
  minutes or longer `#[ignore]`, and give each one a comment that states its
  run time.
- Keep tests deterministic. Seed every random generator.
- Do not remove or weaken a test to make a change pass. Do not add a skip, an
  exemption, a broad lint allowance, or a silent fallback to obtain a pass.

## Repository Hygiene

- Inspect the working tree before you edit. Preserve unrelated user changes.
- Keep formatting-only churn and opportunistic refactors out of a focused
  change.
- Do not run destructive Git commands. Do not discard local work, commit,
  push, or rewrite history unless the user asks for it.
