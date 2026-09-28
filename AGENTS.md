# Repository Agent Guidelines

These rules apply to the whole repository. A more specific `AGENTS.md` in a
subdirectory overrides them for that subdirectory. `CLAUDE.md` only points
here. Put every rule in this file.

## Project

floaty is bit-exact, platform-independent software floating point.
[DESIGN.md](DESIGN.md) is the source of truth for every design decision.

- Read `DESIGN.md` before you change the crate.
- When a change alters a decision, update `DESIGN.md` in the same change.
- Follow the build order in `DESIGN.md`. Do not start a step before the
  previous step passes its oracle tests.

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
  in `floaty`: no `f32` or `f64` operations and no math library calls. A fast
  path is the only exception. It must pass the same oracle tests as the
  generic path, and `DESIGN.md` must list it.
- Round through the one rounding routine in `exact`. Do not write a second
  rounding path.
- Handle special values in the operation. NaNs, infinities, and zeros never
  reach the rounding routine.
- Test every behavior against an established oracle: TestFloat, MPFR,
  `rustc_apfloat`, `ml_dtypes`, the host processor, decTest, the Intel decimal
  library tests, QD, or libgcc. Where no implementation exists, evaluate the
  published definition from IEEE 754 or a vendor manual with MPFR. A behavior
  without an oracle test is not finished.
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
cargo test --workspace --no-fail-fast
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo +1.85 clippy -p floaty --all-targets -- -D warnings -D clippy::pedantic
```

- Install the minimum-version toolchain once:
  `rustup toolchain install 1.85 --profile minimal --component clippy`.
- `--workspace` includes `floaty-verify`. Its build runs on Linux x86-64
  hosts only, and needs the submodules (`git submodule update --init`),
  `make`, `gcc`, and `m4`. It builds TestFloat and MPFR from source. On
  another host, run the gates with `-p floaty` instead of `--workspace`.
- Keep the submodules under `floaty-verify/reference/` at their pinned
  releases. A change of release is a design change: record it in
  `DESIGN.md`.
- Run the long exhaustive sweeps with
  `cargo test -p floaty-verify --release -- --ignored`.

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
- Write a `docs/anomalies/` record when a vendor manual, a reference
  implementation, or an oracle disagrees with itself or with silicon. Start
  with a `Status:` line. Use the sections `Affected rule`, `Conflicting
  reference evidence`, `Resolution`, and `Regression evidence`. Cite the
  manual, revision, volume, and page.

## Generated Files

Never hand-edit a generated file. Change its generator and run it again.

| Generated file | Command |
| --- | --- |
| `floaty-verify/data/fp8-reference.txt`, `fp8-from-f16.bin`, and `fp8-arithmetic.bin` | `floaty-verify/scripts/generate_fp8_reference.py floaty-verify/data`, as its docstring states |

## Tests

- Add a regression test for every bug fix and every observable behavior
  change.
- Test behavior and invariants, not implementation details.
- Cover the edge cases of floating point: zeros of both signs, subnormals,
  the smallest and largest normals, infinities, quiet and signaling NaNs,
  halfway cases, and results that round across a binade.
- Keep unit tests inline in `#[cfg(test)] mod tests` in the file they test.
  Put oracle comparisons in `floaty-verify`.
- Test every FP8 input pair in the normal test run. Mark sweeps that take
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
