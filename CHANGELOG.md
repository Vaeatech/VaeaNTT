# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-13

### Added

- **NTT32 pipeline** — 28-bit primes (< 2²⁸) with ARM NEON native vectorization
  - Harvey lazy reduction (branchless, constant-time)
  - Shoup precomputed quotients
  - Forward, inverse, inverse_lazy (no N⁻¹ normalization)
  - Negacyclic polynomial multiplication (allocating + zero-alloc)
  - Automatic NEON/scalar dispatch (compile-time `#[cfg]`)

- **NTT64 pipeline** — 60-62 bit primes with Barrett/Montgomery reduction
  - Compatible with SEAL/OpenFHE prime conventions
  - Forward, inverse, tiled forward NTT

- **Poly64** — Polynomial arithmetic over Z_q[X]/(X^N+1)
  - NTT-domain operations (add, sub, mul, scalar mul, negate)
  - Random sampling (uniform, ternary, Gaussian) via `rand` feature

- **RNS/CRT** — Multi-prime residue number system
  - RnsContext with per-modulus NTT contexts
  - RnsPoly with component-wise operations

- **Post-quantum coverage** — Validated with NIST standard primes:
  - ML-DSA (NIST post-quantum signature): q = 8380417, N = 256 (full negacyclic NTT)
  - Falcon: q = 12289, N = 512/1024

- **Benchmarks** — 4 Criterion suites:
  - `ntt32_bench`: NTT32 pipeline scaling
  - `ntt64_bench`: NTT64 pipeline scaling
  - `pq_bench`: NIST PQ standard primes
  - `vs_concrete_ntt`: Cross-validation with concrete-ntt

- **Error handling** — `NttError` enum with `try_new()` constructors
- **Documentation** — `#![warn(missing_docs)]`, rustdoc, README
- **CI** — GitHub Actions (ARM + x86 + macOS, clippy, docs, format, MSRV)
- **no_std** — `#![no_std]` with `alloc`, `std` feature (default)
- **FFI** — Diplomat bindings for C, C++, JS/WASM
- **Constant-time** — DudeCT statistical validation (forward, inverse, negacyclic_mul)
- **Send + Sync** — Compile-time assertions for thread safety

[Unreleased]: https://github.com/Vaeatech/VaeaNTT/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Vaeatech/VaeaNTT/releases/tag/v0.1.0
