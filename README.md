<p align="center">
  <h1 align="center">VaeaNTT</h1>
  <p align="center">
    <strong>High-performance Number Theoretic Transform engine for lattice-based cryptography</strong>
  </p>
  <p align="center">
    <a href="https://crates.io/crates/vaea-ntt"><img src="https://img.shields.io/crates/v/vaea-ntt.svg" alt="crates.io"></a>
    <a href="https://docs.rs/vaea-ntt"><img src="https://img.shields.io/docsrs/vaea-ntt" alt="docs.rs"></a>
    <a href="https://github.com/Vaeatech/VaeaNTT/actions"><img src="https://img.shields.io/github/actions/workflow/status/Vaeatech/VaeaNTT/ci.yml?branch=main" alt="CI"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg" alt="License"></a>
    <img src="https://img.shields.io/badge/MSRV-1.87-orange.svg" alt="MSRV">
    <img src="https://img.shields.io/badge/no__std-compatible-green.svg" alt="no_std">
  </p>
</p>

---

VaeaNTT is a Rust library providing NTT (Number Theoretic Transform) implementations
optimized for **ARM NEON** (aarch64), with a portable scalar fallback for all platforms.

- 🚀 **ARM NEON native** — all butterfly stages vectorized with 4-wide `u32` SIMD
- 🔀 **Two pipelines** — 28-bit primes (`ntt32`) and 60–62 bit primes (`ntt64`)
- 📦 **`no_std`** — runs on bare-metal, requires only `alloc`
- 🔒 **Constant-time** — branchless arithmetic, no data-dependent branches
- 🎯 **Runtime-generic** — any NTT-friendly prime, not hardcoded to one scheme
- 🌐 **Multi-language** — C, C++, JS/WASM bindings via Diplomat FFI

## Table of Contents

- [Quick Start](#quick-start)
- [Supported Parameters](#supported-parameters)
- [API Reference](#api-reference)
- [Performance](#performance)
- [Architecture](#architecture)
- [Security](#security)
- [Testing](#testing)
- [License](#license)

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
vaea-ntt = "0.1"
```

### Basic NTT

```rust
use vaea_ntt::ntt32::Ntt32Context;

// Any NTT-friendly prime < 2^28
let ctx = Ntt32Context::new(256, 8_380_417); // ML-DSA prime

let mut data = vec![42u32; 256];
ctx.forward(&mut data);   // Coefficient → NTT domain
ctx.inverse(&mut data);   // NTT domain → Coefficient
assert!(data.iter().all(|&x| x == 42));
```

### Post-Quantum Preset (ML-DSA)

```rust
use vaea_ntt::pq::{PqScheme, PqNtt};

let ntt = PqNtt::new(PqScheme::MlDsa65); // NIST Level 3
let mut poly = vec![0u32; 256];
poly[0] = 1;
ntt.forward(&mut poly);
ntt.inverse(&mut poly);
assert_eq!(poly[0], 1);
```

### Polynomial Multiplication

```rust
use vaea_ntt::ntt32::Ntt32Context;

let ctx = Ntt32Context::new(256, 8_380_417);

// (1 + x) × (1 + x) = 1 + 2x + x²  in Z_q[X]/(X^256 + 1)
let mut a = vec![0u32; 256];
a[0] = 1; a[1] = 1;
let result = ctx.negacyclic_mul(&a, &a);
assert_eq!(&result[..3], &[1, 2, 1]);
```

## Supported Parameters

VaeaNTT accepts any prime `q` and power-of-two `N` satisfying `q ≡ 1 (mod 2N)`.

### `ntt32` — Primes < 2²⁸

| Use Case | q | Bits | Tested N |
|:---------|--:|:----:|:---------|
| **ML-DSA** | 8 380 417 | 23 | 256 |
| **Falcon** | 12 289 | 14 | 512, 1024 |
| **NewHope** | 7 681 | 13 | 512, 1024 |
| **FHE** (CKKS/BGV CRT limbs) | any < 2²⁸ | ≤ 28 | up to 32 768 |

### `ntt64` — Primes 60–62 bits

For FHE-compatible 64-bit primes. Includes built-in constants for common primes
(`PRIME_SEAL`, `PRIME_60_1`, `PRIME_62_1`, etc.).

> **Note on ML-KEM**: ML-KEM uses q = 3329 with an incomplete NTT
> (size-128 over coefficient pairs), not a standard negacyclic NTT.
> VaeaNTT's standard NTT works with q = 3329 for N ≤ 128.
> A dedicated incomplete NTT module for ML-KEM is planned.

## API Reference

### Modules

| Module | Description |
|:-------|:------------|
| [`ntt32`](src/ntt32/) | NTT for primes < 2²⁸. ARM NEON vectorized + scalar fallback. |
| [`ntt64`](src/ntt64/) | NTT for 60–62 bit primes. Barrett and Montgomery arithmetic. |
| [`pq`](src/pq.rs) | Post-quantum presets for ML-DSA. |
| [`poly`](src/poly.rs) | Polynomial arithmetic over Z_q[X]/(X^N + 1), 64-bit coefficients. |
| [`rns`](src/rns.rs) | Residue Number System (multi-prime CRT) for FHE. |
| [`ffi`](src/ffi.rs) | FFI bindings via Diplomat (C, C++, JS/WASM). Requires `ffi` feature. |

### `Ntt32Context`

```rust
// Construction
let ctx = Ntt32Context::new(n, q);           // panics on invalid params
let ctx = Ntt32Context::try_new(n, q)?;      // returns Result<_, NttError>

// Forward / Inverse NTT (in-place)
ctx.forward(&mut data);                       // coefficient → NTT domain
ctx.inverse(&mut data);                       // NTT → coefficient (× N⁻¹)
ctx.inverse_lazy(&mut data);                  // NTT → coefficient (no N⁻¹)

// Polynomial multiplication in Z_q[X]/(X^N + 1)
let result = ctx.negacyclic_mul(&a, &b);     // allocating
ctx.negacyclic_mul_into(&mut a, &mut b, &mut result); // zero-allocation
```

On `aarch64`, `forward`/`inverse` dispatch to NEON automatically.
On other architectures, a scalar fallback using Shoup multiplication and Harvey lazy butterflies is used.

### `PqNtt`

```rust
use vaea_ntt::pq::{PqScheme, PqNtt};

let ntt = PqNtt::new(PqScheme::MlDsa65);
ntt.forward(&mut data);
ntt.inverse(&mut data);
let product = ntt.multiply(&a, &b);

// Available presets:
// PqScheme::MlDsa44  — NIST Level 2 (q=8380417, N=256)
// PqScheme::MlDsa65  — NIST Level 3 (q=8380417, N=256)
// PqScheme::MlDsa87  — NIST Level 5 (q=8380417, N=256)
```

### Utilities

```rust
use vaea_ntt::ntt32::{generate_primes_28, is_prime_32, find_primitive_root};

// Generate NTT-friendly primes < 2^28 for a given N
let primes = generate_primes_28(1024, 3); // 3 primes for N=1024
```

### Features

| Feature | Default | Description |
|:--------|:-------:|:------------|
| `std` | ✅ | Enables `std::error::Error` impl on `NttError` |
| `rand` | — | Random polynomial generation (`Poly64::new_random()`, etc.) |
| `ffi` | — | Diplomat FFI bindings (C, C++, JS/WASM) |

#### `no_std` Usage

```toml
[dependencies]
vaea-ntt = { version = "0.1", default-features = false }
```

Requires `alloc`. Zero runtime dependencies in this configuration.

## Performance

Measured with [Criterion](https://crates.io/crates/criterion) on **Apple M3 Pro** (aarch64), `--release`, single-threaded.

### Forward NTT (`ntt32`, q = 12 289)

| N | Latency | Throughput |
|-----:|--------:|----------:|
| 64 | **66 ns** | 970 M coeff/s |
| 256 | **234 ns** | 1.09 G coeff/s |
| 1 024 | **1.19 µs** | 860 M coeff/s |
| 4 096 | **5.7 µs** | 719 M coeff/s |
| 8 192 | **11.4 µs** | 719 M coeff/s |
| 16 384 | **27.2 µs** | 602 M coeff/s |
| 32 768 | **58.5 µs** | 560 M coeff/s |

### Inverse NTT (`ntt32`, q = 12 289)

| N | Latency |
|-----:|--------:|
| 256 | **320 ns** |
| 1 024 | **1.55 µs** |
| 4 096 | **7.7 µs** |
| 32 768 | **63.8 µs** |

### Negacyclic Polynomial Multiplication

Two forward NTTs + pointwise multiply + inverse NTT.

| N | Total |
|-----:|------:|
| 256 | **1.08 µs** |
| 1 024 | **4.97 µs** |
| 4 096 | **23.3 µs** |

> **Run `cargo bench` on your hardware for your own numbers.**
> Results vary with hardware and system load.
> Disable CPU frequency scaling for reproducible measurements.

## Architecture

```
src/
├── ntt32/           # 28-bit NTT pipeline
│   ├── arith.rs     # Branchless modular arithmetic (add, sub, mul, pow, inv)
│   ├── context.rs   # Ntt32Context — unified API with NEON/scalar dispatch
│   ├── neon.rs      # ARM NEON intrinsics (4-stage fused butterflies)
│   ├── scalar.rs    # Portable scalar (Shoup multiplication, Harvey butterfly)
│   └── prime.rs     # NTT-friendly prime generation, primitive root finding
├── ntt64/           # 64-bit NTT pipeline (Barrett + Montgomery)
│   ├── arith.rs     # 64-bit modular arithmetic
│   ├── context.rs   # Ntt64Context
│   └── prime.rs     # 64-bit prime utilities
├── pq.rs            # Post-quantum presets (ML-DSA)
├── poly.rs          # Poly64 — polynomial over Z_q[X]/(X^N+1)
├── rns.rs           # RNS/CRT multi-prime decomposition
├── ffi.rs           # Diplomat FFI bridge
└── lib.rs
```

### Design Rationale

<details>
<summary><strong>Why 28-bit primes for <code>ntt32</code>?</strong></summary>

- **ARM NEON native**: 4×`u32` lanes. `u32 × u32` products fit in `u64`, no widening to `u128`.
- **Lazy reduction**: With `q < 2²⁸`, intermediates `3q < 2³⁰` fit in `u32`, enabling deferred Barrett reduction across multiple butterfly stages.
- **PQ aligned**: All NIST lattice standards use primes ≤ 23 bits — well within 28 bits.

</details>

<details>
<summary><strong>Why two separate pipelines?</strong></summary>

- FHE schemes (CKKS, BGV) use 60–62 bit primes — these don't fit in `u32`.
- `ntt64` provides Barrett and Montgomery arithmetic for large primes.
- RNS combines multiple `ntt64` contexts for multi-precision FHE computation.

</details>

## Security

| Property | Guarantee |
|:---------|:----------|
| **Constant-time** | All arithmetic uses branchless SIMD masks (`vcgeq` + `vandq`), no data-dependent branches. |
| **Input validation** | `try_new()` rejects non-prime `q`, non-power-of-two `N`, and non-NTT-friendly primes. |
| **Memory safety** | All NEON accesses are bounds-checked via loop guards. `unsafe` limited to NEON intrinsics. |
| **Thread safety** | `Ntt32Context` is `Send + Sync`. Verified with 8 threads × 100 iterations. |

See [SECURITY.md](SECURITY.md) for the vulnerability disclosure policy.

## Testing

```bash
# Unit + integration + doc tests
cargo test --release

# Benchmarks
cargo bench --bench ntt32_bench      # NTT32 full scaling suite
cargo bench --bench ntt64_bench      # NTT64 pipeline
cargo bench --bench pq_bench         # Post-quantum presets

# Security & exhaustive validation
cargo run --release --example exhaustive_test          # 2618 test cases
cargo run --release --example verify_no_false_positive # anti-trivial-pass
cargo run --release --example security_exploits        # exploit suite
```

## License

This project is **dual-licensed**:

### Open Source — AGPL-3.0-or-later

Free for open-source projects. See [LICENSE](LICENSE).

If you use VaeaNTT in a network service or distribute it, you must release your
complete source code under the AGPL. This applies to modified and unmodified usage.

### Commercial License

For closed-source, proprietary, or embedded use, a commercial license is available
that removes all AGPL obligations.

**Contact**: [alexis@vaea.tech](mailto:alexis@vaea.tech)
