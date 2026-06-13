# VaeaNTT

**High-performance Number Theoretic Transform engine for post-quantum cryptography.**

VaeaNTT is a Rust library providing NTT (Number Theoretic Transform) implementations
optimized for ARM NEON, with scalar fallback for all platforms. Constant-time validated
via DudeCT. `no_std` compatible.

## Post-Quantum Coverage

VaeaNTT natively supports all three NIST post-quantum standards:

| Standard | Modulus | Bits | NTT Size |
|----------|:-------:|:----:|:--------:|
| **ML-KEM** (FIPS 203) | q = 3 329 | 12 | 128 |
| **ML-DSA** (FIPS 204) | q = 8 380 417 | 23 | 256 |
| **Falcon** | q = 12 289 | 14 | 512/1024 |

Plus FHE (CKKS/BGV) via 28-bit CRT primes, and any prime < 2²⁸.

## Quick Start

```rust
use vaea_ntt::ntt32::{Ntt32Context, generate_primes_28};

// ML-DSA (Dilithium): q = 8380417, N = 256
let ctx = Ntt32Context::new(256, 8_380_417);

let mut data = vec![42u32; 256];
ctx.forward(&mut data);   // NTT forward
ctx.inverse(&mut data);   // NTT inverse — data is restored

// Negacyclic polynomial multiplication (zero-allocation)
let a = vec![1u32; 256];
let b = vec![2u32; 256];
let mut a_buf = a.clone();
let mut b_buf = b.clone();
let mut result = vec![0u32; 256];
ctx.negacyclic_mul_into(&mut a_buf, &mut b_buf, &mut result);
```

## API

### `Ntt32Context` — 28-bit pipeline (ARM NEON optimized)

```rust
// Construction (panics on invalid params)
let ctx = Ntt32Context::new(n, q);

// Fallible construction
let ctx = Ntt32Context::try_new(n, q)?;

// Forward / Inverse NTT (in-place)
ctx.forward(&mut data);
ctx.inverse(&mut data);         // includes N⁻¹ normalization
ctx.inverse_lazy(&mut data);    // without N⁻¹

// Polynomial multiplication in Z_q[X]/(X^N + 1)
let result = ctx.negacyclic_mul(&a, &b);                     // allocating
ctx.negacyclic_mul_into(&mut a, &mut b, &mut result);        // zero-alloc
```

### `Ntt64Context` — 60-62 bit pipeline

For FHE-compatible 64-bit primes (SEAL, OpenFHE interop).

### Features

| Feature | Default | Description |
|---------|:-------:|-------------|
| `std` | **on** | Enables `std::error::Error` impl |
| `rand` | off | Enables `Poly64::new_random()`, `new_ternary()`, `new_gaussian()` |
| `ffi` | off | Diplomat FFI bindings (C, C++, JS/WASM) |

### `no_std`

```toml
[dependencies]
vaea-ntt = { version = "0.1", default-features = false }
```

Requires `alloc`. No runtime dependencies.

## Architecture

```
vaea-ntt/
├── src/
│   ├── ntt32/           # 28-bit NTT pipeline
│   │   ├── arith.rs     # Branchless modular arithmetic
│   │   ├── context.rs   # Ntt32Context (unified API)
│   │   ├── neon.rs      # ARM NEON intrinsics (all stages vectorized)
│   │   ├── scalar.rs    # Scalar fallback (Shoup/Harvey)
│   │   └── prime.rs     # NTT-friendly prime generation
│   ├── ntt64/           # 64-bit NTT pipeline (Barrett/Montgomery)
│   ├── poly.rs          # Polynomial arithmetic
│   ├── rns.rs           # RNS/CRT multi-prime
│   ├── ffi.rs           # Diplomat FFI bridge
│   └── lib.rs
├── benches/
│   ├── pq_bench.rs      # NIST PQ standard benchmarks
│   └── vs_concrete_ntt.rs  # Competitive benchmarks
├── bindings/
│   ├── c/               # Generated C headers
│   ├── cpp/             # Generated C++ headers
│   └── js/              # Generated JS/TS modules
└── examples/
    └── mldsa_ntt.rs     # ML-DSA/ML-KEM/Falcon demo
```

### Why 28-bit?

The 28-bit prime choice is architecturally motivated:

- **ARM NEON native**: `u32 × u32 → u64` fits perfectly in NEON 128-bit registers (4 lanes). No widening to `u128`.
- **Harvey lazy reduction**: Coefficients stay in `[0, 2q)` between butterfly stages. With `q < 2²⁸`, intermediates `3q < 2³⁰` fit in `u32`.
- **Post-quantum alignment**: All NIST PQ standards use primes ≤ 23 bits — well within our 28-bit pipeline.
- **Branchless**: All operations are constant-time by construction. Validated via DudeCT.

## Testing

```bash
cargo test                          # 123 tests
cargo run --example mldsa_ntt       # PQ demo
cargo bench --bench pq_bench        # PQ benchmarks
cargo bench --bench vs_concrete_ntt # Competitive benchmarks
```

## Benchmarking

Run your own benchmarks:

```bash
cargo bench --bench ntt32_bench     # NTT32 full suite
cargo bench --bench ntt64_bench     # NTT64 full suite
cargo bench --bench pq_bench        # Post-quantum primes
cargo bench --bench vs_concrete_ntt # vs concrete-ntt
```

Benchmark results depend on hardware and system load. We recommend running
on your target platform with CPU frequency scaling disabled for reproducible results.

## License

This project is dual-licensed:

- **Open Source**: [AGPL-3.0-or-later](LICENSE) — free for open-source projects
- **Commercial**: Proprietary license available for closed-source usage

All code in this repository is original work by the authors. No code is derived
from or copied from any other NTT/crypto library.

For commercial licensing inquiries: alexis@veae.io
