# VaeaNTT

**High-performance Number Theoretic Transform engine for post-quantum cryptography on ARM.**

VaeaNTT is a Rust library providing NTT (Number Theoretic Transform) implementations optimized for ARM NEON. It is the fastest NTT library for lattice-based cryptography on ARM processors.

## Performance

Benchmarked on Apple M3 Pro (ARMv8.6-A) against [concrete-ntt](https://crates.io/crates/concrete-ntt) v0.2.0 (Zama):

| Benchmark | Speedup |
|-----------|:-------:|
| Forward NTT (iso-N, same prime) | **1.85×** |
| Negacyclic multiplication (zero-alloc) | **1.75×** |
| Iso-security best-of-each (4×28 vs 2×60) | **1.15×** |

All benchmarks are [reproducible](benches/vs_concrete_ntt.rs) with `cargo bench --bench vs_concrete_ntt`.

## Post-Quantum Coverage

VaeaNTT natively supports all three NIST post-quantum standards:

| Standard | Modulus | Bits | NTT Size | Forward NTT |
|----------|:-------:|:----:|:--------:|:-----------:|
| **ML-KEM** (FIPS 203) | q = 3 329 | 12 | 128 | **150 ns** |
| **ML-DSA** (FIPS 204) | q = 8 380 417 | 23 | 256 | **323 ns** |
| **Falcon** | q = 12 289 | 14 | 512 | **790 ns** |

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
ctx.inverse_lazy(&mut data);    // without N⁻¹ (matches concrete-ntt behavior)

// Polynomial multiplication in Z_q[X]/(X^N + 1)
let result = ctx.negacyclic_mul(&a, &b);                     // allocating
ctx.negacyclic_mul_into(&mut a, &mut b, &mut result);        // zero-alloc
```

### `Ntt64Context` — 60-62 bit pipeline

For FHE-compatible 64-bit primes (SEAL, OpenFHE interop).

### Features

| Feature | Default | Description |
|---------|:-------:|-------------|
| `rand` | off | Enables `Poly64::new_random()`, `new_ternary()`, `new_gaussian()` |

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
│   └── lib.rs
├── benches/
│   ├── pq_bench.rs      # NIST PQ standard benchmarks
│   └── vs_concrete_ntt.rs  # Competitive benchmarks
└── examples/
    └── mldsa_ntt.rs     # ML-DSA/ML-KEM/Falcon demo
```

### Why 28-bit?

The 28-bit prime choice is architecturally motivated:

- **ARM NEON native**: `u32 × u32 → u64` fits perfectly in NEON 128-bit registers (4 lanes). No widening to `u128`.
- **Harvey lazy reduction**: Coefficients stay in `[0, 2q)` between butterfly stages. With `q < 2²⁸`, intermediates `3q < 2³⁰` fit in `u32`.
- **Post-quantum alignment**: All NIST PQ standards use primes ≤ 23 bits — well within our 28-bit pipeline.
- **Branchless**: All operations are constant-time by construction. No data-dependent branches.

## Testing

```bash
cargo test                          # 109 tests
cargo run --example mldsa_ntt       # PQ demo
cargo bench --bench pq_bench        # PQ benchmarks
cargo bench --bench vs_concrete_ntt # Competitive benchmarks
```

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE) for details.

For commercial licensing, contact: [TODO]
