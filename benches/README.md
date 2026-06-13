# benches/

Criterion benchmarks for VaeaNTT.

## Available Benchmarks

| Bench | Description | Command |
|-------|-------------|---------|
| `ntt32_bench` | Forward/inverse NTT, multiplication (all N) | `cargo bench --bench ntt32_bench` |
| `ntt64_bench` | 64-bit NTT pipeline | `cargo bench --bench ntt64_bench` |
| `pq_bench` | Post-quantum presets (ML-DSA) | `cargo bench --bench pq_bench` |
| `vs_concrete_ntt` | Cross-validation with concrete-ntt (Zama) | `cargo bench --bench vs_concrete_ntt` |
| `vs_pqclean` | Cross-validation with PQClean reference C | `cargo bench --bench vs_pqclean` |
| `vs_libcrux` | Cross-validation with libcrux ML-KEM | `cargo bench --bench vs_libcrux` |
| `butterfly_lab` | Isolated butterfly micro-benchmarks | `cargo bench --bench butterfly_lab` |
| `ntt_lab` | Experimental NTT variants | `cargo bench --bench ntt_lab` |

## Notes

- `vs_pqclean` and `vs_libcrux` require third-party C/ASM sources in `mlkem_ntt/` and `pqclean_ntt/`
  (not included in the repo, see benchmark source for setup instructions).
- All benchmarks use [criterion](https://crates.io/crates/criterion) with HTML reports.
- Run with `--release` for accurate results (default for `cargo bench`).
- Disable CPU frequency scaling for reproducible measurements.
