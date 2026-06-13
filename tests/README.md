# tests/

Integration and security tests.

| Test | Description | Command |
|------|-------------|---------|
| `ntt32_integration` | NTT32 roundtrip, reduction, multiplication for all supported N | `cargo test --test ntt32_integration` |
| `ntt64_integration` | NTT64 roundtrip and arithmetic | `cargo test --test ntt64_integration` |
| `constant_time` | Constant-time property tests | `cargo test --test constant_time` |
| `attack_vectors` | Security attack vector tests | `cargo test --test attack_vectors` |

Run all tests:

```bash
cargo test --release
```
