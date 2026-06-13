# examples/

Runnable examples and test suites.

## Examples

| Example | Description | Command |
|---------|-------------|---------|
| `mldsa_ntt` | ML-DSA NTT demo | `cargo run --release --example mldsa_ntt` |
| `exhaustive_test` | 2618 tests across all N×q×pattern combinations | `cargo run --release --example exhaustive_test` |
| `verify_no_false_positive` | Proves tests aren't trivially passing (linearity, convolution theorem, NEON vs scalar) | `cargo run --release --example verify_no_false_positive` |
| `security_exploits` | Security exploit suite (out-of-range inputs, DoS, timing, overflow, thread safety) | `cargo run --release --example security_exploits` |
| `dudect_ntt` | Constant-time validation via DudeCT | `cargo run --release --example dudect_ntt` |
| `bias_check` | Statistical bias check on NTT outputs | `cargo run --release --example bias_check` |
