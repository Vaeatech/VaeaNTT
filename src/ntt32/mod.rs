// Copyright (C) 2024-2026 Vaea SAS
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of VaeaNTT.
//
// VaeaNTT is free software: you can redistribute it and/or modify it under
// the terms of the GNU Affero General Public License as published by the
// Free Software Foundation, either version 3 of the License, or (at your
// option) any later version.
//
// VaeaNTT is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
// License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with VaeaNTT. If not, see <https://www.gnu.org/licenses/>.

//! # ntt32 — 28-bit NTT Pipeline
//!
//! High-performance Number Theoretic Transform for primes < 2^28.
//!
//! ## Architecture
//!
//! | Module     | Description |
//! |------------|-------------|
//! | `arith`    | Branchless modular arithmetic (add, sub, mul, pow, inv) |
//! | `prime`    | NTT-friendly prime generation and primitive root finding |
//! | `scalar`   | Scalar NTT with Shoup trick + Harvey lazy butterfly |
//! | `neon`     | NEON-vectorized NTT (aarch64 only, all stages) |
//! | `context`  | Unified `Ntt32Context` with automatic NEON/scalar dispatch |
//!
//! ## Quick Start
//!
//! ```
//! use vaea_ntt::ntt32::{Ntt32Context, generate_primes_28};
//!
//! let primes = generate_primes_28(1024, 1);
//! let ctx = Ntt32Context::new(1024, primes[0]);
//!
//! let a = vec![1u32; 1024];
//! let b = vec![2u32; 1024];
//! let product = ctx.negacyclic_mul(&a, &b);
//! assert_eq!(product.len(), 1024);
//! ```
//!
//! ## Performance
//!
//! Measured on Apple M3 (single core):
//!
//! | Operation              | N = 256 | Throughput               |
//! |------------------------|---------|--------------------------|
//! | Forward NTT            | 240 ns  | 1.07 billion coeff/s     |
//! | Negacyclic multiply    | 940 ns  | —                        |
//!
//! ## Security
//!
//! All operations are **constant-time**:
//!
//! - No data-dependent branches in any arithmetic or butterfly routine.
//! - Modular reductions use branchless wrapping arithmetic
//!   ([`u32::wrapping_add`] / [`u32::wrapping_sub`]).
//! - Safe to use on secret polynomial coefficients (e.g. ML-DSA keys).
//!
//! ## Primes
//!
//! NTT-friendly primes must satisfy two constraints:
//!
//! 1. **Bit-width**: `q < 2^28` (required by the Shoup / Harvey reduction).
//! 2. **Divisibility**: `q ≡ 1 (mod 2N)` so that a principal 2N-th root
//!    of unity exists in `Z_q`.
//!
//! Use [`generate_primes_28`] to find valid primes for a given `N`:
//!
//! ```
//! use vaea_ntt::ntt32::generate_primes_28;
//!
//! let primes = generate_primes_28(256, 3);
//! assert_eq!(primes.len(), 3);
//! for &q in &primes {
//!     assert!(q < (1 << 28));
//!     assert_eq!(q % (2 * 256), 1);
//! }
//! ```
//!
//! The ML-DSA (Dilithium) prime **8 380 417** is NTT-friendly for `N = 256`
//! (`8_380_417 % 512 == 1`).
//!
//! ## Forward / Inverse Roundtrip
//!
//! ```
//! use vaea_ntt::ntt32::Ntt32Context;
//!
//! let ctx = Ntt32Context::new(256, 8_380_417);
//! let mut data = vec![0u32; 256];
//! data[0] = 1;
//! data[1] = 2;
//! let original = data.clone();
//!
//! ctx.forward(&mut data);
//! // data is now in NTT domain
//! assert_ne!(data, original);
//!
//! ctx.inverse(&mut data);
//! // roundtrip: data restored
//! assert_eq!(data, original);
//! ```

pub mod arith;
pub mod context;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod prime;
pub mod scalar;

// Re-exports for convenience
pub use arith::{mod_add_28, mod_inv_32, mod_mul_28, mod_pow_32, mod_sub_28};
pub use context::Ntt32Context;
pub use prime::generate_primes_28;
pub use prime::{find_primitive_root, is_prime_32};
pub use scalar::{compute_shoup, shoup_mul};
