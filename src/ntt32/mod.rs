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
