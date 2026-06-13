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

//! # 64-bit NTT Pipeline
//!
//! High-performance NTT for 60–62 bit NTT-friendly primes, compatible with
//! SEAL, OpenFHE, and general FHE libraries.
//!
//! ## Modules
//!
//! - [`arith`] — Modular arithmetic (Barrett, Montgomery, add/sub)
//! - [`prime`] — Prime generation and primality testing
//! - [`context`] — NTT context with precomputed twiddle tables

pub mod arith;
pub mod context;
pub mod prime;

// Re-exports for convenience
pub use arith::{
    from_montgomery, mod_add, mod_inv, mod_mul_barrett, mod_mul_mont, mod_pow, mod_sub,
    to_montgomery, Ntt64Arith, PRIME_60_1, PRIME_60_2, PRIME_60_3, PRIME_62_1, PRIME_SEAL,
};

pub use prime::{find_primitive_root, generate_primes_60, is_prime};

pub use context::Ntt64Context;
