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
//! High-performance Number Theoretic Transform for 60–62 bit NTT-friendly
//! primes, targeting Fully Homomorphic Encryption (FHE) workloads and
//! large-field lattice cryptography. All arithmetic is performed on `u64`
//! values with primes up to ~62 bits (strictly < 2⁶²).
//!
//! ## Architecture
//!
//! **Modular arithmetic** is provided via two complementary strategies:
//!
//! - **Barrett reduction** ([`mod_mul_barrett`]) — division-free modular
//!   multiplication using a precomputed constant μ = ⌊2¹²⁸/q⌋. Used as the
//!   default reduction throughout the NTT butterfly loops.
//! - **Montgomery reduction** ([`mod_mul_mont`]) — efficient for long chains
//!   of multiplications in Montgomery domain (a·R mod q). Useful when the
//!   same element is multiplied many times before leaving the domain.
//!
//! **NTT butterflies** follow the standard radix-2 split:
//!
//! - **Forward NTT** — Cooley-Tukey Decimation-In-Time (DIT), layers from
//!   coarsest (gap = N/2) to finest (gap = 1).
//! - **Inverse NTT** — Gentleman-Sande Decimation-In-Frequency (DIF), layers
//!   from finest to coarsest, with a final N⁻¹ mod q normalization pass.
//!
//! **Twiddle ordering** uses the Longa-Naehrig layout (as in SEAL and
//! OpenFHE): twiddle factors are stored in bit-reversed order so that each
//! butterfly layer accesses them sequentially, yielding good cache behaviour.
//! The transform implements *negacyclic* convolution in Z\_q\[X\]/(X^N+1)
//! directly by folding the ψ (2N-th root of unity) factors into the twiddle
//! table.
//!
//! ## Quick Start
//!
//! ```
//! use vaea_ntt::ntt64::{Ntt64Arith, Ntt64Context, generate_primes_60};
//!
//! // Generate one 60-bit NTT-friendly prime for N = 1024
//! let primes = generate_primes_60(1024, 60, 1);
//! let arith  = Ntt64Arith::new(primes[0]);
//! let ctx    = Ntt64Context::new(1024, arith);
//!
//! let mut data = vec![0u64; 1024];
//! data[0] = 42;
//!
//! // Forward NTT (polynomial → evaluation domain)
//! ctx.forward(&mut data);
//!
//! // Inverse NTT (evaluation domain → polynomial)
//! ctx.inverse(&mut data);
//!
//! assert_eq!(data[0], 42);
//! assert!(data[1..].iter().all(|&x| x == 0));
//! ```
//!
//! ## Pre-defined Primes
//!
//! | Constant | Value | Bits | Max N | Origin |
//! |----------|-------|------|-------|--------|
//! | [`PRIME_60_1`] | 1 152 921 504 606 584 833 | 60 | 32 768 | k·2¹⁶+1 |
//! | [`PRIME_60_2`] | 576 460 752 308 273 153 | 60 | 32 768 | k·2¹⁶+1 |
//! | [`PRIME_60_3`] | 576 460 752 312 401 921 | 60 | 32 768 | k·2¹⁶+1 |
//! | [`PRIME_62_1`] | 4 611 686 018 326 724 609 | 62 | 32 768 | k·2¹⁶+1 |
//! | [`PRIME_SEAL`] | 0x1FFF_FFFF_FFE0_0001 | 61 | 1 048 576 | 2⁶¹−2²¹+1 (SEAL) |
//!
//! ## Use Cases
//!
//! - **FHE libraries** — drop-in NTT for SEAL/OpenFHE-compatible prime
//!   towers (BFV, BGV, CKKS schemes).
//! - **Large-field lattice crypto** — any scheme requiring polynomial
//!   arithmetic in Z\_q\[X\]/(X^N+1) with q up to ~62 bits.
//! - **Multi-prime RNS** — combine several 60-bit primes via the [`crate::rns`]
//!   module for coefficient moduli exceeding 64 bits.
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
