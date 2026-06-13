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

//! # VaeaNTT — High-Performance Number Theoretic Transforms
//!
//! **VaeaNTT** is a production-grade NTT engine for lattice-based cryptography
//! and Fully Homomorphic Encryption (FHE), optimized for ARM NEON (aarch64)
//! with a portable scalar fallback.
//!
//! ## What is NTT and why does it matter?
//!
//! The **Number Theoretic Transform** is the finite-field analogue of the FFT.
//! It maps a polynomial from coefficient representation to evaluation
//! representation in O(N log N) over Z_q, where q is a prime modulus. In
//! the evaluation domain, polynomial multiplication becomes pointwise — O(N)
//! instead of the naïve O(N²). This makes NTT the critical hot-path in:
//!
//! - **Post-quantum cryptography** — ML-DSA (FIPS 204, formerly Dilithium)
//!   and other NIST lattice standards multiply polynomials in the ring
//!   Z_q\[X\]/(X^N+1) via NTT.
//! - **Fully Homomorphic Encryption (FHE)** — CKKS, BFV, and BGV schemes
//!   (SEAL, OpenFHE) rely on multi-prime RNS-NTT with 60–62 bit primes.
//!
//! VaeaNTT covers both use-cases with a single engine: 28-bit NTT for
//! post-quantum signatures and 64-bit NTT for FHE workloads.
//!
//! ## Quick Start
//!
//! ```
//! use vaea_ntt::ntt32::Ntt32Context;
//!
//! // Any NTT-friendly prime < 2^28
//! let ctx = Ntt32Context::new(256, 8_380_417);
//!
//! let mut data = vec![42u32; 256];
//! ctx.forward(&mut data);
//! ctx.inverse(&mut data);
//! assert!(data.iter().all(|&x| x == 42));
//! ```
//!
//! ## Architecture Overview
//!
//! | Module | Pipeline | Description |
//! |--------|----------|-------------|
//! | [`ntt32`] | 28-bit primes (< 2²⁸) | ARM NEON vectorized butterfly (Shoup + Harvey lazy reduction). Scalar fallback on non-aarch64 targets. |
//! | [`ntt64`] | 60–62 bit primes | Barrett reduction. SEAL/OpenFHE-compatible. Cooley-Tukey forward, Gentleman-Sande inverse. |
//! | [`poly`] | Polynomial ring | Polynomials over Z_q\[X\]/(X^N+1) with 64-bit coefficients. Tracks coefficient/NTT domain. |
//! | [`rns`] | Multi-prime CRT | Residue Number System decomposition for FHE. Component-wise NTT on each limb. |
//! | [`pq`] | NIST presets | One-line constructors for ML-DSA-44/65/87 (FIPS 204). |
//!
//! ## Negacyclic Polynomial Multiplication (ntt32)
//!
//! Multiply two polynomials in Z_q\[X\]/(X^N+1) using the 28-bit pipeline:
//!
//! ```
//! use vaea_ntt::ntt32::{Ntt32Context, generate_primes_28};
//!
//! // Generate an NTT-friendly prime for N = 256
//! let primes = generate_primes_28(256, 1);
//! let q = primes[0];
//! let ctx = Ntt32Context::new(256, q);
//!
//! // a(X) = 1 + 2X
//! let mut a = vec![0u32; 256];
//! a[0] = 1;
//! a[1] = 2;
//!
//! // b(X) = 3 + X
//! let mut b = vec![0u32; 256];
//! b[0] = 3;
//! b[1] = 1;
//!
//! // c(X) = a(X) · b(X) mod (X^256 + 1, q) = 3 + 7X + 2X²
//! let c = ctx.negacyclic_mul(&a, &b);
//! assert_eq!(c[0], 3);
//! assert_eq!(c[1], 7);
//! assert_eq!(c[2], 2);
//! for i in 3..256 {
//!     assert_eq!(c[i], 0);
//! }
//! ```
//!
//! ## 64-bit Pipeline (FHE)
//!
//! For FHE workloads requiring 60–62 bit primes, use [`ntt64::Ntt64Arith`]
//! and [`ntt64::Ntt64Context`]:
//!
//! ```
//! use vaea_ntt::ntt64::{Ntt64Arith, Ntt64Context, generate_primes_60};
//!
//! // Generate a 60-bit NTT-friendly prime for N = 4096
//! let primes = generate_primes_60(4096, 60, 1);
//! let arith = Ntt64Arith::new(primes[0]);
//! let ctx = Ntt64Context::new(4096, arith);
//!
//! // Forward + inverse roundtrip
//! let mut data = vec![0u64; 4096];
//! data[0] = 42;
//! data[1] = 100;
//! let original = data.clone();
//!
//! ctx.forward(&mut data);
//! assert_ne!(data, original); // now in NTT domain
//! ctx.inverse(&mut data);
//! assert_eq!(data, original); // roundtrip restored
//! ```
//!
//! ## Post-Quantum Presets
//!
//! Zero-configuration NTT for NIST post-quantum standards:
//!
//! ```
//! use vaea_ntt::pq::{PqScheme, PqNtt};
//!
//! // ML-DSA-65 (FIPS 204) — digital signatures, NIST Level 3
//! let ntt = PqNtt::new(PqScheme::MlDsa65);
//! assert_eq!(ntt.n(), 256);
//! assert_eq!(ntt.q(), 8_380_417);
//! assert_eq!(ntt.security_level(), 3);
//!
//! // Multiply two polynomials
//! let mut a = vec![0u32; 256];
//! a[0] = 5;
//! let mut b = vec![0u32; 256];
//! b[0] = 7;
//! let c = ntt.multiply(&a, &b);
//! assert_eq!(c[0], 35);
//! ```
//!
//! ## Modules
//!
//! | Module | Use case |
//! |--------|----------|
//! | [`pq`] | Post-quantum presets for ML-DSA (FIPS 204) |
//! | [`ntt32`] | 28-bit primes (< 2²⁸), ARM NEON vectorized |
//! | [`ntt64`] | 60–62 bit primes for FHE (SEAL/OpenFHE compatible) |
//! | [`poly`] | Polynomials over Z_q\[X\]/(X^N+1) with 64-bit coefficients |
//! | [`rns`] | Multi-prime CRT (Residue Number System) |
//!
//! ## Performance
//!
//! Measured on Apple M3 (single core), `ntt32` pipeline:
//!
//! | Operation | N = 256 | Throughput |
//! |-----------|---------|------------|
//! | Forward NTT | 240 ns | 1.07 billion coeff/s |
//! | Negacyclic multiply | 940 ns | — |
//!
//! The `ntt32` pipeline uses the Shoup precomputed-quotient trick with
//! Harvey lazy butterfly reductions. On aarch64, all butterfly stages are
//! NEON-vectorized (4×u32 lanes). The scalar fallback is used on other
//! architectures.
//!
//! ## Security
//!
//! All arithmetic and butterfly routines are **constant-time**:
//!
//! - **No data-dependent branches** — modular reductions use branchless
//!   wrapping arithmetic ([`u32::wrapping_add`] / [`u32::wrapping_sub`]).
//! - **No secret-dependent memory access patterns** — twiddle factor
//!   indexing depends only on the public transform size N.
//! - Safe to use on secret polynomial coefficients (e.g. ML-DSA signing keys).
//!
//! ## Features
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `std` | **on** | Enables [`std::error::Error`] impl on [`NttError`] |
//! | `rand` | off | Random polynomial generation |
//! | `ffi` | off | C/C++/JS bindings via Diplomat |
//!
//! ## `no_std`
//!
//! This crate is `no_std` compatible (requires `alloc`).
//! Disable default features to use without `std`.
//!
//! ## License
//!
//! VaeaNTT is dual-licensed:
//!
//! - **AGPL-3.0-or-later** for open-source use.
//! - **Commercial license** available from [Vaea SAS](https://vaea.tech)
//!   for proprietary / closed-source integration.

#![no_std]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

/// Errors returned by NTT context construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NttError {
    /// N must be a power of 2 >= 2.
    InvalidSize(usize),
    /// q must be prime.
    NotPrime(u64),
    /// q must satisfy q ≡ 1 (mod 2N) for NTT.
    NotNttFriendly {
        /// The modulus that failed the NTT-friendly check.
        q: u64,
        /// The polynomial size N.
        n: usize,
    },
    /// q must be < 2^28 for the 32-bit pipeline.
    PrimeTooLarge(u64),
}

impl core::fmt::Display for NttError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NttError::InvalidSize(n) => write!(f, "N={n} must be a power of 2 >= 2"),
            NttError::NotPrime(q) => write!(f, "q={q} is not prime"),
            NttError::NotNttFriendly { q, n } => {
                write!(f, "q={q} does not satisfy q ≡ 1 (mod {})", 2 * n)
            }
            NttError::PrimeTooLarge(q) => write!(f, "q={q} must be < 2^28"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for NttError {}

pub mod ntt32;
pub mod ntt64;
pub mod poly;
pub mod pq;
pub mod rns;

#[cfg(feature = "ffi")]
pub mod ffi;
