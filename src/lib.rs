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
//! NTT engine for lattice-based cryptography, optimized for ARM NEON (aarch64)
//! with portable scalar fallback.
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
//! ## Post-Quantum Presets
//!
//! ```
//! use vaea_ntt::pq::{PqScheme, PqNtt};
//!
//! // ML-DSA-65 (FIPS 204) — digital signatures, NIST Level 3
//! let ntt = PqNtt::new(PqScheme::MlDsa65);
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
pub mod pq;
pub mod poly;
pub mod rns;

#[cfg(feature = "ffi")]
pub mod ffi;
