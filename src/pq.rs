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


//! # Post-Quantum Cryptography Presets
//!
//! Pre-configured NTT contexts for NIST post-quantum standards.
//! One import, one line of code — instant access to ML-DSA and custom lattice schemes.
//!
//! ```
//! use vaea_ntt::pq::{PqScheme, PqNtt};
//!
//! // ML-DSA-65 (NIST Level 3 digital signatures)
//! let ntt = PqNtt::new(PqScheme::MlDsa65);
//! let mut poly = vec![0u32; ntt.n()];
//! poly[0] = 42;
//! ntt.forward(&mut poly);
//! ntt.inverse(&mut poly);
//! assert_eq!(poly[0], 42);
//! ```
//!
//! ## Why this matters
//!
//! Other NTT libraries are single-scheme:
//! - `mlkem-native` → ML-KEM only (q=3329, int16, incomplete NTT)
//! - `pqcrystals-dilithium` → ML-DSA only
//! - SEAL/OpenFHE → FHE only, no ARM NEON
//!
//! **VaeaNTT covers ML-DSA + custom lattice + FHE with a single engine**, NEON-optimized.
//!
//! ## Supported schemes
//!
//! | Scheme | Standard | q | N | Notes |
//! |--------|----------|---|---|-------|
//! | ML-DSA-44 | FIPS 204 | 8380417 | 256 | Full negacyclic NTT |
//! | ML-DSA-65 | FIPS 204 | 8380417 | 256 | Full negacyclic NTT |
//! | ML-DSA-87 | FIPS 204 | 8380417 | 256 | Full negacyclic NTT |
//!
//! ### ML-KEM Note
//!
//! ML-KEM uses q=3329 with N=256, but its NTT is an **incomplete NTT** (size-128
//! NTT over coefficient pairs), not a standard size-256 negacyclic NTT. This is
//! because q−1 = 3328 = 2⁸×13 only has a 256th root of unity, not a 512th.
//! A dedicated ML-KEM module with incomplete NTT support is planned.

use crate::ntt32::Ntt32Context;

// ===========================================================================
// PqScheme — Post-Quantum scheme selector
// ===========================================================================

/// NIST post-quantum cryptographic scheme.
///
/// Each variant fully specifies the NTT parameters (N, q) for a given
/// standard, eliminating the risk of misconfiguration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PqScheme {
    // ----- FIPS 204: ML-DSA (Module-Lattice Digital Signature) -----

    /// ML-DSA-44 — NIST Level 2 (128-bit classical security)
    ///
    /// (k,l) = (4,4), N=256, q=8380417.
    MlDsa44,

    /// ML-DSA-65 — NIST Level 3 (192-bit classical security)
    ///
    /// (k,l) = (6,5), N=256, q=8380417.
    MlDsa65,

    /// ML-DSA-87 — NIST Level 5 (256-bit classical security)
    ///
    /// (k,l) = (8,7), N=256, q=8380417.
    MlDsa87,
}

impl PqScheme {
    /// Returns the polynomial degree N for this scheme.
    #[inline]
    pub const fn n(self) -> usize {
        match self {
            Self::MlDsa44 | Self::MlDsa65 | Self::MlDsa87 => 256,
        }
    }

    /// Returns the prime modulus q for this scheme.
    #[inline]
    pub const fn q(self) -> u32 {
        match self {
            Self::MlDsa44 | Self::MlDsa65 | Self::MlDsa87 => 8380417,
        }
    }

    /// Returns the module rank k (number of polynomials).
    #[inline]
    pub const fn k(self) -> usize {
        match self {
            Self::MlDsa44 => 4,
            Self::MlDsa65 => 6,
            Self::MlDsa87 => 8,
        }
    }

    /// Returns the NIST security level (1–5).
    #[inline]
    pub const fn security_level(self) -> u8 {
        match self {
            Self::MlDsa44 => 2,
            Self::MlDsa65 => 3,
            Self::MlDsa87 => 5,
        }
    }

    /// Returns a human-readable name for this scheme.
    #[inline]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MlDsa44 => "ML-DSA-44",
            Self::MlDsa65 => "ML-DSA-65",
            Self::MlDsa87 => "ML-DSA-87",
        }
    }

    /// Returns the NIST FIPS standard number.
    #[inline]
    pub const fn fips(self) -> &'static str {
        match self {
            Self::MlDsa44 | Self::MlDsa65 | Self::MlDsa87 => "FIPS 204",
        }
    }
}

// ===========================================================================
// PqNtt — Post-Quantum NTT engine
// ===========================================================================

/// A ready-to-use NTT engine configured for a specific post-quantum scheme.
///
/// Wraps [`Ntt32Context`] with scheme metadata for safety and convenience.
///
/// # Example
///
/// ```
/// use vaea_ntt::pq::{PqScheme, PqNtt};
///
/// let ntt = PqNtt::new(PqScheme::MlDsa65);
/// assert_eq!(ntt.scheme(), PqScheme::MlDsa65);
/// assert_eq!(ntt.n(), 256);
/// assert_eq!(ntt.q(), 8380417);
///
/// let mut data = vec![0u32; 256];
/// data[0] = 1;
/// ntt.forward(&mut data);
/// ntt.inverse(&mut data);
/// assert_eq!(data[0], 1);
/// ```
pub struct PqNtt {
    /// The underlying NTT context.
    ctx: Ntt32Context,
    /// The scheme this context was created for.
    scheme: PqScheme,
}

impl PqNtt {
    /// Creates a new PQ-NTT engine for the given scheme.
    ///
    /// This precomputes all twiddle factors and modular arithmetic
    /// constants. The context can be reused for multiple NTT calls.
    #[inline]
    pub fn new(scheme: PqScheme) -> Self {
        let ctx = Ntt32Context::new(scheme.n(), scheme.q());
        Self { ctx, scheme }
    }

    /// Returns the scheme this engine was configured for.
    #[inline]
    pub fn scheme(&self) -> PqScheme {
        self.scheme
    }

    /// Returns the polynomial degree N.
    #[inline]
    pub fn n(&self) -> usize {
        self.ctx.n
    }

    /// Returns the prime modulus q.
    #[inline]
    pub fn q(&self) -> u32 {
        self.ctx.q
    }

    /// Returns the NIST security level.
    #[inline]
    pub fn security_level(&self) -> u8 {
        self.scheme.security_level()
    }

    /// Returns a reference to the underlying [`Ntt32Context`].
    #[inline]
    pub fn context(&self) -> &Ntt32Context {
        &self.ctx
    }

    /// Applies forward NTT in-place.
    ///
    /// Transforms from coefficient domain to evaluation (NTT) domain.
    /// In NTT domain, polynomial multiplication is pointwise O(N).
    ///
    /// # Panics
    /// If `data.len() != self.n()`.
    #[inline]
    pub fn forward(&self, data: &mut [u32]) {
        self.ctx.forward(data);
    }

    /// Applies inverse NTT in-place.
    ///
    /// Transforms from evaluation (NTT) domain back to coefficient domain.
    /// Includes the N⁻¹ normalization factor.
    ///
    /// # Panics
    /// If `data.len() != self.n()`.
    #[inline]
    pub fn inverse(&self, data: &mut [u32]) {
        self.ctx.inverse(data);
    }

    /// Computes negacyclic polynomial multiplication: `result = a × b mod (X^N + 1, q)`.
    ///
    /// Both inputs must be in coefficient domain (not NTT).
    /// Result is in coefficient domain.
    ///
    /// # Panics
    /// If `a.len() != self.n()` or `b.len() != self.n()`.
    #[inline]
    pub fn multiply(&self, a: &[u32], b: &[u32]) -> alloc::vec::Vec<u32> {
        self.ctx.negacyclic_mul(a, b)
    }

    /// Computes negacyclic polynomial multiplication: `result = a × b mod (X^N + 1, q)`.
    ///
    /// Both `a` and `b` are consumed (transformed in-place as scratch space).
    /// Result is written to `result`.
    ///
    /// # Panics
    /// If any slice length != `self.n()`.
    #[inline]
    pub fn multiply_into(&self, a: &mut [u32], b: &mut [u32], result: &mut [u32]) {
        self.ctx.negacyclic_mul_into(a, b, result);
    }
}

impl core::fmt::Debug for PqNtt {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PqNtt")
            .field("scheme", &self.scheme.name())
            .field("n", &self.n())
            .field("q", &self.q())
            .field("security_level", &self.security_level())
            .field("fips", &self.scheme.fips())
            .finish()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mldsa_44_roundtrip() {
        let ntt = PqNtt::new(PqScheme::MlDsa44);
        assert_eq!(ntt.q(), 8380417);
        assert_eq!(ntt.n(), 256);
        assert_eq!(ntt.security_level(), 2);

        let mut data: alloc::vec::Vec<u32> = (0..256).map(|i| i * 1000 % 8380417).collect();
        let original = data.clone();
        ntt.forward(&mut data);
        assert_ne!(data, original, "NTT forward did nothing");
        ntt.inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_mldsa_65_roundtrip() {
        let ntt = PqNtt::new(PqScheme::MlDsa65);
        assert_eq!(ntt.security_level(), 3);

        let mut data = alloc::vec![8380416u32; 256]; // q-1
        let original = data.clone();
        ntt.forward(&mut data);
        ntt.inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_mldsa_87_roundtrip() {
        let ntt = PqNtt::new(PqScheme::MlDsa87);
        assert_eq!(ntt.security_level(), 5);

        let mut data = alloc::vec![0u32; 256];
        data[0] = 1;
        let original = data.clone();
        ntt.forward(&mut data);
        ntt.inverse(&mut data);
        assert_eq!(data, original);
    }

    #[test]
    fn test_multiply() {
        let ntt = PqNtt::new(PqScheme::MlDsa44);
        let q = ntt.q();

        // Multiply (1 + x) × (1 + x) mod (x^256 + 1, q)
        // = 1 + 2x + x^2
        let mut a = alloc::vec![0u32; 256];
        a[0] = 1;
        a[1] = 1;
        let result = ntt.multiply(&a, &a);
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 1);
        for i in 3..256 {
            assert_eq!(result[i], 0, "unexpected non-zero at index {i}");
        }
    }

    #[test]
    fn test_scheme_metadata() {
        assert_eq!(PqScheme::MlDsa44.name(), "ML-DSA-44");
        assert_eq!(PqScheme::MlDsa44.fips(), "FIPS 204");
        assert_eq!(PqScheme::MlDsa44.k(), 4);

        assert_eq!(PqScheme::MlDsa65.name(), "ML-DSA-65");
        assert_eq!(PqScheme::MlDsa65.k(), 6);

        assert_eq!(PqScheme::MlDsa87.name(), "ML-DSA-87");
        assert_eq!(PqScheme::MlDsa87.k(), 8);
    }

    #[test]
    fn test_output_fully_reduced() {
        let ntt = PqNtt::new(PqScheme::MlDsa65);
        let mut data: alloc::vec::Vec<u32> =
            (0..ntt.n()).map(|i| (i as u32 * 7 + 13) % ntt.q()).collect();
        ntt.forward(&mut data);
        assert!(
            data.iter().all(|&x| x < ntt.q()),
            "Output not fully reduced for ML-DSA-65"
        );
    }

    #[test]
    fn test_debug_display() {
        let ntt = PqNtt::new(PqScheme::MlDsa65);
        let debug = alloc::format!("{:?}", ntt);
        assert!(debug.contains("ML-DSA-65"));
        assert!(debug.contains("FIPS 204"));
    }
}
