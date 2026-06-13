//! # Ntt32Context — Unified NTT Context for 28-bit Primes
//!
//! Combines the root table from `NttSmallCtx` with the Shoup precomputed
//! quotients from `ShoupCtx` into a single unified context struct.
//!
//! The `forward()` and `inverse()` methods automatically dispatch to
//! NEON on `aarch64` targets, falling back to scalar code otherwise.

use super::prime::NttRootTable;
use super::scalar::compute_shoup;

// ===========================================================================
// Ntt32Context — the unified context
// ===========================================================================

/// Pre-computed NTT context for a single 28-bit prime.
///
/// Stores twiddle factors (root powers) in Longa-Naehrig ordering along
/// with their Shoup precomputed quotients for division-free multiplication.
///
/// # Usage
/// ```
/// use vaea_ntt::ntt32::{Ntt32Context, generate_primes_28};
///
/// let primes = generate_primes_28(1024, 1);
/// let ctx = Ntt32Context::new(1024, primes[0]);
///
/// let mut data = vec![0u32; 1024];
/// data[0] = 42;
/// ctx.forward(&mut data);   // NTT forward
/// ctx.inverse(&mut data);   // NTT inverse (data restored)
/// assert_eq!(data[0], 42);
/// ```
#[derive(Debug, Clone)]
pub struct Ntt32Context {
    /// Polynomial size (power of 2)
    pub n: usize,

    /// log2(n)
    pub log_n: u32,

    /// Prime < 2^28
    pub q: u32,

    /// 2 · q — precomputed for Harvey lazy butterfly
    pub two_q: u32,

    /// Forward root powers (Longa-Naehrig ordering)
    pub root_powers: Vec<u32>,

    /// Shoup quotients for forward root powers: `floor(root_powers[i] · 2^32 / q)`
    pub root_powers_shoup: Vec<u32>,

    /// Inverse root powers for INTT
    pub inv_root_powers: Vec<u32>,

    /// Shoup quotients for inverse root powers
    pub inv_root_powers_shoup: Vec<u32>,

    /// N^{-1} mod q — normalization factor for INTT
    pub n_inv: u32,

    /// Shoup quotient for n_inv
    pub n_inv_shoup: u32,
}

impl Ntt32Context {
    /// Fallible constructor for an NTT context for a 28-bit prime.
    ///
    /// Validates all preconditions and returns an error instead of panicking.
    ///
    /// # Arguments
    /// - `n` — polynomial size, must be a power of 2 ≥ 2
    /// - `q` — prime < 2^28, must satisfy `q ≡ 1 (mod 2N)`
    ///
    /// # Errors
    /// - [`crate::NttError::InvalidSize`] if `n` is not a power of 2 ≥ 2
    /// - [`crate::NttError::PrimeTooLarge`] if `q ≥ 2^28`
    /// - [`crate::NttError::NotPrime`] if `q` is not prime
    /// - [`crate::NttError::NotNttFriendly`] if `(q - 1)` is not divisible by `2N`
    pub fn try_new(n: usize, q: u32) -> Result<Self, crate::NttError> {
        if n < 2 || !n.is_power_of_two() {
            return Err(crate::NttError::InvalidSize(n));
        }
        if q >= (1u32 << 28) {
            return Err(crate::NttError::PrimeTooLarge(q as u64));
        }
        if !super::prime::is_prime_32(q) {
            return Err(crate::NttError::NotPrime(q as u64));
        }
        if !((q - 1) as usize).is_multiple_of(2 * n) {
            return Err(crate::NttError::NotNttFriendly { q: q as u64, n });
        }

        // All preconditions verified — build the root table
        let base = NttRootTable::new(n, q);

        let root_powers_shoup: Vec<u32> = base
            .root_powers
            .iter()
            .map(|&w| compute_shoup(w, q))
            .collect();

        let inv_root_powers_shoup: Vec<u32> = base
            .inv_root_powers
            .iter()
            .map(|&w| compute_shoup(w, q))
            .collect();

        let n_inv_shoup = compute_shoup(base.n_inv, q);

        Ok(Self {
            n,
            log_n: base.log_n,
            q,
            two_q: 2 * q,
            root_powers: base.root_powers,
            root_powers_shoup,
            inv_root_powers: base.inv_root_powers,
            inv_root_powers_shoup,
            n_inv: base.n_inv,
            n_inv_shoup,
        })
    }

    /// Creates a new NTT context for a 28-bit prime.
    ///
    /// Computes primitive roots, twiddle factors (Longa-Naehrig ordering),
    /// and precomputes all Shoup quotients.
    ///
    /// # Arguments
    /// - `n` — polynomial size, must be a power of 2 ≥ 2
    /// - `q` — prime < 2^28, must satisfy `q ≡ 1 (mod 2N)`
    ///
    /// # Panics
    /// - If `n` is not a power of 2 ≥ 2
    /// - If `q ≥ 2^28`
    /// - If `q` is not prime
    /// - If `(q - 1)` is not divisible by `2N`
    pub fn new(n: usize, q: u32) -> Self {
        Self::try_new(n, q).expect("Invalid NTT parameters")
    }

    /// Applies the NTT forward transform in-place.
    ///
    /// On `aarch64`, dispatches to the fully-vectorized NEON implementation.
    /// On other architectures, uses the scalar Shoup NTT.
    #[inline]
    pub fn forward(&self, data: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            super::neon::ntt_fwd_neon(data, self);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            super::scalar::ntt_forward_scalar(data, self);
        }
    }

    /// Applies the NTT inverse transform in-place.
    ///
    /// On `aarch64`, dispatches to the NEON implementation with lazy normalization.
    /// On other architectures, uses the scalar Shoup NTT.
    #[inline]
    pub fn inverse(&self, data: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            super::neon::ntt_inv_neon(data, self);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            super::scalar::ntt_inverse_scalar(data, self);
        }
    }

    /// Applies the NTT inverse transform **without** N⁻¹ normalization.
    ///
    /// Output coefficients are scaled by N relative to the true INTT.
    /// Use this when chaining operations where normalization can be deferred,
    /// or when matching libraries that don't normalize (e.g., concrete-ntt).
    #[inline]
    pub fn inverse_lazy(&self, data: &mut [u32]) {
        #[cfg(target_arch = "aarch64")]
        {
            super::neon::ntt_inv_neon_lazy(data, self);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            super::scalar::ntt_inverse_scalar_lazy(data, self);
        }
    }

    /// Returns N⁻¹ mod q — useful for manual normalization after `inverse_lazy()`.
    #[inline]
    pub fn n_inv(&self) -> u32 {
        self.n_inv
    }

    /// Returns the Shoup quotient for N⁻¹ — for manual Shoup normalization.
    #[inline]
    pub fn n_inv_shoup(&self) -> u32 {
        self.n_inv_shoup
    }

    /// Pointwise multiplication of two vectors in the NTT domain.
    ///
    /// Computes `result[i] = a[i] · b[i] mod q` for each coefficient.
    pub fn pointwise_mul(&self, a: &[u32], b: &[u32], result: &mut [u32]) {
        super::scalar::ntt_pointwise_mul_scalar(a, b, result, self.q, self.n);
    }

    /// Negacyclic polynomial multiplication in Z_q\[X\]/(X^N + 1).
    ///
    /// Computes `result = a · b mod (X^N + 1)` using forward NTT,
    /// pointwise multiplication, and inverse NTT.
    ///
    /// # Returns
    /// A new vector of length N containing the product.
    pub fn negacyclic_mul(&self, a: &[u32], b: &[u32]) -> Vec<u32> {
        let n = self.n;
        let mut a_buf = a.to_vec();
        let mut b_buf = b.to_vec();
        let mut result = vec![0u32; n];
        self.negacyclic_mul_into(&mut a_buf, &mut b_buf, &mut result);
        result
    }

    /// Zero-allocation negacyclic multiplication.
    ///
    /// The caller provides pre-allocated buffers:
    /// - `a_buf` / `b_buf`: input polynomials (overwritten with NTT-domain values)
    /// - `result`: output buffer for the product
    ///
    /// All buffers must have length N. After the call, `a_buf` and `b_buf`
    /// contain NTT-domain data (destroyed); `result` contains the product
    /// in coefficient domain.
    ///
    /// # Example
    /// ```
    /// use vaea_ntt::ntt32::{Ntt32Context, generate_primes_28};
    ///
    /// let primes = generate_primes_28(256, 1);
    /// let ctx = Ntt32Context::new(256, primes[0]);
    ///
    /// let mut a = vec![1u32; 256];
    /// let mut b = vec![2u32; 256];
    /// let mut result = vec![0u32; 256];
    ///
    /// ctx.negacyclic_mul_into(&mut a, &mut b, &mut result);
    /// // result now contains a·b mod (X^256 + 1)
    /// // a and b are now in NTT domain (overwritten)
    /// ```
    pub fn negacyclic_mul_into(&self, a_buf: &mut [u32], b_buf: &mut [u32], result: &mut [u32]) {
        let n = self.n;
        assert_eq!(a_buf.len(), n, "a_buf.len()={} != N={n}", a_buf.len());
        assert_eq!(b_buf.len(), n, "b_buf.len()={} != N={n}", b_buf.len());
        assert_eq!(result.len(), n, "result.len()={} != N={n}", result.len());

        self.forward(a_buf);
        self.forward(b_buf);
        self.pointwise_mul(a_buf, b_buf, result);
        self.inverse(result);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt32::prime::generate_primes_28;

    fn test_prime(n: usize) -> u32 {
        generate_primes_28(n, 1)[0]
    }

    fn make_test_data(n: usize, q: u32) -> Vec<u32> {
        (0..n)
            .map(|i| ((i as u64 * 314_159_265 + 271_828_182) % q as u64) as u32)
            .collect()
    }

    #[test]
    fn test_roundtrip_n16() {
        let n = 16;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        assert_ne!(data, original, "NTT forward did nothing for N={n}");
        ctx.inverse(&mut data);
        assert_eq!(data, original, "NTT roundtrip failed for N={n}");
    }

    #[test]
    fn test_roundtrip_n64() {
        let n = 64;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, original, "NTT roundtrip failed for N={n}");
    }

    #[test]
    fn test_roundtrip_n1024() {
        let n = 1024;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, original, "NTT roundtrip failed for N={n}");
    }

    #[test]
    fn test_roundtrip_n32768() {
        let n = 32768;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, original, "NTT roundtrip failed for N=32768");
    }

    #[test]
    fn test_roundtrip_zeros() {
        let n = 64;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let mut data = vec![0u32; n];
        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, vec![0u32; n]);
    }

    #[test]
    fn test_constant_polynomial() {
        // NTT of [c, 0, 0, ...] should give [c, c, c, ...]
        let n = 64;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let c = 42u32;
        let mut data = vec![0u32; n];
        data[0] = c;

        ctx.forward(&mut data);
        for (i, &v) in data.iter().enumerate() {
            assert_eq!(v, c, "NTT of constant: data[{i}]={v}, expected {c}");
        }
    }

    #[test]
    fn test_negacyclic_mul_identity() {
        // Multiply by [1, 0, 0, ...] should be identity
        let n = 64;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);

        let a: Vec<u32> = (0..n)
            .map(|i| ((i as u64 * 17 + 5) % q as u64) as u32)
            .collect();
        let mut one = vec![0u32; n];
        one[0] = 1;

        let result = ctx.negacyclic_mul(&a, &one);
        assert_eq!(result, a, "Multiply by 1 is not identity");
    }

    #[test]
    fn test_negacyclic_mul_n16() {
        let n = 16;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);

        let a: Vec<u32> = (0..n).map(|i| (i as u32 + 1) % q).collect();
        let b: Vec<u32> = vec![1u32; n];

        // Naive reference
        let mut expected = vec![0u32; n];
        for i in 0..n {
            for j in 0..n {
                let prod = (a[i] as u64 * b[j] as u64) % q as u64;
                if i + j < n {
                    expected[i + j] = ((expected[i + j] as u64 + prod) % q as u64) as u32;
                } else {
                    let idx = i + j - n;
                    expected[idx] = ((expected[idx] as u64 + q as u64 - prod) % q as u64) as u32;
                }
            }
        }

        let result = ctx.negacyclic_mul(&a, &b);
        assert_eq!(result, expected, "Negacyclic multiplication mismatch");
    }

    #[test]
    fn test_inverse_lazy_no_normalization() {
        let n = 256;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);

        // inverse_lazy should NOT equal original (missing N^{-1})
        let mut data = original.clone();
        ctx.forward(&mut data);
        ctx.inverse_lazy(&mut data);
        assert_ne!(
            data, original,
            "inverse_lazy should not match original (no N^{{-1}})"
        );

        // But after manual N^{-1} normalization, it should match
        let n_inv = ctx.n_inv();
        for x in data.iter_mut() {
            *x = ((*x as u64 * n_inv as u64) % q as u64) as u32;
        }
        assert_eq!(
            data, original,
            "inverse_lazy + manual N^{{-1}} should match original"
        );
    }

    #[test]
    fn test_inverse_lazy_matches_concrete_ntt_style() {
        // Verify that inverse_lazy() is exactly inverse() without N^{-1}
        let n = 1024;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);

        let mut data_full = original.clone();
        let mut data_lazy = original.clone();

        ctx.forward(&mut data_full);
        ctx.forward(&mut data_lazy);

        ctx.inverse(&mut data_full);
        ctx.inverse_lazy(&mut data_lazy);

        // data_lazy * N^{-1} should equal data_full
        let n_inv = ctx.n_inv();
        let data_lazy_normalized: Vec<u32> = data_lazy
            .iter()
            .map(|&x| ((x as u64 * n_inv as u64) % q as u64) as u32)
            .collect();
        assert_eq!(data_full, data_lazy_normalized);
    }

    #[test]
    fn test_negacyclic_mul_into_matches_negacyclic_mul() {
        let n = 256;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);

        let a: Vec<u32> = (0..n)
            .map(|i| ((i as u64 * 17 + 3) % q as u64) as u32)
            .collect();
        let b: Vec<u32> = (0..n)
            .map(|i| ((i as u64 * 31 + 7) % q as u64) as u32)
            .collect();

        // Allocating version
        let result_alloc = ctx.negacyclic_mul(&a, &b);

        // Zero-alloc version
        let mut a_buf = a.clone();
        let mut b_buf = b.clone();
        let mut result_inplace = vec![0u32; n];
        ctx.negacyclic_mul_into(&mut a_buf, &mut b_buf, &mut result_inplace);

        assert_eq!(
            result_alloc, result_inplace,
            "negacyclic_mul_into must match negacyclic_mul"
        );
    }

    #[test]
    fn test_negacyclic_mul_into_reusable_buffers() {
        // Verify that buffers can be reused across calls
        let n = 64;
        let q = test_prime(n);
        let ctx = Ntt32Context::new(n, q);

        let mut a_buf = vec![0u32; n];
        let mut b_buf = vec![0u32; n];
        let mut result = vec![0u32; n];

        for round in 0..3u32 {
            // Fill buffers with different data each round
            for i in 0..n {
                a_buf[i] = ((i as u64 * (round as u64 + 17) + 3) % q as u64) as u32;
                b_buf[i] = ((i as u64 * (round as u64 + 31) + 7) % q as u64) as u32;
            }
            let expected = ctx.negacyclic_mul(&a_buf, &b_buf);

            ctx.negacyclic_mul_into(&mut a_buf, &mut b_buf, &mut result);
            assert_eq!(
                result, expected,
                "Reusable buffer mismatch at round {round}"
            );

            // Re-fill for next round (a_buf/b_buf were destroyed)
        }
    }

    // ===================================================================
    // NIST Post-Quantum Standard Primes
    // ===================================================================

    #[test]
    fn test_pq_mldsa_roundtrip() {
        // ML-DSA (FIPS 204): q = 8380417 = 2^23 - 2^13 + 1, N = 256
        let q: u32 = 8_380_417;
        let n = 256;
        assert_eq!((q - 1) % (2 * n as u32), 0, "q-1 must be divisible by 2N");

        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        assert_ne!(data, original, "Forward NTT should change data");
        ctx.inverse(&mut data);
        assert_eq!(data, original, "ML-DSA roundtrip failed");
    }

    #[test]
    fn test_pq_mldsa_negacyclic_mul() {
        let q: u32 = 8_380_417;
        let n = 256;
        let ctx = Ntt32Context::new(n, q);

        // Multiply by [1, 0, 0, ...] should be identity
        let a: Vec<u32> = (0..n)
            .map(|i| ((i as u64 * 17 + 5) % q as u64) as u32)
            .collect();
        let mut one = vec![0u32; n];
        one[0] = 1;

        let result = ctx.negacyclic_mul(&a, &one);
        assert_eq!(result, a, "ML-DSA: multiply by 1 is not identity");
    }

    #[test]
    fn test_pq_falcon512_roundtrip() {
        // Falcon-512: q = 12289, N = 512
        let q: u32 = 12_289;
        let n = 512;
        assert_eq!((q - 1) % (2 * n as u32), 0, "q-1 must be divisible by 2N");

        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, original, "Falcon-512 roundtrip failed");
    }

    #[test]
    fn test_pq_falcon1024_roundtrip() {
        // Falcon-1024: q = 12289, N = 1024
        let q: u32 = 12_289;
        let n = 1024;
        assert_eq!((q - 1) % (2 * n as u32), 0, "q-1 must be divisible by 2N");

        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, original, "Falcon-1024 roundtrip failed");
    }

    #[test]
    fn test_pq_falcon_negacyclic_mul() {
        let q: u32 = 12_289;
        let n = 512;
        let ctx = Ntt32Context::new(n, q);

        let a: Vec<u32> = (0..n)
            .map(|i| ((i as u64 * 17 + 5) % q as u64) as u32)
            .collect();
        let mut one = vec![0u32; n];
        one[0] = 1;

        let result = ctx.negacyclic_mul(&a, &one);
        assert_eq!(result, a, "Falcon: multiply by 1 is not identity");
    }

    #[test]
    fn test_pq_mlkem_proxy_roundtrip() {
        // ML-KEM proxy: q = 3329, N = 128 (Kyber uses incomplete 128-point NTT)
        // 3329 - 1 = 3328 = 2^8 × 13, and 2×128 = 256 | 3328 ✓
        let q: u32 = 3_329;
        let n = 128;
        assert_eq!((q - 1) % (2 * n as u32), 0, "q-1 must be divisible by 2N");

        let ctx = Ntt32Context::new(n, q);
        let original = make_test_data(n, q);
        let mut data = original.clone();

        ctx.forward(&mut data);
        ctx.inverse(&mut data);
        assert_eq!(data, original, "ML-KEM proxy roundtrip failed");
    }

    #[test]
    fn test_pq_mlkem_negacyclic_mul() {
        let q: u32 = 3_329;
        let n = 128;
        let ctx = Ntt32Context::new(n, q);

        // Verify against naive O(N²) multiplication
        let a: Vec<u32> = (0..n).map(|i| (i as u32 + 1) % q).collect();
        let b: Vec<u32> = vec![1u32; n];

        let mut expected = vec![0u32; n];
        for i in 0..n {
            for j in 0..n {
                let prod = (a[i] as u64 * b[j] as u64) % q as u64;
                if i + j < n {
                    expected[i + j] = ((expected[i + j] as u64 + prod) % q as u64) as u32;
                } else {
                    let idx = i + j - n;
                    expected[idx] = ((expected[idx] as u64 + q as u64 - prod) % q as u64) as u32;
                }
            }
        }

        let result = ctx.negacyclic_mul(&a, &b);
        assert_eq!(
            result, expected,
            "ML-KEM negacyclic multiplication mismatch"
        );
    }

    // Compile-time check: Ntt32Context must be Send + Sync
    // (required for safe sharing across threads in crypto applications)
    const _: () = {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        fn check() {
            assert_send::<super::Ntt32Context>();
            assert_sync::<super::Ntt32Context>();
        }
    };
}
