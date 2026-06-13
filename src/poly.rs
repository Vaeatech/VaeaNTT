//! # Polynomial over Z_q\[X\]/(X^N + 1)
//!
//! Polynomials are stored in coefficient domain by default.
//! Use [`Poly64::forward_ntt`] / [`Poly64::inverse_ntt`] to switch between
//! coefficient and NTT (evaluation) domains.
//!
//! In NTT domain, multiplication is pointwise O(N) instead of O(N²).

use crate::ntt64::arith::{mod_mul_barrett, Ntt64Arith};
use crate::ntt64::context::{ntt_forward, ntt_inverse, Ntt64Context};
#[cfg(feature = "rand")]
use rand::Rng;
#[cfg(feature = "rand")]
use rand_distr::{Distribution, Normal};

// ---------------------------------------------------------------------------
// Poly64 — polynomial in Z_q\[X\]/(X^N+1)
// ---------------------------------------------------------------------------

/// Polynomial in R_q = Z_q\[X\]/(X^N + 1) with 64-bit coefficients.
///
/// Tracks whether the data is in coefficient domain or NTT (evaluation) domain.
/// In NTT domain, multiplication is pointwise (O(N) instead of O(N²)).
#[derive(Clone, Debug)]
pub struct Poly64 {
    /// Coefficients (coefficient domain) or evaluations (NTT domain).
    pub data: Vec<u64>,
    /// `true` if the polynomial is in NTT (evaluation) domain.
    pub is_ntt: bool,
}

impl Poly64 {
    // -------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------

    /// Creates the zero polynomial with N coefficients.
    #[inline]
    pub fn new_zero(n: usize) -> Self {
        Self {
            data: vec![0u64; n],
            is_ntt: false,
        }
    }

    /// Creates a polynomial with uniform random coefficients in [0, q).
    ///
    /// Requires the `rand` feature.
    #[cfg(feature = "rand")]
    pub fn new_random(n: usize, arith: &Ntt64Arith) -> Self {
        let mut rng = rand::thread_rng();
        let q = arith.modulus;
        let data: Vec<u64> = (0..n).map(|_| rng.gen_range(0..q)).collect();
        Self {
            data,
            is_ntt: false,
        }
    }

    /// Creates a ternary polynomial with coefficients in {0, 1, q−1}.
    ///
    /// q−1 represents −1 mod q. The distribution is uniform over {−1, 0, 1}.
    /// Used for secret keys in CKKS/BFV.
    ///
    /// Requires the `rand` feature.
    #[cfg(feature = "rand")]
    pub fn new_ternary(n: usize, arith: &Ntt64Arith) -> Self {
        let mut rng = rand::thread_rng();
        let q = arith.modulus;
        let data: Vec<u64> = (0..n)
            .map(|_| match rng.gen_range(0u32..3) {
                0 => 0,
                1 => 1,
                _ => q - 1,
            })
            .collect();
        Self {
            data,
            is_ntt: false,
        }
    }

    /// Creates a polynomial with discrete Gaussian noise.
    ///
    /// Each coefficient is drawn from N(0, σ²), rounded to the nearest integer,
    /// then reduced mod q. Negative values are represented as q + value.
    ///
    /// Requires the `rand` feature.
    #[cfg(feature = "rand")]
    pub fn new_gaussian(n: usize, sigma: f64, arith: &Ntt64Arith) -> Self {
        let mut rng = rand::thread_rng();
        let q = arith.modulus;
        let normal = Normal::new(0.0, sigma).expect("sigma must be > 0");
        let data: Vec<u64> = (0..n)
            .map(|_| {
                let sample: f64 = normal.sample(&mut rng);
                let rounded = sample.round() as i64;
                if rounded >= 0 {
                    (rounded as u64) % q
                } else {
                    let abs_val = (-rounded) as u64;
                    let r = abs_val % q;
                    if r == 0 {
                        0
                    } else {
                        q - r
                    }
                }
            })
            .collect();
        Self {
            data,
            is_ntt: false,
        }
    }

    // -------------------------------------------------------------------
    // NTT transforms
    // -------------------------------------------------------------------

    /// Converts from coefficient domain to NTT domain (in-place).
    ///
    /// # Panics
    /// Panics if the polynomial is already in NTT domain.
    pub fn forward_ntt(&mut self, ntt_ctx: &Ntt64Context) {
        assert!(!self.is_ntt, "polynomial is already in NTT domain");
        ntt_forward(&mut self.data, ntt_ctx);
        self.is_ntt = true;
    }

    /// Converts from NTT domain to coefficient domain (in-place).
    ///
    /// # Panics
    /// Panics if the polynomial is not in NTT domain.
    pub fn inverse_ntt(&mut self, ntt_ctx: &Ntt64Context) {
        assert!(self.is_ntt, "polynomial is not in NTT domain");
        ntt_inverse(&mut self.data, ntt_ctx);
        self.is_ntt = false;
    }

    // -------------------------------------------------------------------
    // Arithmetic
    // -------------------------------------------------------------------

    /// Pointwise addition: `self += other (mod q)`.
    ///
    /// Both polynomials must be in the same domain (NTT or coefficient).
    ///
    /// # Panics
    /// Panics if domains or sizes don't match.
    pub fn add_assign(&mut self, other: &Poly64, arith: &Ntt64Arith) {
        assert_eq!(
            self.is_ntt, other.is_ntt,
            "polynomials must be in the same domain"
        );
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "polynomials must have the same size"
        );
        let q = arith.modulus;
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            let sum = *a + b;
            // Branchless via overflowing_sub
            let (sub, borrow) = sum.overflowing_sub(q);
            *a = if borrow { sum } else { sub };
        }
    }

    /// Pointwise subtraction: `self -= other (mod q)`.
    ///
    /// Both polynomials must be in the same domain.
    ///
    /// # Panics
    /// Panics if domains or sizes don't match.
    pub fn sub_assign(&mut self, other: &Poly64, arith: &Ntt64Arith) {
        assert_eq!(
            self.is_ntt, other.is_ntt,
            "polynomials must be in the same domain"
        );
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "polynomials must have the same size"
        );
        let q = arith.modulus;
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            let (sub, borrow) = (*a).overflowing_sub(b);
            *a = if borrow { sub.wrapping_add(q) } else { sub };
        }
    }

    /// Pointwise multiplication: `self *= other (mod q)`.
    ///
    /// **Both polynomials must be in NTT domain** so that pointwise multiplication
    /// corresponds to negacyclic convolution in coefficient domain.
    ///
    /// # Panics
    /// Panics if polynomials are not in NTT domain or have different sizes.
    pub fn mul_assign(&mut self, other: &Poly64, arith: &Ntt64Arith) {
        assert!(
            self.is_ntt && other.is_ntt,
            "both polynomials must be in NTT domain for multiplication"
        );
        assert_eq!(
            self.data.len(),
            other.data.len(),
            "polynomials must have the same size"
        );
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a = mod_mul_barrett(*a, b, arith);
        }
    }

    /// Scalar multiplication: `self *= scalar (mod q)`.
    pub fn scalar_mul(&mut self, scalar: u64, arith: &Ntt64Arith) {
        for a in self.data.iter_mut() {
            *a = mod_mul_barrett(*a, scalar, arith);
        }
    }

    /// Negation: `self = −self (mod q)`, i.e. `self[i] = q − self[i]`.
    pub fn negate(&mut self, arith: &Ntt64Arith) {
        let q = arith.modulus;
        for a in self.data.iter_mut() {
            // Branchless: mask = (a != 0) as u64 * u64::MAX, then q & mask - *a ...
            // but the branch here is on public data (coefficients), not secrets,
            // and the branch predictor handles it well. Keep it simple.
            *a = if *a == 0 { 0 } else { q - *a };
        }
    }

    // -------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------

    /// Number of coefficients (= max degree + 1).
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the polynomial has zero length.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Naive polynomial multiplication (test-only)
// ---------------------------------------------------------------------------

/// Naive polynomial multiplication in Z_q\[X\]/(X^N+1).
///
/// O(N²) complexity. Used only in tests to verify NTT-based multiplication.
#[cfg(test)]
fn naive_poly_mul(a: &[u64], b: &[u64], q: u64) -> Vec<u64> {
    let n = a.len();
    assert_eq!(n, b.len());
    let mut result = vec![0u64; n];

    for i in 0..n {
        for j in 0..n {
            let prod = (a[i] as u128) * (b[j] as u128);
            let idx = i + j;
            if idx < n {
                let val = (result[idx] as u128 + prod) % (q as u128);
                result[idx] = val as u64;
            } else {
                let wrapped_idx = idx - n;
                let val = (result[wrapped_idx] as u128 + (q as u128) - (prod % (q as u128)))
                    % (q as u128);
                result[wrapped_idx] = val as u64;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt64::arith::Ntt64Arith;
    use crate::ntt64::context::Ntt64Context;

    // Small NTT-friendly prime for N=256: q = 7681 = 15·512+1
    const TEST_Q: u64 = 7681;
    const TEST_N: usize = 256;

    fn test_arith() -> Ntt64Arith {
        Ntt64Arith::new(TEST_Q)
    }

    fn test_ntt_ctx() -> Ntt64Context {
        Ntt64Context::new(TEST_N, test_arith())
    }

    #[test]
    fn test_poly_add_sub() {
        let arith = test_arith();
        let a = Poly64::new_random(TEST_N, &arith);
        let b = Poly64::new_random(TEST_N, &arith);

        let mut c = a.clone();
        c.add_assign(&b, &arith);
        c.sub_assign(&b, &arith);

        for i in 0..TEST_N {
            assert_eq!(c.data[i], a.data[i], "add/sub roundtrip fails at index {i}");
        }
    }

    #[test]
    fn test_poly_add_commutative() {
        let arith = test_arith();
        let a = Poly64::new_random(TEST_N, &arith);
        let b = Poly64::new_random(TEST_N, &arith);

        let mut ab = a.clone();
        ab.add_assign(&b, &arith);

        let mut ba = b.clone();
        ba.add_assign(&a, &arith);

        for i in 0..TEST_N {
            assert_eq!(ab.data[i], ba.data[i], "add not commutative at index {i}");
        }
    }

    #[test]
    fn test_poly_negate() {
        let arith = test_arith();
        let a = Poly64::new_random(TEST_N, &arith);

        let mut neg_a = a.clone();
        neg_a.negate(&arith);

        let mut sum = a.clone();
        sum.add_assign(&neg_a, &arith);

        for i in 0..TEST_N {
            assert_eq!(sum.data[i], 0, "a + (-a) != 0 at index {i}");
        }
    }

    #[test]
    fn test_poly_scalar_mul() {
        let arith = test_arith();
        let a = Poly64::new_random(TEST_N, &arith);

        let mut doubled = a.clone();
        doubled.scalar_mul(2, &arith);

        let mut sum = a.clone();
        sum.add_assign(&a, &arith);

        for i in 0..TEST_N {
            assert_eq!(doubled.data[i], sum.data[i], "2*a != a+a at index {i}");
        }
    }

    #[test]
    fn test_poly_mul_ntt() {
        let arith = test_arith();
        let ntt_ctx = test_ntt_ctx();

        let mut a = Poly64::new_zero(TEST_N);
        a.data[0] = 1;
        a.data[1] = 1;

        let mut b = Poly64::new_zero(TEST_N);
        b.data[0] = 1;
        b.data[2] = 1;

        let expected = naive_poly_mul(&a.data, &b.data, TEST_Q);

        a.forward_ntt(&ntt_ctx);
        b.forward_ntt(&ntt_ctx);
        a.mul_assign(&b, &arith);
        a.inverse_ntt(&ntt_ctx);

        for i in 0..TEST_N {
            assert_eq!(a.data[i], expected[i], "NTT mul != naive at index {i}");
        }
    }

    #[test]
    fn test_poly_mul_random_ntt() {
        let arith = test_arith();
        let ntt_ctx = test_ntt_ctx();

        let a_orig = Poly64::new_random(TEST_N, &arith);
        let b_orig = Poly64::new_random(TEST_N, &arith);

        let expected = naive_poly_mul(&a_orig.data, &b_orig.data, TEST_Q);

        let mut a = a_orig.clone();
        let mut b = b_orig.clone();
        a.forward_ntt(&ntt_ctx);
        b.forward_ntt(&ntt_ctx);
        a.mul_assign(&b, &arith);
        a.inverse_ntt(&ntt_ctx);

        for i in 0..TEST_N {
            assert_eq!(a.data[i], expected[i], "NTT mul != naive at index {i}");
        }
    }

    #[test]
    fn test_ternary_distribution() {
        let arith = test_arith();
        let poly = Poly64::new_ternary(1024, &arith);

        for (i, &coeff) in poly.data.iter().enumerate() {
            assert!(
                coeff == 0 || coeff == 1 || coeff == TEST_Q - 1,
                "invalid ternary coefficient at index {i}: {coeff}"
            );
        }

        let count_zero = poly.data.iter().filter(|&&c| c == 0).count();
        let count_one = poly.data.iter().filter(|&&c| c == 1).count();
        let count_neg = poly.data.iter().filter(|&&c| c == TEST_Q - 1).count();

        assert!(count_zero > 0);
        assert!(count_one > 0);
        assert!(count_neg > 0);
    }

    #[test]
    fn test_gaussian_distribution() {
        let arith = test_arith();
        let sigma = 3.2;
        let n = 8192;
        let poly = Poly64::new_gaussian(n, sigma, &arith);

        let q = TEST_Q as f64;
        let half_q = q / 2.0;
        let centered: Vec<f64> = poly
            .data
            .iter()
            .map(|&c| {
                let c = c as f64;
                if c > half_q {
                    c - q
                } else {
                    c
                }
            })
            .collect();

        let mean = centered.iter().sum::<f64>() / n as f64;
        assert!(mean.abs() < 0.5, "mean too far from 0: {mean}");

        let variance = centered.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();
        assert!(
            (std_dev - sigma).abs() < 1.0,
            "stddev too far from {sigma}: {std_dev}"
        );
    }

    #[test]
    fn test_ntt_roundtrip() {
        let arith = test_arith();
        let ntt_ctx = test_ntt_ctx();
        let original = Poly64::new_random(TEST_N, &arith);

        let mut poly = original.clone();
        poly.forward_ntt(&ntt_ctx);
        assert!(poly.is_ntt);
        poly.inverse_ntt(&ntt_ctx);
        assert!(!poly.is_ntt);

        for i in 0..TEST_N {
            assert_eq!(
                poly.data[i], original.data[i],
                "NTT roundtrip fails at index {i}"
            );
        }
    }

    #[test]
    fn test_new_zero() {
        let poly = Poly64::new_zero(64);
        assert_eq!(poly.len(), 64);
        assert!(!poly.is_ntt);
        for &c in &poly.data {
            assert_eq!(c, 0);
        }
    }

    #[test]
    #[should_panic(expected = "already in NTT domain")]
    fn test_double_forward_ntt_panics() {
        let arith = test_arith();
        let ntt_ctx = test_ntt_ctx();
        let mut poly = Poly64::new_random(TEST_N, &arith);
        poly.forward_ntt(&ntt_ctx);
        poly.forward_ntt(&ntt_ctx);
    }

    #[test]
    #[should_panic(expected = "not in NTT domain")]
    fn test_inverse_ntt_without_forward_panics() {
        let arith = test_arith();
        let ntt_ctx = test_ntt_ctx();
        let mut poly = Poly64::new_random(TEST_N, &arith);
        poly.inverse_ntt(&ntt_ctx);
    }
}
