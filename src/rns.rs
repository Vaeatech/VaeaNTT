//! # Residue Number System (RNS) — Multi-Moduli Decomposition
//!
//! RNS allows working with large integers by decomposing them into residues
//! modulo several small coprime moduli. Each component can be processed
//! independently, which is perfect for parallelism and avoids
//! multi-precision arithmetic.
//!
//! For CKKS, the product Q = q₁·q₂·…·q_L defines the precision level.
//! Rescaling removes one modulus per level.

use crate::ntt64::arith::Ntt64Arith;
use crate::ntt64::context::Ntt64Context;
use crate::poly::Poly64;

// ---------------------------------------------------------------------------
// RnsContext — RNS context
// ---------------------------------------------------------------------------

/// RNS context: a set of coprime moduli.
///
/// Precomputes modular arithmetic and NTT contexts for each modulus,
/// enabling efficient component-wise polynomial operations.
pub struct RnsContext {
    /// The moduli q₁, q₂, …, q_L.
    pub moduli: Vec<u64>,
    /// Modular arithmetic contexts for each modulus (Barrett, Montgomery).
    pub ariths: Vec<Ntt64Arith>,
    /// NTT contexts for each modulus.
    pub ntt_ctxs: Vec<Ntt64Context>,
    /// Polynomial degree N.
    pub poly_degree: usize,
}

impl RnsContext {
    /// Creates an RNS context with the given moduli.
    ///
    /// Precomputes all modular arithmetic and NTT contexts.
    /// Each modulus must be NTT-friendly for the given polynomial degree.
    ///
    /// # Panics
    /// - If `poly_degree` is not a power of 2
    /// - If `moduli` is empty
    /// - If any modulus is not NTT-friendly for the given degree
    pub fn new(poly_degree: usize, moduli: Vec<u64>) -> Self {
        assert!(
            poly_degree.is_power_of_two(),
            "poly_degree must be a power of 2"
        );
        assert!(!moduli.is_empty(), "at least one modulus is required");

        let ariths: Vec<Ntt64Arith> = moduli.iter().map(|&q| Ntt64Arith::new(q)).collect();

        let ntt_ctxs: Vec<Ntt64Context> = ariths
            .iter()
            .map(|arith| Ntt64Context::new(poly_degree, arith.clone()))
            .collect();

        Self {
            moduli,
            ariths,
            ntt_ctxs,
            poly_degree,
        }
    }

    /// Number of moduli (= total number of levels).
    #[inline]
    pub fn num_moduli(&self) -> usize {
        self.moduli.len()
    }
}

// ---------------------------------------------------------------------------
// RnsPoly — polynomial in RNS representation
// ---------------------------------------------------------------------------

/// Polynomial in RNS representation: one component per modulus.
///
/// Each component `components[i]` is a polynomial in Z_{q_i}\[X\]/(X^N+1),
/// stored in NTT domain by default for performance.
///
/// The `level` indicates the number of active moduli. CKKS rescaling reduces
/// the level by removing the last modulus.
#[derive(Clone, Debug)]
pub struct RnsPoly {
    /// `components[i]` = polynomial modulo `moduli[i]`.
    pub components: Vec<Poly64>,
    /// Current level (number of active moduli).
    pub level: usize,
}

impl RnsPoly {
    /// Encodes a signed-integer polynomial into RNS representation.
    ///
    /// For each modulus q_i:
    /// 1. Reduces each coefficient mod q_i (handles negatives)
    /// 2. Converts to NTT domain
    ///
    /// # Arguments
    /// * `coeffs` — polynomial coefficients in Z (signed, coefficient domain)
    /// * `ctx` — RNS context
    pub fn from_coefficients(coeffs: &[i64], ctx: &RnsContext) -> Self {
        let n = ctx.poly_degree;
        assert!(
            coeffs.len() <= n,
            "too many coefficients: {} > {}",
            coeffs.len(),
            n
        );

        let level = ctx.num_moduli();
        let mut components = Vec::with_capacity(level);

        for i in 0..level {
            let q = ctx.moduli[i];
            let q_i64 = q as i64;

            let mut poly = Poly64::new_zero(n);
            for (j, &c) in coeffs.iter().enumerate() {
                let r = c % q_i64;
                poly.data[j] = if r < 0 { (r + q_i64) as u64 } else { r as u64 };
            }

            poly.forward_ntt(&ctx.ntt_ctxs[i]);
            components.push(poly);
        }

        Self { components, level }
    }

    /// Component-wise addition in RNS.
    ///
    /// Both polynomials must have the same level.
    pub fn add(&self, other: &RnsPoly, ctx: &RnsContext) -> RnsPoly {
        assert_eq!(
            self.level, other.level,
            "levels must match: {} != {}",
            self.level, other.level
        );

        let mut result = self.clone();
        for i in 0..self.level {
            result.components[i].add_assign(&other.components[i], &ctx.ariths[i]);
        }
        result
    }

    /// Component-wise subtraction in RNS.
    pub fn sub(&self, other: &RnsPoly, ctx: &RnsContext) -> RnsPoly {
        assert_eq!(self.level, other.level, "levels must match");

        let mut result = self.clone();
        for i in 0..self.level {
            result.components[i].sub_assign(&other.components[i], &ctx.ariths[i]);
        }
        result
    }

    /// Component-wise multiplication in RNS (NTT domain).
    ///
    /// All components must be in NTT domain.
    pub fn mul(&self, other: &RnsPoly, ctx: &RnsContext) -> RnsPoly {
        assert_eq!(self.level, other.level, "levels must match");

        let mut result = self.clone();
        for i in 0..self.level {
            result.components[i].mul_assign(&other.components[i], &ctx.ariths[i]);
        }
        result
    }

    /// Drops the last modulus (CKKS rescaling).
    ///
    /// After this operation, the level decreases by 1 and the last component
    /// is removed. The scale factor Δ is implicitly divided by q_L.
    ///
    /// # Panics
    /// Panics if the level is already 1.
    pub fn drop_last_modulus(&mut self) {
        assert!(self.level > 1, "cannot reduce level below 1");
        self.components.pop();
        self.level -= 1;
    }

    /// Converts all components from NTT domain to coefficient domain.
    pub fn forward_all(&mut self, ctx: &RnsContext) {
        for i in 0..self.level {
            if !self.components[i].is_ntt {
                self.components[i].forward_ntt(&ctx.ntt_ctxs[i]);
            }
        }
    }

    /// Converts all components from NTT domain to coefficient domain.
    pub fn inverse_all(&mut self, ctx: &RnsContext) {
        for i in 0..self.level {
            if self.components[i].is_ntt {
                self.components[i].inverse_ntt(&ctx.ntt_ctxs[i]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ntt64::prime::is_prime;

    const TEST_N: usize = 256;
    const TEST_Q1: u64 = 7681; // 15·512+1
    const TEST_Q2: u64 = 12289; // 24·512+1

    fn test_rns_ctx() -> RnsContext {
        RnsContext::new(TEST_N, vec![TEST_Q1, TEST_Q2])
    }

    #[test]
    fn test_rns_encode_decode() {
        let ctx = test_rns_ctx();
        let coeffs = vec![5i64, -3, 0, 7];
        let mut rns_poly = RnsPoly::from_coefficients(&coeffs, &ctx);

        rns_poly.inverse_all(&ctx);

        assert_eq!(rns_poly.components[0].data[0], 5);
        assert_eq!(rns_poly.components[0].data[1], TEST_Q1 - 3);
        assert_eq!(rns_poly.components[0].data[2], 0);
        assert_eq!(rns_poly.components[0].data[3], 7);

        assert_eq!(rns_poly.components[1].data[0], 5);
        assert_eq!(rns_poly.components[1].data[1], TEST_Q2 - 3);
        assert_eq!(rns_poly.components[1].data[2], 0);
        assert_eq!(rns_poly.components[1].data[3], 7);
    }

    #[test]
    fn test_rns_add_mul_distributivity() {
        let ctx = test_rns_ctx();

        let a_coeffs: Vec<i64> = (0..TEST_N as i64).map(|i| i % 100).collect();
        let b_coeffs: Vec<i64> = (0..TEST_N as i64).map(|i| (i * 3 + 7) % 100).collect();
        let c_coeffs: Vec<i64> = (0..TEST_N as i64).map(|i| (i * 2 + 1) % 50).collect();

        let a = RnsPoly::from_coefficients(&a_coeffs, &ctx);
        let b = RnsPoly::from_coefficients(&b_coeffs, &ctx);
        let c = RnsPoly::from_coefficients(&c_coeffs, &ctx);

        // (a + b) * c
        let ab = a.add(&b, &ctx);
        let mut lhs = ab.mul(&c, &ctx);

        // a*c + b*c
        let ac = a.mul(&c, &ctx);
        let bc = b.mul(&c, &ctx);
        let mut rhs = ac.add(&bc, &ctx);

        lhs.inverse_all(&ctx);
        rhs.inverse_all(&ctx);

        for i in 0..ctx.num_moduli() {
            for j in 0..TEST_N {
                assert_eq!(
                    lhs.components[i].data[j], rhs.components[i].data[j],
                    "(a+b)*c != a*c+b*c — modulus {}, coeff {}",
                    ctx.moduli[i], j
                );
            }
        }
    }

    #[test]
    fn test_rns_drop_last_modulus() {
        let ctx = test_rns_ctx();
        let coeffs = vec![1i64, 2, 3];
        let mut poly = RnsPoly::from_coefficients(&coeffs, &ctx);

        assert_eq!(poly.level, 2);
        assert_eq!(poly.components.len(), 2);

        poly.drop_last_modulus();

        assert_eq!(poly.level, 1);
        assert_eq!(poly.components.len(), 1);
    }

    #[test]
    #[should_panic(expected = "cannot reduce")]
    fn test_rns_drop_last_modulus_panics_at_level_1() {
        let ctx = RnsContext::new(TEST_N, vec![TEST_Q1]);
        let coeffs = vec![1i64];
        let mut poly = RnsPoly::from_coefficients(&coeffs, &ctx);
        poly.drop_last_modulus();
    }

    #[test]
    fn test_rns_sub() {
        let ctx = test_rns_ctx();
        let coeffs: Vec<i64> = (0..TEST_N as i64).map(|i| i % 1000 - 500).collect();
        let a = RnsPoly::from_coefficients(&coeffs, &ctx);

        let mut zero = a.sub(&a, &ctx);
        zero.inverse_all(&ctx);

        for i in 0..ctx.num_moduli() {
            for j in 0..TEST_N {
                assert_eq!(
                    zero.components[i].data[j], 0,
                    "a - a != 0 — modulus {}, coeff {}",
                    ctx.moduli[i], j
                );
            }
        }
    }

    #[test]
    fn test_ntt_friendly_primes_are_valid() {
        assert!(is_prime(TEST_Q1), "q1 = {TEST_Q1} should be prime");
        assert!(is_prime(TEST_Q2), "q2 = {TEST_Q2} should be prime");

        let two_n = 2 * TEST_N as u64;
        assert_eq!((TEST_Q1 - 1) % two_n, 0);
        assert_eq!((TEST_Q2 - 1) % two_n, 0);
    }
}
