//! # Prime Generation and Primality Testing
//!
//! Deterministic Miller-Rabin primality test (correct for all n < 2^64)
//! and generation of NTT-friendly primes of the form k·2N + 1.

use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// Deterministic Miller-Rabin primality test
// ---------------------------------------------------------------------------

/// Deterministic Miller-Rabin primality test for any n < 2^64.
///
/// Uses the 12 witness bases {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
/// which guarantee correctness for all n < 3.317×10²⁴ > 2^64.
///
/// # Reference
/// Sorenson & Webster, "Strong Pseudoprimes to Twelve Prime Bases", 2015.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }

    const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if SMALL_PRIMES.contains(&n) {
        return true;
    }
    if n.is_multiple_of(5) || n.is_multiple_of(7) || n.is_multiple_of(11) || n.is_multiple_of(13) {
        return n <= 13;
    }

    // Decompose n-1 = 2^s * d with d odd
    let mut d = n - 1;
    let mut s: u32 = 0;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    // Miller-Rabin test with each witness base
    'witness: for &a in &SMALL_PRIMES {
        if a >= n {
            continue;
        }

        // x = a^d mod n
        let mut x = mod_pow_raw(a, d, n);

        if x == 1 || x == n - 1 {
            continue 'witness;
        }

        for _ in 0..s - 1 {
            x = mod_mul_raw(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }

        return false;
    }

    true
}

// ---------------------------------------------------------------------------
// Raw modular arithmetic (standalone, no Ntt64Arith dependency)
// ---------------------------------------------------------------------------

/// Raw modular exponentiation: `base^exp mod modulus`.
///
/// Uses square-and-multiply with u128 intermediates.
/// This is a standalone version that does not require an `Ntt64Arith` context.
#[inline]
fn mod_pow_raw(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result: u64 = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mod_mul_raw(result, base, modulus);
        }
        exp >>= 1;
        base = mod_mul_raw(base, base, modulus);
    }
    result
}

/// Raw modular multiplication via u128 to avoid overflow.
#[inline(always)]
fn mod_mul_raw(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

// ---------------------------------------------------------------------------
// NTT-friendly prime generation
// ---------------------------------------------------------------------------

/// Generates NTT-friendly primes of the form `k·2N + 1`.
///
/// An NTT-friendly prime q for polynomial degree N satisfies `2N | (q−1)`,
/// guaranteeing the existence of 2N-th roots of unity in Z_q*.
///
/// # Arguments
/// * `poly_degree` — N, the cyclotomic polynomial degree (must be a power of 2)
/// * `bit_size` — desired bit-length for each prime (in [2, 62])
/// * `count` — number of primes to generate
///
/// # Returns
/// A vector of `count` distinct NTT-friendly primes.
///
/// # Panics
/// - If `poly_degree` is not a power of 2
/// - If not enough primes can be found
pub fn generate_primes_60(poly_degree: usize, bit_size: usize, count: usize) -> Vec<u64> {
    assert!(
        poly_degree.is_power_of_two(),
        "poly_degree must be a power of 2"
    );
    assert!((2..=62).contains(&bit_size), "bit_size must be in [2, 62]");

    let two_n = (2 * poly_degree) as u64;
    let lower = (1u64 << (bit_size - 1)) / two_n + 1;
    let upper = (1u64 << bit_size) / two_n;

    let mut primes = Vec::with_capacity(count);

    for k in lower..=upper {
        if primes.len() == count {
            break;
        }
        let q = k * two_n + 1;
        if q.leading_zeros() != (64 - bit_size as u32) {
            continue;
        }
        if is_prime(q) {
            primes.push(q);
        }
    }

    assert_eq!(
        primes.len(),
        count,
        "could not find {count} NTT-friendly primes of {bit_size} bits for N={poly_degree}"
    );

    primes
}

// ---------------------------------------------------------------------------
// Primitive root finding
// ---------------------------------------------------------------------------

/// Finds a primitive 2N-th root of unity modulo q.
///
/// Returns ψ such that:
/// - ψ^(2N) ≡ 1 (mod q)
/// - ψ^N ≡ −1 (mod q)
///
/// # Algorithm
/// 1. Factor q−1 (trial division for small factors, residue assumed prime)
/// 2. Find a generator g of Z_q* (by testing g^((q−1)/p) ≠ 1 for each prime factor p)
/// 3. Compute ψ = g^((q−1)/(2N))
///
/// # Panics
/// If q ≡ 0 (mod 2N), i.e., 2N does not divide q−1.
pub fn find_primitive_root(n: usize, modulus: u64) -> u64 {
    let two_n = 2 * n as u64;
    assert!(
        (modulus - 1).is_multiple_of(two_n),
        "modulus q={modulus} does not satisfy q ≡ 1 (mod 2N={two_n})"
    );

    let q_minus_1 = modulus - 1;
    let prime_factors = small_factor(q_minus_1);

    // Find a generator g of Z_q*
    let g = find_generator(modulus, &prime_factors);

    // ψ = g^((q−1)/(2N))
    let exp = q_minus_1 / two_n;
    let psi = mod_pow_raw(g, exp, modulus);

    // Safety checks
    debug_assert_eq!(
        mod_pow_raw(psi, two_n, modulus),
        1,
        "ψ^(2N) ≠ 1: not a 2N-th root"
    );
    debug_assert_eq!(
        mod_pow_raw(psi, n as u64, modulus),
        modulus - 1,
        "ψ^N ≠ −1: not a PRIMITIVE 2N-th root"
    );

    psi
}

/// Trial-division factorization: returns the distinct prime factors of `n`.
fn small_factor(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();

    if n.is_multiple_of(2) {
        factors.push(2);
        while n.is_multiple_of(2) {
            n /= 2;
        }
    }

    let mut d = 3u64;
    while d * d <= n {
        if n.is_multiple_of(d) {
            factors.push(d);
            while n.is_multiple_of(d) {
                n /= d;
            }
        }
        d += 2;
    }

    if n > 1 {
        factors.push(n);
    }

    factors
}

/// Finds a generator of Z_q* by trial.
///
/// An element g is a generator iff g^((q−1)/p) ≠ 1 for every prime factor p of q−1.
fn find_generator(q: u64, prime_factors: &[u64]) -> u64 {
    let q_minus_1 = q - 1;

    for g in 2..q {
        let mut is_generator = true;
        for &p in prime_factors {
            let exp = q_minus_1 / p;
            if mod_pow_raw(g, exp, q) == 1 {
                is_generator = false;
                break;
            }
        }
        if is_generator {
            return g;
        }
    }

    panic!("no generator found for q={q} — this should never happen");
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_miller_rabin_primes() {
        let known_primes: Vec<u64> = vec![
            2,
            3,
            5,
            7,
            11,
            13,
            17,
            19,
            23,
            29,
            31,
            37,
            41,
            43,
            97,
            101,
            7681,
            12289,
            65537,
            786433,
            104857601,
            (1u64 << 61) - 1, // Mersenne prime 2^61 - 1
        ];
        for &p in &known_primes {
            assert!(is_prime(p), "{p} should be prime");
        }
    }

    #[test]
    fn test_miller_rabin_composites() {
        let composites: Vec<u64> = vec![
            0, 1, 4, 6, 8, 9, 10, 12, 15, 21, 25, 49, 100, 1000,
            561,  // Carmichael pseudoprime (3·11·17)
            1105, // Carmichael pseudoprime (5·13·17)
            1729, // Hardy-Ramanujan number (7·13·19)
            7680, 12288,
        ];
        for &c in &composites {
            assert!(!is_prime(c), "{c} should not be prime");
        }
    }

    #[test]
    fn test_prime_generation() {
        let primes = generate_primes_60(256, 14, 3);
        assert_eq!(primes.len(), 3);

        for &q in &primes {
            assert!(is_prime(q), "{q} is not prime");
            let two_n: u64 = 2 * 256;
            assert_eq!((q - 1) % two_n, 0);
            let bits = 64 - q.leading_zeros();
            assert_eq!(bits, 14, "{q} has {bits} bits, expected 14");
        }

        // Uniqueness
        let mut sorted = primes.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), primes.len(), "primes must be distinct");
    }

    #[test]
    fn test_find_primitive_root() {
        let q = 7681u64; // q−1 = 7680 = 15·512, supports N=256
        let n = 256;
        let psi = find_primitive_root(n, q);
        assert_eq!(mod_pow_raw(psi, 2 * n as u64, q), 1);
        assert_eq!(mod_pow_raw(psi, n as u64, q), q - 1);
    }

    #[test]
    fn test_find_primitive_root_seal() {
        let q = super::super::PRIME_SEAL;
        for &n in &[16, 64, 1024, 4096] {
            let psi = find_primitive_root(n, q);
            assert_eq!(mod_pow_raw(psi, 2 * n as u64, q), 1);
            assert_eq!(mod_pow_raw(psi, n as u64, q), q - 1);
        }
    }
}
