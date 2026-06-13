//! # 64-bit Modular Arithmetic
//!
//! High-performance modular arithmetic for NTT-friendly primes < 2^62.
//!
//! Provides:
//! - **Barrett reduction** — modular multiplication without division
//! - **Montgomery reduction** — fast chains of multiplications in Montgomery domain
//! - **Branchless add/sub** — constant-time modular addition and subtraction
//! - **Fast exponentiation** — square-and-multiply
//! - **Modular inverse** — via Fermat's little theorem

// ---------------------------------------------------------------------------
// Pre-defined NTT-friendly primes (< 2^62, of the form k·2^s + 1 with s ≥ 16)
// ---------------------------------------------------------------------------

/// 60-bit NTT-friendly prime: 1152921504606584833 = 17592186044412 · 2^16 + 1.
///
/// Satisfies q ≡ 1 (mod 2^16), supports N ≤ 32768.
pub const PRIME_60_1: u64 = 1_152_921_504_606_584_833;

/// Classic SEAL 61-bit prime: 0x1fffffffffe00001 = 2305843009211596801.
///
/// q = 2^61 − 2^21 + 1, satisfies q ≡ 1 (mod 2^21), supports N ≤ 1048576.
pub const PRIME_SEAL: u64 = 0x1FFF_FFFF_FFE0_0001;

/// 62-bit NTT-friendly prime: 4611686018326724609 = 70368744177283 · 2^16 + 1.
///
/// Satisfies q ≡ 1 (mod 2^16), supports N ≤ 32768.
pub const PRIME_62_1: u64 = 4_611_686_018_326_724_609;

/// 60-bit NTT-friendly prime: 576460752308273153 = 8796093022282 · 2^16 + 1.
///
/// Satisfies q ≡ 1 (mod 2^16), supports N ≤ 32768.
pub const PRIME_60_2: u64 = 576_460_752_308_273_153;

/// 60-bit NTT-friendly prime: 576460752312401921 = 8796093022345 · 2^16 + 1.
///
/// Satisfies q ≡ 1 (mod 2^16), supports N ≤ 32768.
pub const PRIME_60_3: u64 = 576_460_752_312_401_921;

// ---------------------------------------------------------------------------
// Ntt64Arith — precomputed modular arithmetic context
// ---------------------------------------------------------------------------

/// Precomputed modular arithmetic context for a given modulus.
///
/// Contains all constants needed for Barrett and Montgomery reductions.
/// The modulus must be odd and < 2^62.
#[derive(Debug, Clone)]
pub struct Ntt64Arith {
    /// The modulus q (must be < 2^62 and odd).
    pub modulus: u64,

    /// Barrett constant: μ = floor(2^128 / q).
    ///
    /// Used to approximate division by q without a `div` instruction.
    pub barrett_mu: u128,

    /// Montgomery constant R = 2^64 mod q.
    pub mont_r: u64,

    /// Montgomery constant R² = (2^64)² mod q = 2^128 mod q.
    ///
    /// Used to convert into Montgomery domain: `to_mont(a) = REDC(a, R²) = a·R mod q`.
    pub mont_r2: u64,

    /// Montgomery constant −q⁻¹ mod 2^64.
    ///
    /// Cancels the low bits of the intermediate product during REDC.
    pub mont_neg_inv: u64,
}

impl Ntt64Arith {
    /// Creates a new modular arithmetic context for the given modulus.
    ///
    /// # Panics
    /// - If `modulus` is even (required for Montgomery)
    /// - If `modulus` < 2 or ≥ 2^62
    pub fn new(modulus: u64) -> Self {
        assert!(modulus >= 2, "modulus must be >= 2");
        assert!(modulus < (1u64 << 62), "modulus must be < 2^62");
        assert!(
            modulus & 1 == 1,
            "modulus must be odd (required for Montgomery)"
        );

        // --- Barrett constant: μ = floor(2^128 / q) ---
        // 2^128 doesn't fit in u128, so we decompose:
        //   floor(2^128 / q) = floor((u128::MAX + 1) / q)
        //                    = floor(u128::MAX / q) + floor((u128::MAX % q + 1) / q)
        let barrett_mu = {
            let q = modulus as u128;
            let div = u128::MAX / q;
            let rem = u128::MAX % q;
            if rem + 1 == q {
                div + 1
            } else {
                div
            }
        };

        // --- Montgomery constants ---

        // R = 2^64 mod q = (u64::MAX % q + 1) % q  since 2^64 = u64::MAX + 1
        let mont_r = {
            let r = (u64::MAX % modulus).wrapping_add(1);
            if r == modulus {
                0
            } else {
                r
            }
        };

        // R² = 2^128 mod q = (R · R) mod q  via u128
        let mont_r2 = ((mont_r as u128 * mont_r as u128) % modulus as u128) as u64;

        // −q⁻¹ mod 2^64 via Newton-Hensel lifting.
        // We find x such that q·x ≡ −1 (mod 2^64).
        // Start with x₀ = 1 (works since q is odd: q·1 ≡ 1 mod 2).
        // Each iteration doubles precision: after 6 iterations → exact mod 2^64.
        let mont_neg_inv = {
            let mut inv: u64 = 1;
            for _ in 0..6 {
                inv = inv.wrapping_mul(2u64.wrapping_sub(modulus.wrapping_mul(inv)));
            }
            inv.wrapping_neg()
        };

        Self {
            modulus,
            barrett_mu,
            mont_r,
            mont_r2,
            mont_neg_inv,
        }
    }
}

// ---------------------------------------------------------------------------
// Branchless modular add / sub
// ---------------------------------------------------------------------------

/// Branchless modular addition.
///
/// Returns `(a + b) mod m` with `a, b < m`.
///
/// Uses `overflowing_sub` to generate a borrow flag which the compiler
/// lowers to a conditional move (`cmov` / `csel`), avoiding branches.
#[inline(always)]
pub fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    debug_assert!(a < m, "mod_add: a={a} >= m={m}");
    debug_assert!(b < m, "mod_add: b={b} >= m={m}");

    let s = a.wrapping_add(b);
    let (sub, borrow) = s.overflowing_sub(m);
    // s >= m → borrow = false → return sub = s - m
    // s <  m → borrow = true  → return s
    if borrow {
        s
    } else {
        sub
    }
}

/// Branchless modular subtraction.
///
/// Returns `(a - b) mod m` with `a, b < m`.
///
/// Uses `overflowing_sub` to detect underflow and conditionally add `m` back,
/// lowered to a branchless `cmov` / `csel` by the compiler.
#[inline(always)]
pub fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    debug_assert!(a < m, "mod_sub: a={a} >= m={m}");
    debug_assert!(b < m, "mod_sub: b={b} >= m={m}");

    let (sub, borrow) = a.overflowing_sub(b);
    // a >= b → borrow = false → return sub = a - b
    // a <  b → borrow = true  → return sub + m = a - b + m
    if borrow {
        sub.wrapping_add(m)
    } else {
        sub
    }
}

// ---------------------------------------------------------------------------
// Barrett multiplication
// ---------------------------------------------------------------------------

/// Modular multiplication via Barrett reduction.
///
/// Returns `(a * b) mod q` with `a, b < q` and `q < 2^62`.
///
/// # Algorithm
/// 1. `p = a * b` (u128, at most ~2^124)
/// 2. `qhat = (p * μ) >> 128` (quotient approximation)
/// 3. `r = p - qhat * q` (approximate remainder)
/// 4. Single conditional subtraction to correct
#[inline(always)]
pub fn mod_mul_barrett(a: u64, b: u64, ctx: &Ntt64Arith) -> u64 {
    debug_assert!(
        a < ctx.modulus,
        "mod_mul_barrett: a={a} >= q={}",
        ctx.modulus
    );
    debug_assert!(
        b < ctx.modulus,
        "mod_mul_barrett: b={b} >= q={}",
        ctx.modulus
    );

    let p = a as u128 * b as u128;

    // Quotient approximation: qhat = floor(p * μ / 2^128)
    // We need the top 128 bits of the 256-bit product p * μ.
    // Decompose into four 64×64 partial products.
    let p_lo = p as u64 as u128;
    let p_hi = (p >> 64) as u64 as u128;
    let mu_lo = ctx.barrett_mu as u64 as u128;
    let mu_hi = (ctx.barrett_mu >> 64) as u64 as u128;

    let mid1 = p_lo * mu_hi;
    let mid2 = p_hi * mu_lo;
    let high = p_hi * mu_hi;

    // Accumulate carries toward the upper 128 bits.
    let t_lo = p_lo * mu_lo;
    let carry_from_lo = t_lo >> 64;

    let mid_sum = mid1 as u64 as u128 + mid2 as u64 as u128 + carry_from_lo;
    let carry_from_mid = mid_sum >> 64;

    let qhat = high + (mid1 >> 64) + (mid2 >> 64) + carry_from_mid;

    // Approximate remainder: r = p - qhat * q
    let q = ctx.modulus as u128;
    let r = (p - qhat * q) as u64;

    // Branchless correction: r may be in [0, 2q), so at most one subtraction
    let (corrected, borrow) = r.overflowing_sub(ctx.modulus);
    if borrow {
        r
    } else {
        corrected
    }
}

// ---------------------------------------------------------------------------
// Montgomery multiplication
// ---------------------------------------------------------------------------

/// Montgomery multiplication (REDC).
///
/// Returns `a_mont * b_mont * R⁻¹ mod q` where both inputs are in Montgomery domain.
///
/// If `a_mont = a·R mod q` and `b_mont = b·R mod q`, the result is `(a·b)·R mod q`.
///
/// # Algorithm (REDC)
/// 1. `t = a * b` (u128)
/// 2. `m = (t mod R) * (−q⁻¹ mod R) mod R`
/// 3. `u = (t + m·q) / R` (exact division since t + m·q ≡ 0 mod R)
/// 4. Conditional subtraction if u ≥ q
#[inline(always)]
pub fn mod_mul_mont(a_mont: u64, b_mont: u64, ctx: &Ntt64Arith) -> u64 {
    let t = a_mont as u128 * b_mont as u128;

    // m = (t mod 2^64) * neg_inv mod 2^64
    let t_lo = t as u64;
    let m = t_lo.wrapping_mul(ctx.mont_neg_inv);

    // u = (t + m * q) >> 64
    // t + m·q is guaranteed divisible by 2^64 (that's the whole point of Montgomery)
    let mq = m as u128 * ctx.modulus as u128;
    let sum = t + mq; // Cannot overflow u128: t < q² < 2^124 and mq < 2^64 * 2^62 = 2^126
    let u = (sum >> 64) as u64;

    // Branchless final correction: u may be in [0, 2q)
    let (corrected, borrow) = u.overflowing_sub(ctx.modulus);
    if borrow {
        u
    } else {
        corrected
    }
}

/// Converts a value into Montgomery domain.
///
/// `to_montgomery(a) = a · R mod q`
///
/// Computed via `REDC(a, R²) = a · R² · R⁻¹ mod q = a · R mod q`.
#[inline(always)]
pub fn to_montgomery(a: u64, ctx: &Ntt64Arith) -> u64 {
    mod_mul_mont(a, ctx.mont_r2, ctx)
}

/// Converts a value out of Montgomery domain.
///
/// `from_montgomery(a_mont) = a_mont · R⁻¹ mod q = REDC(a_mont, 1)`
#[inline(always)]
pub fn from_montgomery(a_mont: u64, ctx: &Ntt64Arith) -> u64 {
    mod_mul_mont(a_mont, 1, ctx)
}

// ---------------------------------------------------------------------------
// Exponentiation and inverse
// ---------------------------------------------------------------------------

/// Fast modular exponentiation (square-and-multiply).
///
/// Returns `base^exp mod q` using Barrett multiplication.
/// Complexity: O(log exp) multiplications.
#[inline(always)]
pub fn mod_pow(base: u64, exp: u64, ctx: &Ntt64Arith) -> u64 {
    if exp == 0 {
        return 1;
    }
    if base == 0 {
        return 0;
    }

    let mut result = 1u64;
    let mut b = base % ctx.modulus;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = mod_mul_barrett(result, b, ctx);
        }
        e >>= 1;
        if e > 0 {
            b = mod_mul_barrett(b, b, ctx);
        }
    }

    result
}

/// Modular inverse via Fermat's little theorem.
///
/// For prime q: `a⁻¹ ≡ a^(q−2) (mod q)`.
///
/// # Panics
/// If `a == 0` (zero has no inverse).
#[inline(always)]
pub fn mod_inv(a: u64, ctx: &Ntt64Arith) -> u64 {
    assert!(a != 0, "zero has no modular inverse");
    assert!(a < ctx.modulus, "mod_inv: a must be < modulus");
    mod_pow(a, ctx.modulus - 2, ctx)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive reference multiplication via u128.
    fn naive_mod_mul(a: u64, b: u64, m: u64) -> u64 {
        ((a as u128 * b as u128) % m as u128) as u64
    }

    /// Naive reference exponentiation via u128.
    fn naive_mod_pow(base: u64, exp: u64, m: u64) -> u64 {
        let mut result = 1u128;
        let mut b = base as u128 % m as u128;
        let mut e = exp;
        while e > 0 {
            if e & 1 == 1 {
                result = result * b % m as u128;
            }
            e >>= 1;
            if e > 0 {
                b = b * b % m as u128;
            }
        }
        result as u64
    }

    // --- Context construction ---

    #[test]
    fn test_ctx_creation() {
        let ctx = Ntt64Arith::new(17);
        assert_eq!(ctx.modulus, 17);
        assert_eq!(ctx.mont_r as u128, (1u128 << 64) % 17u128);
        let r = ctx.mont_r as u128;
        assert_eq!(ctx.mont_r2, (r * r % 17) as u64);
        assert_eq!(
            ctx.modulus.wrapping_mul(ctx.mont_neg_inv).wrapping_add(1),
            0,
        );
    }

    #[test]
    fn test_ctx_ntt_primes() {
        for &p in &[PRIME_60_1, PRIME_SEAL, PRIME_62_1, PRIME_60_2, PRIME_60_3] {
            assert!(p < (1u64 << 62));
            assert!(p & 1 == 1);
            let ctx = Ntt64Arith::new(p);
            assert_eq!(
                ctx.modulus.wrapping_mul(ctx.mont_neg_inv).wrapping_add(1),
                0,
            );
        }
    }

    #[test]
    #[should_panic(expected = "odd")]
    fn test_ctx_even_modulus_panics() {
        Ntt64Arith::new(10);
    }

    #[test]
    #[should_panic(expected = ">= 2")]
    fn test_ctx_modulus_too_small() {
        Ntt64Arith::new(1);
    }

    // --- Modular add ---

    #[test]
    fn test_mod_add_basic() {
        let m = 17;
        assert_eq!(mod_add(3, 5, m), 8);
        assert_eq!(mod_add(10, 10, m), 3);
        assert_eq!(mod_add(0, 0, m), 0);
        assert_eq!(mod_add(16, 0, m), 16);
        assert_eq!(mod_add(0, 16, m), 16);
    }

    #[test]
    fn test_mod_add_edge_cases() {
        let m = 17;
        assert_eq!(mod_add(10, 7, m), 0);
        assert_eq!(mod_add(16, 16, m), 15);
    }

    #[test]
    fn test_mod_add_large_modulus() {
        let m = PRIME_60_1;
        let a = m - 1;
        let b = m - 1;
        let expected = ((a as u128 + b as u128) % m as u128) as u64;
        assert_eq!(mod_add(a, b, m), expected);
        assert_eq!(mod_add(0, a, m), a);
        assert_eq!(mod_add(a, 0, m), a);
    }

    // --- Modular sub ---

    #[test]
    fn test_mod_sub_basic() {
        let m = 17;
        assert_eq!(mod_sub(10, 3, m), 7);
        assert_eq!(mod_sub(3, 10, m), 10);
        assert_eq!(mod_sub(0, 0, m), 0);
        assert_eq!(mod_sub(5, 5, m), 0);
    }

    #[test]
    fn test_mod_sub_edge_cases() {
        let m = 17;
        assert_eq!(mod_sub(0, 16, m), 1);
        assert_eq!(mod_sub(0, 1, m), 16);
        assert_eq!(mod_sub(16, 0, m), 16);
    }

    #[test]
    fn test_mod_sub_large_modulus() {
        let m = PRIME_60_1;
        assert_eq!(mod_sub(0, 0, m), 0);
        assert_eq!(mod_sub(1, 0, m), 1);
        assert_eq!(mod_sub(0, 1, m), m - 1);
        assert_eq!(mod_sub(m - 1, m - 1, m), 0);
    }

    // --- Barrett multiplication ---

    #[test]
    fn test_mod_mul_barrett_basic() {
        let ctx = Ntt64Arith::new(17);
        assert_eq!(mod_mul_barrett(3, 5, &ctx), 15);
        assert_eq!(mod_mul_barrett(4, 5, &ctx), 3);
        assert_eq!(mod_mul_barrett(0, 12, &ctx), 0);
        assert_eq!(mod_mul_barrett(1, 12, &ctx), 12);
    }

    #[test]
    fn test_mod_mul_barrett_vs_naive() {
        let ctx = Ntt64Arith::new(PRIME_60_1);
        let m = ctx.modulus;
        let test_values: Vec<u64> = vec![
            0,
            1,
            2,
            m - 1,
            m - 2,
            m / 2,
            m / 3,
            123456789,
            987654321,
            (1u64 << 30) + 7,
            (1u64 << 40) - 3,
        ];
        for &a in &test_values {
            for &b in &test_values {
                let a = a % m;
                let b = b % m;
                let expected = naive_mod_mul(a, b, m);
                let got = mod_mul_barrett(a, b, &ctx);
                assert_eq!(got, expected, "Barrett fails for a={a}, b={b}, m={m}");
            }
        }
    }

    #[test]
    fn test_mod_mul_barrett_all_primes() {
        for &p in &[PRIME_60_1, PRIME_SEAL, PRIME_62_1, PRIME_60_2, PRIME_60_3] {
            let ctx = Ntt64Arith::new(p);
            let a = p - 1;
            let b = p - 1;
            let expected = naive_mod_mul(a, b, p);
            let got = mod_mul_barrett(a, b, &ctx);
            assert_eq!(got, expected, "Barrett fails for p={p}");
        }
    }

    // --- Montgomery ---

    #[test]
    fn test_montgomery_roundtrip() {
        let ctx = Ntt64Arith::new(17);
        for a in 0..17u64 {
            let a_mont = to_montgomery(a, &ctx);
            let a_back = from_montgomery(a_mont, &ctx);
            assert_eq!(a_back, a, "Roundtrip fails for a={a}, a_mont={a_mont}");
        }
    }

    #[test]
    fn test_montgomery_roundtrip_large() {
        let ctx = Ntt64Arith::new(PRIME_60_1);
        let m = ctx.modulus;
        let values: Vec<u64> = vec![0, 1, 2, m - 1, m - 2, m / 2, 123456789, m / 3];
        for &a in &values {
            let a_mont = to_montgomery(a, &ctx);
            let a_back = from_montgomery(a_mont, &ctx);
            assert_eq!(a_back, a, "Roundtrip fails for a={a}");
        }
    }

    #[test]
    fn test_montgomery_mul() {
        let ctx = Ntt64Arith::new(17);
        for a in 0..17u64 {
            for b in 0..17u64 {
                let expected = naive_mod_mul(a, b, 17);
                let a_m = to_montgomery(a, &ctx);
                let b_m = to_montgomery(b, &ctx);
                let c_m = mod_mul_mont(a_m, b_m, &ctx);
                let c = from_montgomery(c_m, &ctx);
                assert_eq!(c, expected, "Montgomery mul fails for a={a}, b={b}");
            }
        }
    }

    #[test]
    fn test_montgomery_mul_large() {
        let ctx = Ntt64Arith::new(PRIME_SEAL);
        let m = ctx.modulus;
        let values: Vec<u64> = vec![1, 2, m - 1, m / 2, 999999937, 1000000007 % m];
        for &a in &values {
            for &b in &values {
                let expected = naive_mod_mul(a, b, m);
                let a_m = to_montgomery(a, &ctx);
                let b_m = to_montgomery(b, &ctx);
                let c_m = mod_mul_mont(a_m, b_m, &ctx);
                let c = from_montgomery(c_m, &ctx);
                assert_eq!(c, expected, "Montgomery mul fails for a={a}, b={b}, q={m}");
            }
        }
    }

    // --- Modular exponentiation ---

    #[test]
    fn test_mod_pow_known() {
        let ctx = Ntt64Arith::new(1000003);
        assert_eq!(mod_pow(2, 10, &ctx), 1024);

        let ctx2 = Ntt64Arith::new(1009);
        assert_eq!(mod_pow(2, 10, &ctx2), 15);

        assert_eq!(mod_pow(12345, 0, &ctx), 1);
        assert_eq!(mod_pow(42, 1, &ctx), 42);
        assert_eq!(mod_pow(0, 100, &ctx), 0);
    }

    #[test]
    fn test_mod_pow_fermat() {
        let ctx = Ntt64Arith::new(PRIME_60_1);
        let p = ctx.modulus;
        assert_eq!(mod_pow(2, p - 1, &ctx), 1);
        assert_eq!(mod_pow(3, p - 1, &ctx), 1);
        assert_eq!(mod_pow(p - 1, p - 1, &ctx), 1);
        assert_eq!(mod_pow(123456789, p - 1, &ctx), 1);
    }

    #[test]
    fn test_mod_pow_vs_naive() {
        let ctx = Ntt64Arith::new(PRIME_60_1);
        let m = ctx.modulus;
        for base in [2u64, 3, 7, 11, 13, m - 1] {
            for exp in [0u64, 1, 2, 3, 10, 100, 1000] {
                let expected = naive_mod_pow(base, exp, m);
                let got = mod_pow(base, exp, &ctx);
                assert_eq!(got, expected, "mod_pow fails for base={base}, exp={exp}");
            }
        }
    }

    // --- Modular inverse ---

    #[test]
    fn test_mod_inv_basic() {
        let ctx = Ntt64Arith::new(17);
        for a in 1..17u64 {
            let inv = mod_inv(a, &ctx);
            let product = mod_mul_barrett(a, inv, &ctx);
            assert_eq!(product, 1, "a={a}, inv={inv}, a*inv mod 17 = {product}");
        }
    }

    #[test]
    fn test_mod_inv_large_prime() {
        let ctx = Ntt64Arith::new(PRIME_SEAL);
        let m = ctx.modulus;
        let values = [1u64, 2, 3, m - 1, m - 2, m / 2, 999999937 % m];
        for &a in &values {
            let inv = mod_inv(a, &ctx);
            let product = mod_mul_barrett(a, inv, &ctx);
            assert_eq!(product, 1, "Inverse fails for a={a}, q={m}");
        }
    }

    #[test]
    #[should_panic(expected = "zero")]
    fn test_mod_inv_zero_panics() {
        let ctx = Ntt64Arith::new(17);
        mod_inv(0, &ctx);
    }

    // --- SEAL prime properties ---

    #[test]
    fn test_seal_prime_properties() {
        let q = PRIME_SEAL;
        let ctx = Ntt64Arith::new(q);
        assert_eq!(q % (1 << 21), 1, "SEAL prime is not ≡ 1 mod 2^21");
        for &base in &[2u64, 3, 5, 7, 11, 13] {
            assert_eq!(mod_pow(base, q - 1, &ctx), 1);
        }
    }

    // --- Barrett vs Montgomery consistency ---

    #[test]
    fn test_consistency_barrett_vs_montgomery() {
        let ctx = Ntt64Arith::new(PRIME_60_1);
        let m = ctx.modulus;
        let values = [1u64, 2, 3, m - 1, m / 2, 777777777, 42];
        for &a in &values {
            let a = a % m;
            for &b in &values {
                let b = b % m;
                let barrett_result = mod_mul_barrett(a, b, &ctx);
                let a_m = to_montgomery(a, &ctx);
                let b_m = to_montgomery(b, &ctx);
                let mont_result = from_montgomery(mod_mul_mont(a_m, b_m, &ctx), &ctx);
                assert_eq!(barrett_result, mont_result);
            }
        }
    }
}
