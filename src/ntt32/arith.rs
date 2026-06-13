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

//! # 28-bit Modular Arithmetic
//!
//! Ultra-fast modular operations for primes < 2^28.
//!
//! The key insight: the product of two 28-bit numbers fits in u64 (56 bits max),
//! so we never need u128 — the entire NTT pipeline stays in 64-bit registers.
//!
//! All conditional reductions are **branchless** (constant-time) using
//! `wrapping_sub(q & mask)` to avoid timing side-channels.

// ===========================================================================
// Modular multiplication, addition, subtraction — all branchless
// ===========================================================================

/// Modular multiplication for primes < 2^28.
///
/// The product of two 28-bit values fits in a u64 (56 bits max),
/// so reduction uses a single `%` on u64. No u128 needed.
///
/// # Panics (debug only)
/// Asserts that `a < q`, `b < q`, and `q < 2^28`.
#[inline(always)]
pub fn mod_mul_28(a: u32, b: u32, q: u32) -> u32 {
    debug_assert!(a < q, "mod_mul_28: a={a} >= q={q}");
    debug_assert!(b < q, "mod_mul_28: b={b} >= q={q}");
    debug_assert!(q < (1u32 << 28), "mod_mul_28: q={q} >= 2^28");
    ((a as u64 * b as u64) % q as u64) as u32
}

/// Branchless modular addition for primes < 2^28.
///
/// The sum of two values < 2^28 fits in u32 (< 2^29),
/// so there is no overflow risk. The conditional reduction is constant-time.
///
/// # Panics (debug only)
/// Asserts that `a < q` and `b < q`.
#[inline(always)]
pub fn mod_add_28(a: u32, b: u32, q: u32) -> u32 {
    debug_assert!(a < q, "mod_add_28: a={a} >= q={q}");
    debug_assert!(b < q, "mod_add_28: b={b} >= q={q}");
    let s = a + b;
    // Branchless: if s >= q then s - q, else s
    let mask = ((s >= q) as u32).wrapping_neg();
    s.wrapping_sub(q & mask)
}

/// Branchless modular subtraction for primes < 2^28.
///
/// Returns `(a - b) mod q` without branching.
/// If `a >= b`, returns `a - b`. Otherwise returns `a + q - b`.
///
/// # Panics (debug only)
/// Asserts that `a < q` and `b < q`.
#[inline(always)]
pub fn mod_sub_28(a: u32, b: u32, q: u32) -> u32 {
    debug_assert!(a < q, "mod_sub_28: a={a} >= q={q}");
    debug_assert!(b < q, "mod_sub_28: b={b} >= q={q}");
    // Branchless: if a < b then a + q - b, else a - b
    // Equivalent: a.wrapping_sub(b).wrapping_add(q & mask)  where mask = (a < b)
    let mask = ((a < b) as u32).wrapping_neg();
    a.wrapping_sub(b).wrapping_add(q & mask)
}

// ===========================================================================
// Modular exponentiation and inverse (32-bit)
// ===========================================================================

/// Fast modular exponentiation using square-and-multiply (32-bit).
///
/// Computes `base^exp mod q` using only u64 arithmetic.
/// Since q < 2^28, intermediate products fit in u64.
pub fn mod_pow_32(base: u32, exp: u32, q: u32) -> u32 {
    if exp == 0 {
        return 1;
    }
    if base == 0 {
        return 0;
    }

    let mut result = 1u64;
    let mut b = (base % q) as u64;
    let q64 = q as u64;
    let mut e = exp;

    while e > 0 {
        if e & 1 == 1 {
            result = result * b % q64;
        }
        e >>= 1;
        if e > 0 {
            b = b * b % q64;
        }
    }

    result as u32
}

/// Modular inverse via Fermat's little theorem (32-bit).
///
/// For prime q: `a^{-1} ≡ a^{q-2} (mod q)`.
///
/// # Panics
/// If `a == 0` (zero has no inverse).
pub fn mod_inv_32(a: u32, q: u32) -> u32 {
    assert!(a != 0, "No inverse for zero");
    mod_pow_32(a, q - 2, q)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mod_mul_28_basic() {
        let q = 268_435_399u32;
        assert_eq!(mod_mul_28(0, 123, q), 0);
        assert_eq!(mod_mul_28(1, 42, q), 42);
        assert_eq!(mod_mul_28(42, 1, q), 42);

        let a = q - 1;
        let b = q - 1;
        let expected = ((a as u64 * b as u64) % q as u64) as u32;
        assert_eq!(mod_mul_28(a, b, q), expected);
    }

    #[test]
    fn test_mod_add_sub_28() {
        let q = 100_000_007u32;
        assert_eq!(mod_add_28(3, 5, q), 8);
        assert_eq!(mod_add_28(q - 1, 1, q), 0);
        assert_eq!(mod_add_28(q - 1, q - 1, q), q - 2);

        assert_eq!(mod_sub_28(5, 3, q), 2);
        assert_eq!(mod_sub_28(3, 5, q), q - 2);
        assert_eq!(mod_sub_28(0, 0, q), 0);
        assert_eq!(mod_sub_28(0, 1, q), q - 1);
    }

    #[test]
    fn test_mod_pow_32_basic() {
        let q = 1009u32;
        assert_eq!(mod_pow_32(2, 10, q), 15);
        assert_eq!(mod_pow_32(42, 0, q), 1);
        assert_eq!(mod_pow_32(42, 1, q), 42);
        assert_eq!(mod_pow_32(0, 100, q), 0);
        assert_eq!(mod_pow_32(7, q - 1, q), 1);
    }

    #[test]
    fn test_mod_inv_32_basic() {
        let q = 17u32;
        for a in 1..q {
            let inv = mod_inv_32(a, q);
            let prod = mod_mul_28(a, inv, q);
            assert_eq!(prod, 1, "Inverse failed for a={a}, q={q}, inv={inv}");
        }
    }
}
