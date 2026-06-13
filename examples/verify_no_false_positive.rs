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

#![allow(
    unused_variables,
    unused_imports,
    unused_mut,
    dead_code,
    clippy::needless_range_loop
)]
// =============================================================================
// Anti-False-Positive Verification
// =============================================================================
// Ensures our tests aren't passing trivially:
// 1. Forward NTT must actually CHANGE the data
// 2. Inverse NTT must actually CHANGE the data
// 3. Forward != identity (not a no-op)
// 4. Inverse != identity (not a no-op)
// 5. Forward(Forward(x)) != x (not self-inverse)
// 6. Cross-validate NEON vs Scalar
// 7. Known-answer test for small NTT

use vaea_ntt::ntt32::{generate_primes_28, Ntt32Context};

fn main() {
    let mut pass = 0u32;
    let mut fail = 0u32;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  Anti-False-Positive Verification                      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Test 1: Forward NTT changes data for ALL tested configurations
    // =========================================================================
    println!("── Test 1: Forward actually transforms ───────────────────");

    let configs: Vec<(usize, u32)> = vec![
        (2, 5),
        (4, 17),
        (8, 17),
        (16, 97),
        (32, 97),
        (64, 769),
        (128, 769),
        (256, 12289),
        (256, 8380417),
        (512, 12289),
        (1024, 12289),
    ];

    for &(n, q) in &configs {
        // Check q is NTT-friendly
        if !(q as u64 - 1).is_multiple_of(2 * n as u64) {
            continue;
        }

        let ctx = Ntt32Context::new(n, q);

        // Non-trivial input: impulse [1, 0, 0, ...]
        let mut data = vec![0u32; n];
        data[0] = 1;
        let before = data.clone();
        ctx.forward(&mut data);

        if data == before {
            eprintln!("  ❌ Forward is a NO-OP for N={n} q={q}!");
            fail += 1;
        } else {
            // Count how many elements changed
            let changed = data
                .iter()
                .zip(before.iter())
                .filter(|(a, b)| a != b)
                .count();
            if changed < n / 2 {
                eprintln!("  ⚠️  Forward only changed {changed}/{n} elements for N={n} q={q}");
            }
            pass += 1;
        }

        // Also check that forward output is NOT all zeros
        if data.iter().all(|&x| x == 0) {
            eprintln!("  ❌ Forward produced all zeros for N={n} q={q}!");
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!("  Done: {pass} pass, {fail} fail\n");

    // =========================================================================
    // Test 2: Inverse NTT changes data (not identity)
    // =========================================================================
    println!("── Test 2: Inverse actually transforms ───────────────────");

    let t2_start = pass;
    let t2_fail_start = fail;

    for &(n, q) in &configs {
        if !(q as u64 - 1).is_multiple_of(2 * n as u64) {
            continue;
        }
        let ctx = Ntt32Context::new(n, q);

        // Apply inverse to non-NTT data — should change it
        let mut data: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 3) % q).collect();
        let before = data.clone();
        ctx.inverse(&mut data);

        if data == before {
            eprintln!("  ❌ Inverse is a NO-OP for N={n} q={q}!");
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!(
        "  Done: {} pass, {} fail\n",
        pass - t2_start,
        fail - t2_fail_start
    );

    // =========================================================================
    // Test 3: Forward(Forward(x)) != x (forward is NOT self-inverse)
    // =========================================================================
    println!("── Test 3: Forward is not self-inverse ───────────────────");

    let t3_start = pass;
    let t3_fail_start = fail;

    for &(n, q) in &configs {
        if !(q as u64 - 1).is_multiple_of(2 * n as u64) {
            continue;
        }
        let ctx = Ntt32Context::new(n, q);

        let original: Vec<u32> = (0..n).map(|i| (i as u32 * 13 + 5) % q).collect();
        let mut data = original.clone();
        ctx.forward(&mut data);
        ctx.forward(&mut data); // double forward

        if data == original {
            eprintln!("  ❌ Double-forward = identity for N={n} q={q} (forward is self-inverse!)");
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!(
        "  Done: {} pass, {} fail\n",
        pass - t3_start,
        fail - t3_fail_start
    );

    // =========================================================================
    // Test 4: NEON vs Scalar cross-validation
    // =========================================================================
    println!("── Test 4: NEON vs Scalar cross-validation ───────────────");

    let t4_start = pass;
    let t4_fail_start = fail;

    for &(n, q) in &configs {
        if !(q as u64 - 1).is_multiple_of(2 * n as u64) || n < 8 {
            continue; // scalar fallback for n < 8, skip
        }
        let ctx = Ntt32Context::new(n, q);

        let input: Vec<u32> = (0..n).map(|i| (i as u32 * 41 + 17) % q).collect();

        // NEON path (via ctx.forward)
        let mut neon_data = input.clone();
        ctx.forward(&mut neon_data);

        // Scalar path
        let mut scalar_data = input.clone();
        vaea_ntt::ntt32::scalar::ntt_forward_scalar(&mut scalar_data, &ctx);

        if neon_data != scalar_data {
            eprintln!("  ❌ NEON != Scalar forward for N={n} q={q}!");
            // Show first diff
            for (idx, (a, b)) in neon_data.iter().zip(scalar_data.iter()).enumerate() {
                if a != b {
                    eprintln!("      first diff at [{idx}]: NEON={a}, Scalar={b}");
                    break;
                }
            }
            fail += 1;
        } else {
            pass += 1;
        }

        // Also check inverse
        let mut neon_inv = neon_data.clone();
        ctx.inverse(&mut neon_inv);

        let mut scalar_inv = scalar_data.clone();
        vaea_ntt::ntt32::scalar::ntt_inverse_scalar(&mut scalar_inv, &ctx);

        if neon_inv != scalar_inv {
            eprintln!("  ❌ NEON != Scalar inverse for N={n} q={q}!");
            fail += 1;
        } else {
            pass += 1;
        }

        // Both should equal the original
        if neon_inv != input {
            eprintln!("  ❌ NEON roundtrip != original for N={n} q={q}!");
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!(
        "  Done: {} pass, {} fail\n",
        pass - t4_start,
        fail - t4_fail_start
    );

    // =========================================================================
    // Test 5: Known-answer test (hand-computed NTT for N=8, q=17)
    // =========================================================================
    println!("── Test 5: Known-answer test (N=8, q=17) ─────────────────");

    let t5_start = pass;
    let t5_fail_start = fail;

    {
        // q=17, N=8
        // q-1 = 16 = 2*8, so q ≡ 1 (mod 16) ✓
        // Primitive 16th root of unity: g=3 (since 3^16 ≡ 1 mod 17, 3^8 ≡ -1 mod 17)
        // Check: 3^8 mod 17 = 6561 mod 17 = 16 = -1 ✓
        let ctx = Ntt32Context::new(8, 17);

        // Input: [1, 0, 0, 0, 0, 0, 0, 0] (unit impulse)
        // NTT of unit impulse should give all-ones (up to twiddle ordering)
        // Actually in negacyclic NTT, NTT([1,0,...,0]) should give something specific

        let mut impulse = vec![0u32; 8];
        impulse[0] = 1;
        let orig = impulse.clone();
        ctx.forward(&mut impulse);

        // All output values must be in [0, 17)
        let all_valid = impulse.iter().all(|&x| x < 17);
        if !all_valid {
            eprintln!("  ❌ KAT: output not in [0,17)");
            fail += 1;
        } else {
            pass += 1;
        }

        // NTT of [1,0,...,0] in negacyclic should be all 1's
        // (since x_hat[k] = sum(x[i] * psi^(2*bit_rev(k)+1)*i) = psi^0 = 1 for impulse at 0)
        // Actually this depends on the twiddle convention. Let's just check roundtrip.
        ctx.inverse(&mut impulse);
        if impulse != orig {
            eprintln!("  ❌ KAT: roundtrip failed for impulse");
            fail += 1;
        } else {
            pass += 1;
        }

        // Manual polynomial multiplication check:
        // In Z_17[X]/(X^8+1): (1+X) * (1+X) = 1 + 2X + X^2
        let mut a = vec![0u32; 8];
        a[0] = 1;
        a[1] = 1;
        let result = ctx.negacyclic_mul(&a, &a);

        let expected = [1u32, 2, 1, 0, 0, 0, 0, 0];
        if result != expected {
            eprintln!("  ❌ KAT: (1+X)^2 = {:?}, expected {:?}", result, expected);
            fail += 1;
        } else {
            pass += 1;
        }

        // (X^7) * (X) = X^8 = -1 mod (X^8+1) = 17-1 = 16 in Z_17
        let mut x7 = vec![0u32; 8];
        x7[7] = 1;
        let mut x1 = vec![0u32; 8];
        x1[1] = 1;
        let result = ctx.negacyclic_mul(&x7, &x1);

        // X^7 * X = X^8 ≡ -1 mod (X^8+1) → constant term = q-1 = 16, rest = 0
        let mut expected2 = vec![0u32; 8];
        expected2[0] = 16; // -1 mod 17
        if result != expected2 {
            eprintln!("  ❌ KAT: X^7 * X = {:?}, expected {:?}", result, expected2);
            fail += 1;
        } else {
            pass += 1;
        }

        // (X^4) * (X^4) = X^8 ≡ -1 mod (X^8+1)
        let mut x4 = vec![0u32; 8];
        x4[4] = 1;
        let result = ctx.negacyclic_mul(&x4, &x4);
        if result != expected2 {
            eprintln!(
                "  ❌ KAT: X^4 * X^4 = {:?}, expected {:?}",
                result, expected2
            );
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!(
        "  Done: {} pass, {} fail\n",
        pass - t5_start,
        fail - t5_fail_start
    );

    // =========================================================================
    // Test 6: Verify NTT preserves linearity
    // =========================================================================
    println!("── Test 6: Linearity NTT(a+b) = NTT(a) + NTT(b) ────────");

    let t6_start = pass;
    let t6_fail_start = fail;

    for &(n, q) in &configs {
        if !(q as u64 - 1).is_multiple_of(2 * n as u64) {
            continue;
        }
        let ctx = Ntt32Context::new(n, q);

        let a: Vec<u32> = (0..n).map(|i| (i as u32 * 3 + 1) % q).collect();
        let b: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 5) % q).collect();

        // NTT(a)
        let mut ntt_a = a.clone();
        ctx.forward(&mut ntt_a);

        // NTT(b)
        let mut ntt_b = b.clone();
        ctx.forward(&mut ntt_b);

        // NTT(a) + NTT(b) mod q
        let sum_ntt: Vec<u32> = ntt_a
            .iter()
            .zip(ntt_b.iter())
            .map(|(&x, &y)| (x as u64 + y as u64) as u32 % q)
            .collect();

        // a + b mod q, then NTT
        let ab_sum: Vec<u32> = a.iter().zip(b.iter()).map(|(&x, &y)| (x + y) % q).collect();
        let mut ntt_ab = ab_sum;
        ctx.forward(&mut ntt_ab);

        if sum_ntt != ntt_ab {
            eprintln!("  ❌ Linearity violated for N={n} q={q}!");
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!(
        "  Done: {} pass, {} fail\n",
        pass - t6_start,
        fail - t6_fail_start
    );

    // =========================================================================
    // Test 7: Convolution theorem: NTT(a*b) = NTT(a) · NTT(b)
    // =========================================================================
    println!("── Test 7: Convolution theorem ───────────────────────────");

    let t7_start = pass;
    let t7_fail_start = fail;

    for &n in &[8, 16, 64, 256] {
        let primes = generate_primes_28(n, 1);
        let q = primes[0];
        let ctx = Ntt32Context::new(n, q);

        let a: Vec<u32> = (0..n).map(|i| (i as u32 * 11 + 3) % q).collect();
        let b: Vec<u32> = (0..n).map(|i| (i as u32 * 23 + 7) % q).collect();

        // Method 1: Direct negacyclic multiplication
        let product = ctx.negacyclic_mul(&a, &b);

        // Method 2: NTT(a) · NTT(b) pointwise, then INTT
        let mut ntt_a = a.clone();
        ctx.forward(&mut ntt_a);
        let mut ntt_b = b.clone();
        ctx.forward(&mut ntt_b);

        // Pointwise multiply mod q
        let mut pointwise: Vec<u32> = ntt_a
            .iter()
            .zip(ntt_b.iter())
            .map(|(&x, &y)| ((x as u64 * y as u64) % q as u64) as u32)
            .collect();
        ctx.inverse(&mut pointwise);

        if product != pointwise {
            eprintln!("  ❌ Convolution theorem violated for N={n} q={q}!");
            // Show first diff
            for (idx, (&a, &b)) in product.iter().zip(pointwise.iter()).enumerate() {
                if a != b {
                    eprintln!("      first diff at [{idx}]: mul={a}, conv={b}");
                    break;
                }
            }
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!(
        "  Done: {} pass, {} fail\n",
        pass - t7_start,
        fail - t7_fail_start
    );

    // =========================================================================
    // Summary
    // =========================================================================
    let total = pass + fail;
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  TOTAL: {pass:>4} pass | {fail:>3} fail | {total:>4} total              ║");
    if fail == 0 {
        println!("║  ✅  NO FALSE POSITIVES — ALL VERIFICATIONS PASSED      ║");
    } else {
        println!("║  ❌  FALSE POSITIVES DETECTED                           ║");
    }
    println!("╚══════════════════════════════════════════════════════════╝");

    if fail > 0 {
        std::process::exit(1);
    }
}
