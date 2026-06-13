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
// VaeaNTT — Exhaustive Stress Test
// =============================================================================
// Tests every combination of:
//   - N: 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
//   - q: small (3329), medium (12289), large (8380417), near-28bit, boundary primes
//   - Data patterns: zeros, ones, max (q-1), sequential, impulse, alternating
//   - Checks: roundtrip, full reduction, multiplication, PQ presets

use vaea_ntt::ntt32::{generate_primes_28, Ntt32Context};
use vaea_ntt::pq::{PqNtt, PqScheme};

fn main() {
    let mut pass = 0u32;
    let mut fail = 0u32;
    let mut skip = 0u32;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║     VaeaNTT — Exhaustive Stress Test                    ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    // =========================================================================
    // Phase 1: All N × all known primes
    // =========================================================================
    println!("── Phase 1: Roundtrip N × q ──────────────────────────────");

    let sizes: &[usize] = &[
        2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
    ];

    // Known primes used in real cryptographic schemes
    let known_primes: &[(u32, &str)] = &[
        (5, "q=5 (tiny)"),
        (17, "q=17 (tiny)"),
        (97, "q=97 (small)"),
        (257, "q=257 (Fermat)"),
        (769, "q=769"),
        (3329, "q=3329 (ML-KEM)"),
        (7681, "q=7681 (NewHope)"),
        (12289, "q=12289 (NTRU)"),
        (40961, "q=40961"),
        (65537, "q=65537 (Fermat)"),
        (8380417, "q=8380417 (ML-DSA)"),
        (104857601, "q=104857601 (~27bit)"),
        (268369921, "q=268369921 (~28bit)"),
    ];

    for &n in sizes {
        for &(q, name) in known_primes {
            let two_n = 2 * n as u64;
            if (q as u64 - 1) % two_n != 0 {
                skip += 1;
                continue;
            }
            if q >= (1u32 << 28) {
                skip += 1;
                continue;
            }

            let ctx = Ntt32Context::new(n, q);

            // --- Test patterns ---
            let patterns: Vec<(&str, Vec<u32>)> = vec![
                ("zeros", vec![0u32; n]),
                ("ones", vec![1u32; n]),
                ("max", vec![q - 1; n]),
                ("sequential", (0..n as u32).map(|i| i % q).collect()),
                ("impulse", {
                    let mut v = vec![0u32; n];
                    v[0] = 1;
                    v
                }),
                (
                    "alternating",
                    (0..n).map(|i| if i % 2 == 0 { 0 } else { q - 1 }).collect(),
                ),
                (
                    "pseudo_random",
                    (0..n)
                        .map(|i| ((i as u64 * 314159265 + 271828182) % q as u64) as u32)
                        .collect(),
                ),
            ];

            for (pat_name, original) in &patterns {
                // Forward + inverse roundtrip
                let mut data = original.clone();
                ctx.forward(&mut data);

                // Check full reduction after forward
                let all_reduced = data.iter().all(|&x| x < q);
                if !all_reduced {
                    eprintln!("  ❌ FAIL: N={n:>5} {name:>30} [{pat_name:>12}] — not fully reduced after forward");
                    fail += 1;
                } else {
                    pass += 1;
                }

                // Check forward actually changed data (unless input is all zeros)
                if *pat_name != "zeros" && data == *original && n > 1 {
                    eprintln!(
                        "  ❌ FAIL: N={n:>5} {name:>30} [{pat_name:>12}] — forward did nothing"
                    );
                    fail += 1;
                } else {
                    pass += 1;
                }

                // Inverse roundtrip
                ctx.inverse(&mut data);
                if data != *original {
                    eprintln!(
                        "  ❌ FAIL: N={n:>5} {name:>30} [{pat_name:>12}] — roundtrip mismatch"
                    );
                    // Show first diff
                    for (idx, (a, b)) in data.iter().zip(original.iter()).enumerate() {
                        if a != b {
                            eprintln!("         first diff at [{idx}]: got {a}, expected {b}");
                            break;
                        }
                    }
                    fail += 1;
                } else {
                    pass += 1;
                }
            }
        }
    }

    println!("  Phase 1: {pass} pass, {fail} fail, {skip} skip\n");

    // =========================================================================
    // Phase 2: Barrett interval boundary primes
    // =========================================================================
    println!("── Phase 2: Barrett boundary primes ──────────────────────");

    let phase2_start = pass;
    let phase2_fail_start = fail;

    // Find primes where barrett_interval is exactly 4, 5, 6, 7, 8
    // barrett_interval = floor((2^31/q - 1) / 2)
    // For interval=k: q ≈ 2^31 / (2k+1)
    for target_interval in 3..=12 {
        let approx_q = (1u64 << 31) / (2 * target_interval as u64 + 1);
        // Search for NTT-friendly prime near approx_q for N=256
        let two_n = 512u64; // 2*256
        let mut found = false;
        for delta in 0..1000 {
            for sign in &[1i64, -1i64] {
                let candidate = (approx_q as i64 + sign * delta as i64) as u64;
                if candidate < 3 || candidate >= (1u64 << 28) {
                    continue;
                }
                if (candidate - 1) % two_n != 0 {
                    continue;
                }
                let q32 = candidate as u32;
                if !vaea_ntt::ntt32::is_prime_32(q32) {
                    continue;
                }

                // Verify barrett_interval
                let max_b = ((1u64 << 31) / candidate) as u32;
                let bi = max_b.saturating_sub(1) / 2;
                let bi = if bi == 0 { 1 } else { bi };

                // Test this prime
                let ctx = Ntt32Context::new(256, q32);
                let mut data: Vec<u32> = (0..256)
                    .map(|i| ((i as u64 * 7 + 13) % candidate) as u32)
                    .collect();
                let original = data.clone();
                ctx.forward(&mut data);

                let all_reduced = data.iter().all(|&x| x < q32);
                if !all_reduced {
                    eprintln!(
                        "  ❌ FAIL: Barrett boundary q={q32} (interval={bi}) — not fully reduced"
                    );
                    fail += 1;
                } else {
                    pass += 1;
                }

                ctx.inverse(&mut data);
                if data != original {
                    eprintln!(
                        "  ❌ FAIL: Barrett boundary q={q32} (interval={bi}) — roundtrip failed"
                    );
                    fail += 1;
                } else {
                    pass += 1;
                }

                found = true;
                break;
            }
            if found {
                break;
            }
        }
        if !found {
            // Try larger N
            skip += 1;
        }
    }

    // Also test with various N for boundary primes
    for &n in &[64, 128, 512, 1024, 2048, 4096] {
        let primes = generate_primes_28(n, 5);
        for &q in &primes {
            let max_b = ((1u64 << 31) / q as u64) as u32;
            let bi = max_b.saturating_sub(1) / 2;
            let bi = if bi == 0 { 1 } else { bi };

            let ctx = Ntt32Context::new(n, q);
            let mut data: Vec<u32> = (0..n)
                .map(|i| ((i as u64 * 31 + 97) % q as u64) as u32)
                .collect();
            let original = data.clone();
            ctx.forward(&mut data);

            let all_reduced = data.iter().all(|&x| x < q);
            if !all_reduced {
                eprintln!("  ❌ FAIL: N={n} q={q} (bi={bi}) — not reduced");
                fail += 1;
            } else {
                pass += 1;
            }

            ctx.inverse(&mut data);
            if data != original {
                eprintln!("  ❌ FAIL: N={n} q={q} (bi={bi}) — roundtrip");
                fail += 1;
            } else {
                pass += 1;
            }
        }
    }

    println!(
        "  Phase 2: {} pass, {} fail\n",
        pass - phase2_start,
        fail - phase2_fail_start
    );

    // =========================================================================
    // Phase 3: Large N with medium primes (Barrett in early stages)
    // =========================================================================
    println!("── Phase 3: Large N + medium primes (Barrett stress) ─────");

    let phase3_start = pass;
    let phase3_fail_start = fail;

    // These cases stress the periodic Barrett in early stages of the fast path
    // N=4096 with q where barrett_interval ~ 5-10
    for &n in &[1024, 2048, 4096, 8192] {
        let primes = generate_primes_28(n, 3);
        for &q in &primes {
            let max_b = ((1u64 << 31) / q as u64) as u32;
            let bi = max_b.saturating_sub(1) / 2;
            let bi = if bi == 0 { 1 } else { bi };
            let log_n = (n as f64).log2() as u32;
            let early_stages = log_n - 4; // for fast path

            let ctx = Ntt32Context::new(n, q);

            // Max-value stress test
            let mut data = vec![q - 1; n];
            let original = data.clone();
            ctx.forward(&mut data);

            let all_reduced = data.iter().all(|&x| x < q);
            if !all_reduced {
                eprintln!("  ❌ FAIL: N={n} q={q} (bi={bi}, early={early_stages}) — max values not reduced");
                fail += 1;
            } else {
                pass += 1;
            }

            ctx.inverse(&mut data);
            if data != original {
                eprintln!(
                    "  ❌ FAIL: N={n} q={q} (bi={bi}, early={early_stages}) — max values roundtrip"
                );
                fail += 1;
            } else {
                pass += 1;
            }

            // Random-ish stress
            let mut data2: Vec<u32> = (0..n)
                .map(|i| ((i as u64 * 999983 + 1000003) % q as u64) as u32)
                .collect();
            let original2 = data2.clone();
            ctx.forward(&mut data2);
            ctx.inverse(&mut data2);
            if data2 != original2 {
                eprintln!("  ❌ FAIL: N={n} q={q} (bi={bi}) — random roundtrip");
                fail += 1;
            } else {
                pass += 1;
            }
        }
    }

    println!(
        "  Phase 3: {} pass, {} fail\n",
        pass - phase3_start,
        fail - phase3_fail_start
    );

    // =========================================================================
    // Phase 4: Polynomial multiplication correctness
    // =========================================================================
    println!("── Phase 4: Polynomial multiplication ────────────────────");

    let phase4_start = pass;
    let phase4_fail_start = fail;

    for &n in &[8, 16, 64, 256, 1024] {
        let primes = generate_primes_28(n, 1);
        let q = primes[0];
        let ctx = Ntt32Context::new(n, q);

        // (1 + x) × (1 + x) = 1 + 2x + x^2
        let mut a = vec![0u32; n];
        a[0] = 1;
        a[1] = 1;
        let result = ctx.negacyclic_mul(&a, &a);
        if result[0] != 1 || result[1] != 2 || result[2] != 1 {
            eprintln!(
                "  ❌ FAIL: N={n} q={q} — (1+x)^2 wrong: [{}, {}, {}, ...]",
                result[0], result[1], result[2]
            );
            fail += 1;
        } else {
            let rest_zero = result[3..].iter().all(|&x| x == 0);
            if !rest_zero {
                eprintln!("  ❌ FAIL: N={n} q={q} — (1+x)^2 has non-zero tail");
                fail += 1;
            } else {
                pass += 1;
            }
        }

        // (1) × (any) = (any)  — identity
        let _ones = vec![0u32; n];
        let mut id = vec![0u32; n];
        id[0] = 1; // polynomial "1"
        let data: Vec<u32> = (0..n).map(|i| (i as u32 * 7 + 3) % q).collect();
        let result = ctx.negacyclic_mul(&id, &data);
        if result != data {
            eprintln!("  ❌ FAIL: N={n} q={q} — identity multiplication");
            fail += 1;
        } else {
            pass += 1;
        }

        // 0 × anything = 0
        let zero = vec![0u32; n];
        let result = ctx.negacyclic_mul(&zero, &data);
        if !result.iter().all(|&x| x == 0) {
            eprintln!("  ❌ FAIL: N={n} q={q} — zero multiplication");
            fail += 1;
        } else {
            pass += 1;
        }
    }

    println!(
        "  Phase 4: {} pass, {} fail\n",
        pass - phase4_start,
        fail - phase4_fail_start
    );

    // =========================================================================
    // Phase 5: PQ presets
    // =========================================================================
    println!("── Phase 5: Post-quantum presets ─────────────────────────");

    let phase5_start = pass;
    let phase5_fail_start = fail;

    for scheme in [PqScheme::MlDsa44, PqScheme::MlDsa65, PqScheme::MlDsa87] {
        let ntt = PqNtt::new(scheme);

        // Roundtrip
        let mut data: Vec<u32> = (0..256)
            .map(|i| (i * 1000 % ntt.q() as usize) as u32)
            .collect();
        let original = data.clone();
        ntt.forward(&mut data);

        let all_reduced = data.iter().all(|&x| x < ntt.q());
        if !all_reduced {
            eprintln!("  ❌ FAIL: {} — not fully reduced", scheme.name());
            fail += 1;
        } else {
            pass += 1;
        }

        ntt.inverse(&mut data);
        if data != original {
            eprintln!("  ❌ FAIL: {} — roundtrip", scheme.name());
            fail += 1;
        } else {
            pass += 1;
        }

        // Max values
        let mut data2 = vec![ntt.q() - 1; 256];
        let orig2 = data2.clone();
        ntt.forward(&mut data2);
        ntt.inverse(&mut data2);
        if data2 != orig2 {
            eprintln!("  ❌ FAIL: {} — max values roundtrip", scheme.name());
            fail += 1;
        } else {
            pass += 1;
        }

        // Metadata
        assert_eq!(ntt.n(), 256);
        assert_eq!(ntt.q(), 8380417);
        pass += 1;
    }

    println!(
        "  Phase 5: {} pass, {} fail\n",
        pass - phase5_start,
        fail - phase5_fail_start
    );

    // =========================================================================
    // Phase 6: Inverse NTT specific tests
    // =========================================================================
    println!("── Phase 6: Inverse NTT edge cases ───────────────────────");

    let phase6_start = pass;
    let phase6_fail_start = fail;

    for &n in &[8, 16, 32, 64, 128, 256, 512, 1024] {
        let primes = generate_primes_28(n, 2);
        for &q in &primes {
            let ctx = Ntt32Context::new(n, q);

            // Double forward + double inverse should be identity (with N^2 factor, so use roundtrips)
            let mut data: Vec<u32> = (0..n).map(|i| (i as u32 * 37 + 11) % q).collect();
            let original = data.clone();

            ctx.forward(&mut data);
            ctx.inverse(&mut data);

            if data != original {
                eprintln!("  ❌ FAIL: Inverse N={n} q={q} — roundtrip");
                fail += 1;
            } else {
                pass += 1;
            }

            // Forward then check all in [0, q)
            let mut data2: Vec<u32> = (0..n).map(|i| (i as u32) % q).collect();
            ctx.forward(&mut data2);
            let ok = data2.iter().all(|&x| x < q);
            if !ok {
                eprintln!("  ❌ FAIL: Inverse N={n} q={q} — not reduced");
                fail += 1;
            } else {
                pass += 1;
            }
        }
    }

    println!(
        "  Phase 6: {} pass, {} fail\n",
        pass - phase6_start,
        fail - phase6_fail_start
    );

    // =========================================================================
    // Summary
    // =========================================================================
    let total = pass + fail;
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  TOTAL: {pass:>5} pass | {fail:>3} fail | {skip:>4} skip | {total:>5} total   ║");
    if fail == 0 {
        println!("║  ✅  ALL TESTS PASSED                                    ║");
    } else {
        println!("║  ❌  FAILURES DETECTED                                    ║");
    }
    println!("╚══════════════════════════════════════════════════════════╝");

    if fail > 0 {
        std::process::exit(1);
    }
}
