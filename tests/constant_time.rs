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


//! Constant-time integration tests for VaeaNTT using DudeCT methodology.
//!
//! These tests verify that NTT operations do not leak timing information
//! based on input data. They use a simplified Welch's t-test approach
//! inspired by the DudeCT paper (https://eprint.iacr.org/2016/1123).
//!
//! The tests are marked `#[ignore]` because they take ~30-60s each.
//!
//! Run with:
//!   cargo test --release --test constant_time -- --ignored
//!
//! Or use the helper script:
//!   ./scripts/run_dudect.sh

use rand::Rng;
use std::hint::black_box;
use std::time::Instant;
use vaea_ntt::ntt32::Ntt32Context;

// ============================================================================
// Configuration
// ============================================================================

/// Number of measurements per class per test.
/// ~500K total measurements gives good statistical power.
const NUM_MEASUREMENTS: usize = 500_000;

/// Welch's t-test threshold.
/// |t| < THRESHOLD after NUM_MEASUREMENTS means constant-time.
/// DudeCT paper uses 4.5 as "inconclusive" and 5.0 as "non-constant-time".
const T_THRESHOLD: f64 = 4.5;

// NIST PQ primes
const Q_MLDSA: u32 = 8_380_417; // ML-DSA (FIPS 204)
const N_MLDSA: usize = 256;

// ============================================================================
// Welch's t-test
// ============================================================================

/// Online statistics accumulator (Welch's algorithm).
struct OnlineStats {
    n: u64,
    mean: f64,
    m2: f64, // sum of squares of differences from the current mean
}

impl OnlineStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Add a new observation using Welford's online algorithm.
    #[inline]
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }

    fn variance(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        self.m2 / (self.n - 1) as f64
    }
}

/// Compute Welch's t-statistic between two distributions.
fn welch_t(a: &OnlineStats, b: &OnlineStats) -> f64 {
    if a.n < 2 || b.n < 2 {
        return 0.0;
    }
    let var_a = a.variance();
    let var_b = b.variance();
    let denom = (var_a / a.n as f64) + (var_b / b.n as f64);
    if denom <= 0.0 {
        return 0.0;
    }
    (a.mean - b.mean) / denom.sqrt()
}

/// Core dudect test runner for an arbitrary NTT operation.
///
/// Interleaves measurements of fixed and random inputs, then
/// computes Welch's t-test at multiple percentile crops.
fn dudect_run<F>(name: &str, mut operation: F, num_measurements: usize) -> f64
where
    F: FnMut(&mut [u32]),
{
    let mut rng = rand::thread_rng();

    // Interleave measurements randomly to avoid systematic bias
    let mut all_timings: Vec<(f64, bool)> = Vec::with_capacity(num_measurements);

    let q = Q_MLDSA;
    let n = N_MLDSA;

    // Pre-generate ALL inputs for both classes.
    // Class 0: same data repeated. Class 1: distinct random data.
    // Both are stored in arrays with identical structure to avoid
    // cache artifacts — only the DATA values differ.
    let fixed_data: Vec<u32> = (0..n).map(|_| rng.gen::<u32>() % q).collect();

    let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(num_measurements);
    let mut classes: Vec<bool> = Vec::with_capacity(num_measurements);

    for _ in 0..num_measurements {
        let is_fixed = rng.gen::<bool>();
        classes.push(is_fixed);
        if is_fixed {
            inputs.push(fixed_data.clone());
        } else {
            inputs.push((0..n).map(|_| rng.gen::<u32>() % q).collect());
        }
    }

    // Measurement loop — identical access pattern for both classes
    let mut buf = vec![0u32; n];
    for i in 0..num_measurements {
        buf.copy_from_slice(&inputs[i]);
        black_box(&buf);

        let start = Instant::now();
        operation(&mut buf);
        let elapsed = start.elapsed().as_nanos() as f64;

        black_box(&buf);
        all_timings.push((elapsed, classes[i]));
    }

    // Compute t-test at multiple percentile crops (like DudeCT)
    // Sort timings and crop at different percentiles to handle outliers
    let mut max_t = 0.0_f64;

    // Percentiles to crop at
    let percentiles = [100, 99, 95, 90, 80, 70, 60, 50];

    for &pct in &percentiles {
        // Compute the percentile threshold
        let mut all_times: Vec<f64> = all_timings.iter().map(|(t, _)| *t).collect();
        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = all_times.len() * pct / 100;
        let threshold = if threshold_idx >= all_times.len() {
            f64::MAX
        } else {
            all_times[threshold_idx]
        };

        // Compute t-test for this crop
        let mut stats_fixed = OnlineStats::new();
        let mut stats_random = OnlineStats::new();

        for &(timing, is_fixed) in &all_timings {
            if timing > threshold {
                continue;
            }
            if is_fixed {
                stats_fixed.push(timing);
            } else {
                stats_random.push(timing);
            }
        }

        let t = welch_t(&stats_fixed, &stats_random);
        if t.abs() > max_t.abs() {
            max_t = t;
        }
    }

    let n_total = all_timings.len();
    let n_fixed = all_timings.iter().filter(|(_, f)| *f).count();
    let n_random = n_total - n_fixed;

    eprintln!(
        "  {name}: n_total={n_total}, n_fixed={n_fixed}, n_random={n_random}, max |t| = {:.4}",
        max_t.abs()
    );

    max_t
}

/// Core dudect test runner for negacyclic_mul_into (takes 3 mutable slices).
fn dudect_run_mul(name: &str, ctx: &Ntt32Context, num_measurements: usize) -> f64 {
    let mut rng = rand::thread_rng();
    let n = ctx.n;
    let q = ctx.q;

    let mut all_timings: Vec<(f64, bool)> = Vec::with_capacity(num_measurements);

    // Pre-generate ALL inputs for both classes
    let fixed_a: Vec<u32> = (0..n).map(|_| rng.gen::<u32>() % q).collect();
    let fixed_b: Vec<u32> = (0..n).map(|_| rng.gen::<u32>() % q).collect();

    let mut inputs_a: Vec<Vec<u32>> = Vec::with_capacity(num_measurements);
    let mut inputs_b: Vec<Vec<u32>> = Vec::with_capacity(num_measurements);
    let mut classes: Vec<bool> = Vec::with_capacity(num_measurements);

    for _ in 0..num_measurements {
        let is_fixed = rng.gen::<bool>();
        classes.push(is_fixed);
        if is_fixed {
            inputs_a.push(fixed_a.clone());
            inputs_b.push(fixed_b.clone());
        } else {
            inputs_a.push((0..n).map(|_| rng.gen::<u32>() % q).collect());
            inputs_b.push((0..n).map(|_| rng.gen::<u32>() % q).collect());
        }
    }

    // Measurement loop — identical access pattern for both classes
    let mut a = vec![0u32; n];
    let mut b = vec![0u32; n];
    let mut result = vec![0u32; n];

    for i in 0..num_measurements {
        a.copy_from_slice(&inputs_a[i]);
        b.copy_from_slice(&inputs_b[i]);
        black_box((&a, &b));

        let start = Instant::now();
        ctx.negacyclic_mul_into(&mut a, &mut b, &mut result);
        let elapsed = start.elapsed().as_nanos() as f64;

        black_box(&result);
        all_timings.push((elapsed, classes[i]));
    }

    // Multi-percentile t-test
    let mut max_t = 0.0_f64;
    let percentiles = [100, 99, 95, 90, 80, 70, 60, 50];

    for &pct in &percentiles {
        let mut all_times: Vec<f64> = all_timings.iter().map(|(t, _)| *t).collect();
        all_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let threshold_idx = all_times.len() * pct / 100;
        let threshold = if threshold_idx >= all_times.len() {
            f64::MAX
        } else {
            all_times[threshold_idx]
        };

        let mut stats_fixed = OnlineStats::new();
        let mut stats_random = OnlineStats::new();

        for &(timing, is_fixed) in &all_timings {
            if timing > threshold {
                continue;
            }
            if is_fixed {
                stats_fixed.push(timing);
            } else {
                stats_random.push(timing);
            }
        }

        let t = welch_t(&stats_fixed, &stats_random);
        if t.abs() > max_t.abs() {
            max_t = t;
        }
    }

    let n_total = all_timings.len();
    let n_fixed = all_timings.iter().filter(|(_, f)| *f).count();
    let n_random = n_total - n_fixed;

    eprintln!(
        "  {name}: n_total={n_total}, n_fixed={n_fixed}, n_random={n_random}, max |t| = {:.4}",
        max_t.abs()
    );

    max_t
}

// ============================================================================
// Tests
// ============================================================================

#[test]
#[ignore]
fn test_forward_constant_time_mldsa() {
    eprintln!("\n=== DudeCT: NTT forward (ML-DSA q={Q_MLDSA}, N={N_MLDSA}) ===");
    let ctx = Ntt32Context::new(N_MLDSA, Q_MLDSA);

    let t = dudect_run("forward", |data| ctx.forward(data), NUM_MEASUREMENTS);

    assert!(
        t.abs() < T_THRESHOLD,
        "TIMING LEAK DETECTED in forward()! |t| = {:.4} > {T_THRESHOLD}\n\
         This suggests the NTT forward transform may not be constant-time.",
        t.abs()
    );
    eprintln!(
        "  ✅ forward() is constant-time (|t| = {:.4} < {T_THRESHOLD})",
        t.abs()
    );
}

#[test]
#[ignore]
fn test_inverse_constant_time_mldsa() {
    eprintln!("\n=== DudeCT: NTT inverse (ML-DSA q={Q_MLDSA}, N={N_MLDSA}) ===");
    let ctx = Ntt32Context::new(N_MLDSA, Q_MLDSA);

    let t = dudect_run("inverse", |data| ctx.inverse(data), NUM_MEASUREMENTS);

    assert!(
        t.abs() < T_THRESHOLD,
        "TIMING LEAK DETECTED in inverse()! |t| = {:.4} > {T_THRESHOLD}\n\
         This suggests the NTT inverse transform may not be constant-time.",
        t.abs()
    );
    eprintln!(
        "  ✅ inverse() is constant-time (|t| = {:.4} < {T_THRESHOLD})",
        t.abs()
    );
}

#[test]
#[ignore]
fn test_negacyclic_mul_constant_time_mldsa() {
    eprintln!("\n=== DudeCT: negacyclic_mul_into (ML-DSA q={Q_MLDSA}, N={N_MLDSA}) ===");
    let ctx = Ntt32Context::new(N_MLDSA, Q_MLDSA);

    let t = dudect_run_mul("negacyclic_mul_into", &ctx, NUM_MEASUREMENTS);

    assert!(
        t.abs() < T_THRESHOLD,
        "TIMING LEAK DETECTED in negacyclic_mul_into()! |t| = {:.4} > {T_THRESHOLD}\n\
         This suggests negacyclic multiplication may not be constant-time.",
        t.abs()
    );
    eprintln!(
        "  ✅ negacyclic_mul_into() is constant-time (|t| = {:.4} < {T_THRESHOLD})",
        t.abs()
    );
}
