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


//! DudeCT constant-time test for VaeaNTT operations.
//!
//! This example runs the DudeCT statistical test to verify that NTT
//! operations do not leak timing information based on input data.
//!
//! Usage:
//!   cargo run --release --example dudect_ntt
//!   cargo run --release --example dudect_ntt -- --continuous bench_forward
//!
//! Press Ctrl+C to stop the test and see the final results.
//!
//! Interpretation:
//!   - |t| < 4.5 → no detectable timing leak (constant-time)
//!   - |t| > 5.0 → probable timing leak

// dudect-bencher v0.4 macros need clap as a direct dependency (see Cargo.toml)

use dudect_bencher::{ctbench_main_with_seeds, BenchRng, Class, CtRunner};
use rand_core::RngCore;
use std::cell::RefCell;
use vaea_ntt::ntt32::Ntt32Context;

// ============================================================================
// ML-DSA (FIPS 204): q = 8_380_417, N = 256
// ============================================================================
const Q_MLDSA: u32 = 8_380_417;
const N_MLDSA: usize = 256;
const BATCH: usize = 10_000;

/// Benchmark NTT forward: fixed input (class Left) vs random input (class Right).
///
/// If the butterfly operations are truly branchless, the runtime
/// distributions for both classes should be statistically indistinguishable.
fn bench_forward(runner: &mut CtRunner, rng: &mut BenchRng) {
    let ctx = Ntt32Context::new(N_MLDSA, Q_MLDSA);

    let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(BATCH);
    let mut classes: Vec<Class> = Vec::with_capacity(BATCH);

    for _ in 0..BATCH {
        let class = if rng.next_u32() & 1 == 0 {
            Class::Left
        } else {
            Class::Right
        };
        let input = match class {
            Class::Left => vec![0u32; N_MLDSA],
            Class::Right => (0..N_MLDSA).map(|_| rng.next_u32() % Q_MLDSA).collect(),
        };
        inputs.push(input);
        classes.push(class);
    }

    for (input, class) in inputs.into_iter().zip(classes.into_iter()) {
        // run_one requires Fn (not FnMut), so we use RefCell for interior mutability
        let input = RefCell::new(input);
        runner.run_one(class, || {
            ctx.forward(&mut input.borrow_mut());
        });
    }
}

/// Benchmark NTT inverse: fixed vs random input.
fn bench_inverse(runner: &mut CtRunner, rng: &mut BenchRng) {
    let ctx = Ntt32Context::new(N_MLDSA, Q_MLDSA);

    let mut inputs: Vec<Vec<u32>> = Vec::with_capacity(BATCH);
    let mut classes: Vec<Class> = Vec::with_capacity(BATCH);

    for _ in 0..BATCH {
        let class = if rng.next_u32() & 1 == 0 {
            Class::Left
        } else {
            Class::Right
        };
        let input = match class {
            Class::Left => vec![0u32; N_MLDSA],
            Class::Right => (0..N_MLDSA).map(|_| rng.next_u32() % Q_MLDSA).collect(),
        };
        inputs.push(input);
        classes.push(class);
    }

    for (input, class) in inputs.into_iter().zip(classes.into_iter()) {
        let input = RefCell::new(input);
        runner.run_one(class, || {
            ctx.inverse(&mut input.borrow_mut());
        });
    }
}

/// Benchmark negacyclic multiplication: fixed vs random inputs.
fn bench_negacyclic_mul(runner: &mut CtRunner, rng: &mut BenchRng) {
    let ctx = Ntt32Context::new(N_MLDSA, Q_MLDSA);

    let mut inputs: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(BATCH);
    let mut classes: Vec<Class> = Vec::with_capacity(BATCH);

    for _ in 0..BATCH {
        let class = if rng.next_u32() & 1 == 0 {
            Class::Left
        } else {
            Class::Right
        };
        let (a, b) = match class {
            Class::Left => (vec![0u32; N_MLDSA], vec![0u32; N_MLDSA]),
            Class::Right => (
                (0..N_MLDSA).map(|_| rng.next_u32() % Q_MLDSA).collect(),
                (0..N_MLDSA).map(|_| rng.next_u32() % Q_MLDSA).collect(),
            ),
        };
        inputs.push((a, b));
        classes.push(class);
    }

    for ((a, b), class) in inputs.into_iter().zip(classes.into_iter()) {
        let a = RefCell::new(a);
        let b = RefCell::new(b);
        let result = RefCell::new(vec![0u32; N_MLDSA]);
        runner.run_one(class, || {
            ctx.negacyclic_mul_into(
                &mut a.borrow_mut(),
                &mut b.borrow_mut(),
                &mut result.borrow_mut(),
            );
        });
    }
}

ctbench_main_with_seeds!(
    (bench_forward, None),
    (bench_inverse, None),
    (bench_negacyclic_mul, None)
);
