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

//! # NEON-Accelerated NTT — Optimized Butterfly Pipeline
//!
//! Full NEON implementation of the NTT using ARM NEON SIMD intrinsics.
//! ALL stages are vectorized — including t=1 and t=2 which use
//! deinterleaving (`vuzp`/`vzip`) to regroup non-contiguous butterfly elements.
//!
//! ## Forward NTT — Deferred-Reduction Butterfly
//!
//! The forward NTT uses a deferred-reduction butterfly that eliminates all
//! intermediate modular corrections from the hot loop. Coefficients grow by
//! at most `+2q` per stage. An **adaptive Barrett interval** (computed as
//! `(2^31/q - 1) / 2`) determines how many stages can run without reduction.
//! For small primes (q < 2^23), **no intermediate Barrett** is needed.
//! For large primes (q ≈ 2^28), Barrett is applied every 3 stages.
//! A final Barrett pass normalizes all values to `[0, q)`.
//!
//! The last 4 stages (or 3 for large primes) are **fused** into a single
//! memory pass over 16-element (or 8-element) blocks, eliminating redundant
//! loads/stores.
//!
//! The butterfly uses `vqdmulhq_s32` for single-instruction quotient
//! estimation, reducing the core butterfly to **6 NEON arithmetic ops**
//! with zero intermediate reductions per butterfly.
//!
//! ## Inverse NTT — Fast Shoup Multiplication
//!
//! The inverse NTT uses standard modular add/sub with `vqdmulhq_s32`-based
//! Shoup multiplication for the twiddle factor, saving 4 instructions per
//! multiply compared to the traditional `vmull`-based approach. The final
//! N⁻¹ normalization is fused into the last butterfly stage.
//!
//! This module is only compiled on `aarch64` targets.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ===========================================================================
// Barrett normalization — single-pass reduction from [0, Bq) to [0, q)
// ===========================================================================

/// Barrett normalization: reduces values from `[0, max_val)` to `[0, q)` in
/// a single pass using a precomputed Barrett constant.
///
/// The Barrett constant `bc = floor(2^32 / q)` enables computing `v mod q`
/// via a single multiply-high + correction, avoiding the multi-step binary
/// search used by naive approaches.
///
/// This is constant-time: no data-dependent branches.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn barrett_reduce(v: uint32x4_t, q_vec: uint32x4_t, bc: uint32x4_t) -> uint32x4_t {
    // q_hat ≈ v / q via multiply-high
    let prod_lo = vmull_u32(vget_low_u32(v), vget_low_u32(bc));
    let prod_hi = vmull_high_u32(v, bc);
    let q_hat_lo = vshrn_n_u64(prod_lo, 32);
    let q_hat = vshrn_high_n_u64(q_hat_lo, prod_hi, 32);
    // r = v - q_hat * q  (may be in [0, 2q) due to floor rounding)
    let r = vmlsq_u32(v, q_hat, q_vec);
    // Single correction: if r >= q then r -= q
    let mask = vcgeq_u32(r, q_vec);
    vsubq_u32(r, vandq_u32(mask, q_vec))
}

// ===========================================================================
// Forward NTT — Cooley-Tukey DIT, 100% NEON
// ===========================================================================

/// NEON-accelerated NTT forward (Cooley-Tukey DIT) with fused final stages.
///
/// Architecture:
/// - **Phase 1** (early stages, t≥8): standard per-stage butterfly loop with
///   periodic Barrett reduction every 3 stages and 2× unrolled inner loop.
/// - **Phase 2** (last 3 stages, t=4,2,1): fused into a single memory pass
///   over 8-element blocks. Each block is loaded once, 3 stages of butterflies
///   are applied in-register, and the result is stored once. This eliminates
///   2 memory passes over the entire array.
///
/// A final Barrett pass normalizes all coefficients to `[0, q)`.
#[cfg(target_arch = "aarch64")]
pub fn ntt_fwd_neon(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    let n = ctx.n;
    let q = ctx.q;
    assert_eq!(data.len(), n);
    let log_n = ctx.log_n as usize;

    // Security: verify inputs are in [0, q) — active in debug builds only
    debug_assert!(
        data.iter().all(|&x| x < q),
        "NTT forward: input coefficients must be in [0, q). Found value >= q={}",
        q
    );

    // Small NTTs (N < 8): fall back to scalar — NEON needs ≥ 8 elements
    if n < 8 {
        super::scalar::ntt_forward_scalar(data, ctx);
        return;
    }

    unsafe {
        let q_vec = vdupq_n_u32(q);
        let two_q = vdupq_n_u32(2 * q);
        let bc = vdupq_n_u32(((1u64 << 32) / q as u64) as u32);

        // Adaptive Barrett interval: compute max stages without reduction.
        // After k butterfly stages, values grow to (2k+1)*q.
        // For vqdmulhq_s32 (signed interpretation), values must be < 2^31.
        // So: (2k+1)*q < 2^31  →  k < (2^31/q - 1) / 2
        let barrett_interval = {
            let max_b = ((1u64 << 31) / q as u64) as u32;
            let k = max_b.saturating_sub(1) / 2;
            if k == 0 {
                1
            } else {
                k
            }
        };

        if log_n >= 4 && n >= 16 && barrett_interval >= 4 {
            // =============================================================
            // FAST PATH: 4-stage fusion for small/medium primes
            // Requires 4 deferred-reduction stages to be safe.
            // =============================================================
            let early_stages = log_n - 4;
            let mut t = n;
            let mut m = 1usize;
            let mut stages_since_reduce = 0u32;

            for _ in 0..early_stages {
                t >>= 1;

                // Periodic Barrett reduction (adaptive interval)
                if stages_since_reduce >= barrett_interval {
                    for j in (0..n).step_by(4) {
                        let v = vld1q_u32(data.as_ptr().add(j));
                        vst1q_u32(data.as_mut_ptr().add(j), barrett_reduce(v, q_vec, bc));
                    }
                    stages_since_reduce = 0;
                }

                let mut k = 0;
                for i in 0..m {
                    let w_vec = vdupq_n_u32(ctx.root_powers[m + i]);
                    let wq_vec = vdupq_n_s32(ctx.root_powers_qmulh[m + i]);
                    let mut j = k;

                    // 2× unrolled butterfly loop for ILP
                    while j + 8 <= k + t {
                        let u0 = vld1q_u32(data.as_ptr().add(j));
                        let v0 = vld1q_u32(data.as_ptr().add(j + t));
                        let u1 = vld1q_u32(data.as_ptr().add(j + 4));
                        let v1 = vld1q_u32(data.as_ptr().add(j + 4 + t));

                        let qh0 = vqdmulhq_s32(vreinterpretq_s32_u32(v0), wq_vec);
                        let vw0 = vmulq_u32(v0, w_vec);
                        let qh1 = vqdmulhq_s32(vreinterpretq_s32_u32(v1), wq_vec);
                        let vw1 = vmulq_u32(v1, w_vec);

                        let wv0 = vmlsq_u32(vw0, vreinterpretq_u32_s32(qh0), q_vec);
                        let wv1 = vmlsq_u32(vw1, vreinterpretq_u32_s32(qh1), q_vec);

                        vst1q_u32(data.as_mut_ptr().add(j), vaddq_u32(u0, wv0));
                        vst1q_u32(
                            data.as_mut_ptr().add(j + t),
                            vsubq_u32(vaddq_u32(u0, two_q), wv0),
                        );
                        vst1q_u32(data.as_mut_ptr().add(j + 4), vaddq_u32(u1, wv1));
                        vst1q_u32(
                            data.as_mut_ptr().add(j + 4 + t),
                            vsubq_u32(vaddq_u32(u1, two_q), wv1),
                        );
                        j += 8;
                    }
                    // Handle remaining 4-element block
                    if j + 4 <= k + t {
                        let u4 = vld1q_u32(data.as_ptr().add(j));
                        let v4 = vld1q_u32(data.as_ptr().add(j + t));
                        let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v4), wq_vec);
                        let vw = vmulq_u32(v4, w_vec);
                        let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                        vst1q_u32(data.as_mut_ptr().add(j), vaddq_u32(u4, wv));
                        vst1q_u32(
                            data.as_mut_ptr().add(j + t),
                            vsubq_u32(vaddq_u32(u4, two_q), wv),
                        );
                    }
                    k += 2 * t;
                }
                m <<= 1;
                stages_since_reduce += 1;
            }

            // =============================================================
            // Phase 2: Fused last 4 stages (t=8, t=4, t=2, t=1)
            // =============================================================
            // Barrett before fused stages if needed
            if stages_since_reduce >= barrett_interval {
                for j in (0..n).step_by(4) {
                    let v = vld1q_u32(data.as_ptr().add(j));
                    vst1q_u32(data.as_mut_ptr().add(j), barrett_reduce(v, q_vec, bc));
                }
            }

            // At this point: m = n/16
            let m5 = m; // twiddle offset for stage with t=8
            let m6 = m * 2; // twiddle offset for stage with t=4
            let m7 = m * 4; // twiddle offset for stage with t=2
            let m8 = m * 8; // twiddle offset for stage with t=1

            for block in 0..(n / 16) {
                let k = block * 16;

                // Load 16 elements into 4 NEON registers
                let mut r0 = vld1q_u32(data.as_ptr().add(k));
                let mut r1 = vld1q_u32(data.as_ptr().add(k + 4));
                let mut r2 = vld1q_u32(data.as_ptr().add(k + 8));
                let mut r3 = vld1q_u32(data.as_ptr().add(k + 12));

                // --- Stage t=8: butterfly(r0,r2) and (r1,r3) ---
                // 1 twiddle per block
                {
                    let w = vdupq_n_u32(ctx.root_powers[m5 + block]);
                    let wq = vdupq_n_s32(ctx.root_powers_qmulh[m5 + block]);

                    let q_hat0 = vqdmulhq_s32(vreinterpretq_s32_u32(r2), wq);
                    let vw0 = vmulq_u32(r2, w);
                    let wv0 = vmlsq_u32(vw0, vreinterpretq_u32_s32(q_hat0), q_vec);

                    let q_hat1 = vqdmulhq_s32(vreinterpretq_s32_u32(r3), wq);
                    let vw1 = vmulq_u32(r3, w);
                    let wv1 = vmlsq_u32(vw1, vreinterpretq_u32_s32(q_hat1), q_vec);

                    let new_r0 = vaddq_u32(r0, wv0);
                    let new_r2 = vsubq_u32(vaddq_u32(r0, two_q), wv0);
                    let new_r1 = vaddq_u32(r1, wv1);
                    let new_r3 = vsubq_u32(vaddq_u32(r1, two_q), wv1);

                    r0 = new_r0;
                    r1 = new_r1;
                    r2 = new_r2;
                    r3 = new_r3;
                }

                // --- Stage t=4: butterfly(r0,r1) and (r2,r3) ---
                // 2 twiddles per block
                {
                    let wa = vdupq_n_u32(ctx.root_powers[m6 + block * 2]);
                    let wqa = vdupq_n_s32(ctx.root_powers_qmulh[m6 + block * 2]);
                    let wb = vdupq_n_u32(ctx.root_powers[m6 + block * 2 + 1]);
                    let wqb = vdupq_n_s32(ctx.root_powers_qmulh[m6 + block * 2 + 1]);

                    let q_hat0 = vqdmulhq_s32(vreinterpretq_s32_u32(r1), wqa);
                    let vw0 = vmulq_u32(r1, wa);
                    let wv0 = vmlsq_u32(vw0, vreinterpretq_u32_s32(q_hat0), q_vec);

                    let q_hat1 = vqdmulhq_s32(vreinterpretq_s32_u32(r3), wqb);
                    let vw1 = vmulq_u32(r3, wb);
                    let wv1 = vmlsq_u32(vw1, vreinterpretq_u32_s32(q_hat1), q_vec);

                    let new_r0 = vaddq_u32(r0, wv0);
                    let new_r1 = vsubq_u32(vaddq_u32(r0, two_q), wv0);
                    let new_r2 = vaddq_u32(r2, wv1);
                    let new_r3 = vsubq_u32(vaddq_u32(r2, two_q), wv1);

                    r0 = new_r0;
                    r1 = new_r1;
                    r2 = new_r2;
                    r3 = new_r3;
                }

                // --- Stage t=2: within-register shuffle ---
                // 4 twiddles, packed in pairs
                {
                    let base = block * 4;
                    // Process (r0,r1) and (r2,r3) as register pairs
                    macro_rules! fuse_t2 {
                        ($rA:ident, $rB:ident, $idx:expr) => {
                            let w = vcombine_u32(
                                vdup_n_u32(ctx.root_powers[m7 + base + $idx]),
                                vdup_n_u32(ctx.root_powers[m7 + base + $idx + 1]),
                            );
                            let wq = vcombine_s32(
                                vdup_n_s32(ctx.root_powers_qmulh[m7 + base + $idx]),
                                vdup_n_s32(ctx.root_powers_qmulh[m7 + base + $idx + 1]),
                            );
                            let u = vcombine_u32(vget_low_u32($rA), vget_low_u32($rB));
                            let v = vcombine_u32(vget_high_u32($rA), vget_high_u32($rB));

                            let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v), wq);
                            let vw = vmulq_u32(v, w);
                            let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                            let ru = vaddq_u32(u, wv);
                            let rv = vsubq_u32(vaddq_u32(u, two_q), wv);

                            $rA = vcombine_u32(vget_low_u32(ru), vget_low_u32(rv));
                            $rB = vcombine_u32(vget_high_u32(ru), vget_high_u32(rv));
                        };
                    }
                    fuse_t2!(r0, r1, 0);
                    fuse_t2!(r2, r3, 2);
                }

                // --- Stage t=1: deinterleave even/odd ---
                // 8 twiddles, loaded as 2 vector loads
                {
                    let base = block * 8;
                    macro_rules! fuse_t1 {
                        ($rA:ident, $rB:ident, $idx:expr) => {
                            let w = vld1q_u32(ctx.root_powers.as_ptr().add(m8 + base + $idx));
                            let wq =
                                vld1q_s32(ctx.root_powers_qmulh.as_ptr().add(m8 + base + $idx));

                            let u = vuzp1q_u32($rA, $rB);
                            let v = vuzp2q_u32($rA, $rB);

                            let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v), wq);
                            let vw = vmulq_u32(v, w);
                            let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                            let ru = vaddq_u32(u, wv);
                            let rv = vsubq_u32(vaddq_u32(u, two_q), wv);

                            $rA = vzip1q_u32(ru, rv);
                            $rB = vzip2q_u32(ru, rv);
                        };
                    }
                    fuse_t1!(r0, r1, 0);
                    fuse_t1!(r2, r3, 4);
                }

                // Store 16 elements (single write for 4 stages of work)
                vst1q_u32(data.as_mut_ptr().add(k), r0);
                vst1q_u32(data.as_mut_ptr().add(k + 4), r1);
                vst1q_u32(data.as_mut_ptr().add(k + 8), r2);
                vst1q_u32(data.as_mut_ptr().add(k + 12), r3);
            }
        } else if log_n >= 3 && n >= 8 {
            // =============================================================
            // FALLBACK: 3-stage fusion for large primes (q ~ 2^28)
            // =============================================================
            let early_stages = log_n - 3;
            let mut t = n;
            let mut m = 1usize;
            let mut stages_since_reduce = 0u32;

            for _ in 0..early_stages {
                t >>= 1;
                if stages_since_reduce >= barrett_interval {
                    for j in (0..n).step_by(4) {
                        let v = vld1q_u32(data.as_ptr().add(j));
                        vst1q_u32(data.as_mut_ptr().add(j), barrett_reduce(v, q_vec, bc));
                    }
                    stages_since_reduce = 0;
                }
                let mut k = 0;
                for i in 0..m {
                    let w_vec = vdupq_n_u32(ctx.root_powers[m + i]);
                    let wq_vec = vdupq_n_s32(ctx.root_powers_qmulh[m + i]);
                    let mut j = k;
                    while j + 8 <= k + t {
                        let u0 = vld1q_u32(data.as_ptr().add(j));
                        let v0 = vld1q_u32(data.as_ptr().add(j + t));
                        let u1 = vld1q_u32(data.as_ptr().add(j + 4));
                        let v1 = vld1q_u32(data.as_ptr().add(j + 4 + t));
                        let qh0 = vqdmulhq_s32(vreinterpretq_s32_u32(v0), wq_vec);
                        let vw0 = vmulq_u32(v0, w_vec);
                        let qh1 = vqdmulhq_s32(vreinterpretq_s32_u32(v1), wq_vec);
                        let vw1 = vmulq_u32(v1, w_vec);
                        let wv0 = vmlsq_u32(vw0, vreinterpretq_u32_s32(qh0), q_vec);
                        let wv1 = vmlsq_u32(vw1, vreinterpretq_u32_s32(qh1), q_vec);
                        vst1q_u32(data.as_mut_ptr().add(j), vaddq_u32(u0, wv0));
                        vst1q_u32(
                            data.as_mut_ptr().add(j + t),
                            vsubq_u32(vaddq_u32(u0, two_q), wv0),
                        );
                        vst1q_u32(data.as_mut_ptr().add(j + 4), vaddq_u32(u1, wv1));
                        vst1q_u32(
                            data.as_mut_ptr().add(j + 4 + t),
                            vsubq_u32(vaddq_u32(u1, two_q), wv1),
                        );
                        j += 8;
                    }
                    if j + 4 <= k + t {
                        let u4 = vld1q_u32(data.as_ptr().add(j));
                        let v4 = vld1q_u32(data.as_ptr().add(j + t));
                        let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v4), wq_vec);
                        let vw = vmulq_u32(v4, w_vec);
                        let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                        vst1q_u32(data.as_mut_ptr().add(j), vaddq_u32(u4, wv));
                        vst1q_u32(
                            data.as_mut_ptr().add(j + t),
                            vsubq_u32(vaddq_u32(u4, two_q), wv),
                        );
                    }
                    k += 2 * t;
                }
                m <<= 1;
                stages_since_reduce += 1;
            }

            // Fused last 3 stages
            if stages_since_reduce >= barrett_interval {
                for j in (0..n).step_by(4) {
                    let v = vld1q_u32(data.as_ptr().add(j));
                    vst1q_u32(data.as_mut_ptr().add(j), barrett_reduce(v, q_vec, bc));
                }
            }
            let m6 = m;
            let m7 = m * 2;
            let m8 = m * 4;
            for block in 0..(n / 8) {
                let k = block * 8;
                let mut lo = vld1q_u32(data.as_ptr().add(k));
                let mut hi = vld1q_u32(data.as_ptr().add(k + 4));
                {
                    let w = vdupq_n_u32(ctx.root_powers[m6 + block]);
                    let wq = vdupq_n_s32(ctx.root_powers_qmulh[m6 + block]);
                    let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(hi), wq);
                    let vw = vmulq_u32(hi, w);
                    let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                    let tl = vaddq_u32(lo, wv);
                    let th = vsubq_u32(vaddq_u32(lo, two_q), wv);
                    lo = tl;
                    hi = th;
                }
                {
                    let w = vcombine_u32(
                        vdup_n_u32(ctx.root_powers[m7 + block * 2]),
                        vdup_n_u32(ctx.root_powers[m7 + block * 2 + 1]),
                    );
                    let wq = vcombine_s32(
                        vdup_n_s32(ctx.root_powers_qmulh[m7 + block * 2]),
                        vdup_n_s32(ctx.root_powers_qmulh[m7 + block * 2 + 1]),
                    );
                    let u = vcombine_u32(vget_low_u32(lo), vget_low_u32(hi));
                    let v = vcombine_u32(vget_high_u32(lo), vget_high_u32(hi));
                    let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v), wq);
                    let vw = vmulq_u32(v, w);
                    let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                    let ru = vaddq_u32(u, wv);
                    let rv = vsubq_u32(vaddq_u32(u, two_q), wv);
                    lo = vcombine_u32(vget_low_u32(ru), vget_low_u32(rv));
                    hi = vcombine_u32(vget_high_u32(ru), vget_high_u32(rv));
                }
                {
                    let w = vld1q_u32(ctx.root_powers.as_ptr().add(m8 + block * 4));
                    let wq = vld1q_s32(ctx.root_powers_qmulh.as_ptr().add(m8 + block * 4));
                    let u = vuzp1q_u32(lo, hi);
                    let v = vuzp2q_u32(lo, hi);
                    let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v), wq);
                    let vw = vmulq_u32(v, w);
                    let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                    let ru = vaddq_u32(u, wv);
                    let rv = vsubq_u32(vaddq_u32(u, two_q), wv);
                    lo = vzip1q_u32(ru, rv);
                    hi = vzip2q_u32(ru, rv);
                }
                vst1q_u32(data.as_mut_ptr().add(k), lo);
                vst1q_u32(data.as_mut_ptr().add(k + 4), hi);
            }
        } else {
            // Small NTTs (n <= 4): generic per-stage loop
            let mut t = n;
            let mut m = 1usize;
            let mut stages_since_reduce = 0u32;

            for _ in 0..log_n {
                t >>= 1;

                if stages_since_reduce >= barrett_interval {
                    for j in (0..n).step_by(4) {
                        let v = vld1q_u32(data.as_ptr().add(j));
                        vst1q_u32(data.as_mut_ptr().add(j), barrett_reduce(v, q_vec, bc));
                    }
                    stages_since_reduce = 0;
                }

                if t >= 4 {
                    let mut k = 0;
                    for i in 0..m {
                        let w_vec = vdupq_n_u32(ctx.root_powers[m + i]);
                        let wq_vec = vdupq_n_s32(ctx.root_powers_qmulh[m + i]);
                        let mut j = k;
                        while j + 4 <= k + t {
                            let u4 = vld1q_u32(data.as_ptr().add(j));
                            let v4 = vld1q_u32(data.as_ptr().add(j + t));
                            let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v4), wq_vec);
                            let vw = vmulq_u32(v4, w_vec);
                            let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                            vst1q_u32(data.as_mut_ptr().add(j), vaddq_u32(u4, wv));
                            vst1q_u32(
                                data.as_mut_ptr().add(j + t),
                                vsubq_u32(vaddq_u32(u4, two_q), wv),
                            );
                            j += 4;
                        }
                        k += 2 * t;
                    }
                } else if t == 2 {
                    let mut k = 0;
                    let mut i = 0;
                    while i + 2 <= m {
                        let w_vec = vcombine_u32(
                            vdup_n_u32(ctx.root_powers[m + i]),
                            vdup_n_u32(ctx.root_powers[m + i + 1]),
                        );
                        let wq_vec = vcombine_s32(
                            vdup_n_s32(ctx.root_powers_qmulh[m + i]),
                            vdup_n_s32(ctx.root_powers_qmulh[m + i + 1]),
                        );
                        let raw_lo = vld1q_u32(data.as_ptr().add(k));
                        let raw_hi = vld1q_u32(data.as_ptr().add(k + 4));
                        let u = vcombine_u32(vget_low_u32(raw_lo), vget_low_u32(raw_hi));
                        let v = vcombine_u32(vget_high_u32(raw_lo), vget_high_u32(raw_hi));
                        let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v), wq_vec);
                        let vw = vmulq_u32(v, w_vec);
                        let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                        let ru = vaddq_u32(u, wv);
                        let rv = vsubq_u32(vaddq_u32(u, two_q), wv);
                        vst1q_u32(
                            data.as_mut_ptr().add(k),
                            vcombine_u32(vget_low_u32(ru), vget_low_u32(rv)),
                        );
                        vst1q_u32(
                            data.as_mut_ptr().add(k + 4),
                            vcombine_u32(vget_high_u32(ru), vget_high_u32(rv)),
                        );
                        k += 8;
                        i += 2;
                    }
                } else {
                    let mut k = 0;
                    let mut i = 0;
                    while i + 4 <= m {
                        let w_vec = vld1q_u32(ctx.root_powers.as_ptr().add(m + i));
                        let wq_vec = vld1q_s32(ctx.root_powers_qmulh.as_ptr().add(m + i));
                        let raw_lo = vld1q_u32(data.as_ptr().add(k));
                        let raw_hi = vld1q_u32(data.as_ptr().add(k + 4));
                        let u = vuzp1q_u32(raw_lo, raw_hi);
                        let v = vuzp2q_u32(raw_lo, raw_hi);
                        let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v), wq_vec);
                        let vw = vmulq_u32(v, w_vec);
                        let wv = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q_vec);
                        let ru = vaddq_u32(u, wv);
                        let rv = vsubq_u32(vaddq_u32(u, two_q), wv);
                        vst1q_u32(data.as_mut_ptr().add(k), vzip1q_u32(ru, rv));
                        vst1q_u32(data.as_mut_ptr().add(k + 4), vzip2q_u32(ru, rv));
                        k += 8;
                        i += 4;
                    }
                }
                m <<= 1;
                stages_since_reduce += 1;
            }
        }

        // Final Barrett normalization: reduce all to [0, q)
        for j in (0..n).step_by(4) {
            let v = vld1q_u32(data.as_ptr().add(j));
            vst1q_u32(data.as_mut_ptr().add(j), barrett_reduce(v, q_vec, bc));
        }
    }
}

// ===========================================================================
// Inverse NTT helpers
// ===========================================================================

/// Branchless modular addition on 4 lanes using NEON.
/// Inputs must be in `[0, q)`. Output in `[0, q)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mod_add_neon(a: uint32x4_t, b: uint32x4_t, q: uint32x4_t) -> uint32x4_t {
    let sum = vaddq_u32(a, b);
    let mask = vcgeq_u32(sum, q);
    vsubq_u32(sum, vandq_u32(mask, q))
}

/// Branchless modular subtraction on 4 lanes using NEON.
/// Inputs must be in `[0, q)`. Output in `[0, q)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mod_sub_neon(a: uint32x4_t, b: uint32x4_t, q: uint32x4_t) -> uint32x4_t {
    let mask = vcltq_u32(a, b);
    let a_corr = vaddq_u32(a, vandq_u32(mask, q));
    vsubq_u32(a_corr, b)
}

/// Shoup multiplication using vqdmulhq for quotient estimation (no correction needed).
/// The input `v` must be in `[0, q)` for correct Shoup quotient.
/// Output is in `[0, 2q)`, then corrected to `[0, q)`.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shoup_mul_fast(v: uint32x4_t, w: uint32x4_t, wq: int32x4_t, q: uint32x4_t) -> uint32x4_t {
    let q_hat = vqdmulhq_s32(vreinterpretq_s32_u32(v), wq);
    let vw = vmulq_u32(v, w);
    let r = vmlsq_u32(vw, vreinterpretq_u32_s32(q_hat), q);
    let mask = vcgeq_u32(r, q);
    vsubq_u32(r, vandq_u32(mask, q))
}

// ===========================================================================
// Inverse NTT — Gentleman-Sande DIF, 100% NEON
// ===========================================================================

/// NEON-accelerated NTT inverse (Gentleman-Sande DIF) with normalization.
///
/// All stages are vectorized. Uses `vqdmulhq` for fast twiddle multiplication.
/// The final N⁻¹ normalization is fused into the last butterfly stage.
#[cfg(target_arch = "aarch64")]
pub fn ntt_inv_neon(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    ntt_inv_neon_inner(data, ctx, true);
}

/// NEON-accelerated NTT inverse WITHOUT N⁻¹ normalization.
///
/// Identical to [`ntt_inv_neon`] but skips the N⁻¹ factor.
/// Use when normalization is handled externally or not needed.
#[cfg(target_arch = "aarch64")]
pub fn ntt_inv_neon_lazy(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    ntt_inv_neon_inner(data, ctx, false);
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn ntt_inv_neon_inner(data: &mut [u32], ctx: &super::context::Ntt32Context, normalize: bool) {
    let n = ctx.n;
    let q = ctx.q;
    assert_eq!(data.len(), n);
    let log_n = ctx.log_n;

    // Small NTTs (N < 8): fall back to scalar
    if n < 8 {
        if normalize {
            super::scalar::ntt_inverse_scalar(data, ctx);
        } else {
            super::scalar::ntt_inverse_scalar_lazy(data, ctx);
        }
        return;
    }

    unsafe {
        let q_vec = vdupq_n_u32(q);
        let mut t = 1usize;
        let mut m = n >> 1;

        for stage in 0..log_n {
            let is_last = normalize && stage == log_n - 1;

            if t == 1 {
                // Deinterleaving NEON for t=1
                let mut k = 0;
                let mut i = 0;
                while i + 4 <= m {
                    let w_vec = vld1q_u32(ctx.inv_root_powers.as_ptr().add(m + i));
                    let wq_vec = vld1q_s32(ctx.inv_root_powers_qmulh.as_ptr().add(m + i));

                    let raw_lo = vld1q_u32(data.as_ptr().add(k));
                    let raw_hi = vld1q_u32(data.as_ptr().add(k + 4));

                    let u = vuzp1q_u32(raw_lo, raw_hi);
                    let v = vuzp2q_u32(raw_lo, raw_hi);

                    let sum = mod_add_neon(u, v, q_vec);
                    let dif = mod_sub_neon(u, v, q_vec);
                    let wdif = shoup_mul_fast(dif, w_vec, wq_vec, q_vec);

                    let out_lo = vzip1q_u32(sum, wdif);
                    let out_hi = vzip2q_u32(sum, wdif);

                    vst1q_u32(data.as_mut_ptr().add(k), out_lo);
                    vst1q_u32(data.as_mut_ptr().add(k + 4), out_hi);

                    k += 8;
                    i += 4;
                }
            } else if t == 2 {
                // 2 groups at once
                let mut k = 0;
                let mut i = 0;
                while i + 2 <= m {
                    let w_vec = vcombine_u32(
                        vdup_n_u32(ctx.inv_root_powers[m + i]),
                        vdup_n_u32(ctx.inv_root_powers[m + i + 1]),
                    );
                    let wq_vec = vcombine_s32(
                        vdup_n_s32(ctx.inv_root_powers_qmulh[m + i]),
                        vdup_n_s32(ctx.inv_root_powers_qmulh[m + i + 1]),
                    );

                    let raw_lo = vld1q_u32(data.as_ptr().add(k));
                    let raw_hi = vld1q_u32(data.as_ptr().add(k + 4));

                    let u = vcombine_u32(vget_low_u32(raw_lo), vget_low_u32(raw_hi));
                    let v = vcombine_u32(vget_high_u32(raw_lo), vget_high_u32(raw_hi));

                    let sum = mod_add_neon(u, v, q_vec);
                    let dif = mod_sub_neon(u, v, q_vec);
                    let wdif = shoup_mul_fast(dif, w_vec, wq_vec, q_vec);

                    let out_lo = vcombine_u32(vget_low_u32(sum), vget_low_u32(wdif));
                    let out_hi = vcombine_u32(vget_high_u32(sum), vget_high_u32(wdif));

                    vst1q_u32(data.as_mut_ptr().add(k), out_lo);
                    vst1q_u32(data.as_mut_ptr().add(k + 4), out_hi);

                    k += 8;
                    i += 2;
                }
            } else {
                // t >= 4: standard NEON
                let mut k = 0;
                for i in 0..m {
                    let (w, _w_sh) = (ctx.inv_root_powers[m + i], ctx.inv_root_powers_shoup[m + i]);

                    if is_last {
                        // Lazy norm: fuse N⁻¹ into the last stage
                        let w_combined = ((w as u64 * ctx.n_inv as u64) % q as u64) as u32;
                        let w_combined_qmulh =
                            ((w_combined as u64 * (1u64 << 31)) / q as u64) as i32;

                        let ni_vec = vdupq_n_u32(ctx.n_inv);
                        let niq_vec =
                            vdupq_n_s32(((ctx.n_inv as u64 * (1u64 << 31)) / q as u64) as i32);
                        let wc_vec = vdupq_n_u32(w_combined);
                        let wcq_vec = vdupq_n_s32(w_combined_qmulh);

                        let mut j = k;
                        while j + 4 <= k + t {
                            let u4 = vld1q_u32(data.as_ptr().add(j));
                            let v4 = vld1q_u32(data.as_ptr().add(j + t));
                            let sum = mod_add_neon(u4, v4, q_vec);
                            let dif = mod_sub_neon(u4, v4, q_vec);
                            vst1q_u32(
                                data.as_mut_ptr().add(j),
                                shoup_mul_fast(sum, ni_vec, niq_vec, q_vec),
                            );
                            vst1q_u32(
                                data.as_mut_ptr().add(j + t),
                                shoup_mul_fast(dif, wc_vec, wcq_vec, q_vec),
                            );
                            j += 4;
                        }
                    } else {
                        let w_vec = vdupq_n_u32(w);
                        let wq_vec = vdupq_n_s32(ctx.inv_root_powers_qmulh[m + i]);

                        let mut j = k;
                        while j + 4 <= k + t {
                            let u4 = vld1q_u32(data.as_ptr().add(j));
                            let v4 = vld1q_u32(data.as_ptr().add(j + t));
                            let sum = mod_add_neon(u4, v4, q_vec);
                            let dif = mod_sub_neon(u4, v4, q_vec);
                            let wdif = shoup_mul_fast(dif, w_vec, wq_vec, q_vec);
                            vst1q_u32(data.as_mut_ptr().add(j), sum);
                            vst1q_u32(data.as_mut_ptr().add(j + t), wdif);
                            j += 4;
                        }
                    }
                    k += 2 * t;
                }
            }
            t <<= 1;
            m >>= 1;
        }
    }
}
