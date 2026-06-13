//! # NEON-Accelerated NTT — All Stages Vectorized
//!
//! Full NEON implementation of the NTT using ARM NEON SIMD intrinsics.
//! ALL stages are vectorized — including t=1 and t=2 which use
//! deinterleaving (`vuzp`/`vzip`) to regroup non-contiguous butterfly elements.
//!
//! The inverse NTT uses a "lazy norm" optimization: the final N^{-1}
//! normalization is fused into the last butterfly stage to save a pass.
//!
//! This module is only compiled on `aarch64` targets.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use super::scalar::compute_shoup;

// ===========================================================================
// NEON helpers
// ===========================================================================

/// Branchless Shoup multiplication on 4 lanes using NEON.
///
/// Computes `v[i] × w[i] mod q` for 4 lanes simultaneously.
/// Uses `vmull`/`vmull_high` for the Shoup quotient estimation,
/// then `vmlsq` (multiply-subtract) for the residue.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn shoup_mul_neon(
    v: uint32x4_t,
    w: uint32x4_t,
    w_shoup: uint32x4_t,
    q: uint32x4_t,
) -> uint32x4_t {
    let prod_lo = vmull_u32(vget_low_u32(v), vget_low_u32(w_shoup));
    let prod_hi = vmull_high_u32(v, w_shoup);
    let q_hat_lo = vshrn_n_u64(prod_lo, 32);
    let q_hat = vshrn_high_n_u64(q_hat_lo, prod_hi, 32);
    let vw = vmulq_u32(v, w);
    let r = vmlsq_u32(vw, q_hat, q);
    // Branchless correction: r -= q if r >= q
    let mask = vcgeq_u32(r, q);
    vsubq_u32(r, vandq_u32(mask, q))
}

/// Branchless modular addition on 4 lanes using NEON.
///
/// Computes `(a + b) mod q` for 4 lanes. Uses `vcgeq` + `vand` for
/// constant-time conditional subtraction.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mod_add_neon(a: uint32x4_t, b: uint32x4_t, q: uint32x4_t) -> uint32x4_t {
    let sum = vaddq_u32(a, b);
    let mask = vcgeq_u32(sum, q);
    vsubq_u32(sum, vandq_u32(mask, q))
}

/// Branchless modular subtraction on 4 lanes using NEON.
///
/// Computes `(a - b) mod q` for 4 lanes. Adds q conditionally when a < b.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn mod_sub_neon(a: uint32x4_t, b: uint32x4_t, q: uint32x4_t) -> uint32x4_t {
    let mask = vcltq_u32(a, b);
    let a_corr = vaddq_u32(a, vandq_u32(mask, q));
    vsubq_u32(a_corr, b)
}

// ===========================================================================
// Forward NTT — Cooley-Tukey DIT, 100% NEON
// ===========================================================================

/// NEON-accelerated NTT forward (Cooley-Tukey DIT).
///
/// All stages are vectorized:
/// - `t >= 4`: standard 4-wide NEON butterfly on contiguous elements
/// - `t == 2`: processes 2 groups of 2 via `vcombine` lane shuffling
/// - `t == 1`: processes 4 groups of 1 via `vuzp`/`vzip` deinterleaving
#[cfg(target_arch = "aarch64")]
pub fn ntt_fwd_neon(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    let n = ctx.n;
    let q = ctx.q;
    assert_eq!(data.len(), n);
    let log_n = ctx.log_n;

    unsafe {
        let q_vec = vdupq_n_u32(q);
        let mut t = n;
        let mut m = 1usize;

        for _ in 0..log_n {
            t >>= 1;

            if t >= 4 {
                // Standard NEON: 4 contiguous butterflies
                let mut k = 0;
                for i in 0..m {
                    let w_vec = vdupq_n_u32(ctx.root_powers[m + i]);
                    let ws_vec = vdupq_n_u32(ctx.root_powers_shoup[m + i]);
                    let mut j = k;
                    while j + 4 <= k + t {
                        let u4 = vld1q_u32(data.as_ptr().add(j));
                        let v4 = vld1q_u32(data.as_ptr().add(j + t));
                        let wv = shoup_mul_neon(v4, w_vec, ws_vec, q_vec);
                        vst1q_u32(data.as_mut_ptr().add(j), mod_add_neon(u4, wv, q_vec));
                        vst1q_u32(data.as_mut_ptr().add(j + t), mod_sub_neon(u4, wv, q_vec));
                        j += 4;
                    }
                    k += 2 * t;
                }
            } else if t == 2 {
                // t=2: process 2 groups of 2 at once via NEON
                let mut k = 0;
                let mut i = 0;
                while i + 2 <= m {
                    let w_vec = vcombine_u32(
                        vdup_n_u32(ctx.root_powers[m + i]),
                        vdup_n_u32(ctx.root_powers[m + i + 1]),
                    );
                    let ws_vec = vcombine_u32(
                        vdup_n_u32(ctx.root_powers_shoup[m + i]),
                        vdup_n_u32(ctx.root_powers_shoup[m + i + 1]),
                    );

                    let raw_lo = vld1q_u32(data.as_ptr().add(k)); // [a0,a1,a2,a3]
                    let raw_hi = vld1q_u32(data.as_ptr().add(k + 4)); // [a4,a5,a6,a7]

                    // u = [a0,a1,a4,a5], v = [a2,a3,a6,a7]
                    let u = vcombine_u32(vget_low_u32(raw_lo), vget_low_u32(raw_hi));
                    let v = vcombine_u32(vget_high_u32(raw_lo), vget_high_u32(raw_hi));

                    let wv = shoup_mul_neon(v, w_vec, ws_vec, q_vec);
                    let res_u = mod_add_neon(u, wv, q_vec);
                    let res_v = mod_sub_neon(u, wv, q_vec);

                    // Reconstruct layout
                    let out_lo = vcombine_u32(vget_low_u32(res_u), vget_low_u32(res_v));
                    let out_hi = vcombine_u32(vget_high_u32(res_u), vget_high_u32(res_v));

                    vst1q_u32(data.as_mut_ptr().add(k), out_lo);
                    vst1q_u32(data.as_mut_ptr().add(k + 4), out_hi);

                    k += 8;
                    i += 2;
                }
            } else {
                // t=1: process 4 groups of 1 via deinterleaving
                let mut k = 0;
                let mut i = 0;
                while i + 4 <= m {
                    let w_vec = vld1q_u32(ctx.root_powers.as_ptr().add(m + i));
                    let ws_vec = vld1q_u32(ctx.root_powers_shoup.as_ptr().add(m + i));

                    let raw_lo = vld1q_u32(data.as_ptr().add(k)); // [a0,a1,a2,a3]
                    let raw_hi = vld1q_u32(data.as_ptr().add(k + 4)); // [a4,a5,a6,a7]

                    // Deinterleave: u = [a0,a2,a4,a6], v = [a1,a3,a5,a7]
                    let u = vuzp1q_u32(raw_lo, raw_hi);
                    let v = vuzp2q_u32(raw_lo, raw_hi);

                    let wv = shoup_mul_neon(v, w_vec, ws_vec, q_vec);
                    let res_u = mod_add_neon(u, wv, q_vec);
                    let res_v = mod_sub_neon(u, wv, q_vec);

                    // Reinterleave
                    let out_lo = vzip1q_u32(res_u, res_v);
                    let out_hi = vzip2q_u32(res_u, res_v);

                    vst1q_u32(data.as_mut_ptr().add(k), out_lo);
                    vst1q_u32(data.as_mut_ptr().add(k + 4), out_hi);

                    k += 8;
                    i += 4;
                }
            }
            m <<= 1;
        }
    }
}

// ===========================================================================
// Inverse NTT — Gentleman-Sande DIF, 100% NEON + Lazy Norm
// ===========================================================================

/// NEON-accelerated NTT inverse (Gentleman-Sande DIF) with lazy normalization.
///
/// All stages are vectorized. The last stage fuses the N^{-1} normalization
/// into the butterfly to avoid an extra pass over the data.
///
/// Stage dispatch:
/// - `t == 1`: deinterleaving via `vuzp`/`vzip`
/// - `t == 2`: lane shuffling via `vcombine`
/// - `t >= 4`: standard 4-wide NEON butterfly (+ lazy norm on last stage)
#[cfg(target_arch = "aarch64")]
pub fn ntt_inv_neon(data: &mut [u32], ctx: &super::context::Ntt32Context) {
    ntt_inv_neon_inner(data, ctx, true);
}

/// NEON-accelerated NTT inverse WITHOUT N^{-1} normalization.
///
/// Identical to [`ntt_inv_neon`] but skips the N^{-1} factor.
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
                    let ws_vec = vld1q_u32(ctx.inv_root_powers_shoup.as_ptr().add(m + i));

                    let raw_lo = vld1q_u32(data.as_ptr().add(k));
                    let raw_hi = vld1q_u32(data.as_ptr().add(k + 4));

                    let u = vuzp1q_u32(raw_lo, raw_hi); // [a0,a2,a4,a6]
                    let v = vuzp2q_u32(raw_lo, raw_hi); // [a1,a3,a5,a7]

                    let sum = mod_add_neon(u, v, q_vec);
                    let dif = mod_sub_neon(u, v, q_vec);
                    let wdif = shoup_mul_neon(dif, w_vec, ws_vec, q_vec);

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
                    let ws_vec = vcombine_u32(
                        vdup_n_u32(ctx.inv_root_powers_shoup[m + i]),
                        vdup_n_u32(ctx.inv_root_powers_shoup[m + i + 1]),
                    );

                    let raw_lo = vld1q_u32(data.as_ptr().add(k));
                    let raw_hi = vld1q_u32(data.as_ptr().add(k + 4));

                    let u = vcombine_u32(vget_low_u32(raw_lo), vget_low_u32(raw_hi));
                    let v = vcombine_u32(vget_high_u32(raw_lo), vget_high_u32(raw_hi));

                    let sum = mod_add_neon(u, v, q_vec);
                    let dif = mod_sub_neon(u, v, q_vec);
                    let wdif = shoup_mul_neon(dif, w_vec, ws_vec, q_vec);

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
                    let (w, w_sh) = (ctx.inv_root_powers[m + i], ctx.inv_root_powers_shoup[m + i]);

                    if is_last {
                        // Lazy norm: fuse N^{-1} into the last stage
                        let w_combined = ((w as u64 * ctx.n_inv as u64) % q as u64) as u32;
                        let w_combined_sh = compute_shoup(w_combined, q);

                        let ni_vec = vdupq_n_u32(ctx.n_inv);
                        let nis_vec = vdupq_n_u32(ctx.n_inv_shoup);
                        let wc_vec = vdupq_n_u32(w_combined);
                        let wcs_vec = vdupq_n_u32(w_combined_sh);

                        let mut j = k;
                        while j + 4 <= k + t {
                            let u4 = vld1q_u32(data.as_ptr().add(j));
                            let v4 = vld1q_u32(data.as_ptr().add(j + t));
                            let sum = mod_add_neon(u4, v4, q_vec);
                            let dif = mod_sub_neon(u4, v4, q_vec);
                            vst1q_u32(
                                data.as_mut_ptr().add(j),
                                shoup_mul_neon(sum, ni_vec, nis_vec, q_vec),
                            );
                            vst1q_u32(
                                data.as_mut_ptr().add(j + t),
                                shoup_mul_neon(dif, wc_vec, wcs_vec, q_vec),
                            );
                            j += 4;
                        }
                    } else {
                        let w_vec = vdupq_n_u32(w);
                        let ws_vec = vdupq_n_u32(w_sh);

                        let mut j = k;
                        while j + 4 <= k + t {
                            let u4 = vld1q_u32(data.as_ptr().add(j));
                            let v4 = vld1q_u32(data.as_ptr().add(j + t));
                            let sum = mod_add_neon(u4, v4, q_vec);
                            let dif = mod_sub_neon(u4, v4, q_vec);
                            let wdif = shoup_mul_neon(dif, w_vec, ws_vec, q_vec);
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
