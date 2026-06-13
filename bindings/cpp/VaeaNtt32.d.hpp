#ifndef VaeaNtt32_D_HPP
#define VaeaNtt32_D_HPP

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <memory>
#include <functional>
#include <optional>
#include <cstdlib>
#include "diplomat_runtime.hpp"

class VaeaNttError;




namespace diplomat {
namespace capi {
    struct VaeaNtt32;
} // namespace capi
} // namespace

/**
 * Opaque handle to a pre-computed NTT context for 28-bit primes.
 *
 * Create with `VaeaNtt32::try_new()`, then call `forward()`, `inverse()`,
 * or `negacyclic_mul()` on polynomial buffers.
 */
class VaeaNtt32 {
public:

  /**
   * Creates a new NTT context.
   *
   * # Arguments
   * - `n` — polynomial size, must be a power of 2 ≥ 2
   * - `q` — prime < 2^28, must satisfy q ≡ 1 (mod 2N)
   *
   * Returns an error if parameters are invalid.
   */
  inline static diplomat::result<std::unique_ptr<VaeaNtt32>, VaeaNttError> try_new(size_t n, uint32_t q);

  /**
   * Returns the polynomial size N.
   */
  inline size_t get_n() const;

  /**
   * Returns the prime modulus q.
   */
  inline uint32_t get_q() const;

  /**
   * Forward NTT transform (in-place).
   *
   * `data` must have exactly N elements. All elements must be < q.
   */
  inline diplomat::result<std::monostate, VaeaNttError> forward(diplomat::span<uint32_t> data) const;

  /**
   * Inverse NTT transform (in-place, with N⁻¹ normalization).
   *
   * `data` must have exactly N elements.
   */
  inline diplomat::result<std::monostate, VaeaNttError> inverse(diplomat::span<uint32_t> data) const;

  /**
   * Inverse NTT without N⁻¹ normalization (matches concrete-ntt behavior).
   *
   * `data` must have exactly N elements.
   */
  inline diplomat::result<std::monostate, VaeaNttError> inverse_lazy(diplomat::span<uint32_t> data) const;

  /**
   * Negacyclic polynomial multiplication in Z_q[X]/(X^N + 1).
   *
   * Computes `result = a × b mod (X^N + 1)` using NTT.
   * All three buffers must have exactly N elements.
   * `a` and `b` are modified (transformed to NTT domain).
   */
  inline diplomat::result<std::monostate, VaeaNttError> negacyclic_mul(diplomat::span<uint32_t> a, diplomat::span<uint32_t> b, diplomat::span<uint32_t> result) const;

  /**
   * Pointwise multiplication of two NTT-domain polynomials.
   *
   * All three buffers must have exactly N elements.
   * Inputs must already be in NTT domain.
   */
  inline diplomat::result<std::monostate, VaeaNttError> pointwise_mul(diplomat::span<const uint32_t> a, diplomat::span<const uint32_t> b, diplomat::span<uint32_t> result) const;

    inline const diplomat::capi::VaeaNtt32* AsFFI() const;
    inline diplomat::capi::VaeaNtt32* AsFFI();
    inline static const VaeaNtt32* FromFFI(const diplomat::capi::VaeaNtt32* ptr);
    inline static VaeaNtt32* FromFFI(diplomat::capi::VaeaNtt32* ptr);
    inline static void operator delete(void* ptr);
private:
    VaeaNtt32() = delete;
    VaeaNtt32(const VaeaNtt32&) = delete;
    VaeaNtt32(VaeaNtt32&&) noexcept = delete;
    VaeaNtt32 operator=(const VaeaNtt32&) = delete;
    VaeaNtt32 operator=(VaeaNtt32&&) noexcept = delete;
    static void operator delete[](void*, size_t) = delete;
};


#endif // VaeaNtt32_D_HPP
