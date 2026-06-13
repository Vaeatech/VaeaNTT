#ifndef VaeaPrimeList_D_HPP
#define VaeaPrimeList_D_HPP

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <memory>
#include <functional>
#include <optional>
#include <cstdlib>
#include "diplomat_runtime.hpp"


namespace diplomat {
namespace capi {
    struct VaeaPrimeList;
} // namespace capi
} // namespace

/**
 * Result of prime generation: a list of primes.
 */
class VaeaPrimeList {
public:

  /**
   * Generates `count` NTT-friendly 28-bit primes for polynomial size `n`.
   */
  inline static std::unique_ptr<VaeaPrimeList> generate(size_t n, size_t count);

  /**
   * Returns the number of primes in the list.
   */
  inline size_t len() const;

  /**
   * Returns the prime at the given index (0-based).
   */
  inline uint32_t get(size_t index) const;

    inline const diplomat::capi::VaeaPrimeList* AsFFI() const;
    inline diplomat::capi::VaeaPrimeList* AsFFI();
    inline static const VaeaPrimeList* FromFFI(const diplomat::capi::VaeaPrimeList* ptr);
    inline static VaeaPrimeList* FromFFI(diplomat::capi::VaeaPrimeList* ptr);
    inline static void operator delete(void* ptr);
private:
    VaeaPrimeList() = delete;
    VaeaPrimeList(const VaeaPrimeList&) = delete;
    VaeaPrimeList(VaeaPrimeList&&) noexcept = delete;
    VaeaPrimeList operator=(const VaeaPrimeList&) = delete;
    VaeaPrimeList operator=(VaeaPrimeList&&) noexcept = delete;
    static void operator delete[](void*, size_t) = delete;
};


#endif // VaeaPrimeList_D_HPP
