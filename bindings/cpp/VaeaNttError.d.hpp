#ifndef VaeaNttError_D_HPP
#define VaeaNttError_D_HPP

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
    enum VaeaNttError {
      VaeaNttError_InvalidSize = 0,
      VaeaNttError_NotPrime = 1,
      VaeaNttError_NotNttFriendly = 2,
      VaeaNttError_PrimeTooLarge = 3,
      VaeaNttError_InvalidBufferLength = 4,
    };

    typedef struct VaeaNttError_option {union { VaeaNttError ok; }; bool is_ok; } VaeaNttError_option;
} // namespace capi
} // namespace

/**
 * Error codes returned by VaeaNTT operations.
 */
class VaeaNttError {
public:
    enum Value {
        /**
         * N must be a power of 2 >= 2.
         */
        InvalidSize = 0,
        /**
         * q must be prime.
         */
        NotPrime = 1,
        /**
         * q must satisfy q ≡ 1 (mod 2N).
         */
        NotNttFriendly = 2,
        /**
         * q must be < 2^28.
         */
        PrimeTooLarge = 3,
        /**
         * Buffer length does not match the context's N.
         */
        InvalidBufferLength = 4,
    };

    VaeaNttError(): value(Value::InvalidSize) {}

    // Implicit conversions between enum and ::Value
    constexpr VaeaNttError(Value v) : value(v) {}
    constexpr operator Value() const { return value; }
    // Prevent usage as boolean value
    explicit operator bool() const = delete;

    inline diplomat::capi::VaeaNttError AsFFI() const;
    inline static VaeaNttError FromFFI(diplomat::capi::VaeaNttError c_enum);
private:
    Value value;
};


#endif // VaeaNttError_D_HPP
