#ifndef VaeaNttError_HPP
#define VaeaNttError_HPP

#include "VaeaNttError.d.hpp"

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

} // namespace capi
} // namespace

inline diplomat::capi::VaeaNttError VaeaNttError::AsFFI() const {
    return static_cast<diplomat::capi::VaeaNttError>(value);
}

inline VaeaNttError VaeaNttError::FromFFI(diplomat::capi::VaeaNttError c_enum) {
    switch (c_enum) {
        case diplomat::capi::VaeaNttError_InvalidSize:
        case diplomat::capi::VaeaNttError_NotPrime:
        case diplomat::capi::VaeaNttError_NotNttFriendly:
        case diplomat::capi::VaeaNttError_PrimeTooLarge:
        case diplomat::capi::VaeaNttError_InvalidBufferLength:
            return static_cast<VaeaNttError::Value>(c_enum);
        default:
            std::abort();
    }
}
#endif // VaeaNttError_HPP
