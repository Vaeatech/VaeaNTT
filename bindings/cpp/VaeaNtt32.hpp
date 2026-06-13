#ifndef VaeaNtt32_HPP
#define VaeaNtt32_HPP

#include "VaeaNtt32.d.hpp"

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <memory>
#include <functional>
#include <optional>
#include <cstdlib>
#include "VaeaNttError.hpp"
#include "diplomat_runtime.hpp"


namespace diplomat {
namespace capi {
    extern "C" {

    typedef struct VaeaNtt32_try_new_result {union {diplomat::capi::VaeaNtt32* ok; diplomat::capi::VaeaNttError err;}; bool is_ok;} VaeaNtt32_try_new_result;
    VaeaNtt32_try_new_result VaeaNtt32_try_new(size_t n, uint32_t q);

    size_t VaeaNtt32_get_n(const diplomat::capi::VaeaNtt32* self);

    uint32_t VaeaNtt32_get_q(const diplomat::capi::VaeaNtt32* self);

    typedef struct VaeaNtt32_forward_result {union { diplomat::capi::VaeaNttError err;}; bool is_ok;} VaeaNtt32_forward_result;
    VaeaNtt32_forward_result VaeaNtt32_forward(const diplomat::capi::VaeaNtt32* self, diplomat::capi::DiplomatU32ViewMut data);

    typedef struct VaeaNtt32_inverse_result {union { diplomat::capi::VaeaNttError err;}; bool is_ok;} VaeaNtt32_inverse_result;
    VaeaNtt32_inverse_result VaeaNtt32_inverse(const diplomat::capi::VaeaNtt32* self, diplomat::capi::DiplomatU32ViewMut data);

    typedef struct VaeaNtt32_inverse_lazy_result {union { diplomat::capi::VaeaNttError err;}; bool is_ok;} VaeaNtt32_inverse_lazy_result;
    VaeaNtt32_inverse_lazy_result VaeaNtt32_inverse_lazy(const diplomat::capi::VaeaNtt32* self, diplomat::capi::DiplomatU32ViewMut data);

    typedef struct VaeaNtt32_negacyclic_mul_result {union { diplomat::capi::VaeaNttError err;}; bool is_ok;} VaeaNtt32_negacyclic_mul_result;
    VaeaNtt32_negacyclic_mul_result VaeaNtt32_negacyclic_mul(const diplomat::capi::VaeaNtt32* self, diplomat::capi::DiplomatU32ViewMut a, diplomat::capi::DiplomatU32ViewMut b, diplomat::capi::DiplomatU32ViewMut result);

    typedef struct VaeaNtt32_pointwise_mul_result {union { diplomat::capi::VaeaNttError err;}; bool is_ok;} VaeaNtt32_pointwise_mul_result;
    VaeaNtt32_pointwise_mul_result VaeaNtt32_pointwise_mul(const diplomat::capi::VaeaNtt32* self, diplomat::capi::DiplomatU32View a, diplomat::capi::DiplomatU32View b, diplomat::capi::DiplomatU32ViewMut result);

    void VaeaNtt32_destroy(VaeaNtt32* self);

    } // extern "C"
} // namespace capi
} // namespace

inline diplomat::result<std::unique_ptr<VaeaNtt32>, VaeaNttError> VaeaNtt32::try_new(size_t n, uint32_t q) {
    auto result = diplomat::capi::VaeaNtt32_try_new(n,
        q);
    return result.is_ok ? diplomat::result<std::unique_ptr<VaeaNtt32>, VaeaNttError>(diplomat::Ok<std::unique_ptr<VaeaNtt32>>(std::unique_ptr<VaeaNtt32>(VaeaNtt32::FromFFI(result.ok)))) : diplomat::result<std::unique_ptr<VaeaNtt32>, VaeaNttError>(diplomat::Err<VaeaNttError>(VaeaNttError::FromFFI(result.err)));
}

inline size_t VaeaNtt32::get_n() const {
    auto result = diplomat::capi::VaeaNtt32_get_n(this->AsFFI());
    return result;
}

inline uint32_t VaeaNtt32::get_q() const {
    auto result = diplomat::capi::VaeaNtt32_get_q(this->AsFFI());
    return result;
}

inline diplomat::result<std::monostate, VaeaNttError> VaeaNtt32::forward(diplomat::span<uint32_t> data) const {
    auto result = diplomat::capi::VaeaNtt32_forward(this->AsFFI(),
        {data.data(), data.size()});
    return result.is_ok ? diplomat::result<std::monostate, VaeaNttError>(diplomat::Ok<std::monostate>()) : diplomat::result<std::monostate, VaeaNttError>(diplomat::Err<VaeaNttError>(VaeaNttError::FromFFI(result.err)));
}

inline diplomat::result<std::monostate, VaeaNttError> VaeaNtt32::inverse(diplomat::span<uint32_t> data) const {
    auto result = diplomat::capi::VaeaNtt32_inverse(this->AsFFI(),
        {data.data(), data.size()});
    return result.is_ok ? diplomat::result<std::monostate, VaeaNttError>(diplomat::Ok<std::monostate>()) : diplomat::result<std::monostate, VaeaNttError>(diplomat::Err<VaeaNttError>(VaeaNttError::FromFFI(result.err)));
}

inline diplomat::result<std::monostate, VaeaNttError> VaeaNtt32::inverse_lazy(diplomat::span<uint32_t> data) const {
    auto result = diplomat::capi::VaeaNtt32_inverse_lazy(this->AsFFI(),
        {data.data(), data.size()});
    return result.is_ok ? diplomat::result<std::monostate, VaeaNttError>(diplomat::Ok<std::monostate>()) : diplomat::result<std::monostate, VaeaNttError>(diplomat::Err<VaeaNttError>(VaeaNttError::FromFFI(result.err)));
}

inline diplomat::result<std::monostate, VaeaNttError> VaeaNtt32::negacyclic_mul(diplomat::span<uint32_t> a, diplomat::span<uint32_t> b, diplomat::span<uint32_t> result) const {
    auto result = diplomat::capi::VaeaNtt32_negacyclic_mul(this->AsFFI(),
        {a.data(), a.size()},
        {b.data(), b.size()},
        {result.data(), result.size()});
    return result.is_ok ? diplomat::result<std::monostate, VaeaNttError>(diplomat::Ok<std::monostate>()) : diplomat::result<std::monostate, VaeaNttError>(diplomat::Err<VaeaNttError>(VaeaNttError::FromFFI(result.err)));
}

inline diplomat::result<std::monostate, VaeaNttError> VaeaNtt32::pointwise_mul(diplomat::span<const uint32_t> a, diplomat::span<const uint32_t> b, diplomat::span<uint32_t> result) const {
    auto result = diplomat::capi::VaeaNtt32_pointwise_mul(this->AsFFI(),
        {a.data(), a.size()},
        {b.data(), b.size()},
        {result.data(), result.size()});
    return result.is_ok ? diplomat::result<std::monostate, VaeaNttError>(diplomat::Ok<std::monostate>()) : diplomat::result<std::monostate, VaeaNttError>(diplomat::Err<VaeaNttError>(VaeaNttError::FromFFI(result.err)));
}

inline const diplomat::capi::VaeaNtt32* VaeaNtt32::AsFFI() const {
    return reinterpret_cast<const diplomat::capi::VaeaNtt32*>(this);
}

inline diplomat::capi::VaeaNtt32* VaeaNtt32::AsFFI() {
    return reinterpret_cast<diplomat::capi::VaeaNtt32*>(this);
}

inline const VaeaNtt32* VaeaNtt32::FromFFI(const diplomat::capi::VaeaNtt32* ptr) {
    return reinterpret_cast<const VaeaNtt32*>(ptr);
}

inline VaeaNtt32* VaeaNtt32::FromFFI(diplomat::capi::VaeaNtt32* ptr) {
    return reinterpret_cast<VaeaNtt32*>(ptr);
}

inline void VaeaNtt32::operator delete(void* ptr) {
    diplomat::capi::VaeaNtt32_destroy(reinterpret_cast<diplomat::capi::VaeaNtt32*>(ptr));
}


#endif // VaeaNtt32_HPP
