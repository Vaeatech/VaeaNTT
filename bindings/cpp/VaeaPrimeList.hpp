#ifndef VaeaPrimeList_HPP
#define VaeaPrimeList_HPP

#include "VaeaPrimeList.d.hpp"

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
    extern "C" {

    diplomat::capi::VaeaPrimeList* VaeaPrimeList_generate(size_t n, size_t count);

    size_t VaeaPrimeList_len(const diplomat::capi::VaeaPrimeList* self);

    uint32_t VaeaPrimeList_get(const diplomat::capi::VaeaPrimeList* self, size_t index);

    void VaeaPrimeList_destroy(VaeaPrimeList* self);

    } // extern "C"
} // namespace capi
} // namespace

inline std::unique_ptr<VaeaPrimeList> VaeaPrimeList::generate(size_t n, size_t count) {
    auto result = diplomat::capi::VaeaPrimeList_generate(n,
        count);
    return std::unique_ptr<VaeaPrimeList>(VaeaPrimeList::FromFFI(result));
}

inline size_t VaeaPrimeList::len() const {
    auto result = diplomat::capi::VaeaPrimeList_len(this->AsFFI());
    return result;
}

inline uint32_t VaeaPrimeList::get(size_t index) const {
    auto result = diplomat::capi::VaeaPrimeList_get(this->AsFFI(),
        index);
    return result;
}

inline const diplomat::capi::VaeaPrimeList* VaeaPrimeList::AsFFI() const {
    return reinterpret_cast<const diplomat::capi::VaeaPrimeList*>(this);
}

inline diplomat::capi::VaeaPrimeList* VaeaPrimeList::AsFFI() {
    return reinterpret_cast<diplomat::capi::VaeaPrimeList*>(this);
}

inline const VaeaPrimeList* VaeaPrimeList::FromFFI(const diplomat::capi::VaeaPrimeList* ptr) {
    return reinterpret_cast<const VaeaPrimeList*>(ptr);
}

inline VaeaPrimeList* VaeaPrimeList::FromFFI(diplomat::capi::VaeaPrimeList* ptr) {
    return reinterpret_cast<VaeaPrimeList*>(ptr);
}

inline void VaeaPrimeList::operator delete(void* ptr) {
    diplomat::capi::VaeaPrimeList_destroy(reinterpret_cast<diplomat::capi::VaeaPrimeList*>(ptr));
}


#endif // VaeaPrimeList_HPP
