#ifndef VaeaNtt32_H
#define VaeaNtt32_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"

#include "VaeaNttError.d.h"

#include "VaeaNtt32.d.h"






typedef struct VaeaNtt32_try_new_result {union {VaeaNtt32* ok; VaeaNttError err;}; bool is_ok;} VaeaNtt32_try_new_result;
VaeaNtt32_try_new_result VaeaNtt32_try_new(size_t n, uint32_t q);

size_t VaeaNtt32_get_n(const VaeaNtt32* self);

uint32_t VaeaNtt32_get_q(const VaeaNtt32* self);

typedef struct VaeaNtt32_forward_result {union { VaeaNttError err;}; bool is_ok;} VaeaNtt32_forward_result;
VaeaNtt32_forward_result VaeaNtt32_forward(const VaeaNtt32* self, DiplomatU32ViewMut data);

typedef struct VaeaNtt32_inverse_result {union { VaeaNttError err;}; bool is_ok;} VaeaNtt32_inverse_result;
VaeaNtt32_inverse_result VaeaNtt32_inverse(const VaeaNtt32* self, DiplomatU32ViewMut data);

typedef struct VaeaNtt32_inverse_lazy_result {union { VaeaNttError err;}; bool is_ok;} VaeaNtt32_inverse_lazy_result;
VaeaNtt32_inverse_lazy_result VaeaNtt32_inverse_lazy(const VaeaNtt32* self, DiplomatU32ViewMut data);

typedef struct VaeaNtt32_negacyclic_mul_result {union { VaeaNttError err;}; bool is_ok;} VaeaNtt32_negacyclic_mul_result;
VaeaNtt32_negacyclic_mul_result VaeaNtt32_negacyclic_mul(const VaeaNtt32* self, DiplomatU32ViewMut a, DiplomatU32ViewMut b, DiplomatU32ViewMut result);

typedef struct VaeaNtt32_pointwise_mul_result {union { VaeaNttError err;}; bool is_ok;} VaeaNtt32_pointwise_mul_result;
VaeaNtt32_pointwise_mul_result VaeaNtt32_pointwise_mul(const VaeaNtt32* self, DiplomatU32View a, DiplomatU32View b, DiplomatU32ViewMut result);

void VaeaNtt32_destroy(VaeaNtt32* self);





#endif // VaeaNtt32_H
