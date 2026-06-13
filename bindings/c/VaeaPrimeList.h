#ifndef VaeaPrimeList_H
#define VaeaPrimeList_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"


#include "VaeaPrimeList.d.h"






VaeaPrimeList* VaeaPrimeList_generate(size_t n, size_t count);

size_t VaeaPrimeList_len(const VaeaPrimeList* self);

uint32_t VaeaPrimeList_get(const VaeaPrimeList* self, size_t index);

void VaeaPrimeList_destroy(VaeaPrimeList* self);





#endif // VaeaPrimeList_H
