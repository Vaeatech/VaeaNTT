#ifndef VaeaNttError_D_H
#define VaeaNttError_D_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "diplomat_runtime.h"





typedef enum VaeaNttError {
  VaeaNttError_InvalidSize = 0,
  VaeaNttError_NotPrime = 1,
  VaeaNttError_NotNttFriendly = 2,
  VaeaNttError_PrimeTooLarge = 3,
  VaeaNttError_InvalidBufferLength = 4,
} VaeaNttError;

typedef struct VaeaNttError_option {union { VaeaNttError ok; }; bool is_ok; } VaeaNttError_option;



#endif // VaeaNttError_D_H
