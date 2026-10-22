// Frozen-benchmark-only method 1. The original wrappers remain unchanged.
#define TRAIN_TENSOR_SUBGROUP_REDUCTION 1
#define TRAIN_TENSOR_GLOBAL_PIXEL_GRAD 1
#define TRAIN_TENSOR_PIXEL_EARLY_CULL 1
#include "train_backward_vksplat_tensor.comp.slang"
