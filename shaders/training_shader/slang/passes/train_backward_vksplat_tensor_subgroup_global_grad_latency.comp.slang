// Fixed16 Global-gradient A/B: defer RGB consumption until geometry is accumulated.
#define TRAIN_TENSOR_SUBGROUP_REDUCTION 1
#define TRAIN_TENSOR_GLOBAL_PIXEL_GRAD 1
#define TRAIN_TENSOR_RGB_LATENCY_HIDING 1
#include "train_backward_vksplat_tensor.comp.slang"
