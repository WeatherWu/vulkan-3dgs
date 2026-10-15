// Fixed16 A/B variant: retain subgroup reduction, bypass the RGB shared cache.
#define TRAIN_TENSOR_SUBGROUP_REDUCTION 1
#define TRAIN_TENSOR_GLOBAL_PIXEL_GRAD 1
#include "train_backward_vksplat_tensor.comp.slang"
