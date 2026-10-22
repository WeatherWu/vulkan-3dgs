// Same protocol on <=32KiB devices; never used for target performance claims.
#define TRAIN_TENSOR_TEST_BATCH 4
#define TRAIN_TENSOR_ROW_EMPTY 1
#include "train_backward_vksplat_tensor.comp.slang"
