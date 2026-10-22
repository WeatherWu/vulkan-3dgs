// Test-only fallback on GPUs unable to run Tensor. Never selected by the app.
#define TRAIN_PIXEL_CULL_TEST 1
#include "train_backward_pixel_to_2dgs.comp.slang"
