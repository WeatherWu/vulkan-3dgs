# Training Reference

## Main Classes

- `GaussianTraining`: orchestration class for dataset, model initialization, training step, validation, and PLY export.
- `TrainingBuffers`: owns all GPU buffers used by training.
- `GaussianForwardRenderer`: compute forward renderer.
- `GaussianBackwardRenderer`: compute backward renderer and optimizer dispatch.
- `GaussianDensificationRenderer`: clone/split/prune passes.
- `TrainingDatasetLoader`: MipNeRF360/COLMAP dataset loading.

## Dataset Loading

Supported dataset format is MipNeRF360/COLMAP-like:

- `sparse/0/cameras.bin` and `sparse/0/images.bin`, or `sparse/cameras.bin`, or root-level COLMAP files.
- Optional `points3D.bin`.
- Image directory selected from requested `images_N`, then `images_4`, `images_2`, `images_8`, `images`.

`TrainingDatasetLoader::loadMipNeRF360Scene()`:

1. Finds image and sparse directories.
2. Reads COLMAP cameras/images.
3. Resolves image filenames case-insensitively by stem.
4. Reads actual image dimensions with `stbi_info`.
5. Uses actual image dimensions for `TrainingCameraFrame::width/height`.
6. Scales `fx/fy/cx/cy` from COLMAP original dimensions to actual image dimensions.
7. Requires all training images in the selected directory to have consistent size.
8. Reads optional `points3D.bin`.

This avoids mismatches between `camera.width / downscale` and actual image file dimensions.

## COLMAP Point Data

`points3D.bin` provides only sparse initialization data:

- 3D position
- RGB color
- reprojection error
- track length

It does not contain Gaussian scale, opacity, rotation, SH high-order coefficients, gradients, optimizer state, loss, or target pixels.

Sparse initialization:

- position from COLMAP point
- SH DC from RGB: `(color - 0.5) / C0`
- opacity as logit of default initial opacity
- scale from CPU kd-tree 3-NN distance estimate
- rotation as identity quaternion

If no sparse points exist, random fallback samples Gaussians around the camera-position range.

## Training Step

`GaussianTraining::trainStep()` currently performs:

1. Upload current target frame image and camera.
2. Create push constants.
3. Prepare tile items:
   - clear forward buffers
   - project Gaussians
   - clear tile ranges
   - count tile coverage
   - prefix tile ranges
4. Submit and wait.
5. Grow tile item buffer if required.
6. Decide densification/pruning for this iteration.
7. Render prepared tiles:
   - emit tile items
   - sort tile items
   - rebuild tile ranges
   - composite pixels
8. Backward:
   - clear gradients
   - loss to pixel
   - pixel to 2D Gaussian
   - 2D Gaussian to 3D Gaussian
9. Gradient descent / Adam optimizer.
10. Optional densify/prune.
11. Submit and wait.
12. Validation on first 5 steps and every `validationInterval_`.
13. Adopt densified Gaussian count.
14. Increment `trainingIteration_`.
15. Advance or resample frame index according to the active schedule.

Images are selected by `TrainingScheduleConfig`:

- `Sequential`: keeps the previous ordered frame advance:

  ```cpp
  currentDatasetFrameIndex_ = (currentDatasetFrameIndex_ + 1) % dataset_.size();
  ```

- `3DGS Random`: uses a random-without-replacement viewpoint stack matching the original 3DGS training loop. When the stack is empty, it is refilled with all training frame indices, then each iteration samples and removes one index.

Sequential mode has no fixed total iteration count. `3DGS Random` sets `totalIterations = 30000` and the UI stops training automatically when `trainingIteration()` reaches that value.

## Training Buffers

`TrainingBuffers` owns:

- `gaussianParams_`
- `gaussianGrads_`
- `adamStates_`
- `projected_`
- tile item/key/sort/range buffers
- `renderedColor_`
- `targetColor_`
- `pixelGrads_`
- `pixelBlendStates_`
- `gaussianVisibility_`
- `densificationStates_`
- densified params/Adam/counters
- `projectedGrads_`
- `loss_`
- general counters
- preview instances
- camera uniform

Descriptor getters return `vk::DescriptorBufferInfo` by value. Save them into local variables when a pointer is needed for descriptor writes.

## Shader Passes

Training shader common files:

- `train_common.slang`
- `train_loss_common.slang`
- `train_forward_common.slang`
- `train_forward_tile_common.slang`
- `train_forward_project_common.slang`
- `train_backward_types.slang`
- `train_backward_common.slang`
- `train_backward_project_common.slang`

Training pass files:

- `train_clear.comp.slang`
- `train_forward_project.comp.slang`
- `train_forward_tile_clear.comp.slang`
- `train_forward_tile_count.comp.slang`
- `train_forward_tile_prefix.comp.slang`
- `train_forward_tile_emit.comp.slang`
- `train_forward_tile_gather_high.comp.slang`
- `train_forward_tile_gather_items.comp.slang`
- `train_forward_tile_sort.comp.slang`
- `train_forward.comp.slang`
- `train_loss.comp.slang`
- `train_backward_clear.comp.slang`
- `train_backward_loss_to_pixel.comp.slang`
- `train_backward_pixel_to_2dgs.comp.slang`
- `train_backward_2dgs_to_3dgs.comp.slang`
- `train_optimizer.comp.slang`
- `train_densify_clear.comp.slang`
- `train_densify_prune.comp.slang`
- `train_pack_render_buffer.comp.slang`

Add new passes to top-level `CMakeLists.txt`.

## Tile Sorting

Training forward uses tile-based compute compositing.

Tile items are emitted with split 64-bit semantic keys:

- high key: tile index
- low key: depth sortable uint
- value/index: original tile item index

`vulkan_radix_sort` only sorts 32-bit key/value, so the implementation does two stable passes:

1. Sort by low depth key.
2. Gather high keys in low-sort order.
3. Sort by high tile key.
4. Gather final sorted tile items.
5. Rebuild tile ranges by scanning sorted keys.

This assumes `vulkan_radix_sort` key/value sort is stable. If instability appears, replace with a native 64-bit key sorter.

## Loss And Optimizer

Loss is per pixel:

- L1 term
- DSSIM term from 11x11 Gaussian SSIM window, sigma 1.5
- combined as `(1 - lossDssimWeight) * L1 + lossDssimWeight * (1 - SSIM)`

Optimizer uses Adam-style state with bias correction, separate learning rates for:

- position
- feature DC
- feature rest
- opacity
- scale
- rotation

Position LR has delay/final/max-steps scheduling.

## Densification/Pruning

Config defaults are in `training_types.hpp` and mirrored in `Application` UI:

- `densifyFromIteration = 500`
- `densifyUntilIteration = 15000`
- `densificationInterval = 100`
- `opacityResetInterval = 3000`
- `maxGaussianCount = 1000000`
- `splitChildren = 2`
- `densifyGradThreshold = 0.0002`
- `minOpacity = 0.005`
- `percentDense = 0.01`
- `screenSizePruneThreshold = 20`
- `worldSizePruneThreshold = 0.1`

Densification/pruning runs at the end of a training step when the iteration is inside the range and divisible by interval.

Current logic supports clone/split split behavior, local Gaussian-normal offsets, scale shrinking, inherited/scaled Adam state, opacity reset, opacity pruning, screen/world size pruning.

## Training Schedule Modes

The Application UI exposes:

- `Sequential`: ordered image traversal, no automatic iteration stop.
- `3DGS Random`: random-without-replacement frame sampling and fixed 30000 total iterations.

The optimizer defaults still mirror common 3DGS values, but forward/backward math remains project-specific and should not be treated as exact reference parity without further audit.

## Known Differences From Standard 3DGS

Current implementation is not yet standard 3DGS parity:

- Fixed 30000-iteration stop and random viewpoint-stack sampling are available only in `3DGS Random` mode; `Sequential` mode keeps the old behavior.
- Training validation reads back GPU data and may be expensive.
- Forward/backward math is still project-specific and should be audited against the paper/reference implementation before claiming parity.
- Tile sort depends on stable 32-bit radix passes rather than native 64-bit key sort.
