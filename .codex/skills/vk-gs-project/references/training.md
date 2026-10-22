# Training reference

## Ownership

- GaussianTraining: dataset/model lifecycle and training-step semantics.
- TrainingController: device selection, worker, stop/join, immutable run settings,
  timer, benchmark and export; UI reads mutex-protected snapshots.
- TrainingFrameScheduler: frame order, seeded viewpoint stack, completion and prefetch.
- TrainingModelIO: host initialization, KD-tree scale estimates and PLY serialization.
- TrainingImageRuntime: streamer/device cache, targets and upload timeline values.
- TrainingStepExecutor: command pool/buffers, prepare/main submissions and fences,
  count/capacity handling, validation scheduling and densification adoption.
- TrainingBuffers: GPU arrays, capacities and descriptor values.
- GaussianForwardRenderer: projection, count/prefix, emit/sort/ranges and composite.
- GaussianBackwardRenderer: backward facade.
- BackwardLossPass: loss, clear, coefficient filtering and pixel gradients.
- PixelTo2DGSDispatcher: backward capability policy and variant dispatch.
- ProjectionOptimizerPass: fused or separate projection/Adam.
- BackwardValidationPass: Gaussian and compact-result validation.
- GaussianDensificationRenderer: clone/split/prune and opacity reset.

## Dataset and initialization

The loader accepts COLMAP binary cameras/images under sparse/0, sparse or the
scene root, with optional points3D.bin. It resolves image names, reads actual
dimensions and scales intrinsics accordingly. Images must share an extent.

Filename-sorted views at zero-based indices 0,8,16,... are held out; a single-view
scene stays trainable. Fixed Benchmark indices refer to the training subset.
Match this split to evaluation.

Sparse points initialize positions and RGB-derived SH DC. CPU 3-NN distances
initialize scale; opacity is stored as a logit and rotation as an identity
quaternion. Without sparse points, configured random initialization is used.

## Image storage and caches

Targets are packed linear RGBA8, four bytes per pixel. The host ImageStreamer
owns request/prefetch and LRU entries. Host budget defaults to 10% of available
RAM with 2 GiB reserved. Active handles prevent eviction.

The disk cache uses KTX2 chunks with 16 image layers and a 20 GiB default quota.
The device cache stores aligned image slots and considers VK_EXT_memory_budget.
Its normal reserve is max(512 MiB,15% of budget), pre-ADC reserve max(1 GiB,25%),
both capped at half; allocation is capped at one quarter of the device heap
budget and full-dataset size. Timeline uploads use a three-slot staging ring.
VULKAN_3DGS_SYNC_IMAGE_UPLOAD=1 selects the synchronous fallback.

## One step

1. Select/upload target and camera; construct push constants.
2. Record projection, tile coverage count and hierarchical prefix.
3. Copy a 16-byte count, submit prepare and wait.
4. Read the mapped count and grow tile item storage when required.
5. Record emit, two stable 32-bit radix sorts, final gather, tile ranges and composite.
6. Record loss, backward clear, loss-to-pixel and selected pixel-to-2DGS.
7. Record fused projection/dense Adam, or the separate path as configured.
8. Perform scheduled validation and densify/prune.
9. Submit main, wait, consume compact validation and adopt output buffers.
10. Advance iteration and frame scheduler.

The two stable radix passes order depth then tile ID. Preserve equal-key order.
Coverage/prefix gives each Gaussian a contiguous emit range. Empty tile ranges
are zero. Fixed8 and Fixed16 use the corresponding fixed-grid dimensions.

## Backward controls

Auto selects Direct versus subgroup traversal according to actual capabilities
and prefix utilization. It does not choose Per-Splat or Tensor. Unsupported
explicit modes fall back to Direct.

Per-Splat uses 128 threads/tile and a subgroup-aligned batch capped at 128.
Fixed16 Tensor uses 256 threads and Gaussian batch 16. The nine accumulated
components are center XY, conic/opacity XYZW and RGB, in three aligned float4s.
The Tensor capability gate requires at least 45 KiB shared memory.

Tensor subgroup reduction is opt-in, native subgroup32 and compute BASIC/SHUFFLE.
Global pixel-gradient reads require that subgroup path. Both switches default
off; snapshots report effective rather than merely requested settings. Declared
shared storage is 45824 bytes for Shared, 36612 for Subgroup and 33540 for
Subgroup/Global. Tensor does not invoke Tensor Core MMA.

## SSIM

The 11x11 Gaussian-window loss combines L1 and DSSIM with zero padding and fixed
normalization. SsimBackwardState contains reusable derivative coefficients.

- ssim_mode=0, default: 2D loss and 2D pixel-gradient neighborhood.
- ssim_mode=2: the same 2D loss and coefficients, followed by horizontal
  coefficient filtering and vertical pixel gradients.

Hybrid uses three float4 planes (48 bytes/pixel) at binding 42. Storage is checked
against device range and shader indexing limits, changes with extent/mode and is
released in 2D mode. Barriers cover both horizontal writes/vertical reads and
vertical reads before scratch reuse. Invalid API modes are rejected; unsupported
persisted values select 2D. Explicit ssim_mode takes precedence over the compatible
separable_ssim key without reinterpreting unsupported method IDs.

## Optimizer and densification

GaussianTrainParam and GaussianGrad each contain 15 float4s (240 bytes).
AdamState stores two GaussianGrad values (480 bytes). Ordinary fused projection
and Adam avoid full-gradient writeback, but the fallback allocation remains.

VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1 selects separate dispatches.
Active zero-gradient parameters retain dense Adam semantics. Inactive SH bands
keep their own update mask. Preserve pre-update values for projection derivatives.

Densification exposes Visible Support and Compact statistics modes. Default
scheduling is iterations 500–15000, interval100, opacity reset3000, max10M
Gaussians, splitChildren2, gradient threshold0.0002 and minOpacity0.005.
Buffer float32 atomic add is required; float min/max is optional because uint
radius variants are available. Output parameter/Adam buffers are ping-ponged.

## Validation and profiling

Numerical validation runs for the first five steps and then the configured
interval. Pixel partials are 80 bytes, Gaussian partials 32 bytes and final GPU
result112 bytes. A three-slot persistent host ring consumes the compact result;
normal validation does not download full model/rendered/loss arrays.

TrainingProfilingService owns query-pool lifetime and timestamp conversion.
Renderer/pass observers are detached before destruction. CPU submit/wait includes
GPU execution; nested Tile-local backward time is not an extra additive stage.

Fixed Benchmark pins frame/model/iteration and disables optimizer/ADC. Use
0 warmup/1 measured for exported gradient comparisons, and a separate bounded
timing run for a new performance hypothesis. Pure Training measures normal
training with reduced UI refresh and final PLY export.

## Gradient format

The writer emits V3DGGRD1 version7: 84-byte little-endian header, 12 float32s
and one zero membership byte per Gaussian, then width*height float4 pixel
gradients. Active projected components are 0,1,4,5,6,7,8,9,10; padding is zero.
Reserved child counts are zero, child batch8 and RGB schedule0.
SSIM IDs are explicit (0 or2). Actual Tensor flags come from benchmark state.

The comparison script accepts versions1–7. Their metadata/payload descriptions
are input compatibility contracts, not runtime method availability. Preserve
exact path identity for repeats and require an explicit override for different
pixel kernels. Default projected atol1e-5/rtol1e-3, forward atol1e-4 and pixel
atol1e-9 are not adjustable merely to obtain a pass.

## ABI and build

Keep C++/Slang layouts and bindings synchronized. Save descriptor values in
locals when passing their addresses. Register shader passes in CMake. Shader
debug information is controlled by VULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO;
it is a CMake flag, not a runtime optimization selector.
