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
are zero. Fixed8 and Fixed16 use fixed-grid dimensions. Adaptive Spatial builds and appends stable child lists and uses indirect child dispatch.

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

The experimental branch supports ssim_mode0 (2D, default), 1 (full separable)
and2 (2D loss with separable backward). Modes1/2 require five/three float4
scratch planes respectively at binding42. BackwardLossPass owns dispatch and
barriers; TrainingBuffers owns checked extent/mode allocation. Configuration
and exports report exact IDs and effective state.

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

## Physical Morton ordering

The default-off `morton_ordering` setting physically reorders the current model
before displayed steps 1, 5001, 10001 and 15001. Parameters, both moments and ADC
statistics share one permutation. GPU completion and position-status checks
precede adoption. The full 63-bit Morton code uses 21 bits per axis and the
longest bounding-box side for normalization. Dense Adam remains in use.

`GaussianTraining` owns the event schedule and resource lifetime. Fixed Benchmark
does not trigger events. Renderer rebuild preserves event history; a new model
clears it. Disabling stops future events without restoring the previous order.
Physical indices can change split random draws, depth ties and full-run results;
exact state copying does not establish identical training trajectories or quality.
The independent sorting choice remains subject to its own numerical contract.

The Z-order profiling rows average events, not ordinary steps. Amortize using
event count divided by completed steps. Workspace bytes are logical temporary
storage, not native peak VRAM. See `docs/training-morton-ordering.md` for controls
and the pinned upstream method.

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

The writer emits V3DGGRD1 version9 with108-byte header. Payload contains12
float32 projected components and one membership byte perGaussian, then float4
pixel gradients. Metadata records actual paths, Tensor flags, SSIM, checkpoint
state/layout/tasks/fallback and pixel-culling method. The reader accepts1–9.
Failed/skipped/fallback checkpoint captures are rejected rather than mislabeled.
Repeat captures must retain exact effective settings. See docs/DEVELOPMENT.md.

## Experimental tools

Checkpoint state construction and warp backward are frozen-benchmark commands.
Guarded EarlyCull has separate default-off training and benchmark requests.
Scanline/row-mask tools operate on frozen captures without training writeback.
Optimizer capture executes one actual ordinary update and exports pre/post;
replay and streamed Adam use private state. Scheduling sampling retains original
submissions and fences; its sort runner does not enable GPU-driven training.
Application --validation on|off controls the instance request before creation;
default on in both builds. These features require their own numerical/resource
assessment and do not inherit acceptance from their availability.

## ABI and build

Keep C++/Slang layouts and bindings synchronized. Save descriptor values in
locals when passing their addresses. Register shader passes in CMake. Shader
debug information is controlled by VULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO;
it is a CMake flag, not a runtime optimization selector.
