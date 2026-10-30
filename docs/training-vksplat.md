# VkSplat training candidates

The training panel provides four default-off candidates based on
[VkSplat b3ad2b0](https://github.com/harry7557558/vksplat/tree/b3ad2b048918496616e0e9346e35c606d813ab55).
They retain the project training schedule and exact Faster-GS depth sorting.
Availability is separate from correctness, training quality and measured speed.

## Select a method

| Training control | Existing default | Candidate and application boundary |
| --- | --- | --- |
| Training Image Storage | Budgeted Cache | GPU Resident; click Load Dataset |
| Loss Backend | Existing | VkSplat; next start/resume after stopping |
| Training Compute Backend | Existing | VkSplat; click Load Dataset |
| VkSplat Subtile Culling | Off | Requires the VkSplat compute backend; next start/resume |

Controls are locked while the worker runs. Resource settings that differ from
the loaded dataset block start/resume until an explicit reload. Preparing GPU
images or a replacement device must succeed before the current model is released.
Changing settings does not itself reset the model. Use a fresh dataset/model for
each independent training comparison.

GPU Resident preloads both training and held-out RGBA8 images and binds the
selected training descriptor without requesting host pixels. It does not evict
images or silently fall back. The panel reports preload and model setup separately
from training, including their combined accounted time. This total excludes dataset
parsing, model export and external quality evaluation. Resident image allocation
includes held-out images; driver heap usage, when available, is reported separately.
The official model-buffer total of 1.37 GiB excludes its dataset images.

VkSplat Loss uses the official 11x11 SSIM kernel, forward/backward dispatches and
L1 fusion. SSIM centers exclude five pixels on every border and use valid-region
normalization. Border pixels can still receive gradients from interior centers.
The existing SSIM selector is inactive under this backend. Images without a valid
11x11 region use the existing loss, with the reason displayed. The loss display
and numerical statistics use the selected objective. Training losses from the two
objectives are not interchangeable quality measurements. Soft-alpha gradients
follow the upstream program; independent mathematical finite differences are
reported separately instead of silently correcting that program.

## Compute backend requirements

The compute backend currently requires native subgroup32, at least 45 KiB shared
memory, Fixed16 tiles, Faster-GS Sorting, Workgroup composite and VkSplat Tensor
Baseline without EarlyCull or RGB latency hiding. It requires Adam beta1=0.9,
beta2=0.999, epsilon=1e-15 and zero gradient clipping. Unsupported devices or
configurations are rejected with a reason; settings are not overwritten.

Parameters use world-space positions and an identity model transform, exact
pinhole intrinsics and dimensions that fit the official packed image-size ABI.
Geometry and SH/moments use separate persistent buffers; SH is RGB-interleaved
in blocks of 32 Gaussians. Quaternion conversion maps project xyzw to upstream
wxyz, including both moments. Scale and opacity are stored activated. Their
inverse conversions are numerical operations, not bitwise reversible copies.
The backend uses near clipping 0.01, alpha cap 0.999 and the corresponding
saturation derivative. It runs projection and tile counting together and retains
dense Adam updates for active zero-gradient parameters.

The existing ADC/reset/SH schedule, optimizer step counting, view scheduler,
exact-depth sorting and four Faster-GS Morton events remain in effect. This is
an explicit combination of upstream kernels and the project schedule, rather than
the complete upstream training system. Training trajectories can change. The old
Direct Intrinsics and Physical16 options are inactive under this backend.

ADC, Morton, export and diagnostic boundaries materialize canonical storage
explicitly. It is released after the boundary; ordinary steps do not maintain a
complete canonical model mirror. Boundary CPU cost is included in total step time
and reported separately; Morton complete timing includes its boundary work.
Native validation aggregates include Adam-moment finiteness; the existing field
breakdown describes parameters. Native model bytes are a component of memory use, not total training VRAM or
transient peak usage.

The independent subtile variant uses the upstream 4x8 mapping and overlap masks.
It shares the compute backend's numerical and backward-state conventions. The
same-state unmasked backend is its reference. Current Tensor scheduling is retained;
the upstream Thompson scheduler and quantized tile-depth sorting are not enabled.

## Correctness and target-device checks

Build Debug and Release sequentially and run their CTest suites. The existing
optimizer GPU diagnostic accepts an explicit synthetic VkSplat suite:

```bash
build/test/Release/training_optimizer_gpu_tests --vksplat-only \
  --gpu "RTX 3090" --require-validation --require-native-lifecycle \
  --report vksplat_3090_correctness.json
```

On Windows append `.exe`. Choose a new report filename on each run. No capture
BIN is required, and the suite does not load a real dataset or start full training.
Reports include device identity, module fingerprints, source commit, numerical
checks and validation errors including device destruction.

Without `--require-native-lifecycle`, devices with insufficient Tensor capability
can check supported loss, state and forward/subtile components. Such a report
explicitly records zero native lifecycle steps and is not full backend admission.
The required flag rejects this partial coverage. Full native lifecycle checks
exercise synthetic optimizer steps, ADC/reset, SH changes, Morton, pause/rebuild
and subtile reconfiguration. Legacy raw captures that cannot identify the new
loss/backend are rejected; PLY export remains supported.

## Full training comparisons

After target correctness checks, use one Release build with shader debug and
actual Vulkan validation disabled. Keep bonsai images_2, 30000 steps, learning
rates including saved opacity LR, ADC, Faster-GS Sorting and Z-order identical.
Disable legacy direct intrinsics and Physical16. Start each group from a fresh
model: existing baseline; resident images only; VkSplat Loss only; native backend
without subtile; native backend with subtile. Compare complete timing and setup,
profiling counts, Gaussian/items trajectories, whole-scope memory and the same
held-out PSNR. Kernel timing classifications must not be counted twice.

284.671 seconds versus 227.931 seconds is a cross-implementation observation of
56.740 seconds, not a promised saving. No training speed or quality gain is assumed
from source similarity, successful compilation or synthetic checks.
