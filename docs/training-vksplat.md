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

Morton, export and diagnostic boundaries materialize canonical storage
explicitly. ADC operates directly on native parameters and both moments. It is released after the boundary; ordinary steps do not maintain a
complete canonical model mirror. Boundary CPU cost is included in total step time
and reported separately; Morton complete timing includes its boundary work.
Native validation aggregates include Adam-moment finiteness; the existing field
breakdown describes parameters. Native model bytes are a component of memory use, not total training VRAM or
transient peak usage.

## Native default ADC

The native backend uses the pinned `default.slang`, `strategy_utils.slang`,
`cumsum.slang`, `where.slang` and upstream post-backward sequence. A separate owner
accumulates normalized center gradients only for positive official projection
radii. It classifies with a strict gradient threshold, obtains stable indices,
allocates the full grown native state before writes, duplicates/splits, then
computes the official keep mask and, when pruning is nonempty, stably compacts every field and moment.
It does not use the existing densifier or canonical import/export during ADC.
The next step and current model validation use the adopted output count.

Only two split children and Visible Support statistics are supported. The native
backend uses `refine_scale2d_stop_iter=0`; screen-radius growth/pruning is inactive.
The stored Screen Prune Size remains a setting of the Existing backend and is not
converted to a native normalized threshold. Gradient, min-opacity, dense scale
and world-size thresholds use the selected values and upstream units.

The independent ADC generator is the upstream `std::default_random_engine`, starts at seed 42, draws once per real step, and
survives pauses, renderer rebuilds and Morton boundaries. Fixed Benchmark neither
accumulates statistics nor advances this generator. A fresh model resets it.
Duplicate copies preserve original moments and clear all child moments. Split
uses upstream `randn3(seed, source)` and `randn3(seed, destination)`, normalizes the
quaternion, divides both scales by 1.6, and clears both moment sets. Opacity reset
clamps activated opacity to twice min-opacity and clears only opacity moments.

Logs report input, duplicate, split, prune, output, capacity, RNG seed and actual
reset, with complete CPU cost and GPU batch cost. Native ADC rejects a growth
budget overflow before growth writes, without truncation. An empty keep mask reports zero output, sets the training logical count to zero,
and requires explicit dataset reload. An uncertain submitted state blocks
resume and export until reload. Allocation/queue errors do not imply a successful
rollback. CPU event timing includes allocation, compaction and synchronization;
GPU event timing covers the submitted batches, not CPU waits.

The independent subtile variant uses the upstream 4x8 mapping and overlap masks.
It shares the compute backend's numerical and backward-state conventions. The
same-state unmasked backend is its reference. Current Tensor scheduling is retained;
the upstream Thompson scheduler and quantized tile-depth sorting are not enabled.

## Independent optimization controls

The VkSplat controls retain separate default-off reference comparisons:

| Control / persistent key | Reference default | Candidate |
| --- | --- | --- |
| VkSplat ADC Execution / `vksplat_adc_execution` | Reference (0) | Reuse (1); Load Dataset required |
| VkSplat Projection Storage / `vksplat_projection_storage` | Reference (0) | Official SoA (1); Load Dataset required |
| VkSplat Loss Reporting / `vksplat_loss_reporting` | Every Step (0) | On Check (1); next start/resume |
| Training Numeric Checks / `training_numeric_checks` | Periodic (0) | First 5 + Final (1); next start/resume |

ADC and projection choices apply only to VkSplat compute. Loss reporting applies
only to VkSplat Loss. Unsupported combinations show their inactive state. Controls
are locked while running; changing resource settings requires an explicit reload
and never silently resets the model. Stopping a run is not normal completion.

Reuse keeps masks, scan/index workspaces and physical capacities between events.
Counts use the upstream sum reduction before conditional prefix/where operations.
Pruning uses one reusable attribute scratch buffer and copies each compacted field
back to persistent native storage. Duplicate/split/prune/reset arithmetic, the
RNG and project event schedule are identical to the reference ADC.

Official SoA stores XY, exact FP32 depth, conic/opacity, RGB, packed rectangles,
radius and overlap counts separately. Consumers read that layout directly; it
requires no per-step projected-AoS conversion. Composite and Tensor consumers load
only XY, conic/opacity and RGB; sorting reads depth and ADC reads radius. Rectangles use two uint32 words
without a new 16-bit device-feature request. The existing projected-gradient
representation and Tensor thread organization remain unchanged.

On Check omits scalar-loss output on ordinary steps while computing the same
pixel gradients every step. Reports identify the last scalar-loss step. Numerical
checks and explicit finite inspection require fresh scalar output; an older value
is never represented as the current objective.

First 5 + Final performs six checks in a normal 30000-step run: steps 1 through 5
and step 30000. Intermediate non-finite detection may be delayed. It is independent
of Vulkan validation layers; smoke tests keep their periodic contract and Fixed
Benchmark retains its final measured-step check.

## Timing scopes

CPU submit/wait includes required GPU execution and scheduler latency; it is not
removable host overhead. ADC complete includes allocation, recording, copies,
GPU waits and count/workspace adoption. Nested subranges are labelled and must
not be added to their parent. Detailed scheduling diagnostics remain opt-in.

The upstream `Total - 220.930 / 227.931 secs` prints two different quantities:
the sum of named stages excluding names beginning with `_`, and wall-clock time
around the Python training loop. The first can include HostGuard host intervals
and overlapping nested stages (Morton sort is also inside post-backward). The
7.001-second difference is not a direct Python-overhead measurement. Upstream
initialization and dataset preparation occur before that wall timer; compare its
227.931 seconds with this project's training time, not the preparation-inclusive
total. The official 1.37 GiB model-buffer value is not complete process VRAM.

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
and subtile reconfiguration. They require real growth, count conservation,
post-adoption next-step counts, actual reset opacity/moments and zero-count
propagation on an empty keep mask. The ADC component suite also covers strict
threshold ties, stable SH/moment movement, prefix boundaries, budget refusal,
independent RNG progression and positive-radius statistics. An expected empty-mask
error in a negative test is labelled as such; final validation status remains decisive.
Legacy raw captures that cannot identify the new
loss/backend are rejected; PLY export remains supported.

## Full training comparisons

After target correctness checks, use one Release build with shader debug and
actual Vulkan validation disabled. Keep bonsai images_2, 30000 steps, GPU Resident,
VkSplat compute and Loss, the same subtile setting, saved learning rates, ADC,
Faster-GS Sorting and Z-order identical. Disable legacy direct intrinsics and
Physical16. Each group starts from a fresh model:

| Group | Change from the current VkSplat reference |
| --- | --- |
| A | All additional optimization controls at reference defaults |
| B | ADC Reuse only |
| C | Official SoA only |
| D | Loss On Check only |
| E | First 5 + Final numeric checks only |
| F | Combination after the independent comparisons succeed |

Pure Training is a separate comparison condition; do not enable it only for the
candidate. Compare complete timing and setup, profiling counts, Gaussian/items
trajectories, whole-scope memory and the same held-out PSNR. Kernel classifications
and nested CPU ranges must not be counted twice.

Normal completion saves a uniquely named `.training.<id>.json` beside the selected
output model path and logs its filename. It records actual selections, the last
scalar-loss step, complete preparation/training times, stage names, cumulative
milliseconds/sample counts, ADC scale trajectory, device/build identity and shader fingerprints. Report
writing, model export and external evaluation occur outside the training timer.
The configuration describes the last Start/Resume snapshot; keep settings fixed
throughout a fresh run for an independent A/B comparison.

259.929 seconds versus 227.931 seconds is a cross-implementation observation of
31.998 seconds, not a promised saving. No speed or quality gain is assumed from
source similarity, compilation or synthetic checks. Full native Tensor lifecycle
and performance must be checked on the target RTX3090.
