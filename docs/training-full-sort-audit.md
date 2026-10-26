# Complete sorting-chain comparison

`training_range_keys_gpu_tests --full-sort-chain` compares two private GPU list
producers from identical frozen projected Gaussians. It does not select a
training variant or advance training. Without this flag, the runner retains its
[independent range-key comparison](training-range-key-audit.md).

The candidate follows the separate-sorting data flow in
[Faster-GS §3.2 and Appendix B.2](https://arxiv.org/html/2602.09999v1), with explicit
adaptations for the current Vulkan renderer. This is a comparison of a complete
sorting package, not an attribution of savings to one component or a reproduction
of the paper's whole training system.

## Compared work

Both paths recompute coverage counts on GPU with the original FP32 geometry and
run the original Gaussian prefix. Both clear ranges and copy the item count to
the host in the prepare submission. The captured counts and lists are correctness
references; they are not uploaded as free producer inputs.

| Original | Candidate |
| --- | --- |
| Original count and prefix | Original count and prefix, participation flags, stable prefix compaction, scatter depth keys and original IDs |
| Read back item count `I` | Read back `I` and participating count `Nv` in the same submission/fence |
| Original emit in Gaussian order | Stable depth sort of `Nv` Gaussians, gather counts in that order, reconstruct prefix, emit tile keys with original Gaussian IDs |
| 32-bit depth sort of `I` items | No repeated item depth sort |
| Gather tile keys through item permutation | No high-key gather |
| 32-bit tile key/value sort | Effective-width tile key/value sort with Gaussian IDs as payload |
| Gather final Gaussian IDs | No final ID gather |
| Indirect-key boundaries and original range finalization | Direct sorted-key boundaries and original range finalization |

`Nv` means Gaussians whose actual tile count is nonzero. Stable compaction preserves
original Gaussian order before depth sorting, including equal depth keys. It uses
a flag prefix rather than the upstream atomic append and charges that work to the
candidate. The original Gaussian prefix is retained to obtain `I` using the same
counter contract as the reference; this is a conservative Vulkan adaptation.
Original IDs, geometry, depth-key bit transform and permanent storage are unchanged.
Nonfinite depth keys are not cleaned or quantized.

Tile keys occupy **uint32 storage**. Grids with at most 65,536 tiles use two byte
radix passes; larger grids use the original four-pass width. This is a 16/32
effective-width adaptation, not physical uint16 packing or exact arbitrary-bit
CUB dispatch. Both even pass counts finish in the input key/value buffers.
The effective-width recorder uses the existing sorter's shader pipelines and workspace;
it does not alter the submodule or its public sort API. It retains the
four-histogram initialization cost, including for two-pass sorting.

Candidate emit itself writes the direct-key buffer for a one-item list, so that
case does not depend on unproduced radix scratch. Zero items never read a key.
Separate private counter banks hold `I`, `Nv` and the reordered prefix total;
unrelated adaptive counter fields are not reused.

## Target-device commands

Run from the repository root on the RTX 3090. Windows appends `.exe`. Report paths
must be new. Use the matching scanline and raw input files from the original
renderer; the capture's name does not select a scanline optimization.

```text
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --synthetic --report full_sort_edges.json --require-validation
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --capture frame_0_iter_8037.scanline.bin --inputs sort_baseline_inputs.bin --report full_sort_correctness.json --require-validation
```

Only after both validated GPU commands succeed, run one short timing comparison:

```text
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --capture frame_0_iter_8037.scanline.bin --inputs sort_baseline_inputs.bin --report full_sort_timing_preheated.json --timing --no-validation
```

`--require-validation` requires both core and synchronization validation. Timing
requires explicit `--no-validation` and real binary inputs; synthetic timing is
rejected. `--device-index N` selects a device. External Vulkan loader overrides
can still affect layers; keep them identical and record them when comparing runs.
Return code zero and `PRIVATE_CORRECTNESS_PASSED_NOT_TRAINING_ADMISSION` describe
local correctness, not performance admission. A failure returns nonzero and
writes `STOP`; preserve failed reports rather than replacing them.

## Correctness and limits

The raw file is first checked against the captured projected data and final
list/ranges. A/B/A then executes the actual producers and checks exact final
Gaussian IDs, all range components, original per-Gaussian counts/prefixes,
candidate sorted visible IDs/depth keys and reordered prefixes. Readonly input,
unused buffers, tails, counter fields, prefix/radix scratch and output guards are
checked. A recomputed coverage count differing from the capture stops the test
before either candidate sort; the runner never repairs counts to match a GPU.

Synthetic GPU cases cover zero/one, all-inactive Gaussians, multi-tile splats,
equal depths, signed zero, nonfinite depth-key encoding, workgroup/radix partition
edges, a 65,537-Gaussian hierarchical prefix, capacity greater than active count,
and the 65,536/65,537-tile radix-width boundary. CPU tests separately compare
the stable full-chain oracle and width boundary. Device/workload limits, host
memory and device-memory headroom are checked before allocating private buffers.

The sequence excludes projection, composite, loss, backward, Adam, ADC and live
capacity growth. It cannot establish downstream gradient/multi-step equivalence,
lifecycle safety or full-training speed. Different GPUs or compiler flags can
produce different FP32 coverage decisions; a mismatch remains unresolved until
investigated on the matched target setup.

## Timing and decision

The full-chain timing protocol is `fixed_preconditioning_32_aba_2w8_v1`.
After correctness verification and its downloads, the runner executes exactly
16 alternating A/B pairs (32 full invocations) before A1. This count is fixed;
it never waits or retries until a favorable sample appears. All 32 timings are
retained in `preconditioning_samples` with the candidate flag. They are excluded
from the formal timing statistics by protocol, regardless of their values.

Each A1/B/A2 phase then has two warmups and eight retained measured samples.
Its two warmup results are saved in `warmup_samples`; all 24 measured samples
remain in `samples` and determine the original acceptance checks. Fixed warming
does not guarantee stable device conditions. An unstable measured phase still
produces an inconclusive result; measured samples are never discarded.

Every invocation restores private workspaces in a separate fenced submission.
No full-buffer verification downloads occur between preconditioning and A1 or
between measured phases. Restore, allocation,
upload, verification downloads and timestamp-query retrieval are outside the
sequence timing. Restore submit/wait is reported separately; restoring both
paths' private workspace perturbs caches and does not model production allocation.

| Metric | Scope |
| --- | --- |
| `prepare_ms` | Clear, count, prefix, candidate compaction, count copies and host-read barrier |
| `main_ms` | Sort/emit/prefix/gathers as applicable, boundaries and finalization |
| `gpu_active_ms` | `prepare_ms + main_ms`; includes all candidate GPU producers |
| `gpu_span_ms` | Prepare start through main end, including the CPU scheduling/readback gap |
| `cpu_sequence_ms` | Prepare/main recording, submissions, original fence waits and count checks; stops after main fence |
| `ranges_ms` | Boundary construction and finalization; descriptive only |

Reports retain device UUID/driver, build/compiler/shader identities, both source
file fingerprints and all samples. The GPU active, GPU span and CPU sequence each
receive the declared 20% sample-variation, 2% baseline-drift and 5% gain checks,
plus a 0.10 ms absolute engineering target. All three must be inspected; GPU
active savings alone do not prove a scheduling/wall-time benefit. Non-3090 timing
is marked `NON_TARGET_DEVICE_NOT_ADMISSION`. A passing short comparison still
requires downstream and lifecycle review before any production switch or
full-training run. No independent range-only gain is added to this package's
measured gain.

## Private downstream replay

`--full-sort-chain --downstream --require-validation` checks actual GPU-produced
lists with the original Fixed16 Workgroup Shared composite and original Tensor
Subgroup/Global backward shaders. It is a correctness diagnostic with fixed
captured pixel gradients, not a timing run. It requires buffer float32 atomic
add, native subgroup32, the original Tensor capability gate, and core plus
synchronization validation. There is no automatic backward fallback.

The runner executes A1/B/A2. Each invocation rebuilds the complete list on GPU
and records the two consumers in the same main command buffer, directly binding
the producer's projected, final-ID and range buffers. No CPU-sorted list is
uploaded. The existing range-finalize dependency precedes the consumers; a
transfer/compute barrier covers output initialization and producer writes, a
compute barrier makes freshly produced blend states visible to backward, and
output dependencies cover verification downloads. Workspaces are reset before
each invocation. All references remain private and no worker is advanced.

For the RTX 3090, build Release and run from the repository root:

```text
cmake --build build --config Release
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --downstream --synthetic --require-validation --report full_sort_downstream_edges.json
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --downstream --capture build/bin/Release/output/benchmark_scanline/frame_0_iter_8037.scanline.bin --inputs sort_baseline_inputs.bin --require-validation --report full_sort_downstream.json
```

Both report names must be new. Use the actual matching capture location if it
differs. `--downstream` rejects `--timing` and `--no-validation`. A passing report
has schema `faster_sort_downstream_gpu_v1` and status
`PRIVATE_DOWNSTREAM_PASSED_NOT_TRAINING_ADMISSION`. A numerical mismatch preserves
all comparisons in `STOP_DOWNSTREAM_NUMERICAL_MISMATCH` and returns nonzero;
other failures write `STOP` when the report path has been reserved.

RGBA and all blend fields must be bitwise identical between A1/B, A2/B and the
original A1/A2 repeat. Real captures additionally require every produced blend
to match the captured original blend exactly. Projected-gradient comparisons
reuse `compareRowGradients`: per-component absolute tolerance `1e-5` plus relative
tolerance `1e-3`, finite values, zero padding and identical Gaussian counts. All
three comparisons use that same contract; baseline repeat noise never widens it.
Reports retain the differences, relative L2, cosine and maximum-error location,
as well as readonly, tail and canary checks. Logical producer/consumer buffer
bytes are reported; native peak VRAM remains unmeasured. The diagnostic holds
reference and candidate workspace together, so these bytes are not an estimate
of production memory overhead.

Synthetic downstream cases have a 31x15 partial-edge image, zero/one/multiple
Gaussians, inactive Gaussians, tied depths and multi-tile splats. Their pixel
gradients are deterministic fixture inputs; their colors and blends are produced
on GPU, with no historical blend reference. On devices that cannot run the native
Tensor path, explicit `--portable-reference` is allowed only with
`--downstream --synthetic`. It uses the original Direct backward shader, emits
`PRIVATE_PORTABLE_FIXTURE_PASSED_NOT_TENSOR_ADMISSION`, and cannot validate the
3090 Tensor path or serve as a real-capture fallback.

The downstream replay uses the original consumer shaders. This replay does not compute
loss/SSIM or pixel gradients, project gradients back to 3D, update Adam or execute
ADC. It establishes local downstream equivalence and diagnostic dependencies;
live capacity growth, worker integration and training lifecycle still require
their own bounded checks. It does not establish a combined downstream timing or
full-training gain.

## Default-off renderer integration

The experimental `GaussianForwardRenderer` can use the same sorting chain on
its actual `TrainingBuffers`. `VULKAN_3DGS_FASTER_SORT=1` requests it when the
renderer is initialized; unset or `0` selects original sorting. Other values
are rejected. This environment setting is not a persisted UI setting. A process
restart/model reinitialization is needed to read a changed environment value.
The startup/transition log reports `requested`, `active` and `tile_size`.

Only Fixed16 activates the candidate. Fixed8 and Adaptive Spatial retain their
original paths and report the fallback. The default path allocates no candidate
workspace. An opt-in workspace is owned by the renderer, grows with Gaussian
count, rebinds current buffers at prepare/main boundaries and is released by
renderer cleanup. Permanent Gaussian IDs, parameters, moments and the original
Gaussian prefix/count buffer keep their original meanings.

Preparation retains original projection/count/prefix, then records flag-prefix
compaction and a separate private `Nv` readback. That readback uses the original
prepare submission/fence; main consumes it only after the fence. Main sorts
`Nv`, rebuilds the ordered prefix, emits direct IDs, sorts tile keys and builds
ranges in the canonical renderer output buffers. Alignment/capacity checks cover
the key/value arrays; radix storage is sized for the larger possible sort and
reused with explicit transfer/compute dependencies. The one-call `forward()` API
rejects this mode because it has no prepare fence; live training uses
`TrainingStepExecutor`'s existing two submissions.

Profiling includes original prefix plus compaction/readback in `Tile prefix`.
The candidate's main sorting chain, including its ordered prefix and emit, is
included in `Tile sort/ranges`; it has no separate `Tile emit` sample. Allocation
and CPU count handling remain in CPU record/step timings. GPU shader lists alone
do not establish whole-step or whole-training savings.

Before requesting live training, validate the actual renderer path on private
buffers:

```text
cmake --build build --config Release
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --integrated-sort --synthetic --require-validation --report faster_sort_integration_edges.json
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --integrated-sort --capture build/bin/Release/output/benchmark_scanline/frame_0_iter_8037.scanline.bin --inputs sort_baseline_inputs.bin --require-validation --report faster_sort_integration.json
```

This mode requires validation and rejects timing/downstream combinations. Its
schema is `faster_sort_integration_gpu_v1`; a successful status is
`PRIVATE_RENDERER_INTEGRATION_PASSED_NOT_LIVE_TRAINING_ADMISSION`. Both old and
new renderer paths execute on the same private buffers in A/B/A order. Results
check actual path selection, `Nv`, IDs, ranges, readonly projected input, color
and blend equality. Real inputs additionally check the captured blend.

The synthetic sequence changes Gaussian counts through 1,257,33,0,513,2 on one
renderer, exercises buffer growth, nonzero-count buffer adoption, zero-count
reset and repeated original/candidate switching. Buffer adoption here checks
workspace/descriptor lifecycle with uploaded projected input; it does not execute
ADC clone/split/prune or establish Adam-state equivalence. No worker, loss,
backward, optimizer or real training is run by this mode. Bounded live lifecycle
and whole-step performance/quality checks remain outstanding.
