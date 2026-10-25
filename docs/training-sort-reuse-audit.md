# Depth-sort reuse CPU audit

`tests/training_sort_reuse_audit.py` is an offline, standard-library-only
diagnostic. It reads an existing `V3DSCN01` Fixed16 Tensor Subgroup/Global capture
and does not create a Vulkan device, update training state or enable a training
optimization.

The mechanism under examination is sorting participating Gaussian IDs by depth
before expanding their tile memberships, then stably sorting the expanded list
by tile. Its reference is [Faster-GS, Appendix B.2](https://arxiv.org/html/2602.09999v1).
The [renderer](../src/training/core/gaussian_forward_renderer.cpp) already uses
two stable 32-bit sorts; this diagnostic concerns reducing the first sort's
input length, not introducing the existing two-sort scheme.

## Run

From the repository root, with an existing capture and a new output path:

```text
python -B tests/training_sort_reuse_audit.py frame_0_iter_8037.scanline.bin --report sort_reuse_audit.json
python -B -m unittest discover -s tests -p "test_*.py"
```

The report must not exist. The tool refuses to overwrite the input or an earlier
report. It does not create output directories. The default input limit is
256 MiB; `--max-capture-mib` changes that CPU input limit, not a GPU memory limit
or a guarantee about peak Python memory.

`--item-capacity` supplies a hypothetical tile-array capacity. Without it, the
memory model uses the active item count as a minimum-capacity scenario; the
capture does not contain production capacity. `--alignment` supplies an explicit
sort-storage alignment scenario, defaulting to 256 bytes. Neither option
establishes the target device's actual allocation.

## Oracle contract

The reader checks the header, exact payload length, tile-range partition,
Fixed16 range padding, Gaussian IDs and uniqueness of a Gaussian within a tile.
It extracts the original depth bits without float conversion, including signed
zero and exceptional bit patterns. This does not authorize exceptional values
in production; it prevents the diagnostic from silently rewriting its inputs.

Two independent ordering procedures use the same captured memberships:

1. Emit in original Gaussian-ID order, stably sort all items by depth, then
   stably sort them by tile.
2. Stably compact participating Gaussian IDs, sort those IDs by depth, recompute
   expansion offsets, emit their memberships, then stably sort items by tile.

Both final ID lists and all four components of every tile range must match the
capture exactly. Stable compaction preserves original-ID ordering for equal
depth keys; an unordered atomic append followed by a stable sort does not.
Per-Gaussian tile enumeration is normalized to ascending tile IDs when reading
the capture. Since each Gaussian occurs at most once in a tile, that
normalization cannot change the final per-tile tie order.

This is a membership-replay oracle. It does not independently recompute
coverage geometry, original Gaussian counts, GPU emit output or GPU radix
behavior. Matching outputs do not establish shader, synchronization, gradient
or training correctness. The report retains input and source SHA-256 hashes;
UUID, compiled module and compiler metadata absent from the capture remain null.

## Conditional cost model

The report distinguishes source-level logical payload from native memory
transactions and measured time. For the current library, each 32-bit sort has
four radix passes and twelve compute dispatches. Reducing the input length
reduces partition workgroups, not those fixed dispatches, when both lists have
more than one element.

The additional-work scenario retains the original count/prefix path, adds a
hierarchical stable-rank scan and scatter, gathers counts in depth order, and
scans those counts before emit. It removes the original high-key gather. It
assumes a private visible-count readback can share the existing prepare
submission. This scenario is a budget, not an implemented GPU path; no existing
counter field, submission order or production buffer is changed by this tool.

The logical-payload model is explicit:

- A sort contributes `80N` core bytes: per radix pass, an upsweep key read of
  `4N` and downsweep key/value reads and writes of `16N`.
- Removed high-key gather contributes `12I`; removed emit depth-key stores
  contribute `4I`.
- Added stable scan/scatter contributes `28G + 12Nv`; depth-ordered count
  gathering contributes `16Nv`, its prefix contributes `20Nv`, and extra emit
  ID indirection contributes `4Nv`. Together these are `28G + 52Nv`.
- Histograms, hierarchical scratch traffic, shared operations, barriers,
  transaction amplification, caches and downstream changes are not included in
  those core-byte totals. They remain costs to measure, not zero-valued costs.

For memory, the no-alias scenario reserves rank, depth key, ID and sorted
offset/count arrays totaling `20G` bytes, additional prefix scratch and a private
16-byte count. Existing tile arrays remain available for fallback. Sequential
sort scratch can be shared because `Nv <= I`; this does not describe the whole
application's VRAM or its allocation headroom.

The time sensitivity table uses an explicitly assumed depth-sort share of a
0.78 ms historical sort/ranges interval and linear size scaling. Those are
unmatched scenarios, not measurements or guaranteed upper bounds. Fixed costs
and different workload states can invalidate the extrapolation. The tool never
converts its byte reduction into measured milliseconds or full-training speedup.

`CPU_ORACLE_PASSED_BUDGET_CONDITIONAL_NO_GPU_ADMISSION` means that membership
replay passed but the net performance budget remains conditional. A zero exit
code is not permission to enable a GPU candidate. Missing production count/emit
inputs and matching RTX3090 timing must be resolved before claiming that the
runtime/performance gate passed; downstream comparison also requires matching
target and optimizer state.

## Original-renderer GPU replay

`training_sort_baseline_gpu_tests` is an explicit standalone diagnostic, not a
GPU CTest or a candidate implementation. It uploads captured projected values
to private `TrainingBuffers` and calls the existing renderer's count, prefix,
emit, two radix sorts, gathers and range construction. Normal training leaves
its optional sort query pool unset and retains the original shader sequence.

The runner allocates the normal training workspace but does not dispatch
projection, composite, loss, backward, Adam or ADC. It requires measurable
`VK_EXT_memory_budget` headroom and can reject a low-memory device before
allocating that workspace. Its initial tile capacity follows `TrainingBuffers`
allocation and grows if needed. `replay_capacity` describes that private
allocation; `historical_live_capacity` remains null. The capture cannot recover
the old live allocation or historical resize latency.

After building Release, run these commands on the RTX3090 host from the
repository root. Windows uses the same paths with `.exe` appended.

First collect correctness evidence and original renderer inputs:

```text
build/test/Release/training_sort_baseline_gpu_tests --capture frame_0_iter_8037.scanline.bin --report sort_baseline_correctness.json --inputs sort_baseline_inputs.bin --require-validation
```

This command requires core and synchronization validation. If those are
unavailable, it writes `STOP` and exits unsuccessfully; do not treat a run
without them as having passed that gate. Use a new report/output name on a
later invocation. Output files are reserved exclusively before GPU work, so a
failed run can leave an empty binary; discard its evidence when the JSON says
`STOP` or the process fails.

Only after correctness succeeds, collect separate baseline timings:

```text
build/test/Release/training_sort_baseline_gpu_tests --capture frame_0_iter_8037.scanline.bin --report sort_baseline_timing.json --timing --no-validation
```

The timing run rechecks lists outside the measured samples. It retains all
24 segmented baseline samples (three sets of two warmups plus eight measured
samples) and eight further samples without internal sort timestamps (two
additional warmups). These are A/A/A baselines, not candidate A/B/A results.
The final set still has boundary timestamps for prepare/main totals.
Review baseline drift, within-set variation and instrumentation effects before
using the segmented depth share in a budget; no retry-until-fast or candidate
admission is implied by successful collection.

Samples include count/clear, prefix, emit, depth sort, high-key gather, tile sort,
item gather and ranges, plus prepare/main totals and CPU submission/wait costs.
`gpu_span_including_host_gap_ms` crosses the prepare fence and host recording
gap; it is not a sum of shader execution times. Mutable input/work arrays are
cleared in a separately fenced restore outside each sample. Downloads and
verification stay outside timing. Restore cache effects, warmed allocation and
the absence of capacity growth limit comparison with live training.

Device UUID, driver, build/compiler, shader fingerprints and actual validation
state are in the report. Source capture fingerprints identify the input, not a
matching historical module or full training checkpoint. No whole-step or
training speedup is reported.

### Raw input format and CPU verification

`V3DSRT01` version1 has a 104-byte little-endian header: eight magic bytes,
uint32 version and replay capacity, then the exact 88-byte source scanline
header. Payload order is:

| Block | Element count | Bytes per element |
| --- | --- | --- |
| Projected Gaussian | G | 48 |
| Original Gaussian prefix offset/count | G | 8 |
| Original unsorted Gaussian IDs | I | 4 |
| Original unsorted depth keys | I | 4 |
| Original unsorted tile keys | I | 4 |
| Original sort index seeds | I | 4 |

The runner verifies original prefixes, emit ID/depth/seed fields, CPU-stably
sorted output, GPU-sorted output and readonly projected values against the
capture before reporting success. The Python tool additionally compares each
Gaussian's emitted tile-membership set:

```text
python -B tests/training_sort_reuse_audit.py frame_0_iter_8037.scanline.bin --sort-input sort_baseline_inputs.bin --report sort_reuse_with_inputs.json
```

With `--sort-input`, the memory model uses that file's replay capacity unless
an explicit `--item-capacity` scenario overrides it. This still does not recover
historical live capacity. Sync the two GPU JSON files and the binary back for
evaluation; raw captures are diagnostic inputs, not source files to commit.
