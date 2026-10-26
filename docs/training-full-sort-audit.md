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
The private recorder uses the existing sorter's shader pipelines and workspace;
it does not alter the submodule or the production sort API. It retains the
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
build/bin/Release/training_range_keys_gpu_tests --full-sort-chain --capture frame_0_iter_8037.scanline.bin --inputs sort_baseline_inputs.bin --report full_sort_timing.json --timing --no-validation
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

Each A1/B/A2 phase has two warmups and eight retained samples. Every invocation
restores private workspaces in a separate fenced submission. No full-buffer
verification downloads occur between measured phases. Restore, allocation,
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
