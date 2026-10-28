# Physical Z-ordering in training

**Training → Faster-GS Z-ordering (Experimental)** selects a default-off physical
Morton reorder. It is independent of Faster-GS Sorting and is saved as
`morton_ordering`. Workers lock the selection. A paused change applies on resume;
disabling it stops future reorders without restoring the previous model order.
Use a fresh model for an independent comparison.

## Method and implementation boundary

The method source is [Faster-GS §3.3](https://arxiv.org/html/2602.09999v1#S3.SS3).
The implementation references are fixed to:

- Faster-GS [`c60ea269b7cf3ebda527d640b6fdd31462f91b41`](https://github.com/nerficg-project/faster-gaussian-splatting/tree/c60ea269b7cf3ebda527d640b6fdd31462f91b41): model permutation and training configuration.
- NeRFICG [`623ad16ec6e05a5c5cedacc046273b63eac2686e`](https://github.com/nerficg-project/nerficg/tree/623ad16ec6e05a5c5cedacc046273b63eac2686e): Morton encoding and callback dispatch.

The GPU computes the current centers' axis-aligned bounding box. Coordinates are
normalized using the longest box side, clamped to [0,1], and quantized by
truncation with factor 2097151. Each axis contributes 21 bits to the 63-bit Morton
code. Two stable 32-bit radix sorts preserve the complete code: low bits first,
then high bits gathered through the resulting indices. Equal Morton codes retain
input order; PyTorch's unspecified equal-code order is not reproduced bitwise.

Parameters, both Adam moments and ADC statistics use the same permutation.
Temporary output buffers retain the active buffers' capacities and queue sharing.
Their inactive tails are copied unchanged; the live prefix is gathered. Only after
the submission completes and the position-status check passes are all three
arrays adopted together. Transient projection, tile and gradient inputs are
rebuilt by the following training step. Dense Adam and its original step schedule
remain in use. No persistent identity correction or sparse update is applied.

The fixed events occur after 0, 5000, 10000 and 15000 completed steps, before the
next forward pass: displayed steps 1, 5001, 10001 and 15001. Existing ADC/reset
scheduling is retained. Fixed Benchmark never triggers a reorder. Model reset
clears event history; renderer rebuild retains it. Resuming cannot repeat an
already completed event at the same step boundary.

This reproduces the physical locality mechanism within this Vulkan pipeline, not
the entire CUDA training system. Physical index changes can change split random
draws, depth ties and floating-point accumulation, so full-training trajectories
are not required to match the unpermuted baseline. State copying itself remains
bitwise exact. Closing the gap to the CUDA pipeline's backward kernel, allocator
or post-training cleanup is outside this switch.

## Cost and failures

UI and logs report event count, the latest boundary, Gaussian count and CPU/GPU
duration. CPU profiling includes allocation, recording, submission, waiting,
adoption and temporary-buffer release. GPU profiling covers copies, bounding
reduction, encoding, both sorts, gather and status readback. These costs remain
inside the existing total training timer. The reported workspace byte count is
logical temporary-buffer storage, not measured native peak VRAM.

The Z-order profiling rows average per event, with at most four samples in a
complete run. Their amortized cost per step is average event time multiplied by
event count and divided by completed steps; do not add the event average directly
to ordinary per-step averages.

Parameters, moments and ADC statistics alone require about 736 bytes per allocated
Gaussian for the temporary state; codes, keys, indices and sorter scratch add to
that. Capacity padding and a larger ADC workspace can increase the actual total.
No per-step identity lookup is added between reorder events.

Allocation/device-limit failures raise an error with the original model still
owned. Non-finite centers or an overflowing bounding extent are rejected before
adoption. Empty input does not dispatch; coincident finite centers receive code
zero. Device/queue failure aborts execution rather than claiming a successful
reorder or silently selecting the baseline.

## Correctness checks

The existing optimizer GPU diagnostic includes a correctness-only mode:

```bash
build/test/Release/training_optimizer_gpu_tests --morton-only --gpu "RTX 3090" \
  --require-validation --report morton_correctness.json
```

Append `.exe` on Windows. `--gpu` uses the application's existing name/index/UUID
selector; omitting it uses the first enumerated GPU. Use a new report filename.
The command requires actual
core/synchronization validation, push descriptors and float32 atomic add for its
synthetic downstream checks. It reports the actual selected device; a local AMD
result cannot establish RTX 3090 performance or native Tensor behavior.

Checks include 0/1 and workgroup boundaries, repeated positions/codes, capacity
growth/shrink and ADC buffer adoption, recognizable nonzero moments/statistics,
inverse permutation, inactive tails, non-finite rejection, scheduling policy and
synthetic owner steps through forward/backward, ADC, opacity reset and SH changes.
They do not load a real dataset or establish held-out quality. The diagnostic has
no timing-admission mode. Existing smoke reports record actual reorder events;
a smoke ending at step 3200 covers only the initial event. Step 15001 is required
to observe all four events in real training.

After target-device correctness/lifecycle checks, use the [UI A/B procedure](training-ui.md)
for independent 30000-step runs. Hold all settings, including Faster-GS Sorting,
constant and toggle only Z-ordering. Save total time, complete profiling, final
Gaussian count, PLY files and matched held-out PSNR. Single-run loss or a faster
reorder dispatch alone does not establish a training improvement.
