# S4-3 scheduling feasibility diagnostics

This is an audit, not GPU-driven training. Both original prepare/main submissions,
fences, CPU count readback, capacity growth, sort order and training algorithms
remain unchanged. No existing optimizer/culling experiment is promoted.

## Capture bounded ordinary-training timing

From the repository root on the target machine, build Release with your existing
configuration. Do not change shader debug/compiler settings between comparisons.
Linux example:

```bash
cmake --build build --config Release
(
  export VULKAN_3DGS_S4_SCHEDULING_REPORT="$PWD/s4_scheduling_baseline_v2.json"
  export VULKAN_3DGS_S4_SCHEDULING_START=8000
  export VULKAN_3DGS_S4_SCHEDULING_SAMPLES=128
  build/bin/Release/vulkan-3dgs --validation on
)
```

Start normal/Pure Training in the UI using the existing baseline configuration,
not Fixed Benchmark and not the optimizer replay executable. Sample collection
starts at the requested one-based training iteration. The example deliberately
uses a middle-training window; reaching iteration 8000 still requires actual
training, not a PLY-based optimizer-state resume. No complete 30000-step run is
required. Start/sample variables default to 0/128; samples are limited to 1–2048.
On Windows use the same environment names via `$env:NAME` and the `.exe` path.

The log `S4-3 scheduling diagnostic report complete` indicates the bounded report
has been written. Training continues; pause it normally if no more steps are
needed. The output is reserved exclusively when the training profiler initializes,
so file existence alone does not mean completion. Existing files are never
overwritten. Choose a fresh path for another initialization. Pausing early does
not finalize the sampler; destroying the training profiler/application writes
`PARTIAL`, which is not an admission result. Environment variables are startup
controls, not live UI settings. The subshell prevents leaking them to later runs.

Without the report environment variable, there are no extra query pools,
timestamp commands, per-step report samples or output. When armed and within
the selected window, three extra timestamp queries are recorded: before prepare
work, after the prepare readback copy and before main commands. Query results are retrieved
only after the existing main fence; no new GPU waits or full model downloads
are introduced. The probe does not enable Vulkan validation itself.

The JSON records per-step iteration/frame/G, tile item count/capacity, extent,
SH and rendering modes, upload timeline value, ADC/validation/reset/resize/capture
flags, and:

- CPU prepare queue-submit call and fence-wait durations separately;
- count-read and capacity-check times, with resize nested in capacity-check;
- CPU post-prepare-fence to main-submit interval, with main recording nested;
- main queue-submit call duration;
- same-queue GPU prepare duration and prepare-to-main gap, independently null
  when their endpoints are unavailable.

Timing reports now use `s4_scheduling_timing_v2`. Runtime identity records
`cpp_build_config`, `cpp_compiler`, `cpp_ndebug_defined`,
`cpp_debug_macro_defined`, `logger_level_at_initialization` and
`vulkan_validation_layers_enabled`; each sample also records `logger_level`.
Build configuration comes from CMake, macros from the compiled core, and logger
level from the running logger. These fields do not reconstruct arbitrary custom
compiler optimization flags. Shader compiler/debug information remains separate.

The application passes its Vulkan instance's validation-layer configuration
after the availability check through the controller to the sampler. A caller
that does not supply this information records null, not false. This describes
the application's instance configuration, not externally forced loader/layer
overrides. Per-step `validation` still means **training numeric validation**, not
Vulkan validation layers. The probe changes neither validation nor logging
settings; do not assume Release means validation is disabled. Historical v1
reports lack these fields and GPU prepare duration; keep them as historical
evidence rather than filling missing values with guesses.

CPU and GPU clocks are not subtracted from one another. The GPU interval is an
instrumented opportunity upper bound, **not pure GPU idle or guaranteed savings**:
it may include queue/device work or upload dependencies. CPU fence wait includes
required GPU calculation and must not be counted as removable time. Category
groups overlap; `ordinary` excludes ADC, validation, reset, resize, optimizer
capture and optimizer-disabled steps. All raw rows remain available. Final JSON
serialization/I/O is outside GPU intervals but may affect that step’s CPU Total.
Diagnostic write failures disable the probe without preventing iteration advance
after a completed Adam step. An incomplete/failed output is not success.

GPU prepare duration includes the original prepare work, barriers, profiling
commands and count-copy operation, but not CPU fence wait. For a future attempt
to record main commands while prepare executes, the smaller of same-step GPU
prepare duration and CPU main-record duration is only an optimistic overlap
ceiling. Real overlap also depends on submit/record alignment, count-independent
recording and validation overhead. Merely recording both command buffers before
one combined submit does not hide CPU recording behind GPU work. This probe has
`overlap_measured:false` and does not implement or demonstrate overlap.

## Validate the independent sort probe

Linux commands; Windows adds `.exe`. Every report name must be new:

```bash
build/test/Release/training_scheduling_gpu_tests --synthetic --require-validation --report s4_scheduling_synthetic.json
```

This checks eight small cases: empty/one element, 255/256/257 boundaries,
multi-partition tails, capacity slack and all-equal keys. A stable CPU oracle
checks exact final IDs and ranges, including equal-key tie order. It also checks
unused tails, read-only seeds, GPU count and scratch/range canaries after A/B/A.
The command additionally writes `s4_scheduling_synthetic.timing_smoke.json` for
two private-buffer GPU-work samples with all three timestamps. It checks query
availability, runtime metadata and unchanged buffer contents. That file is
explicitly `synthetic:true`; it tests the sampler, not the actual training worker
or an opportunity budget.

The tool requires a compute queue supporting group512, push descriptors, bounded
storage and measurable memory headroom. It does not impose the old Tensor
subgroup32/shared-memory gate. `--require-validation` requires both core and
synchronization validation; missing layers or validation errors reject the run.
`--device-index N` selects the physical GPU; match its recorded UUID to the
training timing report instead of comparing different cards by model name.

## Measure direct versus GPU-count sorting cost

Use a representative `tile_items` / `tile_item_capacity` pair from the ordinary
training report. These are NOT Gaussian count or Gaussian capacity. For example
only, replace the numbers below with the observed pair:

```bash
build/test/Release/training_scheduling_gpu_tests --count 1500000 --capacity 3000000 --require-validation --report s4_scheduling_sort_cost.json
```

No optimizer/scanline binary is needed. Inputs are deterministic synthetic
tile/depth keys, not an actual renderer capture. Capacity is limited to 4194304
elements; out-of-budget inputs are rejected rather than silently sampled.

A1/B/A2 measures 2 warmup and 8 measured dispatches per phase, preserving every
sample. Both variants use the same seed, full restore and private resource set.
Restore is separately fenced/timed and excluded from the stage; its cache
perturbation is disclosed. There are no output comparisons between timed phases.
The stage includes two 32-bit radix sorts, high-key gather and start/count range
construction. It does not include final item gather, emit, forward or backward.
Zero/one direct sorts use the production host bypass; those tiny synthetic cases
are correctness checks, not meaningful training timing substitutes.

The library indirect API reads count from a GPU-written buffer but still launches
sort partitions using `maxElementCount`. The B helper gather/range dispatches
also use capacity bounds with GPU count guards. Thus this probe evaluates a
bounded-capacity implementation, NOT a completed GPU-generated dispatch-argument
path. A future actual-count indirect helper needs its own generation/sync costs.
The synthetic GPU count is produced in setup, outside the timed stage, just as
the normal prefix already produces a count; the tool never reads it to size B.

JSON includes library/header and helper shader fingerprints, exact counts,
launched partition counts, all stage/restore samples, baseline drift and
`indirect_incremental_stage_ms = B - mean(A1,A2)`. Positive means added cost.
Max/min above 1.20 or baseline drift above 2% makes the cost sample inconclusive;
the tool does not retry/drop samples. A stable sample remains only a cost probe,
not evidence of GPU-driven training speedup.

## Decide whether to implement

The target v2 baseline and a separate validation-off run are now available as
`s4_scheduling_baseline_v2.json` and `s4_scheduling_validation_off.json`. Both are
complete 128-step windows (126 ordinary steps) on the same 3090 UUID/driver,
Release/GNU 11.4.0, INFO logging and Slang 2026.7.1/debug ON. Ordinary mean CPU
main recording is 1.357331 versus 0.211023 ms; GPU prepare-to-main gap is 1.782588
versus 0.408167 ms, while GPU prepare is 0.955191 versus 0.940075 ms. Model states
are not identical: this supports substantial validation overhead but is not a
same-state paired timing or a full-training speedup measurement.

Validation is now controlled without source edits by `--validation on|off`
(default on for both builds). Use a new report filename and append
`--validation off` to the capture command for an intentional off-layer sample;
verify `vulkan_validation_layers_enabled:false`. Restore any old manual source
edit by syncing the new implementation, not by keeping a hard-coded false.
Correctness checks should keep validation enabled. No existing same-GPU sort
boundary or cost probe needs repeating solely for this startup option.

Under the validation-off baseline, the same-step optimistic recording-overlap
ceiling is only 0.211023 ms. The entire observed 0.408167 ms gap is not all
guaranteed removable idle. Defer GPU-driven/overlap integration while establishing
the normal-training performance baseline; do not reuse validation-on timing to
inflate the new opportunity budget.

Compare the ordinary-step gap and potential overlap ceiling with the measured incremental sort cost,
then account for still-unmeasured emit/count consumers, dispatch argument
generation, barriers, capacity/overflow checks and fallback. Synthetic cost and
instrumented gap are different measurements, not directly additive measured
training gains. If the budget is inadequate, stop. Do not remove fences first,
sort an uninitialized capacity tail, or proceed to full training to rescue a
negative budget. Only a separately authorized Fixed16 ordinary-step prototype
should follow a credible budget, preserving stable depth/tile order and safe
no-write overflow recovery before expanding lifecycle coverage.

## Establish the full-training baseline

Build Release with the existing shader flags, then run without the bounded
scheduling probe (Linux, from the repository root):

```bash
cmake --build build --config Release
(
  unset VULKAN_3DGS_S4_SCHEDULING_REPORT
  unset VULKAN_3DGS_S4_SCHEDULING_START
  unset VULKAN_3DGS_S4_SCHEDULING_SAMPLES
  build/bin/Release/vulkan-3dgs --validation off
)
```

Confirm the startup log contains `requested=false, enabled=false`. Do not run
under Nsight during this timing run. Keep the same dataset/downscale, modes,
training numeric checks and shader compiler/debug flags; select `3DGS Random`
and Pure Training for the 30000-step run. This switch disables application-wide
Vulkan layer requests, not just training, so keep correctness runs separate.
Record total training time, Total step and major GPU stages, final loss/Gaussian
count and the final PLY. Compare any final quality claims using the same held-out
split. This establishes a runtime-configuration baseline, not evidence that
previously rejected kernels became faster. No new full-training result has yet
been measured by this implementation batch.
