# Scheduling diagnostics

The opt-in sampler observes ordinary training without changing prepare/main
submissions, fences, count readback, capacity growth or algorithms. A separate
private-buffer runner checks and times direct versus GPU-count sorting.
Neither implements GPU-driven training or CPU/GPU recording overlap.

## Bounded training capture

Use the existing Release compiler/shader settings and a fresh output path:

```bash
(
  export VULKAN_3DGS_S4_SCHEDULING_REPORT="$PWD/scheduling_timing.json"
  export VULKAN_3DGS_S4_SCHEDULING_START=8000
  export VULKAN_3DGS_S4_SCHEDULING_SAMPLES=128
  build/bin/Release/vulkan-3dgs --validation on
)
```

Start normal/Pure Training, not Fixed Benchmark. Samples start at the requested
one-based iteration. The window is bounded to1–2048 samples. A fresh run must
actually reach that iteration; PLY is not an optimizer-state resume.

The report is exclusively reserved at profiler initialization. File existence
does not mean completion: check `status:COMPLETE` and collected/requested counts.
Training continues after the final write; pause normally. Early destruction
writes PARTIAL. A failed diagnostic write disables the sampler, not iteration
advancement after completed Adam. No environment request means no extra query
pool, timestamps or report samples.

Three same-queue GPU queries mark prepare begin, prepare end after count copy,
and main begin. They are read after the existing main fence, without a new wait.
Unavailable endpoints yield independent null intervals.

Schema `s4_scheduling_timing_v2` records:

- CPU prepare submit/wait, mapped count read and capacity check;
- resize nested in capacity check and main record nested in post-fence interval;
- CPU main submit call, GPU prepare duration and prepare-to-main gap;
- iteration/frame/G, item count/capacity, modes, extent and upload dependency;
- ADC, numeric validation, reset, resize and capture classifications;
- CMake build config/compiler, macros, logger levels and actual application
  instance validation configuration after availability checking.

The per-step validation flag means training numeric checks, not Vulkan layers.
Unknown caller instance configuration is null. Externally forced loader
overrides are not introspected. Build labels do not reconstruct arbitrary flags.

CPU/GPU clocks are never subtracted. GPU prepare includes required work; CPU
fence wait is not wholly removable. The gap is an instrumented opportunity
bound, not pure idle. For recording overlap, min(same-step GPU prepare, CPU main
record) is only an optimistic ceiling, not measured savings. Other dependencies
and added GPU consumers/barriers/overflow handling remain costs.
Group categories overlap and all raw samples are retained. Final serialization
is outside GPU intervals but may affect the final CPU step.

## Validation control

`--validation on|off` is an application startup option, default on for Debug
and Release. Equal-sign syntax is accepted; invalid/missing/duplicate values
fail before window creation. It does not change numeric checks, logger level
or shader flags. Startup logs show requested/enabled values. An unavailable
requested layer retains the warning-and-disable behavior.

For performance runs use off and confirm actual enabled=false. Keep correctness
runs separate and enabled. Do not mix Nsight instrumentation into ordinary timing.

## Private sort probe

```bash
build/test/Release/training_scheduling_gpu_tests --synthetic --require-validation --report sort_boundaries.json
build/test/Release/training_scheduling_gpu_tests --count 1467157 --capacity 3305808 --require-validation --report sort_cost.json
```

The numbers are examples; select observed tile_items/capacity, not Gaussian
counts. Count/capacity are limited by device storage/budget and a4194304-element
tool limit. Inputs are synthetic, not renderer captures.

Eight synthetic cases cover empty/one,255/256/257, partition tails, capacity
slack and equal keys. A stable CPU oracle checks exact IDs/ranges and
tails/seeds/count/scratch canaries. The sibling timing-smoke report checks
private GPU buffer work and all three query endpoints, not the training worker.

Cost phases use2 warmup and8 samples each for A/B/A, complete separately fenced
restore and no comparisons/downloads between phases. The measured stage is
two stable32-bit sorts, high-key gather and range construction; final gather,
emit, rendering and backward are excluded.

The library reads a GPU count but launches partitions by capacity. Helper
gather/ranges in the GPU-count path also use capacity bounds. Reported
incremental cost is B minus mean(A1,A2), not full-training speedup. Restore
cache perturbation, all samples, drift, fingerprints and partition counts are
reported. Resource/identity/numeric failures reject results; outliers remain
in the report rather than being dropped.
