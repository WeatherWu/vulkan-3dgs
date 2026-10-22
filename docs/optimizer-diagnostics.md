# S4 fused optimizer diagnostics and offline prototype (experimental)

S4-0 provides capture and replay, not an optimized optimizer. Original FP32
projection/SH backward + dense Adam, group256, AoS and defaults are unchanged.
S4-1R adds an explicit offline-only streamed-Adam candidate. It is not admitted:
target 3090 replay passes bitwise but shows a stable performance regression;
local AMD state-sequence differences remain unresolved. No candidate is selectable
by normal/Pure Training; S4-1M and GPU-driven training are not implemented.
The separate [S4-3 scheduling diagnostics](scheduling-diagnostics.md) do not enable
GPU-driven training. This independent runner does not run the unresolved S3 stress test.

## Capture one actual training step

Pause and join training first. Click `Capture ONE S4 Optimizer Step and Pause`.
**This actually trains one step**, using the already initialized model and its
existing configuration, not newly edited UI settings. The next step must be
ordinary fused optimizer enabled, not ADC, opacity reset, final iteration or
an active Fixed Benchmark. Rejection does not advance to another window.

The worker copies pre-state immediately before the original fused dispatch,
executes it once, completes normal validation and iteration/frame/RNG advancement,
queues no further training writes, exports, and returns/joins. Export remains on
the exclusive worker instead of blocking the UI with multi-GB I/O. Resume starts
the next step; pause/shutdown waits for the started capture/export to finish.

Output is `Output Folder/benchmark_optimizer/step_N.optimizer.bin` plus a small
`.optimizer.json` manifest. Files are never overwritten. Failed writes may leave
an invalid partial file: use a new name. Errors report submission, GPU completion
and iteration advancement. Export failure after a completed step does not roll
back Adam. Uncertain submitted state blocks resume until model reinitialization.
Prepare submission and optimizer submission are distinguished: prepare coverage
already updates `maxScreenRadius`. A failure after prepare is therefore not
reported as an entirely untouched model, even when Adam was not submitted.

## Resource and synchronization contract

Pre-state is 840 B/G; post-reference is 736 B/G. At 1,248,125 Gaussians the
binary is 1,967,045,928 bytes. Capacity slack is metadata, not payload. Only pre
is cloned on the GPU; post is streamed from exclusively owned live arrays.
Unrequested steps add no diagnostic GPU allocation, copy, readback or wait.

The conservative policy requires `VK_EXT_memory_budget`, measured device-local
headroom for clones plus 256 MiB reserve, 128 MiB available RAM, and payload plus
64 MiB free disk. File budget is 8 GiB. Storage-range and group256 dispatch
limits are checked. Unknown/insufficient resources reject capture, without
model sampling, shrink-to-fit cache changes, precision changes or layout fallback.

All six source SSBOs already have TransferSrc usage. Explicit barriers cover
producer writes → transfer reads, transfer reads → original shader writes and
fused writes → export reads. A reusable 1 MiB coherent staging buffer downloads
on the compute producer queue. Camera comes from the successful upload shadow;
push bytes come from the actual dispatch, including prepared item count and Adam
bias correction. There is no unsafe UBO download or cross-queue assumption.

## Validate and replay

Build Debug and Release sequentially. Commands are from the repository root on
Windows; Linux uses the same paths without `.exe`.

```powershell
cmake --build build --config Release
build/test/Release/training_optimizer_tests.exe
build/test/Release/training_optimizer_gpu_tests.exe --synthetic --require-validation --report build/s4_synthetic.json
build/test/Release/training_optimizer_gpu_tests.exe --replay path/to/step_N.optimizer.bin --require-validation --report build/s4_replay.json
```

Choose a new report filename each time. With `--require-validation`, absent
core/synchronization validation rejects the run. Otherwise unavailable layers
are explicitly reported false, not passed.

Replay requires matching physical UUID, vendor/device, driver/API version,
Slang/backend/debug configuration and exact SPIR-V fingerprint. There is no
ignore-mismatch switch. Use `--synthetic` for local cross-vendor checks. The
fused kernel has no Tensor native32/45-KiB gate and can run on a 780M.

Only private seed/work buffers are bound. All parameter, m/v, densification and
padding words must match captured post-reference and repeated original A/A
bitwise. Non-finite values and negative second moments reject the run; no
sanitization is used. Every invocation restores all mutable arrays. Synthetic
negative controls detect skipping invisible Gaussians, no-op outputs and restoring
only parameters; the independent CPU oracle checks inactive SH freezing too.

After correctness: 2 warmup, 8 measured baseline invocations. Restore is separately
submitted/fenced before kernel timing to avoid timestamp overlap with transfer.
JSON records individual kernel, restore GPU, kernel CPU submit/wait and restore
CPU submit/wait times, plus restoration bytes. Allocation/upload/download/compare
are outside kernel timestamps. Restoration perturbs caches: these are isolated,
cache-perturbed costs, not ordinary training wall time. Without `--candidate-r1`,
only the original baseline is loaded and no numeric speedup is claimed.

## Binary reference

`V3DGOPT1`, version1, is little endian with a 928-byte header: camera offset560,
240 bytes; actual push constants offset800, 128 bytes. Definitions/static
assertions live in [training_optimizer_capture.hpp](../src/training/core/training_optimizer_capture.hpp).
Scanline inputs and v9 gradient exports are not interchangeable with this format.

| Block | Array | Stride |
| --- | --- | ---: |
| 0 | pre parameters | 240 |
| 1 | pre Adam m/v | 480 |
| 2 | projected Gaussians | 48 |
| 3 | projected gradients | 48 |
| 4 | pre densification | 16 |
| 5 | Gaussian tile ranges | 8 |
| 6 | post parameters | 240 |
| 7 | post Adam m/v | 480 |
| 8 | post densification | 16 |

Header includes exact lengths/strides, pre-global step, completed global/optimizer
steps, G/capacity/frame/extent/modes/SH, UUID, bounded identity strings, shader
fingerprint and block/header checksums. FNV-1a-64 detects accidental corruption;
it is not cryptographic authentication. Strict header, length/budget, ABI, mode,
step, checksum and finite-value checks precede large GPU allocations. Streaming
comparison avoids three full host outputs.

## Target evidence and branch selection

Capture a representative ordinary step on the target GPU. Collect uninstrumented
replay timing separately from native Nsight resources for the original fused
dispatch: registers/shared, theoretical versus active residency, spill/local
load-store, L1/L2/DRAM transactions/throughput and SM utilization. Preserve the
instrumented/debug shader identity separately. The old separate-projection
186-register screenshot does not describe this fused kernel.

The supplied `s4_replay.json` verifies original post-reference and A/A bitwise
on RTX 3090, Slang 2026.7.1 direct SPIR-V, shader debug ON, 1,281,062 Gaussians,
iteration 8615 / optimizer step 8534, frame 33, SH3. Core and synchronization
validation report zero errors. Its first measured kernel/restore samples are
34.815 / 71.3385 ms; the other seven kernels are 2.79347–2.83443 ms. Keep all
samples: the first anomaly prevents treating the seven-sample subset as an
accepted timing baseline. The separate synthetic report has a different device
UUID; it is not proof that both checks ran on the same physical card.

The supplied native capture of the original fused shader reports 255 registers
per thread, 23,552 B shared allocation and group256. The theoretical limit is
8 resident warps, limited by registers (hardware 48; shared-memory limit 32).
This supports one bounded live-range prototype, not proof of spilling. The
source declares no explicit groupshared array; the native shared allocation
must not be relabeled as confirmed spill. Samples and scoreboard percentages
are not elapsed-time or bandwidth shares. The 508.42-ms Entire Trace throughput
and occupancy summaries mix kernels and cannot characterize this dispatch alone.
Unavailable spill and transaction counters remain unknown/null.

## S4-1R offline streamed Adam

`train_backward_2dgs_to_3dgs_optimizer_stream_adam.comp.slang` keeps the original
typed AoS, bindings, FP32, group256 and projection/SH gradient code. All gradients
finish before updates. It loads/stores each pair of vec4 moment fields around
the original Adam helper, rather than keeping a whole local `AdamState`.
Invisible/zero-gradient Gaussians still undergo dense Adam; inactive SH keeps the
original mask behavior. The baseline shader and compilation settings are unchanged.
Reduced source-level lifetime is only a hypothesis, not a measured register gain.

Build on the target with the SAME compiler/backend/debug option as the existing
capture. Start with this command (Linux paths, from the repository root):

```bash
cmake --build build --config Release
build/test/Release/training_optimizer_gpu_tests --synthetic --candidate-r1 --require-validation --report s4_r1_synthetic.json
```

Use a new report filename. This checks 168 isolated cases, including nonzero
projection/rotation/color derivatives, then 12 optimizer-only sequence steps at
G=17/257 with SH0/1/2/3/1/0. The original GPU is the bitwise oracle; five isolated
configurations also use an independent CPU Adam check. Projected inputs in the
sequence are intentionally fixed: this is not a full training test. Failure exits
nonzero, reports device/shader identity, completed counts and first differing
word/bits, and must NOT be bypassed by relaxing tolerance.

Local AMD 780M / Slang 2026.7.1 / shader debug OFF passes the isolated cases but
fails at G=17, SH1, sequence step 1: params=0, m=11, v=0, densification=0 differing
words, nonfinite=0. The first mismatch is block7 word131 (Gaussian1 first-moment
rotation.w), actual/reference bits 3106528930/3106528929, a one-ULP difference.
Using the original scalar Adam wrapper and typed-buffer field accesses did not
remove it. The cause is unresolved; do not label it harmless or a driver bug.
The strict local gate remains; the stable project skill does not promote this
candidate. Subsequent target reports provide a separate conclusion: 3090 synthetic
180 cases and real capture replay pass bitwise, but `s4_r1_replay_2.json` measures
A1/B/A2 = 2.616972/4.166016/2.626688 ms, A drift 0.37058%, verdict
`STOP_NO_SHORT_GAIN` (58.9% slower than mean A). The prior replay used another
physical card/model state and had 4.07% A drift; its times are not same-state
repeats of this result. Stop R1 performance promotion and training integration;
do not repeat timing or full training without a new mechanism/evidence.

Debug/Release builds and each configuration’s 20 CTests pass, as do 47 Python
regressions. Both GPU configurations reproduce the same sequence failure with
zero core/synchronization validation errors; reports are
`build/s4_r1_synthetic_debug_typed.json` and
`build/s4_r1_synthetic_release_typed.json`. Baseline-only Release still passes
140 cases and both capture hooks; Debug/Release original replay on the small
local synthetic file passes. The candidate profiling request is also blocked
by the sequence gate. Successful candidate timing/profile-window paths and
actual training smoke are not locally verified.

ONLY if target synthetic checks pass, replay the original S4 binary on its exact
physical device/driver and original shader identity:

```bash
build/test/Release/training_optimizer_gpu_tests --replay path/to/step_N.optimizer.bin --candidate-r1 --require-validation --report s4_r1_replay.json
```

Replay automatically repeats the 12-step sequence gate before touching real
candidate timing. It verifies baseline post-reference/A/A and candidate/repeat,
then measures A1/B/A2 with 2 warmup and 8 measured invocations per phase. Every
dispatch restores all mutable state; no comparison/download occurs between
timed phases. All samples and restore costs are retained. A phase max/min >1.20
or A drift >2% is inconclusive; gain <5% or not greater than twice the A difference
stops admission. A short gain still requires native review, not training promotion.
Synthetic replay files are explicitly excluded from target performance evidence.

After correctness and short-gain gates, resource capture can use a bounded
headless window, separate from timing:

```bash
build/test/Release/training_optimizer_gpu_tests --replay path/to/step_N.optimizer.bin --candidate-r1 --require-validation --profile-r1-seconds 30 --profile-variant stream-adam --report s4_r1_native.json
```

Launch this command as the Nsight target; manually start collection when stderr
prints `S4-1R PROFILE WINDOW READY`. Use the Nsight host UI collection control,
not F11 in a nonexistent target window. Repeat explicitly with `--profile-variant
baseline` and a different report filename for paired resources. The window is
limited to 1–30 seconds, restores the same seed every dispatch, checks final
state and reports `PROFILE_ONLY_NOT_TIMING`, never speedup. Native register,
shared and spill fields remain null until externally collected. Normal training,
S4-1M layout migration and S4-2 integration remain out of scope.
