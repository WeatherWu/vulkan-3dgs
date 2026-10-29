# Direct intrinsics and physical 16-bit tile keys

These independent experimental options default off. Running workers lock both
choices. Direct projection changes apply to the next start/resume without
resetting the model. Changing physical key width requires **Load Dataset**, which
creates a fresh training state and replaces the training device if its feature
request differs. Use fresh models for independent training comparisons.

## Direct intrinsics projection

**Direct Intrinsics Projection (Experimental)** is saved as
`direct_intrinsics_projection`. It selects both the forward center projection
and its backward derivative, including the fused projection/Adam path and the
separate projection-backward path.

Camera-space X/Y/Z still use the original view/model transformation. The center
uses `u=fx*X/Z+cx-0.5`, `v=fy*Y/Z+cy-0.5` and the corresponding analytical
Jacobian. The half-pixel offset is the existing sampling convention. Covariance,
SH, clipping, alpha rules, dense Adam and ADC keep their original definitions.
Depth and its 32-bit sort encoding are not quantized.

Dataset frames upload original FP32 fx/fy/cx/cy in a separate 16-byte uniform at
binding64. The camera, push-constant and capture binary layouts are unchanged.
Replacing a camera without exact intrinsics invalidates the direct path; forward
and backward both use the matrix fallback. The UI displays the last executed
path and the missing-intrinsics reason, or **not executed** before the first step.

Direct forward/backward pipelines and descriptors have separate owners and are
created only when direct projection actually executes. The intrinsics uniform is
allocated/uploaded only through the explicit intrinsics path. Disabling direct
projection while stopped releases these resources; the ordinary camera upload
uses only the original camera buffer. Matrix descriptors retain their original
pool sizes and bindings.

Different FP32 operation order can change XY, coverage, rendered values and the
training trajectory. It is not a bitwise replacement of matrix projection. The
finite coordinate oracle uses
`8*eps32*(abs(f*coordinate/Z)+abs(principal)+0.5)` and gradient checks use
`1e-5+1e-3*abs(reference)`. These bounds are fixed before evaluation. Matrix-path
differences are reported separately; they do not replace the analytical oracle.

Legacy raw gradient/scanline/adaptive exports are rejected after direct execution,
and legacy optimizer captures cannot execute a direct step. Those formats cannot
identify the original intrinsics and projection choice. PLY export remains
available. A matrix step restores matrix-compatible raw export behavior.

## Physical 16-bit tile keys

**Physical 16-bit Tile Keys (Experimental)** is saved as `physical_tile_key16`.
It requires an actual Fixed16 Faster-GS sorting path, at most 65536 tiles and the
required 16-bit features enabled on the logical training device. Other cases
retain original uint32 tile storage and expose a fallback reason. Shader modules
requiring 16-bit buffers are created only on the enabled path.

The current Slang variants declare `UniformAndStorageBuffer16BitAccess`, so the
enabled feature set includes Vulkan11 `storageBuffer16BitAccess`,
`uniformAndStorageBuffer16BitAccess` and core `shaderInt16`. Physical support alone
is insufficient; callers pass the logical device's actual enabled status.

These features are requested only by a training device configured for physical16;
ordinary training devices do not enable them. The UI separates the device request
from actual enabled capability. Changing the checkbox does not recreate a device
or clear a paused model. Start/resume and benchmarks are blocked until **Load
Dataset** applies the new request; reverting the checkbox cancels that pending
change. A device-creation failure retains the existing device/model and reports
the error. Unsupported hardware uses the stated uint32 fallback without repeatedly
requesting a reload.

Emit writes uint16 tile keys directly. The two stable byte-radix passes read/write
uint16 keys and use a physically 16-bit temporary key bank. IDs, histograms,
offsets and the separate depth sort remain uint32. Range construction consumes
the sorted uint16 keys directly; there is no 32-to-16 conversion pass.

Key and ID banks have separate byte-size/alignment calculations. The private
variant reuses the original histogram/spine algorithm without editing the sorter
submodule. Key buffers and temporary storage include canary space used by the
diagnostics. The UI/report expose active key width, logical key bytes and the
allocated 16-bit key/scratch bytes. Existing fallback buffers may coexist, so
these byte counts do not establish whole-training or native peak-VRAM savings.

The uint32 sorting recorder and uint16 recorder are separate. The ordinary
recorder does not visit a physical16 workspace. With both candidates off, candidate
uniforms, pipelines, descriptor pools and key workspaces are absent; only method
selection and observation metadata remain on the CPU. Shader compilation can
still produce candidate modules without creating their runtime GPU resources.

## Independent correctness commands

Build the existing Release configuration. On Linux, run:

```bash
build/test/Release/training_optimizer_gpu_tests \
  --direct-intrinsics-only --gpu "RTX 3090" --require-validation \
  --report direct_intrinsics_correctness.json

build/bin/Release/training_range_keys_gpu_tests \
  --synthetic --full-sort-chain --integrated-sort --physical-tile-key16 \
  --require-validation --device-index 0 --report tile_key16_correctness.json
```

Append `.exe` on Windows. The optimizer tool is under `build/test`, while the
range tool is under `build/bin`. Use new report filenames. The range tool reports
the selected device and index; choose the RTX3090 index on a multi-GPU machine.

Direct checks cover centered/off-center intrinsics, non-square images, view/model
transformations, near clipping, image edges, FP64/finite-difference oracles,
depth equality and separate matrix/direct coverage/color/gradient differences.
Its private owner sequence includes a declared negative missing-intrinsics
fallback; positive cases must actually execute direct projection.

The 16-bit checks inspect SPIR-V integer width/array stride, compare stable
uint16/uint32 sorts and a CPU oracle over 0/1, odd lengths, workgroup/partition
boundaries, repeated maximum keys, alignment and scratch reuse. They check tails
and canaries, integrated list/range/color equality and private owner lifecycle.
CPU policy tests cover 65536/65537 tiles and unsupported capabilities. A requested
candidate unexpectedly falling back causes the positive diagnostic to fail.

Both modes require actual core/synchronization validation and report the GPU.
Synthetic owner steps exercise ADC/adoption, reset, SH changes, stopped edits,
renderer rebuild, new-model history and PLY export without loading a real dataset.
They do not establish native Tensor performance, held-out quality or full-training
speed. No separate performance executable is added.

## Target-device training comparison

Establish the disabled baseline first. Compare the reference build and the build
with both candidates off using the existing Fixed Benchmark or diagnostic replay
on the same frozen input, GPU and build flags. Keep both backward kernels in the
comparison. Different trained models are different workloads. A resource/source
isolation check alone does not demonstrate recovered training time; investigate
any repeatable remaining regression before adding candidates.

After target-device correctness checks, use the existing smoke entry for bounded
real lifecycle checks, separately selecting one candidate. Smoke reports include
requested/actual projection and key-width fields and reject unexpected fallback.
They retain fingerprints of the direct and key16 shader modules.

Use one Release build with shader debugging off and actual validation off for
complete training. Hold Faster-GS Sorting, Z-order, Tensor Subgroup/Global, mixed
SSIM, image selection, initialization, profiling and all saved optimizer/ADC
settings constant. In particular, record the actual saved Opacity LR rather than
silently replacing it with the code default.

Run three fresh 30000-step models: both options off; direct only; physical16 only.
Keep EarlyCull, RGB latency hiding and Pure Training off. Save full training time,
profiling, final Gaussian count/loss and distinct PLY files such as
`bonsai_base_30000.ply`, `bonsai_direct_30000.ply`, `bonsai_key16_30000.ply`.
Evaluate the same held-out views for all three. Changes in training-view loss
alone do not establish quality. Only after independent retention decisions should
a combined performance experiment be considered.
