# Tensor row-empty R0 contract

This is the shared numerical contract for the experimental R0 classification
gate and isolated R1 backward prototype, not a performance result. The previous
padded SPD root interval is diagnostic only. R0/R1 use a conservative tangent
lower bound, without trusting root, determinant-division or square-root accuracy.
The application dispatchers/defaults do not select the R1 prototype.

## Supported domain

Inputs are the original FP32 projected Gaussian and integer pixel coordinates.
The classifier keeps the whole row on any failed precondition:

- finite center XY, absolute value at most `2^20`; width/height at most `2^20`;
- `A,C` in `[2^-16,2^16]`, `|B|<=2^16`, with nonzero `|B|>=2^-16`;
- finite opacity in `[float(1/255),1]`, matching method-1's existing contract;
- the computed `A*C-B*B > A*C/1024` scope check; this is not used as an exact
  determinant certificate, since the lower-bound proof needs only `A>0`;
- valid tile/row/image coordinates and finite intermediate values;
- the computed magnitude `M` below is at most 128.

This deliberately sacrifices coverage for a bounded arithmetic domain. It is
not an epsilon parameter to tune. Non-SPD, tiny, extreme-scale and non-finite
inputs keep the old path. An invalid image row is never classified as new work.

## Real-arithmetic lower bound

For the row's fixed `y`, let `l,h` be the first/last valid pixel's X deltas,
`Y` its Y delta, and `q(x)=A*x*x+2*B*x*Y+C*Y*Y`. Choose **any** `v` in `[l,h]`.
The implementation uses a clamped approximate vertex for a useful, but not
necessarily optimal, tangent point. No accuracy premise about that division
is needed beyond finite `v` inside the interval.

```text
q(x) = q(v) + 2*(A*v+B*Y)*(x-v) + A*(x-v)^2
     >= q(v) - abs(2*(A*v+B*Y))*(h-l)        for every x in [l,h]
```

Thus `L=q(v)-abs(q'(v))*(h-l)` is a lower bound on the entire geometric row.
It does not consult `processedCount`. A nonempty support span with no eligible
pixels is NOT a geometric empty row under this classifier.

## FP32 error ledger

Let `e=2^-23`, `D=max(abs(l),abs(h))`, and the positive magnitude sum be
`M=A*D*D+2*abs(B)*D*abs(Y)+C*Y*Y+1`. Quantities in this paragraph denote
exact-real magnitudes of the represented operands; the shader computes their
FP32 counterparts. All large intermediates are bounded well below overflow;
`M>=1` also bounds accumulated absolute flush-to-zero errors.

Use `e`, not half-epsilon: either permitted rounding direction is covered.
The following deliberately loose bounds include source-order products/adds,
their intermediate rounding, and permitted subnormal flushing:

| Source of error | Upper allowance, units of `e*M` |
| --- | ---: |
| Candidate/original coordinate subtraction and interval endpoints | 16 |
| Source-order `q(v)` | 16 |
| Slope, width and slope-times-width (`abs(slope)*width <= 4*M` in real arithmetic) | 48 |
| Tangent subtraction and original power evaluation | 32 |
| Magnitude/error evaluation rounding and remaining absolute FTZ terms | 16 |
| Total allocated | 128 |

For example, each triple product has relative forward error bounded by
`gamma(2)`; summing the three signed terms is bounded against their absolute
magnitude sum, not a small cancellation-damaged result. Since `|v|<=D` and
`h-l<=2*D`, the slope penalty is bounded by `4*M`. These inequalities also
cover a fused original power with fewer intermediate roundings. Coordinate
differences use exactly representable integer inputs and bounded finite
centers; normal subtraction errors are relative to the difference, while
absolute subnormal errors are negligible against `e*M`. Cancellation in the
determinant does not enter the exclusion proof.

The shader subtracts `256*e*M_computed` from the computed tangent. Positive
sum rounding perturbs `M` by less than `8*e` relatively on this domain, so this
remains greater than the allocated `128*e*M` error, with room for rounding the
error and its subtraction. Neither the division-selected vertex's error nor
an assumed exact square root is part of this ledger. A non-finite result
always retains the row.

Rejection requires the resulting lower bound to be **strictly** greater than
`-2*threshold`, where the existing method-1 threshold is
`log(float(1/255)/opacity)-2^-10`. Since `M_computed<=128`, every rejected
original power is negative and greater than `-80`, including the above error
budget. The existing bounded exp/log guard therefore applies without extending
its underflow contract. Equality and uncertain cases keep the original sample.

The FP32 instruction assumptions come from the
[Vulkan precision rules](https://docs.vulkan.org/spec/latest/appendices/spirvenv.html#precision-and-operation-of-spirv-instructions).
The guard's exp/log budget remains the one documented for pixel EarlyCull;
R0 neither lowers that guard nor reorders retained sample arithmetic.

## Compiler and validation gates

Source `precise` alone is insufficient. Local Slang 2026.7.1 direct SPIR-V
emits no `NoContraction` even with `-fp-mode precise`, reproducing the
[upstream compiler issue](https://github.com/shader-slang/slang/issues/11933).
Only RowEmpty census/candidate wrappers use Slang's `-emit-spirv-via-glsl`
compatibility path plus `-fp-mode precise`; original training and S3-2 shader
flags are unchanged.
This uses the bundled downstream compiler, not a new renderer or dependency
installation. If that backend is unavailable, compilation must fail rather
than silently use direct emission. Runtime inspection rejects absent
`NoContraction`, non-FP32/relaxed floating-point values or changed exp/log lowering. This
inspection is a guard against known regressions, not a substitute for reviewing
the arithmetic graph after compiler/helper changes.

The CPU tests separately check the bound against an analytic exact-real
clamped-vertex minimum, the original FP32 sample, a worst-direction power
allowance and an independent long-double alpha envelope. GPU tests compare
original sampling counts against the independent scalar oracle, and every
rejection against the original GPU sample function. Different compiler paths
need not produce bit-identical oracle values at every threshold, so the
target numerical admission still requires its own zero-false-drop report.
No integrated Tensor gradients have been computed by R0. R1's SPIR-V audit
allows one bounded-integer case emitted by glslang: the uint32
`SubgroupLocalInvocationId` input (SPIR-V BuiltIn 41) and direct uint32 loads
from that input may carry `RelaxedPrecision`. On admitted native32 devices
their values are 0..31. The audit verifies that builtin, input pointer and load
types; it does not exempt integer arithmetic, Gaussian indices, coordinates,
struct members or any floats. In-memory negative tests reject relaxed float
results, relaxed integer arithmetic, and broken precision decorations.

## Report and next gate

```powershell
build/test/Release/training_checkpoint_gpu_tests.exe --row-empty-r0-capture path/to/frame.scanline.bin
```

Linux omits `.exe`. This reuses the original S3-2 binary input and writes
`*.scanline.row_empty_r0.json`, adding suffixes on collision. It does not read
the old JSON as if it contained per-Gaussian/pixel data. Reports retain the
original counters plus eight geometric-row/native-wave counters; old
`span_*` labels refer to full-or-empty row masks in this report only.
`oracle_empty_eligible_rows_kept` is the count of no-contributor eligible rows
that this geometric classifier keeps, NOT exclusively prefix-induced emptiness.

Native-wave counts use actual BASIC/VOTE subgroup operations in the sampling
census. They are not Tensor's Gaussian-reduction mapping or a prediction of
issued instruction savings. R0 has no timing metric, no gradient output and
no speedup estimate. IDs and driver must match the capture for admission.
Synthetic source IDs zero produce an inconclusive report. An input error or
false rejection stops the method; geometric eligible-pair coverage below 10%
stops this iteration as an engineering screen. A coverage pass permits R1
evaluation, not a claim of correctness or speed.

## R1/R2 integration boundary

`train_backward_vksplat_tensor_row_empty` uses the same predicate, unchanged
threshold/domain and batch16. Sixteen thresholds and sixteen uint row masks
add 128 declared shared bytes. Gaussian/threshold/mask initialization is
published by the original load barrier; 256 threads classify Gaussian×row
pairs, combine keep bits by shared atomic OR, then uniformly publish once.
The sample loop retains every zero pair write, including culled rows. Reduction
tests the same mask before reading that Gaussian row's matrix entries. Original
relative Gaussian order, processed guards, T/suffix state, accumulation order,
shuffle participation and original batch barriers remain unchanged.

The common body also has explicit batch4/Shared test wrappers. They fit smaller
devices and test stale pairs, mask flips, prefix/border cases and independent
CPU gradients, but are never substituted for actual Tensor16 or its timings.
The actual candidate and original Tensor still need target native32 validation.

`--row-empty-capture` rereads the binary input, rechecks R0 on the matching
source GPU/driver, compares all 12 output gradient slots and a baseline repeat,
and verifies the pixel-gradient/BlendState inputs remain unchanged. Forward,
Adam, densification and normal training are not run. Only then does it collect
short A/B/A timestamps for the complete candidate dispatch, including mask
construction/publication/use. An error/validation failure aborts timing.
Candidate and baseline use different compiler backends, so a measured net
difference is not attribution to the mask alone.

Long A/B/A requires both `--timing --resources-reviewed` and a repeated short
gain. The latter flag records operator review, never fills unknown native
register/shared/residency measurements. The reporter uses null for unmeasured
values. Timing policy rejects >2% A drift and requires >=5% net gain beyond
twice the A-to-A difference before resource/multiview review. This is an
engineering screen; it neither guarantees full-training speedup nor enables
the application prototype automatically.
