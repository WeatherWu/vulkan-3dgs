# Experimental row-empty classifier

This reference describes the implemented predicate and its arithmetic assumptions,
not a completed production safety proof or a performance result. The isolated
classifier/candidate are not selected by normal application dispatch. Stress
validation remains unresolved; geometric coverage alone does not establish safe
integrated execution or faster training.

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
inputs use ordinary sampling. An invalid image row is never classified as new work.

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
The classifier neither lowers that guard nor reorders retained sample arithmetic.

## Compiler and runtime checks

Row-empty wrappers use the GLSL SPIR-V emission path with precise floating-point
mode. The runtime audit checks NoContraction, FP32 types, precision decorations
and exp/log lowering. Source annotations alone are not a certificate.

The audit permits RelaxedPrecision only on the native32 uint subgroup-invocation
input and its direct loads. It does not exempt arithmetic, indices, coordinates,
struct members or floats. Numeric changes still require target execution.

## Tools

Linux commands; Windows adds .exe:

```bash
build/test/Release/training_checkpoint_gpu_tests --row-empty-r0-capture path/to/frame.scanline.bin
build/test/Release/training_checkpoint_gpu_tests --row-empty-capture path/to/frame.scanline.bin
```

Inputs use the scanline binary format, not a JSON summary. Capture identity and
driver must match. Reports separate geometric exclusions, already-guarded work,
kept oracle-empty rows and native-wave opportunities; nested counts are not
additive instruction savings.

The census does not emit gradients or measure training speed. The isolated
Tensor candidate checks all projected-gradient slots, repeat behavior and
read-only pixel/BlendState inputs before timing. It includes mask construction,
publication and use, not just the filtered inner loop. Candidate and reference
compiler paths differ, so a net timing difference is not attribution to one
predicate alone. Portable batch4 checks do not substitute for actual Tensor16.

The Tensor candidate retains relative Gaussian order, processed-prefix guards,
T/suffix recursion, zero pair writes and original reduction participation. Masks
add storage, classification, shared atomic OR and a publication barrier.
Forward, Adam, densification and normal training are not executed by this tool.
