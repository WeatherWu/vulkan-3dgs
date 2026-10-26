# Direct sorted-key range audit

`tests/training_range_key_audit.py` is an independent, CPU-only audit of the
Fixed16 range-boundary data path. It does not change shaders, select a training
variant, execute Vulkan or combine with depth-sort reuse.

## Input and invocation

Use matching `V3DSCN01` and `V3DSRT01` inputs from the original renderer:

```text
python -B tests/training_range_key_audit.py --capture frame_0_iter_8037.scanline.bin --sort-input sort_baseline_inputs.bin --report range_sorted_key_audit.json
python -B -m unittest discover -s tests -p "test_*.py"
```

The report path must be new. The tool reuses the bounded capture reader and
raw count/emit verifier from the [sort audit](training-sort-reuse-audit.md),
checks input hashes and preserves input files. It reconstructs both stable
32-bit sorts on CPU and verifies final Gaussian IDs against the capture.

## Producer and lifetime contract

The original boundary shader reads `tileKeyHigh[tileSortIndices[i]]`.
`gatherHighTileKeys` first gathers high keys into `tileSortScratch`, and the
second key/value radix call sorts that key buffer in place. The current
library's four ping-pong passes end in the input key buffer. Thus for at least
two items:

```text
tileSortScratch[i] == tileKeyHigh[tileSortIndices[i]]
```

`tileSortScratch` is the key array at binding23, not the radix library's
histogram/intermediate `tileSortStorage`. No intervening producer overwrites the
key array before range construction. Its existing compute dependency must
remain valid for any direct reader.

For zero or one item, the renderer bypasses radix sorting and does not produce
`tileSortScratch`. A direct-key design must select original `tileKeyHigh` in
these cases, without assuming that stale scratch is initialized. Zero items
must not read any key; one item uses the original identity index.

## Checked equivalence

The CPU model compares both intermediate start/end boundaries and every final
range component. It also compares an independent histogram/prefix range oracle
and the captured final ranges. Tests cover empty lists and tiles, one item,
equal keys, final-item sentinel handling, active counts smaller than capacity,
warp/workgroup/radix partition edges and invalid producer/permutation data.

This checks integer lookup equivalence. It does not independently download
actual GPU sorted-key output or validate a replacement shader. Existing atomics,
boundary predicates, finalization, dispatch count, sorted IDs and tie rules are
not changed by the audited mechanism.

## Access budget and limits

For `I` items, the source expressions perform `2I` current/previous key queries.
The indirect path has `4I` uint references, and direct reading has `2I`.
The resulting logical-byte difference is `8I`; compiler reuse, caches and
coalescing can make native traffic different. The report's consecutive-32-item,
32-byte-address-block comparison is an address-distribution proxy, not measured
warp execution or memory transactions.

Under the producer contract, the proposed direct reader needs no additional
production buffer, pass or per-frame copy. An isolated diagnostic's own
resources are not a claim about the production allocation. The CPU audit does
not measure native instructions, synchronization behavior, cache effects,
target-device speed or downstream cost; runtime checks are separate below.

`CPU_EQUIVALENCE_PASSED_GPU_COMPARISON_REQUIRED` is not production admission.
A GPU comparison must first verify the library-produced sorted keys against
the original-key/permutation relation, then check both range paths exactly on
private buffers, including zero/one-item behavior and canaries. Any timing must
retain an original A/B/A reference, include all candidate-specific preparation
and preserve the declared drift/outlier rules. A range-local gain is not a
whole-training gain.

## Private GPU comparison

`training_range_keys_gpu_tests` is a standalone diagnostic. The candidate
`train_range_boundaries_sorted_keys_test.comp.slang` is compiled as a separate
module and is never loaded or selected by the training renderer. It changes
only key lookup; atomic exchanges, boundary predicates and finalization remain
the same. No training switch is added.

The runner reads the matching scanline and raw sort-input binaries. On private
buffers it executes the actual radix library's depth sort, the original high-key
gather, tile sort and item gather. It verifies GPU-produced sorted keys, stable
indices and final Gaussian IDs against CPU/captured references before accepting
either boundary path. CPU-sorted keys are not uploaded as the candidate input.
For zero/one item, direct lookup binds the original-key buffer and leaves
unproduced scratch unread.

### Run on the target GPU

Build and run from the repository root; Windows appends `.exe`:

```text
cmake --build build --config Release
build/bin/Release/training_range_keys_gpu_tests --synthetic --report range_keys_edges.json --require-validation
build/bin/Release/training_range_keys_gpu_tests --capture frame_0_iter_8037.scanline.bin --inputs sort_baseline_inputs.bin --report range_keys_correctness.json --require-validation
```

The synthetic suite covers zero/one, tied keys and warp/workgroup/radix
partition boundaries with capacity greater than active count. The real-input
command requires exactly matching projected/raw metadata and count/emit data.
Both paths are checked in A/B/A order, including readonly seeds, unused tails,
sort scratch and range-output canaries. Integer comparisons have no tolerance.
Input files are never overwritten; each report path must be new.

Only after validated correctness succeeds, run one bounded timing comparison:

```text
build/bin/Release/training_range_keys_gpu_tests --capture frame_0_iter_8037.scanline.bin --inputs sort_baseline_inputs.bin --report range_keys_timing.json --timing --no-validation
```

Timing is forbidden with instance validation requested or for synthetic-only
input. Default validation uses available layers; `--require-validation` demands
both core and synchronization validation. `--no-validation` explicitly disables
that instance request, not external loader overrides. `--device-index N`
selects the device. A failed run writes `STOP` and returns nonzero; never use its
partial output as passing evidence.

### Measurement scope

Each of A1/B/A2 has two warmups and eight retained measured samples. Every
invocation restores private work arrays, output ranges and radix scratch in a
separately fenced submission. The readonly seeds remain unchanged. Restore and
downloads are outside timing; restore submit/wait cost is reported separately
and its cache perturbation is disclosed. The actual producer sorts execute
inside each measured sequence for both A and B, so B never receives a free
CPU-produced or previously sorted input.

The same invocation records:

- `boundaries_ms`: boundary dispatch and its compute barrier;
- `ranges_ms`: boundary dispatch/barrier and original range finalization/barrier;
- `sort_gather_ranges_ms`: two sorts, two gathers and range construction.

The last interval is a broader private sequence, **not** the whole list
production interval: projection, count, prefix, emit, capacity growth,
composite, backward, Adam and live submissions are not executed. It cannot
establish a full-list or training gain. GPU comparisons include identical
timestamp points for A and B and verify both paths before/after timing, with no
downloads between measured phases.

JSON includes device UUID/driver, actual instance-validation requests, compiler
and shader identities, source binary fingerprints, all samples and separate
sequence/range-local evaluations. The shared 20% outlier, 2% drift and 5% gain
rules are applied without sample deletion; the broader sequence also applies
the 0.10 ms investment target. Non-3090 measurements are not target-device
admission. Even a private short gain still requires native review and matched
complete-list/downstream assessment; `full_training_speedup` remains null and
`production_candidate_enabled` remains false.
