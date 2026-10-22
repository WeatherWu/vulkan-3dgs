# GPU optimization decisions

## Establish the actual baseline

Identify the device, active mode, model/frame, compiler/debug flags and compiled
module. Count work that the current implementation executes, not work already
removed by batching, pruning, coefficient reuse or optimizer fusion.

Separate measured facts, plausible mechanisms and missing evidence. A technique
name, high register count or low occupancy does not establish removable cost.

## Budget combined and end-to-end cost

For each proposal identify:

- which operations, transactions, atomics, barriers or critical-path waits disappear;
- which classification, storage, dispatch, synchronization or dependency costs appear;
- whether saved cost can exceed new cost in the complete affected sequence;
- the upper bound on full-training benefit, not just the fastest subpass;
- the smallest target-device measurement that can support or reject the mechanism.

Count preparation and intermediate traffic. Grouping can add state construction
without reducing backward work. Culling helps only when it removes executed
instructions or transactions; rejected lanes do not imply proportional warp savings.
Reduced convolution tap count is not the same as reduced DRAM traffic.

Source object lifetime is not native register lifetime. Compilers may scalarize
state; streaming loads can also reduce memory/instruction parallelism. Inspect
generated behavior before attributing a resource change to source layout.

A fence waits for required GPU execution as well as scheduling gaps. Removing it
cannot remove that execution. Audit count, capacity, overflow and sort dependencies;
measure real GPU boundaries and CPU recording before proposing overlap.

## Interpret resources at the measured scope

Theoretical occupancy is a residency limit, not measured utilization. Consider
registers, shared-memory allocation granularity and available parallelism together.
Shared allocation and spill/local-memory traffic are distinct. Missing counters
remain unknown, not zero.

Samples and scoreboard stall percentages are not elapsed-time fractions. Whole-trace
throughput mixes dispatches. Use dispatch-specific evidence for kernel claims.
Logical bytes and source strides do not establish actual memory transactions.

## Keep experiments interpretable

Preserve an original reference and vary one mechanism. Prefer a bounded isolated
check before broad integration. Match UUID, driver, module, flags and workload.
Retain all samples and obey the declared drift/outlier rules; do not retry until
favorable or combine different captures into a same-state repeat.

For in-place optimizer measurements restore parameters, both moments and relevant
statistics before each invocation. Record restore costs separately and disclose
cache perturbation. Production adapters remain candidate costs.

Use the numerical contract required by the change. Investigate mismatches instead
of widening tolerances. Dense Adam still updates active zero-gradient parameters;
inactive SH masking is a separate condition.

Keep correctness checks, uninstrumented timing and profiling captures separate.
Check actual Vulkan validation configuration, logger level and build flags; an
executable directory named Release does not establish its complete runtime state.

Do not rescue a failed mechanism with an expensive full-training run or unrelated
parameter scan. Reopen it only for a falsifiable new hypothesis. A retained local
gain still needs lifecycle/fallback validation and matched full-training assessment.

## Preserve research boundaries

Changes to Gaussian budgets, resolution, SH frequency, optimizer density or
training steps change training trajectories. They require an explicitly chosen
quality/time objective and held-out evaluation; do not introduce them as incidental
fixes to an equivalence-preserving optimization.
