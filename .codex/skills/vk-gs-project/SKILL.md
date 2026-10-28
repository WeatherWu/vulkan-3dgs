---
name: vk-gs-project
description: Project guide for the vulkan-3dgs repository. Use for Vulkan rendering, Slang shaders, Gaussian training, datasets, image caches, training UI, PLY I/O, build/debug issues, documentation and architecture in this codebase.
---

# vulkan-3dgs project

Read the affected code before editing. Preserve user changes and use the existing
`build` directory. The current branch is the source of implementation truth;
do not assume that a feature on another branch exists here.

## Hard Rule: Current-State Documentation

All documentation, skill references and code comments must describe the
confirmed current state of the branch. `plan.md` is the only exception.

- Do not record AI work logs, change narratives, temporary development phases,
  milestone labels, superseded decisions or completed-task summaries outside
  `plan.md`. Describe what exists and how it behaves, not how it was developed.
- When a decision is replaced, document only the current decision; use Git
  history for the superseded version. Preserve an old format description only
  when current code still supports it, as a compatibility reference rather than
  a development timeline.
- Verify behavior against code and confirmed requirements. Do not turn an
  unexplained implementation deviation, test failure or tentative hypothesis
  into a new product rule. State an unresolved limitation explicitly when it
  affects current usage; do not invent intended behavior to justify it.
- Keep necessary protocol identifiers and public option names exact. Do not
  confuse those interfaces with temporary project-stage labels.
- Document only the implementations present on the current branch. Experimental
  implementation and usage documentation belong on the experimental branch;
  an available switch is not evidence of accepted correctness or performance.

## Hard Rule: Faithful Paper Reproduction

When adopting a method from another paper, reproduce the selected method as
faithfully and completely as possible. Identify its paper version, official
implementation/branch and relevant configuration before choosing the design.
Do not silently replace, omit, simplify or add algorithmic details, including
visibility predicates, optimizer state/step rules, numerical formulas, schedules,
data layouts or method components. Distinguish a complete reproduction from an
approved adaptation or a partial implementation; never present them as equivalent.

If faithful reproduction conflicts with the project's existing path, numerical
contract or architecture, explain the concrete conflict and ask the user whether
to adopt the paper's original path or retain the project's path with an explicit
adaptation. Obtain that choice before implementing the conflicting part; an
existing explicit choice remains valid and must not be requested again.
Unspecified or unavailable upstream details remain unresolved, not permission to
invent them. Routine Vulkan/API translation that preserves the selected method's
behavior may proceed within the authorized scope; disclose material differences.

## Working rules

- Reuse the existing CMake configuration; do not introduce build-system churn.
- Keep application/UI policy separate from live training and low-level Vulkan.
- Preserve the C++/Slang ABI, descriptor bindings, queue ownership and barriers.
- Do not update skill feature descriptions for provisional work. Describe an
  accepted implementation only after validation and explicit maintainer retention.
  A request to update a durable engineering rule is independent of feature retention.
- Do not replace unresolved correctness failures with wider tolerances or
  document a workaround as an intended requirement without confirmation.
- Experimental implementations and usage documents are owned by their branch.
  Check branch contents before invoking tools or referring to options.
- The quality-new targets cover their explicitly configured module list, not
  every legacy or third-party file. Do not broaden them without a cleanup request.

## References

Read only references needed for the task:

- [Architecture](references/architecture.md): repository layout and ownership.
- [Training](references/training.md): current passes, buffers, controls and data.
- [Tasks and pitfalls](references/tasks-and-pitfalls.md): diagnosis and checks.
- [C++ architecture](references/cpp-architecture-profile.md): module boundaries,
  RAII, threading and refactoring invariants.
- [Documentation](references/documentation-style.md): current-state writing and
  equivalent English/Chinese development guides.
- [GPU decisions](references/gpu-optimization-decisions.md): use before selecting
  a GPU optimization, interpreting a regression or proposing measurements.
- `plan.md`: planning context only, not an implemented-feature inventory.

## Current model

`GaussianRenderer` owns normal PLY graphics rendering. `GaussianTraining`
owns compute training, with separate buffers and forward/backward renderers.
Do not combine the two resource graphs.

`Application` composes controllers and presentation. `TrainingController`
owns the device, dataset lifecycle, worker, immutable run configuration and
mutex-protected UI snapshot. The worker owns normal live training access;
stop/join it before changing device or resource configuration. Panels edit
passive settings and dispatch typed intents, not live Vulkan objects.

`TrainingFrameScheduler` owns frame order and completion. 3DGS Random uses a
seeded random-without-replacement stack and a 30000-step limit. Sequential
cycles frames without a fixed stop. Fixed Benchmark pins one frame and freezes
model/iteration state, optimizer and densification. Pure Training changes
presentation frequency and exports the final PLY.

`TrainingStepExecutor` owns prepare/main command buffers, submissions and fences.
The prepare command copies a 16-byte tile count to persistent host memory;
capacity handling precedes main recording. Do not substitute full-buffer
downloads for that count or call CPU wait time removable GPU work.

Pixel backward supports Auto, Direct, Workgroup Shared, Subgroup, Tile Gaussian
Atomic, VkSplat Per-Splat and VkSplat Tensor. Auto chooses Direct/subgroup behavior,
not VkSplat. Capability checks determine actual paths; unsupported requests fall
back to Direct. Tiles are Fixed8, Fixed16 (default) or experimental Adaptive Spatial.

Fixed16 Tensor offers default-off native32 subgroup reduction and Global reads.
Global requires Subgroup; the experimental RGB latency switch additionally
requires Global. SSIM modes0/1/2 select2D, full separability or2D-loss/separable
backward. Adaptive Spatial, checkpoint benchmarks, EarlyCull and private
optimizer/scheduling tools are experimental branch features, not accepted
speedups. Read docs/DEVELOPMENT.md and the focused diagnostic guides for actual
controls, resource checks and unresolved limitations. Preserve persisted IDs and
actual-path metadata.

Projection backward and dense Adam are fused on ordinary optimizer steps.
`VULKAN_3DGS_SEPARATE_PROJECTION_OPTIMIZER=1` selects the separate path for
diagnosis. The fallback Gaussian-gradient allocation remains present.

Training uses device-local packed RGBA8 targets, host/disk streaming and a
budgeted device cache. Dataset extent must match buffers; intrinsics are scaled
to actual decoded image dimensions. Keep the filename-sorted held-out split
matched to evaluation.

## Validation

After a coherent C++/shader change:

```powershell
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Also validate Release for performance-sensitive work. Run configuration-specific
builds/tests sequentially. Use `python -B` for Python tests. GPU checks must report
the actual device and capabilities; compilation/CPU tests do not prove GPU
runtime correctness, quality or speed.

The Windows application is `build/bin/Debug/vulkan-3dgs.exe`; non-Windows
omits `.exe`. Inspect startup options on the actual branch with `--help`.
