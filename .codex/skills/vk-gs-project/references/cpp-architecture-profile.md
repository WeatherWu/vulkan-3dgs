# C++ Architecture Profile

## Purpose

This repository applies a focused C++ Core Guidelines profile when restructuring or extending C++ code. The target is clearer responsibility, ownership, and concurrency boundaries without weakening Vulkan/Slang performance paths. It complements the existing project style; it does not require adopting Google, LLVM, AUTOSAR, or CERT formatting wholesale.

Apply these rules to the complete requested scope. Both incremental extraction and a stage-wide batch are valid. A refactor is successful when behavior, GPU/CPU synchronization, C++/Slang ABI, and benchmark quality remain unchanged or are deliberately revalidated.

## Responsibility Boundaries

- A class should have one primary reason to change. File length alone is a signal to inspect responsibility, not a rule requiring arbitrary splitting.
- `Application` composes modules, owns GLFW/ImGui routing, and translates user actions into commands. It should not accumulate training-step internals, settings parsing, worker-loop policy, or renderer implementation details.
- `GaussianTraining` owns the semantics of one training step, dataset/model state, GPU training resources, validation, and export. It must not depend on ImGui or window input.
- Normal graphics rendering remains independent of training. `GaussianRenderer` composes focused frame, pipeline, resource, sorter, and command-recorder owners; input policy belongs to `CameraController`/application-level code.
- Configuration parsing/persistence, UI panels, worker lifecycle, and GPU-step orchestration are separate responsibilities when their change cadence diverges.

`TrainingSettingsStore`, `TrainingController`, `TrainingSettingsPanel`, `TrainingDiagnosticsPanel`, and `TrainingCommandPanel` are current modules. Settings persistence is separate from `Application`; the controller owns the asynchronous worker and snapshot/result handoff; editable controls mutate only passive `TrainingSettings`; diagnostics consume only immutable snapshots; command presentation emits typed intent flags. File-dialog, worker, dataset, and export execution stays at the application/controller boundary and must not let a panel mutate `GaussianTraining` directly.

## Dependency Direction

Keep dependencies directed from orchestration toward implementation:

```text
Application / UI
    -> controller or command facade
        -> training or graphics domain object
            -> Vulkan/image/resource wrappers
```

- Lower layers do not include ImGui, GLFW input policy, or application settings.
- Avoid circular includes. Prefer forward declarations in headers and concrete includes in `.cpp` files when ownership permits.
- A renderer may depend on Vulkan wrappers and model data; it must not reach into training UI state.
- Do not use `dynamic_cast` as a general service locator. Keep renderer-specific adaptation at the composition boundary.

## Ownership And Lifetime

- Prefer values for required ownership and `std::unique_ptr` for optional or polymorphic ownership. Raw pointers/references are non-owning observers and require a documented valid lifetime.
- Prefer the Rule of Zero. A class that only owns RAII members should not manually implement copy/move/destruction operations.
- Keep Vulkan handles in focused RAII wrappers. A module that records commands should not also become the owner of unrelated UI or dataset lifetime.
- C++/Slang structs, descriptor binding numbers, and static assertions are an ABI boundary. Centralize their ownership and update both sides in one change.

## Concurrency

- The training worker is the sole normal caller of live `GaussianTraining` mutation APIs while async training is active.
- UI/main-thread code reads immutable `TrainingUiSnapshot` or `TrainingWorkerResult`; it does not inspect live training buffers or mutable training state.
- Stop and join the worker before changing training device, dataset, renderer ownership, configuration that rebuilds resources, or application shutdown.
- Use `std::jthread`/`std::stop_token` for owned worker lifetime. Do not detach threads.
- A UI refresh cadence may change presentation only; it must not silently skip training passes, validation, optimizer steps, or densification.

## Interfaces, Errors, And Invariants

- Make preconditions explicit at public boundaries. Use values such as `std::optional`, typed configuration, or clear exceptions rather than sentinel state spread across callers.
- Keep an invariant close to the type that owns it. Examples: `TrainingBuffers::extent()`, initialized renderer state, and selected GPU capabilities.
- Preserve the existing exception-based error model at module boundaries unless a requested refactor deliberately changes it. Do not mix unchecked error codes and exceptions in one new API without a clear conversion boundary.
- Use `struct` for passive transfer/configuration data with no invariant; use `class` for behavior, ownership, synchronization, or invariants.

## Performance-Sensitive Code

- Optimize from profiling data, not from class size or generic style advice.
- Keep shader selection, Vulkan descriptor writes, barriers, and command-recording details localized to their renderer/controller. Do not pull them into UI code for convenience.
- Preserve Fixed Benchmark and Pure Training semantics during refactors. Benchmark mode freezes model state and disables optimizer/densification; Pure Training changes UI publication only.
- Retain explicit barriers and queue ownership. Refactoring must not replace a known synchronization boundary with implicit ordering.

## Refactoring Procedure

1. State the responsibility being extracted and the behavior that must remain unchanged.
2. Add narrow interfaces and move the planned ownership boundaries. Keep the batch coherent and scoped.
3. Keep compatibility adapters only when they reduce risk; remove them before declaring the extraction complete.
4. Build and run unit tests after the coherent change. Intermediate builds are optional. For training changes also use Fixed Benchmark and, when relevant, a complete training/quality run.
5. Compare profiling, validation, Gaussian count, and output quality before claiming a performance-neutral refactor.
