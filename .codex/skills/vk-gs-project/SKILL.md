---
name: vk-gs-project
description: Project onboarding and maintenance guide for the vk-gs repository. Use when working in this repo on Vulkan rendering, Slang shaders, 3D Gaussian Splatting graphics rendering, compute-based Gaussian training, MipNeRF360/COLMAP dataset loading, UI training controls, PLY import/export, build/debug issues, or architectural questions about this codebase.
---

# vk-gs Project

Use this skill to become productive in the `vk-gs` repository quickly. Treat it as project-specific context, not as a replacement for reading the touched files.

## First Actions

1. Work from the repository root: `C:\Users\weath\Documents\vscode\CG\advanceSearch\vk-gs`.
2. Inspect the exact files involved before editing. Prefer `rg`/`rg --files`.
3. Build after meaningful C++ or shader changes:

```powershell
cmake --build build --config Debug
```

4. Do not introduce Ninja/CMake preset churn unless explicitly requested. Current active build path is the existing `build` directory.
5. Preserve the user's dirty worktree. Do not revert unrelated changes.

## What To Read

Load only the reference file relevant to the task:

- For project layout, build, render bases, and app/UI flow, read `references/architecture.md`.
- For training implementation, data formats, shader passes, optimizer, densification, and known 3DGS gaps, read `references/training.md`.
- For common fixes, error meanings, coding rules, and validation commands, read `references/tasks-and-pitfalls.md`.

## Core Mental Model

The project is a Vulkan + Slang 3D Gaussian Splatting renderer with two major paths:

- `GaussianRenderer`: normal graphics rendering from PLY models using GPU key generation, radix sort, and instanced quad rasterization.
- `GaussianTraining`: compute-based training path using MipNeRF360/COLMAP datasets, compute forward rendering, compute backward passes, Adam optimization, densification/pruning, and PLY export.

The training path is not a thin wrapper over the graphics renderer. It owns its own buffers, compute renderers, and shader passes.

## Important Rules

- Keep C++/Slang struct layouts synchronized. `training_types.hpp` static asserts are important; shader structs in `shaders/training_shader/slang/common/` must match.
- Descriptor buffer infos returned by `TrainingBuffers` are values. If a Vulkan write needs a pointer, keep the returned `vk::DescriptorBufferInfo` in a stable local variable before passing its address.
- Training frame dimensions must match `TrainingBuffers::extent()`. Dataset loading currently uses real image dimensions from `stbi_info` and scales COLMAP intrinsics to those dimensions.
- Training currently selects images sequentially, not randomly, and has no automatic total-iteration stop unless added.
- Training UI `Start Training` can auto validate/load a dataset, then initialize and run. If it is disabled, inspect `training_dataset_loaded_`, `training_dataset_valid_`, and the status text in `Application`.
- Shader additions must be added to top-level `CMakeLists.txt` via `compile_training_shader(...)` or `compile_slang_shader(...)`.

## Validation Expectations

For code changes, normally run:

```powershell
cmake --build build --config Debug
```

For UI/runtime issues, report the executable:

```text
build/bin/Debug/vk_gs_windows.exe
```

If a runtime issue depends on local data or GPU behavior and cannot be reproduced from the shell, explain the exact UI steps and error text to check.
