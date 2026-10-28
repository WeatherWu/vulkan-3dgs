# Configure training in ImGui

Launch the application without arguments to use saved UI preferences:

```bash
build/bin/Release/vulkan-3dgs
```

On Windows, launch `build/bin/Release/vulkan-3dgs.exe`. Select the training GPU,
dataset and downscale in the training window. Select the training mode and
methods, then click **Start Training**. The CLI is optional for ordinary training.

## Sorting selection

**Training → Faster-GS Sorting (Experimental)** selects the experimental sorting
chain. It defaults off and is saved with the other training settings. Running
workers lock this selection; after pausing, it applies to the next start or
resume. Renderer rebuilds retain the choice. The forward renderer uses the
explicit training setting, without consulting an environment variable.

Only an effective Fixed16 tile mode activates Faster-GS sorting. Other effective
tile modes use original sorting. **Last executed sorting** shows the most recently
completed step, including a fallback reason; it shows **not executed** before the
first step. Changing the checkbox does not rewrite that observation or clear the
model. The experimental label does not establish a speed or quality improvement.

## Physical Z-ordering selection

**Training → Faster-GS Z-ordering (Experimental)** independently enables physical
model reordering. It defaults off, is saved, and is locked while training runs.
It moves parameters, Adam moments and ADC statistics together before displayed
steps 1, 5001, 10001 and 15001. The panel shows actual event count and latest
CPU/GPU cost. Disabling it stops future events without undoing the existing order;
Load Dataset creates a fresh comparison. Fixed Benchmark does not reorder.

See [physical Z-ordering](training-morton-ordering.md) for the pinned reference,
state contract, memory costs and correctness command. For a Z-order A/B, keep
Faster-GS Sorting and all other settings identical and toggle only Z-ordering.
Use independent PLY/quality filenames such as `bonsai_zorder_A_30000` and
`bonsai_zorder_B_30000`. Keep Z-ordering off when measuring sorting alone.

## Validation preferences

**Application Settings → Vulkan Validation Layers (next launch)** saves the
validation request for the next process. The initial default is on in both Debug
and Release. The panel separately displays the current request, its source and
whether validation is actually enabled. Changing the checkbox does not rebuild
the Vulkan instance. Close and reopen the program to apply the preference.

An explicit `--validation on|off` overrides the saved preference for that launch
without saving the override. Smoke diagnostics request core/synchronization
validation even when the saved preference is off; combining smoke options with
an explicit `--validation off` is rejected. Unavailable validation layers retain
their startup warning and actual-disabled status. Numeric training validation,
profiling and shader build flags are separate controls.

Application, viewer and training settings share the existing settings file.
The application owns `vulkan_validation`; training owns `faster_sort` and
`morton_ordering`. Each
settings writer preserves keys owned by the other domains. A failed application
settings save is displayed and the checkbox returns to its previous value.

Shader debug information is configured at build time:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DVULKAN_3DGS_TRAINING_SHADER_DEBUG_INFO=OFF
cmake --build build --config Release
```

## Compare two complete training runs

Use the same Release build for both runs. In Application Settings, disable
validation, restart without CLI overrides, and confirm **Current request: Off**
and **Actually enabled: Off**. The startup log must also show
`Vulkan validation layers: requested=0, enabled=0`.

Set bonsai, downscale 2, seed 0, 3DGS Random (30000 steps), Fixed16,
VkSplat Tensor with Subgroup reduction and Global pixel gradients, RGB Baseline
and 2D Loss + Separable Backward. Keep EarlyCull and Pure Training off and all
optimizer, densification, cache and profiling settings identical. Use the same
initialization and parameters as the paired correctness runs. Do not supply
smoke options or run a fixed benchmark before either training run.

1. Leave Faster-GS sorting off for A. Load the dataset to start with a fresh
   model, set **Output Folder** to `output` and **PLY Name** to
   `bonsai_A_30000.ply`, then click **Start Training**.
2. Wait for normal completion at 30000. Save the **Total training time**, final
   Gaussian count/loss and complete profiling screenshots. Pure Training is off,
   so export the model explicitly using the existing PLY export controls.
3. Close and reopen the program, preserve the configuration, enable Faster-GS
   sorting for B and use `bonsai_B_30000.ply`. Start from a fresh model and repeat
   the same collection. The last executed sorting must show Faster-GS on Fixed16.

The existing **Load Dataset** action also clears the model and timer. Resuming a
partially trained model with a different sorting setting is not an independent
A/B comparison. Preserve both output models and use new filenames for repeated
runs.

The application's training timer stops at worker completion, before model export.
Use that timer rather than application lifetime. Compare the entire sorting
scope: original prefix + emit + sort/ranges versus candidate prefix + sort/ranges;
candidate emit is already included in sort/ranges. Each path can produce a
different Gaussian trajectory, so retain final counts when comparing full-run
time. A single pair is an initial comparison, not a stable speedup estimate.

Evaluate both models with the same renderer and held-out views:

```bash
python -B tools/evaluate_psnr.py render \
  output/bonsai_A_30000.ply data/mipnerf360/bonsai \
  --factor 2 --test-every 8 --output-json quality_A_30000.json

python -B tools/evaluate_psnr.py render \
  output/bonsai_B_30000.ply data/mipnerf360/bonsai \
  --factor 2 --test-every 8 --output-json quality_B_30000.json
```

The evaluator uses filename-sorted held-out indices 0, 8, 16, etc. and gsplat
for both PLY files; this is a common external-renderer comparison. Preserve its
reported image directory and view list for matching the training split. If
evaluation dependencies are missing, retain the models and install dependencies
from `tools/requirements-psnr.txt` separately; training need not be repeated.
The evaluator is a separate command, not a training UI action. Export and
evaluation time are outside training time. Training-view loss alone does not
establish held-out quality.
