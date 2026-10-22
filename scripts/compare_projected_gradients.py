#!/usr/bin/env python3
"""Compare frozen benchmark projected and pixel gradients.

The reader accepts little-endian V3DGGRD1 versions 1 through 7. Header sizes
are 64, 68, 72, 76, 80, 84 and 84 bytes. Optional fields describe child batch,
Tensor reduction, pixel-gradient source, RGB schedule and SSIM mode. These
input-format fields do not imply that every captured method is selectable in
the current application. The writer emits version 7.

Payload: 12 float32 projected-gradient components and one membership byte per
Gaussian; versions 6 and 7 also contain width*height float4 pixel gradients.
Pixel RGB is active and W is padding. Version 6 permits SSIM IDs 0 and 1;
version 7 permits IDs 0, 1 and 2. Runtime SSIM selections are 0 and 2.

Freeze the model/frame, set warmup 0 and measured 1, export each path, and
compare using the default tolerances. A repeat must use the same actual path.
"""


from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import struct


HEADER = struct.Struct("<8s12I2f")
HEADER_V2 = struct.Struct("<8s13I2f")
HEADER_V3 = struct.Struct("<8s14I2f")
HEADER_V4 = struct.Struct("<8s15I2f")
HEADER_V5 = struct.Struct("<8s16I2f")
HEADER_V6 = struct.Struct("<8s17I2f")
HEADER_V7 = struct.Struct("<8s17I2f")
PIXEL_GRADIENT = struct.Struct("<4f")
GRADIENT = struct.Struct("<12f")
MAGIC = b"V3DGGRD1"
ACTIVE_COMPONENTS = (0, 1, 4, 5, 6, 7, 8, 9, 10)
COMPONENT_NAMES = (
    "center.x", "center.y", "conic.x", "conic.y", "conic.z",
    "opacity", "color.r", "color.g", "color.b",
)
TILE_NAMES = ("Fixed8", "Fixed16", "Adaptive Spatial")
PIXEL_NAMES = ("Auto", "Direct", "Workgroup Shared", "Subgroup", "Tile Gaussian Atomic",
               "VkSplat Per-Splat", "VkSplat Tensor")
SSIM_NAMES = ("2D", "Separable", "2D Loss + Separable Backward")


@dataclass(frozen=True)
class Capture:
    path: Path
    data: bytes
    gaussian_count: int
    iteration: int
    frame: int
    tile_mode: int
    pixel_mode: int
    composite_mode: int
    width: int
    height: int
    parent_items: int
    child_items: int
    split_parents: int
    loss: float
    alpha: float
    child_batch: int = 8
    header_size: int = HEADER.size
    tensor_reduction: bool = False
    tensor_global_pixel_grad: bool = False
    tensor_rgb_latency_hiding: bool = False
    ssim_mode: int = 0
    has_pixel_gradients: bool = False

    def gradient(self, index: int) -> tuple[float, ...]:
        return GRADIENT.unpack_from(self.data, self.header_size + index * GRADIENT.size)

    def child_touched(self, index: int) -> bool:
        return self.data[self.header_size + self.gaussian_count * GRADIENT.size + index] != 0

    def pixel_gradient(self, index: int) -> tuple[float, ...]:
        if not self.has_pixel_gradients:
            raise ValueError("pixel gradients require a version-6 or newer export")
        offset = self.header_size + self.gaussian_count * (GRADIENT.size + 1)
        return PIXEL_GRADIENT.unpack_from(self.data, offset + index * PIXEL_GRADIENT.size)


def tensor_label(capture: Capture, value: bool, enabled: str, disabled: str) -> str:
    return (enabled if value else disabled) if capture.pixel_mode == 6 else "N/A"


def read_capture(path: Path) -> Capture:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: truncated header")
    magic, version = struct.unpack_from("<8sI", data)
    if magic != MAGIC or version not in (1, 2, 3, 4, 5, 6, 7):
        raise ValueError(f"{path}: unsupported gradient export format")
    header = {1: HEADER, 2: HEADER_V2, 3: HEADER_V3, 4: HEADER_V4, 5: HEADER_V5,
              6: HEADER_V6, 7: HEADER_V7}[version]
    if len(data) < header.size:
        raise ValueError(f"{path}: truncated version-{version} header")
    values = list(header.unpack_from(data))
    ssim_mode = values.pop(-3) if version >= 6 else 0
    if ssim_mode not in ((0, 1, 2) if version >= 7 else (0, 1)):
        raise ValueError(f"{path}: invalid SSIM mode")
    tensor_rgb_latency_hiding = values.pop(-3) if version >= 5 else 0
    tensor_global_pixel_grad = values.pop(-3) if version >= 4 else 0
    tensor_reduction = values.pop(-3) if version >= 3 else 0
    child_batch = values.pop(-3) if version >= 2 else 8
    if tensor_reduction not in (0, 1):
        raise ValueError(f"{path}: invalid Tensor reduction flag")
    if tensor_global_pixel_grad not in (0, 1):
        raise ValueError(f"{path}: invalid Tensor pixel-gradient source")
    if tensor_global_pixel_grad and not tensor_reduction:
        raise ValueError(f"{path}: Global Tensor pixel-gradient source requires subgroup reduction")
    if tensor_rgb_latency_hiding not in (0, 1):
        raise ValueError(f"{path}: invalid Tensor RGB schedule")
    if tensor_rgb_latency_hiding and not tensor_global_pixel_grad:
        raise ValueError(f"{path}: Tensor RGB latency hiding requires Global pixel gradients")
    if child_batch not in (8, 16, 32):
        raise ValueError(f"{path}: invalid Tensor child batch size")
    (magic, version, count, iteration, frame, tile_mode, pixel_mode,
     composite_mode, width, height, parent_items, child_items,
     split_parents, loss, alpha) = values
    if not math.isfinite(loss) or not math.isfinite(alpha):
        raise ValueError(f"{path}: non-finite forward loss/alpha")
    if (count == 0 or tile_mode >= len(TILE_NAMES) or pixel_mode >= len(PIXEL_NAMES)):
        raise ValueError(f"{path}: invalid Gaussian count, tile mode or pixel backward mode")
    if tensor_reduction and (tile_mode != 1 or pixel_mode != 6):
        raise ValueError(f"{path}: subgroup Tensor reduction requires Fixed16 + VkSplat Tensor")
    expected_size = header.size + count * (GRADIENT.size + 1)
    if version >= 6:
        if width == 0 or height == 0:
            raise ValueError(f"{path}: invalid pixel gradient extent")
        expected_size += width * height * PIXEL_GRADIENT.size
    if len(data) != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, found {len(data)}")
    membership = data[header.size + count * GRADIENT.size:header.size + count * (GRADIENT.size + 1)]
    if any(value not in (0, 1) for value in membership):
        raise ValueError(f"{path}: invalid child-membership bytes")
    if tile_mode != 2 and (child_items != 0 or split_parents != 0 or any(membership)):
        raise ValueError(f"{path}: fixed tile export unexpectedly contains child items")
    if tile_mode == 2:
        if (child_items == 0) != (split_parents == 0):
            raise ValueError(f"{path}: inconsistent child and split-parent counts")
        if child_items > 0 and not any(membership):
            raise ValueError(f"{path}: child items have no member Gaussians")
    return Capture(path, data, count, iteration, frame, tile_mode, pixel_mode,
                   composite_mode, width, height, parent_items, child_items,
                   split_parents, loss, alpha, child_batch, header.size,
                   tensor_reduction=bool(tensor_reduction),
                   tensor_global_pixel_grad=bool(tensor_global_pixel_grad),
                   tensor_rgb_latency_hiding=bool(tensor_rgb_latency_hiding),
                   ssim_mode=ssim_mode, has_pixel_gradients=version >= 6)


def check_pair(reference: Capture, candidate: Capture, forward_atol: float,
               allow_pixel_mode_change: bool = False) -> None:
    fields = ("gaussian_count", "iteration", "frame", "composite_mode", "width", "height")
    if not allow_pixel_mode_change:
        fields += ("pixel_mode",)
    mismatches = [name for name in fields
                  if getattr(reference, name) != getattr(candidate, name)]
    if mismatches:
        raise ValueError("captures differ in " + ", ".join(mismatches))
    if (abs(reference.loss - candidate.loss) > forward_atol or
            abs(reference.alpha - candidate.alpha) > forward_atol):
        raise ValueError("forward loss/alpha differ beyond --forward-atol; "
                         "check that both runs use the same frozen model and frame")


@dataclass
class Stats:
    gaussians: int = 0
    components: int = 0
    mismatched_components: int = 0
    nonfinite_components: int = 0
    missing_gaussians: int = 0
    extra_gaussians: int = 0
    reference_sq: float = 0.0
    candidate_sq: float = 0.0
    difference_sq: float = 0.0
    dot: float = 0.0
    max_abs: float = 0.0
    max_index: int = 0
    max_component: str = ""

    @property
    def relative_l2(self) -> float:
        if self.reference_sq == 0.0:
            return 0.0 if self.difference_sq == 0.0 else math.inf
        return math.sqrt(self.difference_sq / self.reference_sq)

    @property
    def cosine(self) -> float:
        scale = math.sqrt(self.reference_sq * self.candidate_sq)
        return self.dot / scale if scale else math.nan


def compare(reference: Capture, candidate: Capture, atol: float, rtol: float,
            zero_threshold: float) -> tuple[Stats, Stats]:
    overall, child = Stats(), Stats()
    touched_source = candidate if candidate.tile_mode == 2 else reference
    for index in range(reference.gaussian_count):
        ref = reference.gradient(index)
        got = candidate.gradient(index)
        groups = (overall, child) if touched_source.child_touched(index) else (overall,)
        ref_sq = 0.0
        got_sq = 0.0
        for group in groups:
            group.gaussians += 1
        for component, name in zip(ACTIVE_COMPONENTS, COMPONENT_NAMES):
            a, b = ref[component], got[component]
            for group in groups:
                group.components += 1
                if not math.isfinite(a) or not math.isfinite(b):
                    group.nonfinite_components += 1
                    continue
                diff = a - b
                group.reference_sq += a * a
                group.candidate_sq += b * b
                group.difference_sq += diff * diff
                group.dot += a * b
                if abs(diff) > atol + rtol * max(abs(a), abs(b)):
                    group.mismatched_components += 1
                if abs(diff) > group.max_abs:
                    group.max_abs = abs(diff)
                    group.max_index = index
                    group.max_component = name
            if math.isfinite(a) and math.isfinite(b):
                ref_sq += a * a
                got_sq += b * b
        for group in groups:
            if math.sqrt(ref_sq) > zero_threshold and math.sqrt(got_sq) <= zero_threshold:
                group.missing_gaussians += 1
            elif math.sqrt(ref_sq) <= zero_threshold and math.sqrt(got_sq) > zero_threshold:
                group.extra_gaussians += 1
    return overall, child


def print_stats(label: str, stats: Stats) -> None:
    if stats.gaussians == 0:
        print(f"{label}: no Gaussians in this group")
        return
    print(f"{label}: {stats.gaussians} Gaussians, relative L2 {stats.relative_l2:.6g}, "
          f"cosine {stats.cosine:.8g}")
    print(f"  differing components {stats.mismatched_components}/{stats.components}, "
          f"non-finite {stats.nonfinite_components}, "
          f"missing/extra Gaussian gradients {stats.missing_gaussians}/{stats.extra_gaussians}")
    print(f"  max |difference| {stats.max_abs:.6g} at Gaussian {stats.max_index} "
          f"({stats.max_component})")


def compare_pixels(reference: Capture, candidate: Capture, atol: float, rtol: float) -> Stats:
    if not reference.has_pixel_gradients or not candidate.has_pixel_gradients:
        raise ValueError("pixel comparison needs two version-6 or newer exports")
    if (reference.width, reference.height) != (candidate.width, candidate.height):
        raise ValueError("pixel gradient extents differ")
    stats = Stats()
    for index in range(reference.width * reference.height):
        stats.gaussians += 1  # Reuse the numeric accumulator; this count denotes pixels.
        for component, (a, b) in enumerate(zip(reference.pixel_gradient(index)[:3],
                                              candidate.pixel_gradient(index)[:3])):
            stats.components += 1
            if not math.isfinite(a) or not math.isfinite(b):
                stats.nonfinite_components += 1
                continue
            difference = abs(a - b)
            stats.reference_sq += a * a
            stats.candidate_sq += b * b
            stats.difference_sq += difference * difference
            stats.dot += a * b
            if difference > atol + rtol * max(abs(a), abs(b)):
                stats.mismatched_components += 1
            if difference > stats.max_abs:
                stats.max_abs, stats.max_index = difference, index
                stats.max_component = "rgb"[component]
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixed", type=Path, help="fixed-mode benchmark export")
    parser.add_argument("candidate", type=Path, help="candidate benchmark export")
    parser.add_argument("--repeat-fixed", type=Path,
                        help="optional second fixed export to measure atomic-add noise")
    parser.add_argument("--allow-pixel-mode-change", action="store_true",
                        help="explicitly compare different backward kernels, e.g. Direct and Tensor")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--zero-threshold", type=float, default=1e-8)
    parser.add_argument("--forward-atol", type=float, default=1e-4)
    parser.add_argument("--pixel-atol", type=float, default=1e-9,
                        help="absolute pixel-gradient tolerance (already divided by pixel count)")
    args = parser.parse_args()
    try:
        fixed, candidate = read_capture(args.fixed), read_capture(args.candidate)
        check_pair(fixed, candidate, args.forward_atol, args.allow_pixel_mode_change)
        print(f"Model: {fixed.gaussian_count} Gaussians, iteration {fixed.iteration}, "
              f"frame {fixed.frame}, {fixed.width}x{fixed.height}")
        print(f"Modes: {TILE_NAMES[fixed.tile_mode]} -> {TILE_NAMES[candidate.tile_mode]}; "
              f"split parents {candidate.split_parents}, child items {candidate.child_items}")
        print(f"Pixel backward modes: {PIXEL_NAMES[fixed.pixel_mode]} ({fixed.pixel_mode}) -> "
              f"{PIXEL_NAMES[candidate.pixel_mode]} ({candidate.pixel_mode})")
        if fixed.tile_mode == 2 or candidate.tile_mode == 2:
            print(f"Adaptive Tensor child batch: {fixed.child_batch} -> {candidate.child_batch}")
        if ((fixed.tile_mode == 1 and fixed.pixel_mode == 6) or
                (candidate.tile_mode == 1 and candidate.pixel_mode == 6)):
            print(f"Fixed16 Tensor reduction: "
                  f"{tensor_label(fixed, fixed.tensor_reduction, 'Subgroup', 'Shared')} -> "
                  f"{tensor_label(candidate, candidate.tensor_reduction, 'Subgroup', 'Shared')}")
            print(f"Fixed16 Tensor pixel-gradient source: "
                  f"{tensor_label(fixed, fixed.tensor_global_pixel_grad, 'Global', 'Shared')} -> "
                  f"{tensor_label(candidate, candidate.tensor_global_pixel_grad, 'Global', 'Shared')}")
            if fixed.tensor_global_pixel_grad or candidate.tensor_global_pixel_grad:
                print(f"Fixed16 Tensor RGB schedule: "
                      f"{tensor_label(fixed, fixed.tensor_rgb_latency_hiding, 'Latency Hiding', 'Baseline')} -> "
                      f"{tensor_label(candidate, candidate.tensor_rgb_latency_hiding, 'Latency Hiding', 'Baseline')}")
        print(f"Forward difference: loss {abs(fixed.loss - candidate.loss):.6g}, "
              f"alpha {abs(fixed.alpha - candidate.alpha):.6g}")
        print(f"SSIM: {SSIM_NAMES[fixed.ssim_mode]} -> {SSIM_NAMES[candidate.ssim_mode]}")
        if fixed.has_pixel_gradients and candidate.has_pixel_gradients:
            pixels = compare_pixels(fixed, candidate, args.pixel_atol, args.rtol)
            print(f"Pixel gradients: relative L2 {pixels.relative_l2:.6g}, cosine {pixels.cosine:.8g}, "
                  f"differing components {pixels.mismatched_components}/{pixels.components}, "
                  f"non-finite {pixels.nonfinite_components}, max |difference| {pixels.max_abs:.6g} "
                  f"at pixel {pixels.max_index} ({pixels.max_component})")
        else:
            print("Pixel gradients: unavailable in legacy exports (need v6+ on both sides)")
        overall, child = compare(fixed, candidate, args.atol, args.rtol,
                                 args.zero_threshold)
        print_stats("All", overall)
        print_stats("Child-touched", child)
        if args.repeat_fixed:
            repeat = read_capture(args.repeat_fixed)
            check_pair(fixed, repeat, args.forward_atol)
            if repeat.tile_mode != fixed.tile_mode:
                raise ValueError("--repeat-fixed must use the same tile mode as the fixed export")
            if (repeat.tensor_reduction != fixed.tensor_reduction or
                    repeat.child_batch != fixed.child_batch):
                raise ValueError("--repeat-fixed must use the same Tensor reduction and child batch")
            if repeat.tensor_global_pixel_grad != fixed.tensor_global_pixel_grad:
                raise ValueError("--repeat-fixed must use the same Tensor pixel-gradient source")
            if repeat.tensor_rgb_latency_hiding != fixed.tensor_rgb_latency_hiding:
                raise ValueError("--repeat-fixed must use the same Tensor RGB schedule")
            if repeat.ssim_mode != fixed.ssim_mode:
                raise ValueError("--repeat-fixed must use the same SSIM mode")
            noise, _ = compare(fixed, repeat, args.atol, args.rtol,
                               args.zero_threshold)
            print_stats("Fixed repeat (noise floor)", noise)
            if noise.relative_l2:
                print(f"Candidate / repeat relative-L2 ratio: "
                      f"{overall.relative_l2 / noise.relative_l2:.6g}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
