"""CPU membership-replay oracle and conditional budget for depth-sort reuse.

Reads V3DSCN01 without Vulkan or training writeback. Captured memberships are
inputs to the oracle, not an independent reproduction of the coverage shader.
"""

from __future__ import annotations

import argparse
from array import array
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
import sys


HEADER = struct.Struct("<8s20I")
RANGE = struct.Struct("<4I")
UINT_MAX = (1 << 32) - 1
ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/training/core/gaussian_forward_renderer.cpp",
    "src/training/core/training_buffers.cpp",
    "src/training/core/training_scanline_capture.hpp",
    "shaders/training_shader/slang/passes/train_forward_tile_emit.comp.slang",
    "shaders/training_shader/slang/passes/train_forward_tile_gather_high.comp.slang",
    "third_party/vulkan_radix_sort/src/vk_radix_sort.h.in",
    "third_party/vulkan_radix_sort/src/shader/upsweep.slang",
    "third_party/vulkan_radix_sort/src/shader/downsweep.slang",
    "tests/training_sort_reuse_audit.py",
)


def depth_sort_key(bits: int) -> int:
    """Bit-exact floatDepthSortKey, including signed zero and NaN payloads."""
    if not 0 <= bits <= UINT_MAX:
        raise ValueError("Depth bits must fit uint32")
    return (~bits & UINT_MAX) if bits & 0x80000000 else bits ^ 0x80000000


def uints(data: bytes) -> array:
    result = array("I")
    result.frombytes(data)
    if sys.byteorder != "little":
        result.byteswap()
    return result


def uint_sha(values) -> str:
    packed = array("I", values)
    if sys.byteorder != "little":
        packed.byteswap()
    return hashlib.sha256(packed.tobytes()).hexdigest()


def verify_sort_inputs(path: Path, capture: Capture, max_bytes=256 * 1024 * 1024) -> dict:
    """Check original renderer count/emit arrays against the matching capture."""
    if path.stat().st_size > max_bytes:
        raise ValueError("Sort input exceeds byte budget")
    data = path.read_bytes()
    if len(data) < 104:
        raise ValueError("Truncated sort input header")
    magic, version, capacity = struct.unpack_from("<8s2I", data)
    with Path(capture.metadata["path"]).open("rb") as original:
        expected_header = original.read(HEADER.size)
        projected_bytes = original.read(48 * capture.membership.gaussian_count)
    g, items = capture.membership.gaussian_count, capture.membership.item_count
    if (magic != b"V3DSRT01" or version != 1 or capacity < items or
            data[16:104] != expected_header or len(data) != 104 + 56 * g + 16 * items):
        raise ValueError("Sort input header/capacity/length mismatch")
    if data[104:104 + 48 * g] != projected_bytes:
        raise ValueError("Sort input projected state mismatch")
    range_start = 104 + 48 * g
    range_end = range_start + 8 * g
    ranges = list(struct.iter_unpack("<2I", data[range_start:range_end]))
    arrays = [uints(data[range_end + 4 * items * i:range_end + 4 * items * (i + 1)])
              for i in range(4)]
    emit_ids, depth_keys, tile_keys, seeds = arrays
    cursor = 0
    membership = capture.membership
    for gaussian, (start, count) in enumerate(ranges):
        expected_tiles = membership.tiles[membership.offsets[gaussian]:membership.offsets[gaussian + 1]]
        if start != cursor or count != len(expected_tiles) or start + count > items:
            raise ValueError("Sort input Gaussian count/prefix mismatch")
        if sorted(tile_keys[start:start + count]) != list(expected_tiles):
            raise ValueError("Sort input emit membership mismatch")
        key = depth_sort_key(membership.depth_bits[gaussian])
        for i in range(start, start + count):
            if emit_ids[i] != gaussian or depth_keys[i] != key or seeds[i] != i:
                raise ValueError("Sort input emit ID/depth/seed mismatch")
        cursor += count
    if cursor != items:
        raise ValueError("Sort input does not cover items")
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                production_count_emit_matches=True, readonly_projected_matches=True,
                replay_capacity=capacity, historical_live_capacity=None,
                scope="Original renderer private replay; not historical live-buffer capacity")


@dataclass
class Membership:
    depth_bits: array
    offsets: array
    tiles: array
    tile_count: int

    @property
    def gaussian_count(self) -> int:
        return len(self.depth_bits)

    @property
    def item_count(self) -> int:
        return len(self.tiles)


@dataclass
class Capture:
    metadata: dict
    membership: Membership
    items: array
    ranges: list[tuple[int, int, int, int]]


def read_capture(path: Path, max_capture_bytes: int = 256 * 1024 * 1024) -> Capture:
    """Validate the file before allocation; accept the existing Fixed16 contract."""
    file_size = path.stat().st_size
    if file_size > max_capture_bytes:
        raise ValueError("Capture exceeds configured CPU input byte budget")
    with path.open("rb") as stream:
        header_bytes = stream.read(HEADER.size)
        if len(header_bytes) != HEADER.size:
            raise ValueError("Truncated capture header")
        magic, *fields = HEADER.unpack(header_bytes)
        (version, gaussians, iteration, frame, width, height, item_count, tile_count,
         composite, pixel, subgroup, global_grad, latency, ssim, cull, vendor,
         device, driver, reserved0, reserved1) = fields
        if (magic != b"V3DSCN01" or version != 1 or not width or not height or
                width * height > UINT_MAX or
                tile_count != ((width + 15) // 16) * ((height + 15) // 16) or
                (composite, pixel, subgroup, global_grad, latency, cull) != (1, 6, 1, 1, 0, 0) or
                ssim > 2 or reserved0 or reserved1):
            raise ValueError("Unsupported capture contract")
        items_offset = HEADER.size + 48 * gaussians
        ranges_offset = items_offset + 4 * item_count
        blends_offset = ranges_offset + RANGE.size * tile_count
        expected = blends_offset + 32 * width * height
        if expected != file_size:
            raise ValueError("Capture has truncated/trailing data")
        payload = header_bytes + stream.read()
    if len(payload) != expected:
        raise ValueError("Capture changed size during read")
    items = uints(payload[items_offset:ranges_offset])
    ranges = list(struct.iter_unpack("<4I", payload[ranges_offset:blends_offset]))
    counts = array("I", [0]) * gaussians
    cursor = 0
    for start, count, z, w in ranges:
        if z or w or (not count and start) or start + count > item_count:
            raise ValueError("Invalid Fixed16 range or nonzero padding")
        if not count:
            continue
        if start != cursor:
            raise ValueError("Ranges do not partition the item list")
        seen = set()
        for index in range(start, start + count):
            gaussian = items[index]
            if gaussian >= gaussians:
                raise ValueError("Invalid Gaussian ID")
            if gaussian in seen:
                raise ValueError("Duplicate Gaussian in one tile")
            seen.add(gaussian)
            counts[gaussian] += 1
        cursor += count
    if cursor != item_count:
        raise ValueError("Ranges do not cover all items")
    offsets = array("I", [0])
    total = 0
    for count in counts:
        total += count
        offsets.append(total)
    tiles = array("I", [0]) * item_count
    positions = array("I", offsets[:-1])
    for tile, (start, count, _, _) in enumerate(ranges):
        for index in range(start, start + count):
            gaussian = items[index]
            tiles[positions[gaussian]] = tile
            positions[gaussian] += 1
    depths = array("I", (struct.unpack_from("<I", payload, HEADER.size + 48 * g + 8)[0]
                         for g in range(gaussians)))
    return Capture(
        dict(path=str(path), sha256=hashlib.sha256(payload).hexdigest(), bytes=file_size,
             iteration=iteration, frame=frame, width=width, height=height,
             vendor_id=vendor, device_id=device, driver_version=driver, ssim_mode=ssim,
             pixel_mode=pixel, composite_mode=composite, tensor_subgroup=subgroup,
             tensor_global_pixel_grad=global_grad, rgb_latency=latency, pixel_cull=cull,
             device_uuid=None, compiler_flags=None, module_fingerprint=None),
        Membership(depths, offsets, tiles, tile_count), items, ranges)


def from_memberships(depth_bits, memberships, tile_count: int) -> Membership:
    """Build synthetic oracle input; tile enumeration need not be row-major."""
    if tile_count <= 0 or len(depth_bits) != len(memberships):
        raise ValueError("Invalid membership dimensions")
    offsets, tiles = array("I", [0]), array("I")
    for group in memberships:
        if len(set(group)) != len(group) or any(t < 0 or t >= tile_count for t in group):
            raise ValueError("Invalid or duplicate tile membership")
        tiles.extend(group)
        offsets.append(len(tiles))
    for bits in depth_bits:
        depth_sort_key(bits)
    return Membership(array("I", depth_bits), offsets, tiles, tile_count)


def _finish(ids: array, tiles: array, order: list[int], tile_count: int):
    result = array("I", (ids[i] for i in order))
    ranges = [(0, 0, 0, 0)] * tile_count
    cursor = 0
    while cursor < len(order):
        tile = tiles[order[cursor]]
        end = cursor + 1
        while end < len(order) and tiles[order[end]] == tile:
            end += 1
        ranges[tile] = (cursor, end - cursor, 0, 0)
        cursor = end
    return result, ranges


def baseline_lists(membership: Membership):
    """Original-ID emit, stable depth sort over I, then stable tile sort over I."""
    ids = array("I")
    for gaussian in range(membership.gaussian_count):
        count = membership.offsets[gaussian + 1] - membership.offsets[gaussian]
        ids.extend(array("I", [gaussian]) * count)
    keys = array("I", (depth_sort_key(bits) for bits in membership.depth_bits))
    order = list(range(membership.item_count))
    order.sort(key=lambda i: keys[ids[i]])
    order.sort(key=lambda i: membership.tiles[i])
    return _finish(ids, membership.tiles, order, membership.tile_count)


def reuse_lists(membership: Membership):
    """Stable compact, depth sort Nv, recompute offsets, expand, stable tile sort."""
    visible = [g for g in range(membership.gaussian_count)
               if membership.offsets[g + 1] != membership.offsets[g]]
    visible.sort(key=lambda g: depth_sort_key(membership.depth_bits[g]))
    sorted_offsets = array("I", [0])
    for gaussian in visible:
        count = membership.offsets[gaussian + 1] - membership.offsets[gaussian]
        sorted_offsets.append(sorted_offsets[-1] + count)
    ids, tiles = array("I"), array("I")
    for rank, gaussian in enumerate(visible):
        start, end = membership.offsets[gaussian:gaussian + 2]
        if len(ids) != sorted_offsets[rank]:
            raise ValueError("Sorted emit offset mismatch")
        ids.extend(array("I", [gaussian]) * (end - start))
        tiles.extend(membership.tiles[start:end])
    if len(ids) != membership.item_count:
        raise ValueError("Sorted emit count mismatch")
    order = list(range(len(ids)))
    order.sort(key=lambda i: tiles[i])
    result, ranges = _finish(ids, tiles, order, membership.tile_count)
    return result, ranges, visible, sorted_offsets


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def prefix_model(elements: int) -> dict:
    if not elements:
        return dict(dispatches=0, scratch_uints=0)
    levels = [ceil_div(elements, 256)]
    while levels[-1] > 256:
        levels.append(ceil_div(levels[-1], 256))
    return dict(dispatches=2 * len(levels) + 1, scratch_uints=sum(levels))


def sorter_storage(elements: int, alignment: int) -> int:
    """Source formula, not measured target allocation; alignment is explicit."""
    align = lambda x: ceil_div(x, alignment) * alignment
    histogram = align(4 * (4 + 4 * 256 + ceil_div(elements, 4096) * 256))
    return align(4) + histogram + 2 * align(4 * elements)


def budget(gaussians: int, visible: int, items: int, item_capacity: int,
           alignment: int = 256) -> dict:
    if (min(gaussians, visible, items, item_capacity) < 0 or
            visible > min(gaussians, items) or item_capacity < items or
            bool(visible) != bool(items) or alignment < 1 or alignment & (alignment - 1)):
        raise ValueError("Invalid workload/capacity/alignment budget")
    baseline_sort = 80 * items if items > 1 else 0
    candidate_depth_sort = 80 * visible if visible > 1 else 0
    unchanged_tile_sort = baseline_sort
    depth_saving = baseline_sort - candidate_depth_sort
    removed_high_gather = 12 * items if items > 1 else 0
    removed_emit_depth_key = 4 * items
    # Hypothetical no-alias design: rank scan + scatter; sorted counts + prefix.
    extra_payload = 28 * gaussians + 52 * visible if items else 0
    rank_prefix = prefix_model(gaussians)
    sorted_prefix = prefix_model(visible)
    extra_dispatches = (rank_prefix["dispatches"] + 1 + 1 +
                        sorted_prefix["dispatches"]) if items else 0
    removed_dispatches = int(items > 1)
    depth_fraction = (1 - visible / items) if items else None
    scenarios = []
    for share in (0.25, 0.5, 0.75):
        saved_ms = 0.78 * share * depth_fraction if items else 0.0
        scenarios.append(dict(assumed_depth_share_of_sort_ranges=share,
                              assumed_sort_ranges_ms=0.78,
                              linear_depth_sort_saving_ms=saved_ms,
                              max_added_ms_for_0_10ms_net=saved_ms - 0.10,
                              measured=False))
    return dict(
        gaussians=gaussians, visible_gaussians=visible, items=items,
        items_per_visible=items / visible if visible else None,
        depth_input_reduction_fraction=depth_fraction,
        all_sort_input_reduction_fraction=(items - visible) / (2 * items) if items else None,
        depth_partition_workgroups_per_pass=dict(baseline=ceil_div(items, 4096),
                                                 candidate=ceil_div(visible, 4096)),
        baseline_depth_sort_dispatches=12 if items > 1 else 0,
        candidate_depth_sort_dispatches=12 if visible > 1 else 0,
        unchanged_tile_sort_dispatches=12 if items > 1 else 0,
        extra_prepare_and_sorted_prefix_dispatches=extra_dispatches,
        removed_high_gather_dispatches=removed_dispatches,
        logical_payload_model=dict(
            baseline_depth_sort_bytes=baseline_sort,
            candidate_depth_sort_bytes=candidate_depth_sort,
            unchanged_tile_sort_bytes=unchanged_tile_sort,
            removed_depth_sort_bytes=depth_saving,
            removed_high_gather_bytes=removed_high_gather,
            removed_emit_depth_key_bytes=removed_emit_depth_key,
            added_rank_compact_and_sorted_prefix_core_bytes=extra_payload,
            modeled_core_net_bytes_saved=depth_saving + removed_high_gather +
            removed_emit_depth_key - extra_payload,
            formula="4 radix passes: each reads 4B key in upsweep, reads/writes "
                    "8B key/value in downsweep => 80N. Added core: 28G + 52Nv.",
            excluded="Histograms, block-scratch, source-level control/shared work, cache "
                     "and transaction amplification, barriers, fixed dispatch cost, "
                     "projection/coverage changes and downstream cache effects.",
            native_dram_bytes=None, time_prediction_ms=None),
        memory_model=dict(
            item_capacity=item_capacity, item_capacity_is_captured=False,
            alignment_scenario=alignment, target_alignment_known=False,
            existing_item_arrays_bytes=24 * item_capacity,
            existing_sort_storage_bytes=sorter_storage(items, alignment) if items > 1 else 0,
            candidate_extra_arrays_bytes=20 * gaussians,
            candidate_extra_prefix_scratch_bytes=4 * rank_prefix["scratch_uints"],
            candidate_private_count_bytes=16,
            candidate_extra_total_bytes=20 * gaussians + 4 * rank_prefix["scratch_uints"] + 16,
            extra_sort_storage_bytes=0,
            assumptions="Reserve rank(4G), depth key(4G), ID(4G), sorted offset/count(8G); "
                        "no alias with original arrays. Nv<=I permits sequential sorter scratch "
                        "reuse; a separate sorter allocation is not needed. Not total VRAM/RSS."),
        count_dependency="Nv must be generated on GPU before a direct Nv sort. A private "
                         "counter copied in the existing prepare submission can avoid an extra "
                         "submit, but is not implemented. Do not overwrite adaptive y/z counters "
                         "or the I counter. Sorted prefix must not overwrite I.",
        sensitivity_only=scenarios, measured_net_ms=None,
        full_training_speedup=None,
        required_timing="Complete count/prefix/compact/readback/sort/prefix/emit/tile-sort/"
                        "gather/ranges sequence, CPU submit/wait changes and downstream effects.")


def audit(path: Path, item_capacity: int | None = None, alignment: int = 256,
          max_capture_bytes: int = 256 * 1024 * 1024, sort_input: Path | None = None) -> dict:
    capture = read_capture(path, max_capture_bytes)
    membership = capture.membership
    baseline, baseline_ranges = baseline_lists(membership)
    if baseline != capture.items or baseline_ranges != capture.ranges:
        raise ValueError("Captured list disagrees with original stable depth/tile oracle")
    candidate, candidate_ranges, visible, offsets = reuse_lists(membership)
    if candidate != baseline or candidate_ranges != baseline_ranges:
        raise ValueError("Depth-sort reuse disagrees with baseline")
    equal_depth_neighbors = 0
    for start, count, _, _ in capture.ranges:
        for index in range(start + 1, start + count):
            a, b = capture.items[index - 1], capture.items[index]
            equal_depth_neighbors += membership.depth_bits[a] == membership.depth_bits[b]
    hashes = {}
    for name, values in (("items", candidate), ("ranges", (v for r in candidate_ranges for v in r)),
                         ("sorted_visible_ids", visible), ("sorted_emit_offsets", offsets)):
        hashes[name + "_sha256"] = uint_sha(values)
    raw = verify_sort_inputs(sort_input, capture, max_capture_bytes) if sort_input else None
    selected_capacity = item_capacity if item_capacity is not None else (
        raw["replay_capacity"] if raw else membership.item_count)
    cost = budget(membership.gaussian_count, len(visible), membership.item_count,
                  selected_capacity, alignment)
    cost["memory_model"]["item_capacity_source"] = (
        "explicit_scenario" if item_capacity is not None else
        "renderer_private_replay" if raw else "active_item_minimum_scenario")
    return dict(
        schema="depth_sort_reuse_cpu_audit_v1",
        status="CPU_ORACLE_PASSED_BUDGET_CONDITIONAL_NO_GPU_ADMISSION",
        gpu_executed=False, timing_measured=False, training_advanced=False,
        capture=capture.metadata,
        renderer_replay_inputs=raw,
        source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS},
        oracle=dict(baseline_matches_capture=True, candidate_matches_baseline=True,
                    all_range_components_match=True, equal_depth_adjacent_item_pairs=equal_depth_neighbors,
                    participating_gaussians=len(visible), **hashes,
                    scope="Recorded (Gaussian,tile) membership replay with original depth bits. "
                          "Per-Gaussian tile enumeration is normalized; one membership per tile "
                          "means this does not change final tie order. Geometry/count/emit shader "
                          "and GPU radix behavior are not independently validated."),
        budget=cost,
        missing_evidence=["Matching per-depth-sort and complete-sequence RTX3090 timings",
                          "Candidate native resource/traffic and synchronization checks",
                          "Matching target and optimizer state for downstream/full-step comparison"] +
                         ([] if raw else ["Raw renderer gaussianTileRanges/count and emit state"]),
        decision="Do not enable or integrate a GPU candidate from CPU replay alone. The "
                 "conditional operation budget is not a passed runtime/performance gate.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--item-capacity", type=int, help="Explicit scenario; not read from capture")
    parser.add_argument("--alignment", type=int, default=256, help="Storage alignment scenario")
    parser.add_argument("--sort-input", type=Path, help="Matching V3DSRT01 renderer count/emit replay")
    parser.add_argument("--max-capture-mib", type=int, default=256)
    args = parser.parse_args(argv)
    if (args.report.resolve() == args.capture.resolve() or args.report.exists() or
            (args.sort_input and args.report.resolve() == args.sort_input.resolve())):
        parser.error("Report must be a new file and must not overwrite capture")
    try:
        result = audit(args.capture, args.item_capacity, args.alignment,
                       args.max_capture_mib * 1024 * 1024, args.sort_input)
        with args.report.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Sort reuse audit failed: {error}\n")
    print(result["status"])
    print(f"G={result['budget']['gaussians']}, Nv={result['budget']['visible_gaussians']}, "
          f"I={result['budget']['items']}; report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
