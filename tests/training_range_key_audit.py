"""Independent CPU audit of direct sorted-key range construction; no Vulkan execution."""

from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path

import training_sort_reuse_audit as capture_audit


ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "src/training/core/gaussian_forward_renderer.cpp",
    "shaders/training_shader/slang/passes/train_forward_tile_range_boundaries.comp.slang",
    "shaders/training_shader/slang/passes/train_forward_tile_gather_high.comp.slang",
    "shaders/training_shader/slang/passes/train_forward_tile_sort.comp.slang",
    "third_party/vulkan_radix_sort/src/vk_radix_sort.h.in",
    "third_party/vulkan_radix_sort/include/vk_radix_sort.h",
)


def boundaries(count, tile_count, key_at):
    words = array("I", [0]) * (4 * tile_count)
    for index in range(count + 1):
        current = tile_count if index == count else key_at(index)
        previous = tile_count if index == 0 else key_at(index - 1)
        if index < count and current < tile_count and (index == 0 or current != previous):
            words[4 * current] = index
        if index > 0 and previous < tile_count and (index == count or current != previous):
            words[4 * previous + 1] = index
    return words


def finalize(words, count):
    result = array("I", words)
    for offset in range(0, len(result), 4):
        start, end = words[offset:offset + 2]
        if end <= start or count == 0:
            result[offset:offset + 4] = array("I", [0, 0, 0, 0])
        else:
            result[offset] = min(start, count)
            result[offset + 1] = min(end, count) - min(start, count)
    return result


def compare(original_keys, sorted_indices, sorted_keys, tile_count, count=None):
    count = len(sorted_indices) if count is None else count
    if tile_count <= 0 or count < 0 or count > min(len(original_keys), len(sorted_indices)):
        raise ValueError("Invalid active count/tile capacity")
    seen = bytearray(count)
    for index in range(count):
        source = sorted_indices[index]
        if not isinstance(source, int) or not 0 <= source < count or seen[source]:
            raise ValueError("Invalid sorted permutation")
        seen[source] = 1
        if not isinstance(original_keys[source], int) or not 0 <= original_keys[source] < tile_count:
            raise ValueError("Invalid tile key")
    if count > 1:
        if sorted_keys is None or len(sorted_keys) < count:
            raise ValueError("Sorted key producer missing")
        direct = sorted_keys
    else:
        # Production bypasses both radix calls for zero/one item; scratch is not produced.
        direct = original_keys
    for index in range(count):
        if not isinstance(direct[index], int) or direct[index] != original_keys[sorted_indices[index]] or (
                index and direct[index] < direct[index - 1]):
            raise ValueError("Sorted key/permutation relation is invalid")
    indirect_words = boundaries(count, tile_count, lambda index: original_keys[sorted_indices[index]])
    direct_words = boundaries(count, tile_count, lambda index: direct[index])
    if indirect_words != direct_words:
        raise ValueError("Range boundary mismatch")
    result = finalize(direct_words, count)
    # Independent histogram/prefix oracle, not the neighbor-boundary implementation.
    histogram = [0] * tile_count
    for index in range(count):
        histogram[direct[index]] += 1
    expected, cursor = array("I"), 0
    for size in histogram:
        expected.extend((cursor if size else 0, size, 0, 0))
        cursor += size
    if result != expected:
        raise ValueError("Independent histogram range oracle mismatch")
    return indirect_words, result


def address_blocks(indices, count, lanes=32, block_bytes=32):
    """Address grouping proxy, not native transactions, cache misses or warp mapping."""
    indirect = direct = groups = 0
    for start in range(0, count, lanes):
        end = min(start + lanes, count)
        first = max(0, start - 1)
        indirect += len({indices[i] * 4 // block_bytes for i in range(first, end)})
        direct += len({i * 4 // block_bytes for i in range(first, end)})
        groups += 1
    return dict(groups=groups, lanes=lanes, address_block_bytes=block_bytes,
                sum_indirect_distinct_key_blocks=indirect,
                sum_direct_distinct_key_blocks=direct,
                interpretation="Consecutive-item address-block proxy only; not native "
                               "transactions, residency, bandwidth or measured speedup")


def audit(capture_path, sort_input_path):
    capture = capture_audit.read_capture(capture_path)
    raw = capture_audit.verify_sort_inputs(sort_input_path, capture)
    payload = sort_input_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != raw["sha256"]:
        raise ValueError("Sort input changed during audit")
    g, count = capture.membership.gaussian_count, capture.membership.item_count
    offset = 104 + 56 * g
    arrays = [capture_audit.uints(payload[offset + 4 * count * i:offset + 4 * count * (i + 1)])
              for i in range(4)]
    emit_ids, depth_keys, tile_keys, seeds = arrays
    order = list(seeds)
    order.sort(key=lambda i: depth_keys[i])
    order.sort(key=lambda i: tile_keys[i])
    ordered_keys = array("I", (tile_keys[index] for index in order))
    words, ranges = compare(tile_keys, order, ordered_keys, capture.membership.tile_count)
    if array("I", (emit_ids[index] for index in order)) != capture.items:
        raise ValueError("Sorted Gaussian IDs differ from capture")
    if ranges != array("I", (value for row in capture.ranges for value in row)):
        raise ValueError("Direct-key ranges differ from capture")
    return dict(schema="range_sorted_key_cpu_audit_v1",
                status="CPU_EQUIVALENCE_PASSED_GPU_COMPARISON_REQUIRED",
                gpu_executed=False, timing_measured=False, training_advanced=False,
                source_capture=capture.metadata, raw_inputs=raw,
                items=count, tile_count=capture.membership.tile_count,
                original_untouched=True, all_boundary_words_match=True, all_final_range_words_match=True,
                boundaries_sha256=capture_audit.uint_sha(words),
                ranges_sha256=capture_audit.uint_sha(ranges),
                sorted_keys_sha256=capture_audit.uint_sha(ordered_keys),
                permutation_sha256=capture_audit.uint_sha(order),
                logical_access_model=dict(
                    key_queries=2 * count, indirect_uint_loads=4 * count, direct_uint_loads=2 * count,
                    indirect_source_bytes=16 * count, direct_source_bytes=8 * count,
                    removed_source_bytes=8 * count, additional_buffers=0, additional_dispatches=0,
                    changed_atomic_operations=0, native_traffic=None, measured_gain_ms=None,
                    limits="Source-level reference counts exclude compiler reuse, cache effects, "
                           "coalescing and scheduling. Same I+1 neighbor checks and range writes."),
                address_proxy=address_blocks(order, count),
                producer_contract="For I>=2, second 32-bit radix sort writes sorted keys back to "
                                  "tileSortScratch (binding23), not tileSortStorage. Four ping-pong "
                                  "passes end in the original key buffer; no later key writes occur "
                                  "before range construction. For I<=1 bind original tileKeyHigh; "
                                  "scratch is unproduced. Preserve existing compute dependencies.",
                limits="CPU reconstructs the producer relation from captured unsorted keys and "
                       "the library contract. It does not download/verify actual GPU sorted keys, "
                       "test a replacement shader or measure its speed. Not production admission.",
                source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--sort-input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists() or args.report.resolve() in (args.capture.resolve(), args.sort_input.resolve()):
        parser.error("Report must be a new file, not an input")
    result = audit(args.capture, args.sort_input)
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
