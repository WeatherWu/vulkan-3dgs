#!/usr/bin/env python3
"""Analyze CSV from Export Adaptive Tile Diagnostics after a frozen benchmark.

    python scripts/analyze_adaptive_tiles.py capture.csv --json analysis.json
    python scripts/analyze_adaptive_tiles.py adaptive.csv --coarse-reference no_child.csv

Ranking uses the current geometric proxy and the same 25% child-item budget.
It is an offline hypothesis test, not a predicted GPU speedup. Only parents
classified by the configured threshold have four-child intersection counts.
The max0..max3 columns contain local child-list prefixes for selected parents,
and quadrant maxima in the coarse parent list for unselected parents. New
exports also include a .masks.bin sidecar. With a coarse reference, stable
mask filtering reconstructs child prefixes for all classified parents and
checks selected parents against their measured prefixes before using them.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import struct


BASE_META_FIELDS = ("frame", "iteration", "gaussians", "width", "height", "threshold", "pixel_mode")
META_FIELDS = BASE_META_FIELDS + ("child_batch",)
ROW_FIELDS = ("parent_index", "parent_x", "parent_y", "parent_items",
              "child0", "child1", "child2", "child3", "selected",
              "max0", "max1", "max2", "max3")
MASK_HEADER = struct.Struct("<8s10I")


@dataclass(frozen=True)
class Tile:
    index: int
    x: int
    y: int
    parent_items: int
    children: tuple[int, ...]
    selected: bool
    maxima: tuple[int, ...]

    @property
    def cost(self) -> int:
        return sum(self.children)

    @property
    def saved_tests(self) -> int:
        return 256 * self.parent_items - 64 * self.cost

    @property
    def profit(self) -> int:
        return self.saved_tests - 64 * self.cost - 4096

    @property
    def profitable(self) -> bool:
        return self.cost > 0 and self.profit > 0


def read_tiles(path: Path) -> tuple[dict[str, int], list[Tile]]:
    metadata: dict[str, int] = {}
    tiles: list[Tile] = []
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not set(BASE_META_FIELDS + ROW_FIELDS).issubset(reader.fieldnames or ()):
            raise ValueError("CSV is missing adaptive diagnostic columns")
        for row in reader:
            values = {name: int(row[name]) for name in BASE_META_FIELDS + ROW_FIELDS}
            values["child_batch"] = int(row.get("child_batch") or 8)
            if values["child_batch"] not in (8, 16, 32):
                raise ValueError("CSV contains an invalid Tensor child batch")
            if any(value < 0 for value in values.values()):
                raise ValueError("CSV contains negative counts or indices")
            current_meta = {name: values[name] for name in META_FIELDS}
            if metadata and metadata != current_meta:
                raise ValueError("CSV mixes benchmark metadata")
            metadata = current_meta
            width = (metadata["width"] + 15) // 16
            index = values["parent_index"]
            if width == 0 or index != len(tiles) or values["parent_x"] != index % width or values["parent_y"] != index // width:
                raise ValueError("CSV parent indices/coordinates are inconsistent")
            if values["selected"] not in (0, 1):
                raise ValueError("CSV contains an invalid selected flag")
            tile = Tile(index, values["parent_x"], values["parent_y"], values["parent_items"],
                        tuple(values[f"child{i}"] for i in range(4)),
                        bool(values["selected"]), tuple(values[f"max{i}"] for i in range(4)))
            limits = tile.children if tile.selected else (tile.parent_items,) * 4
            if any(maximum > limit for maximum, limit in zip(tile.maxima, limits)):
                raise ValueError("CSV processed prefix exceeds its list length")
            if tile.selected and not tile.profitable:
                raise ValueError("selected tile does not meet the current profitability rule")
            tiles.append(tile)
    if not tiles or metadata["height"] == 0:
        raise ValueError("CSV has no tiles or an empty extent")
    expected = ((metadata["width"] + 15) // 16) * ((metadata["height"] + 15) // 16)
    if len(tiles) != expected:
        raise ValueError(f"CSV has {len(tiles)} parents; extent requires {expected}")
    return metadata, tiles


def rank_selection(tiles: list[Tile], budget: int) -> list[Tile]:
    ranked = sorted((tile for tile in tiles if tile.profitable),
                    key=lambda tile: (-tile.profit / tile.cost, tile.index))
    result: list[Tile] = []
    used = 0
    for tile in ranked:
        if tile.cost <= budget - used:
            result.append(tile)
            used += tile.cost
    return result


def selection_stats(tiles: list[Tile]) -> dict[str, int]:
    return {"parents": len(tiles), "child_items": sum(tile.cost for tile in tiles),
            "geometric_saved_tests": sum(tile.saved_tests for tile in tiles),
            "proxy_profit": sum(tile.profit for tile in tiles)}


def analyze(metadata: dict[str, int], tiles: list[Tile]) -> dict:
    parent_items = sum(tile.parent_items for tile in tiles)
    budget = parent_items // 4
    selected = [tile for tile in tiles if tile.selected]
    profitable = [tile for tile in tiles if tile.profitable]
    ranked = rank_selection(tiles, budget)
    running = 0
    scan_ids = []
    for tile in profitable:
        if running + tile.cost <= budget:
            scan_ids.append(tile.index)
        running += tile.cost
    current = selection_stats(selected)
    alternative = selection_stats(ranked)
    if current["child_items"] > budget:
        raise ValueError("selected child items exceed the current budget")
    batch = metadata.get("child_batch", 8)
    batches = sum(sum((value + batch - 1) // batch for value in tile.maxima) if tile.selected
                  else (max(tile.maxima) + 15) // 16 for tile in tiles)
    return {"metadata": metadata, "parent_items": parent_items, "child_budget": budget,
            "profitable_parents": len(profitable),
            "profitable_rejected_parents": sum(not tile.selected for tile in profitable),
            "scan_selection_matches": scan_ids == [tile.index for tile in selected],
            "current": current, "ranked_proxy": alternative,
            "current_active_list_items": parent_items + sum(tile.cost - tile.parent_items for tile in selected),
            "current_tensor_batches": batches if metadata["pixel_mode"] == 6 else None,
            "selected_parent_ids": [tile.index for tile in selected],
            "ranked_parent_ids": [tile.index for tile in ranked]}


def compare_coarse_reference(metadata: dict[str, int], tiles: list[Tile],
                             reference_metadata: dict[str, int], reference: list[Tile]) -> dict:
    matching_fields = ("frame", "iteration", "gaussians", "width", "height", "pixel_mode")
    if any(metadata[name] != reference_metadata[name] for name in matching_fields):
        raise ValueError("coarse reference uses different benchmark metadata")
    if metadata["pixel_mode"] != 6:
        raise ValueError("observed Tensor work accounting requires VkSplat Tensor exports")
    if len(tiles) != len(reference) or any(tile.selected for tile in reference):
        raise ValueError("coarse reference must have the same extent and zero split parents")
    if any(tile.parent_items != coarse.parent_items for tile, coarse in zip(tiles, reference)):
        raise ValueError("coarse reference has different parent candidate counts")
    if any(not tile.selected and tile.maxima != coarse.maxima for tile, coarse in zip(tiles, reference)):
        raise ValueError("unselected parents differ from the coarse reference; check the frozen model")
    coarse_entries = sum(max(tile.maxima) for tile in reference)
    current_entries = sum(sum(tile.maxima) if tile.selected else max(tile.maxima) for tile in tiles)
    coarse_slots = 256 * coarse_entries
    current_slots = sum(64 * sum(tile.maxima) if tile.selected else 256 * max(tile.maxima) for tile in tiles)
    batch = metadata.get("child_batch", 8)
    return {"coarse_prefix_entries": coarse_entries, "current_prefix_entries": current_entries,
            "coarse_prefix_pixel_slots": coarse_slots, "current_prefix_pixel_slots": current_slots,
            "coarse_tensor_batches": sum((max(tile.maxima) + 15) // 16 for tile in reference),
            "current_tensor_batches": sum(sum((value + batch - 1) // batch for value in tile.maxima) if tile.selected
                                          else (max(tile.maxima) + 15) // 16 for tile in tiles)}


def read_masks(path: Path, metadata: dict[str, int], tiles: list[Tile]) -> bytes:
    data = path.read_bytes()
    if len(data) < MASK_HEADER.size:
        raise ValueError("truncated child-mask header")
    (magic, version, gaussians, iteration, frame, width, height, threshold,
     pixel_mode, items, parents) = MASK_HEADER.unpack_from(data)
    expected_meta = (metadata["gaussians"], metadata["iteration"], metadata["frame"],
                     metadata["width"], metadata["height"], metadata["threshold"], metadata["pixel_mode"])
    if magic != b"V3DMSK01" or version != 1:
        raise ValueError("unsupported child-mask format")
    if (gaussians, iteration, frame, width, height, threshold, pixel_mode) != expected_meta:
        raise ValueError("child masks and CSV use different benchmark metadata")
    if items != sum(tile.parent_items for tile in tiles) or parents != len(tiles):
        raise ValueError("child masks and CSV have different parent/item counts")
    if len(data) != MASK_HEADER.size + items:
        raise ValueError("child-mask file size does not match its header")
    masks = data[MASK_HEADER.size:]
    if any(mask > 15 for mask in masks):
        raise ValueError("child masks contain invalid bits")
    return masks


def predict_child_prefixes(metadata: dict[str, int], tiles: list[Tile],
                           coarse: list[Tile], masks: bytes) -> dict[int, tuple[int, ...]]:
    if len(masks) != sum(tile.parent_items for tile in tiles) or len(coarse) != len(tiles):
        raise ValueError("child-mask prediction input sizes differ")
    predictions = {}
    offset = 0
    for tile, reference in zip(tiles, coarse):
        span = masks[offset:offset + tile.parent_items]
        offset += tile.parent_items
        if tile.parent_items < metadata["threshold"]:
            if any(span):
                raise ValueError("unclassified parent has nonzero exported masks")
            continue
        counts = [0] * 4
        prefixes = [0] * 4
        for item, mask in enumerate(span):
            for child in range(4):
                if mask & (1 << child):
                    counts[child] += 1
                    if item < reference.maxima[child]:
                        prefixes[child] += 1
        if tuple(counts) != tile.children:
            raise ValueError(f"parent {tile.index}: masks do not match full child counts")
        prediction = tuple(prefixes)
        if tile.selected and prediction != tile.maxima:
            raise ValueError(f"parent {tile.index}: predicted child prefixes {prediction} "
                             f"differ from measured {tile.maxima}; do not use inferred costs")
        predictions[tile.index] = prediction
    return predictions


def analyze_prefix_choices(metadata: dict[str, int], tiles: list[Tile],
                            coarse: list[Tile], predictions: dict[int, tuple[int, ...]]) -> dict:
    budget = sum(tile.parent_items for tile in tiles) // 4
    batch = metadata.get("child_batch", 8)
    candidates = []
    for tile in tiles:
        if tile.index not in predictions or tile.cost == 0:
            continue
        child = predictions[tile.index]
        parent_prefix = max(coarse[tile.index].maxima)
        candidates.append({"parent_index": tile.index, "selected": tile.selected,
                           "child_items": tile.cost, "child_prefixes": child,
                           "pixel_slots_saved": 256 * parent_prefix - 64 * sum(child),
                           "prefix_entries_delta": sum(child) - parent_prefix,
                           "tensor_batches_delta": sum((value + batch - 1) // batch for value in child)
                               - (parent_prefix + 15) // 16})

    def choose(conservative: bool) -> list[int]:
        eligible = [item for item in candidates if item["pixel_slots_saved"] > 0 and
                    (not conservative or (item["prefix_entries_delta"] <= 0 and item["tensor_batches_delta"] <= 0))]
        eligible.sort(key=lambda item: (-item["pixel_slots_saved"] / item["child_items"], item["parent_index"]))
        used, result = 0, []
        for item in eligible:
            if used + item["child_items"] <= budget:
                result.append(item["parent_index"])
                used += item["child_items"]
        return result

    def work(ids: list[int]) -> dict:
        selected = set(ids)
        entries, slots, batches, items = 0, 0, 0, 0
        for tile in tiles:
            if tile.index in selected:
                child = predictions[tile.index]
                entries += sum(child)
                slots += 64 * sum(child)
                batches += sum((value + batch - 1) // batch for value in child)
                items += tile.cost
            else:
                prefix = max(coarse[tile.index].maxima)
                entries += prefix
                slots += 256 * prefix
                batches += (prefix + 15) // 16
        return {"parents": len(selected), "child_items": items, "prefix_entries": entries,
                "prefix_pixel_slots": slots, "tensor_batches": batches, "parent_ids": ids}

    geometric = [tile.index for tile in rank_selection(tiles, budget)]
    return {"classified_parents": len(predictions),
            "validated_selected_parents": sum(tile.selected for tile in tiles),
            "coarse": work([]),
            "current": work([tile.index for tile in tiles if tile.selected]),
            "geometric_ranked": work(geometric),
            "prefix_savings_ranked": work(choose(False)),
            "conservative": work(choose(True)),
            "candidate_costs": candidates}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--coarse-reference", type=Path,
                        help="same frozen model/frame exported with zero child, for observed prefix costs")
    parser.add_argument("--masks", type=Path,
                        help="normal adaptive .masks.bin; automatically detected beside the input CSV")
    parser.add_argument("--json", type=Path, help="save report and proposed parent IDs")
    args = parser.parse_args()
    try:
        metadata, tiles = read_tiles(args.capture)
        report = analyze(metadata, tiles)
        if args.coarse_reference:
            reference_metadata, reference = read_tiles(args.coarse_reference)
            report["coarse_comparison"] = compare_coarse_reference(
                metadata, tiles, reference_metadata, reference)
            mask_path = args.masks or args.capture.with_suffix(".masks.bin")
            if args.masks or mask_path.exists():
                predictions = predict_child_prefixes(metadata, tiles, reference,
                                                      read_masks(mask_path, metadata, tiles))
                report["mask_prefix_analysis"] = analyze_prefix_choices(metadata, tiles, reference, predictions)
        elif args.masks:
            raise ValueError("--masks requires --coarse-reference")
        print(f"Frame {metadata['frame']}, iteration {metadata['iteration']}, "
              f"{metadata['gaussians']} Gaussians, threshold {metadata['threshold']}, "
              f"child Tensor batch {metadata['child_batch']}")
        print(f"Profitable parents {report['profitable_parents']}; "
              f"rejected by budget {report['profitable_rejected_parents']}")
        print(f"Selection matches current scan-prefix rule: {report['scan_selection_matches']}")
        for label, key in (("Current", "current"), ("Ranked geometric proxy", "ranked_proxy")):
            item = report[key]
            print(f"{label}: {item['parents']} parents, "
                  f"{item['child_items']}/{report['child_budget']} child items, "
                  f"saved-test proxy {item['geometric_saved_tests']}, profit proxy {item['proxy_profit']}")
        current_profit = report["current"]["proxy_profit"]
        if current_profit:
            gain = report["ranked_proxy"]["proxy_profit"] / current_profit - 1
            print(f"Ranked proxy-profit change: {100 * gain:+.2f}% (not a GPU speedup)")
        print(f"Current active list items: {report['current_active_list_items']}")
        if report["current_tensor_batches"] is not None:
            print(f"Current Tensor batches after prefix trimming: {report['current_tensor_batches']}")
        if "coarse_comparison" in report:
            observed = report["coarse_comparison"]
            print("Observed coarse -> adaptive work (counts, not milliseconds):")
            for name in ("prefix_entries", "prefix_pixel_slots", "tensor_batches"):
                print(f"  {name}: {observed['coarse_' + name]} -> {observed['current_' + name]}")
        if "mask_prefix_analysis" in report:
            inferred = report["mask_prefix_analysis"]
            print(f"Mask prediction validated on {inferred['validated_selected_parents']} selected parents; "
                  f"prefixes inferred for {inferred['classified_parents']} classified parents")
            for policy in ("coarse", "current", "geometric_ranked", "prefix_savings_ranked", "conservative"):
                cost = inferred[policy]
                print(f"  {policy}: {cost['parents']} parents, {cost['child_items']} child items, "
                      f"entries {cost['prefix_entries']}, pixel slots {cost['prefix_pixel_slots']}, "
                      f"batches {cost['tensor_batches']}")
            print("These are work counts; classification, scatter and dispatch costs still require GPU timing.")
        else:
            print("Unselected child-prefix costs are unmeasured; export masks to infer them.")
        if args.json:
            if args.json.resolve() == args.capture.resolve():
                raise ValueError("--json must not overwrite the input CSV")
            args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
