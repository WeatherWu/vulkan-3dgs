"""Validate baseline evidence and calculate conditional depth-reuse budgets (CPU only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


STAGES = ("count_clear_ms", "prefix_ms", "emit_ms", "depth_sort_ms", "high_gather_ms",
          "tile_sort_ms", "item_gather_ms", "ranges_ms", "prepare_gpu_ms", "main_gpu_ms")
RANGE_PARTS = ("range_boundaries_ms", "range_boundary_barrier_ms", "range_finalize_ms",
               "range_finalize_barrier_ms")
IDENTITY = ("device", "device_uuid", "vendor_id", "device_id", "driver_version", "cpp_build",
            "compiler", "shader_compiler", "sorter_header_sha256", "shader_fnv1a64",
            "source_capture_fnv1a64", "iteration", "frame", "width", "height", "gaussians",
            "items", "replay_capacity", "timestamp_valid_bits", "timestamp_period_ns")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "Missing/nonfinite/negative timing")
    return value


def summarize(values):
    return dict(mean_ms=statistics.mean(values), minimum_ms=min(values), maximum_ms=max(values),
                max_min_ratio=max(values) / min(values) if min(values) > 0 else None)


def evaluate(correctness, timing, audit):
    for report in (correctness, timing):
        require(report.get("schema") == "sort_baseline_replay_v1" and
                report.get("status") == "BASELINE_REPLAY_PASSED_NOT_CANDIDATE_ADMISSION",
                "Baseline report failed or unsupported schema")
        require(report.get("validation_errors") == 0 and
                report.get("gpu_count_emit_CPU_oracle_and_sorted_capture_match") is True and
                report.get("readonly_projected_match") is True and
                report.get("candidate_enabled") is False and report.get("training_advanced") is False,
                "Baseline correctness/state contract not satisfied")
    require(correctness.get("validation_enabled") is True and
            correctness.get("synchronization_validation_enabled") is True,
            "Correctness must pass core and synchronization validation")
    require(timing.get("validation_enabled") is False and
            timing.get("synchronization_validation_enabled") is False and
            timing.get("timing_requested") is True, "Timing must be separate and unvalidated")
    for key in IDENTITY:
        require(key in correctness and correctness[key] is not None and
                correctness[key] == timing.get(key), f"Evidence identity mismatch: {key}")
    require(bool(timing["device_uuid"]) and bool(timing["shader_fnv1a64"]), "Missing device/module identity")
    require(audit.get("schema") == "depth_sort_reuse_cpu_audit_v1" and
            audit.get("oracle", {}).get("candidate_matches_baseline") is True,
            "Missing membership oracle")
    raw = audit.get("renderer_replay_inputs") or {}
    require(raw.get("production_count_emit_matches") is True and
            raw.get("readonly_projected_matches") is True and
            raw.get("replay_capacity") == timing["replay_capacity"], "Missing/mismatched raw renderer inputs")
    workload = audit["budget"]
    require(workload["gaussians"] == timing["gaussians"] and workload["items"] == timing["items"],
            "Input audit workload mismatch")
    visible, items = workload["visible_gaussians"], timing["items"]
    require(0 < visible <= min(items, timing["gaussians"]), "Invalid participating count")
    require(timing.get("warmup_per_phase") == 2 and timing.get("measured_per_phase") == 8,
            "Unexpected sample budget")
    samples = timing.get("samples", [])
    require(len(samples) == 32, "All 32 baseline samples are required")
    phases = [[s for s in samples if s.get("phase") == p] for p in range(4)]
    require(all(len(p) == 8 for p in phases), "Missing baseline phase samples")
    has_split = all(all(k in s["timing"] and s["timing"][k] is not None for k in RANGE_PARTS)
                    for phase in phases[:3] for s in phase)
    any_split = any(any(k in s["timing"] for k in RANGE_PARTS)
                    for phase in phases[:3] for s in phase)
    require(not any_split or has_split, "Incomplete range split timestamps")
    summary, unstable = [], False
    for p, phase in enumerate(phases):
        fields = STAGES + RANGE_PARTS if p < 3 and has_split else STAGES if p < 3 else (
            "prepare_gpu_ms", "main_gpu_ms")
        require(all(s.get("segmented") is (p < 3) for s in phase), "Unexpected timestamp mode")
        for s in phase:
            t = s["timing"]
            for field in fields:
                number(t.get(field))
            if p < 3:
                require(abs(t["prepare_gpu_ms"] - t["count_clear_ms"] - t["prefix_ms"]) <= 1e-6,
                        "Prepare timestamp accounting mismatch")
                require(abs(t["main_gpu_ms"] - sum(t[k] for k in STAGES[2:8])) <= 1e-6,
                        "Main timestamp accounting mismatch")
                if has_split:
                    require(abs(t["ranges_ms"] - sum(t[k] for k in RANGE_PARTS)) <= 1e-6,
                            "Range timestamp accounting mismatch")
        stats = {k: summarize([s["timing"][k] for s in phase]) for k in fields}
        stats["active_sequence_ms"] = summarize([
            s["timing"]["prepare_gpu_ms"] + s["timing"]["main_gpu_ms"] for s in phase])
        unstable |= any(v["max_min_ratio"] is None or v["max_min_ratio"] > 1.20
                        for k, v in stats.items() if k in ("prepare_gpu_ms", "main_gpu_ms", "active_sequence_ms"))
        summary.append(stats)
    a1, a2 = [summary[p]["active_sequence_ms"]["mean_ms"] for p in (0, 2)]
    drift = 100 * abs(a2 - a1) / statistics.mean((a1, a2))
    segmented = [s["timing"] for phase in phases[:3] for s in phase]
    means = {k: statistics.mean(t[k] for t in segmented) for k in STAGES}
    active = means["prepare_gpu_ms"] + means["main_gpu_ms"]
    plain = summary[3]["active_sequence_ms"]["mean_ms"]
    require(active > 0 and means["emit_ms"] > 0, "Timing resolution cannot support this budget")
    target = max(.10, .05 * active)
    depth_saving = means["depth_sort_ms"] * (1 - visible / items)
    partial_gross = depth_saving + means["high_gather_ms"]
    scenarios = [dict(assumed_added_ms=added,
                      required_emit_saving_ms=max(0, target + added - partial_gross),
                      required_emit_saving_percent=100 * max(0, target + added - partial_gross) / means["emit_ms"])
                 for added in (0, .05, .10, .15, .20)]
    target_device = timing["vendor_id"] == 0x10DE and "RTX 3090" in timing["device"]
    status = "NON_TARGET_DEVICE" if not target_device else (
        "STOP_UNSTABLE_BASELINE" if unstable or drift > 2 else
        "NEEDS_RANGE_SPLIT" if not has_split else "BUDGET_REVIEW_REQUIRED")
    return dict(schema="sort_reuse_budget_v1", status=status, candidate_admitted=False,
                gpu_executed_by_analyzer=False, full_training_speedup=None,
                source_device=timing["device"], source_uuid=timing["device_uuid"],
                phases=summary, baseline_drift_percent=drift,
                boundary_only_difference_percent=100 * (plain / active - 1),
                mean_stages_ms=means, active_sequence_ms=active,
                range_share_percent=100 * means["ranges_ms"] / active,
                range_split_present=has_split, net_target_ms=target,
                conditional_model=dict(linear_depth_saving_ms=depth_saving,
                    removed_high_gather_ms=means["high_gather_ms"],
                    partial_gross_saving_ms=partial_gross,
                    remaining_emit_ms=means["emit_ms"],
                    added_cost_allowance_if_emit_unchanged_ms=partial_gross - target,
                    added_cost_allowance_if_emit_free_ms=partial_gross + means["emit_ms"] - target,
                    scenarios=scenarios,
                    assumptions="Depth time scales linearly with Nv/I; high gather is removed. "
                                "Partial gross excludes emit savings. Emit is not assumed to scale "
                                "with Nv/G because the I-item output is unchanged. Free emit is an "
                                "unattainable sensitivity bound, not a speed prediction."),
                decision="Do not implement/enable a candidate from this baseline alone. Review range "
                         "attribution and whether required emit savings plus new preparation cost "
                         "have a credible budget; no range optimization is included.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("correctness", "timing", "input-audit", "report"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    require(not args.report.exists(), "Report must be a new path")
    paths = (args.correctness, args.timing, args.input_audit)
    payloads = [p.read_bytes() for p in paths]
    result = evaluate(*(json.loads(data) for data in payloads))
    result["evidence"] = {str(p): hashlib.sha256(data).hexdigest() for p, data in zip(paths, payloads)}
    with args.report.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write("\n")
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
