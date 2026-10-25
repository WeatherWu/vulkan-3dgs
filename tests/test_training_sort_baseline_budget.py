import copy
import unittest

import training_sort_baseline_budget as budget


def evidence(split=False):
    correct = dict(schema="sort_baseline_replay_v1",
                   status="BASELINE_REPLAY_PASSED_NOT_CANDIDATE_ADMISSION",
                   validation_enabled=True, synchronization_validation_enabled=True,
                   validation_errors=0, gpu_count_emit_CPU_oracle_and_sorted_capture_match=True,
                   readonly_projected_match=True, candidate_enabled=False, training_advanced=False,
                   device="NVIDIA GeForce RTX 3090", device_uuid="test-uuid", vendor_id=0x10DE,
                   device_id=1, driver_version=1, cpp_build="Release", compiler="GNU",
                   shader_compiler="Slang", sorter_header_sha256="header-hash",
                   shader_fnv1a64={"range": "shader-hash"}, source_capture_fnv1a64="capture-hash",
                   iteration=8037, frame=0, width=32, height=16, gaussians=1200, items=1000,
                   replay_capacity=9600, timestamp_valid_bits=64, timestamp_period_ns=1)
    timing = dict(correct, validation_enabled=False, synchronization_validation_enabled=False,
                  timing_requested=True, warmup_per_phase=2, measured_per_phase=8, samples=[])
    base = dict(count_clear_ms=.12, prefix_ms=.06, emit_ms=.178, depth_sort_ms=.324,
                high_gather_ms=.039, tile_sort_ms=.324, item_gather_ms=.098, ranges_ms=4.89)
    base["prepare_gpu_ms"] = base["count_clear_ms"] + base["prefix_ms"]
    base["main_gpu_ms"] = sum(base[k] for k in budget.STAGES[2:8])
    if split:
        base.update(range_boundaries_ms=4.8, range_boundary_barrier_ms=.04,
                    range_finalize_ms=.04, range_finalize_barrier_ms=.01)
    for phase in range(4):
        for _ in range(8):
            values = {k: v * (1 + phase * .001) for k, v in base.items()}
            if phase == 3:
                values = {k: values[k] for k in ("prepare_gpu_ms", "main_gpu_ms")}
            timing["samples"].append(dict(phase=phase, segmented=phase < 3, timing=values))
    audit = dict(schema="depth_sort_reuse_cpu_audit_v1", oracle=dict(candidate_matches_baseline=True),
                 renderer_replay_inputs=dict(production_count_emit_matches=True,
                                            readonly_projected_matches=True, replay_capacity=9600),
                 budget=dict(gaussians=1200, visible_gaussians=300, items=1000))
    return copy.deepcopy((correct, timing, audit))


class BaselineBudgetTests(unittest.TestCase):
    def test_legacy_reports_require_range_split_without_candidate_admission(self):
        result = budget.evaluate(*evidence())
        self.assertEqual(result["status"], "NEEDS_RANGE_SPLIT")
        self.assertFalse(result["candidate_admitted"])
        self.assertIsNone(result["full_training_speedup"])
        self.assertAlmostEqual(result["net_target_ms"], .05 * result["active_sequence_ms"])

    def test_split_reports_check_accounting_and_remain_conditional(self):
        result = budget.evaluate(*evidence(split=True))
        self.assertEqual(result["status"], "BUDGET_REVIEW_REQUIRED")
        self.assertTrue(result["range_split_present"])
        self.assertEqual(result["phases"][0]["range_boundaries_ms"]["mean_ms"], 4.8)
        model = result["conditional_model"]
        self.assertGreater(model["remaining_emit_ms"], 0)
        self.assertAlmostEqual(model["linear_depth_saving_ms"], .324 * 1.001 * .7)
        self.assertLess(model["added_cost_allowance_if_emit_unchanged_ms"], 0)
        self.assertGreater(model["scenarios"][-1]["required_emit_saving_percent"], 100)

    def test_correctness_validation_cannot_be_skipped(self):
        for field, value in (("validation_enabled", False), ("synchronization_validation_enabled", False),
                             ("validation_errors", 1), ("candidate_enabled", True),
                             ("training_advanced", True), ("readonly_projected_match", False)):
            data = evidence()
            data[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                budget.evaluate(*data)

    def test_timing_with_validation_enabled_is_rejected(self):
        data = evidence()
        data[1]["validation_enabled"] = True
        with self.assertRaises(ValueError):
            budget.evaluate(*data)

    def test_different_gpu_modules_or_workload_are_rejected(self):
        for field in ("device_uuid", "shader_fnv1a64", "source_capture_fnv1a64", "replay_capacity"):
            data = evidence()
            data[1][field] = "different"
            with self.subTest(field=field), self.assertRaises(ValueError):
                budget.evaluate(*data)

    def test_missing_raw_input_verification_is_rejected(self):
        data = evidence()
        data[2]["renderer_replay_inputs"] = None
        with self.assertRaises(ValueError):
            budget.evaluate(*data)

    def test_missing_samples_and_wrong_modes_are_rejected(self):
        for missing in (True, False):
            data = evidence()
            if missing:
                data[1]["samples"].pop()
            else:
                data[1]["samples"][0]["segmented"] = False
            with self.assertRaises(ValueError):
                budget.evaluate(*data)

    def test_negative_nan_or_wrong_sums_are_rejected(self):
        for value in (-1, float("nan"), None, .7):
            data = evidence(split=True)
            data[1]["samples"][0]["timing"]["range_boundaries_ms"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                budget.evaluate(*data)

    def test_outlier_stops_the_budget_without_dropping_sample(self):
        data = evidence()
        sample = data[1]["samples"][0]["timing"]
        for key in sample:
            sample[key] *= 1.3
        result = budget.evaluate(*data)
        self.assertEqual(result["status"], "STOP_UNSTABLE_BASELINE")
        self.assertGreater(result["phases"][0]["active_sequence_ms"]["max_min_ratio"], 1.2)

    def test_drift_stops_the_budget(self):
        data = evidence()
        for sample in data[1]["samples"]:
            if sample["phase"] == 2:
                for key in sample["timing"]:
                    sample["timing"][key] *= 1.05
        self.assertEqual(budget.evaluate(*data)["status"], "STOP_UNSTABLE_BASELINE")

    def test_non_target_device_is_not_3090_evidence(self):
        data = evidence(split=True)
        for report in data[:2]:
            report["device"] = "AMD Radeon 780M"
            report["vendor_id"] = 0x1002
        self.assertEqual(budget.evaluate(*data)["status"], "NON_TARGET_DEVICE")


if __name__ == "__main__":
    unittest.main()
