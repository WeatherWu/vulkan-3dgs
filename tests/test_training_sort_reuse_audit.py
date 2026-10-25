from array import array
import contextlib
import io
import json
from pathlib import Path
import random
import struct
import tempfile
import unittest

import training_sort_reuse_audit as sort_audit


def capture_bytes(bits, groups, width=32, height=16, *, items_override=None):
    tiles = ((width + 15) // 16) * ((height + 15) // 16)
    membership = sort_audit.from_memberships(bits, groups, tiles)
    items, ranges = sort_audit.baseline_lists(membership)
    if items_override is not None:
        items = array("I", items_override)
    header = sort_audit.HEADER.pack(b"V3DSCN01", 1, len(bits), 8037, 0, width, height,
                                   len(items), tiles, 1, 6, 1, 1, 0, 2, 0, 0x10DE,
                                   0x2204, 123, 0, 0)
    projected = bytearray(48 * len(bits))
    for gaussian, depth in enumerate(bits):
        struct.pack_into("<I", projected, 48 * gaussian + 8, depth)
    return (header + projected + struct.pack(f"<{len(items)}I", *items) +
            b"".join(sort_audit.RANGE.pack(*r) for r in ranges) +
            bytes(32 * width * height))


class SortReuseOracleTests(unittest.TestCase):
    def assert_oracles_match(self, bits, groups, tiles):
        membership = sort_audit.from_memberships(bits, groups, tiles)
        baseline, ranges = sort_audit.baseline_lists(membership)
        candidate, candidate_ranges, visible, offsets = sort_audit.reuse_lists(membership)
        self.assertEqual(candidate, baseline)
        self.assertEqual(candidate_ranges, ranges)
        self.assertEqual(offsets[-1], len(baseline))
        self.assertEqual(len(visible), sum(bool(g) for g in groups))
        return candidate, ranges, visible, offsets

    def test_empty_and_singleton(self):
        self.assert_oracles_match([], [], 3)
        self.assert_oracles_match([0x3F800000], [[]], 3)
        items, ranges, _, _ = self.assert_oracles_match([0x3F800000], [[2]], 3)
        self.assertEqual(list(items), [0])
        self.assertEqual(ranges, [(0, 0, 0, 0), (0, 0, 0, 0), (0, 1, 0, 0)])

    def test_recomputed_prefix_not_original_id_prefix(self):
        _, _, visible, offsets = self.assert_oracles_match(
            [0x40000000, 0x3F800000, 0], [[2, 0], [1], []], 3)
        self.assertEqual(visible, [1, 0])
        self.assertEqual(list(offsets), [0, 1, 3])

    def test_ties_require_original_gaussian_order(self):
        bits, groups = [0x3F800000] * 5, [[1, 0], [], [0, 1], [1], [0]]
        items, ranges, visible, _ = self.assert_oracles_match(bits, groups, 3)
        self.assertEqual(visible, [0, 2, 3, 4])
        self.assertEqual(list(items), [0, 2, 4, 0, 2, 3])
        self.assertEqual(ranges[2], (0, 0, 0, 0))
        # An unordered compaction remains unordered after a stable equal-key sort.
        wrong = list(reversed(visible))
        wrong.sort(key=lambda g: sort_audit.depth_sort_key(bits[g]))
        self.assertNotEqual(wrong, visible)

    def test_depth_transform_preserves_raw_exceptional_bits(self):
        bits = [0xFFC00001, 0xFF800000, 0xBF800000, 0x80000000, 0,
                0x3F800000, 0x7F800000, 0x7FC00001, 0x7FC00002]
        self.assertEqual([sort_audit.depth_sort_key(x) for x in bits],
                         sorted(sort_audit.depth_sort_key(x) for x in bits))
        self.assert_oracles_match(bits, [[1, 0]] * len(bits), 2)
        self.assertNotEqual(sort_audit.depth_sort_key(0x80000000), sort_audit.depth_sort_key(0))
        with self.assertRaises(ValueError):
            sort_audit.depth_sort_key(1 << 32)

    def test_deterministic_random_and_partition_boundaries(self):
        rng = random.Random(10091)
        for size in (0, 1, 31, 32, 33, 255, 256, 257, 4095, 4096, 4097):
            with self.subTest(size=size):
                bits = [rng.choice([0, 0x80000000, 0x3F800000, rng.getrandbits(32)])
                        for _ in range(size)]
                groups = [rng.sample(range(9), rng.randrange(5)) for _ in range(size)]
                self.assert_oracles_match(bits, groups, 9)

    def test_invalid_memberships(self):
        for bits, groups, tiles in (([0], [[0, 0]], 1), ([0], [[1]], 1),
                                     ([0], [[-1]], 1), ([], [[0]], 1), ([], [], 0)):
            with self.assertRaises(ValueError):
                sort_audit.from_memberships(bits, groups, tiles)


class SortReuseCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "capture.bin"
        self.data = capture_bytes([0x40000000, 0x3F800000, 0x3F800000, 0],
                                  [[0, 1], [0], [1, 0], []])
        self.path.write_bytes(self.data)

    def test_reader_and_audit_do_not_modify_input(self):
        result = sort_audit.audit(self.path, item_capacity=16)
        self.assertTrue(result["oracle"]["baseline_matches_capture"])
        self.assertTrue(result["oracle"]["candidate_matches_baseline"])
        self.assertEqual(result["budget"]["visible_gaussians"], 3)
        self.assertIsNone(result["budget"]["measured_net_ms"])
        self.assertFalse(result["gpu_executed"])
        self.assertEqual(self.path.read_bytes(), self.data)

    def test_reject_truncated_trailing_and_input_budget(self):
        for payload in (b"", self.data[:40], self.data[:-1], self.data + b"x"):
            self.path.write_bytes(payload)
            with self.assertRaises(ValueError):
                sort_audit.read_capture(self.path)
        self.path.write_bytes(self.data)
        with self.assertRaisesRegex(ValueError, "byte budget"):
            sort_audit.read_capture(self.path, max_capture_bytes=100)

    def test_reject_wrong_header_and_unrealistic_counts_before_allocation(self):
        for field, value in ((0, 2), (1, sort_audit.UINT_MAX), (4, 0), (10, 0), (19, 1)):
            payload = bytearray(self.data)
            struct.pack_into("<I", payload, 8 + 4 * field, value)
            self.path.write_bytes(payload)
            with self.assertRaises(ValueError):
                sort_audit.read_capture(self.path)

    def test_reject_bad_id_duplicate_range_and_padding(self):
        item_offset = sort_audit.HEADER.size + 48 * 4
        range_offset = item_offset + 4 * 5
        variants = [(item_offset, 4), (item_offset + 4, 1),
                    (range_offset, 1), (range_offset + 8, 1), (range_offset + 4, 99)]
        for offset, value in variants:
            with self.subTest(offset=offset, value=value):
                payload = bytearray(self.data)
                struct.pack_into("<I", payload, offset, value)
                self.path.write_bytes(payload)
                with self.assertRaises(ValueError):
                    sort_audit.read_capture(self.path)

    def test_reader_accepts_memberships_but_audit_rejects_wrong_tie_order(self):
        payload = bytearray(self.data)
        item_offset = sort_audit.HEADER.size + 48 * 4
        struct.pack_into("<2I", payload, item_offset, 2, 1)
        self.path.write_bytes(payload)
        sort_audit.read_capture(self.path)
        with self.assertRaisesRegex(ValueError, "stable depth/tile oracle"):
            sort_audit.audit(self.path)

    def test_empty_capture_and_edge_extent(self):
        for width, height in ((1, 1), (17, 17)):
            self.path.write_bytes(capture_bytes([], [], width, height))
            result = sort_audit.audit(self.path)
            self.assertEqual(result["budget"]["items"], 0)

    def test_cli_exclusive_output_and_no_capture_overwrite(self):
        report = Path(self.temp.name) / "result.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sort_audit.main([str(self.path), "--report", str(report)]), 0)
        self.assertEqual(json.loads(report.read_text())["schema"], "depth_sort_reuse_cpu_audit_v1")
        for output in (report, self.path):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                sort_audit.main([str(self.path), "--report", str(output)])
        self.assertEqual(self.path.read_bytes(), self.data)

    def sort_input_bytes(self):
        projected = self.data[sort_audit.HEADER.size:sort_audit.HEADER.size + 48 * 4]
        ranges = b"".join(struct.pack("<2I", *r) for r in ((0, 2), (2, 1), (3, 2), (5, 0)))
        depth2, depth1 = sort_audit.depth_sort_key(0x40000000), sort_audit.depth_sort_key(0x3F800000)
        arrays = ((0, 0, 1, 2, 2), (depth2, depth2, depth1, depth1, depth1),
                  (0, 1, 0, 1, 0), (0, 1, 2, 3, 4))
        return (struct.pack("<8s2I", b"V3DSRT01", 1, 8) + self.data[:sort_audit.HEADER.size] +
                projected + ranges + b"".join(struct.pack("<5I", *x) for x in arrays))

    def test_count_emit_replay_is_verified_and_not_historical_capacity(self):
        raw = Path(self.temp.name) / "input.bin"
        raw.write_bytes(self.sort_input_bytes())
        result = sort_audit.audit(self.path, sort_input=raw)
        self.assertTrue(result["renderer_replay_inputs"]["production_count_emit_matches"])
        self.assertEqual(result["renderer_replay_inputs"]["replay_capacity"], 8)
        self.assertIsNone(result["renderer_replay_inputs"]["historical_live_capacity"])

    def test_count_emit_replay_rejects_corruption_and_mismatched_state(self):
        raw = Path(self.temp.name) / "input.bin"
        original = self.sort_input_bytes()
        for offset in (8, 12, 16 + 8 + 4 * 3, 104, 104 + 48 * 4,
                       104 + 56 * 4, 104 + 56 * 4 + 4 * 5,
                       104 + 56 * 4 + 8 * 5, 104 + 56 * 4 + 12 * 5):
            payload = bytearray(original)
            struct.pack_into("<I", payload, offset, 0 if offset == 12 else 0xFFFFFFFF)
            raw.write_bytes(payload)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                sort_audit.audit(self.path, sort_input=raw)
        for payload in (original[:40], original[:-1], original + b"x"):
            raw.write_bytes(payload)
            with self.assertRaises(ValueError):
                sort_audit.audit(self.path, sort_input=raw)


class SortReuseBudgetTests(unittest.TestCase):
    def test_existing_sorter_formula_and_prefix_hierarchy(self):
        self.assertEqual(sort_audit.sorter_storage(4096, 256), 38400)
        self.assertEqual(sort_audit.prefix_model(256), dict(dispatches=3, scratch_uints=1))
        self.assertEqual(sort_audit.prefix_model(65537), dict(dispatches=5, scratch_uints=259))

    def test_existing_workload_has_conditional_not_measured_savings(self):
        result = sort_audit.budget(1248125, 442675, 1611449, 1611449)
        self.assertEqual(result["depth_partition_workgroups_per_pass"],
                         dict(baseline=394, candidate=109))
        self.assertEqual(result["candidate_depth_sort_dispatches"], 12)
        self.assertEqual(result["extra_prepare_and_sorted_prefix_dispatches"], 12)
        self.assertEqual(result["memory_model"]["candidate_extra_arrays_bytes"], 24962500)
        self.assertIsNone(result["logical_payload_model"]["native_dram_bytes"])
        self.assertTrue(all(not s["measured"] for s in result["sensitivity_only"]))

    def test_no_duplication_is_not_automatically_worthwhile(self):
        result = sort_audit.budget(1000, 1000, 1000, 2048)
        self.assertLess(result["logical_payload_model"]["modeled_core_net_bytes_saved"], 0)
        self.assertEqual(result["depth_input_reduction_fraction"], 0)
        self.assertEqual(result["memory_model"]["item_capacity"], 2048)

    def test_budget_rejects_invalid_counts_alignment_and_capacity(self):
        for args in ((1, 2, 2, 2), (2, 2, 1, 1), (2, 1, 2, 1),
                     (2, 0, 2, 2), (-1, 0, 0, 0), (2, 1, 2, 2, 3)):
            with self.assertRaises(ValueError):
                sort_audit.budget(*args)


if __name__ == "__main__":
    unittest.main()
