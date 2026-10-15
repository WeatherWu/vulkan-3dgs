from pathlib import Path
import csv
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_adaptive_tiles as analysis


class AdaptiveTileAnalysisTests(unittest.TestCase):
    def test_ranking_replaces_low_profit_early_tile(self) -> None:
        early = analysis.Tile(0, 0, 0, 40, (3, 2, 0, 0), True, (3, 2, 0, 0))
        later = analysis.Tile(1, 1, 0, 100, (3, 2, 0, 0), False, (20, 10, 0, 0))
        ranked = analysis.rank_selection([early, later], 5)
        self.assertEqual([tile.index for tile in ranked], [1])
        self.assertGreater(analysis.selection_stats(ranked)["proxy_profit"], early.profit)

    def test_budget_and_broad_support_are_respected(self) -> None:
        broad = analysis.Tile(0, 0, 0, 100, (100,) * 4, False, (0,) * 4)
        narrow = analysis.Tile(1, 1, 0, 100, (3, 3, 0, 0), False, (0,) * 4)
        self.assertFalse(broad.profitable)
        self.assertEqual(analysis.rank_selection([broad, narrow], 5), [])

    def test_tensor_batches_use_leaf_prefixes_only_when_selected(self) -> None:
        coarse = analysis.Tile(0, 0, 0, 100, (0,) * 4, False, (17, 5, 0, 0))
        refined = analysis.Tile(1, 1, 0, 100, (8,) * 4, True, (8, 8, 8, 8))
        report = analysis.analyze({"pixel_mode": 6}, [coarse, refined])
        self.assertEqual(report["current_tensor_batches"], 2 + 4)
        self.assertTrue(report["scan_selection_matches"])

    def test_incomplete_csv_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.csv"
            path.write_text("parent_index,parent_items\n0,100\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                analysis.read_tiles(path)

    def test_observed_cost_separates_pixel_work_from_repeated_reductions(self) -> None:
        metadata = dict(frame=0, iteration=8000, gaussians=1000, width=16, height=16, pixel_mode=6)
        coarse = analysis.Tile(0, 0, 0, 300, (0,) * 4, False, (100,) * 4)
        child = analysis.Tile(0, 0, 0, 300, (50,) * 4, True, (40,) * 4)
        report = analysis.compare_coarse_reference(metadata, [child], metadata, [coarse])
        self.assertEqual(report["coarse_prefix_entries"], 100)
        self.assertEqual(report["current_prefix_entries"], 160)
        self.assertEqual(report["coarse_prefix_pixel_slots"], 25600)
        self.assertEqual(report["current_prefix_pixel_slots"], 10240)
        self.assertGreater(report["current_tensor_batches"], report["coarse_tensor_batches"])

    def test_valid_csv_preserves_selection_and_prefix_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiles.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=analysis.META_FIELDS + analysis.ROW_FIELDS)
                writer.writeheader()
                for index in range(2):
                    row = dict(frame=0, iteration=8000, gaussians=1000, width=32,
                               height=16, threshold=256, pixel_mode=6,
                               parent_index=index, parent_x=index, parent_y=0,
                               parent_items=300, selected=int(index == 0))
                    row.update({f"child{i}": 25 for i in range(4)})
                    row.update({f"max{i}": 25 if index == 0 else 80 for i in range(4)})
                    writer.writerow(row)
            metadata, tiles = analysis.read_tiles(path)
            report = analysis.analyze(metadata, tiles)
            self.assertTrue(report["scan_selection_matches"])
            self.assertEqual(report["current"]["child_items"], 100)
            self.assertEqual(report["child_budget"], 150)

    def test_coarse_reference_requires_same_checkpoint(self) -> None:
        metadata = dict(frame=0, iteration=8000, gaussians=1000, width=16, height=16, pixel_mode=6)
        other = dict(metadata, iteration=8001)
        tile = analysis.Tile(0, 0, 0, 300, (0,) * 4, False, (100,) * 4)
        with self.assertRaisesRegex(ValueError, "metadata"):
            analysis.compare_coarse_reference(metadata, [tile], other, [tile])

    def test_accounting_uses_recorded_child_batch(self) -> None:
        tile = analysis.Tile(0, 0, 0, 1000, (50,) * 4, True, (40,) * 4)
        for batch, expected in ((8, 20), (16, 12), (32, 8)):
            report = analysis.analyze({"pixel_mode": 6, "child_batch": batch}, [tile])
            self.assertEqual(report["current_tensor_batches"], expected)

    def test_mask_filter_recovers_selected_and_rejected_prefixes(self) -> None:
        # Child lists [0, 2, 4] and [1, 3] preserve the parent's sorted order.
        masks = bytes([1, 2, 1, 2, 1, 3, 1, 2, 0, 3])
        selected = analysis.Tile(0, 0, 0, 5, (3, 2, 0, 0), True, (3, 2, 0, 0))
        rejected = analysis.Tile(1, 1, 0, 5, (3, 3, 0, 0), False, (2, 3, 0, 0))
        reference = [analysis.Tile(0, 0, 0, 5, (0,) * 4, False, (5, 4, 0, 0)),
                     analysis.Tile(1, 1, 0, 5, (0,) * 4, False, (2, 3, 0, 0))]
        predicted = analysis.predict_child_prefixes({"threshold": 1}, [selected, rejected], reference, masks)
        self.assertEqual(predicted[0], (3, 2, 0, 0))
        self.assertEqual(predicted[1], (2, 2, 0, 0))

    def test_mask_prediction_requires_exact_selected_match(self) -> None:
        tile = analysis.Tile(0, 0, 0, 3, (3, 0, 0, 0), True, (1, 0, 0, 0))
        reference = [analysis.Tile(0, 0, 0, 3, (0,) * 4, False, (2, 0, 0, 0))]
        with self.assertRaisesRegex(ValueError, "differ from measured"):
            analysis.predict_child_prefixes({"threshold": 1}, [tile], reference, bytes([1, 1, 1]))

    def test_mask_counts_and_unclassified_regions_are_checked(self) -> None:
        tile = analysis.Tile(0, 0, 0, 3, (2, 0, 0, 0), False, (2, 0, 0, 0))
        with self.assertRaisesRegex(ValueError, "full child counts"):
            analysis.predict_child_prefixes({"threshold": 1}, [tile], [tile], bytes([1, 1, 1]))
        with self.assertRaisesRegex(ValueError, "unclassified"):
            analysis.predict_child_prefixes({"threshold": 4}, [tile], [tile], bytes([1, 0, 0]))

    def test_mask_header_matches_csv_checkpoint(self) -> None:
        metadata = dict(frame=0, iteration=8000, gaussians=1000, width=16,
                        height=16, threshold=1, pixel_mode=6)
        tile = analysis.Tile(0, 0, 0, 3, (3, 0, 0, 0), False, (2, 0, 0, 0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tile.masks.bin"
            header = analysis.MASK_HEADER.pack(b"V3DMSK01", 1, 1000, 8000, 0, 16, 16, 1, 6, 3, 1)
            path.write_bytes(header + bytes([1, 1, 1]))
            self.assertEqual(analysis.read_masks(path, metadata, [tile]), bytes([1, 1, 1]))
            with self.assertRaisesRegex(ValueError, "metadata"):
                analysis.read_masks(path, dict(metadata, iteration=8001), [tile])

    def test_conservative_policy_respects_budget_and_work_counts(self) -> None:
        tiles = [analysis.Tile(i, i, 0, 40, (5, 0, 0, 0), False, (3, 0, 0, 0)) for i in range(2)]
        predictions = {0: (3, 0, 0, 0), 1: (3, 0, 0, 0)}
        report = analysis.analyze_prefix_choices({"threshold": 1}, tiles, tiles, predictions)
        result, coarse = report["conservative"], report["coarse"]
        self.assertEqual(result["parents"], 2)
        self.assertLessEqual(result["child_items"], sum(tile.parent_items for tile in tiles) // 4)
        for name in ("prefix_entries", "prefix_pixel_slots", "tensor_batches"):
            self.assertLessEqual(result[name], coarse[name])

    def test_cli_automatically_reads_mask_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adaptive_path, coarse_path = Path(directory) / "adaptive.csv", Path(directory) / "coarse.csv"
            report_path = Path(directory) / "analysis.json"
            for path, selected, threshold in ((adaptive_path, 1, 1), (coarse_path, 0, 1000000)):
                row = dict(frame=0, iteration=8000, gaussians=1000, width=16, height=16,
                           threshold=threshold, pixel_mode=6, parent_index=0, parent_x=0,
                           parent_y=0, parent_items=100, selected=selected)
                row.update({f"child{i}": 20 if selected and i == 0 else 0 for i in range(4)})
                row.update({f"max{i}": 20 if i == 0 else 0 for i in range(4)})
                with path.open("w", newline="", encoding="utf-8") as output:
                    writer = csv.DictWriter(output, fieldnames=analysis.META_FIELDS + analysis.ROW_FIELDS)
                    writer.writeheader()
                    writer.writerow(row)
            header = analysis.MASK_HEADER.pack(b"V3DMSK01", 1, 1000, 8000, 0, 16, 16, 1, 6, 100, 1)
            adaptive_path.with_suffix(".masks.bin").write_bytes(header + bytes([1] * 20 + [0] * 80))
            command = ["analyze_adaptive_tiles.py", str(adaptive_path), "--coarse-reference",
                       str(coarse_path), "--json", str(report_path)]
            with mock.patch.object(sys, "argv", command), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(analysis.main(), 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["mask_prefix_analysis"]["validated_selected_parents"], 1)


if __name__ == "__main__":
    unittest.main()
