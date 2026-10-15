"""Format and comparison checks for benchmark gradient exports."""

from pathlib import Path
import sys
import subprocess
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_projected_gradients as gradients  # noqa: E402


def write_capture(path: Path, vectors: list[tuple[float, ...]], *, tile_mode: int,
                  touched: bytes, frame: int = 3, pixel_mode: int = 6,
                  version: int = 1, child_batch: int = 8, tensor_reduction: int = 0,
                  tensor_global_pixel_grad: int = 0, tensor_rgb_latency_hiding: int = 0) -> None:
    count = len(vectors)
    fields = (gradients.MAGIC, version, count, 8000, frame, tile_mode, pixel_mode,
              1, 16, 16, 4, sum(touched), int(any(touched)))
    if version == 5:
        header = gradients.HEADER_V5.pack(*fields, child_batch, tensor_reduction,
                                         tensor_global_pixel_grad, tensor_rgb_latency_hiding, 0.25, 0.75)
    elif version == 4:
        header = gradients.HEADER_V4.pack(*fields, child_batch, tensor_reduction,
                                         tensor_global_pixel_grad, 0.25, 0.75)
    elif version == 3:
        header = gradients.HEADER_V3.pack(*fields, child_batch, tensor_reduction, 0.25, 0.75)
    elif version == 2:
        header = gradients.HEADER_V2.pack(*fields, child_batch, 0.25, 0.75)
    else:
        header = gradients.HEADER.pack(*fields, 0.25, 0.75)
    payload = b"".join(gradients.GRADIENT.pack(*vector) for vector in vectors)
    path.write_bytes(header + payload + touched)


class CompareProjectedGradientsTests(unittest.TestCase):
    def test_v5_rgb_schedule_and_legacy_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captures = []
            vector = tuple(float(i) for i in range(12))
            settings = ((1, 0, 0, 0), (2, 0, 0, 0), (3, 1, 0, 0), (4, 1, 1, 0),
                        (5, 0, 0, 0), (5, 1, 0, 0), (5, 1, 1, 0), (5, 1, 1, 1))
            for version, reduction, global_grad, latency in settings:
                path = Path(directory) / f"v{version}_{reduction}_{global_grad}_{latency}.bin"
                write_capture(path, [vector], tile_mode=1, touched=b"\x00", version=version,
                              tensor_reduction=reduction, tensor_global_pixel_grad=global_grad,
                              tensor_rgb_latency_hiding=latency)
                capture = gradients.read_capture(path)
                self.assertEqual(capture.tensor_rgb_latency_hiding, bool(latency))
                self.assertEqual(capture.tensor_global_pixel_grad, bool(global_grad))
                self.assertEqual(capture.tensor_reduction, bool(reduction))
                self.assertEqual(capture.gradient(0), vector)
                self.assertFalse(capture.child_touched(0))
                captures.append(capture)
            for candidate in captures[1:]:
                gradients.check_pair(captures[0], candidate, 1e-4)
                self.assertEqual(gradients.compare(captures[0], candidate, 1e-5, 1e-3, 1e-8)[0].relative_l2, 0)

    def test_invalid_v5_rgb_schedule_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            for tile, pixel, reduction, source, latency in (
                    (1, 6, 1, 1, 2), (1, 6, 1, 0, 1), (1, 6, 0, 0, 1),
                    (1, 6, 0, 1, 1), (0, 6, 1, 1, 1), (2, 6, 1, 1, 1), (1, 1, 1, 1, 1)):
                with self.subTest(tile=tile, pixel=pixel, reduction=reduction, source=source, latency=latency):
                    write_capture(path, [(0.0,) * 12], tile_mode=tile, pixel_mode=pixel,
                                  touched=b"\x00", version=5, tensor_reduction=reduction,
                                  tensor_global_pixel_grad=source, tensor_rgb_latency_hiding=latency)
                    with self.assertRaisesRegex(ValueError, "schedule|requires|reduction"):
                        gradients.read_capture(path)

    def test_truncated_v5_header_and_payload_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.bin"
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=5)
            data = path.read_bytes()
            for length, message in ((gradients.HEADER_V5.size - 1, "truncated version-5 header"),
                                    (len(data) - 1, "expected")):
                path.write_bytes(data[:length])
                with self.assertRaisesRegex(ValueError, message):
                    gradients.read_capture(path)

    def test_cli_reports_rgb_schedule_and_checks_noise_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline, latency = Path(directory) / "baseline.bin", Path(directory) / "latency.bin"
            for path, schedule in ((baseline, 0), (latency, 1)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=5,
                              tensor_reduction=1, tensor_global_pixel_grad=1,
                              tensor_rgb_latency_hiding=schedule)
            command = [sys.executable, str(Path(gradients.__file__)), str(baseline), str(latency)]
            result = subprocess.run(command + ["--repeat-fixed", str(baseline)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Tensor RGB schedule: Baseline -> Latency Hiding", result.stdout)
            result = subprocess.run(command + ["--repeat-fixed", str(latency)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same Tensor RGB schedule", result.stderr)

    def test_v4_pixel_grad_source_and_legacy_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captures = []
            vector = tuple(float(i) for i in range(12))
            for version, reduction, global_grad in ((1, 0, 0), (2, 0, 0), (3, 1, 0),
                                                     (4, 0, 0), (4, 1, 0), (4, 1, 1)):
                path = Path(directory) / f"v{version}_{reduction}_{global_grad}.bin"
                write_capture(path, [vector], tile_mode=1, touched=b"\x00", version=version,
                              tensor_reduction=reduction, tensor_global_pixel_grad=global_grad)
                capture = gradients.read_capture(path)
                self.assertEqual(capture.tensor_reduction, bool(reduction))
                self.assertEqual(capture.tensor_global_pixel_grad, bool(global_grad))
                self.assertEqual(capture.gradient(0), vector)
                self.assertFalse(capture.child_touched(0))
                captures.append(capture)
            for candidate in captures[1:]:
                gradients.check_pair(captures[0], candidate, 1e-4)
                self.assertEqual(gradients.compare(captures[0], candidate, 1e-5, 1e-3, 1e-8)[0].relative_l2, 0)

    def test_invalid_v4_pixel_grad_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            for tile, pixel, reduction, source in ((1, 6, 1, 2), (1, 6, 0, 1),
                                                    (0, 6, 1, 1), (2, 6, 1, 1), (1, 1, 1, 1)):
                with self.subTest(tile=tile, pixel=pixel, reduction=reduction, source=source):
                    write_capture(path, [(0.0,) * 12], tile_mode=tile, pixel_mode=pixel,
                                  touched=b"\x00", version=4, tensor_reduction=reduction,
                                  tensor_global_pixel_grad=source)
                    with self.assertRaisesRegex(ValueError, "source|reduction"):
                        gradients.read_capture(path)

    def test_truncated_v4_header_and_payload_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.bin"
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=4)
            data = path.read_bytes()
            for length, message in ((gradients.HEADER_V4.size - 1, "truncated version-4 header"),
                                    (len(data) - 1, "expected")):
                path.write_bytes(data[:length])
                with self.assertRaisesRegex(ValueError, message):
                    gradients.read_capture(path)

    def test_cli_reports_pixel_grad_source_and_checks_noise_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shared, global_grad = Path(directory) / "shared.bin", Path(directory) / "global.bin"
            for path, source in ((shared, 0), (global_grad, 1)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=4,
                              tensor_reduction=1, tensor_global_pixel_grad=source)
            command = [sys.executable, str(Path(gradients.__file__)), str(shared), str(global_grad)]
            result = subprocess.run(command + ["--repeat-fixed", str(shared)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("pixel-gradient source: Shared -> Global", result.stdout)
            result = subprocess.run(command + ["--repeat-fixed", str(global_grad)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same Tensor pixel-gradient source", result.stderr)

    def test_cli_reports_actual_reduction_and_rejects_mixed_noise_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            shared, subgroup = Path(directory) / "shared.bin", Path(directory) / "subgroup.bin"
            for path, reduction in ((shared, 0), (subgroup, 1)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00",
                              version=3, tensor_reduction=reduction)
            command = [sys.executable, str(Path(gradients.__file__)), str(shared), str(subgroup)]
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Fixed16 Tensor reduction: Shared -> Subgroup", result.stdout)
            result = subprocess.run(command + ["--repeat-fixed", str(subgroup)],
                                    capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same Tensor reduction", result.stderr)

    def test_truncated_v3_header_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.bin"
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=3)
            path.write_bytes(path.read_bytes()[:gradients.HEADER_V3.size - 1])
            with self.assertRaisesRegex(ValueError, "truncated version-3 header"):
                gradients.read_capture(path)

    def test_v3_tensor_reduction_and_legacy_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captures = []
            vector = tuple(float(i) for i in range(12))
            for version, reduction in ((1, 0), (2, 0), (3, 0), (3, 1)):
                path = Path(directory) / f"v{version}_{reduction}.bin"
                write_capture(path, [vector], tile_mode=1, touched=b"\x00",
                              version=version, tensor_reduction=reduction)
                capture = gradients.read_capture(path)
                self.assertEqual(capture.tensor_reduction, bool(reduction))
                self.assertEqual(capture.gradient(0), vector)
                self.assertFalse(capture.child_touched(0))
                captures.append(capture)
            for candidate in captures[1:]:
                gradients.check_pair(captures[0], candidate, 1e-4)
                self.assertEqual(gradients.compare(captures[0], candidate, 1e-5, 1e-3, 1e-8)[0].relative_l2, 0)

    def test_invalid_v3_reduction_metadata_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            for tile, pixel, reduction in ((1, 6, 2), (0, 6, 1), (2, 6, 1), (1, 1, 1)):
                with self.subTest(tile=tile, pixel=pixel, reduction=reduction):
                    write_capture(path, [(0.0,) * 12], tile_mode=tile, pixel_mode=pixel,
                                  touched=b"\x00", version=3, tensor_reduction=reduction)
                    with self.assertRaisesRegex(ValueError, "reduction"):
                        gradients.read_capture(path)

    def test_missing_child_gradient_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixed_path = Path(directory) / "fixed.bin"
            adaptive_path = Path(directory) / "adaptive.bin"
            present = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            empty = (0.0,) * 12
            write_capture(fixed_path, [present, present], tile_mode=1,
                          touched=b"\x00\x00")
            write_capture(adaptive_path, [empty, present], tile_mode=2,
                          touched=b"\x01\x00")
            fixed = gradients.read_capture(fixed_path)
            adaptive = gradients.read_capture(adaptive_path)
            gradients.check_pair(fixed, adaptive, 1e-4)
            overall, child = gradients.compare(fixed, adaptive, 1e-5, 1e-3, 1e-8)
            self.assertEqual(overall.missing_gaussians, 1)
            self.assertEqual(child.gaussians, 1)
            self.assertEqual(child.missing_gaussians, 1)
            self.assertAlmostEqual(overall.relative_l2, 2 ** -0.5)

    def test_different_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.bin"
            second_path = Path(directory) / "second.bin"
            empty = (0.0,) * 12
            write_capture(first_path, [empty], tile_mode=1, touched=b"\x00")
            write_capture(second_path, [empty], tile_mode=1, touched=b"\x00", frame=4)
            with self.assertRaisesRegex(ValueError, "frame"):
                gradients.check_pair(gradients.read_capture(first_path),
                                     gradients.read_capture(second_path), 1e-4)

    def test_truncated_capture_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.bin"
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00")
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "expected"):
                gradients.read_capture(path)

    def test_kernel_change_requires_explicit_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            direct_path = Path(directory) / "direct.bin"
            tensor_path = Path(directory) / "tensor.bin"
            empty = (0.0,) * 12
            write_capture(direct_path, [empty], tile_mode=1, touched=b"\x00", pixel_mode=1)
            write_capture(tensor_path, [empty], tile_mode=1, touched=b"\x00")
            direct, tensor = gradients.read_capture(direct_path), gradients.read_capture(tensor_path)
            with self.assertRaisesRegex(ValueError, "pixel_mode"):
                gradients.check_pair(direct, tensor, 1e-4)
            gradients.check_pair(direct, tensor, 1e-4, allow_pixel_mode_change=True)

    def test_v2_batch_metadata_and_v1_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            legacy_path, current_path = Path(directory) / "old.bin", Path(directory) / "new.bin"
            vector = (1.0, 2.0) + (0.0,) * 10
            write_capture(legacy_path, [vector], tile_mode=2, touched=b"\x01")
            write_capture(current_path, [vector], tile_mode=2, touched=b"\x01", version=2, child_batch=32)
            legacy, current = gradients.read_capture(legacy_path), gradients.read_capture(current_path)
            self.assertEqual(legacy.child_batch, 8)
            self.assertEqual(current.child_batch, 32)
            self.assertEqual(current.gradient(0), legacy.gradient(0))
            gradients.check_pair(legacy, current, 1e-4)
            self.assertEqual(gradients.compare(legacy, current, 1e-5, 1e-3, 1e-8)[0].relative_l2, 0)


if __name__ == "__main__":
    unittest.main()
