"""Format and comparison checks for benchmark gradient exports."""

from pathlib import Path
import struct
import sys
import subprocess
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_projected_gradients as gradients  # noqa: E402


def write_capture(path: Path, vectors: list[tuple[float, ...]], *, tile_mode: int,
                  touched: bytes, frame: int = 3, pixel_mode: int = 6,
                  version: int = 1, child_batch: int = 8, tensor_reduction: int = 0,
                  tensor_global_pixel_grad: int = 0, tensor_rgb_latency_hiding: int = 0,
                  ssim_mode: int = 0, pixel_gradient: tuple[float, ...] = (1e-7, 2e-7, 3e-7, 0.0),
                  checkpoint_bucket: int = 0, checkpoint_layout: int = 0,
                  checkpoint_tasks: int = 0, checkpoint_state_built: int = 0,
                  checkpoint_fallbacks: int = 0, pixel_cull_method: int = 0) -> None:
    count = len(vectors)
    fields = (gradients.MAGIC, version, count, 8000, frame, tile_mode, pixel_mode,
              1, 16, 16, 4, sum(touched), int(any(touched)))
    if version >= 9:
        header = gradients.HEADER_V9.pack(*fields, child_batch, tensor_reduction,
            tensor_global_pixel_grad, tensor_rgb_latency_hiding, ssim_mode,
            checkpoint_bucket, checkpoint_layout, checkpoint_tasks,
            checkpoint_state_built, checkpoint_fallbacks, pixel_cull_method, 0.25, 0.75)
    elif version >= 8:
        header = gradients.HEADER_V8.pack(*fields, child_batch, tensor_reduction,
            tensor_global_pixel_grad, tensor_rgb_latency_hiding, ssim_mode,
            checkpoint_bucket, checkpoint_layout, checkpoint_tasks,
            checkpoint_state_built, checkpoint_fallbacks, 0.25, 0.75)
    elif version >= 6:
        layout = gradients.HEADER_V7 if version == 7 else gradients.HEADER_V6
        header = layout.pack(*fields, child_batch, tensor_reduction,
                                         tensor_global_pixel_grad, tensor_rgb_latency_hiding,
                                         ssim_mode, 0.25, 0.75)
    elif version == 5:
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
    pixels = gradients.PIXEL_GRADIENT.pack(*pixel_gradient) * (16 * 16) if version >= 6 else b""
    path.write_bytes(header + payload + touched + pixels)


class CompareProjectedGradientsTests(unittest.TestCase):
    def test_v9_pixel_cull_roundtrip_and_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old, new = (Path(directory) / name for name in ("v8.bin", "v9.bin"))
            write_capture(old, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=8)
            write_capture(new, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=9,
                          tensor_reduction=1, tensor_global_pixel_grad=1, pixel_cull_method=1)
            a, b = gradients.read_capture(old), gradients.read_capture(new)
            self.assertEqual(a.pixel_cull_method, 0)
            self.assertEqual(b.pixel_cull_method, 1)
            self.assertEqual(b.header_size, 108)
            self.assertEqual(a.gradient(0), b.gradient(0))
            self.assertEqual(a.pixel_gradient(0), b.pixel_gradient(0))
            gradients.check_pair(a, b, 1e-4)
            data = new.read_bytes()
            for broken in (data[:107], data[:-1], data + b"\x00"):
                new.write_bytes(broken)
                with self.assertRaises(ValueError): gradients.read_capture(new)

    def test_v9_pixel_cull_rejects_incompatible_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            base = dict(version=9, pixel_cull_method=1, tensor_reduction=1,
                        tensor_global_pixel_grad=1)
            for change in (dict(pixel_cull_method=2), dict(tensor_global_pixel_grad=0),
                           dict(tensor_rgb_latency_hiding=1), dict(pixel_mode=1),
                           dict(checkpoint_state_built=1, checkpoint_bucket=32,
                                checkpoint_layout=2, checkpoint_tasks=1)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00", **(base | change))
                with self.assertRaises(ValueError): gradients.read_capture(path)

    def test_cli_v9_pixel_cull_repeat_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            a, b = (Path(directory) / name for name in ("a.bin", "b.bin"))
            for path, method in ((a, 0), (b, 1)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=9,
                              tensor_reduction=1, tensor_global_pixel_grad=1, pixel_cull_method=method)
            command = [sys.executable, "-B", str(Path(gradients.__file__)), str(a), str(b)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Off -> EarlyCull FP32 guarded v1", result.stdout)
            result = subprocess.run(command + ["--repeat-fixed", str(b)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same pixel-culling method", result.stderr)

    def test_v8_checkpoint_roundtrip_and_legacy_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline, warp = Path(directory) / "v7.bin", Path(directory) / "warp.bin"
            write_capture(baseline, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=7)
            write_capture(warp, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=8,
                          pixel_mode=7, checkpoint_bucket=32, checkpoint_layout=2,
                          checkpoint_tasks=1, checkpoint_state_built=1)
            old, new = gradients.read_capture(baseline), gradients.read_capture(warp)
            self.assertEqual(new.header_size, 104)
            self.assertEqual(new.checkpoint_bucket, 32)
            self.assertEqual(new.checkpoint_layout, 2)
            self.assertEqual(new.gradient(0), old.gradient(0))
            self.assertEqual(new.pixel_gradient(0), old.pixel_gradient(0))
            with self.assertRaisesRegex(ValueError, "pixel_mode"):
                gradients.check_pair(old, new, 1e-4)
            gradients.check_pair(old, new, 1e-4, True)
            self.assertEqual(gradients.compare(old, new, 1e-5, 1e-3, 1e-8)[0].relative_l2, 0)

    def test_v8_rejects_false_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.bin"
            base = dict(version=8, pixel_mode=7, checkpoint_bucket=32,
                        checkpoint_layout=2, checkpoint_state_built=1, checkpoint_tasks=1)
            for change in (dict(checkpoint_bucket=16), dict(checkpoint_layout=1),
                           dict(checkpoint_state_built=2), dict(checkpoint_tasks=100),
                           dict(checkpoint_fallbacks=1), dict(checkpoint_state_built=0),
                           dict(tensor_reduction=1)):
                write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", **(base | change))
                with self.assertRaises(ValueError):
                    gradients.read_capture(path)
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=7, pixel_mode=7)
            with self.assertRaisesRegex(ValueError, "mode"):
                gradients.read_capture(path)

    def test_v8_truncation_and_p1_state_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "p1.bin"
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=8,
                          checkpoint_bucket=32, checkpoint_layout=2, checkpoint_state_built=1,
                          checkpoint_tasks=1, tensor_reduction=1, tensor_global_pixel_grad=1)
            self.assertEqual(gradients.read_capture(path).pixel_mode, 6)
            data = path.read_bytes()
            for broken in (data[:103], data[:-1], data + b"\x00"):
                path.write_bytes(broken)
                with self.assertRaises(ValueError):
                    gradients.read_capture(path)

    def test_cli_v8_labels_and_repeat_checkpoint_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain, p1, warp = (Path(directory) / name for name in ("plain.bin", "p1.bin", "warp.bin"))
            write_capture(plain, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=8)
            kwargs = dict(checkpoint_bucket=32, checkpoint_layout=2,
                          checkpoint_state_built=1, checkpoint_tasks=1)
            write_capture(p1, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=8, **kwargs)
            write_capture(warp, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=8, pixel_mode=7, **kwargs)
            command = [sys.executable, str(Path(gradients.__file__)), str(plain), str(warp),
                       "--allow-pixel-mode-change"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Checkpointed Warp", result.stdout)
            self.assertIn("Checkpoint bucket/layout", result.stdout)
            result = subprocess.run(command + ["--repeat-fixed", str(p1)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same checkpoint path", result.stderr)

    def test_v7_modes_keep_v6_and_legacy_meaning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captures = []
            for version, mode in ((1, 0), (6, 0), (6, 1), (7, 0), (7, 1), (7, 2)):
                path = Path(directory) / f"v{version}_{mode}.bin"
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00",
                              version=version, ssim_mode=mode)
                capture = gradients.read_capture(path)
                self.assertEqual(capture.ssim_mode, mode)
                self.assertEqual(capture.gradient(0), (1.0,) * 12)
                captures.append(capture)
            self.assertEqual(gradients.SSIM_NAMES[captures[2].ssim_mode], "Separable")
            self.assertEqual(gradients.SSIM_NAMES[captures[-1].ssim_mode], "2D Loss + Separable Backward")
            for candidate in captures[1:]:
                gradients.check_pair(captures[0], candidate, 1e-4)
                self.assertEqual(gradients.compare(captures[0], candidate, 1e-5, 1e-3, 1e-8)[0].relative_l2, 0)
            for candidate in captures[2:]:
                self.assertEqual(gradients.compare_pixels(captures[1], candidate, 1e-9, 1e-3).relative_l2, 0)

    def test_v7_invalid_modes_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            for mode in (3, 0xffffffff):
                write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=7, ssim_mode=mode)
                with self.assertRaisesRegex(ValueError, "SSIM mode"):
                    gradients.read_capture(path)
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=7, ssim_mode=2)
            data = path.read_bytes()
            for broken in (data[:gradients.HEADER_V7.size - 1], data[:-1], data + b"\x00"):
                path.write_bytes(broken)
                with self.assertRaises(ValueError):
                    gradients.read_capture(path)

    def test_cli_v7_hybrid_label_and_repeat_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline, hybrid = Path(directory) / "2d.bin", Path(directory) / "hybrid.bin"
            for path, mode in ((baseline, 0), (hybrid, 2)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=7, ssim_mode=mode)
            command = [sys.executable, str(Path(gradients.__file__)), str(baseline), str(hybrid)]
            result = subprocess.run(command + ["--repeat-fixed", str(baseline)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SSIM: 2D -> 2D Loss + Separable Backward", result.stdout)
            self.assertIn("Pixel gradients: relative L2 0", result.stdout)
            result = subprocess.run(command + ["--repeat-fixed", str(hybrid)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same SSIM mode", result.stderr)

    def test_repeat_v6_full_separable_matches_v7_full_not_hybrid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            old, full, hybrid = (Path(directory) / name for name in ("v6.bin", "full.bin", "hybrid.bin"))
            for path, version, mode in ((old, 6, 1), (full, 7, 1), (hybrid, 7, 2)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=version, ssim_mode=mode)
            command = [sys.executable, str(Path(gradients.__file__)), str(old), str(hybrid)]
            result = subprocess.run(command + ["--repeat-fixed", str(full)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(command + ["--repeat-fixed", str(hybrid)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same SSIM mode", result.stderr)

    def test_v6_ssim_pixels_and_legacy_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            captures = []
            for version, mode in ((1, 0), (5, 0), (6, 0), (6, 1)):
                path = Path(directory) / f"v{version}_{mode}.bin"
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00",
                              version=version, ssim_mode=mode)
                capture = gradients.read_capture(path)
                self.assertEqual(capture.ssim_mode, mode)
                self.assertEqual(capture.has_pixel_gradients, version >= 6)
                self.assertEqual(capture.gradient(0), (1.0,) * 12)
                self.assertFalse(capture.child_touched(0))
                captures.append(capture)
            for candidate in captures[1:]:
                gradients.check_pair(captures[0], candidate, 1e-4)
                self.assertEqual(gradients.compare(captures[0], candidate, 1e-5, 1e-3, 1e-8)[0].relative_l2, 0)
            self.assertAlmostEqual(captures[-1].pixel_gradient(255)[2], 3e-7)
            pixels = gradients.compare_pixels(captures[-2], captures[-1], 1e-9, 1e-3)
            self.assertEqual(pixels.components, 256 * 3)
            self.assertEqual(pixels.relative_l2, 0)
            with self.assertRaisesRegex(ValueError, "version-6"):
                gradients.compare_pixels(captures[0], captures[-1], 1e-9, 1e-3)

    def test_v6_invalid_mode_truncation_and_trailing_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=6, ssim_mode=2)
            with self.assertRaisesRegex(ValueError, "SSIM mode"):
                gradients.read_capture(path)
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=6)
            data = path.read_bytes()
            for broken in (data[:gradients.HEADER_V6.size - 1], data[:-1], data + b"\x00"):
                path.write_bytes(broken)
                with self.assertRaises(ValueError):
                    gradients.read_capture(path)

    def test_v6_pixels_detect_small_errors_nonfinite_and_ignore_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory) / "2d.bin", Path(directory) / "separable.bin"
            write_capture(baseline, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=6)
            for pixel, mismatched, nonfinite in (
                    ((1e-7, 2e-7, 3e-7, 100.0), 0, 0),
                    ((1.1e-7, 2e-7, 3e-7, 0.0), 256, 0),
                    ((float("nan"), 2e-7, 3e-7, 0.0), 0, 256)):
                write_capture(candidate, [(1.0,) * 12], tile_mode=1, touched=b"\x00",
                              version=6, ssim_mode=1, pixel_gradient=pixel)
                result = gradients.compare_pixels(gradients.read_capture(baseline),
                                                  gradients.read_capture(candidate), 1e-9, 1e-3)
                self.assertEqual(result.mismatched_components, mismatched)
                self.assertEqual(result.nonfinite_components, nonfinite)

    def test_nonfinite_forward_and_zero_v6_extent_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.bin"
            write_capture(path, [(0.0,) * 12], tile_mode=1, touched=b"\x00", version=6)
            original = path.read_bytes()
            for value in (float("nan"), float("inf")):
                for offset in (gradients.HEADER_V6.size - 8, gradients.HEADER_V6.size - 4):
                    data = bytearray(original)
                    struct.pack_into("<f", data, offset, value)
                    path.write_bytes(data)
                    with self.assertRaisesRegex(ValueError, "non-finite forward"):
                        gradients.read_capture(path)
            data = bytearray(original)
            struct.pack_into("<I", data, 8 + 7 * 4, 0)  # width
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "extent"):
                gradients.read_capture(path)

    def test_cli_v6_ssim_and_repeat_mode_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = Path(directory) / "2d.bin", Path(directory) / "separable.bin"
            for path, mode in ((baseline, 0), (candidate, 1)):
                write_capture(path, [(1.0,) * 12], tile_mode=1, touched=b"\x00", version=6, ssim_mode=mode)
            command = [sys.executable, str(Path(gradients.__file__)), str(baseline), str(candidate)]
            result = subprocess.run(command + ["--repeat-fixed", str(baseline)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("SSIM: 2D -> Separable", result.stdout)
            self.assertIn("Pixel gradients: relative L2 0", result.stdout)
            result = subprocess.run(command + ["--repeat-fixed", str(candidate)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same SSIM mode", result.stderr)

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
