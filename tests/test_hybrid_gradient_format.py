"""Current hybrid SSIM export format and comparator contracts."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_projected_gradients as gradients


def capture(path: Path, mode: int, version: int = 7) -> None:
    header = gradients.HEADER_V7.pack(
        gradients.MAGIC, version, 1, 8000, 3, 1, 6, 1, 16, 16,
        4, 0, 0, 8, 1, 1, 0, mode, 0.25, 0.75)
    path.write_bytes(header + gradients.GRADIENT.pack(*([0.0] * 12)) + b"\0" +
                     gradients.PIXEL_GRADIENT.pack(0.0, 0.0, 0.0, 0.0) * 256)


class HybridGradientFormatTests(unittest.TestCase):
    def test_current_payload_and_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "2d.bin", Path(directory) / "hybrid.bin"
            capture(first, 0)
            capture(second, 2)
            baseline, hybrid = gradients.read_capture(first), gradients.read_capture(second)
            gradients.check_pair(baseline, hybrid, 1e-4)
            self.assertEqual(hybrid.ssim_mode, 2)
            self.assertEqual(hybrid.header_size, 84)
            self.assertTrue(hybrid.tensor_global_pixel_grad)
            self.assertEqual(gradients.compare_pixels(baseline, hybrid, 1e-9, 1e-3).relative_l2, 0)

    def test_invalid_mode_and_incomplete_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            capture(path, 3)
            with self.assertRaisesRegex(ValueError, "SSIM"):
                gradients.read_capture(path)
            capture(path, 2)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "expected"):
                gradients.read_capture(path)

    def test_input_versions_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.bin"
            capture(path, 2, version=6)
            with self.assertRaisesRegex(ValueError, "SSIM"):
                gradients.read_capture(path)
            capture(path, 2, version=8)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                gradients.read_capture(path)

    def test_repeat_requires_identical_ssim(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory) / "2d.bin", Path(directory) / "hybrid.bin"
            capture(first, 0)
            capture(second, 2)
            result = subprocess.run(
                [sys.executable, "-B", str(Path(gradients.__file__)), str(first), str(second),
                 "--repeat-fixed", str(second)], capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 2)
            self.assertIn("same SSIM mode", result.stderr)


if __name__ == "__main__":
    unittest.main()
