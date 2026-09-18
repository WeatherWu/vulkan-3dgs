#!/usr/bin/env python3
"""Render/evaluate 3DGS models or compare existing image directories.

The ``render`` command loads an Inria-style 3DGS PLY, reads COLMAP cameras
from a Mip-NeRF360 scene, renders the held-out views with gsplat, and reports
mean per-image RGB PSNR using torchmetrics. The default split matches gsplat
and VkSplat: filenames are sorted and every eighth image is held out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


IMAGE_EXTENSIONS = {
    ".bmp", ".exr", ".jpeg", ".jpg", ".png", ".tif", ".tiff"
}


def add_metric_arguments(parser: argparse.ArgumentParser, default_device: str) -> None:
    parser.add_argument(
        "--device",
        default=default_device,
        help="Torch device, for example cpu, cuda, or cuda:1",
    )
    parser.add_argument("--output-json", type=Path, help="Detailed JSON report")
    parser.add_argument("--quiet", action="store_true", help="Only print final PSNR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3DGS PLY and calculate PSNR with torchmetrics."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    render = commands.add_parser(
        "render", help="Render a PLY at COLMAP validation views and evaluate it"
    )
    render.add_argument("model_ply", type=Path, help="Inria-style 3DGS PLY")
    render.add_argument(
        "scene_dir",
        type=Path,
        help="Scene containing images/images_N and sparse/0 COLMAP data",
    )
    render.add_argument(
        "--factor",
        type=int,
        default=2,
        help="Image downscale used to auto-select images_N (default: 2)",
    )
    render.add_argument(
        "--image-dir",
        help="Image directory relative to scene_dir; overrides --factor auto-selection",
    )
    render.add_argument(
        "--sparse-dir",
        help="COLMAP directory relative to scene_dir (default: sparse/0, then sparse)",
    )
    render.add_argument(
        "--test-every",
        type=int,
        default=8,
        help="After filename sorting, render indices divisible by N; 0 renders all",
    )
    render.add_argument(
        "--render-dir",
        type=Path,
        help="Rendered PNG directory (default: MODEL_PARENT/psnr_renders)",
    )
    render.add_argument(
        "--background",
        choices=("black", "white"),
        default="black",
        help="Rasterization background (default: black)",
    )
    render.add_argument(
        "--save-bit-depth",
        choices=(8, 16),
        type=int,
        default=16,
        help="Quantization used for saved renders and PSNR (default: 16)",
    )
    add_metric_arguments(render, "cuda")

    compare = commands.add_parser(
        "compare", help="Calculate PSNR for two existing image directories"
    )
    compare.add_argument("reference_dir", type=Path)
    compare.add_argument("render_dir", type=Path)
    compare.add_argument(
        "--match",
        choices=("relative", "name", "stem"),
        default="relative",
        help="Case-insensitive pairing key (default: relative)",
    )
    compare.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Ignore files without a partner instead of failing",
    )
    add_metric_arguments(compare, "auto")
    return parser.parse_args()


def collect_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"Image directory does not exist: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
    )


def make_key_function(root: Path, mode: str) -> Callable[[Path], str]:
    if mode == "relative":
        return lambda path: path.relative_to(root).as_posix().casefold()
    if mode == "name":
        return lambda path: path.name.casefold()
    return lambda path: path.stem.casefold()


def index_images(root: Path, mode: str) -> dict[str, Path]:
    key_for = make_key_function(root, mode)
    indexed: dict[str, Path] = {}
    for path in collect_images(root):
        key = key_for(path)
        if key in indexed:
            raise ValueError(
                f"Duplicate {mode} key {key!r}: {indexed[key]} and {path}"
            )
        indexed[key] = path
    return indexed


def pair_images(
    reference_dir: Path,
    render_dir: Path,
    mode: str,
    allow_unmatched: bool,
) -> list[tuple[str, Path, Path]]:
    references = index_images(reference_dir, mode)
    renders = index_images(render_dir, mode)
    reference_keys = set(references)
    render_keys = set(renders)
    missing_renders = sorted(reference_keys - render_keys)
    missing_references = sorted(render_keys - reference_keys)
    if not allow_unmatched and (missing_renders or missing_references):
        details: list[str] = []
        if missing_renders:
            details.append("missing renders: " + ", ".join(missing_renders[:10]))
        if missing_references:
            details.append(
                "missing references: " + ", ".join(missing_references[:10])
            )
        raise ValueError("Image sets do not match; " + "; ".join(details))
    keys = sorted(reference_keys & render_keys)
    if not keys:
        raise ValueError("No matching image pairs were found")
    return [(key, references[key], renders[key]) for key in keys]


def load_rgb_image(path: Path):
    import cv2
    import numpy as np

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape}: {path}")

    channels = image.shape[2]
    if channels == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif channels == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        raise ValueError(f"Unsupported channel count {channels}: {path}")

    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)
        if not np.isfinite(image).all():
            raise ValueError(f"Image contains NaN or Inf: {path}")
        minimum, maximum = float(image.min()), float(image.max())
        if minimum < 0.0 or maximum > 1.0:
            raise ValueError(
                f"Floating image is outside [0, 1] ({minimum}, {maximum}): {path}"
            )
        return image
    raise ValueError(f"Unsupported image dtype {image.dtype}: {path}")


def select_device(requested: str, torch_module, require_cuda: bool = False) -> str:
    selected = requested
    if requested == "auto":
        selected = "cuda" if torch_module.cuda.is_available() else "cpu"
    if selected.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(f"{selected} was requested, but CUDA is unavailable")
    if require_cuda and not selected.startswith("cuda"):
        raise RuntimeError("gsplat rendering requires a CUDA device")
    try:
        torch_module.empty((), device=selected)
    except Exception as error:
        raise RuntimeError(f"Invalid or unavailable torch device {selected!r}") from error
    return selected


def image_to_tensor(image, torch_module, np_module, device: str):
    return torch_module.from_numpy(
        np_module.ascontiguousarray(image.transpose(2, 0, 1))
    ).unsqueeze(0).to(device)


def calculate_psnr(reference, rendered, metric, torch_module, np_module, device: str) -> float:
    if reference.shape != rendered.shape:
        raise ValueError(
            f"Image shape mismatch: reference {reference.shape}, render {rendered.shape}"
        )
    reference_tensor = image_to_tensor(reference, torch_module, np_module, device)
    render_tensor = image_to_tensor(rendered, torch_module, np_module, device)
    value = float(metric(render_tensor, reference_tensor).item())
    metric.reset()
    return value


def json_number(value: float):
    if math.isfinite(value):
        return value
    return "inf" if value > 0.0 else "-inf"


def mean_psnr(values: list[float]) -> float:
    if not values:
        raise ValueError("No PSNR values were produced")
    if any(math.isinf(value) and value > 0.0 for value in values):
        return math.inf
    return math.fsum(values) / len(values)


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def compare_directories(args, torch_module, np_module, metric_class) -> int:
    device = select_device(args.device, torch_module)
    metric = metric_class(data_range=1.0).to(device)
    pairs = pair_images(
        args.reference_dir.resolve(),
        args.render_dir.resolve(),
        args.match,
        args.allow_unmatched,
    )
    results: list[dict[str, Any]] = []
    values: list[float] = []
    with torch_module.no_grad():
        for key, reference_path, render_path in pairs:
            value = calculate_psnr(
                load_rgb_image(reference_path),
                load_rgb_image(render_path),
                metric,
                torch_module,
                np_module,
                device,
            )
            values.append(value)
            results.append(
                {
                    "key": key,
                    "reference": str(reference_path),
                    "render": str(render_path),
                    "psnr_db": json_number(value),
                }
            )
            if not args.quiet:
                print(f"{key}: {value:.6f} dB")
    average = mean_psnr(values)
    print(f"Mean PSNR ({len(values)} images): {average:.6f} dB")
    write_report(
        args.output_json,
        {
            "mode": "compare",
            "reference_dir": str(args.reference_dir.resolve()),
            "render_dir": str(args.render_dir.resolve()),
            "match": args.match,
            "device": device,
            "data_range": 1.0,
            "image_count": len(values),
            "mean_psnr_db": json_number(average),
            "images": results,
        },
    )
    return 0


def camera_model_name(camera: Any) -> str:
    name = getattr(camera, "model_name", "")
    if name:
        return str(name).replace("CameraModelId.", "")
    model = getattr(camera, "model", "")
    return getattr(model, "name", str(model)).replace("CameraModelId.", "")


def image_world_to_camera(image: Any, np_module):
    transform = image.cam_from_world
    if callable(transform):
        transform = transform()
    matrix = np_module.eye(4, dtype=np_module.float32)
    matrix[:3, :4] = np_module.asarray(transform.matrix(), dtype=np_module.float32)
    return matrix


def select_sparse_dir(scene_dir: Path, requested: str | None) -> Path:
    if requested:
        path = scene_dir / requested
        if not path.is_dir():
            raise ValueError(f"COLMAP sparse directory does not exist: {path}")
        return path
    for relative in (Path("sparse/0"), Path("sparse")):
        path = scene_dir / relative
        if path.is_dir():
            return path
    raise ValueError(f"No sparse/0 or sparse COLMAP directory under {scene_dir}")


def select_image_dir(scene_dir: Path, factor: int, requested: str | None) -> Path:
    if requested:
        path = scene_dir / requested
        if not path.is_dir():
            raise ValueError(f"Image directory does not exist: {path}")
        return path
    candidates = []
    if factor > 1:
        candidates.extend((f"images_{factor}_png", f"images_{factor}"))
    candidates.append("images")
    for relative in candidates:
        path = scene_dir / relative
        if path.is_dir():
            return path
    raise ValueError(
        f"No image directory found under {scene_dir}; tried {', '.join(candidates)}"
    )


def relative_stem(path: Path) -> str:
    return path.with_suffix("").as_posix().casefold()


def map_colmap_images(image_dir: Path) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in collect_images(image_dir):
        key = relative_stem(path.relative_to(image_dir))
        if key in mapped:
            raise ValueError(f"Duplicate image stem {key!r} in {image_dir}")
        mapped[key] = path
    return mapped


def load_validation_views(args, np_module, pycolmap_module) -> tuple[list[dict[str, Any]], Path, Path]:
    scene_dir = args.scene_dir.resolve()
    sparse_dir = select_sparse_dir(scene_dir, args.sparse_dir)
    image_dir = select_image_dir(scene_dir, max(args.factor, 1), args.image_dir)
    available_images = map_colmap_images(image_dir)
    reconstruction = pycolmap_module.Reconstruction(str(sparse_dir))
    cameras = {int(key): value for key, value in reconstruction.cameras.items()}
    images = {int(key): value for key, value in reconstruction.images.items()}
    registered = [images[int(image_id)] for image_id in reconstruction.reg_image_ids()]
    registered.sort(key=lambda image: image.name.casefold())

    views: list[dict[str, Any]] = []
    for sorted_index, image in enumerate(registered):
        if args.test_every > 0 and sorted_index % args.test_every != 0:
            continue
        image_key = relative_stem(Path(image.name))
        reference_path = available_images.get(image_key)
        if reference_path is None:
            raise ValueError(
                f"No image in {image_dir} matches COLMAP image {image.name!r}"
            )
        camera = cameras[int(image.camera_id)]
        model = camera_model_name(camera)
        if model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(
                f"Camera {image.camera_id} uses unsupported model {model}; "
                "provide an undistorted PINHOLE/SIMPLE_PINHOLE COLMAP scene"
            )
        reference = load_rgb_image(reference_path)
        height, width = reference.shape[:2]
        calibration = np_module.asarray(
            camera.calibration_matrix(), dtype=np_module.float32
        ).copy()
        calibration[0, :] *= width / float(camera.width)
        calibration[1, :] *= height / float(camera.height)
        views.append(
            {
                "name": image.name,
                "reference_path": reference_path,
                "reference": reference,
                "width": width,
                "height": height,
                "K": calibration,
                "viewmat": image_world_to_camera(image, np_module),
            }
        )
    if not views:
        raise ValueError("The selected validation split contains no images")
    return views, image_dir, sparse_dir


def require_properties(properties: set[str], required: set[str], model_path: Path) -> None:
    missing = sorted(required - properties)
    if missing:
        raise ValueError(f"PLY {model_path} is missing properties: {', '.join(missing)}")


def load_gaussian_ply(model_path: Path, device: str, np_module, torch_module, plydata_class):
    if not model_path.is_file():
        raise ValueError(f"Model PLY does not exist: {model_path}")
    ply = plydata_class.read(str(model_path))
    vertex = ply["vertex"].data
    properties = set(vertex.dtype.names or ())
    required = {
        "x", "y", "z", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
        "f_dc_0", "f_dc_1", "f_dc_2",
    }
    require_properties(properties, required, model_path)

    means = np_module.stack([vertex["x"], vertex["y"], vertex["z"]], axis=1)
    log_scales = np_module.stack(
        [vertex[f"scale_{index}"] for index in range(3)], axis=1
    )
    quaternions = np_module.stack(
        [vertex[f"rot_{index}"] for index in range(4)], axis=1
    )
    raw_opacity = np_module.asarray(vertex["opacity"], dtype=np_module.float32)
    dc = np_module.stack(
        [vertex[f"f_dc_{index}"] for index in range(3)], axis=1
    )[:, None, :]

    rest_names = sorted(
        (name for name in properties if name.startswith("f_rest_")),
        key=lambda name: int(name.removeprefix("f_rest_")),
    )
    if len(rest_names) % 3 != 0:
        raise ValueError(f"PLY has invalid SH-rest property count: {len(rest_names)}")
    if rest_names:
        rest = np_module.stack([vertex[name] for name in rest_names], axis=1)
        rest = rest.reshape(len(vertex), 3, len(rest_names) // 3).transpose(0, 2, 1)
        colors = np_module.concatenate((dc, rest), axis=1)
    else:
        colors = dc
    coefficient_count = colors.shape[1]
    sh_degree = int(round(math.sqrt(coefficient_count) - 1))
    if (sh_degree + 1) ** 2 != coefficient_count:
        raise ValueError(f"PLY SH coefficient count {coefficient_count} is not square")

    import torch.nn.functional as functional

    means_tensor = torch_module.from_numpy(
        np_module.ascontiguousarray(means, dtype=np_module.float32)
    ).to(device)
    quats_tensor = functional.normalize(
        torch_module.from_numpy(
            np_module.ascontiguousarray(quaternions, dtype=np_module.float32)
        ).to(device),
        dim=-1,
    )
    scales_tensor = torch_module.exp(
        torch_module.from_numpy(
            np_module.ascontiguousarray(log_scales, dtype=np_module.float32)
        ).to(device)
    )
    opacities_tensor = torch_module.sigmoid(
        torch_module.from_numpy(raw_opacity).to(device)
    )
    colors_tensor = torch_module.from_numpy(
        np_module.ascontiguousarray(colors, dtype=np_module.float32)
    ).to(device)
    return (
        means_tensor,
        quats_tensor,
        scales_tensor,
        opacities_tensor,
        colors_tensor,
        sh_degree,
    )


def quantize_and_save_render(image, path: Path, bit_depth: int, np_module):
    import cv2

    maximum = 65535 if bit_depth == 16 else 255
    dtype = np_module.uint16 if bit_depth == 16 else np_module.uint8
    quantized = np_module.rint(np_module.clip(image, 0.0, 1.0) * maximum).astype(dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Failed to write rendered image: {path}")
    return quantized.astype(np_module.float32) / float(maximum)


def render_and_evaluate(args, torch_module, np_module, metric_class) -> int:
    try:
        import pycolmap
        from gsplat.rendering import rasterization
        from plyfile import PlyData
    except ImportError as error:
        raise RuntimeError(
            "Rendering dependencies are missing. Install pycolmap, plyfile, and gsplat."
        ) from error

    device = select_device(args.device, torch_module, require_cuda=True)
    views, image_dir, sparse_dir = load_validation_views(args, np_module, pycolmap)
    model = load_gaussian_ply(
        args.model_ply.resolve(), device, np_module, torch_module, PlyData
    )
    means, quats, scales, opacities, colors, sh_degree = model
    render_dir = (
        args.render_dir.resolve()
        if args.render_dir
        else args.model_ply.resolve().parent / "psnr_renders"
    )
    output_json = args.output_json or (render_dir / "psnr.json")
    metric = metric_class(data_range=1.0).to(device)
    background = None
    if args.background == "white":
        background = torch_module.ones((1, 3), dtype=torch_module.float32, device=device)

    results: list[dict[str, Any]] = []
    values: list[float] = []
    with torch_module.no_grad():
        for view_index, view in enumerate(views):
            viewmat = torch_module.from_numpy(view["viewmat"]).unsqueeze(0).to(device)
            calibration = torch_module.from_numpy(view["K"]).unsqueeze(0).to(device)
            kwargs: dict[str, Any] = {}
            if background is not None:
                kwargs["backgrounds"] = background
            rendered, _, _ = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=viewmat,
                Ks=calibration,
                width=view["width"],
                height=view["height"],
                sh_degree=sh_degree,
                packed=False,
                render_mode="RGB",
                camera_model="pinhole",
                **kwargs,
            )
            rendered_rgb = rendered[0, ..., :3].clamp(0.0, 1.0).cpu().numpy()
            render_path = render_dir / Path(view["name"]).with_suffix(".png")
            rendered_quantized = quantize_and_save_render(
                rendered_rgb, render_path, args.save_bit_depth, np_module
            )
            value = calculate_psnr(
                view["reference"],
                rendered_quantized,
                metric,
                torch_module,
                np_module,
                device,
            )
            values.append(value)
            results.append(
                {
                    "index": view_index,
                    "name": view["name"],
                    "reference": str(view["reference_path"]),
                    "render": str(render_path),
                    "psnr_db": json_number(value),
                }
            )
            if not args.quiet:
                print(f"[{view_index + 1}/{len(views)}] {view['name']}: {value:.6f} dB")

    average = mean_psnr(values)
    print(f"Mean PSNR ({len(values)} validation images): {average:.6f} dB")
    write_report(
        output_json,
        {
            "mode": "render",
            "model_ply": str(args.model_ply.resolve()),
            "scene_dir": str(args.scene_dir.resolve()),
            "image_dir": str(image_dir),
            "sparse_dir": str(sparse_dir),
            "test_every": args.test_every,
            "background": args.background,
            "save_bit_depth": args.save_bit_depth,
            "device": device,
            "data_range": 1.0,
            "gaussian_count": int(means.shape[0]),
            "sh_degree": sh_degree,
            "image_count": len(values),
            "mean_psnr_db": json_number(average),
            "images": results,
        },
    )
    if not args.quiet:
        print(f"Renders: {render_dir}")
        print(f"Report: {output_json}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        import numpy as np
        import torch
        from torchmetrics.image import PeakSignalNoiseRatio

        if args.command == "compare":
            return compare_directories(args, torch, np, PeakSignalNoiseRatio)
        return render_and_evaluate(args, torch, np, PeakSignalNoiseRatio)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"PSNR evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
