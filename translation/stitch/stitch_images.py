from __future__ import annotations

import os
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

try:
    from stitch.autobc import auto_adjust_to_unified_brightness_contrast
except ImportError:

    def auto_adjust_to_unified_brightness_contrast(image, *args, **kwargs):
        if isinstance(image, Image.Image):
            return image, False, {}
        return image, False, {}

ENABLE_BC_ADJUSTMENT = False

WORKDIR = Path(__file__).resolve().parent
IMAGES_DIR = WORKDIR / "images"

def parse_tile_configuration(config_path: Path) -> List[Tuple[Path, float, float]]:

    tiles: List[Tuple[Path, float, float]] = []
    with config_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("dim"):
                continue

            try:
                coords_str = line[line.index("(") + 1 : line.index(")")]
                x_str, y_str = [p.strip() for p in coords_str.split(",")]
                x, y = float(x_str), float(y_str)
                img_name = line.split(";")[0].strip()
                img_path = (IMAGES_DIR / img_name).resolve()
                tiles.append((img_path, x, y))
            except Exception:
                continue
    return tiles

def compute_canvas_size(tiles: List[Tuple[Path, float, float]]) -> Tuple[int, int]:
    max_w = 0.0
    max_h = 0.0
    for img_path, x, y in tiles:
        with Image.open(img_path) as im:
            w, h = im.size
        max_w = max(max_w, x + w)
        max_h = max(max_h, y + h)
    return int(np.ceil(max_w)), int(np.ceil(max_h))

def _cosine_ramp(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 - np.cos(np.pi * np.clip(values, 0.0, 1.0)))

def _feather_mask(width: int, height: int, feather_px: int) -> np.ndarray:
    if feather_px <= 0:
        return np.ones((height, width), dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    dist_y = np.minimum(y, (height - 1) - y)[:, None]
    dist_x = np.minimum(x, (width - 1) - x)
    wy = _cosine_ramp(dist_y / float(feather_px))
    wx = _cosine_ramp(dist_x / float(feather_px))
    return (wy * wx).astype(np.float32)

def blend_tiles_batch(tiles: List[Tuple[Path, float, float]], canvas_size: Tuple[int, int], feather_px: int, batch_size: int = 100) -> Image.Image:

    canvas_w, canvas_h = canvas_size
    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    for batch_start in range(0, len(tiles), batch_size):
        batch_end = min(batch_start + batch_size, len(tiles))
        batch_tiles = tiles[batch_start:batch_end]

        print(f"  Processing batch {batch_start//batch_size + 1}/{(len(tiles) + batch_size - 1)//batch_size} ({len(batch_tiles)} tiles)")

        for img_path, x, y in batch_tiles:
            with Image.open(img_path) as im:
                tile = np.asarray(im.convert("RGB"), dtype=np.float32)
            h, w, _ = tile.shape
            xi = int(round(x))
            yi = int(round(y))

            x0 = max(0, xi)
            y0 = max(0, yi)
            x1 = min(canvas_w, xi + w)
            y1 = min(canvas_h, yi + h)
            if x0 >= x1 or y0 >= y1:
                continue

            tx0 = x0 - xi
            ty0 = y0 - yi
            tx1 = tx0 + (x1 - x0)
            ty1 = ty0 + (y1 - y0)

            roi = tile[ty0:ty1, tx0:tx1, :]
            mask_full = _feather_mask(w, h, feather_px)
            mroi = mask_full[ty0:ty1, tx0:tx1][:, :, None]


            roi_h, roi_w = mroi.shape[:2]
            if xi + w >= canvas_w:
                mroi[:, -feather_px:, :] = 1.0
            if yi + h >= canvas_h:
                mroi[-feather_px:, :, :] = 1.0
            if xi <= 0:
                mroi[:, :feather_px, :] = 1.0
            if yi <= 0:
                mroi[:feather_px, :, :] = 1.0

            acc[y0:y1, x0:x1, :] += roi * mroi
            weight[y0:y1, x0:x1, 0] += mroi[:, :, 0]

    mask = weight > 0
    out = np.zeros_like(acc, dtype=np.float32)
    out[mask.repeat(3, axis=2)] = (
        acc[mask.repeat(3, axis=2)] / weight.repeat(3, axis=2)[mask.repeat(3, axis=2)]
    )


    uncovered_mask = ~mask
    uncovered_count = uncovered_mask.sum()
    if uncovered_count > 0:
        uncovered_ratio = uncovered_count / (canvas_h * canvas_w)
        print(f"  Warning: {uncovered_count} pixels ({uncovered_ratio:.2%}) uncovered, filling with nearest neighbor")


        covered_mask = mask[:, :, 0]
        if covered_mask.any():
            for c in range(3):
                channel = out[:, :, c].copy()
                uncovered = ~covered_mask
                if uncovered.any():

                    for y in range(canvas_h):
                        for x in range(canvas_w):
                            if uncovered[y, x]:

                                found = False
                                for radius in range(1, min(10, max(canvas_w, canvas_h))):
                                    for dy in range(-radius, radius + 1):
                                        for dx in range(-radius, radius + 1):
                                            if dx*dx + dy*dy > radius*radius:
                                                continue
                                            ny, nx = y + dy, x + dx
                                            if 0 <= ny < canvas_h and 0 <= nx < canvas_w:
                                                if covered_mask[ny, nx]:
                                                    channel[y, x] = channel[ny, nx]
                                                    found = True
                                                    break
                                        if found:
                                            break
                                    if found:
                                        break
                out[:, :, c] = channel
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    else:
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def blend_tiles(tiles: List[Tuple[Path, float, float]], canvas_size: Tuple[int, int], feather_px: int) -> Image.Image:

    canvas_w, canvas_h = canvas_size
    acc = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

    for img_path, x, y in tiles:
        with Image.open(img_path) as im:
            tile = np.asarray(im.convert("RGB"), dtype=np.float32)
        h, w, _ = tile.shape
        xi = int(round(x))
        yi = int(round(y))

        x0 = max(0, xi)
        y0 = max(0, yi)
        x1 = min(canvas_w, xi + w)
        y1 = min(canvas_h, yi + h)
        if x0 >= x1 or y0 >= y1:
            continue

        tx0 = x0 - xi
        ty0 = y0 - yi
        tx1 = tx0 + (x1 - x0)
        ty1 = ty0 + (y1 - y0)

        roi = tile[ty0:ty1, tx0:tx1, :]
        mask_full = _feather_mask(w, h, feather_px)
        mroi = mask_full[ty0:ty1, tx0:tx1][:, :, None]


        roi_h, roi_w = mroi.shape[:2]
        if xi + w >= canvas_w:
            mroi[:, -feather_px:, :] = 1.0
        if yi + h >= canvas_h:
            mroi[-feather_px:, :, :] = 1.0
        if xi <= 0:
            mroi[:, :feather_px, :] = 1.0
        if yi <= 0:
            mroi[:feather_px, :, :] = 1.0

        acc[y0:y1, x0:x1, :] += roi * mroi
        weight[y0:y1, x0:x1, 0] += mroi[:, :, 0]

    mask = weight > 0
    out = np.zeros_like(acc, dtype=np.float32)
    out[mask.repeat(3, axis=2)] = (
        acc[mask.repeat(3, axis=2)] / weight.repeat(3, axis=2)[mask.repeat(3, axis=2)]
    )


    uncovered_mask = ~mask
    uncovered_count = uncovered_mask.sum()
    if uncovered_count > 0:
        uncovered_ratio = uncovered_count / (canvas_h * canvas_w)
        print(f"  Warning: {uncovered_count} pixels ({uncovered_ratio:.2%}) uncovered, filling with nearest neighbor")


        covered_mask = mask[:, :, 0]
        if covered_mask.any():
            for c in range(3):
                channel = out[:, :, c].copy()
                uncovered = ~covered_mask
                if uncovered.any():

                    for y in range(canvas_h):
                        for x in range(canvas_w):
                            if uncovered[y, x]:

                                found = False
                                for radius in range(1, min(10, max(canvas_w, canvas_h))):
                                    for dy in range(-radius, radius + 1):
                                        for dx in range(-radius, radius + 1):
                                            if dx*dx + dy*dy > radius*radius:
                                                continue
                                            ny, nx = y + dy, x + dx
                                            if 0 <= ny < canvas_h and 0 <= nx < canvas_w:
                                                if covered_mask[ny, nx]:
                                                    channel[y, x] = channel[ny, nx]
                                                    found = True
                                                    break
                                        if found:
                                            break
                                    if found:
                                        break
                out[:, :, c] = channel
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    else:
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def stitch_images(
    input_dir: str | Path,
    output_path: str | Path,
    pattern: str = "*_fake_B.png",
    pre_scale: int | None = None,
    rows: int | None = None,
    cols: int | None = None,
    overlap: float = 0.2,
    use_batch: bool = False,
    batch_size: int = 100,
    enable_bc_adjustment: bool = False,
    config_dir: Path | None = None,
) -> Path:

    global ENABLE_BC_ADJUSTMENT
    ENABLE_BC_ADJUSTMENT = enable_bc_adjustment

    input_dir = Path(input_dir).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if config_dir is None:
        config_dir = WORKDIR.parent / "datasets" / "stitch" / "output"
    created_scaled_dir: Path | None = None
    if pre_scale is not None:
        size = int(pre_scale)
        scaled_dir = WORKDIR / "_scaled_tmp"
        print(f"[1/3] Resizing images in {input_dir} to {size}x{size} ...")
        if scaled_dir.exists():
            shutil.rmtree(scaled_dir.as_posix(), ignore_errors=True)
        scaled_dir.mkdir(exist_ok=True)

        src_paths = sorted([p for p in input_dir.glob(pattern) if p.is_file()], key=lambda p: p.name)
        if not src_paths:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {input_dir} to pre-scale")
        for idx, p in enumerate(src_paths, start=1):
            with Image.open(p) as im:
                im = im.convert("RGB").resize((size, size), Image.BILINEAR)
                im.save((scaled_dir / p.name).as_posix())
            if idx % 500 == 0 or idx == len(src_paths):
                print(f"  resized {idx}/{len(src_paths)}")
        input_dir = scaled_dir
        created_scaled_dir = scaled_dir
    cfg_registered = input_dir / "TileConfiguration.registered.txt"
    cfg_plain = input_dir / "TileConfiguration.txt"
    cfg_path = cfg_registered if cfg_registered.exists() else cfg_plain

    def grid_stitch() -> Image.Image:

        step_size_from_config = None
        padded_size_from_config = None
        possible_config_paths = [
            config_dir / "processing_config.json",
            Path(input_dir).parent / "processing_config.json",
            Path(input_dir).parent.parent / "processing_config.json",
        ]
        config_path = None
        for cp in possible_config_paths:
            if cp.exists():
                config_path = cp
                break
        if config_path is not None:
            try:
                import json
                with config_path.open("r", encoding="utf-8") as f:
                    cfg = json.load(f)
                step_size_from_config = cfg.get("processing_parameters", {}).get("step_size")
                patch_dims = cfg.get("output_statistics", {}).get("patch_dimensions", {})
                if patch_dims:
                    first_key = next(iter(patch_dims))
                    padded_size = patch_dims[first_key].get("padded_image_size", {})
                    if padded_size:
                        padded_size_from_config = (padded_size.get("width"), padded_size.get("height"))
            except Exception as e:
                print(f"  Warning: Could not load step_size from config: {e}")


        paths = [p for p in input_dir.glob(pattern) if p.is_file()]
        def numeric_key(p: Path) -> tuple:
            name = p.stem
            num = 10**9
            i = 0
            while i < len(name) and name[i].isdigit():
                i += 1
            if i > 0:
                try:
                    num = int(name[:i])
                except Exception:
                    num = 10**9
            return (num, name)
        paths.sort(key=numeric_key)
        if not paths:
            raise FileNotFoundError(f"No files matching pattern '{pattern}' found in {input_dir}")

        n = len(paths)
        if rows is not None and cols is not None:
            grid_rows = int(rows)
            grid_cols = int(cols)
        else:
            grid_cols = int(round(np.sqrt(n)))
            grid_cols = max(grid_cols, 1)
            grid_rows = int(np.ceil(n / grid_cols))

        total_needed = min(n, grid_rows * grid_cols)
        paths = paths[:total_needed]

        with Image.open(paths[0]) as im0:
            w0, h0 = im0.size


        if step_size_from_config is not None:
            step_x = step_size_from_config
            step_y = step_size_from_config
            print(f"  Using step_size from config: {step_size_from_config}")
        else:
            step_x = int(round(w0 * (1.0 - float(overlap))))
            step_y = int(round(h0 * (1.0 - float(overlap))))
            step_x = max(1, step_x)
            step_y = max(1, step_y)
            print(f"  Calculated step_size from tile size: {step_x}")

        overlap_px_x = max(0, w0 - step_x)
        overlap_px_y = max(0, h0 - step_y)
        feather_px = max(1, int(round(0.5 * min(overlap_px_x, overlap_px_y))))

        tiles: List[Tuple[Path, float, float]] = []
        for idx, p in enumerate(paths):
            r = idx // grid_cols
            c = idx % grid_cols
            x = c * step_x
            y = r * step_y
            tiles.append((p, float(x), float(y)))


        if padded_size_from_config is not None:
            canvas_size = (padded_size_from_config[0], padded_size_from_config[1])
            print(f"  Using padded_size from config: {canvas_size}")

            if tiles:
                last_tile = tiles[-1]
                last_x, last_y = last_tile[1], last_tile[2]
                last_tile_end_x = last_x + w0
                last_tile_end_y = last_y + h0
                if last_tile_end_x < canvas_size[0] or last_tile_end_y < canvas_size[1]:
                    print(f"  Warning: Last tile ends at ({last_tile_end_x}, {last_tile_end_y}), canvas is {canvas_size}, may cause black edges")
        else:
            canvas_size = compute_canvas_size(tiles)

        if use_batch:
            return blend_tiles_batch(tiles, canvas_size, feather_px, batch_size)
        else:
            return blend_tiles(tiles, canvas_size, feather_px)

    if cfg_path.exists():
        tiles = parse_tile_configuration(cfg_path)
        if not tiles:

            print("[2/3] TileConfiguration empty; falling back to grid stitching ...")
            mosaic_img = grid_stitch()
        else:
            print("[2/3] Stitching using TileConfiguration positions ...")

            with Image.open(tiles[0][0]) as t0:
                w0, h0 = t0.size
            xs = sorted(int(round(x)) for _, x, _ in tiles)
            ys = sorted(int(round(y)) for _, _, y in tiles)
            def min_pos_delta(vals: list[int]) -> int:
                best = 10**9
                for i in range(1, len(vals)):
                    d = vals[i] - vals[i-1]
                    if d > 0 and d < best:
                        best = d
                return best if best != 10**9 else 0
            dx = min_pos_delta(xs)
            dy = min_pos_delta(ys)
            overlap_px_x = max(0, w0 - dx) if dx > 0 else 0
            overlap_px_y = max(0, h0 - dy) if dy > 0 else 0
            feather_px = max(1, int(round(0.5 * min(overlap_px_x, overlap_px_y))))
            canvas_size = compute_canvas_size(tiles)
            if use_batch:
                mosaic_img = blend_tiles_batch(tiles, canvas_size, feather_px, batch_size)
            else:
                mosaic_img = blend_tiles(tiles, canvas_size, feather_px)
    else:
        print("[2/3] Stitching using grid layout ...")
        mosaic_img = grid_stitch()


    if ENABLE_BC_ADJUSTMENT:
        output_name_lower = str(out_path).lower()
        if "drum" in output_name_lower or "mzw" in output_name_lower or "test" in output_name_lower or "hp" in output_name_lower:
            print("[3/3] Applying brightness and contrast adjustment to stitched image ...")
            try:
                adjusted_img, _, _ = auto_adjust_to_unified_brightness_contrast(
                    mosaic_img,
                    target_brightness=128,
                    target_contrast=50,
                    check_overexposure_flag=False,
                    max_iterations=1,
                    enable_brightness=True,
                    enable_contrast=True
                )
                mosaic_img = adjusted_img
                print("  Adjustment applied successfully")
            except Exception as e:
                print(f"  Warning: Failed to apply color adjustment: {e}")
                print("  Saving original image without adjustment")

    mosaic_img.save(out_path.as_posix(), quality=95)
    print(f"[3/3] Saved mosaic to: {out_path}")

    if created_scaled_dir is not None and created_scaled_dir.exists():
        try:
            print(f"Cleaning up temporary resized files in {created_scaled_dir} ...")
            shutil.rmtree(created_scaled_dir.as_posix(), ignore_errors=True)
        except Exception as e:
            print(f"Warning: could not remove temp directory {created_scaled_dir}: {e}")

    return out_path

def main() -> None:

    parser = argparse.ArgumentParser(description="Stitch images into a mosaic")
    parser.add_argument(
        "--input",
        default=(WORKDIR / "images").as_posix(),
        help="Input images directory (will try TileConfiguration.*; else grid stitching)",
    )
    parser.add_argument(
        "--pattern",
        default="*_fake_B.png",
        help="Filename glob pattern to include (e.g. '*.png' or '*_fake_B.png')",
    )
    parser.add_argument(
        "--pre-scale",
        type=int,
        default=None,
        help="If set, resize all images in input dir to NxN before stitching",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Grid rows (use with --cols). If omitted, inferred square-ish.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Grid cols (use with --rows). If omitted, inferred square-ish.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.2,
        help="Assumed tile overlap ratio for grid stitching (e.g. 0.2 = 20%)",
    )
    parser.add_argument(
        "--output",
        default="mosaic.jpg",
        help="Output filename (jpg/png/tif)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for memory-efficient processing (default: 100)",
    )
    parser.add_argument(
        "--use-batch",
        action="store_true",
        help="Use batch processing for large datasets",
    )
    parser.add_argument(
        "--enable-bc-adjustment",
        action="store_true",
        help="Enable brightness and contrast adjustment for stitched images",
    )
    args = parser.parse_args()

    stitch_images(
        input_dir=args.input,
        output_path=WORKDIR / args.output,
        pattern=args.pattern,
        pre_scale=args.pre_scale,
        rows=args.rows,
        cols=args.cols,
        overlap=args.overlap,
        use_batch=args.use_batch,
        batch_size=args.batch_size,
        enable_bc_adjustment=args.enable_bc_adjustment,
    )

if __name__ == "__main__":
    main()
