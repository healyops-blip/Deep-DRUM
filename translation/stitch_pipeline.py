from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import json
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent

DATA_ROOT = (REPO_ROOT / "datasets" / "stitch").resolve()
IMAGE_ROOT = (DATA_ROOT / "test.jpg").resolve()
TEST_ROOT = (DATA_ROOT / "output").resolve()
MODEL_NAME = "deep_drum_pix2pix"
MODEL = "pix2pix"
DIRECTION = "AtoB"
NET_G = "unet_256"
NETWORK_SIZE = 256

ENABLE_CROP = True
ENABLE_TEST = True

ENABLE_BC_ADJUSTMENT = False
ENABLE_CROP_BC_ADJUSTMENT = True

PHASE = f"{IMAGE_ROOT.stem}_pix2pix"

def run_crop() -> tuple[int, list[Path], int, int]:

    from stitch import cropTo128patches_overlap as crop_pipeline

    print("[1/3] Cropping DRUM/HE images and generating combined patches ...")

    size = 128

    crop_pipeline.ENABLE_BRIGHTNESS_ADJUSTMENT = False
    crop_pipeline.ENABLE_CONTRAST_ADJUSTMENT = False
    total_patches, combined_dirs = crop_pipeline.cropTo128patches(IMAGE_ROOT, size)
    combined_dirs = [Path(p) for p in combined_dirs]
    print(f"[1/3] Done. Output patches under {TEST_ROOT}")
    print(f"[1/3] Combined dirs: {combined_dirs}")
    if ENABLE_CROP_BC_ADJUSTMENT:
        print("[1/3] Brightness/contrast adjustment deferred to parallel DataLoader workers")

    config_path = TEST_ROOT / "processing_config.json"
    grid_rows = grid_cols = 0
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        patch_dims = cfg.get("output_statistics", {}).get("patch_dimensions", {})
        if patch_dims:
            first_key = next(iter(patch_dims))
            drum_info = patch_dims[first_key]["drum_patches"]

            grid_cols = int(drum_info["x"])
            grid_rows = int(drum_info["y"])
            print(f"[1/3] Grid from step-1 (rows, cols) = ({grid_rows}, {grid_cols})")
    except Exception as e:
        print(f"[warn] Could not read grid config from {config_path}: {e}")

    return total_patches, combined_dirs, grid_rows, grid_cols

def run_test(num_test: int, combined_dirs: list[Path]) -> None:

    def count_pngs(directory: Path) -> int:
        if not directory.exists() or not directory.is_dir():
            return 0
        return len([p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".png"])

    expected_count = num_test
    if not combined_dirs:
        raise FileNotFoundError("No combined output directory was found")
    combined_dir = Path(combined_dirs[0])
    combined_count = count_pngs(combined_dir)
    if combined_count < expected_count:
        raise FileNotFoundError(
            f"Insufficient pix2pix data: expected {expected_count} images, "
            f"but {combined_dir} contains {combined_count}"
        )

    model = MODEL
    model_name = MODEL_NAME
    phase = PHASE
    direction = DIRECTION
    test_root = TEST_ROOT
    num_test_str = str(num_test)

    results_images = REPO_ROOT.parent / "results" / model_name / f"{phase}_latest" / "images"
    existing = 0
    if results_images.exists():
        existing = len(list(results_images.glob("*_fake_B.png")))
    if existing >= num_test and not ENABLE_TEST :
        print(
            f"[2/3] Skip test.py: found {existing} >= {num_test} fake_B images in {results_images}"
        )
        return
    """Invoke test.py with the requested pix2pix options."""
    cmd = [
        sys.executable,
        "test.py",
        "--dataroot",
        test_root.as_posix(),
        "--name",
        model_name,
        "--phase",
        phase,
        "--model",
        model,
        "--direction",
        direction,
        "--num_test",
        num_test_str,
        "--checkpoints_dir",
        (REPO_ROOT / "checkpoints").as_posix(),
        "--results_dir",
        (REPO_ROOT.parent / "results").as_posix(),
    ]
    print("[2/3] Running pix2pix test.py ...")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT.as_posix())
    print("[2/3] pix2pix testing finished")

def run_stitch(grid_rows: int, grid_cols: int, generated_at_stitch_size: bool = False) -> None:

    from stitch.stitch_images import stitch_images

    model_name = MODEL_NAME
    phase = PHASE

    results_images = REPO_ROOT.parent / "results" / model_name / f"{phase}_latest" / "images"

    stem = IMAGE_ROOT.stem
    out_stem = stem.replace("DRUM", "Hp") if "DRUM" in stem else f"{stem}_Hp"
    mosaic_name = f"{out_stem}.png"
    mosaic_path = (TEST_ROOT / mosaic_name).resolve()
    mosaic_path.parent.mkdir(parents=True, exist_ok=True)

    if not results_images.exists():
        print(
            f"[warn] Expected results directory {results_images} not found; "
            "stitch_images may fail if test outputs are missing."
        )

    pre_scale = None if generated_at_stitch_size else 128

    print("[3/3] Stitching fake_B images ...")
    stitch_images(
        input_dir=results_images,
        output_path=mosaic_path,
        pattern="*_fake_B.png",
        pre_scale=pre_scale,
        rows=grid_rows,
        cols=grid_cols,
        overlap=0.2,
        use_batch=True,
        batch_size=100,
        enable_bc_adjustment=ENABLE_BC_ADJUSTMENT,
        config_dir=TEST_ROOT,
    )
    print("[3/3] Stitching completed")


    if mosaic_path.exists():
        print("[3/3] Cropping stitched output to match original input size ...")
        try:

            orig_width, orig_height = None, None
            config_path = TEST_ROOT / "processing_config.json"
            if config_path.exists():
                try:
                    with config_path.open("r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    patch_dims = cfg.get("output_statistics", {}).get("patch_dimensions", {})
                    if patch_dims:
                        first_key = next(iter(patch_dims))
                        orig_size = patch_dims[first_key].get("original_image_size", {})
                        if orig_size:
                            orig_width = orig_size.get("width")
                            orig_height = orig_size.get("height")
                            print(f"  Original image size from config: {orig_width}x{orig_height}")
                except Exception as e:
                    print(f"  Could not read original size from config: {e}")


            if orig_width is None or orig_height is None:
                with Image.open(IMAGE_ROOT) as orig_img:
                    orig_width, orig_height = orig_img.size
                    print(f"  Original image size from file: {orig_width}x{orig_height}")


            with Image.open(mosaic_path) as mosaic_img:
                mosaic_width, mosaic_height = mosaic_img.size
                print(f"  Stitched mosaic size: {mosaic_width}x{mosaic_height}")


                if mosaic_width >= orig_width and mosaic_height >= orig_height:

                    cropped_img = mosaic_img.crop((0, 0, orig_width, orig_height))
                    cropped_img.save(mosaic_path.as_posix(), quality=95)
                    print(f"  ✓ Cropped to {orig_width}x{orig_height} and saved")
                else:

                    print(f"  ⚠ Warning: Stitched mosaic ({mosaic_width}x{mosaic_height}) is smaller than original ({orig_width}x{orig_height})")
                    print(f"  Saving mosaic as-is without cropping")
        except Exception as e:
            print(f"  ⚠ Warning: Failed to crop stitched output: {e}")
            print(f"  Saving original stitched mosaic without cropping")
    else:
        print(f"  ⚠ Warning: Stitched mosaic not found at {mosaic_path}")

def main() -> None:

    if ENABLE_CROP:
        total_patches, combined_dirs, grid_rows, grid_cols = run_crop()
    else:
        print("[1/3] Skipping crop step (ENABLE_CROP = False)")

        config_path = TEST_ROOT / "processing_config.json"
        grid_rows = grid_cols = 0
        combined_dirs = []
        total_patches = 0
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            patch_dims = cfg.get("output_statistics", {}).get("patch_dimensions", {})
            if patch_dims:
                first_key = next(iter(patch_dims))
                drum_info = patch_dims[first_key]["drum_patches"]
                grid_cols = int(drum_info["x"])
                grid_rows = int(drum_info["y"])
                total_patches = grid_rows * grid_cols
                print(f"[1/3] Loaded grid from config: (rows, cols) = ({grid_rows}, {grid_cols})")

            combined_dirs = [d for d in TEST_ROOT.glob("*_pix2pix") if d.is_dir()]
        except Exception as e:
            print(f"[warn] Could not read grid config from {config_path}: {e}")
            print(f"[warn] Will use default grid_rows=0, grid_cols=0")


    if ENABLE_TEST:
        run_test(total_patches, combined_dirs)
    else:
        print("[2/3] Skipping test step (ENABLE_TEST = False)")


    run_stitch(grid_rows, grid_cols)

if __name__ == "__main__":
    main()
