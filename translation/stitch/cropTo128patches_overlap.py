import os
import sys
import numpy as np
import cv2
import json
from datetime import datetime
from pathlib import Path
from PIL import Image

WORKDIR = Path(__file__).resolve().parent
if str(WORKDIR) not in sys.path:
    sys.path.insert(0, str(WORKDIR))

try:
    from stitch.autobc import auto_adjust_to_unified_brightness_contrast
except ImportError:

    def auto_adjust_to_unified_brightness_contrast(image, *args, **kwargs):
        if isinstance(image, Image.Image):
            return image, False, {}
        return image, False, {}

ENABLE_BRIGHTNESS_ADJUSTMENT = False
ENABLE_CONTRAST_ADJUSTMENT = False

def save_configuration(output_dir, drum_dir, he_dir, window_size, step_size, total_patches, patch_dimensions):

    config = {
        "timestamp": datetime.now().isoformat(),
        "input_directories": {
            "drum_dir": drum_dir,
            "he_dir": he_dir
        },
        "processing_parameters": {
            "window_size": window_size,
            "step_size": step_size
        },
        "output_statistics": {
            "total_patches_generated": total_patches,
            "output_directory": output_dir,
            "patch_dimensions": patch_dimensions
        },
        "processing_info": {
            "script_name": "cropTo128patches_overlap.py",
            "description": "Crop DRUM and HE images using sliding window with overlap and combine them side by side"
        }
    }

    config_file_path = os.path.join(output_dir, "processing_config.json")
    with open(config_file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"Configuration saved to: {config_file_path}")
    return config_file_path

def cropTo128patches(drum_path: Path, size: int = 128):

    drum_path = drum_path.resolve()
    if not drum_path.exists():
        raise FileNotFoundError(f"Not found DRUM : {drum_path}")

    import math
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    dataset_dir = drum_path.parent
    output_dir = dataset_dir / "output"
    combined_output_dir = output_dir / f"{drum_path.stem}_pix2pix"
    combined_output_dir.mkdir(parents=True, exist_ok=True)

    drum_img = cv2.imread(str(drum_path))
    if drum_img is None:
        raise RuntimeError(f"Unable to read DRUM : {drum_path}")
    original_rows, original_cols = drum_img.shape[:2]
    step_size = int(round(size * (1 - 0.2)))
    num_patches_x = math.ceil((original_cols - size) / step_size) + 1
    num_patches_y = math.ceil((original_rows - size) / step_size) + 1
    padded_cols = step_size * (num_patches_x - 1) + size
    padded_rows = step_size * (num_patches_y - 1) + size
    pad_right = padded_cols - original_cols
    pad_bottom = padded_rows - original_rows
    if pad_right > 0 or pad_bottom > 0:
        drum_img = cv2.copyMakeBorder(
            drum_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT_101
        )

    total_patches = num_patches_x * num_patches_y
    white_patch = np.full((size, size, 3), 255, dtype=np.uint8)
    print(
        f"  Input-only crop: {original_cols}x{original_rows} -> "
        f"grid {num_patches_y}x{num_patches_x} ({total_patches} patches)"
    )
    print("  Writing combined patches with 8 workers; no full-size HE or *_B patches")

    pending = set()
    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="crop-writer") as executor:
        patch_number = 0
        for row in range(num_patches_y):
            y0 = row * step_size
            for col in range(num_patches_x):
                x0 = col * step_size
                patch_number += 1
                drum_patch = drum_img[y0:y0 + size, x0:x0 + size]
                combined = np.concatenate((drum_patch, white_patch), axis=1)
                output_path = combined_output_dir / f"{patch_number}.png"
                pending.add(executor.submit(cv2.imwrite, str(output_path), combined))
                if len(pending) >= 256:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        if not future.result():
                            raise RuntimeError("Write combined patch Failed")
                if patch_number % 1000 == 0:
                    print(f"  queued {patch_number}/{total_patches} patches")
        if pending:
            done, _ = wait(pending)
            for future in done:
                if not future.result():
                    raise RuntimeError("Write combined patch Failed")

    patch_dimensions = {
        f"{drum_path.stem}_input_only": {
            "drum_patches": {"x": num_patches_x, "y": num_patches_y},
            "he_patches": {"x": num_patches_x, "y": num_patches_y},
            "original_image_size": {"width": original_cols, "height": original_rows},
            "padded_image_size": {"width": padded_cols, "height": padded_rows},
        }
    }
    save_configuration(
        output_dir.as_posix(),
        dataset_dir.as_posix(),
        "input-only (patch-sized white B)",
        (size, size),
        step_size,
        total_patches,
        patch_dimensions,
    )
    print(f"Processing complete!Generated a total of {total_patches}  items input-only combined patches")
    return total_patches, [combined_output_dir]

if __name__ == "__main__":
    cropTo128patches(Path("../datasets/stitch/test.jpg"))
