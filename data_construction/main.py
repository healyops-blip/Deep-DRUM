import os
import time

from registration_workflow import registration_workflow
from utils import combine_images_side_by_side, creatmulDir, find_images_with_common_part


def create_combined_images(output_folder):
    print("Creating paired images...")
    a_train_path = os.path.join(output_folder, "A", "train")
    b_train_path = os.path.join(output_folder, "B", "train")
    fold_ab_train_path = os.path.join(output_folder, "fold_AB", "train")

    for path in (a_train_path, b_train_path, fold_ab_train_path):
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Required directory does not exist: {path}")

    a_images = {name for name in os.listdir(a_train_path) if name.endswith(".png")}
    b_images = {name for name in os.listdir(b_train_path) if name.endswith(".png")}
    common_filenames = sorted(a_images & b_images)
    if not common_filenames:
        raise FileNotFoundError("A and B have no matching PNG filenames")

    success_count = 0
    for filename in common_filenames:
        a_path = os.path.join(a_train_path, filename)
        b_path = os.path.join(b_train_path, filename)
        output_path = os.path.join(fold_ab_train_path, filename)
        success_count += combine_images_side_by_side(a_path, b_path, output_path)

    print(f"Created {success_count}/{len(common_filenames)} paired images in {fold_ab_train_path}")


def main():
    start_time = time.time()
    input_folder = "./data/input"
    output_folder = "./output/dataset"
    drum_quality_factor = 0.3

    he_images = sorted(find_images_with_common_part(input_folder, "_HE"))
    drum_images = sorted(find_images_with_common_part(input_folder, "_DRUM"))
    if len(he_images) != len(drum_images):
        raise ValueError(
            f"DRUM/H&E image count mismatch: {len(drum_images)} DRUM and {len(he_images)} H&E"
        )

    print(f"Found {len(drum_images)} DRUM/H&E image pairs")
    creatmulDir(output_folder)

    for index, (drum_path, he_path) in enumerate(zip(drum_images, he_images)):
        print(f"Processing image pair {index + 1}/{len(he_images)}")
        workflow = registration_workflow(
            drum_path,
            he_path,
            index,
            output_folder,
            drum_quality_factor,
        )
        workflow.search_win()

    create_combined_images(output_folder)
    print(f"Data construction completed in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
