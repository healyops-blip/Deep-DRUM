import time

import numpy as np
import SimpleITK as itk
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp

from image_processing import adjust_contrast_brightness


def elastix_rigistration(fixed_imagepath, moving_imagepath):
    fixed_image = itk.imread(fixed_imagepath, itk.UC)
    moving_image = itk.imread(moving_imagepath, itk.UC)
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterMap(parameter_object.GetDefaultParameterMap("rigid"))
    result_image, _ = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        log_to_console=False,
    )
    elastix_object = itk.ElastixRegistrationMethod.New(fixed_image, moving_image)
    elastix_object.SetFixedImage(fixed_image)
    elastix_object.SetMovingImage(moving_image)
    elastix_object.SetParameterObject(parameter_object)
    elastix_object.SetLogToConsole(False)
    elastix_object.UpdateLargestPossibleRegion()
    result_image = elastix_object.GetOutput()
    filename = fixed_imagepath.split(".")[1].split("\\")[1]
    itk.imwrite(result_image, f"./cach/{filename}_elastixrig.jpg")
    return result_image


def optical_flow_rigistration(fixed_image, moving_image):
    fixed_gray = rgb2gray(fixed_image)
    moving_gray = rgb2gray(moving_image)
    vertical_flow, horizontal_flow = optical_flow_tvl1(fixed_gray, moving_gray)
    rows, columns = fixed_gray.shape
    row_coords, column_coords = np.meshgrid(
        np.arange(rows), np.arange(columns), indexing="ij"
    )
    warped = warp(
        moving_gray,
        np.array([row_coords + vertical_flow, column_coords + horizontal_flow]),
        mode="edge",
    )
    registered = np.zeros((rows, columns, 3))
    registered[..., 0] = warped
    registered[..., 1] = fixed_gray
    registered[..., 2] = fixed_gray
    return registered


def tolerance_optimization(
    source_image,
    target_image,
    lower_bound,
    upper_bound,
    step,
    num_sampling=3,
):
    from utils import random_crop

    objective_optimal = 0
    tolerance_optimal = lower_bound
    for tolerance in range(lower_bound, upper_bound, step):
        start_time = time.time()
        for _ in range(num_sampling):
            source_patch, _ = random_crop(source_image, tolerance)
            target_patch, _ = random_crop(target_image, tolerance)
            objective = ssim(
                np.array(source_patch),
                np.array(target_patch),
                data_range=255,
                channel_axis=2,
                win_size=5,
            )
            if objective > objective_optimal:
                tolerance_optimal = tolerance
                objective_optimal = objective
        print(f"Tolerance {tolerance} evaluated in {time.time() - start_time:.2f} seconds")
    print(f"Optimal tolerance: {tolerance_optimal}; objective: {objective_optimal:.4f}")
    return tolerance_optimal


def calculate_ssim_score(image1, image2):
    image1 = np.array(image1) if isinstance(image1, Image.Image) else image1
    image2 = np.array(image2) if isinstance(image2, Image.Image) else image2
    return ssim(image1, image2, data_range=255, channel_axis=2, win_size=5)


def reduce_image_quality_post_registration(image, quality_factor=0.5):
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_array = np.array(image)
    if quality_factor < 0.3:
        quantized = (image_array // 16) * 16
    elif quality_factor < 0.5:
        quantized = (image_array // 4) * 4
    elif quality_factor < 0.7:
        quantized = (image_array // 2) * 2
    else:
        quantized = np.round(image_array * quality_factor / quality_factor) * quality_factor
    return Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))


def adjust_image_quality_post_registration(image, quality_factor=0.5):
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image_array = np.array(image)
    if quality_factor >= 0.8:
        quantized = np.round(image_array * quality_factor / quality_factor) * quality_factor
    elif quality_factor >= 0.6:
        quantized = np.round(image_array / 2) * 2
    elif quality_factor >= 0.4:
        quantized = np.round(image_array / 4) * 4
    elif quality_factor >= 0.2:
        quantized = np.round(image_array / 8) * 8
    else:
        quantized = np.round(image_array / 16) * 16
    return Image.fromarray(np.clip(quantized, 0, 255).astype(np.uint8))


def enhance_and_save_images(
    source_image,
    target_image,
    output_folder,
    num,
    column,
    row,
    max_ssim,
    rig_ssim,
    drum_quality_factor=1.0,
    drum_resolution_scale=1.0,
):
    from autobc import auto_adjust_to_unified_brightness_contrast
    from utils import save_synchronized_images

    adjusted_source = adjust_contrast_brightness(source_image, target_image.convert("L"))
    if drum_resolution_scale != 1.0:
        adjusted_source = adjust_image_quality_post_registration(
            adjusted_source, drum_resolution_scale
        )
    if drum_quality_factor != 1.0:
        adjusted_source = reduce_image_quality_post_registration(
            adjusted_source, drum_quality_factor
        )

    adjusted_source, is_overexposed, adjustment_info = (
        auto_adjust_to_unified_brightness_contrast(
            adjusted_source,
            target_brightness=128,
            target_contrast=50,
            check_overexposure_flag=True,
            max_iterations=2,
            enable_brightness=False,
            enable_contrast=True,
        )
    )
    if is_overexposed:
        print(
            f"Patch {num}_{column}_{row} remains overexposed: "
            f"saturation={adjustment_info.get('final_saturation_ratio', 0):.2%}, "
            f"mean={adjustment_info.get('final_mean_brightness', 0):.1f}"
        )
    if adjusted_source.mode != "RGB":
        adjusted_source = adjusted_source.convert("RGB")
    return save_synchronized_images(
        target_image,
        adjusted_source,
        output_folder,
        num,
        column,
        row,
        max_ssim,
        rig_ssim,
    )
