import numpy as np
from PIL import Image
import os

def calculate_brightness_contrast(image):

    brightness = np.mean(image)


    contrast = np.std(image)

    return brightness, contrast

def imagej_auto_adjust_contrast_brightness(image_np, saturated_pixels=0.35):

    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    if len(image_np.shape) == 3:

        channel_pixels = image_np.shape[0] * image_np.shape[1]
        low_saturated_count = int(channel_pixels * saturated_pixels / 100.0)
        high_saturated_count = int(channel_pixels * saturated_pixels / 100.0)

        adjusted_np = np.zeros_like(image_np, dtype=np.float64)


        for channel in range(3):
            channel_data = image_np[:, :, channel].flatten()


            hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))


            cumsum = np.cumsum(hist)


            min_val = 0
            for i in range(256):
                if cumsum[i] >= low_saturated_count:
                    min_val = i
                    break


            max_val = 255
            for i in range(255, -1, -1):
                if cumsum[i] <= channel_pixels - high_saturated_count:
                    max_val = i
                    break


            if min_val >= max_val:
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)
                if min_val >= max_val:

                    adjusted_np[:, :, channel] = image_np[:, :, channel].astype(np.float64)
                    continue


            channel_float = image_np[:, :, channel].astype(np.float64)
            adjusted_channel = (channel_float - min_val) * 255.0 / (max_val - min_val)
            adjusted_channel = np.clip(adjusted_channel, 0, 255)
            adjusted_np[:, :, channel] = adjusted_channel

        adjusted_np = adjusted_np.astype(np.uint8)

    else:

        total_pixels = image_np.size
        low_saturated_count = int(total_pixels * saturated_pixels / 100.0)
        high_saturated_count = int(total_pixels * saturated_pixels / 100.0)


        hist, bins = np.histogram(image_np.flatten(), bins=256, range=(0, 256))


        cumsum = np.cumsum(hist)


        min_val = 0
        for i in range(256):
            if cumsum[i] >= low_saturated_count:
                min_val = i
                break


        max_val = 255
        for i in range(255, -1, -1):
            if cumsum[i] <= total_pixels - high_saturated_count:
                max_val = i
                break


        if min_val >= max_val:
            min_val = np.min(image_np)
            max_val = np.max(image_np)
            if min_val >= max_val:

                return image_np


        image_float = image_np.astype(np.float64)
        adjusted_np = (image_float - min_val) * 255.0 / (max_val - min_val)
        adjusted_np = np.clip(adjusted_np, 0, 255).astype(np.uint8)

    return adjusted_np

def auto_adjust_contrast_brightness(image_np, saturated_pixels=0.35):

    return imagej_auto_adjust_contrast_brightness(image_np, saturated_pixels=saturated_pixels)

def auto_adjust_brightness(image_np, target_brightness=128):

    current_brightness = np.mean(image_np)
    brightness_factor = target_brightness / current_brightness


    adjusted_np = image_np * brightness_factor
    adjusted_np = np.clip(adjusted_np, 0, 255)

    return adjusted_np.astype(np.uint8)

def adjust_brightness_contrast(image_np, brightness_target, contrast_target):

    current_mean = np.mean(image_np)
    current_std = np.std(image_np)


    if current_std < 1e-6:
        current_std = 1.0


    alpha = contrast_target / current_std
    alpha = np.clip(alpha, 0.5, 2.2)


    beta = brightness_target - current_mean

    beta = np.clip(beta, -50, 30)


    adjusted_np = np.float64(alpha) * image_np + beta


    predicted_mean = alpha * current_mean + beta
    if predicted_mean > 180:
        excess = predicted_mean - 180
        beta = beta - excess * 0.5
        adjusted_np = np.float64(alpha) * image_np + beta

    adjusted_np = np.clip(adjusted_np, 0, 255)
    return adjusted_np.astype(np.uint8)

def adjust_contrast_brightness(fixed_image,moving_image):

    brightness1, contrast1 = calculate_brightness_contrast(np.array(fixed_image))

    adjusted_image2_np = adjust_brightness_contrast(moving_image, brightness1, contrast1)
    adjusted_image2 = Image.fromarray(adjusted_image2_np)

    return adjusted_image2

def check_overexposure(image_np, saturation_threshold=250, saturation_ratio_threshold=0.05, mean_brightness_threshold=200):

    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)


    if len(image_np.shape) == 3:
        total_pixels = image_np.shape[0] * image_np.shape[1] * image_np.shape[2]
        saturated_pixels = (image_np >= saturation_threshold).sum()
        mean_brightness = image_np.mean()
    else:
        total_pixels = image_np.size
        saturated_pixels = (image_np >= saturation_threshold).sum()
        mean_brightness = image_np.mean()

    saturation_ratio = saturated_pixels / total_pixels


    is_overexposed = (saturation_ratio >= saturation_ratio_threshold) or (mean_brightness >= mean_brightness_threshold)

    details = {
        'saturation_ratio': saturation_ratio,
        'mean_brightness': mean_brightness,
        'saturated_pixels': saturated_pixels,
        'total_pixels': total_pixels
    }

    return is_overexposed, saturation_ratio, mean_brightness, details

def auto_adjust_to_unified_brightness_contrast(image, target_brightness=128, target_contrast=50,
                                               check_overexposure_flag=True, max_iterations=8,
                                               enable_brightness=True, enable_contrast=True,
                                               original_image_path=None):

    if isinstance(image, Image.Image):
        image_np = np.array(image)
        is_pil = True
    else:
        image_np = image
        is_pil = False

    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)


    original_image_np = image_np.copy()

    original_overexposed = False
    if check_overexposure_flag:
        original_overexposed, _, _, _ = check_overexposure(
            image_np,
            saturation_threshold=240,
            saturation_ratio_threshold=0.03,
            mean_brightness_threshold=190
        )
        if original_overexposed:

            target_brightness = min(target_brightness, 100)

    current_target_brightness = target_brightness
    current_saturated_pixels = 0.35
    if original_overexposed:
        current_saturated_pixels = 5.0

    adjustment_info = {
        'original_target_brightness': target_brightness,
        'original_target_contrast': target_contrast,
        'final_target_brightness': target_brightness,
        'final_target_contrast': target_contrast,
        'final_saturated_pixels': current_saturated_pixels,
        'iterations': 0,
        'overexposure_detected': False,
        'adjustment_applied': False,
        'original_overexposed': original_overexposed
    }

    adjusted_np = image_np

    for iteration in range(max_iterations):
        adjustment_info['iterations'] = iteration + 1

        if enable_contrast:

            contrast_adjusted = auto_adjust_contrast_brightness(adjusted_np, saturated_pixels=current_saturated_pixels)
        else:
            contrast_adjusted = adjusted_np

        if enable_brightness:
            brightness_adjusted = auto_adjust_brightness(contrast_adjusted, current_target_brightness)
        else:
            brightness_adjusted = contrast_adjusted

        brightness_adjusted = np.clip(brightness_adjusted, 0, 240).astype(np.uint8)

        adjusted_np = brightness_adjusted

        if check_overexposure_flag:

            is_overexposed, sat_ratio, mean_bright, details = check_overexposure(
                adjusted_np,
                saturation_threshold=240,
                saturation_ratio_threshold=0.03,
                mean_brightness_threshold=190
            )
            if is_overexposed:
                adjustment_info['overexposure_detected'] = True
                adjustment_info['adjustment_applied'] = True

                current_target_brightness = max(80, current_target_brightness - 20)

                current_saturated_pixels = min(10.0, current_saturated_pixels + 1.0)

                adjustment_info['final_target_brightness'] = current_target_brightness
                adjustment_info['final_saturated_pixels'] = current_saturated_pixels

                if iteration < max_iterations - 1:

                    if original_image_path and os.path.exists(original_image_path):
                        try:
                            original_img = Image.open(original_image_path).convert('RGB')
                            original_image_np = np.array(original_img)
                        except Exception:
                            pass
                    adjusted_np = original_image_np
                    continue
            else:
                adjustment_info['final_target_brightness'] = current_target_brightness
                adjustment_info['final_saturated_pixels'] = current_saturated_pixels
                break
        else:
            break

    if check_overexposure_flag:
        is_overexposed_final, sat_ratio_final, mean_bright_final, details_final = check_overexposure(
            adjusted_np,
            saturation_threshold=240,
            saturation_ratio_threshold=0.03,
            mean_brightness_threshold=190
        )

        if is_overexposed_final:

            final_target_brightness = max(70, current_target_brightness - 10)
            final_saturated_pixels = min(15.0, current_saturated_pixels + 2.0)


            final_source_np = original_image_np
            if original_image_path and os.path.exists(original_image_path):
                try:
                    original_img = Image.open(original_image_path).convert('RGB')
                    final_source_np = np.array(original_img)
                except Exception:
                    pass

            if enable_contrast:
                contrast_adjusted = auto_adjust_contrast_brightness(final_source_np, saturated_pixels=final_saturated_pixels)
            else:
                contrast_adjusted = final_source_np

            if enable_brightness:
                adjusted_np = auto_adjust_brightness(contrast_adjusted, final_target_brightness)
            else:
                adjusted_np = contrast_adjusted


            adjusted_np = np.clip(adjusted_np, 0, 230).astype(np.uint8)


            adjustment_info['final_target_brightness'] = final_target_brightness
            adjustment_info['final_saturated_pixels'] = final_saturated_pixels


            is_overexposed_final, sat_ratio_final, mean_bright_final, details_final = check_overexposure(
                adjusted_np,
                saturation_threshold=240,
                saturation_ratio_threshold=0.03,
                mean_brightness_threshold=190
            )

        adjustment_info['final_saturation_ratio'] = sat_ratio_final
        adjustment_info['final_mean_brightness'] = mean_bright_final
        adjustment_info['overexposure_detected'] = is_overexposed_final

    if is_pil:
        result_image = Image.fromarray(adjusted_np)
    else:
        result_image = adjusted_np

    return result_image, adjustment_info['overexposure_detected'], adjustment_info
