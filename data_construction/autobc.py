import numpy as np
import cv2
from PIL import Image

def calculate_brightness_contrast(image):

    brightness = np.mean(image)

    contrast = np.std(image)

    return brightness, contrast

def imagej_auto_adjust_contrast_brightness(image_np, saturated_pixels=0.35):

    PEAK_CLIP_FRACTION = 0.10

    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    sat = max(float(saturated_pixels), 0.0) / 100.0

    if len(image_np.shape) == 3:

        channel_pixels = image_np.shape[0] * image_np.shape[1]
        tail_count = int(channel_pixels * sat / 2.0)

        adjusted_np = np.zeros_like(image_np, dtype=np.float64)

        for channel in range(3):
            channel_data = image_np[:, :, channel].flatten()

            hist, bins = np.histogram(channel_data, bins=256, range=(0, 256))

            peak_limit = max(1, int(channel_pixels * PEAK_CLIP_FRACTION))
            hist = np.minimum(hist, peak_limit)

            cumsum = np.cumsum(hist)

            min_val = 0
            for i in range(256):
                if cumsum[i] >= tail_count:
                    min_val = i
                    break

            actual_max = np.max(channel_data)
            max_val = actual_max
            for i in range(int(actual_max), -1, -1):
                if cumsum[i] <= channel_pixels - tail_count:
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
        tail_count = int(total_pixels * sat / 2.0)

        hist, bins = np.histogram(image_np.flatten(), bins=256, range=(0, 256))
        peak_limit = max(1, int(total_pixels * PEAK_CLIP_FRACTION))
        hist = np.minimum(hist, peak_limit)

        cumsum = np.cumsum(hist)

        min_val = 0
        for i in range(256):
            if cumsum[i] >= tail_count:
                min_val = i
                break

        actual_max = np.max(image_np)
        max_val = actual_max
        for i in range(int(actual_max), -1, -1):
            if cumsum[i] <= total_pixels - tail_count:
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

def auto_adjust_contrast_brightness(image_np):

    return imagej_auto_adjust_contrast_brightness(image_np, saturated_pixels=0.35)

def auto_adjust_brightness(image_np, target_brightness=128):

    import json
    import time

    current_min = np.min(image_np)
    current_max = np.max(image_np)
    current_brightness = np.mean(image_np)



    if current_min == 0 and current_max == 255:
        return image_np.astype(np.uint8)


    if current_max > current_min:
        normalized = (image_np.astype(np.float64) - current_min) / (current_max - current_min) * 255
        normalized = np.clip(normalized, 0, 255)
        adjusted_np = normalized.astype(np.uint8)


        return adjusted_np
    else:

        return image_np.astype(np.uint8)

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

def adjust_contrast_brightness(fixed_image, moving_image):

    brightness1, contrast1 = calculate_brightness_contrast(np.array(fixed_image))

    adjusted_image2_np = adjust_brightness_contrast(moving_image, brightness1, contrast1)
    adjusted_image2 = Image.fromarray(adjusted_image2_np)

    return adjusted_image2

def check_overexposure(image_np, saturation_threshold=250, saturation_ratio_threshold=0.05,
                       mean_brightness_threshold=200):

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
                                               check_overexposure_flag=True, max_iterations=2,
                                               enable_brightness=True, enable_contrast=True):

    if isinstance(image, Image.Image):
        image_np = np.array(image)
        is_pil = True
    else:
        image_np = image
        is_pil = False

    if image_np.dtype != np.uint8:
        image_np = image_np.astype(np.uint8)

    current_target_brightness = target_brightness

    adjustment_info = {
        'original_target_brightness': target_brightness,
        'original_target_contrast': target_contrast,
        'final_target_brightness': target_brightness,
        'final_target_contrast': target_contrast,
        'iterations': 0,
        'overexposure_detected': False,
        'adjustment_applied': False
    }

    adjusted_np = image_np

    for iteration in range(max_iterations):
        adjustment_info['iterations'] = iteration + 1

        if enable_contrast:
            contrast_adjusted = auto_adjust_contrast_brightness(adjusted_np)
        else:
            contrast_adjusted = adjusted_np

        if enable_brightness:
            brightness_adjusted = auto_adjust_brightness(contrast_adjusted, current_target_brightness)
        else:
            brightness_adjusted = contrast_adjusted

        adjusted_np = brightness_adjusted

        if check_overexposure_flag:
            is_overexposed, sat_ratio, mean_bright, details = check_overexposure(adjusted_np)
            if is_overexposed:
                adjustment_info['overexposure_detected'] = True
                adjustment_info['adjustment_applied'] = True

                current_target_brightness = max(100, current_target_brightness - 15)
                adjustment_info['final_target_brightness'] = current_target_brightness

                if iteration < max_iterations - 1:
                    continue
            else:
                adjustment_info['final_target_brightness'] = current_target_brightness
                break
        else:
            break

    if is_pil:
        result_image = Image.fromarray(adjusted_np)
    else:
        result_image = adjusted_np

    if check_overexposure_flag:
        is_overexposed_final, sat_ratio_final, mean_bright_final, details_final = check_overexposure(adjusted_np)
        adjustment_info['final_saturation_ratio'] = sat_ratio_final
        adjustment_info['final_mean_brightness'] = mean_bright_final
        adjustment_info['overexposure_detected'] = is_overexposed_final

    return result_image, adjustment_info['overexposure_detected'], adjustment_info

def to_uint8(image_np: np.ndarray) -> np.ndarray:

    if image_np.dtype == np.uint8:
        return image_np

    img = image_np.astype(np.float32)
    if np.isfinite(img).all() and img.min() >= 0.0 and img.max() <= 1.0:
        img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    mn = np.nanmin(img)
    mx = np.nanmax(img)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(image_np, dtype=np.uint8)

    img = (img - mn) * (255.0 / (mx - mn))
    return np.clip(img, 0, 255).astype(np.uint8)

def imagej_enhance_contrast_normalize_like(
    image_u8: np.ndarray, saturated_percent: float = 0.35, need_stats: bool = False
):

    if image_u8.dtype != np.uint8:
        image_u8 = to_uint8(image_u8)

    channel_stats = []

    def _process_single(channel: np.ndarray) -> np.ndarray:
        flat = channel.reshape(-1)
        total = flat.size
        if total == 0:
            return channel
        hist = np.bincount(flat, minlength=256).astype(np.int64)
        limit = max(1, total // 10)
        hist = np.minimum(hist, limit)

        hist_sum = float(hist.sum())
        cutoff_side = hist_sum * (saturated_percent / 100.0) / 2.0
        cumsum = np.cumsum(hist)
        hmin = int(np.searchsorted(cumsum, cutoff_side))
        cumsum_rev = np.cumsum(hist[::-1])
        hmax = 255 - int(np.searchsorted(cumsum_rev, cutoff_side))

        if hmax <= hmin:
            hmin = int(flat.min())
            hmax = int(flat.max())
            if hmax <= hmin:
                if need_stats:
                    hist_nonzero = np.nonzero(hist)[0]
                    channel_stats.append({
                        "hmin": hmin,
                        "hmax": hmax,
                        "scale": 1.0,
                        "cutoff_side": float(cutoff_side),
                        "hist_sum": float(hist_sum),
                        "limit": int(limit),
                        "hist_min_val": int(hist_nonzero.min()) if len(hist_nonzero) > 0 else 0,
                        "hist_max_val": int(hist_nonzero.max()) if len(hist_nonzero) > 0 else 255,
                        "hist_peak_idx": int(np.argmax(hist)),
                        "hist_peak_val": int(hist[np.argmax(hist)]),
                        "cumsum_at_0": int(cumsum[0]) if len(cumsum) > 0 else 0,
                        "cumsum_at_255": int(cumsum[-1]) if len(cumsum) > 0 else 0,
                        "note": "hmax<=hmin, no stretch",
                    })
                return channel

        scale = 255.0 / float(hmax - hmin)
        out = (channel.astype(np.float32) - hmin) * scale
        out = np.clip(out, 0, 255).astype(np.uint8)
        if need_stats:
            hist_nonzero = np.nonzero(hist)[0]
            channel_stats.append({
                "hmin": hmin,
                "hmax": hmax,
                "scale": float(scale),
                "cutoff_side": float(cutoff_side),
                "hist_sum": float(hist_sum),
                "limit": int(limit),
                "hist_min_val": int(hist_nonzero.min()) if len(hist_nonzero) > 0 else 0,
                "hist_max_val": int(hist_nonzero.max()) if len(hist_nonzero) > 0 else 255,
                "hist_peak_idx": int(np.argmax(hist)),
                "hist_peak_val": int(hist[np.argmax(hist)]),
                "cumsum_at_0": int(cumsum[0]) if len(cumsum) > 0 else 0,
                "cumsum_at_255": int(cumsum[-1]) if len(cumsum) > 0 else 0,
            })
        return out

    if image_u8.ndim == 2:
        out = _process_single(image_u8)
    else:
        chs = cv2.split(image_u8)
        chs_proc = [_process_single(c) for c in chs]
        out = cv2.merge(chs_proc)

    if need_stats:
        return out, channel_stats
    return out
