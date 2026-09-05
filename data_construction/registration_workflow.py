import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

from register_function import enhance_and_save_images, optical_flow_rigistration
from utils import calculate_mse, read_image


class registration_workflow:
    def __init__(
        self,
        drum_path,
        he_path,
        num,
        output_folder,
        drum_quality_factor=1.0,
    ):
        self.slide_step = 5
        self.tol = 64
        self.patch_drum = Image.fromarray(np.array(read_image(drum_path)))
        self.patch_he = read_image(he_path)
        self.search_width = 10
        self.search_height = 10
        self.num = num
        self.output_folder = output_folder
        self.drum_quality_factor = 1.0
        self.drum_resolution_scale = 1.0

    def search_win(self):
        width, height = self.patch_drum.size
        for column in range(width // self.tol):
            pos_x = self.tol * column
            for row in range(height // self.tol):
                pos_y = self.tol * row
                self.confirm_position(pos_x, pos_y, column, row)

        score = ssim(
            np.array(self.patch_drum),
            np.array(self.patch_he),
            data_range=255,
            channel_axis=2,
            win_size=5,
        )
        print(f"Image {self.num} source SSIM: {score:.4f}")
        return 0

    def confirm_position(self, pos_x, pos_y, column, row):
        drum_patch, patch_position = self.fix_crop(self.patch_drum, pos_x, pos_y)
        start_x = max(0, patch_position[0] - self.tol)
        start_y = max(0, patch_position[1] - self.tol)
        max_x = self.patch_he.size[0] - self.tol
        max_y = self.patch_he.size[1] - self.tol
        correlations = np.zeros((self.search_width, self.search_height))

        for x_index in range(self.search_width):
            for y_index in range(self.search_height):
                x = start_x + x_index * self.slide_step
                y = start_y + y_index * self.slide_step
                if x > max_x or y > max_y:
                    continue
                he_patch = self.patch_he.crop((x, y, x + self.tol, y + self.tol))
                correlations[x_index, y_index] = ssim(
                    np.array(he_patch),
                    np.array(drum_patch),
                    data_range=255,
                    channel_axis=2,
                    win_size=5,
                )

        if np.all(np.isnan(correlations)) or np.all(correlations == 0):
            print("Skipping a patch with no valid SSIM candidate")
            return 0.0

        max_position = np.unravel_index(np.argmax(correlations), correlations.shape)
        max_ssim = float(np.max(correlations))
        final_x = start_x + max_position[0] * self.slide_step
        final_y = start_y + max_position[1] * self.slide_step
        final_he = self.patch_he.crop(
            (final_x, final_y, final_x + self.tol, final_y + self.tol)
        )

        if 0 <= max_ssim <= 1 and calculate_mse(drum_patch) >= 0:
            registered = optical_flow_rigistration(final_he, drum_patch)
            registered_uint8 = np.clip(registered * 100, 0, 255).astype(np.uint8)
            rig_ssim = ssim(
                registered_uint8,
                np.array(final_he),
                data_range=255,
                channel_axis=2,
                win_size=5,
            )
            enhance_and_save_images(
                Image.fromarray(registered_uint8),
                final_he,
                self.output_folder,
                self.num,
                column,
                row,
                max_ssim,
                rig_ssim,
                self.drum_quality_factor,
                self.drum_resolution_scale,
            )
        return max_ssim

    def fix_crop(self, image, pos_x, pos_y):
        width, height = image.size
        pos_x = min(pos_x, width - self.tol)
        pos_y = min(pos_y, height - self.tol)
        return image.crop((pos_x, pos_y, pos_x + self.tol, pos_y + self.tol)), (pos_x, pos_y)

    def cal_mse(self, image):
        return calculate_mse(image)
