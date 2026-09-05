import os
from PIL import Image
import numpy as np

def creatDir(folder_name):

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Directory '{folder_name}' created.")
    else:
        print(f"Directory '{folder_name}' already exists.")

def creatmulDir(output_folder):

    creatDir(output_folder)
    creatDir(os.path.join(output_folder, 'A'))
    creatDir(os.path.join(output_folder, 'B'))
    creatDir(os.path.join(output_folder, 'fold_AB'))
    creatDir(os.path.join(output_folder, 'test'))
    creatDir(os.path.join(output_folder, 'A', 'val'))
    creatDir(os.path.join(output_folder, 'B', 'val'))
    creatDir(os.path.join(output_folder, 'fold_AB', 'val'))
    creatDir(os.path.join(output_folder, 'A', 'test'))
    creatDir(os.path.join(output_folder, 'B', 'test'))
    creatDir(os.path.join(output_folder, 'fold_AB', 'test'))
    creatDir(os.path.join(output_folder, 'B', 'train'))
    creatDir(os.path.join(output_folder, 'A', 'train'))
    creatDir(os.path.join(output_folder, 'fold_AB', 'train'))

def read_image(image_path):

    image = Image.open(image_path).convert('RGB')
    if image is None:
        raise Exception("Error: Image reading failed, please check the filepath.")
    return image

def find_images_with_common_part(folder_path, common_part):

    matching_images = []
    for entry in os.scandir(folder_path):
        if entry.is_file():
            if common_part in entry.name:
                matching_images.append(entry.path)
    return matching_images

def combine_images_side_by_side(image_a_path, image_b_path, output_path):

    try:
        img_a = Image.open(image_a_path)
        img_b = Image.open(image_b_path)

        if img_a.mode != 'RGB':
            img_a = img_a.convert('RGB')
        if img_b.mode != 'RGB':
            img_b = img_b.convert('RGB')

        width_a, height_a = img_a.size
        width_b, height_b = img_b.size

        combined_width = width_a + width_b
        combined_height = max(height_a, height_b)

        combined_image = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))


        combined_image.paste(img_a, (0, 0))
        combined_image.paste(img_b, (width_a, 0))

        combined_image.save(output_path, 'PNG')
        return True

    except Exception as e:
        print(f"Could not process images {image_a_path} + {image_b_path}: {e}")
        return False

def random_crop(image, tolerance):

    width, height = image.size
    if tolerance > min(width, height):
        tolerance = min(width, height)

    max_x = width - tolerance
    max_y = height - tolerance

    if max_x <= 0 or max_y <= 0:

        return image, (0, 0)

    x = np.random.randint(0, max_x + 1)
    y = np.random.randint(0, max_y + 1)

    cropped_image = image.crop((x, y, x + tolerance, y + tolerance))
    return cropped_image, (x, y)

def calculate_mse(image):

    if image.mode != 'L':
        image = image.convert('L')

    img_array = np.array(image)
    mse = np.mean((img_array - np.mean(img_array)) ** 2)
    return mse

def save_synchronized_images(final_target, adjusted_source, output_folder, num, m, n, max_ssim, rig_ssim):

    output_folder_a = os.path.join(output_folder, 'A', 'train')
    output_folder_b = os.path.join(output_folder, 'B', 'train')


    filename = f"{num}_{m}_{n}_ssim_{max_ssim:.4f}_{rig_ssim:.4f}.png"


    final_target.save(os.path.join(output_folder_b, filename), "PNG")
    adjusted_source.save(os.path.join(output_folder_a, filename), "PNG")

    print(f"Saved synchronized images to A and B: {filename}")

    return os.path.join(output_folder_a, filename), os.path.join(output_folder_b, filename)
