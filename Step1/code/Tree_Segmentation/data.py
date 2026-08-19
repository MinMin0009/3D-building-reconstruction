import os
import cv2
import numpy as np
from PIL import Image


# Load images and masks
def load_images_and_masks(image_dir, mask_dir):
    images = []
    masks = []
    image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))])

    for img_path, mask_path in zip(image_files, mask_files):
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"Image file {img_path} not found")
        if mask is None:
            raise FileNotFoundError(f"Mask file {mask_path} not found")

        # Convert mask values: white [255, 255, 255] to 1, others to 0
        mask_binary = np.all(mask == [255, 255, 255], axis=-1).astype(np.uint8)

        images.append(image)
        masks.append(mask_binary)

    return images, masks


# Normalize images
def normalize_images(images):
    images = np.array(images).astype('float32') / 255.0
    return images


# Resize images and masks
def resize_images_and_masks(images_folder, masks_folder, output_images_folder, output_masks_folder, scale=0.5):
    if not os.path.exists(output_images_folder):
        os.makedirs(output_images_folder)
    if not os.path.exists(output_masks_folder):
        os.makedirs(output_masks_folder)

    image_files = sorted(os.listdir(images_folder))
    mask_files = sorted(os.listdir(masks_folder))

    for image_file, mask_file in zip(image_files, mask_files):
        if image_file.endswith(('.png', '.jpg', '.jpeg')) and mask_file.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(images_folder, image_file)
            mask_path = os.path.join(masks_folder, mask_file)

            with Image.open(image_path) as img:
                new_size = (int(img.width * scale), int(img.height * scale))
                resized_img = img.resize(new_size, Image.LANCZOS)
                resized_img.save(os.path.join(output_images_folder, image_file))

            with Image.open(mask_path) as msk:
                resized_msk = msk.resize(new_size, Image.LANCZOS)
                resized_msk.save(os.path.join(output_masks_folder, mask_file))


# Resize images and masks to a fixed size
def resize_images_and_masks_to_size(images, masks, size=(256, 256)):
    images_resized = [cv2.resize(image, size, interpolation=cv2.INTER_LINEAR) for image in images]
    masks_resized = [cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST) for mask in masks]
    return images_resized, masks_resized


# Augmentation functions using OpenCV
def horizontal_flip(image, mask):
    return cv2.flip(image, 1), cv2.flip(mask, 1)


def vertical_flip(image, mask):
    return cv2.flip(image, 0), cv2.flip(mask, 0)


def rotate_image(image, mask, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(image, matrix, (w, h))
    rotated_mask = cv2.warpAffine(mask, matrix, (w, h))
    return rotated_img, rotated_mask


def add_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def change_brightness_contrast(image, brightness=30, contrast=30):
    new_image = np.int16(image)
    new_image = new_image * (contrast / 127 + 1) - contrast + brightness
    new_image = np.clip(new_image, 0, 255)
    return np.uint8(new_image)


# Augment data without using Albumentations
def augment_data(image_path, json_entry, augmented_image_dir, augmented_mask_dir, num_augments=3):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        return

    mask = create_mask_from_json(json_entry, image.shape)
    if mask is None or mask.sum() == 0:
        print(f"No valid mask found for {image_path}")
        return

    for i in range(num_augments):
        aug_img, aug_mask = image.copy(), mask.copy()

        if np.random.rand() > 0.5:
            aug_img, aug_mask = horizontal_flip(aug_img, aug_mask)
        if np.random.rand() > 0.5:
            aug_img, aug_mask = vertical_flip(aug_img, aug_mask)
        if np.random.rand() > 0.5:
            aug_img, aug_mask = rotate_image(aug_img, aug_mask, angle=np.random.randint(-45, 45))
        if np.random.rand() > 0.5:
            aug_img = add_gaussian_blur(aug_img)
        if np.random.rand() > 0.5:
            aug_img = change_brightness_contrast(aug_img)

        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        augmented_image_path = os.path.join(augmented_image_dir, f"{base_filename}_aug_{i}.png")
        augmented_mask_path = os.path.join(augmented_mask_dir, f"{base_filename}_aug_{i}.png")

        cv2.imwrite(augmented_image_path, aug_img)
        cv2.imwrite(augmented_mask_path, aug_mask)

        print(f"Saved augmented image and mask: {augmented_image_path}, {augmented_mask_path}")


# Create mask from JSON annotation
def create_mask_from_json(json_entry, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if 'regions' not in json_entry:
        print(f"Warning: 'regions' key not found in JSON entry.")
        return mask

    for region in json_entry['regions'].values():
        shape_attributes = region['shape_attributes']
        if shape_attributes['name'] == 'polygon':
            all_points_x = shape_attributes['all_points_x']
            all_points_y = shape_attributes['all_points_y']
            polygon = np.array(list(zip(all_points_x, all_points_y)), dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 255)
    return mask
