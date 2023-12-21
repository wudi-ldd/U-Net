import os
from PIL import Image

def resize_images(input_images_folder, input_masks_folder, output_images_folder, output_masks_folder, target_size):
    # Create the output folders if they don't exist.
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_masks_folder, exist_ok=True)

    for filename in os.listdir(input_images_folder):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_images_folder, filename)
            mask_path = os.path.join(input_masks_folder, filename.replace('.jpg', '.png'))

            # Resize image
            image = Image.open(image_path)
            resized_image = image.resize(target_size, Image.ANTIALIAS)
            resized_image_path = os.path.join(output_images_folder, filename)
            resized_image.save(resized_image_path)

            # Resize corresponding mask
            if os.path.exists(mask_path):
                mask = Image.open(mask_path)
                resized_mask = mask.resize(target_size, Image.NEAREST)
                resized_mask_path = os.path.join(output_masks_folder, filename.replace('.jpg', '.png'))
                resized_mask.save(resized_mask_path)

# Set the desired size, for example, (256, 256).
target_size = (1024, 768)

# Replace with the paths to your original image and mask folders.
input_images_folder = 'datasets/JPEGImages'
input_masks_folder = 'datasets/SegmentationClass'

# Replace with the folder paths where you want to save the resized images and masks.
output_images_folder = 'VOCdevkit\VOC2007/JPEGImages'
output_masks_folder = 'VOCdevkit\VOC2007/SegmentationClass'

resize_images(input_images_folder, input_masks_folder, output_images_folder, output_masks_folder, target_size)
