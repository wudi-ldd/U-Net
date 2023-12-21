#----------------------------------------------------#
# If the dataset images are too large, you can use this script to divide the images into smaller pieces for training, which improves the utilization rate of the video memory.
# If you have your own annotation json files, you can use patch_png_json.py to divide both the images and labels at the same time, and then convert them to VOC format using json_to_dataset.py.
# If you're using an open-source dataset that only has images and masks, you can use patch_png.py to split the images and labels at the same time. After splitting, you can divide them into training and validation sets and start training.
#----------------------------------------------------#
from PIL import Image
import os

# Set the path for the original image and label folders.
original_images_folder = 'datasets/JPEGImages/'
segmentation_class_folder = 'datasets/SegmentationClass/'

# Set the path for the folder where the sliced images will be saved.
output_images_folder = 'datasets/JPEGImagesPatch/'
output_segmentation_folder = 'datasets/SegmentationClassPatch/'

# Ensure the output folder exists.
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_segmentation_folder, exist_ok=True)

# Retrieve all the filenames of the original images.
image_files = os.listdir(original_images_folder)

# Iterate through each image.
for image_file in image_files:
    # Construct the file path.
    image_path = os.path.join(original_images_folder, image_file)
    segmentation_path = os.path.join(segmentation_class_folder, image_file.replace('.jpg', '.png'))

    # Open the image and corresponding label.
    image = Image.open(image_path)
    segmentation = Image.open(segmentation_path)

    # Get the size of the image.
    width, height = image.size

    # Parameters for slicing.
    slice_width = width // 2
    slice_height = height // 2

    # Cut the image and save.
    for i in range(2):
        for j in range(2):
            left = j * slice_width
            upper = i * slice_height
            right = left + slice_width
            lower = upper + slice_height

            image_slice = image.crop((left, upper, right, lower))
            segmentation_slice = segmentation.crop((left, upper, right, lower))

            # Construct the save path.
            output_image_path = os.path.join(output_images_folder, f'{image_file.split(".")[0]}_{i}{j}.jpg')
            output_segmentation_path = os.path.join(output_segmentation_folder, f'{image_file.split(".")[0]}_{i}{j}.png')

            # Save the slices.
            image_slice.save(output_image_path)
            segmentation_slice.save(output_segmentation_path)

            print(f'Saved: {output_image_path} and {output_segmentation_path}')

print('All images have been sliced and saved.')
