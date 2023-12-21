import os
import random
import shutil

def distribute_images(src_folder, target_folders, num_images):
    # Ensure the source folder exists.
    if not os.path.exists(src_folder):
        print("Source folder does not exist.")
        return

    # Get all images.
    all_images = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    random.shuffle(all_images)  # Randomly shuffle the order of images.

    # Ensure that the target folder exists, and create it if it doesn't.
    for folder in target_folders:
        os.makedirs(folder, exist_ok=True)

    # Allocate images to the target folder.
    start = 0
    for folder, count in zip(target_folders, num_images):
        end = start + count
        for img in all_images[start:end]:
            shutil.copy(os.path.join(src_folder, img), os.path.join(folder, img))
        start = end

# Call the function.
source_folder = 'img'  # Source folder path.
target_folders = ['VOCdevkit\VOC2007/val', 'VOCdevkit\VOC2007/train1', 'VOCdevkit\VOC2007/train2', 'VOCdevkit\VOC2007/train3', 'VOCdevkit\VOC2007/train4'] # Target folder name.
num_images = [0, 0, 36, 36, 36] # Number of images allocated per folder.

distribute_images(source_folder, target_folders, num_images)
