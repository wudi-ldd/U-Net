import os
import cv2
import numpy as np
import json

# Set the image folder path.
image_folder = 'mask/img_out'
# Create the folder path to save JSON files.
json_folder = 'mask/json'

# Ensure that the folder to save JSON files exists.
os.makedirs(json_folder, exist_ok=True)

# Get all the image file names in the image folder.
image_files = os.listdir(image_folder)

for image_file in image_files:
    # Construct the image file paths.
    image_path = os.path.join(image_folder, image_file)

    # Read the mask image.
    mask_image = cv2.imread(image_path)

    # Convert the mask image to RGB mode.
    mask_image_rgb = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

    # Create the Labelme JSON data structure.
    labelme_data = {
        "version": "LabelMe",
        "imagePath": image_file,
        "imageData": None,
        "imageHeight": mask_image_rgb.shape[0],
        "imageWidth": mask_image_rgb.shape[1],
        "flags": {},
        "shapes": []
    }

    # Convert the mask image to a grayscale image.
    mask_image_gray = cv2.cvtColor(mask_image_rgb, cv2.COLOR_RGB2GRAY)

    # Get the unique colors in the image.
    unique_colors = np.unique(mask_image_rgb.reshape(-1, mask_image_rgb.shape[2]), axis=0)

    # Define a dictionary mapping colors to class names.
    color_to_label = {
        (128, 0, 0): "Quartz",
        (0, 128, 0): "Feldspar",
        (128, 128, 0): "Rock fragment",
        (0, 0, 128): "Cement",
        (128, 0, 128): "Primary pore",
        (0, 128, 128): "Secondary pore"
    }

    # Create a dictionary to count group_id for each class.
    group_id_dict = {
        "Quartz": 0,
        "Feldspar": 0,
        "Rock fragment": 0,
        "Cement": 0,
        "Primary pore": 0,
        "Secondary pore": 0
    }

    # Create contours for each region in Labelme.
    for color in unique_colors:
        # Convert colors to integer tuples.
        color = tuple(color.astype(int).tolist())

        # Check if the color is (0, 0, 0).
        if color == (0, 0, 0):
            continue

        # Check if this color has already been processed.
        if any(shape["label"] == color_to_label[color] for shape in labelme_data["shapes"]):
            continue

        # Create a target region with a specific color.
        mask = np.all(mask_image_rgb == color, axis=-1)

        # Extract the contours of the target region.
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Create contours for each region in Labelme.
        for i, contour in enumerate(contours):
            # Check the number of contour points.
            if len(contour) < 50:
                continue

            contour = contour.flatten().tolist()
            shape = {
                "label": color_to_label[color],
                "points": [],
                "group_id": group_id_dict[color_to_label[color]] + 1,
                "description": "",
                "shape_type": "polygon",
                "flags": {}
            }

            # Convert contour coordinates to [x, y] format.
            for j in range(0, len(contour), 2):
                shape["points"].append([contour[j], contour[j + 1]])

            # Update the group_id count dictionary.
            group_id_dict[color_to_label[color]] += 1

            labelme_data["shapes"].append(shape)

    # Construct the path to save the JSON file.
    json_file_path = os.path.join(json_folder, os.path.splitext(image_file)[0] + '.json')

    # Save the Labelme data as a JSON file and format it.
    with open(json_file_path, 'w') as json_file:
        json.dump(labelme_data, json_file, indent=4)
    print("Saved to {}".format(json_file_path))












