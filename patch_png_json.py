# If the images in the dataset are too large, you can use this script to divide the images into smaller pieces for training, which can improve the utilization rate of the GPU memory.
# If you have your own annotation json files, you can use patch_png_json.py to cut the images and labels at the same time, and then use json_to_dataset.py to convert them into the VOC format. After that, you can divide them into training and validation sets for training.
import os
import json
from PIL import Image

# Enter the folder path
input_folder = 'datasets/before'

# Output folder path
output_folder = 'datasets/after'

# Traverse all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        json_path = os.path.join(input_folder, filename.replace('.png', '.json'))

        # Open the JSON file and read the annotation information
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

        # Open the image file
        image = Image.open(image_path)
        width, height = 2560, 1920  # You need to set this to your own image size because later on you will need to calculate the coordinates in the annotation file for slicing.
        num_rows = 4
        num_cols = 4

       # Slice the image and save it.
        for row in range(num_rows):
            for col in range(num_cols):
                left = col * (width // num_cols)
                upper = row * (height // num_rows)
                right = (col + 1) * (width // num_cols)
                lower = (row + 1) * (height // num_rows)
                cropped_image = image.crop((left, upper, right, lower))
                cropped_image_filename = f"{filename.replace('.png', '')}_r{row}_c{col}.png"
                cropped_image_path = os.path.join(output_folder, cropped_image_filename)
                cropped_image.save(cropped_image_path)
                print(f"Saved{cropped_image_filename}")

                # Slice the JSON data and save it.
                cropped_json_data = json_data.copy()
                cropped_json_data['imagePath'] = cropped_image_filename
                cropped_json_data['imageHeight'] = cropped_image.height
                cropped_json_data['imageWidth'] = cropped_image.width
                cropped_json_data['shapes'] = []

                for shape in json_data['shapes']:
                    points = shape['points']
                    mask_area = (max(points, key=lambda x: x[0])[0] - min(points, key=lambda x: x[0])[0]) * \
                                (max(points, key=lambda x: x[1])[1] - min(points, key=lambda x: x[1])[1])
                    image_area = cropped_image.width * cropped_image.height
                    # Remove masks where the bounding area is too large.
                    # if mask_area <= image_area * 4 / 5:
                    new_shape = {'label': shape['label'], 'points': []}
                    for point in points:
                        new_point = [
                            max(0, min((point[0] - left), cropped_image.width - 1)),
                            max(0, min((point[1] - upper), cropped_image.height - 1))
                        ]
                        new_shape['points'].append(new_point)
                    cropped_json_data['shapes'].append(new_shape)

                cropped_json_filename = f"{filename.replace('.png', '')}_r{row}_c{col}.json"
                cropped_json_path = os.path.join(output_folder, cropped_json_filename)
                with open(cropped_json_path, 'w') as cropped_json_file:
                    json.dump(cropped_json_data, cropped_json_file, indent=2)
                print(f"Saved{cropped_json_filename}")

print("Slicing and processing complete!")






