import os
from PIL import Image

def convert_image_bitdepth(input_folder, output_folder):
    # Get a list of all files in the input folder.
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # Construct the full paths for the input files.
        input_path = os.path.join(input_folder, file_name)

        # Open the image.
        image = Image.open(input_path)

        # Convert the image's bit depth to 8 bits.
        converted_image = image.convert("P", dither=None, palette=Image.ADAPTIVE)

        # Construct the full paths for the output files.
        output_path = os.path.join(output_folder, file_name)

        # Save the converted image.
        converted_image.save(output_path)

# Specify the paths for the input folder and the output folder.
input_folder = "datasets/mix"
output_folder = "datasets\SegmentationClass_Origin"

# Create the output folder (if it doesn't exist).
os.makedirs(output_folder, exist_ok=True)

# Call the function to perform the conversion.
convert_image_bitdepth(input_folder, output_folder)



