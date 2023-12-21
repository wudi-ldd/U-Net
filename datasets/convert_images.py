import cv2
import os

def convert_images(folder_path, output_folder):
    # Create the output folder.
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through the image files in the folder.
    for file_name in os.listdir(folder_path):
        if file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # Construct the full path for the input image.
            input_path = os.path.join(folder_path, file_name)

            # Generate the output file name.
            output_file_name = file_name.rsplit('.', maxsplit=1)[0] + '.jpg'

            # Construct the full path for the output image.
            output_path = os.path.join(output_folder, output_file_name)

            # Open the image file.
            image = cv2.imread(input_path)

            # Save the image as a JPG format.
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])  # Set the quality parameter.

            print(f"Converted {input_path} to {output_path}")

# Set the input folder path and the output folder path.
input_folder = "datasets\JPEGImages_Origin"
output_folder = "datasets\JPEGImages"
# Call the function to perform image conversion.
convert_images(input_folder, output_folder)




