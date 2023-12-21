#--------------------------------------------------------#
# This file is used to adjust the format of labels.
#--------------------------------------------------------#
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

#-----------------------------------------------------------------------------------#
# Origin_SegmentationClass_path: The path where the original labels are located.
# Out_SegmentationClass_path: The path where the output labels will be located.
# The processed labels are grayscale images. If the specified value is too small, details may not be visible.
#-----------------------------------------------------------------------------------#
Origin_SegmentationClass_path   = "datasets/SegmentationClass"
Out_SegmentationClass_path      = "datasets/SegmentationClass"

#-----------------------------------------------------------------------------------#
# Origin_Point_Value: The pixel values in the original labels.
# Out_Point_Value: The corresponding pixel values in the output labels.
# Origin_Point_Value and Out_Point_Value should have a one-to-one correspondence.

# For example, if:
# Origin_Point_Value = np.array([0, 255])
# Out_Point_Value = np.array([0, 1])

# It means that the pixels with a value of 0 in the original labels will be adjusted to 0, and the pixels with a value of 255 in the original labels will be adjusted to 1.

# You can adjust more than two pixel values, for example:
# Origin_Point_Value = np.array([0, 128, 255])
# Out_Point_Value = np.array([0, 1, 2])

# It can also be an array (when label values are RGB pixels), like:
# Origin_Point_Value = np.array([[0, 0, 0], [1, 1, 1]])
# Out_Point_Value = np.array([0, 1])
#-----------------------------------------------------------------------------------#
Origin_Point_Value              = np.array([0, 1])
Out_Point_Value                 = np.array([1, 0])

if __name__ == "__main__":
    if not os.path.exists(Out_SegmentationClass_path):
        os.makedirs(Out_SegmentationClass_path)

    #---------------------------#
    # Iterate through the labels and assign values.
    #---------------------------#
    png_names = os.listdir(Origin_SegmentationClass_path)
    print("Iterating through all labels.")
    for png_name in tqdm(png_names):
        png     = Image.open(os.path.join(Origin_SegmentationClass_path, png_name))
        w, h    = png.size
        
        png     = np.array(png)
        out_png = np.zeros([h, w])
        for i in range(len(Origin_Point_Value)):
            mask = png[:, :] == Origin_Point_Value[i]
            if len(np.shape(mask)) > 2:
                mask = mask.all(-1)
            out_png[mask] = Out_Point_Value[i]
        
        out_png = Image.fromarray(np.array(out_png, np.uint8))
        out_png.save(os.path.join(Out_SegmentationClass_path, png_name))

    #-------------------------------------#
    # Count the occurrences of each pixel value in the output labels.
    #-------------------------------------#
    print("Counting the number of pixels for each value in the output image.")
    classes_nums        = np.zeros([256], np.int)
    for png_name in tqdm(png_names):
        png_file_name   = os.path.join(Out_SegmentationClass_path, png_name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label image %s not detected. Please check if the file exists in the specified path and if the file extension is .png."%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        classes_nums    += np.bincount(np.reshape(png, [-1]), minlength=256)
        
    print("Print the pixel values and their respective counts.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)