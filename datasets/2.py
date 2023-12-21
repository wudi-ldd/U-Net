import cv2
import numpy as np

# Reading the mask image.
mask_image = cv2.imread('miou_out/1\X2-G12m1-.png', cv2.IMREAD_GRAYSCALE)  # 假设掩码图像为灰度图像

if mask_image is not None:
    # Checking if the background pixel value is 0, and if the target pixel value is 1 or 255.
    background_pixels = (mask_image == 0)
    target_pixels = (mask_image == 1)

    # If you want to count the number of background pixels.
    background_pixel_count = background_pixels.sum()
    
    # If you want to count the number of target pixels.
    target_pixel_count = target_pixels.sum()

    print(f"Whether the background pixel value is 0：{background_pixel_count > 0}")
    print(f"Whether the target pixel value is 1 or 255：{target_pixel_count > 0}")
else:
    print("Unable to read the mask image.")
# Printing the number of pixels with a value of 0 and the number of pixels with a value of 1.
print(np.sum(mask_image == 0))
print(np.sum(mask_image == 1))
print(np.unique(mask_image))