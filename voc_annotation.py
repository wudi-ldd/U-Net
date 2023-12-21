import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
# If you want to increase the size of the test set, you can modify the `trainval_percent` parameter.
# To change the validation set ratio to 9:1, you can modify the `train_percent` parameter.

# Currently, the library treats the test set as the validation set and does not separately split a test set.
#-------------------------------------------------------#
trainval_percent    = 1
train_percent       = 0.9
#-------------------------------------------------------#
# Pointing to the folder where the VOC dataset is located.
# By default, it points to the VOC dataset in the root directory.
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007\SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("Checking whether the dataset format meets the requirements may take some time.")
    classes_nums        = np.zeros([256], np.int)
    for i in tqdm(list):
        name            = total_seg[i]
        png_file_name   = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("Label image %s was not detected. Please check whether the file exists in the specific path and whether the file extension is .png."%(png_file_name))
        
        png             = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("The shape of label image %s is %s, which does not appear to be a grayscale or eight-bit color image. Please carefully review the dataset format."%(name, str(np.shape(png))))
            print("Label images should be grayscale or eight-bit color images, where the value of each pixel in the label image represents the class of that pixel."%(name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
            
    print("Printing the values and counts of pixels.")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)
    
    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("It appears that the label contains only pixel values of 0 and 255, which indicates a data format issue.")
        print("For a binary classification problem, you should modify the labels so that background pixels have a value of 0, and target pixels have a value of 1.")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("It seems that the label only contains background pixels, which indicates a data format issue. Please review the dataset format carefully.")

    print("The images in the JPEGImages folder should be in .jpg format, while the images in the SegmentationClass folder should be in .png format. Please ensure that the file formats match these requirements.")