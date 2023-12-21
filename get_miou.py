import os

from PIL import Image
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

'''
To perform metric evaluation, please consider the following points:
1. The generated images in this file are grayscale, and they may appear almost entirely black when viewed as JPG images due to their small values. It is normal to see such images.
2. This file calculates the mIoU (mean Intersection over Union) for the validation set. Currently, this library treats the test set as the validation set and does not separate a dedicated test set.
3. This file is designed to calculate mIoU for models trained on data in the VOC format. It is specifically tailored for this format and may not work correctly with other datasets or formats.
'''
if __name__ == "__main__":
    #---------------------------------------------------------------------------#
    # miou_mode is used to specify what this file should calculate during runtime.
    # miou_mode 0 means the entire mIoU calculation process, including obtaining predictions and calculating mIoU.
    # miou_mode 1 means only obtaining predictions.
    # miou_mode 2 means only calculating mIoU.
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    # Number of classes + 1, for example, 2 + 1.
    #------------------------------#
    num_classes     = 2
    #--------------------------------------------#
    # Categories to be distinguished, the same as in json_to_dataset.
    #--------------------------------------------#
    name_classes    =["_background_","pore"]
    #-------------------------------------------------------#
    # Point to the folder where the VOC dataset is located.
    # By default, it points to the VOC dataset in the root directory.
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),'r').read().splitlines() 
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)
            
        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            image       = unet.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
            print("Get miou.")
            hist, IoUs, PA_Recall, Precision, Dice, FPR = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  
            print("Get miou done.")

            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, Dice, FPR, name_classes)
