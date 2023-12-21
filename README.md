# Model Training Guide

This guide provides detailed steps for training semantic segmentation models using a custom dataset. Below is an overview of the entire training process:

## Table of Contents
- [Preparing the Dataset](#preparing-the-dataset)
- [Training the Model](#training-the-model)
- [Making Predictions](#making-predictions)
- [Model Evaluation](#model-evaluation)
- [Additional Tools](#additional-tools)

## Preparing the Dataset
1. **Original Dataset Preparation**: Place the original images and corresponding annotation JSON files into the `datasets\before` folder.
2. **Generating Segmentation Masks**: Run `json_to_dataset.py` and modify the categories in the file to your annotated categories. This script will generate JPG format original images and PNG format segmentation masks, saved in `datasets\JPEGImages` and `datasets\SegmentationClass` folders respectively.
3. **Dataset Division**: Put `datasets\JPEGImages` and `datasets\SegmentationClass` into the `VOCdevkit\VOC2007` folder. Run `voc_annotation.py` to divide the dataset into a training set and a validation set in a 9:1 ratio. The divided TXT files are saved in the `VOCdevkit\VOC2007\ImageSets\SegmentationClass` folder. For smaller datasets, you can choose to run `voc_annotation k-folders.py` to divide the training and validation sets using k-fold cross-validation.

## Training the Model
1. **Setting Parameters**: Modify the parameters in `train.py`.
   - `num_classes`: Set to the number of your dataset categories + 1.
   - `backbone`: Choose `resnet50` or `vgg16` as the backbone network.
   - `pretrained`: Set to `True` to use pre-trained weights for accelerated training.
   - `model_path`: If using pre-trained weights, set to `""`; if you prefer self-training on existing weights, specify the weight path.
   - `input_shape`: Adjust the training image size, it's recommended to resize beforehand to speed up training.
2. **Starting Training**: After modifying the parameters, run `train.py` to start training.

## Making Predictions
1. **Setting Prediction Parameters**: Modify the parameters in `predict.py`.
   - `mode`: Choose between `predict` and `dir_predict`.
   - `name_classes`: Modify according to your categories.
2. **Configuring UNet Parameters**: Modify the parameters in the `unet.py` file.
3. **Start Prediction**: Run `predict.py` for predictions.

## Model Evaluation
1. **Test Set Preparation**: Ensure that the names of the test images are included in `VOCdevkit\VOC2007\ImageSets\Segmentation\test.txt`.
2. **Performing Evaluation**: Modify the parameters in `get_miou.py` and run it directly to calculate evaluation metrics. The results will be saved in the `miou_out` folder.

## Additional Tools
- `cam.py`: Generates heatmaps of predicted images.
- `extract_matching_images.py`: Extracts specified images from a target folder to another, particularly useful for semi-supervised training or evaluating uncertainty in images.
- `copy_random_images.py`: For random sample selection, useful for balancing and diversifying datasets.
- `datasets\8.py`: Converts the bit depth of mask images to 8 bits, typically used for format standardization.
- `datasets\2.py`: Checks mask pixel values to ensure annotation accuracy.
- `datasets\Convert_SegmentationClass.py`: Converts background pixels to a specific value to meet model input requirements.
- `datasets\convert_images.py`: Converts image formats, such as from PNG to JPG.
- `mask2json.py`: Converts predicted mask images into annotated JSON files for subsequent correction or analysis.
- `patch_png_json.py`: Cuts both images and JSON files at the same time, particularly useful for handling large images to avoid the impact of resizing.
- `patch.py`: Cuts both images and their corresponding annotation masks at the same time, useful for generating smaller image segments for training or testing.
