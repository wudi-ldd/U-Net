import os
import random

#-------------------------------------------------------#
# Pointing to the folder where the VOC dataset is located.
# By default, it points to the VOC dataset in the root directory.
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'

def generate_txt_files(fold, total_segs, segfilepath, saveBasePath):
    num = len(total_segs)
    indices = list(range(num))
    random.shuffle(indices)

    # Allocating the approximate quantity for each fold; the last fold may have a slight difference.
    fold_size = num // fold
    folds_extra = num % fold

    for i in range(fold):
        print(f"Processing fold {i + 1}/{fold}")
        start_index = i * fold_size + min(i, folds_extra)
        end_index = (i + 1) * fold_size + min(i + 1, folds_extra)
        val_indices = indices[start_index:end_index]
        train_indices = list(set(indices) - set(val_indices))

        fold_dir = os.path.join(saveBasePath, f'fold{i + 1}')
        os.makedirs(fold_dir, exist_ok=True)

        with open(os.path.join(fold_dir, 'trainval.txt'), 'w') as ftrainval, \
             open(os.path.join(fold_dir, 'train.txt'), 'w') as ftrain, \
             open(os.path.join(fold_dir, 'val.txt'), 'w') as fval:
            for idx in train_indices + val_indices:
                ftrainval.write(total_segs[idx][:-4] + '\n')
                if idx in train_indices:
                    ftrain.write(total_segs[idx][:-4] + '\n')
                elif idx in val_indices:
                    fval.write(total_segs[idx][:-4] + '\n')

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets for cross-validation.")

    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')

    total_segs = [f for f in os.listdir(segfilepath) if f.endswith('.png')]

    trainval_percent = 1
    train_percent = 0.9
    fold = int(1 / (1 - train_percent)) if trainval_percent == 1 else None

    if fold:
        generate_txt_files(fold, total_segs, segfilepath, saveBasePath)
    else:
        print("Invalid combination of trainval_percent and train_percent for cross-validation.")
