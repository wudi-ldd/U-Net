import base64
import json
import os
import os.path as osp
import cv2
import numpy as np
import PIL.Image
from labelme import utils

'''
To create your own semantic segmentation dataset, you should consider the following points:
1. Use labelme version 3.16.7 for the best results. It is recommended to use this specific version of labelme, as some other versions may encounter errors like "Too many dimensions: 3 > 2." You can install it via the command: `pip install labelme==3.16.7`.
2. The label images generated here are 8-bit color images, which may appear different from the dataset format shown in videos. Despite their appearance as color images, they are actually 8-bit images. In this format, each pixel's value represents the category to which that pixel belongs.
'''
if __name__ == '__main__':
    jpgs_path   = "datasets/JPEGImages"
    pngs_path   = "datasets/SegmentationClass"
    classes     = ["_background_","pore",'__background__',]
    count = os.listdir("./datasets/before/") 
    for i in range(0, len(count)):
        path = os.path.join("./datasets/before", count[i])

        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))
            
            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    img = cv2.imread(imagePath)
                    
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV defaults to using BGR, so convert to RGB.     
            label_name_to_value = {'_background_': 0,'__background__':0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            
            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            
            lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)
            
                
            PIL.Image.fromarray(img).save(osp.join(jpgs_path, count[i].split(".")[0]+'.jpg'))

            new = np.zeros([np.shape(img)[0],np.shape(img)[1]])
            for name in label_names:
                index_json = label_names.index(name)
                index_all = classes.index(name)
                new = new + index_all*(np.array(lbl) == index_json)

            utils.lblsave(osp.join(pngs_path, count[i].split(".")[0]+'.png'), new)
            print('Saved ' + count[i].split(".")[0] + '.jpg and ' + count[i].split(".")[0] + '.png')
