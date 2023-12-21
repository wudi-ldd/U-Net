import colorsys
import copy
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.unet import Unet as unet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


#--------------------------------------------#
# When using your own trained model for prediction, you need to modify two parameters: model_path and num_classes.
# If you encounter shape mismatches, make sure to double-check that you have correctly updated model_path and num_classes to match your training configuration.
#--------------------------------------------#
class Unet(object):
    _defaults = {
        #-------------------------------------------------------------------#
        # model_path should point to the weight file in the "logs" folder.
        # After training, there may be multiple weight files in the "logs" folder. You should choose the one with the lowest validation loss.
        # Note that a lower validation loss does not necessarily mean a higher mean intersection over union (mIoU). It simply indicates that the chosen weight performs well in terms of generalization on the validation set.
        #-------------------------------------------------------------------#
        "model_path"    : 'logs/best_epoch_weights.pth',
        # The number of classes you need to distinguish plus one (num_classes + 1).
        #--------------------------------#
        "num_classes"   : 2,
        #--------------------------------#
        # The backbone networks used are VGG and ResNet50.  
        #--------------------------------#
        "backbone"      : "resnet50",
        #--------------------------------#
        # The input image size.
        #--------------------------------#
        "input_shape"   : [768,1024],
        #-------------------------------------------------#
        # The `mix_type` parameter is used to control the visualization of detection results.

        # When `mix_type` is set to 0, it means a mix of the original image and the generated image.
        # When `mix_type` is set to 1, it means only the generated image is retained.
        # When `mix_type` is set to 2, it means only the background is removed, retaining only the objects from the original image.
        #-------------------------------------------------#
        "mix_type"      : 1,
        #--------------------------------#
        # Whether to use CUDA.
        # Set to False if you don't have a GPU.
        #--------------------------------#
        "cuda"          : True,
    }

    #---------------------------------------------------#
    # Initializing UNET.
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        # Setting different colors for drawing bounding boxes.
        #---------------------------------------------------#
        if self.num_classes <= 21:
            # self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
            #                 (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
            #                 (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
            #                 (128, 64, 12)]
            self.colors = [ (0, 0, 0), (255, 255, 255), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        # Obtaining the model.
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    # Obtaining all the classes/categories.
    #---------------------------------------------------#
    def generate(self, onnx=False):
        self.net = unet(num_classes = self.num_classes, backbone=self.backbone)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    # Detecting an image.
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        # Here, the image is converted to an RGB image to prevent errors during prediction.
        # The code only supports RGB image prediction, so all other types of images will be converted to RGB.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        # Making a backup of the input image for later use in drawing.
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        # Adding gray bars to the image to achieve non-distorted resizing.
        # Alternatively, you can directly resize the image for recognition.
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        # Adding the batch_size dimension to the input.
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            # Passing the image into the network for prediction.
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            # Extracting the class for each pixel.
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #---------------------------------------------------#
            # Calculating uncertainty - using entropy as a measure of uncertainty.
            #---------------------------------------------------#
            uncertainty = -np.sum(pr* np.log(pr + 1e-6), axis=-1)
            uncertainty = np.mean(uncertainty)
            #--------------------------------------#
            # Cropping the gray bars from the image.
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            # Resizing the image.
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            # Extracting the class for each pixel.
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        # Counting.
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            # Converting the new image to the Image format.
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            # Mixing the new image with the original image.
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            # Converting the new image to the Image format.
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            # Converting the new image to the Image format.
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image,uncertainty

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        # Here, the image is converted to an RGB image to prevent errors during prediction.
        # The code only supports RGB image prediction, so all other types of images will be converted to RGB.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        # Adding gray bars to the image to achieve non-distorted resizing.
        # Alternatively, you can directly resize the image for recognition.
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        # Adding the batch_size dimension to the input.
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            # Passing the image into the network for prediction.
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            # Extracting the class for each pixel.
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            # Cropping the gray bars from the image.
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------#
                # Passing the image into the network for prediction.
                #---------------------------------------------------#
                pr = self.net(images)[0]
                #---------------------------------------------------#
                # Extracting the class for each pixel.
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                # Cropping the gray bars from the image.
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_miou_png(self, image):
        #---------------------------------------------------------#
        # Here, the image is converted to an RGB image to prevent errors during prediction.
        # The code only supports RGB image prediction, so all other types of images will be converted to RGB.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        # Adding gray bars to the image to achieve non-distorted resizing.
        # Alternatively, you can directly resize the image for recognition.
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        # Adding the batch_size dimension to the input.
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            # Passing the image into the network for prediction.        
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            # Extracting the class for each pixel.
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            # Cropping the gray bars from the image.
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            # Resizing the image.
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            # Extracting the class for each pixel.
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image

class Unet_ONNX(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        # `onnx_path` should point to the ONNX weight file in the "model_data" folder.
        #-------------------------------------------------------------------#
        "onnx_path"    : 'model_data/models.onnx',
        #--------------------------------#
        # The number of classes you need to distinguish plus one (num_classes + 1).
        #--------------------------------#
        "num_classes"   : 21,
        #--------------------------------#
        # The backbone networks used are VGG and ResNet50.  
        #--------------------------------#
        "backbone"      : "vgg",
        #--------------------------------#
        # The input image size.
        #--------------------------------#
        "input_shape"   : [512, 512],
        #-------------------------------------------------#
        # The `mix_type` parameter is used to control the visualization of detection results.

        # When `mix_type` is set to 0, it means a mix of the original image and the generated image.
        # When `mix_type` is set to 1, it means only the generated image is retained.
        # When `mix_type` is set to 2, it means only the background is removed, retaining only the objects from the original image.
        #-------------------------------------------------#
        "mix_type"      : 0,
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    # Initializing YOLO.
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        import onnxruntime
        self.onnx_session   = onnxruntime.InferenceSession(self.onnx_path)
        # Getting all input nodes.
        self.input_name     = self.get_input_name()
        # Getting all output nodes.
        self.output_name    = self.get_output_name()

        #---------------------------------------------------#
        # Setting different colors for drawing bounding boxes.
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        show_config(**self._defaults)

    def get_input_name(self):
        # Getting all input nodes.
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_output_name(self):
        # Getting all output nodes.
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_feed(self,image_tensor):
        # Obtaining the input tensor using `input_name`.
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor
        return input_feed
    
    #---------------------------------------------------#
    # Resizing the input image.
    #---------------------------------------------------#
    def resize_image(self, image, size):
        iw, ih  = image.size
        w, h    = size

        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))

        return new_image, nw, nh

    #---------------------------------------------------#
    # Detecting an image.
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        #---------------------------------------------------------#
        # Here, the image is converted to an RGB image to prevent errors during prediction.
        # The code only supports RGB image prediction, so all other types of images will be converted to RGB.
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------#
        # Making a backup of the input image for later use in drawing.
        #---------------------------------------------------#
        old_img     = copy.deepcopy(image)
        orininal_h  = np.array(image).shape[0]
        orininal_w  = np.array(image).shape[1]
        #---------------------------------------------------------#
        # Adding gray bars to the image to achieve non-distorted resizing.
        # Alternatively, you can directly resize the image for recognition.
        #---------------------------------------------------------#
        image_data, nw, nh  = resize_image(image, (self.input_shape[1],self.input_shape[0]))
        #---------------------------------------------------------#
        # Adding the batch_size dimension to the input.
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        input_feed  = self.get_input_feed(image_data)
        pr          = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)[0][0]

        def softmax(x, axis):
            x -= np.max(x, axis=axis, keepdims=True)
            f_x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
            return f_x
        print(np.shape(pr))
        #---------------------------------------------------#
        # Extracting the class for each pixel.
        #---------------------------------------------------#
        pr = softmax(np.transpose(pr, (1, 2, 0)), -1)
        #--------------------------------------#
        # Cropping the gray bars from the image.
        #--------------------------------------#
        pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        #---------------------------------------------------#
        # Resizing the image.
        #---------------------------------------------------#
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        #---------------------------------------------------#
        # Extracting the class for each pixel.
        #---------------------------------------------------#
        pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        # Counting.
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)

        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            # Converting the new image to the Image format.
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            # Mixing the new image with the original image.
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            # Converting the new image to the Image format.
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            # Converting the new image to the Image format.
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image
