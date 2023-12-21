#----------------------------------------------------#
# Combined single-image prediction, camera detection, and FPS testing 
# functionalities into a single Python file, allowing mode modification by specifying the mode.
#----------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image

from unet import Unet_ONNX, Unet

if __name__ == "__main__":
    #-------------------------------------------------------------------------#
    # If you want to modify the color of the corresponding class, you can do so by modifying `self.colors` in the `__init__` function.
    # ----------------------------------------------------------------------#
    # ----------------------------------------------------------------------#
    #   The 'mode' is used to specify the testing mode:
    #   'predict'           for single-image prediction. If you want to modify the prediction process, such as saving images, cropping objects, etc., you can refer to the detailed comments below.
    #   'video'             for video detection, which can use a camera or video for detection. See the comments below for details.
    #   'fps'               for testing FPS, using the 'street.jpg' image in the 'img' folder. See the comments below for details.
    #   'dir_predict'       for folder traversal detection and saving. It defaults to traversing the 'img' folder and saving to the 'img_out' folder. See the comments below for details.
    #   'export_onnx'       for exporting the model to ONNX. Requires PyTorch 1.7.1 or above.
    #   'predict_onnx'      for prediction using the exported ONNX model. Modify the relevant parameters in 'Unet_ONNX' around line 346 in 'unet.py'.
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    #-------------------------------------------------------------------------#
    #   'count'              Specifies whether to calculate the pixel count (i.e., area) and ratios of objects.
    #   'name_classes'       Categories to be distinguished, the same as in 'json_to_dataset', used for printing categories and quantities.
    #
    #   'count' and 'name_classes' are only valid when 'mode' is set to 'predict'.
    #-------------------------------------------------------------------------#
    count           = False
    name_classes    = ["_background_","pore"]
    #----------------------------------------------------------------------------------------------------------#
    #   'video_path'          Specifies the path to the video. When 'video_path=0', it means detecting from the camera.
    #                         To detect from a video, set 'video_path' to the file path, e.g., 'video_path = "xxx.mp4"' to read the 'xxx.mp4' file from the root directory.
    #   'video_save_path'     Represents the path to save the video. When 'video_save_path=""', it means not saving the video.
    #                         To save the video, set 'video_save_path' to the file path, e.g., 'video_save_path = "yyy.mp4"' to save it as 'yyy.mp4' in the root directory.
    #   'video_fps'           Represents the FPS for the saved video.
    #
    #   'video_path', 'video_save_path', and 'video_fps' are only valid when 'mode' is set to 'video'.
    #   To complete the saving process, you need to exit with Ctrl+C or run until the last frame.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   'test_interval'       Specifies the number of times image detection is performed when measuring FPS. In theory, a larger 'test_interval' yields a more accurate FPS measurement.
    #   'fps_image_path'      Specifies the image used for FPS testing.
    #
    #   'test_interval' and 'fps_image_path' are only valid when 'mode' is set to 'fps'.
    #----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path  = "img/street.jpg"
    #-------------------------------------------------------------------------#
    #   'dir_origin_path'     Specifies the folder path containing images for detection.
    #   'dir_save_path'       Specifies the saving path for images after detection.
    #
    #   'dir_origin_path' and 'dir_save_path' are only valid when 'mode' is set to 'dir_predict'.
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   'simplify'            Use Simplify on ONNX.
    #   'onnx_save_path'      Specifies the path to save the ONNX model.
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode != "predict_onnx":
        unet = Unet()
    else:
        yolo = Unet_ONNX()

    if mode == "predict":
        '''
        There are a few important points to note in the `predict.py` code:
        1. This code cannot directly perform batch predictions. If you want to perform batch predictions, you can use `os.listdir()` to traverse a folder and use `Image.open` to open image files for prediction. You can refer to the `get_miou_prediction.py` for the specific process, as it implements folder traversal.
        2. If you want to save the result, you can use `r_image.save("img.jpg")` to save it.
        3. If you don't want to mix the original image and the segmented image, you can set the `blend` parameter to `False`.
        4. If you want to obtain specific regions based on the mask, you can refer to the part of the `detect_image` function that draws the prediction result. You can check each pixel's class and then obtain the corresponding region based on the class. Here's an example:
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr == c) * (self.colors[c][2])).astype('uint8')
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image,_ = unet.detect_image(image, count=count, name_classes=name_classes)
                r_image.show()
                output_path = f"img_out/output.png"
                r_image.save(output_path)


    elif mode == "video":
        capture=cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read the camera (video) correctly. Please make sure you have installed the camera properly and have correctly specified the video path.")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a specific frame.
            ref, frame = capture.read()
            if not ref:
                break
            # Format conversion, from BGR to RGB.
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert it to an Image.
            frame = Image.fromarray(np.uint8(frame))
            # Perform detection.
            frame = np.array(unet.detect_image(frame))
            # Convert it back to BGR for displaying with OpenCV.
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = unet.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
        
    elif mode == "dir_predict":
        import os
        from tqdm import tqdm
        # Store the image name and corresponding uncertainty in a dictionary.
        uncertainty_dict = {}
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image,uncertainty     = unet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                # Store the uncertainty.
                uncertainty_dict[img_name] = uncertainty
        # Sort the uncertainties in descending order and save them to a text file.
        sorted_uncertainty = sorted(uncertainty_dict.items(), key=lambda x: x[1], reverse=True)

        # Create or overwrite a text file.
        with open(os.path.join(dir_save_path, 'uncertainty_ranking.txt'), 'w') as f:
            for img_name, uncertainty in sorted_uncertainty:
                f.write(f"{img_name}: {uncertainty}\n")
    elif mode == "export_onnx":
        unet.convert_to_onnx(simplify, onnx_save_path)
                
    elif mode == "predict_onnx":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
