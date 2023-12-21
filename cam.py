import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from nets.unet import Unet
import torch
import torch.functional as F
import numpy as np
# import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

image_url = "VOCdevkit\VOC2007\JPEGImages/S1-D12m2-.jpg"
desired_size = (1024, 768)
# Open and convert the image to RGB
image = Image.open(image_url).convert('RGB')
img_resized = image.resize(desired_size)
# Convert the resized image to numpy array
rgb_img = np.float32(np.array(img_resized)) / 255

# Convert to tensor.
image_data  = np.expand_dims(np.transpose(rgb_img, (2, 0, 1)), 0)
input_tensor = torch.from_numpy(image_data)
if torch.cuda.is_available():
    input_tensor = input_tensor.cuda()

model = Unet(num_classes=2, pretrained=False, backbone='resnet50')
weight_path = "logs/best_epoch_weights.pth"
model.load_state_dict(torch.load(weight_path))
model = model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

output = model(input_tensor)
print(type(output))
normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
sem_classes = ["_background_","pore"]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

car_category = sem_class_to_idx["pore"]
car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
car_mask_float = np.float32(car_mask == car_category)

both_images = np.hstack((img_resized, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
image_to_show = Image.fromarray(both_images.astype('uint8'))
image_to_show.show()
# Save car_mask_uint8.
car_mask_image=Image.fromarray(car_mask_uint8)
car_mask_image.save('cam/car_mask_uint8.jpg')


from pytorch_grad_cam import GradCAM

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

    
target_layers = [model.final]
targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

image_to_show = Image.fromarray(cam_image.astype('uint8'))
image_to_show.show()
# Save Heatmap
image_to_show.save('cam/cam.jpg')
    

