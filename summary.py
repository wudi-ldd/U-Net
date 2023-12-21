#--------------------------------------------#
# This part of the code is used to view the network architecture.
#--------------------------------------------#
import torch
from thop import clever_format, profile
from torchsummary import summary

from nets.unet import Unet

if __name__ == "__main__":
    input_shape     = [512, 512]
    num_classes     = 21
    backbone        = 'vgg'
    
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Unet(num_classes = num_classes, backbone = backbone).to(device)
    summary(model, (3, input_shape[0], input_shape[1]))
    
    dummy_input     = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
    flops, params   = profile(model.to(device), (dummy_input, ), verbose=False)
    #--------------------------------------------------------#
    # `flops * 2` is because the profile doesn't count convolution as two operations.
    # Some papers count convolution as both multiplication and addition (two operations). In this case, multiply by 2.
    # Some papers only consider the number of multiplications and ignore additions. In this case, don't multiply by 2.
    # This code chooses to multiply by 2, following the approach used in YOLOX.
    #--------------------------------------------------------#
    flops           = flops * 2
    flops, params   = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))
