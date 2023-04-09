#!/usr/bin/python3

from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101 
from torch import nn

def deeplabv3_v1(backbone_model="mbv3", num_classes=1):
    weights = 'DEFAULT'    # Initialize model with pre-trained weights.
    if backbone_model == "mbv3":
        model = deeplabv3_mobilenet_v3_large(weights=weights)
    elif backbone_model == "r50":
        model = deeplabv3_resnet50(weights=weights)
    elif backbone_model == "r101":
        model = deeplabv3_resnet101(weights=weights)
    else:
        raise ValueError("Wrong backbone model passed. Must be one of 'mbv3', 'r50' and 'r101' ")
 
    # Update the number of output channels for the output layer.
    # This will remove the pre-trained weights for the last layer.
    model.classifier[4]         = nn.LazyConv2d(num_classes, 1)
    model.aux_classifier[4] = nn.LazyConv2d(num_classes, 1)
 
    return model
