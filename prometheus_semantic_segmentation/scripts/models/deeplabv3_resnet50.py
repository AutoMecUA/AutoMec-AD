#!/usr/bin/python3

""" DeepLabv3 Model download and change the head for your prediction"""
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
from torch import nn


def createDeepLabv3_resnet50():
    """DeepLabv3 class with custom head
    Args:
        outputchannels (int, optional): The number of output channels
        in your dataset masks. Defaults to 1.
    Returns:
        model: Returns the DeepLabv3 model with the ResNet101 backbone.
    """
    Net = models.segmentation.deeplabv3_resnet50(weights= models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT  ,progress=True,num_classes=21)
    Net.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes

    for param in Net.parameters():
        param.requires_grad = True
    # Set the model in training mode
    Net.train()
    return Net