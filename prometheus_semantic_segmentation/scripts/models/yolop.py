#!/usr/bin/python3

import torch
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def yolop():

    # load model
    model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)

    return model