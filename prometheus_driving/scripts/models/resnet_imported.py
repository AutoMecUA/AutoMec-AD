import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict

class ResNetV1(nn.Module):
    def __init__(self):
        super(ResNetV1, self).__init__()

        feature_extractor = models.resnet50(wheights=models.ResNet50_Weights.DEFAULT)
        d = OrderedDict(feature_extractor.named_children())
        _, fc = d.popitem(last=True)
        fe_out_planes = fc.in_features  # this is the input dimention of the fc layer
        self.feature_extraction = nn.Sequential(d)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(fe_out_planes, 1024)
        self.fc2 = nn.Linear(1024, 1)


    def forward(self, input):
        x = self.feature_extraction(input)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)


        return x



