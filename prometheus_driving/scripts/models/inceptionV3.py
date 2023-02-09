import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class InceptionV3(nn.Module):
    def __init__(self):
        self.pre_trained_model = models.inception_v3(pretrained=False)

        # Freeze Layers if set to False
        for param in self.pre_trained_model.parameters():
            param.requires_grad = False

        last_layer = self.pre_trained_model._modules.get('mixed7')
        print('last layer output shape: ', last_layer.output_shape)
        last_output = last_layer.output

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.pre_trained_model.features(x) 
        x = self.fc(x)
        return x
