import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # Initialize Base Model / Feature Extractor
        base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Freeze Layers if set to False
        for param in base_model.parameters():
            param.requires_grad = False

        # Define top layers
        self.global_average_layer = nn.AdaptiveAvgPool2d((1, 1)) # This will collect the features and pool them into a nice vector
        self.fc = nn.Sequential(
            nn.Linear(1280, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            #nn.ReLU() # we might have to remove this activation in the future
            # As suggested by Azevedo I had to remove the ReLU activation due to negative steering angles not being predicted
        )
        self.dropout = nn.Dropout(0.2)
        self.base_model = base_model

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.global_average_layer(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # Implement Dropout
        x = self.fc(x)
        return x