import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3, self).__init__()
        # Expects input to be 3 x 299 x 299
        pre_trained_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT) # Shouldn't it be pretrained?

        # Freeze Layers if set to False
        for param in pre_trained_model.parameters():
            param.requires_grad = False

        self.features = nn.Sequential(*list(pre_trained_model.children())[:4])

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(341056, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        self.pre_trained_model = pre_trained_model


    def forward(self, x):
        x = self.features(x) 
        x = self.fc(x)
        return x
