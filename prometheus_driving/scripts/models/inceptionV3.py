import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class InceptionV3(nn.Module):
    def __init__(self,hidden_size = 128, pretrained = True, aux_logits=False):
        super(InceptionV3, self).__init__()
        # Expects input to be 3 x 299 x 299
        self.hidden_size = hidden_size
        self.aux_logits = aux_logits

        if pretrained:
            base_model = models.inception_v3(weights='Inception_V3_Weights.DEFAULT')
        else:
            base_model = models.inception_v3()
        base_model.aux_logits = False

        self.Conv2d_1a_3x3 = base_model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = base_model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = base_model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = base_model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = base_model.Conv2d_4a_3x3
        self.Mixed_5b = base_model.Mixed_5b
        self.Mixed_5c = base_model.Mixed_5c
        self.Mixed_5d = base_model.Mixed_5d
        self.Mixed_6a = base_model.Mixed_6a
        self.Mixed_6b = base_model.Mixed_6b
        self.Mixed_6c = base_model.Mixed_6c
        self.Mixed_6d = base_model.Mixed_6d
        self.Mixed_6e = base_model.Mixed_6e
        self.Mixed_7a = base_model.Mixed_7a
        self.Mixed_7b = base_model.Mixed_7b
        self.Mixed_7c = base_model.Mixed_7c
        self.avgpool = base_model.avgpool
        self.dropout = base_model.dropout

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )


        # This network needs to be revised since it doesn't have the auxiliary network

    def forward(self, x):
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x) # mixed is the inception module!!
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288
    
        # if self.aux_logits and self.training:
        #     pose_aux1 = self.aux1(x)
        
        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768
        
        
        # if self.aux_logits and self.training:
        #     pose_aux2 = self.aux2(x)
        
        
        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)       
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.dropout(x)
        # N x 2048
        x = self.fc(x)
        return x