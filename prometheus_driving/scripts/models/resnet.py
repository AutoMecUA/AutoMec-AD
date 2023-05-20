import torch.nn as nn
import torch.nn.functional as F

class IdentityBlock(nn.Module):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__()
        self.conv = nn.Conv2d(filters , filters, kernel_size, padding=1)
        self.norm = nn.BatchNorm2d(filters)
        self.add = nn.Identity()

    def forward(self, input):
        x = self.conv(input)
        x = self.norm(x)
        x = F.relu(x)

        x = self.conv(x)
        x = self.norm(x)

        x = x + input # This is where the skip connection is made
        x = F.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv7 = nn.Conv2d(3,64, 7, padding=1)
        self.norm = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(3,3))
        self.idbl1 = IdentityBlock(64, 3)
        self.idbl2 = IdentityBlock(64, 3)
        self.gpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, 1)

    def forward(self, input):
        x = self.conv7(input)
        x = self.norm(x)
        x = self.pool(x)
        x = self.idbl1(x)
        x = self.idbl2(x)
        x = self.gpool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x



