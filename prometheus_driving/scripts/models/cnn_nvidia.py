#!/usr/bin/python3

from torch import nn

# Definition of the model. For now a 1 neuron network

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # bx3x224x224 input images
        self.layer1 = nn.Sequential(
            # 3 input channels, 16 output depth, padding and stride
            nn.Conv2d(3,24,kernel_size=5,stride=2),
            # normalizes the batch data setting the average to 0 and std to 1
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2) # similar to image pyrdown, reduces size
        )

        
        self.layer2 = nn.Sequential(
            nn.Conv2d(24,36, kernel_size=5 , stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(36,48, kernel_size=5, stride=2),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(48,64, kernel_size=5),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=5),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(16704,100)
        self.fc2 = nn.Linear(100,50)
        self.fc3= nn.Linear(50,10)
        self.fc4= nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
        
    def forward(self,x):
        # print('input x = ' + str(x.shape))
        out = self.layer1(x)
        # print('layer 1 out = ' + str(out.shape))

        out = self.layer2(out)
        # print('layer 2 out = ' + str(out.shape))

        out = self.layer3(out)
        # print('layer 3 out = ' + str(out.shape))

        out = self.layer4(out)
        # print('layer 3 out = ' + str(out.shape))

        out = self.layer5(out)
        # print('layer 3 out = ' + str(out.shape))

        out = out.view(out.size(0),-1) # flatten to keep batch dimension and compact all others into the second dimension

        out = self.relu(self.fc1(out))
        # print('fc1 out = ' + str(out.shape))

        out = self.elu(self.fc2(out))

        out = self.elu(self.fc3(out))

        out = self.fc4(out)
        # print('fc2 out = ' + str(out.shape))
        # exit(0)
        return out

