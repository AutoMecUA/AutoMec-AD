#!/usr/bin/python3
from torch import nn

# Definition of the model. For now a 1 neuron network

class Rota_Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # bx3x224x224 input images
        self.layer1 = nn.Sequential(
            # 3 input channels, 16 output depth, padding and stride
            nn.Conv2d(3,8,kernel_size=5,stride=2),
            # normalizes the batch data setting the average to 0 and std to 1
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(2) # similar to image pyrdown, reduces size
        )

        
        self.layer2 = nn.Sequential(
            nn.Conv2d(8,16, kernel_size=5 , stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=5, stride=2),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=5),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            #nn.MaxPool2d(2)
        )
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(27456,375)
        self.fc2 = nn.Linear(375,125)
        self.fc3= nn.Linear(125,25)
        self.fc4= nn.Linear(25,1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
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

        out = self.flatten(out)
        # print('layer 3 out = ' + str(out.shape))

        #out = out.view(out.size(0),-1) # flatten to keep batch dimension and compact all others into the second dimension

        out = self.relu(self.fc1(out))
        # print('fc1 out = ' + str(out.shape))

        out = self.relu(self.fc2(out))

        out = self.relu(self.fc3(out))

        out = self.tanh(self.fc4(out))
        # print('fc2 out = ' + str(out.shape))
        # exit(0)
        return out
