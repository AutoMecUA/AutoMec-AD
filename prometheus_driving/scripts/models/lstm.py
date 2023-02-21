#!/usr/bin/python3
from torch import nn , zeros
from torch.autograd import Variable 
# Definition of the model. For now a 1 neuron network

class LSTM(nn.Module):
    def __init__(self,hidden_dim=100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        
        # bx3x224x224 input images
        self.layer1 = nn.Sequential(
            # 3 input channels, 16 output depth, padding and stride
            nn.Conv2d(3,24,kernel_size=5, stride=2),
            # normalizes the batch data setting the average to 0 and std to 1
            nn.BatchNorm2d(24),
            nn.ELU(),
            #nn.MaxPool2d(2) # similar to image pyrdown, reduces size
        )

        
        self.layer2 = nn.Sequential(
            nn.Conv2d(24,36, kernel_size=5, stride=2 ),
            nn.BatchNorm2d(36),
            nn.ELU(),
            #nn.MaxPool2d(2)
            )
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(36,48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),
            #nn.MaxPool2d(2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(48,64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            #nn.MaxPool2d(2)
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64,64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ELU(),
            #nn.MaxPool2d(2)
            nn.Flatten()
        )

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
        self.fc1 = nn.Linear(27456,1152)
        self.fc2 = nn.Linear(1152,512)

        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim,
                          num_layers=self.num_layers, batch_first=True) #lstm

        self.linear = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.20)
        
        
    def forward(self,x , hs=None):
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

        out = self.fc1(out)

        out = self.elu(out)

        out = self.dropout(out)

        out = self.fc2(out)

        out = self.elu(out)

        out = self.dropout(out)

        lstm_out, hs = self.lstm(out, hs)
        x = lstm_out.reshape(-1 , self.hidden_dim)

        out = self.linear(x)

        return out , hs