#!/usr/bin/python3
from torch import nn , zeros , cuda , device
from torch.autograd import Variable 
from collections import OrderedDict
# Definition of the model. For now a 1 neuron network


class CNN(nn.Module):
    def __init__(self):
        super().__init__()        
        # bx3x224x224 input images
        self.layer1 = nn.Sequential(
            # 3 input channels, 16 output depth, padding and stride
            nn.Conv2d(3,24,kernel_size=5, stride=2),
            # normalizes the batch data setting the average to 0 and std to 1
            nn.BatchNorm2d(24),
            nn.ELU(),
            nn.MaxPool2d(3, stride=2) # similar to image pyrdown, reduces size
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
            #nn.MaxPool2d(2),
            nn.AvgPool2d(3, stride=2),
            nn.Flatten()
        )

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
        #self.fc1 = nn.Linear(27456,1152)
        #self.fc2 = nn.Linear(1152,512)

        self.dropout = nn.Dropout(0.20)
        
        
    def forward(self,x ):
        out = self.layer1(x)
        # print('layer 1 out = ' + str(out.shape))

        out = self.layer2(out)
        # print('layer 2 out = ' + str(out.shape))

        out = self.layer3(out)
        # print('layer 3 out = ' + str(out.shape))

        out = self.layer4(out)
        # print('layer 3 out = ' + str(out.shape))

        out = self.layer5(out)

        return out

class LSTM(nn.Module):
    def __init__(self,hidden_dim=1024 , num_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = f'cuda:0' if cuda.is_available() else 'cpu' # cuda: 0 index of gpu

        feature_extraction = CNN()

        d = OrderedDict(feature_extraction.named_children())
        self.feature_extraction = nn.Sequential(d)

        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim,
                          num_layers=self.num_layers, batch_first=True) #lstm

        self.linear = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(0.20)
        
        
    def forward(self,x):
        s = x.size()
        #print(s)
        h0 = zeros(self.num_layers, x.size(0), self.hidden_dim).to(device(self.device)) # we have to have a hidden value for each unit of all rnn-layers.
        c0 = zeros(self.num_layers, x.size(0), self.hidden_dim).to(device(self.device))
        
        #x = x.view(-1, *s[2:]) # x: (B*T)x(C)x(H)x(W))
        #print(x.size()) # (B*T), C, H, W)
        x = self.feature_extraction(x).squeeze()
        #print(x.size()) # (B*T), d)  
        x = x.view(s[0], s[1], -1)  # x: BxTxd
        #print(x.size())
        
        lstm_out , (self.fc_h, self.fc_c) = self.lstm(x , (h0,c0))
        #print('lstm_out = ' + str(lstm_out.shape))
        out = lstm_out[:,-1,:]

        out = self.linear(out)

        return out