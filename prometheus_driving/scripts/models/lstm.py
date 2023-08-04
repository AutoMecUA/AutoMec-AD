#!/usr/bin/python3
import torch
from torch import nn , zeros , cuda , device
from torch.autograd import Variable 
from collections import OrderedDict

class CNN(nn.Module):
    def __init__(self):
        super().__init__()        
        # bx3x224x224 input images
        self.layer1 = nn.Sequential(
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
            nn.Conv2d(64,128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ELU(),
            #nn.MaxPool2d(2),
            nn.AvgPool2d(3, stride=2),
            nn.Flatten()
        )

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
    def __init__(self,hidden_dim=1024 , num_layers=4 , dropout=0.20):
        super().__init__()
        # TODO Calculate the hidden dim based on the input image size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.first_time = True
        self.device = f'cuda:0' if cuda.is_available() else 'cpu' # cuda: 0 index of gpu

        self.feature_extraction = CNN()
        
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim,
                          num_layers=self.num_layers, batch_first=True) #lstm
        
        self.linear = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, batch_size=70):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device(self.device)).detach(),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device(self.device)).detach())
        
        return hidden


        
    def forward(self,x,vel):
        s = x.size()
        #print(s)
        if self.first_time:
            self.hidden = self.init_hidden(s[0])
            self.first_time = False

        
        #x = x.view(-1, *s[2:]) # x: (B*T)x(C)x(H)x(W))
        # print(f'first is {x.size()}') # (B*T), C, H, W)
        x = self.feature_extraction(x) # x: (B*T), d)
        # print(x.size()) # (B*T), d)  
        x = x.view(x.size(0) , 1 , -1)  # x: BxTxd
        # print(x.size()) 
        
        lstm_out , self.hidden = self.lstm(x , self.hidden)
        #print('lstm_out = ' + str(lstm_out.shape))
        out = lstm_out[:,-1,:]
        self.hidden = (self.hidden[0].detach(), self.hidden[1].detach())

        out = self.dropout(out)

        out = self.linear(out)

        return out