import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, input , filters, kernel_size, repetitions, pool_size=2, strides=2):
        super(Block, self).__init__()
        self.kernel_size = kernel_size
        self.repetitions = repetitions
        self.pool_size = pool_size
        self.strides = strides 
        self.input = input
        self.filters = filters
        
        # Define a conv2D_0, conv2D_1, etc based on the number of repetitions
        self.conv2D = nn.ModuleList([nn.Conv2d(self.filters, self.filters, self.kernel_size, 
                                              padding=1) for _ in range(self.repetitions)])

        for idx in range(self.repetitions):
            if idx == 0:
                self.conv2D[idx] = nn.Conv2d(self.input, self.filters, self.kernel_size, padding=1)
            else:
                self.conv2D[idx] = nn.Conv2d(self.filters, self.filters, self.kernel_size, padding=1)

            self.conv2D = nn.ModuleList(conv2D for conv2D in self.conv2D)
        
        # Define the max pool layer that will be added after the Conv2D blocks
        self.max_pool = nn.MaxPool2d(pool_size, strides)
        
    def forward(self, inputs):
        x = inputs
        # Connect the conv2D_0 layer to inputs
        for conv in self.conv2D:
            x = F.relu(conv(x))

        # Finally, add the max_pool layer
        x = self.max_pool(x)
      
        return x

class MyVGG(nn.Module):

    def __init__(self):
        super(MyVGG, self).__init__()

        # Creating blocks of VGG with the following 
        # (filters, kernel_size, repetitions) configurations
        self.block_a = Block(3,64,3,2)
        self.block_b = Block(64,128,3,2)
        self.block_c = Block(128,256,3,3)
        self.block_d = Block(256,512,3,3)
        self.block_e = Block(512,512,3,3)
        
        # Classification head
        # Define a Flatten layer
        self.flatten = nn.Flatten()
        
        # Create a Dense layer with 256 units and ReLU as the activation function
        self.fc = nn.Linear(25600, 256)
        self.fc_relu = nn.ReLU()
        # Finally add the softmax classifier using a Dense layer
        self.classifier = nn.Linear(256, 1)

    def forward(self, inputs):
        # Chain all the layers one after the other
        x = self.block_a(inputs)
        x = self.block_b(x)
        x = self.block_c(x)
        x = self.block_d(x)
        x = self.block_e(x)
        
        x = self.flatten(x)
        x = self.fc_relu(self.fc(x))
        x = self.classifier(x)
        
        return x