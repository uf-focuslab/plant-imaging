import torch.nn as nn
import torch

class CNN(nn.Module):
    # input_dim = images*4 (channels)
    def __init__(self, input_dim, kernel_size, class_num=None):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        #self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.c1 = nn.Conv3d(
                in_channels = 4,
                out_channels = 128,
                kernel_size = self.kernel_size)



        self.c2 = nn.Conv3d(
                128, 
                256,
                kernel_size = (1,4,4))#, padding = self.padding

        self.c3 = nn.Conv3d(
                256, 
                512,
                kernel_size = (1,4,4))#, padding = self.padding

        self.gap = nn.AdaptiveAvgPool2d((1,1)) # not sure what this is
        self.linear = torch.nn.Linear(512, 1)


    def forward(self, x): 
        x = x.float()
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x
