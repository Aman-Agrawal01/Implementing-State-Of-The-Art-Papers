import torch
import torch.nn as nn

class Discriminator(nn.Module):
    
    def __init__(self,channels,features):
        super(Discriminator,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=4,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=features,out_channels=features*2,kernel_size=4,stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=features*2,out_channels=features*4,kernel_size=4,stride=2,padding=1)
        self.conv4 = nn.Conv2d(in_channels=features*4,out_channels=features*8,kernel_size=4,stride=2,padding=1)
        self.final = nn.Conv2d(in_channels=features*8,out_channels=1,kernel_size=4,stride=2,padding=0)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.final(x)
        x = self.sigmoid(x)
        return x
    
class Generator(nn.Module):

    def __init__(self,noise,channels,features):
        super(Generator,self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=noise,out_channels=features*16,kernel_size=4,stride=1,padding=0)
        self.conv2 = nn.ConvTranspose2d(in_channels=features*16,out_channels=features*8,kernel_size=4,stride=2,padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=features*8,out_channels=features*8,kernel_size=4,stride=2,padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels=features*4,out_channels=features*2,kernel_size=4,stride=2,padding=1)            
        self.conv5 = nn.ConvTranspose2d(in_channels=features*2,out_channels=channels,kernel_size=4,stride=2,padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.tanh(x)
        return x 