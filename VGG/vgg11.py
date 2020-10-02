import torch
import torch.nn as nn
from torchsummary import summary

class vgg11(nn.Module):

    def __init__(self):
        super(vgg11,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.conv6 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(7*7*512,4096)
        self.fc2 = nn.Linear(4096,4096)
        self.output = nn.Linear(4096,10)

    def forward(self,x):

        #1st layer
        x = self.conv1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        #2nd layer
        x = self.conv2(x)
        x = self.relu(x)

        x = self.maxpool(x)

        #3rd layer
        x = self.conv3(x)
        x = self.relu(x)

        #4th layer
        x = self.conv4(x)
        x = self.relu(x)

        x = self.maxpool(x)

        #5th layer
        x = self.conv5(x)
        x = self.relu(x)

        #6th layer
        x = self.conv6(x)
        x = self.relu(x)

        x = self.maxpool(x)

        #7th layer
        x = self.conv6(x)
        x = self.relu(x)

        #8th layer
        x = self.conv6(x)
        x = self.relu(x)

        x = self.maxpool(x)
        
        #Flatten
        x = x.view(-1,7*7*512)

        #9th layer
        x = self.fc1(x)
        x = self.relu(x)

        #10th layer
        x = self.fc2(x)
        x = self.relu(x)

        #11th layer
        x = self.output(x)
        x = self.relu(x)
        
        return x

model = vgg11()
print(summary(model,(1,224,224)))