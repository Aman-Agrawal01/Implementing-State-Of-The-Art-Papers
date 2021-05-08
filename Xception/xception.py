#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
from torchsummary import summary


# In[17]:


class EntryflowConv(nn.Module):
    """
    First Part in Entry Flow having only Convolution layers.
    In Xception:
        in_channel = 3
        out_channel = 64
    """
    def __init__(self,in_channel,out_channel):
        super(EntryflowConv,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=32,kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=out_channel,kernel_size=3)

        self.relu = nn.ReLU()

    def forward(self,x):
        # 299x299x3
        x = self.conv1(x)
        x = self.relu(x)
        # 149x149x32

        x = self.conv2(x)
        x = self.relu(x)
        # 147x147x64

        return x


# In[3]:


sample = EntryflowConv(in_channel=3,out_channel=64)
summary(sample,input_size=(3,299,299))


# In[8]:


class DepthwiseSeparable(nn.Module):
    """
    Depthwise Separable Convolution is Depthwise Convolution + Pointwise Convolution.
        Depthwise Convolution : Convolution over each channel independently
            Divide input channels into "in_channel" groups and then apply convolution over each
            Group independently : Depth is not used
        Pointwise Convolution : Normal Convolution with kernel Size (1,1)
            Only depth Used.

    In Xception Architecture the Order of operation is different:
        Pointwise Convolution + Depthwise Convolution

    groups : No of groups the input channel should be divided into
             For depthwise convolution = in_channel
    padding = default: "same" (1 for kernel_size = 3)
    """
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=1):
        super(DepthwiseSeparable,self).__init__()

        self.pointwise = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1)
        self.depthwise = nn.Conv2d(in_channels=out_channel,out_channels=out_channel,kernel_size=kernel_size,stride=stride,padding=padding,groups=out_channel)

    def forward(self,x):
        x = self.pointwise(x)
        x = self.depthwise(x)

        return x


# In[9]:


# Padding = 1 ('same') in all such layers
sample = DepthwiseSeparable(in_channel=64,out_channel=128,kernel_size=3)
summary(sample,input_size=(64,147,147))


# In[13]:


class EntryflowSeparable(nn.Module):
    """
    This part contains depthwise separable convolutions and is repeated 3 times in original implementation.

        in_channel, out_channel : Different for each repetition
        pool_padding: default :1 , Padding value for max_pool layer
        kernel_size = 3 : For all repetitions
    """
    def __init__(self,in_channel,out_channel,pool_padding=1):
        super(EntryflowSeparable,self).__init__()

        # 1st branch
        self.sepconv1 = DepthwiseSeparable(in_channel=in_channel,out_channel=out_channel,kernel_size=3)
        self.sepconv2 = DepthwiseSeparable(in_channel=out_channel,out_channel=out_channel,kernel_size=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=pool_padding)
        self.relu = nn.ReLU()

        # 2nd branch (left)
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=2)

    def forward(self,x):
        # 2nd branch
        y = self.conv(x)

        # 1st branch
        x = self.sepconv1(x)
        x = self.relu(x)

        x = self.sepconv2(x)

        x = self.maxpool(x)

        # Add two branch
        x = x + y
        return x


# In[14]:


sample = EntryflowSeparable(in_channel=64,out_channel=128)
summary(sample,input_size=(64,147,147))


# In[18]:


class EntryFlow(nn.Module):
    """
    Entry Flow Part of Xception :

        EntryflowConv + 3 x EntryflowSeparable
        in_channel = 3
        out_channel = 728
    """
    def __init__(self):
        super(EntryFlow,self).__init__()
        self.conv = EntryflowConv(in_channel=3,out_channel=64)
        self.sep1 = EntryflowSeparable(in_channel=64,out_channel=128)
        self.sep2 = EntryflowSeparable(in_channel=128,out_channel=256)
        self.sep3 = EntryflowSeparable(in_channel=256,out_channel=728)

    def forward(self,x):
        x = self.conv(x)
        x = self.sep1(x)
        x = self.sep2(x)
        x = self.sep3(x)

        return x


# In[19]:


xception_entry = EntryFlow()
summary(xception_entry,input_size=(3,299,299))


# In[ ]:




