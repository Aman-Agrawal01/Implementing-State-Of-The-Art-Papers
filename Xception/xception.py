#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn
from torchsummary import summary


# In[2]:


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
        self.bnm1 = nn.BatchNorm2d(32)
        self.bnm2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        # 299x299x3
        x = self.conv1(x)
        x = self.bnm1(x)
        x = self.relu(x)
        # 149x149x32

        x = self.conv2(x)
        x = self.bnm2(x)
        x = self.relu(x)
        # 147x147x64

        return x


# In[3]:


sample = EntryflowConv(in_channel=3,out_channel=64)
summary(sample,input_size=(3,299,299))


# In[4]:


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


# In[5]:


# Padding = 1 ('same') in all such layers
sample = DepthwiseSeparable(in_channel=64,out_channel=128,kernel_size=3)
summary(sample,input_size=(64,147,147))


# In[6]:


class EntryflowSeparable(nn.Module):
    """
    This part contains depthwise separable convolutions and is repeated 3 times in original implementation.

        in_channel, out_channel : Different for each repetition
        pool_padding: default :1 , Padding value for max_pool layer
        kernel_size = 3 : For all repetitions
        relu_extra : bool, default : false : Whether or not put a relu layer in the beginning
    """
    def __init__(self,in_channel,out_channel,pool_padding=1,relu_extra=False):
        super(EntryflowSeparable,self).__init__()

        # 1st branch
        self.sepconv1 = DepthwiseSeparable(in_channel=in_channel,out_channel=out_channel,kernel_size=3)
        self.bnm1 = nn.BatchNorm2d(out_channel)
        self.sepconv2 = DepthwiseSeparable(in_channel=out_channel,out_channel=out_channel,kernel_size=3)
        self.bnm2 = nn.BatchNorm2d(out_channel)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=pool_padding)
        self.relu = nn.ReLU()
        self.relu_extra = relu_extra

        # 2nd branch (left)
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=2)
        self.bnmy = nn.BatchNorm2d(out_channel)

    def forward(self,x):
        # 2nd branch
        y = self.conv(x)
        y = self.bnmy(y)

        # 1st branch
        if self.relu_extra:
            x = self.relu(x)
        x = self.sepconv1(x)
        x = self.bnm1(x)
        x = self.relu(x)

        x = self.sepconv2(x)
        x = self.bnm2(x)
        x = self.maxpool(x)

        # Add two branch
        x = x + y
        return x


# In[7]:


sample = EntryflowSeparable(in_channel=64,out_channel=128)
summary(sample,input_size=(64,147,147))


# In[8]:


sample = EntryflowSeparable(in_channel=128,out_channel=256,relu_extra=True)
summary(sample,input_size=(128,74,74))


# In[9]:


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
        self.bnm_conv = nn.BatchNorm2d(64)
        self.sep1 = EntryflowSeparable(in_channel=64,out_channel=128)
        self.bnm1 = nn.BatchNorm2d(128)
        self.sep2 = EntryflowSeparable(in_channel=128,out_channel=256,relu_extra=True)
        self.bnm2 = nn.BatchNorm2d(256)
        self.sep3 = EntryflowSeparable(in_channel=256,out_channel=728,relu_extra=True)
        self.bnm3 = nn.BatchNorm2d(728)

    def forward(self,x):
        x = self.conv(x)
        x = self.bnm_conv(x)
        x = self.sep1(x)
        x = self.bnm1(x)
        x = self.sep2(x)
        x = self.bnm2(x)
        x = self.sep3(x)
        x = self.bnm3(x)

        return x


# In[10]:


xception_entry = EntryFlow()
summary(xception_entry,input_size=(3,299,299))


# In[11]:


class MiddleflowSeperable(nn.Module):
    """
    This part contains depthwise separable convolutions and is repeated 3 times in original implementation.

        in_channel, out_channel : Both of them are actually equal!
        kernel_size = 3 : For all repetitions
    """
    def __init__(self,in_channel,out_channel):
        super(MiddleflowSeperable,self).__init__()

        # 1st branch
        self.sep1 = DepthwiseSeparable(in_channel=in_channel,out_channel=out_channel,kernel_size=3)
        self.sep2 = DepthwiseSeparable(in_channel=out_channel,out_channel=out_channel,kernel_size=3)
        self.sep3 = DepthwiseSeparable(in_channel=out_channel,out_channel=out_channel,kernel_size=3)
        self.bnm = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self,x):
        # 2nd branch
        y = x

        # 1st branch
        x = self.relu(x)
        x = self.sep1(x)
        x = self.bnm(x)

        x = self.relu(x)
        x = self.sep2(x)
        x = self.bnm(x)

        x = self.relu(x)
        x = self.sep3(x)
        x = self.bnm(x)

        # Add two branch
        x = x + y
        return x


# In[12]:


model = MiddleflowSeperable(in_channel=728,out_channel=728)
summary(model=model,input_size=(728,19,19))


# In[13]:


class MiddleFlow(nn.Module):
    """
    This is the Middle Flow part -
        MiddleFlowSeperable is repeated 8 times 
    
    input_size = (728,19,19)
    output_size = (728,19,19)       
    """
    def __init__(self):
        super(MiddleFlow,self).__init__()
        self.sep = MiddleflowSeperable(in_channel=728,out_channel=728)
        self.bnm = nn.BatchNorm2d(728)
    def forward(self,x):
        for i in range(8):
            x = self.bnm(self.sep(x))
        return x


# In[14]:


sample = MiddleFlow()
summary(model=sample,input_size=(728,19,19))


# In[15]:


class ExitflowSeperable(nn.Module):
    """
    This part contains depthwise separable convolutions and is repeated 2 times in original implementation with max pool layer.

        in_channel, out_channel : Both of them are different
        kernel_size = 3 : For all repetitions
        max pool kernel_size :3 with stride:2
    """
    def __init__(self,in_channel,out_channel,padding=1):
        super(ExitflowSeperable,self).__init__()

        #1st branch
        self.sep1 = DepthwiseSeparable(in_channel=in_channel,out_channel=in_channel,kernel_size=3)
        self.sep2 = DepthwiseSeparable(in_channel=in_channel,out_channel=out_channel,kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=padding)
        
        #2nd branch
        self.conv = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,kernel_size=1,stride=2)
        
        self.bnm1 = nn.BatchNorm2d(in_channel)
        self.bnm2 = nn.BatchNorm2d(out_channel)

    def forward(self,x):

        #2nd branch 
        y = self.conv(x)
        y = self.bnm2(y)

        #1st branch
        x = self.relu(x)
        x = self.sep1(x)
        x = self.bnm1(x)
        x = self.relu(x)
        x = self.sep2(x)
        x = self.bnm2(x)
        x = self.pool(x)

        return x+y    


# In[16]:


sample = ExitflowSeperable(in_channel=728,out_channel=1024)
summary(model=sample,input_size=(728,19,19))


# In[17]:


class ExitFlow(nn.Module):
    """
    This part contains ExitFlowSeperable part with 2 different depthwise seperable convolutions followed by Global Avgerage Pool(Avg Pool of kernel size 10) and connecting with output layer

    input_size  :(728,19,19)
    output_size :(output_layer)
    """
    def __init__(self,in_channel=728,out_channel=1024,first_layer=1536,second_layer=2048,output_layer=1000):
        super(ExitFlow,self).__init__()
        self.block = ExitflowSeperable(in_channel=in_channel,out_channel=out_channel)
        self.sep1 = DepthwiseSeparable(in_channel=1024,out_channel=first_layer,kernel_size=3)
        self.bnm1 = nn.BatchNorm2d(first_layer)
        self.sep2 = DepthwiseSeparable(in_channel=first_layer,out_channel=second_layer,kernel_size=3)
        self.bnm2 = nn.BatchNorm2d(second_layer)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=10)
        self.flatten = nn.Flatten()
        self.output = nn.Linear(in_features=second_layer,out_features=output_layer)

    def forward(self,x):
        x = self.block(x)
        x = self.sep1(x)
        x = self.bnm1(x)
        x = self.relu(x)
        x = self.sep2(x)
        x = self.bnm2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.output(x)
        return x


# In[18]:


sample = ExitFlow()
summary(model=sample,input_size=(728,19,19))


# In[19]:


class Xception(nn.Module):
    """
        Now, this is the final part where we merge all the flow i.e. entry, middle and exit flow to get the Xception Model
    """
    def __init__(self):
        super(Xception,self).__init__()
        self.entry = EntryFlow()
        self.mid = MiddleFlow()
        self.exit = ExitFlow()
    def forward(self,x):
        x = self.entry(x)
        x = self.mid(x)
        x = self.exit(x)
        return x


# In[20]:


xception = Xception()
summary(model=xception,input_size=(3,299,299))

