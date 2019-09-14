# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:14:26 2019

@author: Gabriel Hsu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#improve dynamic convolution and maxpool

class textCNN(nn.Module):
    
    def __init__(self, vocab_built, emb_dim, dim_channel, kernel_wins, num_class):
        super(textCNN, self).__init__()
        #load pretrained embedding in embedding layer.
        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors)
    
        #Convolutional Layers with different window size kernels
#        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, w, emb_dim) for w in kernel_wins])
        self.conv_w3 = nn.Conv2d(1, dim_channel, (3, emb_dim))
        self.conv_w4 = nn.Conv2d(1, dim_channel, (4, emb_dim))
        self.conv_w5 = nn.Conv2d(1, dim_channel, (5, emb_dim))
    
        #dropout layer
        self.dropout = nn.Dropout(0.6)
        
        #FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        
    def forward(self, x):
        x = self.embed(x)
        x = x.unsqueeze(1)
        x_3 = F.relu(self.conv_w3(x))
        x_4 = F.relu(self.conv_w4(x))
        x_5 = F.relu(self.conv_w5(x))
     
        
        
        x_3 = F.max_pool1d(x_3.squeeze(-1), x_3.size()[2])
        x_4 = F.max_pool1d(x_4.squeeze(-1), x_4.size()[2])
        x_5 = F.max_pool1d(x_5.squeeze(-1), x_5.size()[2])
        
        xx = torch.cat((x_3,x_4,x_5), dim=1)
        xx = xx.squeeze(-1)
        xx = self.dropout(xx)
        logit = self.fc(xx)
        return logit