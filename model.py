# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:14:26 2019

@author: HSU, CHIH-CHAO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#%% Text CNN model
class textCNN(nn.Module):
    
    def __init__(self, vocab_built, dim_channel, kernel_wins, dropout_rate, num_class):
        super(textCNN, self).__init__()
        #load pretrained embedding in embedding layer.
        emb_dim = vocab_built.vectors.size()[1]
        self.embed = nn.Embedding(len(vocab_built), emb_dim)
        self.embed.weight.data.copy_(vocab_built.vectors)
    
        #Convolutional Layers with different window size kernels
        self.convs = nn.ModuleList([nn.Conv2d(1, dim_channel, (w, emb_dim)) for w in kernel_wins])
        #Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        #FC layer
        self.fc = nn.Linear(len(kernel_wins)*dim_channel, num_class)
        
    def forward(self, x):
        emb_x = self.embed(x)
        emb_x = emb_x.unsqueeze(1)

        con_x = [conv(emb_x) for conv in self.convs]

        pool_x = [F.max_pool1d(x.squeeze(-1), x.size()[2]) for x in con_x]
        
        fc_x = torch.cat(pool_x, dim=1)
        
        fc_x = fc_x.squeeze(-1)

        fc_x = self.dropout(fc_x)
        logit = self.fc(fc_x)
        return logit
