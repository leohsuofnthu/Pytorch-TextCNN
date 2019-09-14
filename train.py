# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:50:00 2019

@author: Gabriel Hsu
"""

#try to use nn for crossentropy

import torch
import torch.nn.functional as F

def train(model, device, train_itr, optimizer, epoch, max_epoch):
    model.train()
    train_loss = 0
    for batch in train_itr:
        text, target = batch.text, batch.label
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        
        optimizer.zero_grad()
        logit = model(text)
        
        
        loss = F.cross_entropy(logit, target)
        loss.backward()
        optimizer.step()
        train_loss+= loss.item()
        
        return train_loss
    
def eval(model, device, data_iter):
    model.eval()
    corrects, avg_loss = 0,0
    for batch in data_iter:
        text, target = batch.text, batch.label
        target.data.sub_(1)
        text, target = text.to(device), target.to(device)
        
        logit = model(text)
        loss = F.cross_entropy(logit, target)

        
        avg_loss += loss.item()
        result = torch.max(logit,1)[1]
        corrects += (result.view(target.size()).data == target.data).sum()
    
    size = len(data_iter.dataset)
    avg_loss /= size 
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f} acc: {:.4f}%({}/{}) \n'.format(avg_loss,accuracy,corrects,size))
    
    return accuracy