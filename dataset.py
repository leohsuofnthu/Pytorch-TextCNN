# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 17:29:01 2019

@author: HSU, CHIH-CHAO

"""
import re

import pandas as pd
from numpy.random import RandomState

import torchtext
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator
import torchtext.datasets

import spacy

def split_train_valid(path_data, path_train, path_valid, frac=0.7):
    df = pd.read_csv(path_data)
    rng = RandomState()
    tr = df.sample(frac=0.7, random_state=rng)
    tst = df.loc[~df.index.isin(tr.index)]
    print("Spliting original file to train/valid set...")
    tr.to_csv(path_train, index=False)
    tst.to_csv(path_valid, index=False)

"""
Code taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
"""
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()
    
def create_tabular_dataset(path_train, path_valid, 
                          lang='en', pretrained_emb='glove.6B.300d'):
    
    spacy_en = spacy.load('en', disable=['tagger', 'parser', 'ner', 'textcat'
                                     'entity_ruler', 'sentencizer', 
                                     'merge_noun_chunks', 'merge_entities',
                                     'merge_subtokens'])

    def tokenizer(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    #Creating field for text and label
    TEXT = Field(sequential=True, tokenize=tokenizer, lower=True)
    LABEL = Field(sequential=False)
    
    print('Preprocessing the text...')
    #clean the text
    TEXT.preprocessing = torchtext.data.Pipeline(clean_str)

    print('Creating tabular datasets...It might take a while to finish!')
    train_datafield = [('text', TEXT),  ('label', LABEL)]
    tabular_train = TabularDataset(path = path_train,  
                                 format= 'csv',
                                 skip_header=True,
                                 fields=train_datafield)
    
    valid_datafield = [('text', TEXT),  ('label',LABEL)]
    
    tabular_valid = TabularDataset(path = path_valid, 
                           format='csv',
                           skip_header=True,
                           fields=valid_datafield)
    
    print('Building vocaulary...')
    TEXT.build_vocab(tabular_train, vectors= pretrained_emb)
    LABEL.build_vocab(tabular_train)

    
    return tabular_train, tabular_valid, TEXT.vocab

def create_data_iterator(tr_batch_size, val_batch_size,tabular_train, 
                         tabular_valid, d):
    #Create the Iterator for datasets (Iterator works like dataloader)
    
    train_iter = Iterator(
            tabular_train, 
            batch_size=tr_batch_size,
            device = d, 
            sort_within_batch=False,
            repeat=False)
    
    valid_iter = Iterator(
            tabular_valid, 
            batch_size=val_batch_size,
            device=d,
            sort_within_batch=False, 
            repeat=False)
    
    return train_iter, valid_iter