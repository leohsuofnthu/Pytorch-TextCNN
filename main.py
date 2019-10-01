# -*- coding: utf-8 -*-
"""main.ipynb

@author: HSU, CHIH-CHAO
"""

import argparse

import torch
import torch.optim as optim

import dataset
import model
import training

import matplotlib.pyplot as plt



#%%

def main():
    
    print("Pytorch Version:", torch.__version__)
    parser = argparse.ArgumentParser(description='TextCNN')
    #Training args
    parser.add_argument('--data-csv', type=str, default='./IMDB_Dataset.csv',
                        help='file path of training data in CSV format (default: ./train.csv)')
    
    parser.add_argument('--spacy-lang', type=str, default='en', 
                        help='language choice for spacy to tokenize the text')
                        
    parser.add_argument('--pretrained', type=str, default='glove.6B.300d',
                    help='choice of pretrined word embedding from torchtext')              
                        
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
    
    parser.add_argument('--val-batch-size', type=int, default=64,
                        help='input batch size for testing (default: 64)')
    
    parser.add_argument('--kernel-height', type=str, default='3,4,5',
                    help='how many kernel width for convolution (default: 3, 4, 5)')
    
    parser.add_argument('--out-channel', type=int, default=100,
                    help='output channel for convolutionaly layer (default: 100)')
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate for linear layer (default: 0.5)')
    
    parser.add_argument('--num-class', type=int, default=2,
                        help='number of category to classify (default: 2)')
    
    #if you are using jupyternotebook with argparser
    args = parser.parse_known_args()[0]
    #args = parser.parse_args()
    
    
    #Use GPU if it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    #%% Split whole dataset into train and valid set
    dataset.split_train_valid(args.data_csv, './train.csv', './valid.csv', 0.7)
    
    trainset, validset, vocab = dataset.create_tabular_dataset('./train.csv',
                                 './valid.csv',args.spacy_lang, args.pretrained)
    
    #%%Show some example to show the dataset
    print("Show some examples from train/valid..")
    print(trainset[0].text,  trainset[0].label)
    print(validset[0].text,  validset[0].label)
    
    train_iter, valid_iter = dataset.create_data_iterator(args.batch_size, args.val_batch_size,
                                                         trainset, validset,device)
                
    #%%Create
    kernels = [int(x) for x in args.kernel_height.split(',')]
    m = model.textCNN(vocab, args.out_channel, kernels, args.dropout , args.num_class).to(device)
    # print the model summery
    print(m)    
        
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []
    best_test_acc = -1
    
    #optimizer
    optimizer = optim.Adam(m.parameters(), lr=args.lr)
    
    for epoch in range(1, args.epochs+1):
        #train loss
        tr_loss, tr_acc = training.train(m, device, train_iter, optimizer, epoch, args.epochs)
        print('Train Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, tr_loss, tr_acc))
        
        ts_loss, ts_acc = training.valid(m, device, valid_iter)
        print('Valid Epoch: {} \t Loss: {} \t Accuracy: {}%'.format(epoch, ts_loss, ts_acc))
        
        if ts_acc > best_test_acc:
            best_test_acc = ts_acc
            #save paras(snapshot)
            print("model saves at {}% accuracy".format(best_test_acc))
            torch.save(m.state_dict(), "best_validation")
            
        train_loss.append(tr_loss)
        train_acc.append(tr_acc)
        test_loss.append(ts_loss)
        test_acc.append(ts_acc)
    
    #plot train/validation loss versus epoch
    #plot train/validation loss versus epoch
    x = list(range(1, args.epochs+1))
    plt.figure()
    plt.title("train/validation loss versus epoch")
    plt.xlabel("epoch")
    plt.ylabel("Average loss")
    plt.plot(x, train_loss,label="train loss")
    plt.plot(x, test_loss, color='red', label="test loss")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    #plot train/validation accuracy versus epoch
    x = list(range(1, args.epochs+1))
    plt.figure()
    plt.title("train/validation accuracy versus epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy(%)")
    plt.plot(x, train_acc,label="train accuracy")
    plt.plot(x, test_acc, color='red', label="test accuracy")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()

