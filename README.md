# Convolutional Neural Networks for Sentence Classification

This is an Pytorch implementation of the paper [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882), the structure in this project is named as **CNN-non-static** in the paper. I wrap this model to facilitate anyone who want to train and validate their own dataset with TextCNN rapidly by preparing certain format of *.csv* file and providing parameters from users.

## Getting Started

Follow the following section to conduct your multi-calss or binary-class sentimental analysis.

### 1. Check Environment

This project is developed under following enviroment:

```
python 3.6
pytorch 1.2.0
torchtext 0.4.0
numpy 1.16.5
pandas 0.25.1
matplotlib 3.1.1
spacy 2.1.8
```

### 2. Prepare .csv file

Before running the code, you need to make sure your .csv file follows the fomat with 2 column only. One for text and the other for label, the structure is depicted as follows. (you can also checked on the format of *IMDB_Dataset.csv*. The data under label coulmn could be numericalized or not.

|text|label|
|---|---|
|I am happy|positive|
|I hate you|negtive|


### 3. Run the code

There are two ways that enable to train on your dataset.

#### With Jupyter notebook

*textCNN_IMDB.ipynb* contains a complete procedure of sentiment analysis on IMDB dataset as provided in this repo, which will let you quickly train (Simply change the path in first line of code) with some pre-defined parameters. The version with argparser will be updated soon.

```
df = pd.read_csv('your file path')
```

#### With main.py

Directly execute the main.py with prefered parameters, the details of parameters is provided in following section.

```
python main.py --data-csv ./IMDB_Dataset.csv --spacy-lang en --pretrained glove.6B.300d --epochs 10 --lr 0.01 --batch-size 64 
--val-batch-size 64 --kernel-height 3,4,5 --out-channel 100 --dropout 0.5 -num-class 2
```

#### Parameters
```
  -h, --help                         show this help message and exit
  --data-csv DATA_CSV                file path of training data in CSV format (default:./train.csv)
  --spacy-lang SPACY_LANG            language choice for spacy to tokenize the text (default:en)
  --pretrained PRETRAINED            choice of pretrined word embedding from torchtext (default:glove.6B.300d)
  --epochs EPOCHS                    number of epochs to train (default: 10)
  --lr LR                            learning rate (default: 0.01)
  --momentum MOMENTUM                SGD momentum (default: 0.9)
  --batch-size BATCH_SIZE            input batch size for training (default: 64)
  --val-batch-size VAL_BATCH_SIZE    input batch size for testing (default: 64)
  --kernel-height KERNEL_HEIGHT      kernels for convolution (default: 3, 4, 5)
  --out-channel OUT_CHANNEL          output channel for convolutionaly layer (default: 100)
  --dropout DROPOUT                  dropout rate for linear layer (default: 0.5)
  --num-class NUM_CLASS              number of category to classify (default: 2)
```
**Pleace check out the available option for '--spacy-lang' at [spacy language support](https://spacy.io/usage/models#languages) 
and '--pretrained' at [torchtext pretrained_alias](https://torchtext.readthedocs.io/en/latest/vocab.html#pretrained-aliases)**

### 4. Check the results

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
