
# coding: utf-8

# In[ ]:





# # Import library

# In[ ]:


from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchtext

import numpy as np

import time
import math
import random
import unicodedata
import string
import re
from tqdm import tqdm

import scripts.text
import utils

use_cuda = True
batch_size = 2
learning_rate = 0.01
# # Load data

# In[ ]:


data_path = './processed-data/id.1000/'
en_vocab_path = data_path + 'train.10k.en.vocab'
de_vocab_path = data_path + 'train.10k.de.vocab'


# In[ ]:


en_words, en_vocab, _ = scripts.text.load_vocab(en_vocab_path)
de_words, de_vocab, _ = scripts.text.load_vocab(de_vocab_path)


# In[ ]:


class LuongNMTDataset(torchtext.data.Dataset):
    """
        Custom Dataset for Machine Translation dataset based on torchtext's Dataset class.
    """
    
    def __init__(self, src_path, trg_path, fields, MAX_LENGTH=None, **kwargs):
        """
            Arguments:
                src_path (string): path to source language data.
                trg_path (string): path to target language data.
                fields: A tuple containing the fields that will be used for data in each language.
                Remaining keyword arguments: Passed to the constructor of data.Dataset.
        """
        
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1])]
        
        examples = []
        with open(src_path) as src_file, open(trg_path) as trg_file:
            for src_line, trg_line in tqdm(zip(src_file, trg_file)):
                src_line = map(int, src_line.strip().split(' '))
                trg_line = map(int, trg_line.strip().split(' '))
                if MAX_LENGTH is not None:
                    if len(src_line) > MAX_LENGTH or len(trg_line) > MAX_LENGTH:
                        continue
                if src_line != '' and trg_line != '':
#                     print(src_line)
                    examples.append(torchtext.data.Example.fromlist([src_line, trg_line], fields))
        
        super(LuongNMTDataset, self).__init__(examples, fields, **kwargs)


# In[ ]:


with open(data_path + 'train.10k.en') as f:
    for line in f:
        print(map(int, line.split()))
        break


# In[ ]:


src_field = torchtext.data.Field(sequential=False,
#                                  tokenize=(lambda line: int(line)),
                                 use_vocab=False, 
                                 batch_first=True
                                 )
trg_field = torchtext.data.Field(sequential=False,
#                                  tokenize=(lambda line: int(line)),
                                 use_vocab=False, 
                                 batch_first=True
                                 )


# In[ ]:


train_dataset = LuongNMTDataset(src_path=data_path + 'train.10k.en', 
                            trg_path=data_path + 'train.10k.de', 
                            fields=(src_field, trg_field)
                           )


# In[ ]:


train_loader = torchtext.data.BucketIterator(dataset=train_dataset, 
                                             batch_size=2, 
                                             repeat=False, 
                                             shuffle=True,
                                             sort_within_batch=True, 
                                             sort_key=lambda x: len(x.src)
                                            )


# In[ ]:


len(train_loader)


# In[ ]:


for batch in train_loader:
    print(batch)
    break
    pass

