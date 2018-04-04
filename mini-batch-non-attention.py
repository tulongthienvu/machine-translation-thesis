
# coding: utf-8

# TODO
# - Import
# - Read data (vocab, sentences)
# - Build model
# - Train model
#     - Mini-batch
# - Evaluate model
#     - Loss
#     - Perplexity
#     - BLEU

# In[1]:





# # Import library

# In[2]:


from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
import torchtext
from tqdm import tqdm

import numpy as np

import time
import math
import random
import unicodedata
import string
import re

import scripts.text
import utils

use_cuda = True
batch_size = 16
learning_rate = 0.01
# # Load data

# In[3]:


data_path = './processed-data/id.1000/'
en_vocab_path = data_path + 'train.10k.en.vocab'
de_vocab_path = data_path + 'train.10k.de.vocab'


# In[4]:


en_words, en_vocab, _ = scripts.text.load_vocab(en_vocab_path)
de_words, de_vocab, _ = scripts.text.load_vocab(de_vocab_path)


# In[5]:


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
                # src_line = map(int, src_line.strip().split(' '))
                # trg_line = map(int, trg_line.strip().split(' '))
                src_line = src_line.strip().split(' ')
                trg_line = trg_line.strip().split(' ')
                if MAX_LENGTH is not None:
                    if len(src_line) > MAX_LENGTH or len(trg_line) > MAX_LENGTH:
                        continue
                if src_line != '' and trg_line != '':
#                     print(src_line)
                    examples.append(torchtext.data.Example.fromlist([src_line, trg_line], fields))
        
        super(LuongNMTDataset, self).__init__(examples, fields, **kwargs)


# In[6]:


def post_processing(arr, field_vocab, train):
    for index in range(0, len(arr)):
        arr[index] = map(int, arr[index])
    return arr


# In[7]:


src_field = torchtext.data.Field(sequential=True,
#                                  tokenize=(lambda line: int(line)),
                                 postprocessing=post_processing,
                                 use_vocab=False,
                                 pad_token='1000',
                                 include_lengths=True,
                                 batch_first=True,
                                 )
trg_field = torchtext.data.Field(sequential=True,
#                                  tokenize=(lambda line: int(line)),
                                 postprocessing=post_processing,
                                 use_vocab=False,
                                 include_lengths=True,
                                 pad_token='1000',
                                 batch_first=True
                                 )

# In[8]:


train_dataset = LuongNMTDataset(src_path=data_path + 'train.10k.en', 
                            trg_path=data_path + 'train.10k.de', 
                            fields=(src_field, trg_field)
                           )


# In[9]:


train_loader = torchtext.data.BucketIterator(dataset=train_dataset, 
                                             batch_size=batch_size,
                                             repeat=False, 
                                             shuffle=True,
                                             sort_within_batch=True, 
                                             sort_key=lambda x: len(x.src)
                                            )


valid_dataset = LuongNMTDataset(src_path=data_path + 'valid.100.en',
                            trg_path=data_path + 'valid.100.de',
                            fields=(src_field, trg_field)
                           )


# In[9]:


valid_loader = torchtext.data.BucketIterator(dataset=valid_dataset,
                                             batch_size=batch_size,
                                             repeat=False,
                                             shuffle=False,
                                             sort_within_batch=True,
                                             sort_key=lambda x: len(x.src)
                                            )

# In[10]:


# Read validation data
en_valid_sentences = []
with open(data_path + 'valid.100.en', 'r') as f:
    for line in f:
        en_valid_sentences.append(map(lambda x: int(x), line.split()))
        
de_valid_sentences = []
with open(data_path + 'valid.100.de', 'r') as f:
    for line in f:
        de_valid_sentences.append(map(lambda x: int(x), line.split()))


# # Build model

# ## Using RNNs

# In[11]:


class EncoderRNN(nn.Module):
    """
        Model's encoder using RNN.
    """

    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size + 1, embedding_size, padding_idx=1000)
        # self.embedding = nn.Embedding(input_size + 1, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.init_weights()

    def forward(self, input_sentences, input_lengths, hidden):
#         sentence_len = len(input_sentences)
        longest_length = input_lengths[0]
        embedded = self.embedding(input_sentences)
        
        packed = rnn_utils.pack_padded_sequence(embedded, input_lengths, batch_first=True)
#         embedded = embedded.view(sentence_len, batch_size, -1)
#         embedded = embedded.view(batch_size, longest_length, -1)
        output, hidden = self.rnn(packed, hidden)
        # print(output)
        return output, hidden
    
    def init_hidden(self):
        """
            Initialize hidden state.
        """
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            hidden = hidden.cuda()
        return hidden
    
    def init_weights(self):
        """
            Initialize weights.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)


# In[12]:


class DecoderRNN(nn.Module):
    """
        Model's decoder using RNN.
    """

    def __init__(self, embedding_size, hidden_size, output_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        output_size += 1
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=1000)
        # self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.init_weights()

    def forward(self, input_vector, hidden):
        # view(batch_size, sentence_length, -1)
        output = self.embedding(input_vector).view(batch_size, 1, -1)
        output = F.relu(output)
        # output = rnn_utils.pack_sequence(output)
        output, hidden = self.rnn(output, hidden)
        # print("Output: ", output)
        # output = self.log_softmax(self.linear_out(output[-1]))
        output = self.log_softmax(self.linear_out(output))   # Change dim for packed sequence
        output = output.view(batch_size, -1)
        # print("output final: ", output)
        return output, hidden
    
    def init_hidden(self):
        """
            Initialize hidden state.
        """
        hidden = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        if use_cuda:
            hidden = hidden.cuda()
        return hidden
    
    def init_weights(self):
        """
            Initialize weights.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear_out.weight.data.uniform_(-0.1, 0.1)
        self.linear_out.bias.data.fill_(0)


# # Training model

# ## Using RNN



teacher_forcing_ratio = 0.5
clip = 5.0
MAX_LENGTH = 50


# In[14]:


def train(input_variables, input_lengths, target_variables, target_lengths, encoder, decoder, 
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Zero gradient
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    
#     # Get size of input and target sentences
#     input_length = input_variable.size()[0]
#     target_length = target_variable.size()[0]

    # Run words through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variables, input_lengths, encoder_hidden)

    # Prepare input for decoder and output variables
    # decoder_input = Variable(torch.LongTensor([[de_vocab['<s>']]]))
    decoder_input = torch.zeros((batch_size, 1), ).type(torch.LongTensor)
    decoder_input += de_vocab['<s>']
    # decoder_input = rnn_utils.pack_sequence(decoder_input)
    # print("Decoder input: ", decoder_input)
    decoder_hidden = encoder_hidden  # Use last hidden from the encoder

    if use_cuda:
        decoder_input = decoder_input.cuda()
    # print("Input_variables: ", input_variables)
    # print("Target_variables: ", target_variables)
    # Transpose target variables for mini-batch implement
    target_variables.t_()
    # Find max target lengths
    max_target_length = max(target_lengths)
    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        # Teacher forcing: use the ground-truth target as the next input
        for d_i in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variables[d_i])
            decoder_input = target_variables[d_i]
    else:
        # Without teacher forcing use its own predictions as the next input
        for d_i in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
#             print(decoder_output)
            loss += criterion(decoder_output, target_variables[d_i])
            # Pick most likely word index (highest value) from output (greedy search)
            top_value, top_index = decoder_output.data.topk(1)
            # n_i = top_index[0][0]   # Modify for mini-batch implement
            n_i = top_index
#             print(n_i)
#             print(torch.LongTensor([n_i]))
#             decoder_input = Variable(torch.LongTensor([[n_i]])) # Chosen word is next input
            # Modify for mini-batch implement
            decoder_input = torch.zeros((batch_size, 1), ).type(torch.LongTensor)
            if use_cuda:    # not optimized yet
                decoder_input = decoder_input.cuda()
            decoder_input += n_i

            # Stop at end of sentence (not necessary when using known targers)
            # if n_i == en_vocab['</s>']:
            #     break
    # Backpropagation
    loss.backward()
    nn.utils.clip_grad_norm(encoder.parameters(), clip)
    nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data / max_target_length


# ### Run training

# In[15]:


# Set hyperparameters
embedding_size = 64
hidden_size = 64
num_layers = 2
dropout_p = 0.00

# Initialize models
encoder = EncoderRNN(len(en_vocab), embedding_size, hidden_size, num_layers)
decoder = DecoderRNN(embedding_size, hidden_size, len(de_vocab), num_layers)

# Move models to GPU
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    
# Initialize parameters and criterion
# learning_rate = 0.0001
# encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(ignore_index=1000)
# criterion = nn.NLLLoss()

# In[16]:


# Configuring training
num_epochs = 1
plot_every = 100
print_every = 100

# Keep track of time elapsed and running averages
plot_losses = []
print_loss_total = 0 # Reset every print every
plot_loss_total = 0 # Reset every plot every


# In[ ]:


# Convert all sentences to Variable
# if use_cuda:
#     for i in range(len(en_train_sentences)):
#         en_train_sentences[i] = Variable(torch.LongTensor(en_train_sentences[i]).view(-1, 1)).cuda()
#         de_train_sentences[i] = Variable(torch.LongTensor(de_train_sentences[i]).view(-1, 1)).cuda()
# else:
#     for i in range(len(en_train_sentences)):
#         en_train_sentences[i] = Variable(torch.LongTensor(en_train_sentences[i]).view(-1, 1))
#         de_train_sentences[i] = Variable(torch.LongTensor(de_train_sentences[i]).view(-1, 1))
#
# if use_cuda:
#     for i in range(len(en_valid_sentences)):
#         en_valid_sentences[i] = Variable(torch.LongTensor(en_valid_sentences[i]).view(-1, 1)).cuda()
#         de_valid_sentences[i] = Variable(torch.LongTensor(de_valid_sentences[i]).view(-1, 1)).cuda()
# else:
#     for i in range(len(en_valid_sentences)):
#         en_valid_sentences[i] = Variable(torch.LongTensor(en_valid_sentences[i]).view(-1, 1))
#         de_valid_sentences[i] = Variable(torch.LongTensor(de_valid_sentences[i]).view(-1, 1))


# In[ ]:


# Running training
# start = time.time()
# for epoch in range(0, num_epochs):
#     #start epoch
#     # Shuffle
#     indexes = np.arange(0, len(en_train_sentences))
#     np.random.shuffle(indexes)
#     step = 1
#     num_steps = math.ceil(len(en_train_sentences) / batch_size)
#     for index in indexes:
#         input_variable = en_train_sentences[index:index + 2]
#         print(input_variable)
#         target_variable = de_train_sentences[index:index + 2]
#         loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer,
#                      decoder_optimizer, criterion)
#         print_loss_total += loss
#         plot_loss_total += loss
#
#         if step == 0:
#             step += 1
# #             continue
#             break
#
#         if step % print_every == 0 or step == num_steps:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print_summary = 'Epoch %s/%s, Time: %s, Step: %d/%d, train_loss: %.4f' % (epoch, num_epochs,
#                                                                 utils.time_since(start, step / num_steps),
#                                                                 step,
#                                                                 num_steps, print_loss_avg)
#             print(print_summary)
#
#         if step % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total
#         step += 1
#
#         # stop when reaching certain steps
#         if step == 2000:
#             break
#
#     # end epoch
#     # evaluate on validation set
#     valid_total_loss = 0
#     for i in range(len(en_valid_sentences)):
#         input_variable = en_valid_sentences[i]
#         output_varible = de_valid_sentences[i]
#         valid_loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer,
#                      decoder_optimizer, criterion)
#         valid_total_loss += valid_loss
#     print('Validation loss: %.4f' % (valid_total_loss / len(en_valid_sentences)))
        


# In[17]:


# Running training
start = time.time()
for epoch in range(0, num_epochs):
    #start epoch
    # Shuffle
    step = 1
    num_steps = math.ceil(len(train_dataset) / batch_size)
    for batch in train_loader:
        # input_variables, input_lengths = Variable(batch.src)
        # target_variables, target_lengths = Variable(batch.trg)
        input_variables, input_lengths = batch.src
        # print(input_variables)
        # print(input_lengths)
        target_variables, target_lengths = batch.trg
        loss = train(input_variables, input_lengths, target_variables, target_lengths, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        
        if step == 0:
            step += 1
#             continue
            break
        
        if step % print_every == 0 or step == num_steps:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_summary = 'Epoch %s/%s, Time: %s, Step: %d/%d, train_loss: %.4f' % (epoch + 1, num_epochs,
                                                                utils.time_since(start, step / num_steps),
                                                                step,
                                                                num_steps, print_loss_avg)
            print(print_summary)
        
        if step % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total
        step += 1
        
        # stop when reaching certain steps
        if step == 1000:
            break
        
    # end epoch
    # evaluate on validation set
    valid_total_loss = 0
    # for i in range(len(en_valid_sentences)):
    #     input_variable = en_valid_sentences[i]
    #     output_varible = de_valid_sentences[i]
    # for batch in valid_loader:
    #     input_variables, input_lengths = batch.src
    #     target_variables, target_lengths = batch.trg
    #     valid_loss = train(input_variables, input_lengths, target_variables, target_lengths, encoder, decoder,
    #                        encoder_optimizer, decoder_optimizer, criterion)
    #     valid_total_loss += valid_loss
    # print('Validation loss: %.4f' % (valid_total_loss / len(valid_dataset)))
        


# # Evaluating the model

# In[ ]:


def evaluate(sentence, max_length=MAX_LENGTH):
    input_variable = Variable(torch.LongTensor(scripts.text.to_id(sentence.split(), en_vocab)))
    print(input_variable)
    if use_cuda:
        input_variable = input_variable.cuda()
    
    input_length = len(input_variable)
    
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, [len(input_variable)], encoder_hidden)
    
    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[de_vocab['<s>']]]))
    decoder_hidden = encoder_hidden
    
    if use_cuda:
        decoder_input = decoder_input.cuda()
    
    decoded_words = []
    
    # Run through decoder
    for d_i in range(max_length):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden)
        # Pick most likely word index (highest value) from output (greedy search)
        top_value, top_index = decoder_output.data.topk(1)
        n_i = top_index[0][0]
        print(n_i)
        decoded_words += scripts.text.to_text([n_i], de_words)

        # Stop at end of sentence (not necessary when using known targers)
        if n_i == de_vocab['</s>']:
            break

        decoder_input = Variable(torch.LongTensor([[n_i]])) # Chosen word is next input

        if use_cuda:
            decoder_input = decoder_input.cuda()

            
    return decoded_words


# In[ ]:


def evaluate_sentence(s):
    valid_sentence = s
    
    output_words = evaluate(valid_sentence)
    output_sentence = ' '.join(output_words)
    
    print('>', valid_sentence)
    print('<', output_sentence)
    print('')


# In[ ]:


evaluate_sentence('i am a student and he is a teacher')


# In[ ]:


evaluate_sentence('luck is no excuse and who has luck is successful')

