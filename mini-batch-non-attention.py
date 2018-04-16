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

# Import library

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
from model import EncoderRNN, DecoderRNN
from torch.optim.lr_scheduler import MultiStepLR
from evaluate import evaluate

import numpy as np

import time
import math
import random
import unicodedata
import string
import re
import gc

import scripts.text
import utils

torch.backends.cudnn.enabled = False

use_cuda = True
batch_size = 64
batch_size_valid = 1
learning_rate = 1.0

# # Load data

data_path = './processed-data/50k_vocab/'
en_vocab_path = data_path + 'train.10k.en.vocab'
de_vocab_path = data_path + 'train.10k.de.vocab'

en_words, en_vocab, _ = scripts.text.load_vocab(en_vocab_path)
de_words, de_vocab, _ = scripts.text.load_vocab(de_vocab_path)


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
                    examples.append(torchtext.data.Example.fromlist([src_line, trg_line], fields))

        super(LuongNMTDataset, self).__init__(examples, fields, **kwargs)


def post_processing(arr, field_vocab, train):
    for index in range(0, len(arr)):
        arr[index] = map(int, arr[index])
    return arr


src_field = torchtext.data.Field(sequential=True,
                                 #                                  tokenize=(lambda line: int(line)),
                                 postprocessing=post_processing,
                                 use_vocab=False,
                                 pad_token='50000',
                                 include_lengths=True,
                                 batch_first=True,
                                 )
trg_field = torchtext.data.Field(sequential=True,
                                 #                                  tokenize=(lambda line: int(line)),
                                 postprocessing=post_processing,
                                 use_vocab=False,
                                 include_lengths=True,
                                 pad_token='50000',
                                 batch_first=True
                                 )

train_dataset = LuongNMTDataset(src_path=data_path + 'train.10k.en',
                                trg_path=data_path + 'train.10k.de',
                                fields=(src_field, trg_field)
                                )

train_loader = torchtext.data.BucketIterator(dataset=train_dataset,
                                             batch_size=batch_size,
                                             repeat=False,
                                             shuffle=True,
                                             sort_within_batch=True,
                                             sort_key=lambda x: len(x.src)
                                             )

valid_dataset = LuongNMTDataset(src_path=data_path + 'valid.100.en',
                                trg_path=data_path + 'valid.100.de',
                                fields=(src_field, trg_field),
                                MAX_LENGTH=50
                                )

valid_loader = torchtext.data.BucketIterator(dataset=valid_dataset,
                                             batch_size=batch_size_valid,
                                             repeat=False,
                                             shuffle=False,
                                             sort_within_batch=True,
                                             sort_key=lambda x: len(x.src)
                                             )

# # Training model

# ## Using RNN
teacher_forcing_ratio = 0.5
clip = 5.0
MAX_LENGTH = 50


def train(input_variables, input_lengths, target_variables, target_lengths, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    # Zero gradient
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0.0

    # Run words through encoder
    encoder_hidden = encoder.init_hidden(len(input_variables))
    encoder_outputs, encoder_hidden = encoder(input_variables, input_lengths, encoder_hidden)

    # Prepare input for decoder and output variables
    # decoder_input = torch.zeros((batch_size, 1), ).type(torch.LongTensor)
    decoder_input = torch.zeros((len(input_variables), 1)).type(torch.LongTensor)
    decoder_input += de_vocab['<s>']
    decoder_hidden = encoder_hidden  # Use last hidden from the encoder

    if use_cuda:
        decoder_input = decoder_input.cuda()
    # Transpose target variables for mini-batch implement
    target_variables = target_variables.t()
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
            loss += criterion(decoder_output, target_variables[d_i])
            # Pick most likely word index (highest value) from output (greedy search)
            top_value, top_index = decoder_output.data.topk(1)
            n_i = top_index
            # Modify for mini-batch implement
            # decoder_input = torch.zeros((batch_size, 1), ).type(torch.LongTensor)
            decoder_input = torch.zeros((len(input_variables), 1)).type(torch.LongTensor)
            if use_cuda:  # not optimized yet
                decoder_input = decoder_input.cuda()
            decoder_input += n_i

            # Stop at end of sentence (not necessary when using known targers)
            # if n_i == en_vocab['</s>']:
            #     break

    # Backpropagation
    loss.backward(retain_graph=False)
    # loss.backward()
    nn.utils.clip_grad_norm(encoder.parameters(), clip)
    nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    # Convert Cuda Tensor to Float for saving VRAM
    # gc.collect()
    return loss.data[0] / max_target_length


# ### Run training

# Set hyperparameters
embedding_size = 1000
hidden_size = 500
num_layers = 4
dropout_p = 0.2
bidirectional = True

# Initialize models
encoder = EncoderRNN(len(en_vocab), embedding_size, hidden_size, num_layers, dropout_p=dropout_p, bidirectional=bidirectional)
decoder = DecoderRNN(embedding_size, hidden_size, len(de_vocab), num_layers, dropout_p=dropout_p, bidirectional=bidirectional)

# Move models to GPU
if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# Initialize parameters and criterion
# learning_rate = 0.0001
encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=learning_rate)
# encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
# decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss(ignore_index=50000)
# criterion = nn.NLLLoss()


# Configuring training
num_epochs = 12
plot_every = 100
print_every = 50
encoder_scheduler = MultiStepLR(encoder_optimizer, milestones=[8, 9, 10, 11, 12], gamma=0.5)
decoder_scheduler = MultiStepLR(decoder_optimizer, milestones=[8, 9, 10, 11, 12], gamma=0.5)

# Keep track of time elapsed and running averages
plot_losses = []
print_loss_total = 0.0  # Reset every print every
plot_loss_total = 0.0  # Reset every plot every

# Running training
start = time.time()
for epoch in range(0, num_epochs):
    encoder_scheduler.step()
    decoder_scheduler.step()
    # for param_group in encoder_optimizer.param_groups:
    #     print(param_group['lr'])
    # for param_group in decoder_optimizer.param_groups:
    #     print(param_group['lr'])
    # print("*******")
    # gc.collect()
    # start epoch
    # Shuffle
    prev_step = 1
    step = 1
    num_steps = math.ceil(len(train_dataset) / batch_size)

    for batch in train_loader:
        input_variables, input_lengths = batch.src
        target_variables, target_lengths = batch.trg

        loss = float(train(input_variables, input_lengths, target_variables, target_lengths, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion))
        print_loss_total += loss
        # plot_loss_total += loss
        # print(print_loss_total)
        # if step == 0:
        #     step += 1
            #             continue
            # break

        if step % print_every == 0 or step == num_steps:
            print_loss_avg = print_loss_total / (step - prev_step)
            prev_step = step
            print_loss_total = 0
            print_summary = 'Epoch %s/%s, Time: %s, Step: %d/%d, train_loss: %.4f' % (epoch + 1, num_epochs,
                                                                                      utils.time_since(start,
                                                                                                       step / num_steps),
                                                                                      step,
                                                                                      num_steps, print_loss_avg)
            print(print_summary)

        # if step % plot_every == 0:
        #     plot_loss_avg = plot_loss_total / plot_every
        #     plot_losses.append(plot_loss_avg)
        #     plot_loss_total
        step += 1

        # stop when reaching certain steps
        # if step == 100:
        #     break

    # end epoch
    # evaluate on validation set
    valid_total_loss = evaluate(valid_loader, encoder, decoder, criterion, en_vocab, de_vocab)
    print('Validation loss: %.4f' % (valid_total_loss / len(valid_dataset)))
    # gc.collect()



# # Evaluating the model

# def evaluate(sentence, max_length=MAX_LENGTH):
#     input_variable = Variable(torch.LongTensor(scripts.text.to_id(sentence.split(), en_vocab)))
#     print(input_variable)
#     if use_cuda:
#         input_variable = input_variable.cuda()
#
#     input_length = len(input_variable)
#
#     encoder_hidden = encoder.init_hidden()
#     encoder_outputs, encoder_hidden = encoder(input_variable, [len(input_variable)], encoder_hidden)
#
#     # Create starting vectors for decoder
#     decoder_input = Variable(torch.LongTensor([[de_vocab['<s>']]]))
#     decoder_hidden = encoder_hidden
#
#     if use_cuda:
#         decoder_input = decoder_input.cuda()
#
#     decoded_words = []
#
#     # Run through decoder
#     for d_i in range(max_length):
#         decoder_output, decoder_hidden = decoder(
#             decoder_input, decoder_hidden)
#         # Pick most likely word index (highest value) from output (greedy search)
#         top_value, top_index = decoder_output.data.topk(1)
#         n_i = top_index[0][0]
#         print(n_i)
#         decoded_words += scripts.text.to_text([n_i], de_words)
#
#         # Stop at end of sentence (not necessary when using known targers)
#         if n_i == de_vocab['</s>']:
#             break
#
#         decoder_input = Variable(torch.LongTensor([[n_i]]))  # Chosen word is next input
#
#         if use_cuda:
#             decoder_input = decoder_input.cuda()
#
#     return decoded_words
#
# def evaluate_sentence(s):
#     valid_sentence = s
#
#     output_words = evaluate(valid_sentence)
#     output_sentence = ' '.join(output_words)
#
#     print('>', valid_sentence)
#     print('<', output_sentence)
#     print('')
#
#
# evaluate_sentence('i am a student and he is a teacher')
#
# evaluate_sentence('luck is no excuse and who has luck is successful')
