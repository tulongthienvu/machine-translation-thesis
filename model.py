from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class EncoderRNN(nn.Module):
    """
        Model's encoder using RNN.
    """

    def __init__(self, input_size, embedding_size, hidden_size, batch_size, num_layers=1, use_cuda=True):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(input_size + 1, embedding_size, padding_idx=1000)
        # self.embedding = nn.Embedding(input_size + 1, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.init_weights()
        self.batch_size = batch_size
        self.use_cuda = use_cuda

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
        hidden = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
        return hidden

    def init_weights(self):
        """
            Initialize weights.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)


class DecoderRNN(nn.Module):
    """
        Model's decoder using RNN.
    """

    def __init__(self, embedding_size, hidden_size, output_size, batch_size, num_layers=1, use_cuda=True):
        super(DecoderRNN, self).__init__()
        output_size += 1
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=1000)
        # self.embedding = nn.Embedding(output_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear_out = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.init_weights()

        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def forward(self, input_vector, hidden):
        # view(batch_size, sentence_length, -1)
        output = self.embedding(input_vector).view(self.batch_size, 1, -1)
        output = F.relu(output)
        # output = rnn_utils.pack_sequence(output)
        output, hidden = self.rnn(output, hidden)
        # print("Output: ", output)
        # output = self.log_softmax(self.linear_out(output[-1]))
        output = self.log_softmax(self.linear_out(output))  # Change dim for packed sequence
        output = output.view(self.batch_size, -1)
        # print("output final: ", output)
        return output, hidden

    def init_hidden(self):
        """
            Initialize hidden state.
        """
        hidden = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
        return hidden

    def init_weights(self):
        """
            Initialize weights.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear_out.weight.data.uniform_(-0.1, 0.1)
        self.linear_out.bias.data.fill_(0)