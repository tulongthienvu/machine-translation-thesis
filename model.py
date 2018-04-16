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

    def __init__(self, input_size, embedding_size, hidden_size, num_layers=1, dropout_p=0.0, bidirectional=False, use_cuda=True):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional

        input_size += 1
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=50000)
        # self.embedding = nn.Embedding(input_size + 1, embedding_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # self.rnn = nn.GRU(embedding_size, hidden_size, num_layers, batch_first=True)
        if self.bidirectional:
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.init_weights()
        self.use_cuda = use_cuda

    def forward(self, input_sentences, input_lengths, hidden):
        #         sentence_len = len(input_sentences)
        longest_length = input_lengths[0]
        output = self.embedding(input_sentences)
        output = self.dropout(output)
        output = rnn_utils.pack_padded_sequence(output, input_lengths, batch_first=True)
        #         embedded = embedded.view(sentence_len, batch_size, -1)
        #         embedded = embedded.view(batch_size, longest_length, -1)
        output, (hidden, cell) = self.lstm(output, hidden)
        # print(output)
        return output, (hidden, cell)

    def init_hidden(self, actual_batch_size):
        """
            Initialize hidden state.
        """
        # hidden = Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size))
        if self.bidirectional:
            num_direction = 2
        else:
            num_direction = 1
        hidden = Variable(torch.zeros(self.num_layers * num_direction, actual_batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.num_layers * num_direction, actual_batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)

    def init_weights(self):
        """
            Initialize weights.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)


class DecoderRNN(nn.Module):
    """
        Model's decoder using RNN.
    """

    def __init__(self, embedding_size, hidden_size, output_size, num_layers=1, dropout_p=0.0, bidirectional=False, use_cuda=True):
        super(DecoderRNN, self).__init__()
        output_size += 1
        self.embedding = nn.Embedding(output_size, embedding_size, padding_idx=50000)
        # self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(self.dropout_p)
        # self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        if self.bidirectional:
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        else:
            self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        if self.bidirectional:
            self.linear_out = nn.Linear(hidden_size * 2, output_size)
        else:
            self.linear_out = nn.Linear(hidden_size, output_size)

        self.log_softmax = nn.LogSoftmax(dim=2)
        self.init_weights()

        self.use_cuda = use_cuda

    def forward(self, input_vector, hidden):
        # view(batch_size, sentence_length, -1)
        # output = self.embedding(input_vector).view(self.batch_size, 1, -1)
        output = self.embedding(input_vector).view(len(input_vector), 1, -1)
        output = F.relu(output)
        output = self.dropout(output)
        # output = rnn_utils.pack_sequence(output)
        output, (hidden, cell) = self.lstm(output, hidden)
        # print("Output: ", output)
        # output = self.log_softmax(self.linear_out(output[-1]))
        output = self.log_softmax(self.linear_out(output))  # Change dim for packed sequence
        output = output.view(len(input_vector), -1)
        # print("output final: ", output)
        return output, (hidden, cell)

    def init_hidden(self, actual_batch_size):
        """
            Initialize hidden state.
        """
        if self.bidirectional:
            num_direction = 2
        else:
            num_direction = 1
        hidden = Variable(torch.zeros(self.num_layers * num_direction, actual_batch_size, self.hidden_size))
        cell = Variable(torch.zeros(self.num_layers * num_direction, actual_batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
            cell = cell.cuda()
        return (hidden, cell)

    def init_weights(self):
        """
            Initialize weights.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear_out.weight.data.uniform_(-0.1, 0.1)
        self.linear_out.bias.data.fill_(0)


class Seq2Seq(nn.Module):
    """
    Sequence to Sequence model that contains encoder, decoder.
    """

    def __init__(self, encoder, decoder):
        """
        Constructor.
        :param encoder: encoder model for Seq2Seq
        :param decoder: decoder model for Seq2Seq
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_sentences, input_lengths, target):
        """

        :param input_sentences:
        :param input_lengths:
        :param target:
        :return:
        """
        encoder_outputs, encoder_hidden = self.encoder(input_sentences, input_lengths)
        decoder_input = torch.zeros((len(input_variables), 1)).type(torch.LongTensor)
        decoder_input += de_vocab['<s>']
        decoder_hidden = encoder_hidden  # Use last hidden from the encoder