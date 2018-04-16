
import sys
import torch

import torch.nn.functional as F
from torch.autograd import Variable
from collections import Counter
import math
import numpy as np
import subprocess

MAX_LENGTH = 50


def evaluate(data_loader, encoder, decoder, criterion, en_vocab, de_vocab, max_lenth=MAX_LENGTH, use_cuda=True):
    # Zero gradient
    i = 0
    total_loss = 0
    for batch in data_loader:
        input_variables, input_lengths = batch.src
        target_variables, target_lengths = batch.trg
        # print(i)
        i += 1
        loss = 0

        # Run words through encoder
        encoder_hidden = encoder.init_hidden(len(input_variables))
        encoder_outputs, encoder_hidden = encoder(input_variables, input_lengths, encoder_hidden)
        #
        # Prepare input for decoder and output variables
        decoder_input = torch.zeros((len(input_variables), 1)).type(torch.LongTensor)
        decoder_input += de_vocab['<s>']
        decoder_hidden = encoder_hidden  # Use last hidden from the encoder

        if use_cuda:
            decoder_input = decoder_input.cuda()
        # Transpose target variables for mini-batch implement
        target_variables.t_()
        # Find max target lengths
        max_target_length = max(target_lengths)
        for d_i in range(max_target_length):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variables[d_i])
            # Pick most likely word index (highest value) from output (greedy search)
            top_value, top_index = decoder_output.data.topk(1)
            n_i = top_index
            # Modify for mini-batch implement
            decoder_input = torch.zeros((len(input_variables), 1)).type(torch.LongTensor)
            if use_cuda:  # not optimized yet
                decoder_input = decoder_input.cuda()
            decoder_input += n_i

            # Stop at end of sentence (not necessary when using known targers)
            if n_i[0] == de_vocab['</s>']:
                break
        # Convert Cuda Tensor to Float for saving VRAM
        total_loss += float(loss.data[0] / (max_target_length / data_loader.batch_size))
    # print(len(data_loader))
    return total_loss


def decode_minibatch(input_variables, target_variables, encoder, decoder):
    """
    Decode a minibatch.
    :param input_variables:
    :param target_variables:
    :param encoder:
    :param decoder:
    :return:
    """


def evaluate_model(data_loader, encoder, decoder, criterion, en_vocab, de_vocab, max_lenth=MAX_LENGTH, use_cuda=True):
    """
    Evaluate model by BLEU score.
    :param data_loader:
    :param encoder:
    :param decoder:
    :param criterion:
    :param en_vocab:
    :param de_vocab:
    :param max_lenth:
    :param use_cuda:
    :return:
    """
    preds = []
    ground_truths = []
    for batch in data_loader:
        # Get source and target mini-batch
        input_variables, input_lengths = batch.src
        target_variables, target_lengths = batch.trg

    # Initialize target with <s> for every sentence



"""
    This BLEU computing code is based on https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/evaluate.py
"""

def bleu_stats(hypothesis, reference):
    """
    Compute statistics for BLEU.
    :param hypothesis: predicted sentence
    :param reference: reference (target) sentence
    :return: BLEU stats
    """
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    # 1-5 grams
    for n in xrange(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + 1]) for i in xrange(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """
    Compute BLEU given n-gram statistics.
    :param stats: n-gram statistics for BLEU
    :return: BLEU score
    """
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, references):
    """
    Get validation BLEU score for validation set.
    :param hypotheses: predicted sentences
    :param references: reference sentences
    :return:
    """
    # stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    stats = np.zeros(10)
    for hyp, ref in zip(hypotheses, references):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


