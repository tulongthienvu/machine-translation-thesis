import torch

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
