'Map a sequence to another sequence.'


import argparse
import math
import pickle
import random
import sys
import time

# Assume script is called from ../../
sys.path.insert(0, 'utils/seq2seq')

from seq2seq import *
from torch import optim


def evaluate(encoder, decoder, sentence, input_lang, output_lang, use_gpu=False,
             max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)

    # Run through encoder
    device = torch.device('cpu')
    if use_gpu:
        device = torch.device('cuda')
    encoder_hidden = encoder.init_hidden(device)
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([[SOS_token]]) # SOS
    decoder_context = torch.zeros(1, decoder.hidden_size)
    if use_gpu:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        _, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[int(ni)])

        # Next input is chosen word
        decoder_input = torch.LongTensor([[ni]])
        if use_gpu: decoder_input = decoder_input.cuda()

    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-gpu', action='store_true',
                        help='train on gpu')
    parser.add_argument('model', help='mapping model')
    parser.add_argument('langs', help='mapping langs')
    args = parser.parse_args()

    with open(args.model, 'rb') as fh:
        encoder, decoder = pickle.load(fh)
    encoder.eval(), decoder.eval()

    with open(args.langs, 'rb') as fh:
        input_lang, output_lang = pickle.load(fh)

    for line in sys.stdin:
        out_seq, attn = evaluate(encoder, decoder, line.strip(), input_lang,
                                 output_lang, args.use_gpu)
        print(' '.join(out_seq[:-1]))


if __name__ == '__main__':
    run()
