'Train a sequence to sequence model.'


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


teacher_forcing_ratio = 0.5
clip = 5.0

def train(input_variable, target_variable, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, use_gpu=False):

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Get size of input and target sentences
    target_length = target_variable.size()[0]

    # Run words through encoder
    device = torch.device('cpu')
    if use_gpu:
        device = torch.device('cuda')
    encoder_hidden = encoder.init_hidden(device)
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([[SOS_token]])
    decoder_context = torch.zeros(1, decoder.hidden_size)
    decoder_hidden = encoder_hidden # Use last hidden state from encoder to start decoder
    if use_gpu:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    # Choose whether to use teacher forcing
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:

        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, _ = \
                 decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input

    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, _ = \
                decoder(decoder_input, decoder_context, decoder_hidden,
                        encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])

            # Get most likely word index (highest value) from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = torch.LongTensor([[ni]]) # Chosen word is next input
            if use_gpu:
                decoder_input = decoder_input.cuda()

            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token:
                break

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data.item() / target_length


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-size', type=int, default=500,
                        help='hidden layer size')
    parser.add_argument('--n-layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--dropout', type=float, default=0.05,
                        help='dropout probability')
    parser.add_argument('--lrate', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=50000,
                        help='number of epochs')
    parser.add_argument('--log-rate', type=int, default=1000,
                        help='logging rate')
    parser.add_argument('--use-gpu', action='store_true',
                        help='train on gpu')
    parser.add_argument('data', help='File of "tab" separated pairs')
    parser.add_argument('out_model', help='output model')
    parser.add_argument('out_langs', help='output lang')
    args = parser.parse_args()

    input_lang, output_lang, pairs = prepare_data(args.data)
    print('random pair:', random.choice(pairs))

    # Initialize models
    encoder = EncoderRNN(input_lang.n_words, args.hidden_size, args.n_layers)
    decoder = AttnDecoderRNN('general', args.hidden_size, output_lang.n_words,
                             args.n_layers, dropout_p=args.dropout)

    with open(args.out_model, 'wb') as fh:
        pickle.dump((encoder.cpu(), decoder.cpu()), fh)

    with open(args.out_langs, 'wb') as fh:
        pickle.dump((input_lang, output_lang), fh)

    if args.use_gpu:
        encoder.cuda()
        decoder.cuda()

    # Initialize optimizers and criterion
    learning_rate = 0.0001
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    start = time.time()
    print_loss_total = 0 # Reset every print_every
    print_loss_total = 0 # Reset every print_every

    # Begin!
    for epoch in range(1, args.epochs + 1):

        # Get training data for this cycle
        training_pair = variables_from_pair(random.choice(pairs), input_lang,
                                            output_lang)
        input_variable = training_pair[0]
        target_variable = training_pair[1]

        if args.use_gpu:
            input_variable = input_variable.cuda(), target_variable.cuda()

        # Run the train function
        loss = train(input_variable, target_variable, encoder, decoder,
                     encoder_optimizer, decoder_optimizer, criterion,
                     args.use_gpu)

        # Keep track of loss
        print_loss_total += loss

        if epoch == 0:
            continue

        if epoch % args.log_rate == 0:
            print_loss_avg = print_loss_total / args.log_rate
            print_loss_total = 0
            print_summary = '%s (%d %d%%) %.4f' % \
                (time_since(start, epoch / args.epochs), epoch, \
                epoch / args.epochs * 100, print_loss_avg)
            print(print_summary)

    with open(args.out_model, 'wb') as fh:
        pickle.dump((encoder.cpu(), decoder.cpu()), fh)

    with open(args.out_langs, 'wb') as fh:
        pickle.dump((input_lang, output_lang), fh)


if __name__ == '__main__':
    run()
