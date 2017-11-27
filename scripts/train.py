
"""Train a Bayesian PPCA model."""

import argparse
import numpy as np
import pickle
import ppca

logging_fmt = 'epoch: {epoch: >#e}/{epochs}    batch: {batch: >#b}/' \
              '{n_batches}   avg. ln p(x) >= {elbo:.3f}'


def callback(model, epoch, epochs, batch, n_batches, elbos, logging_rate,
             update_count):
    # Log the progression of the training.
    if update_count % logging_rate == 0:
        fmt = logging_fmt.replace('#e', str(1 + int(np.floor(np.log10(epochs)))))
        fmt = fmt.replace('#b', str(1 + int(np.floor(np.log10(n_batches)))))
        print(fmt.format(epoch=epoch, epochs=epochs, batch=batch,
              n_batches=n_batches, elbo=elbos[-1]))

    return False

def train(model, data, callback, epochs, batch_size, lrate, logging_rate):
    previous_elbo = - np.inf
    burn_in_phase = True
    stochastic = True
    if batch_size == -1:
        stochastic = False
        batch_size = data.shape[1]
    n_batches = data.shape[1] // batch_size
    elbos = []
    update_count = 0
    epoch = 0
    has_converged = False

    while epoch <  epochs and not has_converged:
        epoch += 1

        # Shuffle the data.
        np.random.shuffle(data.T)

        for batch in range(n_batches):
            # Get the data of the minibatch
            start, end = batch * batch_size, (batch + 1) * batch_size
            X = data[:, start: end]

            # Accumulate the stats to update the model.
            exp_llh, acc_stats = model.exp_llh(X, accumulate_stats=True)

            # Compute the scale of the statistics.
            scale = data.shape[1] / X.shape[1]

            # Compute the lower-bound.
            kl_latent, kl = model.kl_div(X)
            elbo = scale * (exp_llh - kl_latent) - kl
            elbo /= data.shape[1]

            # Update the parameters.
            if stochastic:
                if burn_in_phase and elbo > previous_elbo:
                    lr = 1.
                else:
                    #lr = 1 / (update_count + 1)
                    lr = lrate * (1. / epoch)
                    burn_in_phase = False
            else:
                lr = lrate
            model.natural_gradient_update(acc_stats, scale, lrate=lr)
            update_count += 1

            # Store the progression of the lower-bound.
            elbos.append(elbo)
            previous_elbo = elbo

            # Give hand to the caller to let it monitor the
            # convergence.
            has_converged = callback(model, epoch, epochs, batch + 1,
                                     n_batches, elbos, logging_rate,
                                     update_count)

    return elbos


def run():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save-stats',
                        help='save the progress of the training to a file')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to run the training')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size for each update')
    parser.add_argument('--learning-rate', type=float, default=1.,
                        help='learning rates')
    parser.add_argument('--logging-rate', type=int, default=1,
                        help='how frequently to log the progress')
    parser.add_argument('in_model', help='model to train')
    parser.add_argument('data', help='data to train on in numpy binary '
                                     'format')
    parser.add_argument('out_model', help='output trained model')
    args = parser.parse_args()

    # Load the model.
    with open(args.in_model, 'rb') as fid:
        model = pickle.load(fid)

    # Load the data.
    data = np.load(args.data)

    # Train the model.
    elbos =  train(
        model,
        data,
        callback,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lrate=args.learning_rate,
        logging_rate=args.logging_rate
    )



    # Save it on disk.
    with open(args.out_model, 'wb') as fid:
        pickle.dump(model, fid)

    if args.save_stats:
        with open(args.save_stats, 'wb') as fid:
            pickle.dump({'elbos': elbos}, fid)


if __name__ == '__main__':
    run()

