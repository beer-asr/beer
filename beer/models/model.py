
'''Abstract Base Class for a model.'''

import abc
import numpy as np


log_fmt = 'ln p(X) = {0: <12,.3f} E[ln p(X|...)] = {1: <12,.3f} ' \
          'D(q || p) = {2:.3f}'


class ConjugateExponentialModel(metaclass=abc.ABCMeta):
    '''Abstract base class for "Conjugate Exponential model".'''


    def fit(self, data, minibatch_size=-1, max_epochs=2, threshold=1e-6,
            lrate=1., verbose=False):
        '''Fit the model to the data using Variational Bayes training.

        Args:
            data (numpy.ndarray): Training data.
            minibatch_size (int): Size of the minibatches.
            max_epochs (int): Maximum number of epochs.
            threshold (float): Convergence threshold.
            lrate (float): Learning rate.
            verbose (boolean): Print training progress.

        '''
        bsize = minibatch_size if minibatch_size > 0 else len(data)
        epoch = 0
        has_converged = False
        previous_lower_bound = -np.inf

        # Iterate until convergence or the maximum number of epochs is
        # reached.
        while epoch < max_epochs and not has_converged:
            # Evalute the expected value of the log-likelihood.
            exp_llhs, acc_stats = self.exp_llh(data, accumulate=True)

            # Scale of the sufficient statistics.
            scale = float(len(data)) / bsize

            # Evaluate the lower-bound of the data.
            exp_llh = np.sum(exp_llhs)
            kl = self.kl_div_posterior_prior()
            lower_bound = (scale * exp_llh - kl) / len(data)

            if verbose:
                print(log_fmt.format(lower_bound, exp_llh / bsize,
                      kl / bsize))

            # Monitor convergence.
            if abs(previous_lower_bound - lower_bound) <= threshold:
                has_converged = True
            previous_lower_bound = lower_bound

            # Update the parameters.
            self.natural_grad_update(acc_stats, scale, lrate)

            epoch += 1

