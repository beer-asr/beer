
'''Abstract Base Class for a model.'''

import abc
import numpy as np


log_fmt = 'ln p(X) = {0: <12,.3f} E[ln p(X|...)] = {1: <12,.3f} ' \
          'D(q || p) = {2:.3f}'



def _mini_batches(data, mini_batch_size, seed=None):
    rng = np.random.RandomState()
    if seed is not None:
        rng.seed(seed)
    indices = rng.choice(data.shape[0], size=data.shape[0], replace=False)
    splits = np.array_split(indices, data.shape[0] // mini_batch_size)
    for split in splits:
        yield data[split]


class Model(metaclass=abc.ABCMeta):
    """Abstract Base Class for all beer's model."""

    @abc.abstractmethod
    def _fit_step(self, mini_batch):
        NotImplemented

    def fit(self, data, mini_batch_size=-1, max_epochs=1, seed=None):
        '''Fit the model to the data using Variational Bayes training.

        Args:
            data (numpy.ndarray): Training data.
            mini_batch_size (int): Size of the mini-batches. If 0 or
                negative, the size of the mini-batches will be the size
                of the whole data set.
            max_epochs (int): Maximum number of epochs.
            seed (integer): Seed the random generator. For the same
                seed number, the iteration over the mini-batches will
                be exactly the same (does not affect the model itself).

        '''
        mb_size = mini_batch_size if mini_batch_size > 0 else len(data)
        for epoch in range(1, max_epochs + 1):
            for mini_batch in _mini_batches(data, mb_size, seed):
                lower_bound, llh, kld = self._fit_step(mini_batch)
            print('ln p(x) >=', lower_bound[0], llh[0], kld[0])


class ConjugateExponentialModel(Model, metaclass=abc.ABCMeta):
    '''Abstract base class for Conjugate Exponential models.'''

    def _fit_step(self, mini_batch):
        mini_batch_size = np.prod(mini_batch.shape[:-1])
        scale = float(self._fit_cache['data_size']) / mini_batch_size
        exp_llhs, acc_stats = self.exp_llh(mini_batch, accumulate=True)
        exp_llh = np.sum(exp_llhs)
        kld = self.kl_div_posterior_prior()
        lower_bound = (scale * exp_llh - kld)
        self.natural_grad_update(acc_stats, scale, self._fit_cache['lrate'])
        return lower_bound, exp_llh, kld

    def fit(self, data, mini_batch_size=-1, max_epochs=1, seed=None,
            lrate=1.):
        self._fit_cache = {
            'lrate': lrate,
            'data_size': np.prod(data.shape[:-1])
        }
        super().fit(data, mini_batch_size, max_epochs,seed)

