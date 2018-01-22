
'''Abstract Base Class for a model.'''

import abc
import numpy as np
from ..training import mini_batches


class Model(metaclass=abc.ABCMeta):
    """Abstract Base Class for all beer's model."""

    @abc.abstractmethod
    def _fit_step(self, mini_batch):
        NotImplemented


class ConjugateExponentialModel(metaclass=abc.ABCMeta):
    '''Abstract base class for Conjugate Exponential models.'''

    def _fit_step(self, mini_batch):
        mini_batch_size = np.prod(mini_batch.shape[:-1])
        scale = float(self._fit_cache['data_size']) / mini_batch_size
        exp_llhs, acc_stats = self.exp_llh(mini_batch, accumulate=True)
        exp_llh = np.sum(exp_llhs)
        kld = self.kl_div_posterior_prior()
        lower_bound = (scale * exp_llh - kld)
        self.natural_grad_update(acc_stats, scale, self._fit_cache['lrate'])
        return lower_bound / self._fit_cache['data_size'], \
            exp_llh / self._fit_cache['data_size'], \
            kld / self._fit_cache['data_size']

    def fit(self, data, mini_batch_size=-1, max_epochs=1, seed=None,
            lrate=1., callback=None):
        self._fit_cache = {
            'lrate': lrate,
            'data_size': np.prod(data.shape[:-1])
        }

        mb_size = mini_batch_size if mini_batch_size > 0 else len(data)
        for epoch in range(1, max_epochs + 1):
            for mini_batch in mini_batches(data, mb_size, seed):
                lower_bound, llh, kld = self._fit_step(mini_batch)

                if callback is not None:
                    callback(lower_bound, llh, kld)

