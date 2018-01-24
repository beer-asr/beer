
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

    def fit(self, data, mini_batch_size=-1, max_epochs=1, seed=None,
            lrate=1., callback=None):
        data_size= np.prod(data.shape[:-1])

        mb_size = mini_batch_size if mini_batch_size > 0 else len(data)
        for epoch in range(1, max_epochs + 1):
            for mini_batch in mini_batches(data, mb_size, seed):
                mini_batch_size = np.prod(mini_batch.shape[:-1])
                scale = float(data_size) / mini_batch_size

                exp_llhs, acc_stats = self.exp_llh(mini_batch, accumulate=True)
                exp_llh = np.sum(exp_llhs)
                kld = self.kl_div_posterior_prior()
                lower_bound = (scale * exp_llh - kld)
                self.natural_grad_update(acc_stats, scale, lrate)

                lower_bound = lower_bound / data_size
                llh = exp_llh / data_size
                kld = kld / data_size


                if callback is not None:
                    callback(lower_bound, llh, kld)

