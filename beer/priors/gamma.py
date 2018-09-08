'''Implementation of the Gamma distribution.'''

import torch
from .baseprior import ExpFamilyPrior


class GammaPrior(ExpFamilyPrior):
    '''Gamma distribution.

    parameters:
        a: shape
        b: rate

    natural parameters:
        eta1 = -b
        eta2 = a - 1

    sufficient statistics:
        T_1(x) = x
        T_2(x) = ln x

    '''
    __repr_str = '{classname}(shape={shape}, rate={rate})'

    def __init__(self, shape, rate):
        nparams = self.to_natural_parameters(shape, rate)
        super().__init__(nparams)

    def __repr__(self):
        shape, rate = self.to_std_parameters(self.natural_parameters)
        return self.__repr_str.format(
            classname=self.__class__.__name__,
            shape=repr(shape), rate=repr(rate)
        )

    def expected_value(self):
        shape, rate = self.to_std_parameters(self.natural_parameters)
        return shape / rate

    def to_natural_parameters(self, shape, rate):
        return torch.cat([-rate.view(1), (shape - 1).view(1)])

    def _to_std_parameters(self, natural_parameters):
        shape, rate = natural_parameters[1] + 1, -natural_parameters[0]
        return  shape, rate

    def _expected_sufficient_statistics(self):
        shape, rate = self.to_std_parameters(self.natural_parameters)
        return torch.cat([(shape / rate).view(1),
                          (torch.digamma(shape) - torch.log(rate)).view(1)])

    def _log_norm(self, natural_parameters=None):
        if natural_parameters is None:
            natural_parameters = self.natural_parameters
        shape, rate = self.to_std_parameters(natural_parameters)
        return torch.lgamma(shape) - shape * torch.log(rate)


__all__ = ['GammaPrior']
