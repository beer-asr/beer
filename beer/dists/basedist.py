

import abc
from dataclasses import dataclass
import torch


__all__ = ['ExponentialFamily', 'kl_div']


# Error raised when the user is providing parameters object with a
#  missing attribute.
class MissingParameterAttribute(Exception): pass

# Error raised when a "ExponentialFamily" subclass does not define the
# parameters of the distribution.
class UndefinedParameters(Exception): pass

# Error raised when trying to compute the KL-divergence between two
# distribution of different type (i.e. Dirichlet and Normal).
class DistributionTypeMismatch(Exception): pass

# Error raised when trying to compute the KL-divergence between two
# distribution of the same type but with different support dimension.
class SupportDimensionMismatch(Exception): pass


# Check if a parameter object has a specific attribute.
def _check_params_have_attr(params, attrname):
    if not hasattr(params, attrname):
        raise MissingParameterAttribute(
                    f'Parameters have no "{attrname}" attribute')


class ExponentialFamily(torch.nn.Module, metaclass=abc.ABCMeta):
    'Base class to all distribution from the exponential family.'

    # Sucbclasses need to define the parameters of the distribution
    # in a dictionary stored in a class variable named
    # "_std_params_def". For example:
    #_std_params_def = {
    #
    #      +---------------------------- Parameter's name which will be
    #      |                             a read-only attribute of the
    #      |                             subclass.
    #      |
    #      |             +-------------- Documentation of the parameter
    #      |             |               which will be converted into
    #      |             |               the docstring of the attribute.
    #      v             v
    #    'mean': 'Mean parameter.',
    #    'var': 'Variance parameter.'
    #}

    def __init_subclass__(cls):
        if not hasattr(cls, '_std_params_def'):
            raise UndefinedParameters('Parameters of the distribution are ' \
                                      'undefined. You need to specify the ' \
                                      'field "_std_params_def" in your ' \
                                      'class definition.')


    def __init__(self, params):
        super().__init__()
        self.params = params
        for param_name, param_doc in self._std_params_def.items():
            _check_params_have_attr(params, param_name)
            getter = lambda self, name=param_name: getattr(self.params, name)
            setattr(self.__class__, param_name, property(getter, doc=param_doc))

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            for param_name in self._std_params_def:
                param0 = getattr(self, param_name)
                param1 = getattr(other, param_name)
                if param0.shape != param1.shape:
                    # Check if the dimension matches.
                    return False
                elif not torch.allclose(param0, param1):
                    # Check if the parameters are the same.
                    return False
            else:
                return True
        return NotImplemented

    # Even if two instances are equal (have the same parameters) they
    # will have different hash and considered to be unique. This is
    # necessary as when training a model, two distribution may be equal
    # (at initialization for instance) but we need different hash so the
    # optimizer don't confuse them.
    def __hash__(self):
        return hash(id(self))

    ####################################################################
    ## Subclass interface
    # To be implemented by subclasses in addition to the parameters'
    # definition.

    # The forward method is defined in the ``torch.nn.Module`` class.
    # should compute the log-likelihood of the inputs (X) given
    # the parameters.
    #def forward(self, X):
    #    pass

    @property
    @abc.abstractmethod
    def dim(self):
        'Dimension of the support.'
        pass

    @abc.abstractmethod
    def expected_sufficient_statistics(self):
        'Expectation of the sufficient statistics.'
        pass

    @abc.abstractmethod
    def expected_value(self):
        'Expected value of the random variable.'
        pass

    @abc.abstractmethod
    def log_norm(self):
        'Log-normalization constant given the current parameters.'
        pass

    @abc.abstractmethod
    def sample(self, nsamples):
        'Draw values of the random variable given the current parameters.'
        pass

    @abc.abstractmethod
    def natural_parameters(self):
        'Natural/canonical form of the current parameterization.'
        pass

    @abc.abstractmethod
    def update_from_natural_parameters(self):
        'Uptate the parameters given the new natural parameters.'
        pass


def kl_div(pdf1, pdf2):
    '''KL-divergence between two exponential family members of the same
    type and the same support.

    '''
    if pdf1.__class__ is not pdf2.__class__:
        raise DistributionTypeMismatch(
            'Cannot compute KL-divergence between distribution of ' \
            f'different type ({pdf1.__class__} != {pdf2.__class__})')

    if pdf1.dim != pdf2.dim:
        raise SupportDimensionMismatch(
            'Cannot compute KL-divergence between distribution with '\
            f'different support dimension ({pdf1.dim} != {pdf2.dim})')

    nparams1 = pdf1.natural_parameters()
    nparams2 = pdf2.natural_parameters()
    exp_stats = pdf1.expected_sufficient_statistics()
    lnorm1 = pdf1.log_norm()
    lnorm2 = pdf2.log_norm()
    return lnorm2 - lnorm1 - exp_stats @ (nparams2 - nparams1)

