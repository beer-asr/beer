

import abc
from dataclasses import dataclass
import torch


__all__ = ['ConjugateLikelihood', 'ExponentialFamily',
           'ParametersView', 'kl_div']


# Error raised when the user is providing parameters with missing
# attribute.
class MissingParameterAttribute(Exception): pass

# Error raised when an "ExponentialFamily" subclass does not define the
# parameters of the distribution.
class UndefinedParameters(Exception): pass

# Error raised when an "ExponentialFamily" subclass does not define the
# standard parameter class of the distribution.
class UndefinedStdParametersClass(Exception): pass

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


class ParametersView(torch.nn.Module):
    '''Set of parameters own the memory of the parameters. Instead, they
    point to chunk of memory from another parameters.
    '''

    def from_natural_parameters(self, natural_parameters):
        return self.ref.from_natural_parameters(natural_parameters)

    def __init__(self, ref, names, idx):
        super().__init__()
        self.ref = ref
        self.idx = idx
        for name in names:
            def getter(self, name=name):
                return getattr(self.ref, name)[self.idx]
            setattr(self.__class__, name, property(fget=getter))
        self.names = names

    def __repr__(self):
        paramstr = ', '.join([f'{name}={getattr(self, name)}'
                              for name in self.names])
        clsname = self.ref.__class__.__qualname__
        return f'view<{clsname}({paramstr})>'


class ExponentialFamily(torch.nn.Module, metaclass=abc.ABCMeta):
    '''Base class to all distribution from the exponential family.

    Note that one instance of the exponential family can represent one
    or several distribution (of the same type). You can check how
    many distributions are handled by one instance by using:

    >>> len(myinstance)
    3

    '''

    # Sucbclasses need to define the parameters of the distribution
    # in a dictionary stored in a class variable named
    # "_std_params_def" and the standard parameters class in the
    # "_std_params_cls" variable. For example:
    #_std_params_cls = NormalStdParams
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
            raise UndefinedParameters(
                'Parameters of the distribution are undefined. ' \
                'You need to specify the field "_std_params_def" in ' \
                'your class definition.')
        if not hasattr(cls, '_std_params_cls'):
            raise UndefinedStdParametersClass(
                'Class of the standard parameters is not defined. ' \
                'You need to set the class variable "_std_params_cls" ' \
                'in your class definition.')

    def __init__(self, params):
        super().__init__()
        self.params = params

    @classmethod
    def from_std_parameters(cls, *args, **kwargs):
        params = cls._std_params_cls(*args, **kwargs)
        return cls(params)

    ####################################################################
    ## Subclass interface
    # To be implemented by subclasses in addition to the parameters'
    # definition.

    # The forward method is defined in the ``torch.nn.Module`` class.
    # If definied, it should compute the log-likelihood of the inputs
    # (X) given the parameters.
    #def forward(self, X):
    #    pass

    @abc.abstractmethod
    def __len__(self):
        pass

    def __getitem__(self, idx):
        names = tuple(self._std_params_def.keys())
        return self.__class__(ParametersView(self.params, names, idx))

    @abc.abstractmethod
    def conjugate(self):
        '''Returns a descriptor of the conjugate llh function object
        assiocated with the given pdf.
        '''
        pass

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


class ConjugateLikelihood(metaclass=abc.ABCMeta):
    'Base class for conjugate likelihood function.'

    @abc.abstractmethod
    def sufficient_statistics_dim(self, zero_stats=False):
        '''Dimension of the sufficient statistics of lihekihood
        function.

        Args:
            zero_stats (boolean): Include the zero order statistics
                as well.

        Returns:
            int: Dimension of the sufficient statistics.
        '''
        pass

    @abc.abstractmethod
    def sufficient_statistics(self, data):
        '''Sufficient statistics of the likelihood function for the given
        data.

        Args:
            data (``torch.Tensor[N, D]``): Input data.

        Returns:
            stats (``torch.Tensor[N, Q]``): Sufficient statistics.
        '''
        pass

    @abc.abstractmethod
    def pdfvectors_from_rvectors(self, rvecs):
        '''Transform real value vectors into equivalent pdf vectors
        using a default mapping (distribution dependent). For a pdf
        with natural parameters u and log-normalization function A(u)
        the pdf vector is defined as:

            pvec = (u, -A(u))^T

        Args:
            rvecs (``torch.Tensor[N,D]``): Real value vectors of
                dimenion D = self.conjugate_stats_dim

        Returns:
            pdfs (``torch.Tensor[N,Q]``): Equivalent pdf vectors.

        '''
        pass

    @abc.abstractmethod
    def parameters_from_pdfvector(self, pdfvec):
        '''Standard parameters of the likelihood function extracted from
        a pdf vector.
        '''
        pass

    @abc.abstractmethod
    def __call__(self, natural_parameters, stats):
        '''Compute the log likelihood of the data (represented as
        sufficient statistics) given the natural parameters.
        '''
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
    return lnorm2 - lnorm1 - torch.sum(exp_stats * (nparams2 - nparams1), dim=-1)

