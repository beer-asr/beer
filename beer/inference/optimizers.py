import warnings


__all__ = ['BayesianModelOptimizer', 'VariationalBayesOptimizer']


def BayesianModelOptimizer(*args, **kwargs):
    warnings.warn('"BayesianModelOptimizer" is deprecated. Used ' \
                  'VariationalBayesOptimizer" instead.', DeprecationWarning,
                  stacklevel=2)
    return VariationalBayesOptimizer(*args, **kwargs)


class VariationalBayesOptimizer:
    '''Generic optimizer for :any:`BayesianModel` subclasses.

    Args:
        parameters (list): List of :any:`BayesianParameter`.
        lrate (float): Learning rate for the :any:`BayesianParameter`.
        std_optim (``torch.Optimizer``): pytorch optimizer.

    '''

    def __init__(self, groups, lrate=1., std_optim=None):
        '''
        Args:
            parameters (list): List of ``BayesianParameters``.
            lrate (float): learning rate.
            std_optim (``torch.optim.Optimizer``): Optimizer for
                non-Bayesian parameters (i.e. standard ``pytorch``
                parameters)
        '''
        self._parameters = None # will be set when we defined the grouprs.
        self.groups = groups
        self._lrate = lrate
        self._std_optim = std_optim
        self._groups = groups
        self._update_count = 0

    @property
    def groups(self):
        return self._groups

    @groups.setter
    def groups(self, value):
        self._groups = value
        parameters = []
        for group in value:
            parameters += [param for param in group]
        self._parameters = parameters

    def init_step(self):
        'Set all the standard/Bayesian parameters gradient to zero.'
        if self._std_optim is not None:
            self._std_optim.zero_grad()
        for parameter in self._parameters:
            parameter.zero_stats()

    def step(self):
        'Update one group the standard/Bayesian parameters.'
        if self._std_optim is not None:
            self._std_optim.step()
        if self._update_count >= len(self._groups):
            self._update_count = 0
        for parameter in self._groups[self._update_count]:
            parameter.natural_grad_update(self._lrate)

        self._update_count += 1

