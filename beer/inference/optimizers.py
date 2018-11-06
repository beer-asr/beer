
class BayesianModelOptimizer:
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
            parameter.stats.zero_()

    def step(self):
        'Update one group the standard/Bayesian parameters.'
        if self._std_optim is not None:
            self._std_optim.step()
        if self._update_count >= len(self._groups):
            self._update_count = 0
        for parameter in self._groups[self._update_count]:
            parameter.natural_grad_update(self._lrate)

        self._update_count += 1


class CVBOptimizer:

    def __init__(self, params, std_optim=None):
        '''
        Args:
            parameters (list): List of ``BayesianParameters``.
            lrate (float): learning rate.
            std_optim (``torch.optim.Optimizer``): Optimizer for
                non-Bayesian parameters (i.e. standard ``pytorch``
                parameters)
        '''
        self._parameters = list(params)
        self._std_optim = std_optim

    def init_step(self, stats=None):
        'Set all the standard/Bayesian parameters gradient to zero.'
        if self._std_optim is not None:
            self._std_optim.zero_grad()
        for parameter in self._parameters:
            if stats is not None and stats[parameter] is not None:
                parameter.remove_stats(stats[parameter])

    def step(self):
        'Update one group the standard/Bayesian parameters.'
        if self._std_optim is not None:
            self._std_optim.step()
        for parameter in self._parameters:
            parameter.add_stats(parameter.stats)


class SCVBOptimizer:

    def __init__(self, params, std_optim=None, lrate=1.):
        '''
        Args:
            parameters (list): List of ``BayesianParameters``.
            lrate (float): learning rate.
            std_optim (``torch.optim.Optimizer``): Optimizer for
                non-Bayesian parameters (i.e. standard ``pytorch``
                parameters)
        '''
        self._parameters = list(params)
        self._std_optim = std_optim
        self._lrate = lrate

    def init_step(self, stats=None):
        'Set all the standard/Bayesian parameters gradient to zero.'
        if self._std_optim is not None:
            self._std_optim.zero_grad()
        for parameter in self._parameters:
            if stats is not None and stats[parameter] is not None:
                parameter.remove_stats(stats[parameter])

    def step(self, burn_in=False):
        'Update one group the standard/Bayesian parameters.'
        if self._std_optim is not None:
            self._std_optim.step()
        for parameter in self._parameters:
            if burn_in:
                parameter.add_stats(parameter.stats)
            else:
                parameter.natural_grad_update(self._lrate)



__all__ = ['BayesianModelOptimizer', 'CVBOptimizer', 'SCVBOptimizer']
