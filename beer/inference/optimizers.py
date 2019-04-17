
__all__ = ['VBConjugateOptimizer', 'VBOptimizer']


class VBConjugateOptimizer:
    'Variational Bayes optimizer for "conjugate parameters".'

    def __init__(self, groups, lrate=1.):
        # List of list of parameters. Note that we expand the list as
        # the user may pass a generator.
        self.groups = [[param for param in group] for group in groups]
        self.lrate = lrate
        self.update_count = 0

    def state_dict(self):
        return {'lrate': self.lrate, 'update_count': self.update_count}

    def load_state_dict(self, state_dict):
        self.lrate = state_dict['lrate']
        self.update_count = state_dict['update_count']

    def init_step(self):
        for group in self.groups:
            for param in group:
                param.zero_stats()

    def step(self):
        if len(self.groups) > 0:
            for parameter in self.groups[self.update_count % len(self.groups)]:
                parameter.natural_grad_update(self.lrate)
        self.update_count += 1


class VBOptimizer:
    '''Generic Variational Bayes optimizer which combined conjugate and
    std optimizers.
    '''

    def __init__(self, cjg_optim=None, std_optim=None):
        self.cjg_optim = cjg_optim
        self.std_optim = std_optim

    def state_dict(self):
        state = {}
        if self.cjg_optim is not None:
            state['cjg_optim'] = self.cjg_optim.state_dict()
        if self.std_optim is not None:
            state['std_optim'] = self.std_optim.state_dict()
        return state

    def load_state_dict(self, state_dict):
        if self.cjg_optim is not None:
            self.cjg_optim.load_state_dict(state_dict['cjg_optim'])
        if self.std_optim is not None:
            self.std_optim.load_state_dict(state_dict['std_optim'])

    def load_state(self, path):
        with open(path, 'rb') as f:
            self.lrate, self.update_count = pickle.load(f)

    def init_step(self):
        if self.cjg_optim is not None: self.cjg_optim.init_step()
        if self.std_optim is not None: self.std_optim.zero_grad()

    def step(self):
        if self.std_optim is not None: self.std_optim.step()
        if self.cjg_optim is not None: self.cjg_optim.step()

