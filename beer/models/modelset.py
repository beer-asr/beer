

import torch
from .bayesmodel import BayesianModelSet


class JointModelSet(BayesianModelSet):
    '''Set of concatenated model sets having the same type of sufficient
    statistics.
    '''

    def __init__(self, modelsets):
        '''Args:
        modelsets: (seq): list of model set.
        '''
        super().__init__()
        self.modelsets = modelsets
        for i, modelset in enumerate(self.modelsets):
            self._register_submodel('smodel' + str(i), modelset)

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        param_groups = []
        for modelset in self.modelsets:
            m_param_groups = modelset.mean_field_factorization()
            if len(m_param_groups) > 1:
                raise ValueError('Invalid model set: more than 1 mean field group')
            param_groups += m_param_groups[0]
        return [param_groups]

    def sufficient_statistics(self, data):
        return self.modelsets[0].sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        return torch.cat([
            modelset.expected_log_likelihood(stats)
            for modelset in self.modelsets
        ], dim=-1)

    def accumulate(self, stats, resps):
        acc_stats = {}
        start_idx = 0
        for modelset in self.modelsets:
            length = len(modelset)
            modelset_resps = resps[:, start_idx: start_idx + length]
            acc_stats.update(modelset.accumulate(stats, modelset_resps))
            start_idx += length
        return acc_stats

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        '''Args:
        key (int): state index.

        '''
        if key < 0:
            raise ValueError("Unsupported negative index")
        total_length = 0
        for modelset in self.modelsets:
            if key < total_length + len(modelset):
                return modelset[key - total_length]
            total_length += len(modelset)
        raise IndexError('index out of range')

    def __len__(self):
        length = 0
        for model in self.modelsets:
            length += len(model)
        return length


class RepeatedModelSet(BayesianModelSet):
    '''Model set where an internal model set is repeated K times. This
    object is used in mixture-like models when the components of the
    mixture are shared across several classes.

    '''

    def __init__(self, modelset, repeat):
        '''Args:
            modelset (seq): Internal model set.
            repeat (int):Number of time to repeat the the model set.
        '''
        super().__init__()
        self.modelset = modelset
        self.repeat = repeat


    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats):
        llhs = self.modelset.expected_log_likelihood(stats)
        rep_llhs = llhs[:, None, :].repeat(1, self.repeat, 1)
        return rep_llhs.view(len(stats), -1)

    def accumulate(self, stats, resps):
        new_resps = resps.reshape(len(stats), self.repeat, -1).sum(dim=1)
        return self.modelset.accumulate(stats, new_resps)

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        return self.modelset[key % len(self.modelset)]

    def __len__(self):
        return len(self.modelset) * self.repeat


__all__ = ['RepeatedModelSet', 'JointModelSet']
