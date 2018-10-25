
import abc
import torch
from .bayesmodel import BayesianModel


class BayesianModelSet(BayesianModel, metaclass=abc.ABCMeta):
    '''Abstract base class for a set of the :any:`BayesianModel`.

    This model is used by model having discrete latent variable such
    as Mixture models  or Hidden Markov models.

    Note:
        subclasses of :any:`BayesianModelSet` are expected to be
        iterable and therefore should implement at minima:

        .. code-block:: python

           MyBayesianModelSet:

               def __getitem__(self, key):
                  ...

               def __len__(self):
                  ...

    '''

    @abc.abstractmethod
    def __getitem__(self, key):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass



class JointModelSet(BayesianModelSet):
    '''Set of concatenated model sets having the same type of
    sufficient statistics.
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


class DynamicallyOrderedModelSet(BayesianModelSet):
    '''Set of model for which the order of the components might
    change for each called.

    Attributes:
        original_modelset (:any:`BayesianModelSet`): original model.
        order (sequence of integer): new order of the model set.

    Note:
        The ordering sequence can contain several time the index
        of the same components. This is useful for sharing parameters.

    '''

    def __init__(self, original_modelset):
        super().__init__()
        self.original_modelset = original_modelset

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    def mean_field_factorization(self):
        return self.original_modelset.mean_field_factorization()

    def sufficient_statistics(self, data):
        return self.original_modelset.sufficient_statistics(data)

    def expected_log_likelihood(self, stats, order=None):
        if order is None:
            order = list(range(len(self.original_modelset)))
        dtype, device = stats.dtype, stats.device
        pc_exp_llh = self.original_modelset.expected_log_likelihood(stats)
        self.cache['order'] = order
        return pc_exp_llh[:, order]

    def accumulate(self, stats, resps):
        order = self.cache['order']
        new_resps = torch.zeros((len(stats), len(self.original_modelset)),
                                 dtype=resps.dtype, device=resps.device)
        for i, val in enumerate(resps.t()):
            new_resps[:, order[i]] += val
        return self.original_modelset.accumulate(stats, new_resps)

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        return self.original_modelset[key]

    def __len__(self):
        return len(self.original_modelset)


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


__all__ = ['DynamicallyOrderedModelSet', 'JointModelSet', 'RepeatedModelSet']

