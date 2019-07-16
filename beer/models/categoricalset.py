import copy
import torch
from dataclasses import dataclass


from .basemodel import Model
from .parameters import ConjugateBayesianParameter
from .modelset import ModelSet
from .categorical import Categorical, _default_param
from ..dists import Dirichlet, DirichletStdParams

__all__ = ['CategoricalSet', 'SBCategoricalSet']


class CategoricalSet(ModelSet):
    'Set of Categorical distributions with a Dirichlet prior.'

    @classmethod
    def create(cls, weights, prior_strength=1.):
        '''Create a Categorical model.

        Args:
            weights (``torch.Tensor[dim]``): Initial mean distribution.
            prior_strength (float): Strength of the Dirichlet prior.

        Returns:
            :any:`Categorical`

        '''
        return cls(_default_param(weights.detach(), prior_strength))

    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    @property
    def mean(self):
        return self.weights.value()

    def sufficient_statistics(self, data):
        return self.weights.likelihood_fn.sufficient_statistics(data)

    def mean_field_factorization(self):
        return [[self.weights]]

    def expected_log_likelihood(self, stats):
        nparams = self.weights.natural_form()
        return self.weights.likelihood_fn(nparams, stats)

    def accumulate(self, stats, resps):
        w_stats = resps.t() @ stats
        return {self.weights: w_stats}

    def accumulate_from_jointresps(self, jointresps_stats):
        return {self.weights: jointresps_stats.sum(dim=0)}

    ####################################################################
    # ModelSet interface.

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.__class__(self.weights[key])
        return Categorical(self.weights[key])


########################################################################
# Parameterization of the Dirichlet pdf with log-concentrations. This
# is useful to run a gradient ascent based optimization.

@dataclass(init=False, unsafe_hash=True)
class DirichletLogParams(torch.nn.Module):
    log_concentrations: torch.Tensor

    def __init__(self, log_concentrations):
        super().__init__()
        if log_concentrations.requires_grad:
            self.register_parameter('log_concentrations', 
                                    torch.nn.Parameter(log_concentrations))
        else:
            self.register_buffer('log_concentrations', log_concentrations)

    @property
    def concentrations(self):
        return self.log_concentrations.exp()


########################################################################
# Gradient ascent optimization of the variational posterior of the 
# root base measure of the HDP.

# Lower-bound of the objective function.
def _lower_bound(sb_set, root_sbc, concentration):
    # Extract needed quantity to compute the obj. function.
    prob1 = root_sbc.mean[root_sbc.ordering]
    log_prob1, _ = root_sbc._log_prob()
    log_v2, log_1_v2 = sb_set._log_v()
    r_cumsum = torch.flip(torch.flip(log_prob1[1:],
                          dims=(0,)).cumsum(dim=0), dims=(0,))
    
    # Objective function: 
    #   L < E[ ln p(V^2 | V^1) ] - D( q(V^1) || p(V^1) )
    llh = 0 * log_v2.shape[0] * log_prob1.sum() + 0 * log_v2.shape[0] * r_cumsum.sum() \
            + (log_v2 @ (concentration * prob1)).sum() \
            + (log_1_v2 @ (concentration * (1 - prob1.cumsum(dim=0)))).sum() 

    # Normalize the log-likelihood to scale down the gradient.
    llh /= (log_v2.shape[0] * log_v2.shape[1])  

    return llh - root_sbc.stickbreaking.kl_div_posterior_prior().sum()


def _optimize_root_sb(sb_set, concentration, epochs, optim_cls, optim_args):
    '''Update the root stick-breaking process of a Hierarchical
    Dirichlet Process.
    
    Args:
        sb_set (:any:`SBCategoricalSet`): Set of stick-breaking process
            tied by a root stick-breaking process.
        concentration (float): Concentration of the bottom level
            stick-breaking process.
        epochs (int): Number of epochs to update the variational
            posterior.
        optim_cls (:any:`torch.optim`): Optimizer class.
        optim_args (dict): Arguments for the initialization of the 
            optimizer.

    '''    
    root_sb = sb_set.root_sb_categorical.stickbreaking

    # Change the parameterization of the variational posterior
    # so we can run a gradient ascent optimization.
    params_data = root_sb.posterior.params.concentrations.log().clone()
    root_sb.posterior.params = \
            DirichletLogParams(params_data.requires_grad_(True))
    
    # Create the optimzer.
    optim = optim_cls(root_sb.parameters(), **optim_args)

    # Optimization.
    elbos = []
    for i in range(epochs):
        optim.zero_grad()
        L = _lower_bound(sb_set, sb_set.root_sb_categorical, concentration)
        (-L).backward() # pytorch minize the loss !
        elbos.append(float(L))
        optim.step()

    # Change again the parameterization of the variational posterior
    # to avoid unnecessary gradient computation.
    params_data = root_sb.posterior.params.concentrations.detach().clone()
    root_sb.posterior.params = \
            DirichletStdParams(params_data.requires_grad_(False))

    # Notify observers that the variational posterior has been updated.
    sb_set.root_sb_categorical.stickbreaking.dispatch()

    return elbos


########################################################################
# Helper to build the default parameters.

def _default_set_sb_param(n_components, root_sb_categorical, prior_strength):
    mean = root_sb_categorical.mean
    tensorconf = {'dtype': mean.dtype, 'device': mean.device}
    params = torch.ones(n_components, len(root_sb_categorical.stickbreaking), 2,
                        **tensorconf)
    params[:, :, 0] = prior_strength * mean
    params[:, :, 1] = prior_strength * (1 - mean.cumsum(dim=0))
    params = params.reshape(-1, 2)
    prior = Dirichlet.from_std_parameters(params)
    post = root_sb_categorical.stickbreaking.posterior
    params = post.params.concentrations.repeat(n_components, 1)
    posterior = Dirichlet.from_std_parameters(params.clone())
    return ConjugateBayesianParameter(prior, posterior)


########################################################################


class SBCategoricalSet(Model):
    'Set of categorical with a truncated stick breaking prior.'

    # NOTE: contrary to the single SBCategorical, this class does not
    # include a re-ordering mechanism.  

    @classmethod
    def create(cls, n_components, root_sb_categorical, prior_strength=1.,
               optim_cls=torch.optim.Adam, optim_args=None, 
               epochs=15_000):
        '''Create a set of Categorical model.
        Args:
            n_components (int): Number of components in the set.
            root_sb_categorical (int): Root stick-breaking process.
            prior_strength (float): Strength (i.e. concentration) of
                the bottom level stick breaking prior.
            optim_cls (`torch.optim`): Optimizer class to update the 
                root base measure.
            optim_args (dict): Arguments to initialize the optimzer.
            epochs (int): Number of iterations to run the optimization 
                for the base measure.

        Returns:
            :any:`SBCategoricalSet`
        '''
        if optim_args is None:
            optim_args = {'lr': 1e-3}
        param = _default_set_sb_param(n_components, root_sb_categorical, 
                                      prior_strength)
        return cls(n_components, param, root_sb_categorical, prior_strength,
                   optim_cls, optim_args, epochs)

    def __init__(self, n_components, stickbreaking, root_sb_categorical,
                 concentration, optim_cls, optim_args, epochs):
        super().__init__()
        self.n_components = n_components
        self.stickbreaking = stickbreaking
        self.root_sb_categorical = root_sb_categorical
        self.concentration = concentration
        self.optim_cls = optim_cls
        self.optim_args = optim_args
        self.epochs = epochs
        self.stickbreaking.register_callback(self._transform_stats, 
                                             notify_before_update=True)
        #self.stickbreaking.register_callback(self._update_root_sb)    

    def _transform_stats(self):
        stats = self.stickbreaking.stats
        stats = stats[:, self.ordering]
        s2 = torch.zeros_like(stats)
        s2[:, :-1] = stats[:,  1:]
        s2 = torch.flip(torch.flip(s2, dims=(1,)).cumsum(dim=1), dims=(1,))
        new_stats = torch.cat([stats[:, :, None], s2[:, :, None]],  dim=-1)
        new_stats[:, :, -1] += new_stats[:, :, :-1].sum(dim=-1)
        self.stickbreaking.stats = \
                new_stats[:, self.reverse_ordering, :].reshape(-1, 2)

    def _update_root_sb(self):
        # Update the variational posterior
        _optimize_root_sb(self, self.concentration, self.epochs, self.optim_cls,
                          self.optim_args)

        # Update the prior of the bottom level stick-breaking process.
        mean = self.root_sb_categorical.mean
        tensorconf = {'dtype': mean.dtype, 'device': mean.device}
        params = torch.ones(self.n_components, 
                            len(self.root_sb_categorical.stickbreaking), 2,
                            **tensorconf)
        params[:, :, 0] = self.concentration * mean
        params[:, :, 1] = self.concentration * (1 - mean.cumsum(dim=0))
        params = params.reshape(-1, 2)
        self.stickbreaking.prior = Dirichlet.from_std_parameters(params)

    @property
    def ordering(self):
        return self.root_sb_categorical.ordering
    
    @property
    def reverse_ordering(self):
        return self.root_sb_categorical.reverse_ordering

    def _log_v(self):
        c = self.stickbreaking.posterior.params.concentrations
        c = c.reshape(self.n_components, -1, 2)[:, self.ordering, :]
        s_dig = torch.digamma(c.sum(dim=-1))
        log_v = torch.digamma(c[:, :, 0]) - s_dig
        log_1_v = torch.digamma(c[:, :, 1]) - s_dig
        return log_v, log_1_v

    def _log_prob(self):
        log_v, log_1_v = self._log_v()
        log_prob = log_v
        log_prob[:, 1:] += log_1_v[:, :-1].cumsum(dim=1)
        return log_prob, log_1_v

    @property
    def mean(self):
        c = self.stickbreaking.posterior.params.concentrations
        c = c.reshape(self.n_components, -1, 2)[:, self.ordering, :]
        norm = c.sum(dim=-1) + torch.finfo(c.dtype).eps 
        weights = c[:, :, 0] / norm
        residual = (c[:, :, 1] / norm).cumprod(dim=1)
        weights[:, 1:] *= residual[:, :-1]
        return weights[:, self.reverse_ordering]

    ####################################################################

    def sufficient_statistics(self, data):
        # Data is a matrix of one-hot encoding vectors.
        return data

    def mean_field_factorization(self):
        return [[self.stickbreaking]]

    def expected_log_likelihood(self, stats):
        log_prob, _ = self._log_prob()
        return stats @ log_prob[:, self.reverse_ordering]

    def accumulate(self, stats):
        raise NotImplementedError 

    def accumulate_from_jointresps(self, stats):
        return {self.stickbreaking: stats.sum(dim=0)}
