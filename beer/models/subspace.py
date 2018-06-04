'''Bayesian Subspace models.'''


import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianModel
from ..expfamilyprior import JointExpFamilyPrior
from ..expfamilyprior import DirichletPrior
from ..expfamilyprior import GammaPrior
from ..expfamilyprior import NormalIsotropicCovariancePrior
from ..expfamilyprior import MatrixNormalPrior
from ..expfamilyprior import NormalFullCovariancePrior


########################################################################
# Probabilistic Principal Component Analysis (PPCA)
########################################################################

class PPCA(BayesianModel):
    '''Probabilistic Principal Component Analysis (PPCA).

    Attributes:
        mean (``torch.Tensor``): Expected mean.
        precision (``torch.Tensor[1]``): Expected precision (scalar).
        subspace (``torch.Tensor[subspace_dim, data_dim]``): Mean of the
            matrix definining the prior.

    Example:
        >>> subspace_dim, dim = 2, 4
        >>> mean = torch.zeros(dim)
        >>> precision = 1.
        >>> subspace = torch.randn(subspace_dim ,dim)
        >>> ppca = beer.PPCA.create(mean, precision, subspace)
        >>> ppca.mean
        tensor([ 0.,  0.,  0.,  0.])
        >>> ppca.precision
        tensor(1.)
        >>> ppca.subspace
        tensor([[ 0.0328, -2.8251,  2.6031, -0.9721],
                [ 0.5193,  0.6301, -0.3425,  1.2708]])

    '''

    @staticmethod
    def kl_div_latent_posterior(l_means, l_cov):
        '''KL divergence between the posterior distribution of the
        latent variables and their prior (standard normal).

        Args:
            l_means (``torch.Tensor[N, s_dim]``): Means of the
                posteriors where N is  the number of frames (= the
                number of posterior) and s_dim is the dimension of
                the subspace.
            l_cov (``torch.Tensor[s_dim, s_dim]``): Covariance matrix
                shared accross posteriors.

        Returns:
            ``torch.Tensor[N]``: Per-frame KL-divergence.

        '''
        s_dim = l_means.size(1)
        _, logdet = torch.slogdet(l_cov)
        return .5 * (- s_dim - logdet + torch.trace(l_cov) + \
            torch.sum(l_means ** 2, dim=1))

    def __init__(self, prior_mean, posterior_mean, prior_prec, posterior_prec,
                 prior_subspace, posterior_subspace):
        '''
        Args:
            prior_mean (``NormalIsotropicCovariancePrior``): Prior over
                the global mean.
            posterior_mean (``NormalIsotropicCovariancePrior``):
                Posterior over the global mean.
            prior_prec (``GammaPrior``): Prior over the global precision.
            posterior_prec (``GammaPrior``): Posterior over the global
                precision.
            prior_subspace (``MatrixNormalPrior``): Prior over
                the subpace (i.e. the linear transform of the model).
            posterior_subspace (``MatrixNormalPrior``): Posterior over
                the subspace (i.e. the linear transform of the model).
            subspace_dim (int): Dimension of the subspace.

        '''
        super().__init__()
        self.mean_param = BayesianParameter(prior_mean, posterior_mean)
        self.precision_param = BayesianParameter(prior_prec, posterior_prec)
        self.subspace_param = BayesianParameter(prior_subspace,
                                                 posterior_subspace)
        self._subspace_dim, self._data_dim = self.subspace.size()

    @classmethod
    def create(cls, mean, precision, subspace, pseudo_counts=1.):
        '''Create a Probabilistic Principal Ccomponent model.

        Args:
            mean (``torch.Tensor``): Mean of the model.
            precision (float): Global precision of the model.
            subspace (``torch.Tensor[subspace_dim, data_dim]``): Mean of
                the subspace matrix. `subspace_dim` and `data_dim` are
                the dimension of the subspace and the data respectively.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.

        Returns:
            :any:`PPCA`
        '''
        shape = torch.tensor([pseudo_counts]).type(mean.type())
        rate = torch.tensor([pseudo_counts / float(precision)]).type(mean.type())
        prior_prec = GammaPrior(shape, rate)
        posterior_prec = GammaPrior(shape, rate)
        variance = torch.tensor([1. / float(pseudo_counts)]).type(mean.type())
        prior_mean = NormalIsotropicCovariancePrior(mean, variance)
        posterior_mean = NormalIsotropicCovariancePrior(mean, variance)

        cov = torch.eye(subspace.size(0)).type(mean.type()) / pseudo_counts
        prior_subspace = MatrixNormalPrior(subspace, cov)
        posterior_subspace = MatrixNormalPrior(subspace, cov)

        return cls(prior_mean, posterior_mean, prior_prec, posterior_prec,
                   prior_subspace, posterior_subspace)

    @property
    def mean(self):
        _, mean = self.mean_param.expected_value(concatenated=False)
        return mean

    @property
    def precision(self):
        return self.precision_param.expected_value()[1]

    @property
    def subspace(self):
        _, exp_value2 =  self.subspace_param.expected_value(concatenated=False)
        return exp_value2

    def _get_expectation(self):
        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        s_quad, s_mean = self.subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)
        return log_prec, prec, s_quad, s_mean, m_quad, m_mean

    def _compute_distance_term(self, s_stats, l_means, l_quad):
        _, _, s_quad, s_mean, m_quad, m_mean = self._get_expectation()
        data_mean = s_stats[:, 1:] - m_mean.view(1, -1)
        distance_term = torch.zeros(len(s_stats)).type(s_stats.type())
        distance_term += s_stats[:, 0]
        distance_term += - 2 * s_stats[:, 1:] @ m_mean
        distance_term += - 2 * torch.sum((l_means @ s_mean) * data_mean, dim=1)
        distance_term += l_quad.view(len(s_stats), -1) @ s_quad.view(-1)
        distance_term += m_quad
        return distance_term

    def latent_posterior(self, stats):
        data = stats[:, 1:]
        _, prec, s_quad, s_mean, _, m_mean = self._get_expectation()
        lposterior_cov = torch.inverse(
            torch.eye(self._subspace_dim).type(stats.type()) + prec * s_quad)
        lposterior_means = prec * lposterior_cov @ s_mean @ (data - m_mean).t()
        return lposterior_means.t(), lposterior_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data],
                          dim=-1)

    def forward(self, s_stats, latent_variables=None):
        feadim = s_stats.size(1) - 1

        if latent_variables is not None:
            l_means = latent_variables
            l_quad = l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = torch.zeros(len(s_stats)).type(s_stats.type())
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = self.kl_div_latent_posterior(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, _, s_mean, _, m_mean = self._get_expectation()

        distance_term = self._compute_distance_term(s_stats, l_means, l_quad)
        exp_llh = torch.zeros(len(s_stats)).type(s_stats.type())
        exp_llh += -.5 * feadim * math.log(2 * math.pi)
        exp_llh += .5 * feadim * log_prec
        exp_llh += -.5 * prec * distance_term

        # Cache some computation for a quick accumulation of the
        # sufficient statistics.
        self.cache['distance'] = distance_term.sum()
        self.cache['latent_means'] = l_means
        self.cache['latent_quad'] = l_quad
        self.cache['kl_divergence'] = l_kl_div
        self.cache['precision'] = prec
        self.cache['subspace_mean'] = s_mean
        self.cache['mean_mean'] = m_mean

        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        # Load cached values and clear the cache.
        distance_term = self.cache['distance']
        l_means = self.cache['latent_means']
        l_quad = self.cache['latent_quad']
        prec = self.cache['precision']
        s_mean = self.cache['subspace_mean']
        m_mean = self.cache['mean_mean']
        self.clear_cache()

        t_type = s_stats.type()
        feadim = s_stats.size(1) - 1

        data_mean = s_stats[:, 1:] - m_mean[None, :]
        acc_s_mean = (l_means.t() @ data_mean)
        return {
            self.precision_param: torch.cat([
                .5 * torch.tensor(len(s_stats) * feadim).view(1).type(t_type),
                -.5 * distance_term.view(1)
            ]),
            self.mean_param: torch.cat([
                - .5 * torch.tensor(len(s_stats) * prec).view(1).type(t_type),
                prec * torch.sum(s_stats[:, 1:] - l_means @ s_mean, dim=0)
            ]),
            self.subspace_param:  torch.cat([
                - .5 * prec * l_quad.sum(dim=0).view(-1),
                prec * acc_s_mean.view(-1)
            ])
        }

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([torch.sum(mean ** 2 + var, dim=1).view(-1, 1), mean],
                         dim=1)

    def local_kl_div_posterior_prior(self):
        return self.cache['kl_divergence']

    def expected_natural_params(self, mean, var, latent_variables=None,
                                nsamples=1):
        '''Interface for the VAE model. Returns the expected value of the
        natural params of the latent model given the per-frame means
        and variances.

        Args:
            mean (Tensor): Per-frame mean of the posterior distribution.
            var (Tensor): Per-frame variance of the posterior
                distribution.
            labels (Tensor): Frame labelling (if any).
            nsamples (int): Number of samples to estimate the
                natural parameters (ignored).

        Returns:
            (Tensor): Expected value of the natural parameters.

        Note:
            The expected natural parameters can be estimated in close
            form and therefore does not requires sampling. Hence,
            the parameters `nsamples` will be ignored.

        '''
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)

        if latent_variables is not None:
            l_means = latent_variables
            l_quad = l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = torch.zeros(len(s_stats)).type(s_stats.type())
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = self.kl_div_latent_posterior(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, s_quad, s_mean, m_quad, m_mean = self._get_expectation()

        np1 = -.5 * prec * torch.ones(len(s_stats),
                                      mean.size(1)).type(mean.type())
        np2 = prec * (l_means @ s_mean + m_mean)
        np3 = torch.zeros(len(s_stats), mean.size(1)).type(mean.type())
        np3 += -.5 * prec * (l_quad.view(len(s_stats), -1) @ s_quad.view(-1)).view(-1, 1)
        np3 += -(prec * l_means @ s_mean @ m_mean).reshape(-1, 1)
        np3 += -.5 * prec * m_quad
        np3 /= self._data_dim
        np4 = .5 * log_prec * torch.ones(s_stats.size(0),
                                          mean.size(1)).type(mean.type())

        # Cache some computation for a quick accumulation of the
        # sufficient statistics.
        self.cache['distance'] = self._compute_distance_term(s_stats, l_means,
                                                             l_quad).sum()
        self.cache['latent_means'] = l_means
        self.cache['latent_quad'] = l_quad
        self.cache['kl_divergence'] = l_kl_div
        self.cache['precision'] = prec
        self.cache['subspace_mean'] = s_mean
        self.cache['mean_mean'] = m_mean
        return torch.cat([np1, np2, np3, np4], dim=1).detach()


########################################################################
# Probabilistic Linear Discriminant Analysis (PLDA)
########################################################################

class PLDA(BayesianModel):
    '''Probabilistic Linear Discriminant Analysis (PPCA).

    Attributes:
        mean (``torch.Tensor``): Expected mean.
        precision (``torch.Tensor[1]``): Expected precision (scalar).
        noise_subspace (``torch.Tensor[noise_subspace_dim, data_dim]``):
            Mean of the matrix defining the "noise" subspace.
            (within-class variations).
        class_subspace (``torch.Tensor[class_subspace_dim, data_dim]``):
            Mean of the matrix defining the "class" subspace.
            (across-class variations).
        class_means: (``torch.Tensor[]``)
        weights (``torch.Tensor[n_class]``): Prior weight for each
            class.

    '''

    @staticmethod
    def kl_div_latent_posterior(l_means, l_cov):
        '''KL divergence between the posterior distribution of the
        latent variables and their prior (standard normal).

        Args:
            l_means (``torch.Tensor[N, s_dim]``): Means of the
                posteriors where N is  the number of frames (= the
                number of posterior) and s_dim is the dimension of
                the subspace.
            l_cov (``torch.Tensor[s_dim, s_dim]``): Covariance matrix
                shared accross posteriors.

        Returns:
            ``torch.Tensor[N]``: Per-frame KL-divergence.

        '''
        s_dim = l_means.size(1)
        _, logdet = torch.slogdet(l_cov)
        return .5 * (- s_dim - logdet + torch.trace(l_cov) + \
            torch.sum(l_means ** 2, dim=1))

    def __init__(self, prior_mean, posterior_mean, prior_prec, posterior_prec,
                 prior_noise_subspace, posterior_noise_subspace,
                 prior_class_subspace, posterior_class_subspace,
                 prior_means, posterior_means, prior_weights,
                 posterior_weights):
        '''
        Args:
            prior_mean (``NormalIsotropicCovariancePrior``): Prior over
                the global mean.
            posterior_mean (``NormalIsotropicCovariancePrior``):
                Posterior over the global mean.
            prior_prec (``GammaPrior``): Prior over the global precision.
            posterior_prec (``GammaPrior``): Posterior over the global
                precision.
            prior_noise_subspace (``MatrixNormalPrior``): Prior over
                the subpace.
            posterior_noise_subspace (``MatrixNormalPrior``): Posterior
                over the noise subspace.
            prior_class_subspace (``MatrixNormalPrior``): Prior over
                the class subspace.
            posterior_class_subspace (``MatrixNormalPrior``): Posterior
                over the class subspace.
            prior_means (``JointExpFamilyPrior``): Prior over the class'
                means.
            posterior_means (``JointExpFamilyPrior``):
                Posterior over the class' means.
            prior_weights (``DirichletPrior``): Prior over the mixing
                weights.
            posterior_weights (``DirichletPrior``): Posterior over the
                mixing weights.

        '''
        super().__init__()
        self.mean_param = BayesianParameter(prior_mean, posterior_mean)
        self.precision_param = BayesianParameter(prior_prec, posterior_prec)
        self.noise_subspace_param = BayesianParameter(prior_noise_subspace,
                                                      posterior_noise_subspace)
        self.class_subspace_param = BayesianParameter(prior_class_subspace,
                                                      posterior_class_subspace)
        self.class_means_param = BayesianParameter(prior_means,
                                                   posterior_means)
        self.weights_param = BayesianParameter(prior_weights, posterior_weights)
        self._subspace1_dim, self._data_dim = self.noise_subspace.size()
        self._subspace2_dim, _ = self.class_subspace.size()

    @classmethod
    def create(cls, mean, precision, noise_subspace, class_subspace,
               class_means, weights, pseudo_counts=1.):
        '''Create a Probabilistic Principal Ccomponent model.

        Args:
            mean (``torch.Tensor``): Mean of the model.
            precision (float): Global precision of the model.
            noise_subspace (``torch.Tensor[subspace_dim, data_dim]``):
                Mean of the noise subspace matrix. `subspace_dim` and
                `data_dim` are the dimension of the subspace and the
                data respectively.
            class_subspace (``torch.Tensor[subspace_dim, data_dim]``):
                Mean of the class subspace matrix. `subspace_dim` and
                `data_dim` are the dimension of the subspace and the
                data respectively.
            class_means (``torch.Tensor[nclass, dim]``): Means of the
                class as a matrix.
            weights (``torch.Tensor``): Mixing weights.
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.

        Returns:
            :any:`PPCA`
        '''
        # Precision.
        shape = torch.tensor([pseudo_counts]).type(mean.type())
        rate = torch.tensor([pseudo_counts / float(precision)]).type(mean.type())
        prior_prec = GammaPrior(shape, rate)
        posterior_prec = GammaPrior(shape, rate)

        # Global mean.
        variance = torch.tensor([1. / float(pseudo_counts)]).type(mean.type())
        prior_mean = NormalIsotropicCovariancePrior(mean, variance)
        posterior_mean = NormalIsotropicCovariancePrior(mean, variance)

        # Noise subspace.
        cov = torch.eye(noise_subspace.size(0)).type(mean.type())
        cov /= pseudo_counts
        prior_noise_subspace = MatrixNormalPrior(noise_subspace, cov)
        posterior_noise_subspace = MatrixNormalPrior(noise_subspace, cov)

        # Class subspace.
        cov = torch.eye(class_subspace.size(0)).type(mean.type())
        cov /= pseudo_counts
        prior_class_subspace = MatrixNormalPrior(class_subspace, cov)
        posterior_class_subspace = MatrixNormalPrior(class_subspace, cov)

        # cov = same as class subspace.
        class_mean_priors, class_mean_posteriors  = [], []
        for mean in class_means:
            class_mean_priors.append(NormalFullCovariancePrior(mean, cov))
            class_mean_posteriors.append(NormalFullCovariancePrior(mean, cov))

        # Weights.
        prior_weights = DirichletPrior(weights * pseudo_counts)
        posterior_weights = DirichletPrior(weights * pseudo_counts)

        return cls(prior_mean, posterior_mean, prior_prec, posterior_prec,
                   prior_noise_subspace, posterior_noise_subspace,
                   prior_class_subspace, posterior_class_subspace,
                   JointExpFamilyPrior(class_mean_priors),
                   JointExpFamilyPrior(class_mean_posteriors),
                   prior_weights, posterior_weights)

    @property
    def mean(self):
        _, mean = self.mean_param.expected_value(concatenated=False)
        return mean

    @property
    def precision(self):
        return self.precision_param.expected_value()[1]

    @property
    def noise_subspace(self):
        _, exp_value2 = \
            self.noise_subspace_param.expected_value(concatenated=False)
        return exp_value2

    @property
    def class_subspace(self):
        _, exp_value2 =  \
            self.class_subspace_param.expected_value(concatenated=False)
        return exp_value2

    @property
    def class_means(self):
        exp_nparams =  self.class_means_param.expected_value(concatenated=False)
        return torch.stack([mean for _, mean in exp_nparams])

    @property
    def weights(self):
        'Expected value of the weights of the mixture.'
        concentrations = self.weights_param.posterior.natural_hparams + 1
        return concentrations / concentrations.sum()

    def _get_expectation(self):
        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        s_quad, s_mean = self.subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)
        return log_prec, prec, s_quad, s_mean, m_quad, m_mean

    def _compute_distance_term(self, s_stats, l_means, l_quad):
        _, _, s_quad, s_mean, m_quad, m_mean = self._get_expectation()
        data_mean = s_stats[:, 1:] - m_mean.view(1, -1)
        distance_term = torch.zeros(len(s_stats)).type(s_stats.type())
        distance_term += s_stats[:, 0]
        distance_term += - 2 * s_stats[:, 1:] @ m_mean
        distance_term += - 2 * torch.sum((l_means @ s_mean) * data_mean, dim=1)
        distance_term += l_quad.view(len(s_stats), -1) @ s_quad.view(-1)
        distance_term += m_quad
        return distance_term

    def latent_posterior(self, stats):
        data = stats[:, 1:]
        _, prec, s_quad, s_mean, _, m_mean = self._get_expectation()
        lposterior_cov = torch.inverse(
            torch.eye(self._subspace_dim).type(stats.type()) + prec * s_quad)
        lposterior_means = prec * lposterior_cov @ s_mean @ (data - m_mean).t()
        return lposterior_means.t(), lposterior_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data],
                          dim=-1)

    def forward(self, s_stats, latent_variables=None):
        feadim = s_stats.size(1) - 1

        if latent_variables is not None:
            l_means = latent_variables
            l_quad = l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = torch.zeros(len(s_stats)).type(s_stats.type())
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = self.kl_div_latent_posterior(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, _, s_mean, _, m_mean = self._get_expectation()

        distance_term = self._compute_distance_term(s_stats, l_means, l_quad)
        exp_llh = torch.zeros(len(s_stats)).type(s_stats.type())
        exp_llh += -.5 * feadim * math.log(2 * math.pi)
        exp_llh += .5 * feadim * log_prec
        exp_llh += -.5 * prec * distance_term

        # Cache some computation for a quick accumulation of the
        # sufficient statistics.
        self.cache['distance'] = distance_term.sum()
        self.cache['latent_means'] = l_means
        self.cache['latent_quad'] = l_quad
        self.cache['kl_divergence'] = l_kl_div
        self.cache['precision'] = prec
        self.cache['subspace_mean'] = s_mean
        self.cache['mean_mean'] = m_mean

        return exp_llh

    def accumulate(self, s_stats, parent_msg=None):
        # Load cached values and clear the cache.
        distance_term = self.cache['distance']
        l_means = self.cache['latent_means']
        l_quad = self.cache['latent_quad']
        prec = self.cache['precision']
        s_mean = self.cache['subspace_mean']
        m_mean = self.cache['mean_mean']
        self.clear_cache()

        t_type = s_stats.type()
        feadim = s_stats.size(1) - 1

        data_mean = s_stats[:, 1:] - m_mean[None, :]
        acc_s_mean = (l_means.t() @ data_mean)
        return {
            self.precision_param: torch.cat([
                .5 * torch.tensor(len(s_stats) * feadim).view(1).type(t_type),
                -.5 * distance_term.view(1)
            ]),
            self.mean_param: torch.cat([
                - .5 * torch.tensor(len(s_stats) * prec).view(1).type(t_type),
                prec * torch.sum(s_stats[:, 1:] - l_means @ s_mean, dim=0)
            ]),
            self.subspace_param:  torch.cat([
                - .5 * prec * l_quad.sum(dim=0).view(-1),
                prec * acc_s_mean.view(-1)
            ])
        }

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    @staticmethod
    def sufficient_statistics_from_mean_var(mean, var):
        return torch.cat([torch.sum(mean ** 2 + var, dim=1).view(-1, 1), mean],
                         dim=1)

    def local_kl_div_posterior_prior(self):
        return self.cache['kl_divergence']

    def expected_natural_params(self, mean, var, latent_variables=None,
                                nsamples=1):
        '''Interface for the VAE model. Returns the expected value of the
        natural params of the latent model given the per-frame means
        and variances.

        Args:
            mean (Tensor): Per-frame mean of the posterior distribution.
            var (Tensor): Per-frame variance of the posterior
                distribution.
            labels (Tensor): Frame labelling (if any).
            nsamples (int): Number of samples to estimate the
                natural parameters (ignored).

        Returns:
            (Tensor): Expected value of the natural parameters.

        Note:
            The expected natural parameters can be estimated in close
            form and therefore does not requires sampling. Hence,
            the parameters `nsamples` will be ignored.

        '''
        s_stats = self.sufficient_statistics_from_mean_var(mean, var)

        if latent_variables is not None:
            l_means = latent_variables
            l_quad = l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = torch.zeros(len(s_stats)).type(s_stats.type())
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = self.kl_div_latent_posterior(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, s_quad, s_mean, m_quad, m_mean = self._get_expectation()

        np1 = -.5 * prec * torch.ones(len(s_stats),
                                      mean.size(1)).type(mean.type())
        np2 = prec * (l_means @ s_mean + m_mean)
        np3 = torch.zeros(len(s_stats), mean.size(1)).type(mean.type())
        np3 += -.5 * prec * (l_quad.view(len(s_stats), -1) @ s_quad.view(-1)).view(-1, 1)
        np3 += -(prec * l_means @ s_mean @ m_mean).reshape(-1, 1)
        np3 += -.5 * prec * m_quad
        np3 /= self._data_dim
        np4 = .5 * log_prec * torch.ones(s_stats.size(0),
                                          mean.size(1)).type(mean.type())

        # Cache some computation for a quick accumulation of the
        # sufficient statistics.
        self.cache['distance'] = self._compute_distance_term(s_stats, l_means,
                                                             l_quad).sum()
        self.cache['latent_means'] = l_means
        self.cache['latent_quad'] = l_quad
        self.cache['kl_divergence'] = l_kl_div
        self.cache['precision'] = prec
        self.cache['subspace_mean'] = s_mean
        self.cache['mean_mean'] = m_mean
        return torch.cat([np1, np2, np3, np4], dim=1).detach()


__all__ = ['PPCA', 'PLDA']
