'''Bayesian Subspace models.'''


import math
import torch

from .bayesmodel import BayesianParameter
from .bayesmodel import BayesianParameterSet
from .bayesmodel import BayesianModel
from .bayesmodel import BayesianModelSet
from .normal import NormalSetElement
from ..expfamilyprior import GammaPrior
from ..expfamilyprior import NormalIsotropicCovariancePrior
from ..expfamilyprior import MatrixNormalPrior
from ..expfamilyprior import NormalFullCovariancePrior
from ..utils import make_symposdef


def kl_div_std_norm(means, cov):
    '''KL divergence between a set of Normal distributions with a
    shared covariance matrix and a standard Normal N(0, I).

    Args:
        means (``torch.Tensor[N, dim]``): Means of the
            Normal distributions where N is  the number of frames and
            dim is the dimension of the random variable.
        cov (``torch.Tensor[s_dim, s_dim]``): Shared covariance matrix.

    Returns:
        ``torch.Tensor[N]``: Per-distribution KL-divergence.

    '''
    dim = means.size(1)
    _, logdet = torch.slogdet(cov)
    return .5 * (-dim - logdet + torch.trace(cov) + \
        torch.sum(means ** 2, dim=1))



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
        shape = torch.tensor([pseudo_counts], dtype=mean.dtype,
                             device=mean.device)
        rate = torch.tensor([pseudo_counts / float(precision)],
                            dtype=mean.dtype, device=mean.device)
        prior_prec = GammaPrior(shape, rate)
        posterior_prec = GammaPrior(shape, rate)
        variance = torch.tensor([1. / float(pseudo_counts)], dtype=mean.dtype,
                                device=mean.device)
        prior_mean = NormalIsotropicCovariancePrior(mean, variance)
        posterior_mean = NormalIsotropicCovariancePrior(mean, variance)

        cov = torch.eye(subspace.size(0), dtype=mean.dtype,
                        device=mean.device) / pseudo_counts
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
        _, exp_value2 = self.subspace_param.expected_value(concatenated=False)
        return exp_value2

    def _get_expectation(self):
        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        s_quad, s_mean = self.subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)
        return log_prec, prec, s_quad, s_mean, m_quad, m_mean

    def _compute_distance_term(self, s_stats, l_means, l_quad):
        _, _, s_quad, s_mean, m_quad, m_mean = self._get_expectation()
        data_mean = s_stats[:, 1:] - m_mean.view(1, -1)
        distance_term = torch.zeros(len(s_stats), dtype=s_stats.dtype,
                                    device=s_stats.device)
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
            torch.eye(self._subspace_dim, dtype=stats.dtype,
                      device=stats.device) + prec * s_quad)
        lposterior_means = prec * lposterior_cov @ s_mean @ (data - m_mean).t()
        return lposterior_means.t(), lposterior_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @property
    def grouped_parameters(self):
        return [
            [self.mean_param, self.precision_param],
            [self.subspace_param]
        ]

    @staticmethod
    def sufficient_statistics(data):
        return torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data],
                         dim=-1)

    def float(self):
        return self.__class__(
            self.mean_param.prior.float(),
            self.mean_param.posterior.float(),
            self.precision_param.prior.float(),
            self.precision_param.posterior.float(),
            self.subspace_param.prior.float(),
            self.subspace_param.posterior.float()
        )

    def double(self):
        return self.__class__(
            self.mean_param.prior.double(),
            self.mean_param.posterior.double(),
            self.precision_param.prior.double(),
            self.precision_param.posterior.double(),
            self.subspace_param.prior.double(),
            self.subspace_param.posterior.double()
        )

    def to(self, device):
        return self.__class__(
            self.mean_param.prior.to(device),
            self.mean_param.posterior.to(device),
            self.precision_param.prior.to(device),
            self.precision_param.posterior.to(device),
            self.subspace_param.prior.to(device),
            self.subspace_param.posterior.to(device)
        )


    def forward(self, s_stats, latent_variables=None):
        feadim = s_stats.size(1) - 1

        if latent_variables is not None:
            l_means = latent_variables
            l_quad = l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = torch.zeros(len(s_stats), dtype=s_stats.dtype,
                                   device=s_stats.device)
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = kl_div_std_norm(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, _, s_mean, _, m_mean = self._get_expectation()

        distance_term = self._compute_distance_term(s_stats, l_means, l_quad)
        exp_llh = torch.zeros(len(s_stats), dtype=s_stats.dtype,
                              device=s_stats.device)
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

        dtype = s_stats.dtype
        device = s_stats.device
        feadim = s_stats.size(1) - 1

        data_mean = s_stats[:, 1:] - m_mean[None, :]
        acc_s_mean = (l_means.t() @ data_mean)
        return {
            self.precision_param: torch.cat([
                .5 * torch.tensor(len(s_stats) * feadim, dtype=dtype,
                                  device=device).view(1),
                -.5 * distance_term.view(1)
            ]),
            self.mean_param: torch.cat([
                - .5 * torch.tensor(len(s_stats) * prec, dtype=dtype,
                                    device=device).view(1),
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

    def local_kl_div_posterior_prior(self, parent_msg=None):
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
            l_kl_div = torch.zeros(len(s_stats), dtype=mean.dtype,
                                   device=mean.device)
        else:
            l_means, l_cov = self.latent_posterior(s_stats)
            l_quad = l_cov + l_means[:, :, None] * l_means[:, None, :]
            l_kl_div = kl_div_std_norm(l_means, l_cov)
        l_quad = l_quad.view(len(s_stats), -1)

        log_prec, prec, s_quad, s_mean, m_quad, m_mean = self._get_expectation()

        np1 = -.5 * prec * torch.ones(len(s_stats), mean.size(1),
                                      dtype=mean.dtype, device=mean.device)
        np2 = prec * (l_means @ s_mean + m_mean)
        np3 = torch.zeros(len(s_stats), mean.size(1), dtype=mean.dtype,
                          device=mean.device)
        np3 += -.5 * prec * (l_quad.view(len(s_stats), -1) @ s_quad.view(-1)).view(-1, 1)
        np3 += -(prec * l_means @ s_mean @ m_mean).reshape(-1, 1)
        np3 += -.5 * prec * m_quad
        np3 /= self._data_dim
        np4 = .5 * log_prec * torch.ones(s_stats.size(0), mean.size(1),
                                         dtype=mean.dtype, device=mean.device)

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
        return torch.cat([np1, np2, np3, np4], dim=1), s_stats


########################################################################
# Probabilistic Linear Discriminant Analysis (PLDA)
########################################################################

class PLDASet(BayesianModelSet):
    '''Set of Normal distribution for the Probabilistic Linear
    Discriminant Analysis (PLDA) model. When combined with a
    :any:`Mixture` it is a PLDA.

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

    '''

    def __init__(self, prior_mean, posterior_mean, prior_prec, posterior_prec,
                 prior_noise_subspace, posterior_noise_subspace,
                 prior_class_subspace, posterior_class_subspace,
                 prior_means, posterior_means):
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
            prior_means (``list``): List of prior distributions over
                the class' means.
            posterior_means (``list``): List of posterior distributions
                over the class' means.

        '''
        super().__init__()
        self.mean_param = BayesianParameter(prior_mean, posterior_mean)
        self.precision_param = BayesianParameter(prior_prec, posterior_prec)
        self.noise_subspace_param = BayesianParameter(prior_noise_subspace,
                                                      posterior_noise_subspace)
        self.class_subspace_param = BayesianParameter(prior_class_subspace,
                                                      posterior_class_subspace)
        self.class_mean_params = BayesianParameterSet([
            BayesianParameter(prior, posterior)
            for prior, posterior in zip(prior_means, posterior_means)
        ])
        self._subspace1_dim, self._data_dim = self.noise_subspace.size()
        self._subspace2_dim, _ = self.class_subspace.size()

    @classmethod
    def create(cls, mean, precision, noise_subspace, class_subspace,
               class_means, pseudo_counts=1.):
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
            pseudo_counts (``torch.Tensor``): Strength of the prior.
                Should be greater than 0.

        Returns:
            :any:`PPCA`
        '''
        # Precision.
        shape = torch.tensor([pseudo_counts], dtype=mean.dtype,
                             device=mean.device)
        rate = torch.tensor([pseudo_counts / float(precision)],
                            dtype=mean.dtype, device=mean.device)
        prior_prec = GammaPrior(shape, rate)
        posterior_prec = GammaPrior(shape, rate)

        # Global mean.
        variance = torch.tensor([1. / float(pseudo_counts)], dtype=mean.dtype,
                                device=mean.device)
        prior_mean = NormalIsotropicCovariancePrior(mean, variance)
        posterior_mean = NormalIsotropicCovariancePrior(mean, variance)

        # Noise subspace.
        cov = torch.eye(noise_subspace.size(0), dtype=mean.dtype,
                        device=mean.device)
        cov /= pseudo_counts
        prior_noise_subspace = MatrixNormalPrior(noise_subspace, cov)
        posterior_noise_subspace = MatrixNormalPrior(noise_subspace, cov)

        # Class subspace.
        cov = torch.eye(class_subspace.size(0), dtype=mean.dtype,
                        device=mean.device)
        cov /= pseudo_counts
        prior_class_subspace = MatrixNormalPrior(class_subspace, cov)
        posterior_class_subspace = MatrixNormalPrior(class_subspace, cov)

        # cov = same as class subspace.
        class_mean_priors, class_mean_posteriors = [], []
        for mean_i in class_means:
            class_mean_priors.append(NormalFullCovariancePrior(mean_i, cov))
            class_mean_posteriors.append(NormalFullCovariancePrior(mean_i, cov))

        return cls(prior_mean, posterior_mean, prior_prec, posterior_prec,
                   prior_noise_subspace, posterior_noise_subspace,
                   prior_class_subspace, posterior_class_subspace,
                   class_mean_priors, class_mean_posteriors)

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
        means = []
        for mean_param in self.class_mean_params:
            _, mean = mean_param.expected_value(concatenated=False)
            means.append(mean)
        return torch.stack(means)


    def _get_expectation(self):
        log_prec, prec = self.precision_param.expected_value(concatenated=False)
        noise_s_quad, noise_s_mean = \
            self.noise_subspace_param.expected_value(concatenated=False)
        class_s_quad, class_s_mean = \
            self.class_subspace_param.expected_value(concatenated=False)
        m_quad, m_mean = self.mean_param.expected_value(concatenated=False)
        class_mean_mean, class_mean_quad = [], []
        for mean_param in self.class_mean_params:
            mean_quad, mean = mean_param.expected_value(concatenated=False)
            class_mean_mean.append(mean)
            class_mean_quad.append(mean_quad)
        return {
            'log_prec': log_prec, 'prec': prec,
            'm_quad':m_quad, 'm_mean': m_mean,
            'noise_s_quad':noise_s_quad, 'noise_s_mean': noise_s_mean,
            'class_s_quad':class_s_quad, 'class_s_mean': class_s_mean,
            'class_mean_mean': torch.stack(class_mean_mean),
            'class_mean_quad': torch.stack(class_mean_quad)
        }

    def _precompute(self, s_stats):
        '''Pre-compute intermediary results for the expected
        log-likelihood/expected natural parameters.

        '''
        # Expected value of the parameters.
        self.cache.update(self._get_expectation())

        # Estimate the posterior distribution of the latent variables.
        self.latent_posterior(s_stats)

        # Load the necessary value from the cache.
        noise_means = self.cache['noise_means']
        global_mean = self.cache['m_mean']
        class_s_mean = self.cache['class_s_mean']
        class_mean_mean = self.cache['class_mean_mean']
        l_quads = self.cache['l_quads']
        class_s_quad = self.cache['class_s_quad']
        class_mean_quad = self.cache['class_mean_quad']
        noise_s_quad = self.cache['noise_s_quad'].view(-1)
        m_quad = self.cache['m_quad']
        class_means = class_mean_mean @ class_s_mean
        class_quads = class_mean_quad.view(len(self), -1) @ \
            class_s_quad.view(-1)

        # Separate the s. statistics.
        data_quad, data = s_stats[:, 0], s_stats[:, 1:]
        npoints = len(data)

        # Intermediary computations to compute the exp. llh for each
        # components.
        means = noise_means + class_means.view(len(self), 1, -1) + \
            global_mean.view(1, 1, -1)
        lnorm = l_quads @ noise_s_quad.view(-1)
        lnorm += (class_quads).view(len(self), 1) + m_quad
        lnorm += 2 * torch.sum(
            noise_means * (class_means.view(len(self), 1, -1) + \
            global_mean.view(1, 1, -1)), dim=-1)
        lnorm += 2 * (class_means @ global_mean).view(-1, 1)

        deltas = -2 * torch.sum(means * data.view(1, npoints, -1), dim=-1)
        deltas += data_quad.view(1, -1)
        deltas += lnorm

        # We store values necessary for the accumulation and to compute
        # the expected value of the natural parameters.
        self.cache['deltas'] = deltas
        self.cache['means'] = means
        self.cache['lnorm'] = lnorm
        self.cache['npoints'] = npoints

    def latent_posterior(self, stats):
        # Extract the portion of the s. statistics we need.
        data = stats[:, 1:]
        length = len(stats)

        prec = self.cache['prec']
        noise_s_quad = self.cache['noise_s_quad']
        noise_s_mean = self.cache['noise_s_mean']
        m_mean = self.cache['m_mean']
        class_means = self.cache['class_mean_mean'] @ self.cache['class_s_mean']

        # The covariance matrix is the same for all the latent variables.
        l_cov = torch.inverse(
            torch.eye(self._subspace1_dim, dtype=stats.dtype,
                      device=stats.device) + prec * noise_s_quad)

        # Compute the means conditioned on the class.
        data_mean = data.view(length, 1, -1) - class_means
        data_mean -= m_mean.view(1, 1, -1)
        l_means = prec *  data_mean @ noise_s_mean.t() @ l_cov

        # Intermediary computation that will be necessary to compute the
        # elbo / accumuate the statistics.
        self.cache['noise_means'] = torch.stack([
            l_means[:, i, :] @ noise_s_mean
            for i in range(len(self))
        ])
        l_quad = torch.stack([
            l_cov + l_means[:, i, :, None] * l_means[:, i, None, :]
            for i in range(len(self))
        ])
        self.cache['l_quads'] = l_quad.view(len(self), len(stats), -1)
        self.cache['l_means'] = l_means
        self.cache['l_cov'] = l_cov

        self.cache['l_kl_divs'] = torch.stack([
            kl_div_std_norm(l_means[:, i, :], l_cov)
            for i in range(len(self))
        ])

        return l_means, l_cov

    ####################################################################
    # BayesianModel interface.
    ####################################################################

    @property
    def grouped_parameters(self):
        return [
            [self.precision_param],
            [*self.class_mean_params, self.mean_param],
            [self.class_subspace_param, self.noise_subspace_param],
        ]

    def sufficient_statistics(self, data):
        stats = torch.cat([torch.sum(data ** 2, dim=1).view(-1, 1), data],
                          dim=-1)

        # To avoid repeating many computation we compute and store in
        # the cache all the necessary intermediate results.
        self._precompute(stats)

        return stats

    def float(self):
        return self.__class__(
            self.mean_param.prior.float(),
            self.mean_param.posterior.float(),
            self.precision_param.prior.float(),
            self.precision_param.posterior.float(),
            self.noise_subspace_param.prior.float(),
            self.noise_subspace_param.posterior.float(),
            self.class_subspace_param.prior.float(),
            self.class_subspace_param.posterior.float(),
            [param.prior.float() for param in self.class_mean_params],
            [param.posterior.float() for param in self.class_mean_params]
        )

    def double(self):
        return self.__class__(
            self.mean_param.prior.double(),
            self.mean_param.posterior.double(),
            self.precision_param.prior.double(),
            self.precision_param.posterior.double(),
            self.noise_subspace_param.prior.double(),
            self.noise_subspace_param.posterior.double(),
            self.class_subspace_param.prior.double(),
            self.class_subspace_param.posterior.double(),
            [param.prior.double() for param in self.class_mean_params],
            [param.posterior.double() for param in self.class_mean_params]
        )

    def to(self, device):
        return self.__class__(
            self.mean_param.prior.to(device),
            self.mean_param.posterior.to(device),
            self.precision_param.prior.to(device),
            self.precision_param.posterior.to(device),
            self.noise_subspace_param.prior.to(device),
            self.noise_subspace_param.posterior.to(device),
            self.class_subspace_param.prior.to(device),
            self.class_subspace_param.posterior.to(device),
            [param.prior.to(device) for param in self.class_mean_params],
            [param.posterior.to(device) for param in self.class_mean_params]
        )

    def forward(self, s_stats, latent_variables=None):
        # Load the necessary value from the cache.
        prec = self.cache['prec']
        log_prec = self.cache['log_prec']
        deltas = self.cache['deltas']

        exp_llhs = -.5 * (prec * deltas - self._data_dim * log_prec + \
            self._data_dim * math.log(2 * math.pi))

        return exp_llhs.t()

    def accumulate(self, s_stats, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        resps = parent_msg

        dtype = s_stats.dtype
        device = s_stats.device

        # Separate the s. statistics.
        _, data = s_stats[:, 0], s_stats[:, 1:]

        # Load cached values and clear the cache.
        deltas = self.cache['deltas']
        l_means = self.cache['l_means']
        l_quads = self.cache['l_quads']
        prec = self.cache['prec']
        noise_means = self.cache['noise_means']
        m_mean = self.cache['m_mean']
        class_s_mean = self.cache['class_s_mean']
        class_s_quad = self.cache['class_s_quad']
        class_mean_mean = self.cache['class_mean_mean']
        class_mean_quad = self.cache['class_mean_quad']
        class_means = class_mean_mean @ class_s_mean
        self.clear_cache()

        data_noise_class_mean = data - (noise_means + class_means[:, None, :])
        data_noise_class_mean = (resps.t()[:, :, None] * data_noise_class_mean).sum(dim=0)
        acc_mean = torch.sum(data_noise_class_mean, dim=0)

        acc_noise_s_stats1 = (resps.t()[:, :, None] * l_quads).sum(dim=0)
        acc_noise_s_stats1 = acc_noise_s_stats1.sum(dim=0)
        acc_noise_s_stats1 = acc_noise_s_stats1.view(self._subspace1_dim,
                                                     self._subspace1_dim)
        acc_noise_s_stats1 = make_symposdef(acc_noise_s_stats1)
        data_class_means = (class_means + m_mean[None, :])
        data_class_means = resps.t()[:, :, None] * (data[None, :, :] - \
            data_class_means[:, None, :])
        acc_noise_s_stats2 = torch.stack([
            l_means[:, i, :].t() @ data_class_means[i]
            for i in range(len(self))
        ]).sum(dim=0).view(-1)

        acc_class_s_stats1 = \
            (resps @ class_mean_quad.view(len(self), -1)).sum(dim=0)
        acc_class_s_stats1 = acc_class_s_stats1.view(self._subspace2_dim,
                                                     self._subspace2_dim)
        acc_class_s_stats1 = make_symposdef(acc_class_s_stats1)
        data_noise_means = (data - noise_means - m_mean)
        acc_means = (resps.t()[:, :, None] * data_noise_means).sum(dim=1)
        acc_class_s_stats2 = acc_means[:, :, None] * class_mean_mean[:, None, :]
        acc_class_s_stats2 = acc_class_s_stats2.sum(dim=0).t().contiguous().view(-1)
        acc_stats = {
            self.precision_param: torch.cat([
                .5 * torch.tensor(len(s_stats) * self._data_dim, dtype=dtype,
                                  device=device).view(1),
                -.5 * torch.sum(resps.t() * deltas).view(1)
            ]),
            self.mean_param: torch.cat([
                - .5 * torch.tensor(len(s_stats) * prec, dtype=dtype,
                                    device=device),
                prec * acc_mean
            ]),
            self.noise_subspace_param:  torch.cat([
                - .5 * prec * acc_noise_s_stats1.view(-1),
                prec * acc_noise_s_stats2
            ]),
            self.class_subspace_param:  torch.cat([
                - .5 * prec * acc_class_s_stats1.view(-1),
                prec * acc_class_s_stats2
            ])
        }

        # Accumulate the statistics for the class means.
        class_mean_acc_stats1 = class_s_quad
        class_mean_acc_stats1 = make_symposdef(class_mean_acc_stats1)
        w_data_noise_means = resps.t()[:, :, None] * data_noise_means
        class_mean_acc_stats2 = \
            (class_s_mean @ w_data_noise_means.sum(dim=1).t()).t()
        for i, mean_param in enumerate(self.class_mean_params):
            class_mean_acc_stats = {
                mean_param: torch.cat([
                    -.5 * prec * resps[:, i].sum() * class_mean_acc_stats1.view(-1),
                    prec * class_mean_acc_stats2[i]
                ])
            }
            acc_stats.update(class_mean_acc_stats)

        return acc_stats

    ####################################################################
    # BayesianModelSet interface.
    ####################################################################

    def __getitem__(self, key):
        _, class_s_mean = self.class_subspace_param.expected_value(concatenated=False)
        _, class_mean = self.class_mean_params[key].expected_value(concatenated=False)
        mean = self.mean + class_s_mean.t() @ class_mean
        _, noise_s_mean = self.noise_subspace_param.expected_value(concatenated=False)
        cov = noise_s_mean.t() @ noise_s_mean
        cov += torch.eye(self._data_dim, dtype=cov.dtype,
                         device=cov.device) / self.precision
        return NormalSetElement(mean=mean, cov=cov)

    def __len__(self):
        return len(self.class_mean_params)

    def local_kl_div_posterior_prior(self, parent_msg=None):
        if parent_msg is None:
            raise ValueError('"parent_msg" should not be None')
        resps = parent_msg
        return torch.sum(resps * self.cache['l_kl_divs'].t(), dim=-1)

    def expected_natural_params_from_resps(self, resps):
        means = self.cache['means']
        prec = self.cache['prec']
        log_prec = self.cache['log_prec']
        lnorm = self.cache['lnorm']

        broadcasting_array = torch.ones(len(resps), self._data_dim,
                                        dtype=means.dtype, device=means.device)
        np1 = -.5 * prec * broadcasting_array
        np2 = prec * (resps.t()[:, :, None] * means).sum(dim=0)
        np3 = -.5 * prec * (lnorm.t() * resps).sum(dim=-1).view(-1, 1) * \
            broadcasting_array / self._data_dim
        np4 = .5 * log_prec * broadcasting_array

        return torch.cat([np1, np2, np3, np4], dim=-1)

    ####################################################################
    # VAELatentPrior interface.
    ####################################################################

    def sufficient_statistics_from_mean_var(self, mean, var):
        stats = torch.cat([torch.sum(mean ** 2 + var, dim=1).view(-1, 1), mean],
                          dim=1)

        # To avoid repeating many computation we compute and store in
        # the cache all the necessary intermediate results.
        self._precompute(stats)

        return stats


__all__ = ['PPCA', 'PLDASet']
