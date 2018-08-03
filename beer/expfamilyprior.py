'''Implementation of the Conjugate prior from the exponential family
of distribution.

'''

import abc
import math
import torch
import torch.autograd as ta


def _bregman_divergence(f_val1, f_val2, grad_f_val2, val1, val2):
    return f_val1 - f_val2 - torch.sum(grad_f_val2 * (val1 - val2))


# The following code compute the log of the determinant of a
# positive definite matrix. This is equivalent to:
#   >>> torch.log(torch.det(mat))
# Note: the hook is necessary to correct the gradient as pytorch
# will return upper triangular gradient.
def _logdet(mat):
    if mat.requires_grad:
        mat.register_hook(lambda grad: .5 * (grad + grad.t()))
    return 2 * torch.log(torch.diag(torch.potrf(mat))).sum()


class ExpFamilyPrior(metaclass=abc.ABCMeta):
    '''Abstract base class for (conjugate) priors from the exponential
    family of distribution. Prior distributions subclassing
    ``ExpFamilyPrior`` are of the form:

    .. math::
       p(x | \\theta ) = \\exp \\big\\{ \\eta(\\theta)^T T(x)
        - A\\big(\\eta(\\theta) \\big) \\big\\}

    where

      * :math:`x` is the parameter for a model for which we want to
        have a prior/posterior distribution.
      * :math:`\\theta` is the set of *standard hyper-parameters*
      * :math:`\\eta(\\theta)` is the vector of *natural
        hyper-parameters*
      * :math:`T(x)` are the sufficient statistics.
      * :math:`A\\big(\\eta(\\theta) \\big)` is the log-normalizing
        function

    '''

    @staticmethod
    def kl_div(model1, model2):
        '''Kullback-Leibler divergence between two densities of the same
        type from the exponential familyt of distribution. For this
        particular case, the divergence is defined as:

        .. math::
           D(q || p) = (\\eta_q - \\eta_p) \\langle T(x) \\rangle_q +
                A(\\eta_p) - A(\\eta_q)

        Args:
            model1 (:any:`beer.ExpFamilyPrior`): First model.
            model2 (:any:`beer.ExpFamilyPrior`): Second model.

        Returns
            float: Value of te KL. divergence between these two models.

        '''
        return _bregman_divergence(
            model2._log_norm_value,
            model1._log_norm_value,
            model1.expected_sufficient_statistics,
            model2.natural_hparams,
            model1.natural_hparams
        ).detach()

    def __init__(self, natural_hparams):
        '''Initialize the base class.

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Note:
            When subclassing ``beer.ExpFamilyPrior``, the child class
            should call the ``__init__`` method.

            .. code-block:: python

               class MyPrior(beer.ExpFamilyPrior):

                   def __init__(self, std_hparams):
                        # Transfrom the standard hyper-parameters into
                        # the natural hyper-parameters.
                        natural_hparams = transform(std_hparams)
                        super().__init__(natural_hparams)

                   ...

        '''
        # This will be initialized when setting the natural params
        # property.
        self._expected_sufficient_statistics = None
        self._natural_hparams = None
        self._log_norm_value = None

        self.natural_hparams = natural_hparams

    def __repr__(self):
        return self.__class__.__name__.replace('Prior', '')

    @property
    def expected_sufficient_statistics(self):
        '''``torch.Tensor``: Expected value of the sufficient statistics.

        .. math::
           \\langle T(x) \\rangle_{p(x | \\theta)} = \\nabla_{\\eta} \\;
                A\\big(\\eta(\\theta) \\big)

        '''
        return self._expected_sufficient_statistics

    @property
    def natural_hparams(self):
        '``torch.Tensor``: Natural hyper-parameters vector.'
        return self._natural_hparams.detach()

    @natural_hparams.setter
    def natural_hparams(self, value):
        if value.grad is not None:
            value.grad.zero_()
        copied_value = torch.tensor(value.detach(), requires_grad=True)
        log_norm_value = self.log_norm(copied_value).clone()
        ta.backward(log_norm_value)
        self._expected_sufficient_statistics = torch.tensor(copied_value.grad)
        self._natural_hparams = copied_value
        self._log_norm_value = torch.tensor(log_norm_value)

    def float(self):
        '''Create a new :any:`ExpFamilyPrior` with all the parameters set
        to float precision.

        Returns:
            :any:`ExpFamilyPrior`

        '''
        return self.copy_with_new_params(self.natural_hparams.float())

    def double(self):
        '''Create a new :any:`ExpFamilyPrior` with all the parameters set to
        double precision.

        Returns:
            :any:`ExpFamilyPrior`

        '''
        return self.copy_with_new_params(self.natural_hparams.double())

    def to(self, device):
        '''Create a new :any:`ExpFamilyPrior` with all the parameters
        allocated on `device`.

        Returns:
            :any:`ExpFamilyPrior`

        '''
        return self.copy_with_new_params(self.natural_hparams.to(device))

    @abc.abstractmethod
    def copy_with_new_params(self, params):
        '''Abstract method to be implemented by subclasses of
        ``beer.ExpFamilyPrior``.

        Copy the prior and set new (natural) parameters.

        Args:
            params (``torch.Tensor``): New natural parameters.

        Returns:
            :any:`ExpFamilyPrior`

        '''
        pass

    @abc.abstractmethod
    def split_sufficient_statistics(self, s_stats):
        '''Abstract method to be implemented by subclasses of
        ``beer.ExpFamilyPrior``.

        Split the sufficient statistics vector into meaningful groups.
        The notion of *meaningful group* depends on the type of the
        subclass. For instance, the sufficient statistics of the
        Normal density are :math:`T(x) = (x^2, x)^T` leading to
        the following groups: :math:`x^2` and :math:`x`.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            A ``torch.Tensor`` or a tuple of ``torch.Tensor`` depending
            on the type of density.

        '''
        pass

    @abc.abstractmethod
    def log_norm(self, natural_hparams):
        '''Abstract method to be implemented by subclasses of
        ``beer.ExpFamilyPrior``.

        Log-normalizing function of the density.

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        pass


class JointExpFamilyPrior(ExpFamilyPrior):
    '''Composite prior of K independent distributions from the
    exponential family.

    '''
    def __init__(self, priors):
        '''
        Args:
            prior (list): List of :any:`ExpFamilyPrior`` to combine.
        '''
        self._priors = priors
        self._dims = [len(prior.natural_hparams) for prior in priors]
        n_hparams = torch.cat([prior.natural_hparams for prior in self._priors])
        super().__init__(n_hparams)

    def copy_with_new_params(self, params):
        raise NotImplementedError

    def split_sufficient_statistics(self, s_stats):
        previous_dim = 0
        retval = []
        for prior, dim in zip(self._priors, self._dims):
            stats = s_stats[previous_dim: previous_dim + dim]
            retval.append(prior.split_sufficient_statistics(stats))
            previous_dim += dim
        return retval

    def log_norm(self, natural_hparams):
        previous_dim = 0
        lnorm = 0
        for prior, dim in zip(self._priors, self._dims):
            n_hparams = natural_hparams[previous_dim: previous_dim + dim]
            lnorm += prior.log_norm(n_hparams)
            previous_dim += dim
        return lnorm


class DirichletPrior(ExpFamilyPrior):
    '''The Dirichlet density defined as:

    .. math::
       p(x | \\alpha) = \\frac{\\Gamma(\\sum_{i=1}^K \\alpha_i)}
            {\\prod_{i=1}^K \\Gamma(\\alpha_i)}
            \\prod_{i=1}^K x_i^{\\alpha_i - 1}

    where :math:`\\alpha` is the concentration parameter.

    '''

    def __init__(self, concentrations):
        '''
        Args:
            concentrations (``torch.Tensor``): Concentration for each
                dimension.
        '''
        natural_hparams = torch.tensor(concentrations - 1, requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''For the Dirichlet density, this is simply the identity
        function as there is only a single "group" of sufficient
        statistics.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: ``s_stats`` unchanged.

        '''
        return s_stats

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        return - torch.lgamma((natural_hparams + 1).sum()) +\
            torch.lgamma(natural_hparams + 1).sum()


class IsotropicNormalGammaPrior(ExpFamilyPrior):
    '''The (isotropic) Normal-Gamma density defined as:

    .. math::
       p(\\mu, \\lambda | m, \\kappa, a, b) = \\mathcal{N} \\big(\mu | m,
        (\\kappa \; \\text{diag}(\\lambda))^{-1} \\big)
        \\mathcal{G} \\big( \\lambda | a, b \\big)

    where:

      * :math:`\\mu`, :math:`\\lambda` are the mean and the diagonal
        of the precision matrix of a multivariate normal density.
      * :math:`m` is the hyper-parameter mean of the Normal density.
      * :math:`\\kappa` is the hyper-parameter scale of the Normal
        density.
      * :math:`a` is the hyper-paramater shape of the Gamma density.
      * :math:`b` is the hyper-parameter rate of the Gamma density.

    Note:
        Strictly speaking, the Normal-Gamma density is a
        distribution over a 1 dimensional mean and precision parameter.
        In our case, :math:`\\mu` and :math:`\\lambda` are a
        D-dimensional vector and the diagonal of a :math:`D \\times D`
        precision matrix respectively. The ``beer.NormalGammaPrior``
        can be seen as the concatenation of :math:`D` indenpendent
        "standard" Normal-Gamma densities.

    '''

    def __init__(self, mean, scale, shape, rate):
        '''
        Args:
            mean (``torch.Tensor[d]``): Mean of the Normal.
            scale (``torch.Tensor[1]``): Scale of the Normal.
            shape (``torch.Tensor[1]``): Shape parameter of the Gamma.
            rate (``torch.Tensor[1]``): Rate parameter of the Gamma.

        '''
        natural_hparams = torch.tensor(torch.cat([
            (scale * (mean ** 2).sum() + 2 * rate).view(1),
            scale * mean,
            scale.view(1),
            ((2 * (shape - 1) / len(mean)) + 1).view(1)
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 4 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: ``s_stats`` unchanged.

        '''
        return s_stats[0], s_stats[1: -2], s_stats[-2], s_stats[-1]

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        np1, np2, np3, np4 = self.split_sufficient_statistics(natural_hparams)
        dim = len(np2)
        shape = .5 * dim * (np4 - 1) + 1
        rate = .5 * (np1 - ((np2 ** 2).sum() / np3))
        scale = np3
        lognorm = torch.lgamma(shape)
        lognorm += -.5 * dim * torch.log(scale)
        lognorm += -shape * torch.log(rate)
        return lognorm


class JointIsotropicNormalGammaPrior(ExpFamilyPrior):
    '''Joint isotropic NormalGamma prior.'''

    def __init__(self, means, scales, shape, rate):
        '''
        Args:
            means (``torch.Tensor[k,d]``): Mean of the Normals.
            scales (``torch.Tensor[k]``): Scale of the Normals.
            shape (``torch.Tensor[1]``): Shape parameter of the Gamma.
            rate (``torch.Tensor[1]``): Rate parameter of the Gamma.

        '''
        self.ncomp, self.dim = means.size()
        natural_hparams = torch.tensor(torch.cat([
            ((scales * (means**2).sum(dim=1)).sum() + 2 * rate).view(-1),
            (scales[:, None] * means).view(-1),
            scales.view(-1),
            ((2 * (shape - 1) / self.dim) + self.ncomp).view(1)
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.ncomp = self.ncomp
        new_instance.dim = self.dim
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        hnp1 = s_stats[0]
        hnp2s = s_stats[1: 1 + self.dim * self.ncomp]
        hnp3s = s_stats[1 + self.dim * self.ncomp:
                        1 + self.dim * self.ncomp + self.ncomp]
        hnp4 = s_stats[-1]
        return hnp1, hnp2s.view(self.ncomp, self.dim), hnp3s, hnp4

    def log_norm(self, natural_hparams):
        hnp1, hnp2s, hnp3s, hnp4 = self.split_sufficient_statistics(
            natural_hparams)
        shape = .5 * (self.dim * hnp4 - self.dim * self.ncomp + 2)
        rate = .5 * (hnp1 - ((hnp2s ** 2).sum(dim=1) / hnp3s).sum())
        scales = hnp3s
        lognorm = torch.lgamma(shape)
        lognorm -= shape * torch.log(rate)
        lognorm -= .5 * self.dim * torch.log(scales).sum()
        return lognorm


class NormalGammaPrior(ExpFamilyPrior):
    '''The Normal-Gamma density defined as:

    .. math::
       p(\\mu, \\lambda | m, \\kappa, a, b) = \\mathcal{N} \\big(\mu | m,
        (\\kappa \; \\text{diag}(\\lambda))^{-1} \\big)
        \\mathcal{G} \\big( \\lambda | a, b \\big)

    where:

      * :math:`\\mu`, :math:`\\lambda` are the mean and the diagonal
        of the precision matrix of a multivariate normal density.
      * :math:`m` is the hyper-parameter mean of the Normal density.
      * :math:`\\kappa` is the hyper-parameter scale of the Normal
        density.
      * :math:`a` is the hyper-paramater shape of the Gamma density.
      * :math:`b` is the hyper-parameter rate of the Gamma density.

    Note:
        Strictly speaking, the Normal-Gamma density is a
        distribution over a 1 dimensional mean and precision parameter.
        In our case, :math:`\\mu` and :math:`\\lambda` are a
        D-dimensional vector and the diagonal of a :math:`D \\times D`
        precision matrix respectively. The ``beer.NormalGammaPrior``
        can be seen as the concatenation of :math:`D` indenpendent
        "standard" Normal-Gamma densities.

    '''

    def __init__(self, mean, scale, shape, rate):
        '''
        Args:
            mean (``torch.Tensor``): Mean of the Normal.
            scale (``torch.Tensor``): Scale of the Normal.
            shape (``torch.Tensor``): Shape parameter of the Gamma.
            rate (``torch.Tensor``): Rate parameter of the Gamma.

        '''
        natural_hparams = torch.tensor(torch.cat([
            scale * (mean ** 2) + 2 * rate,
            scale * mean,
            scale,
            2 * shape - 1
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 4 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: ``s_stats`` unchanged.

        '''
        return tuple(s_stats.view(4, -1))

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        np1, np2, np3, np4 = self.split_sufficient_statistics(natural_hparams)
        lognorm = torch.lgamma(.5 * (np4 + 1))
        lognorm += -.5 * torch.log(np3)
        lognorm += -.5 * (np4 + 1) * torch.log(.5 * (np1 - ((np2**2) / np3)))
        return torch.sum(lognorm)


class JointNormalGammaPrior(ExpFamilyPrior):
    '''Joint NormalGamma is the distribution over a set of
    :math:`D` dimensional mean vectors :math:`M = (\\mu_1, ...,
    \\mu_K)^T` and the diagonal of a precision matrix :math:`\\lambda`.
    It is defined as:

    .. math::
       p(M, \\lambda | M, \\kappa, a, b) = \\big\\lbrack \\prod_{i=1}^K
            \\mathcal{N} \\big(\\mu_i | m_i, (\\kappa_i \;
            \\text{diag}(\\lambda))^{-1} \\big)  \\big\\rbrack
            \\mathcal{G} \\big( \\lambda | a, b \\big)

    The parameters are defined in the same way as for the
    :any:`beer.NormalGammaPrior`

    Attributes:
        dim (int): Dimension of the mean parameter.
        ncomp (int): Number of Normal densities.
    '''

    def __init__(self, means, scales, shape, rate):
        '''
        Args:
            means (``torch.Tensor``): Mean of the Normal.
            scales (``torch.Tensor``): Scale of the Normal.
            shape (``torch.Tensor``): Shape parameter of the Gamma.
            rate (``torch.Tensor``): Rate parameter of the Gamma.

        '''
        self.ncomp, self.dim = means.size()
        natural_hparams = torch.tensor(torch.cat([
            ((scales * means**2).sum(dim=0) + 2 * rate).view(-1),
            (scales * means).view(-1),
            scales.view(-1),
            2 * shape - 1
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.ncomp = self.ncomp
        new_instance.dim = self.dim
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 4 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        hnp1 = s_stats[:self.dim]
        hnp2s = s_stats[self.dim: self.dim + self.dim * self.ncomp]
        hnp3s = s_stats[self.dim + self.dim * self.ncomp:
                        self.dim + 2 * self.dim * self.ncomp]
        hnp4 = s_stats[self.dim + 2 * self.dim * self.ncomp:]
        return hnp1, hnp2s.view(self.ncomp, self.dim), \
            hnp3s.view(self.ncomp, self.dim), hnp4

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        hnp1, hnp2s, hnp3s, hnp4 = self.split_sufficient_statistics(
            natural_hparams)
        lognorm = torch.lgamma(.5 * (hnp4 + 1)).sum()
        lognorm += -.5 * torch.log(hnp3s).sum()
        tmp = ((hnp2s ** 2) / hnp3s).view(self.ncomp, self.dim)
        lognorm += torch.sum(-.5 * (hnp4 + 1) * \
            torch.log(.5 * (hnp1 - tmp.sum(dim=0))))
        return lognorm


class NormalWishartPrior(ExpFamilyPrior):
    '''The Normal-Wishart density defined as:

    .. math::
       p(\\mu, \\Lambda | m, \\kappa, W, \\nu) = \\mathcal{N} \\big(\mu |
        m, (\\kappa \; \\Lambda)^{-1} \\big)
        \\mathcal{W} \\big( \\Lambda | w, \\nu \\big)

    where:

      * :math:`\\mu`, :math:`\\Lambda` are the mean and the precision
        omatrix of a multivariate normal density.
      * :math:`m` is the hyper-parameter mean of the Normal density.
      * :math:`\\kappa` is the hyper-parameter scale of the Normal
        density (scalar).
      * :math:`W` is the hyper-paramater scale matrix of the Wishart
        density.
      * :math:`\\nu` is the hyper-parameter degree of freedom of the
        Wishart density.

    Attributes:
        dim (int): Dimension of the mean parameter.

    '''

    def __init__(self, mean, scale, scale_matrix, dof):
        '''
        Args:
            mean (``torch.Tensor``): Mean of the Normal.
            scale (float): Scale of the normal.
            scale_matrix (``torch.Tensor``): Scale matrix of the
                Wishart.
            dof (float): Degree of freedom of the Wishart.
        '''
        self.dim = mean.size(0)
        inv_scale = torch.inverse(scale_matrix)
        natural_hparams = torch.tensor(torch.cat([
            (scale * torch.ger(mean, mean) + inv_scale).view(-1),
            scale * mean,
            (torch.ones(1) * scale).type(mean.type()),
            (torch.ones(1) * (dof - self.dim)).type(mean.type())
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.dim = self.dim
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 4 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        grp1, grp2 = s_stats[:self.dim ** 2].view(self.dim, self.dim), \
            s_stats[self.dim ** 2:-2]
        grp3, grp4 = s_stats[-2:]
        return grp1, grp2, grp3, grp4

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        hnp1, hnp2, hnp3, hnp4 = self.split_sufficient_statistics(
            natural_hparams)
        lognorm = .5 * ((hnp4 + self.dim) * self.dim * math.log(2) - \
            self.dim * torch.log(hnp3))
        lognorm += -.5 * (hnp4 + self.dim) * \
            _logdet(hnp1 - torch.ger(hnp2, hnp2) / hnp3)
        seq = torch.arange(1, self.dim + 1, 1).type(natural_hparams.type())
        lognorm += torch.lgamma(.5 * (hnp4 + self.dim + 1 - seq)).sum()
        return lognorm


class JointNormalWishartPrior(ExpFamilyPrior):
    '''Joint Normal-Wishart is the distribution over a set of
    :math:`D` dimensional mean vectors :math:`M = (\\mu_1, ...,
    \\mu_K)^T` and a precision matrix :math:`\\Lambda`.
    It is defined as:

    .. math::
       p(M, \\lambda | M, \\kappa, W, \\nu) = \\big\\lbrack \\prod_{i=1}^K
            \\mathcal{N} \\big(\\mu_i | m_i, (\\kappa_i \;
            \\Lambda)^{-1} \\big)  \\big\\rbrack
            \\mathcal{W} \\big( \\Lambda | W, \\nu \\big)

    The parameters are defined in the same way as for the
    :any:`beer.NormalWishartPrior`

    Attributes:
        dim (int): Dimension of the mean parameter.
        ncomp (int): Number of Normal densities.
    '''

    def __init__(self, means, scales, scale_matrix, dof):
        '''
        Args:
            means (``torch.Tensor``): Means of the Normal densities.
            scales (float): s of the Normal densities.
            scale_matrix (``torch.Tensor``): Scale matrix of the
                Wishart.
            dof (float): Degree of freedom of the Wishart.
        '''
        self.ncomp, self.dim = means.size()
        inv_scale = torch.inverse(scale_matrix)
        mmT = ((scales.view(-1, 1) * means)[:, None, :] * \
            means[:, :, None]).sum(dim=0)
        natural_hparams = torch.tensor(torch.cat([
            (mmT + inv_scale).view(-1),
            (scales.view(-1, 1) * means).view(-1),
            scales,
            (torch.ones(1, dtype=means.dtype, device=means.device) * \
                (dof - self.dim))
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.ncomp = self.ncomp
        new_instance.dim = self.dim
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 4 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        grp1 = s_stats[:self.dim ** 2].view(self.dim, self.dim)
        grp2s = s_stats[self.dim ** 2:-(self.ncomp + 1)].view(self.ncomp,
                                                              self.dim)
        grp3s = s_stats[-(self.ncomp + 1):-1]
        grp4 = s_stats[-1]
        return grp1, grp2s, grp3s, grp4

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        hnp1, hnp2s, hnp3s, hnp4 = self.split_sufficient_statistics(
            natural_hparams)
        lognorm = .5 * ((hnp4 + self.dim) * self.dim * math.log(2) - \
            self.dim * torch.log(hnp3s).sum())
        quad_exp = ((hnp2s[:, None, :] * hnp2s[:, :, None]) / \
            hnp3s[:, None, None]).sum(dim=0)
        lognorm += -.5 * (hnp4 + self.dim) * _logdet(hnp1 - quad_exp)
        seq = torch.arange(1, self.dim + 1, 1, dtype=natural_hparams.dtype,
                           device=natural_hparams.device)
        lognorm += torch.lgamma(.5 * (hnp4 + self.dim + 1 - seq)).sum()
        return lognorm


class NormalFullCovariancePrior(ExpFamilyPrior):
    '''Normal density prior:

    .. math::
       p(\\mu | m, \\Sigma) = \\mathcal{N} \\big( \\mu | m, \\Sigma
            \\big)

    where:

      * :math:`\\mu` is the mean of multivariate Normal density.
      * :math:`m` is the hyper-parameter mean of the Normal prior.
      * :math:`\\Sigma` is the hyper-parameter covariance matrix of the
         Normal prior.

    Attributes:
        dim (int): Dimension of the mean parameter.

    '''

    def __init__(self, mean, cov):
        '''
        Args:
            mean (``torch.Tensor``): Hyper-parameter mean.
            cov (``torch.Tensor``): Hyper-parameter covariance matrix.
        '''
        self.dim = len(mean)
        prec = torch.inverse(cov)
        natural_hparams = torch.tensor(torch.cat([
            -.5 * prec.contiguous().view(-1),
            prec @ mean,
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.dim = self.dim
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 2 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        grp1, grp2 = s_stats[:self.dim ** 2].view(self.dim, self.dim), \
            s_stats[self.dim ** 2:]
        return grp1, grp2

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        hnp1, hnp2 = self.split_sufficient_statistics(
            natural_hparams)
        inv_hnp1 = torch.inverse(hnp1)
        return -.5 * _logdet(-2 * hnp1) - \
            .25 * ((hnp2[None, :] @ inv_hnp1) @ hnp2)[0]


class NormalIsotropicCovariancePrior(ExpFamilyPrior):
    '''Normal density prior with an isotropic covariance matrix:

    .. math::
       p(\\mu | m, \\tau) = \\mathcal{N} \\big( \\mu | m, \\tau I \\big)

    where:

      * :math:`\\mu` is the mean of multivariate Normal density.
      * :math:`m` is the hyper-parameter mean of the Normal prior.
      * :math:`\\tau` is the hyper-parameter variance of the
        Normal prior.

    Attributes:
        dim (int): Dimension of the mean parameter.
    '''

    def __init__(self, mean, variance):
        '''
        Args:
            mean (``torch.Tensor``): Mean hyper-parameter of the prior.
            variance (float): Globale variance hyper-parameter of the
                prior.
        '''
        prec = 1 / variance
        natural_hparams = torch.tensor(torch.cat([
            -.5 * prec,
            prec * mean,
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        super(type(new_instance), new_instance).__init__(params)
        return new_instance


    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 2 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        return s_stats[0], s_stats[1:]

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        hnp1, hnp2 = self.split_sufficient_statistics(natural_hparams)
        inv_hnp1 = 1 / hnp1
        logdet = len(hnp2) * torch.log(-2 * hnp1)
        return -.5 * logdet - .25 * inv_hnp1 * (hnp2[None, :] @ hnp2)


class MatrixNormalPrior(ExpFamilyPrior):
    '''Matrix Normal density prior over a real matrix parameter:

    .. math::
       p(M | U, \\Sigma) = \\mathcal{N} \\big( M | U, \\Sigma \\big)

    where:

      * :math:`M` is a :math:`q \\times d` real matrix.
      * :math:`U` is the :math:`q \\times d` hyper-parameter mean of the
        matrix Normal prior.
      * :math:`\\Sigma` is the :math:`q \\times q` hyper-parameter
        covariance variance of the matrix Normal prior.

    Attributes:
        dim1 (int): Number of rows of :math:`M`.
        dim2 (int): Number of columns of :math:`M`.
    '''

    def __init__(self, mean, cov):
        '''
        Args:
            mean (``torch.Tensor``): Hyper-parameter mean.
            cov (``torch.Tensor``): Hyper-parameter covariance matrix.
        '''
        self.dim1, self.dim2 = mean.size()
        prec = torch.inverse(cov)
        natural_hparams = torch.tensor(torch.cat([
            -.5 * prec.contiguous().view(-1),
            (prec @ mean).view(-1),
        ]), dtype=mean.dtype, device=mean.device, requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        new_instance.dim1 = self.dim1
        new_instance.dim2 = self.dim2
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 2 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        return s_stats[:self.dim1 ** 2].view(self.dim1, self.dim1), \
            s_stats[self.dim1 ** 2:].view(self.dim1, self.dim2)

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            natural_hparams (``torch.Tensor``): Natural hyper-parameters
                of the distribution.

        Returns:
            ``torch.Tensor`` of size 1: Log-normalization value.

        '''
        hnp1, hnp2 = self.split_sufficient_statistics(natural_hparams)
        inv_hnp1 = torch.inverse(hnp1)
        #mat1, mat2 = np2.t() @ inv_np1, np2
        #trace_mat1_mat2 = mat1.view(-1) @ mat2.t().contiguous().view(-1)
        _, logdet = torch.slogdet(-2 * hnp1)
        return -.5 * self.dim2 * logdet - \
            .25 * torch.trace(hnp2.t() @ inv_hnp1 @ hnp2)


class GammaPrior(ExpFamilyPrior):
    '''Gamma density prior:

    .. math::
       p(\\lambda | a, b) = \\mathcal{G} \\big( \\lambda | a, b \\big)

    where:

      * :math:`\\lambda` is the precision of Normal density.
      * :math:`a` is the shape hyper-parameter of the Gamma prior.
      * :math:`b` is the rate hyper-parameter of the Gamma prior.

    '''

    def __init__(self, shape, rate):
        '''
        Args:
            shape (float): Shape hyper-parameter.
            rate (float): Rate hyper-parameter.
        '''
        natural_hparams = torch.tensor(torch.cat([shape - 1, -rate]),
                                       requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 2 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: ``s_stats`` unchanged.

        '''
        return tuple(s_stats.view(2, -1))

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        hnp1, hnp2 = self.split_sufficient_statistics(natural_hparams)
        return torch.lgamma(hnp1 + 1) - (hnp1 + 1) * torch.log(-hnp2)


class WishartPrior(ExpFamilyPrior):
    'Wishart density prior.'

    def __init__(self, cov, dof):
        '''
        Args:
            inv_mean (float): Mean of the covariance matrix.
            dof (float): degree of freedom.
        '''
        natural_hparams = torch.tensor(torch.cat([
            -.5 * cov.view(-1),
            .5 * (dof - cov.shape[0] - 1).view(1),
        ]), requires_grad=True)
        super().__init__(natural_hparams)

    def copy_with_new_params(self, params):
        new_instance = self.__class__.__new__(self.__class__)
        super(type(new_instance), new_instance).__init__(params)
        return new_instance

    def split_sufficient_statistics(self, s_stats):
        '''Split the sufficient statistics into 2 groups.

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split.

        Returns:
            ``torch.Tensor``: ``s_stats`` unchanged.

        '''
        dim = int(math.sqrt(len(s_stats) - 1))
        stats1 = s_stats[:-1].view(dim, dim)
        stats2 = s_stats[-1]
        return stats1, stats2

    def log_norm(self, natural_hparams):
        '''Log-normalizing function

        Args:
            s_stats (``torch.Tensor``): Sufficients statistics to
                split

        Returns:
            ``torch.Tensor``: tuple of sufficient statistics.

        '''
        hnp1, hnp2 = self.split_sufficient_statistics(natural_hparams)
        dim = len(hnp1)

        # Get the standard parameters.
        W = -2 * hnp1
        dof = 2 * hnp2 + dim + 1

        # Log normalizer.
        lognorm = -.5 * dof * _logdet(W)
        lognorm += .5 * dof * dim * math.log(2)
        lognorm += .25 * dim * (dim - 1) * math.log(math.pi)
        seq = torch.arange(1, dim + 1, 1).type(natural_hparams.dtype)
        lognorm += torch.lgamma(.5 * (dof + 1 - seq)).sum()

        return lognorm


__all__ = [
    'ExpFamilyPrior', 'JointExpFamilyPrior', 'DirichletPrior', 'NormalGammaPrior',
    'JointNormalGammaPrior', 'NormalWishartPrior', 'JointNormalWishartPrior',
    'NormalFullCovariancePrior', 'NormalIsotropicCovariancePrior', 'GammaPrior',
    'MatrixNormalPrior', 'IsotropicNormalGammaPrior',
    'JointIsotropicNormalGammaPrior', 'WishartPrior'
]
