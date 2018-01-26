
from .priors import NormalGammaPrior

from .models import NormalDiagonalCovariance
from .models import NormalFullCovariance
from .models import Mixture
from .models import NestedMixture

from . import models

from .training import train_vae, train_conj_exp
