
from .priors import NormalGammaPrior

from .models import NormalDiagonalCovariance
from .models import NormalFullCovariance
from .models import Mixture
from .models import NestedMixture

from . import models

from .expfamily import ExpFamilyDensity
from .expfamily import kl_divergence
from .expfamily import dirichlet
from .expfamily import normalgamma
