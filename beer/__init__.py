'''BEER -- the Bayesian spEEch Recognizer.'''

from .models import *
from .inference import *
from . import features
from . import nnet
from . import dists
from . import graph

import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module='beer')
