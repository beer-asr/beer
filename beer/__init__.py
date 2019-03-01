'''BEER -- the Bayesian spEEch Recognizer.

BEER is a machine learning library focused on Bayesian Generative Models
for speech technologies. 

'''

from .models import *
from .inference import *
from . import features
from . import nnet
from . import dists
from . import graph

import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module='beer')
