
from .bayesmodel import *
from .normal import *
from .mixture import *
from .hmm import *
from .subspace import *
from .vae import *

import yaml


_model_types = {
    'Normal': None,
    'NormalSet': None,
    'Mixture': None,
    'HMM': None,
    'PPCA': None,
    'PLDA': None,
    'VAE': None,
}


def create_model(conf):
    '''Create one or several models from a YAML configuration string.

    Args:
        conf (string): YAML formatted string defining the model.

    Returns:
        :any:`BayesianModel` or a list of :any:`BayesianModel`

    '''
    model_conf = yaml.load(conf)
    requested_type = model_conf['type']
    if requested_type not in _model_types:
        raise ValueError('Unknown model type: {}'.format(requested_type))
    return _model_types[requested_type](model_conf)
