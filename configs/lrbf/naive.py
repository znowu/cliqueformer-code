import ml_collections
from ml_collections.config_dict import config_dict
import torch 
import torch.nn as nn 

def get_config():

    config = ml_collections.ConfigDict()

    config.data = {
        'cls': 'LRBF',
        'N': int(1e5),
        'd': 41 
    }

    config.model = {
        'cls': 'Naive',
        'hidden_dims': 2 * (2048,),
        'lr': 1e-3
    }

    config.learner = {
        'cls': 'GradientAscent',
        'design_steps': 50,
        'decay': 0.,
        'lr': 5e-2
    }

    return config 