import ml_collections
from ml_collections.config_dict import config_dict
import torch 
import torch.nn as nn 

def get_config():

    config = ml_collections.ConfigDict()

    config.data = {
        'cls': 'TFBind8'
    }

    config.model = {
        'cls': 'NaiveDiscrete',
        'hidden_dims': 2 * (2048,),
        'lr': 1e-3
    }

    config.learner = {
        'cls': 'RWRDiscrete',
        'design_steps': 50,
        'temp': 0.1,
        'decay': 0.,
        'lr': 2
    }

    return config 