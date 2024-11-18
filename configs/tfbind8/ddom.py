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
        'cls': 'DDOMDiscrete',
        'hidden_dims': 2 * (1024,),
        'gamma': 2,
        'n_timesteps': 1000,
        'K_factor': 0.1,
        'N_bins': 64,
        'temp': 0.1,
        'uncond_rate': 0.15,
        'lr': 1e-3
    }

    return config 