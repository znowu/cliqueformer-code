import numpy as np
import torch
import torch.nn as nn
import argparse

import wandb
import pickle 
import os 

from absl import app, flags
from ml_collections import config_flags

from models import DDOM, DDOMDiscrete
from optimization.design import Design
from optimization.lerners import GradientAscent, RWR, GradientAscentDiscrete, RWRDiscrete 
from data import Dataset, LRBF, Superconductor, DNA, TFBind8
import models.graphops as graphops

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', int(1), 'Random seed.') 
flags.DEFINE_integer('design_batch_size', int(1000), 'Design batch size.') 
flags.DEFINE_integer('design_mini_batch_size', int(100), 'Design mini-batch size.') 
flags.DEFINE_integer('top_k', int(10), 'The best designs for evaluation.') 
flags.DEFINE_float('split_ratio', 0.8, 'Train-test split.') 

config_flags.DEFINE_config_file(
    'config',
    'configs/tfbind8/ddom.py',
    'File with hyperparameter configurations.',
    lock_config=False
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(_):

    #
    # Parser to extract seed
    # 
    parser = argparse.ArgumentParser(description="Pass in the random seed.")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    args = parser.parse_args()
    seed = FLAGS.seed if args.seed is None else args.seed

    #
    # Initialize a Wandb project
    #
    wandb.init(project='ddom-generation')
    wandb.config.update(FLAGS)
    kwargs = dict(**FLAGS.config)
    torch.manual_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    #
    # Extract specific kwargs
    #
    data_kwargs = dict(kwargs['data'])
    model_kwargs = dict(kwargs['model'])

    #
    # Build model spec tring for loading
    #
    spec = ', '.join([f'{key}: {value}' for key, value in model_kwargs.items()])

    #
    # Initialize the dataset
    #
    data_kwargs["seed"] = FLAGS.seed
    dataset_cls = data_kwargs.pop('cls')
    dataset = globals()[dataset_cls](**data_kwargs)

    #
    # Change the seed for this run
    #
    torch.manual_seed(seed)
    np.random.seed(seed)

    if dataset_cls not in ["DNA", "TFBind8"]:
        dataset.standardize_x()
    
    dataset.standardize_y()

    #
    # Derive the model path
    #
    model_cls = model_kwargs['cls']
    model_dir = os.path.join('saved_models', model_cls, dataset_cls)

    if dataset_cls == 'LRBF':
        model_dir += str(data_kwargs['d'])

    model_path = os.path.join(model_dir, spec + '.pickle')

    #
    # Load the model
    #
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    model = nn.DataParallel(model)

    #
    # Sample designs
    #
    design = np.zeros((FLAGS.design_batch_size,) + dataset.x[0].shape)
    mini = FLAGS.design_mini_batch_size
    n_batches = FLAGS.design_batch_size // mini

    for i in range(n_batches):
        new_design = model.module.sample(mini, dataset.y.max())
        design[i * mini : (i + 1) * mini] = new_design.detach().cpu().numpy()

    #
    # Evaluate the designs
    # 
    true_val = dataset.evaluate(design, from_standardized_x=True, to_standardized_y=False)
    true_val = dataset.max_min_normalize(true_val)

    #
    # Count up nans
    #
    isnan = np.isnan(true_val)
    valid = 1 - isnan

    #
    # Remove nans
    #
    true_val = true_val[np.logical_not(np.isnan(true_val))]

    #
    # Get top-k of design values
    #
    true_val = true_val[np.argsort(true_val)[::-1][:FLAGS.top_k]]

    ascendinfo = {
        "true val": true_val.mean(),
        "true_val_max": true_val.max(),
        "true_val_std": true_val.std(),
        'valid': valid.mean()
    }

    wandb.log({f'ascend/{k}': v for k, v in ascendinfo.items()}, step=0)



if __name__ == '__main__':
    app.run(main)