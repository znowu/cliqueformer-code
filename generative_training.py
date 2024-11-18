import numpy as np
import torch
import torch.nn as nn

import wandb
import pickle 
import os 

from absl import app, flags
from ml_collections import config_flags

from models import DDOM, DDOMDiscrete
from data import Dataset, LRBF, Superconductor, TFBind8, DNA
import data.extras as dx 
import models.graphops as graphops
 
FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', int(1), 'Random seed.') 
flags.DEFINE_integer('batch_size', int(512), 'Batch size.') 
flags.DEFINE_integer('model_steps', int(4e4), 'Model learning size.') 
flags.DEFINE_integer('N_eval', int(2e2), 'Evaluation frequency.')
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
    # Initialize a Wandb project
    #
    wandb.init(project='ddom-training')
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
    # Build model spec tring for saving
    #
    spec = ', '.join([f'{key}: {value}' for key, value in model_kwargs.items()])

    #
    # Initialize the dataset
    #
    data_kwargs["seed"] = FLAGS.seed
    dataset_cls = data_kwargs.pop('cls')
    dataset = globals()[dataset_cls](**data_kwargs)

    if dataset_cls not in ["DNA", "TFBind8"]:
        dataset.standardize_x()
    
    dataset.standardize_y()

    #
    # Split the dataset into train-test partition
    #
    dataset_train, dataset_test = dataset.split(FLAGS.split_ratio)

    #
    # Initialize the model
    #
    model_cls = model_kwargs.pop('cls')
    
    if "Discrete" in model_cls:
        model = globals()[model_cls](dataset.seq_len, dataset.dim, **model_kwargs).to(device) 

    else:
        model = globals()[model_cls](dataset.dim, **model_kwargs).to(device) 
    
    model.get_bins(dataset_train.y)
    model = nn.DataParallel(model)

    #
    # Train the model 
    #
    model.train()

    for step in range(FLAGS.model_steps):
        #
        # Draw a random batch and put it on torch device
        #   
        x, y = dataset_train.sample(FLAGS.batch_size)
        x, y = dx.move_to_device((x, y), device)

        #
        # Compute the loss and take a gradient step
        #
        info = model.module.training_step(x, y)

        #
        # Evaluate the model on the test set
        #
        if step % FLAGS.N_eval == 0:

            model.eval()

            x, y = dataset_test.sample(FLAGS.batch_size)
            x, y = dx.move_to_device((x, y), device)

            evalinfo = model.module.eval_step(x, y)

            wandb.log({f'eval/{k}': v for k, v in evalinfo.items()}, step=step)
            wandb.log({f'train/{k}': v for k, v in info.items()}, step=step)
            model.train()

    #
    # Save the model
    #
    model_dir = os.path.join('saved_models', model_cls, dataset_cls)
    if dataset_cls == 'LRBF':
        model_dir += str(data_kwargs['d'])

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_path = os.path.join(model_dir, spec) + '.pickle'
    with open(model_path, 'wb') as model_file:
        pickle.dump(model.module, model_file)


if __name__ == '__main__':
    app.run(main)