# Copyright 2018 Brendan Duke.
#
# This file is part of Mobile Prune.
#
# Mobile Prune is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# Mobile Prune is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Mobile Prune. If not, see <http://www.gnu.org/licenses/>.

"""Configuration-related."""
import argparse
import os


class Choices:
    """For adding argparse choices config options."""

    pass


class ListParam:
    """For adding lists of params, in the JSON format [a, b, c]."""

    pass


class ListParamFloat(ListParam):
    """For adding lists of floats."""

    dtype = float


class ListParamInt(ListParam):
    """For adding lists of ints."""

    dtype = int


CONFIG_OPTIONS = [
    (bool,
     'is_baseline',
     """Run the (retrained) baseline, without learned fusion op?"""),

    (bool,
     'is_gumbel_discrete',
     """Discrete (or continous) Gumbel softmax output?"""),

    (bool,
     'retrain',
     """Retrain an architecture from scratch?"""),

    (str,
     'retrain_path',
     """Path to controller weights for re-training fusion operator."""),

    (str,
     'checkpoint_path',
     """Paths to take checkpoint files (e.g., inception_v3.ckpt) from.
     Checkpoint paths should be separated by commas.

     Only needed when loading models with some pre-training already (e.g., a
     model part way through training).
     """),

    (str,
     'dataset',
     """Name of dataset that is being trained on.

     This argument is needed to, for example, choose
     an input pipeline function.
     """),

    (str,
     'description',
     """Description of the experiment run, to be added to the Visdom
     environment name.
     """),

    (str,
     'glove_path',
     """Path to Glove embedding file (300 dim)."""),

    (str,
     'log_dir',
     """Path to take summaries and checkpoints from, and write them to."""),

    (str,
     'train_log_file_path',
     """Path to log timestamped messages to with experiment.logging (train).
     """),

    (str,
     'val_log_file_path',
     """Same as above (val)."""),

    (str,
     'h5_file',
     """Path to HDF5 file with dataset."""),

    (str,
     'language_model',
     """Language model to use. Has to be one of {skipthoughts, gru_glove}."""),

    (str,
     'model_name',
     """Name of desired model to use, e.g. vgg, inception_v3."""),

    (str,
     'vis_server',
     """Server to post visualization data to."""),

    (int,
     'architecture_index',
     """Index of architecture in architectures to re-train."""),

    (int,
     'batch_size',
     """Size of training minibatch."""),

    (int,
     'dim_mm',
     """Dimension of the space that the fusion operator projects combined
     feature vectors into before the final prediction layer.
     """),

    (int,
     'dim_q',
     """Dimension of the question features vector."""),

    (int,
     'dim_v',
     """Dimension of the visual features vector."""),

    (int,
     'entropy_decay_epochs',
     """Epochs before entropy decay (by 1/2)."""),

    (int,
     'max_epochs',
     """Maximum number of epochs in training run."""),

    (int,
     'num_classes',
     """Number of classes in the dataset."""),

    (int,
     'num_glimpses',
     """Number of glimpses to use for attention."""),

    (int,
     'num_nodes',
     """Number of nodes in fusion operator."""),

    (int,
     'num_workers',
     """Number of worker threads to use in the PyTorch dataloader."""),

    (int,
     'num_steps_per_summary',
     """Number of training/evaluation steps between writing a summary."""),

    (int,
     'rank',
     """Rank of mutan fusion operator."""),

    (float,
     'dropout_q',
     """Proportion of dropout to use on the question feature vector."""),

    (float,
     'dropout_v',
     """Proportion of dropout to use on the visual feature vector."""),

    (float,
     'dropout_mm',
     """Proportion of dropout to use on the combined feature vector."""),

    (float,
     'dropout_classif',
     """Amount of dropout to use before the linear predictive layer."""),

    (float,
     'entropy_coef',
     """Entropy loss coefficient."""),

    (float,
     'initial_learning_rate',
     """Initial learning rate."""),

    (float,
     'max_grad_norm',
     """Maximum L2-norm of gradients."""),

    (float,
     'momentum',
     """Momentum, for optimizers that use momentum."""),

    (float,
     'selector_lr',
     """Selector learning rate."""),

    (float,
     'sparsity',
     """Weight on L1 norm for whatever."""),

    (float,
     'tau',
     """Temperature scaling in Gumbel softmax."""),

    (float,
     'weight_decay',
     """Amount of weight decay to apply to non-bias weights."""),

    (Choices,
     'dataset_split',
     ['train', 'train+val'],
     """Which split of the data to train on."""),

    (Choices,
     'selection_algorithm',
     ['darts', 'selector'],
     """Which split of the data to train on."""),
]


def parse_args():
    """Parses arguments from the command line."""
    parser = argparse.ArgumentParser(
        description='Train/validate using PyTorch.')

    for opt in CONFIG_OPTIONS:
        if opt[0] == bool:
            parser.add_argument('--' + opt[1],
                                dest=opt[1],
                                action='store_true')
            parser.add_argument('--no-' + opt[1],
                                dest=opt[1],
                                action='store_false')
        elif opt[0] == Choices:
            parser.add_argument('--' + opt[1],
                                choices=opt[2],
                                type=str,
                                help=opt[3])
        elif issubclass(opt[0], ListParam):
            parser.add_argument('--' + opt[1],
                                nargs='+',
                                type=opt[0].dtype,
                                help=opt[2])
        else:
            parser.add_argument('--' + opt[1],
                                type=opt[0],
                                default=None,
                                help=opt[2])

    return parser.parse_args()


def get_unique_id(flags):
    """Returns a unique ID based on the set of hyperparameters and other
    configuration options given by `flags`.
    """
    return '_'.join([flags.description,
                     flags.model_name,
                     os.path.basename(flags.log_dir)])
