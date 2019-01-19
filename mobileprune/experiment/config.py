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
     'use_fp16',
     """Use FP16, dynamic loss scaling."""),

    (str,
     'checkpoint_path',
     """Paths to take checkpoint files (e.g., inception_v3.ckpt) from.
     Checkpoint paths should be separated by commas.

     Only needed when loading models with some pre-training already (e.g., a
     model part way through training).
     """),

    (str,
     'description',
     """Description of the experiment run, to be added to the Visdom
     environment name.
     """),

    (str,
     'h5_file',
     """Path to HDF5 file with dataset."""),

    (str,
     'log_dir',
     """Path to take summaries and checkpoints from, and write them to."""),

    (str,
     'log_file_path',
     """Path to log timestamped messages to with experiment.logging.
     """),

    (str,
     'model_name',
     """Name of desired model to use, e.g. vgg, inception_v3."""),

    (str,
     'vis_server',
     """Server to post visualization data to."""),

    (int,
     'batch_size',
     """Size of training minibatch."""),

    (int,
     'checkpoint_save_interval',
     """Save the model whenever `checkpoint_save_interval` epochs have
     passed.
     """),

    (int,
     'input_size',
     """Image input dimension to model."""),

    (int,
     'max_epochs',
     """Maximum number of epochs in training run."""),

    (int,
     'num_steps_per_summary',
     """Number of training/evaluation steps between writing a summary."""),

    (int,
     'num_workers',
     """Number of worker threads to use in the PyTorch dataloader."""),

    (float,
     'lr',
     """Initial learning rate."""),

    (float,
     'max_grad_norm',
     """Maximum L2-norm of gradients."""),

    (float,
     'momentum',
     """Momentum, for optimizers that use momentum."""),

    (float,
     'prog_resize_after',
     """Progressive upsize after this amount of the current LR epochs."""),

    (float,
     'scale',
     """MobileNetV2 scale."""),

    (float,
     'weight_decay',
     """Amount of weight decay to apply to non-bias weights."""),

    (ListParamInt,
     'lr_schedule',
     """List of epochs to drop LR at."""),
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
