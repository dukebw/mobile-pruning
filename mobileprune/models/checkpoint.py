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

"""Checkpoint-related functionality."""
import os

import path
import torch

from ..experiment import logging


def load_checkpoint(boxs_loop, checkpoint_path, log_file_path):
    """Load a checkpoint from checkpoint_path, if it exists.

    Update boxs_loop.model and boxs_loop.optimizer with the state in any
    checkpoint found.
    """
    if checkpoint_path is None:
        return None

    checkpoint = checkpoint_path
    if not isinstance(checkpoint_path, str):
        # NOTE(brendan): assume checkpoint_path is file-like.
        checkpoint_path = 'bytes I/O'
    elif not os.path.isfile(checkpoint_path):
        logging.log(
            f'=> no checkpoint found at "{checkpoint_path}"', log_file_path)
        return None

    logging.log(f'=> loading checkpoint "{checkpoint_path}"', log_file_path)
    # NOTE(brendan): At this point checkpoint is either a filepath or
    # file-like (I/O buffer).
    checkpoint = torch.load(checkpoint)

    try:
        boxs_loop.model.load_state_dict(checkpoint['state_dict'])
    except KeyError as e:
        logging.log(e, log_file_path)

    try:
        if boxs_loop.optimizer is None:
            raise KeyError
        boxs_loop.optimizer.load_state_dict(checkpoint['optimizer'])
    except KeyError:
        logging.log('Evaluation run: no optimizer.', log_file_path)

    try:
        epoch = checkpoint['epoch']
    except KeyError:
        epoch = -1
        logging.log('Epoch not found in checkpoint: setting epoch = -1.',
                    log_file_path)

    logging.log(f'=> loaded checkpoint "{checkpoint_path}" (epoch {epoch})',
                log_file_path)

    return epoch


def save_checkpoint(boxs_loop, epoch, flags, ckpt_name=None):
    """Save the checkpoint for this epoch.

    Also, delete a checkpoint from five saves ago, if said checkpoint exists.

    Args:
        boxs_loop: Box's loop, containing the model to be saved, and the
            optimizer state.
        epoch: This is epoch number ?
        flags.*: See experiment.config.CONFIG_OPTIONS.
    """
    should_save = ((ckpt_name is not None) or
                   (flags.checkpoint_save_interval is None) or
                   ((epoch % flags.checkpoint_save_interval) == 0))
    if not should_save:
        return

    five_back = 5
    if flags.checkpoint_save_interval is not None:
        five_back *= flags.checkpoint_save_interval

    log_dir = path.Path(flags.log_dir)
    fname = f'model_{os.getpid()}_checkpoint{epoch - five_back}.pth.tar'
    checkpoint_5_back = log_dir/fname
    if (epoch >= five_back) and os.path.exists(checkpoint_5_back):
        os.remove(checkpoint_5_back)

    if ckpt_name is None:
        ckpt_name = f'model_{os.getpid()}_checkpoint{epoch}.pth.tar'
    save_state = {'epoch': epoch,
                  'arch': flags.model_name,
                  'state_dict': boxs_loop.model.state_dict(),
                  'optimizer': boxs_loop.optimizer.state_dict()}
    torch.save(save_state, log_dir/ckpt_name)

    with open(os.path.join(flags.log_dir, 'latest_checkpoint'), 'w') as f:
        f.write(ckpt_name)
