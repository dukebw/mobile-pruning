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

"""Choo choo."""
import torch

from ..experiment import config
from ..data import dataset


def _get_data_loader(split, drop_last, shuffle, flags):
    """Create and return dataset/loader for split."""
    return torch.utils.data.dataloader.DataLoader(
        dataset=dataset.H5Dataset(flags.h5_file, flags.input_size, split),
        batch_size=flags.batch_size,
        shuffle=shuffle,
        num_workers=flags.num_workers,
        pin_memory=True,
        drop_last=drop_last)


def train(flags):
    """Entry point for model training and validation."""
    torch.backends.cudnn.benchmark = True

    train_loader = _get_data_loader('train', True, True, flags)
    val_loader = _get_data_loader('val', False, False, flags)


if __name__ == '__main__':
    train(config.parse_args())
