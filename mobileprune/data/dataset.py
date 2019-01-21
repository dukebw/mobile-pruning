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

"""ImageNet."""
import io

import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


# NOTE(brendan): from https://github.com/Randl/MobileNetV2-pytorch/blob/3518846c69971c10cae89b6b29497a502200da65/data.py#L13
def inception_preprocess(input_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
    ])


def scale_crop(input_size):
    t_list = [
        transforms.CenterCrop(input_size),
    ]

    scale_size = int(input_size / 0.875)
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


class H5Dataset(torch.utils.data.Dataset):
    """Load HDF5 records."""

    def __init__(self, h5_file, input_size, split):
        self.h5_data = None
        self.h5_file = h5_file
        self.split = split

        if split == 'train':
            self.transform = inception_preprocess(input_size)
        else:
            assert split == 'val'
            self.transform = scale_crop(input_size)

        with h5py.File(h5_file, mode='r') as h5_data:
            self.class_upper_ind = np.empty(len(h5_data[split]),
                                            dtype=np.int32)

            self.classes = list(h5_data[split].keys())
            self.classes.sort()
            class_to_idx = {self.classes[i]: i
                            for i in range(len(self.classes))}

            num_per_class = np.empty(len(h5_data[split]), dtype=np.int32)
            for cls_name, class_imgs in h5_data[split].items():
                num_per_class[class_to_idx[cls_name]] = len(class_imgs)

        self.num_examples = 0
        for i, num in enumerate(num_per_class):
            self.num_examples += num
            self.class_upper_ind[i] = self.num_examples

    def __getitem__(self, index):
        """ImageNet dataset: returns (img, target class)."""
        # NOTE(brendan): Opening of the HDF5 file is delayed until the first
        # slice into the H5Dataset, because in libhdf5 it is unsafe to access a
        # single file descriptor from multiple processes.
        #
        # This way, a separate fd is opened for each worker process.
        if self.h5_data is None:
            self.h5_data = h5py.File(self.h5_file, mode='r')

        class_idx = np.searchsorted(self.class_upper_ind, index, side='right')
        cls_name = self.classes[class_idx]
        if class_idx > 0:
            index = index % self.class_upper_ind[class_idx - 1]

        img_io = io.BytesIO(self.h5_data[self.split][cls_name][index])
        img = Image.open(img_io).convert('RGB')

        return self.transform(img), class_idx

    def __len__(self):
        return self.num_examples
