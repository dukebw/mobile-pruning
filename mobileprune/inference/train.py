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
import math
import time

import torch
from torch.optim.lr_scheduler import MultiStepLR

from ..experiment import config
from ..experiment import logging
from ..data import dataset
from ..data import types
from ..models import checkpoint
from ..models.mobilenetv2 import MobileNetV2


class BoxsLoop(types.Struct):  # pylint:disable=too-few-public-methods
    """Statistician George Edward Pelham Box's method of solving problems:
    collecting data, making predictions, and analyzing the results to improve
    the model.

    Fields:
        criterion: Criticism object, i.e. loss or objective function.
        data: DataLoader object that can iterate over videos, labels and ids of
            training data.
        model: Probabilistic model to optimize.
        optimizer: Optimizer object, e.g. RMSProp, SGD, Adam, etc.

    1. Build a probabilistic model of the phenomena.

    2. Reason about the phenomena given model and data.

    3. Criticize the model, revise, and repeat.

    http://edwardlib.org/api/
    (Blei, 2014)
    """

    _fields = ['criterion', 'data', 'model', 'optimizer']


class Meters(types.Struct):  # pylint:disable=too-few-public-methods
    """Metrics tracking."""

    _fields = ['batch_time',
               'data_time',
               'epoch',
               'top1',
               'top5',
               'grad_norm',
               'step']


class Loaders(types.Struct):  # pylint:disable=too-few-public-methods
    """Train and val loaders."""

    _fields = ['train', 'val']


def _cma(val, avg, step_num):
    """Returns cumulative moving average of val with avg."""
    return (val + step_num*avg)/(step_num + 1)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, num_steps_per_summary=math.inf):
        self.val = 0
        self.avg = 0
        self.count = 0
        self.per_summary_avg = 0
        self.num_steps_per_summary = num_steps_per_summary

    def update(self, val):
        self.val = val
        self.avg = _cma(self.val, self.avg, self.count)
        self.per_summary_avg = _cma(self.val,
                                    self.per_summary_avg,
                                    self.count % self.num_steps_per_summary)
        self.count += 1


def _get_meters(epoch):
    """Meters, like mIoU and stuff."""
    return Meters(batch_time=AverageMeter(),
                  data_time=AverageMeter(),
                  epoch=epoch,
                  top1=AverageMeter(),
                  top5=AverageMeter(),
                  grad_norm=AverageMeter(),
                  step=None)


# https://github.com/pytorch/examples/blob/e0929a4253f9ae6ccdde24e787788a9955fdfe1c/imagenet/main.py#L381
def _accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified
    values of k
    """
    topk = (1, 5)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct_k.mul_(100.0 / batch_size)
            correct_k = correct_k.detach().cpu().numpy()
            res.append(correct_k)
        return res


def _should_summarize(total_num_steps, step, flags):
    """Check if metrics should be logged for this step."""
    summary_step = flags.num_steps_per_summary - 1

    return (
        ((step % flags.num_steps_per_summary) == summary_step) or
        ((step + 1) == total_num_steps))


def _log_metrics(split, meters, num_data, flags):
    logging.log(f'{split} top 1: ({meters.top1.val}, {meters.top1.avg})',
                flags.log_file_path)
    logging.log(f'{split} top 5: ({meters.top5.val}, {meters.top5.avg})',
                flags.log_file_path)
    logging.log(f'{split} {meters.step}/{num_data} '
                f'({100.0*meters.step/num_data}%)',
                flags.log_file_path)


def _train_single_epoch(boxs_loop, epoch, flags):
    """Train one epoch.

    Args:
        boxs_loop.*: See BoxsLoop.
        epoch: Number of the current epoch, starting at epoch 0.
        flags.*: See experiment.config.CONFIG_OPTIONS.

    Load one epoch's worth of data and do inference and criticism to train on
    that data.

    Compute evaluation metrics on the training data and log.
    """
    boxs_loop.model.train()

    meters = _get_meters(epoch)

    end = time.time()
    for meters.step, (inputs, labels) in enumerate(boxs_loop.data.train):
        meters.data_time.update(time.time() - end)
        labels = labels.cuda(non_blocking=True)

        preds = boxs_loop.model(inputs)
        loss = boxs_loop.criterion(preds, labels)

        boxs_loop.optimizer.zero_grad()
        loss.backward()
        boxs_loop.optimizer.step()

        top1, top5 = _accuracy(preds, labels)
        meters.top1.update(top1)
        meters.top5.update(top5)

        num_train = len(boxs_loop.data.train)
        if _should_summarize(num_train, meters.step, flags):
            _log_metrics('train', meters, num_train, flags)


def _validation(boxs_loop, epoch, flags):
    """Validate the model."""
    boxs_loop.model.eval()

    meters = _get_meters(epoch)
    end = time.time()
    for meters.step, (inputs, labels) in enumerate(boxs_loop.data.val):
        meters.data_time.update(time.time() - end)
        labels = labels.cuda(non_blocking=True)

        preds = boxs_loop.model(inputs)

        top1, top5 = _accuracy(preds, labels)
        meters.top1.update(top1)
        meters.top5.update(top5)

        num_val = len(boxs_loop.data.val)
        if _should_summarize(num_val, meters.step, flags):
            _log_metrics('val', meters, num_val, flags)


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

    model = MobileNetV2(input_size=flags.input_size, scale=flags.scale)
    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                flags.initial_learning_rate,
                                momentum=flags.momentum,
                                weight_decay=flags.weight_decay,
                                nesterov=True)

    boxs_loop = BoxsLoop(criterion=criterion,
                         data=Loaders(train=train_loader, val=val_loader),
                         model=model,
                         optimizer=optimizer)

    scheduler = MultiStepLR(optimizer, milestones=[200, 300], gamma=0.1)

    epoch = checkpoint.load_checkpoint(boxs_loop,
                                       flags.checkpoint_path,
                                       flags.log_file_path)
    epoch = 0 if epoch is None else (epoch + 1)

    for epoch in range(epoch, flags.max_epochs):
        logging.log(f'=> Epochs {epoch}, learning rate = {scheduler.get_lr()}.',
                    flags.log_file_path)

        scheduler.step()

        _train_single_epoch(boxs_loop, epoch, flags)

        with torch.no_grad():
            _validation(boxs_loop, epoch, flags)

        checkpoint.save_checkpoint(boxs_loop, epoch, flags)


if __name__ == '__main__':
    train(config.parse_args())
