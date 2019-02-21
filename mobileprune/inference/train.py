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
import os
import time

try:
    from apex.fp16_utils import network_to_half, FP16_Optimizer
except:
    print('could not import apex')
import numpy as np
import torch

from ..experiment import config
from ..experiment import logging
from ..data import dataset
from ..data import types
from ..models.checkpoint import load_checkpoint
from ..models.checkpoint import save_checkpoint
from ..models.mobilenetv2 import MobileNetV2
from ..models.mobilenetv2 import MobileNetV2IB


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
               'kld',
               'loss',
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
                  kld=AverageMeter(),
                  loss=AverageMeter(),
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
            correct_k = correct_k.detach().cpu().numpy().item()
            res.append(correct_k)
        return res


def _should_summarize(total_num_steps, step, flags):
    """Check if metrics should be logged for this step."""
    summary_step = flags.num_steps_per_summary - 1

    return (
        ((step % flags.num_steps_per_summary) == summary_step) or
        ((step + 1) == total_num_steps))


def _log_metrics(split, meters, num_data, lr, flags):
    logging.log(f'{split} lr: {lr} step: {meters.step}/{num_data} '
                f'({100.0*meters.step/num_data}%)',
                flags.log_file_path)
    logging.log(f'batch time: ({meters.batch_time.val}, {meters.batch_time.avg})',
                flags.log_file_path)
    logging.log(f'data time: ({meters.data_time.val}, {meters.data_time.avg})',
                flags.log_file_path)
    if split == 'train':
        logging.log(f'KL divergence: ({meters.kld.val}, {meters.kld.avg})',
                    flags.log_file_path)
        logging.log(f'loss: ({meters.loss.val}, {meters.loss.avg})',
                    flags.log_file_path)
    logging.log(f'top 1: ({meters.top1.val}, {meters.top1.avg})',
                flags.log_file_path)
    logging.log(f'top 5: ({meters.top5.val}, {meters.top5.avg})',
                flags.log_file_path)


# NOTE(brendan): https://github.com/NVIDIA/apex/blob/3c7a0e4442c41bc5afa183fe4dd0672a729a3ec9/examples/imagenet/main_fp16_optimizer.py#L249
class DataPrefetcher:
    def __init__(self, loader, use_fp16):
        self.use_fp16 = use_fp16

        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor(
            [0.485*255, 0.456*255, 0.406*255]).cuda().view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229*255, 0.224*255, 0.225*255]).cuda().view(1, 3, 1, 1)
        if use_fp16:
            self.mean = self.mean.half()
            self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            if self.use_fp16:
                self.next_input = self.next_input.half()
            else:
                self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def _adjust_learning_rate(optimizer, epoch, step, len_epoch, flags):
    """LR schedule that should yield 76% converged accuracy with batch size
    256.
    """
    lr = flags.lr
    for sched_epoch in flags.lr_schedule:
        if epoch >= sched_epoch:
            lr *= 0.1

    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.0*len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

    prefetcher = DataPrefetcher(boxs_loop.data.train, flags.use_fp16)
    inputs, labels = prefetcher.next()
    meters.step = -1

    end = time.time()
    while inputs is not None:
        meters.data_time.update(time.time() - end)
        meters.step += 1

        _adjust_learning_rate(boxs_loop.optimizer,
                              epoch,
                              meters.step,
                              len(boxs_loop.data.train),
                              flags)

        is_ib = flags.model_name == 'MobileNetV2IB'
        if is_ib:
            preds, kl_total = boxs_loop.model(inputs)
            kl_total = kl_total.mean()
            meters.kld.update(kl_total)
        else:
            preds = boxs_loop.model(inputs)

        loss = boxs_loop.criterion(preds, labels)
        meters.loss.update(loss)
        if is_ib:
            loss += flags.kl_fac*kl_total

        boxs_loop.optimizer.zero_grad()
        if flags.use_fp16:
            boxs_loop.optimizer.backward(loss)
        else:
            loss.backward()
        boxs_loop.optimizer.step()

        torch.cuda.synchronize()
        meters.batch_time.update(time.time() - end)

        top1, top5 = _accuracy(preds, labels)
        meters.top1.update(top1)
        meters.top5.update(top5)

        end = time.time()
        inputs, labels = prefetcher.next()

        num_train = len(boxs_loop.data.train)
        if _should_summarize(num_train, meters.step, flags):
            lr = boxs_loop.optimizer.param_groups[0]['lr']
            _log_metrics('train', meters, num_train, lr, flags)


def _validation(boxs_loop, epoch, flags):
    """Validate the model."""
    boxs_loop.model.eval()

    meters = _get_meters(epoch)

    prefetcher = DataPrefetcher(boxs_loop.data.val, flags.use_fp16)
    inputs, labels = prefetcher.next()
    meters.step = -1

    end = time.time()
    while inputs is not None:
        meters.data_time.update(time.time() - end)
        meters.step += 1

        labels = labels.cuda(non_blocking=True)

        is_ib = flags.model_name == 'MobileNetV2IB'
        if is_ib:
            preds, _ = boxs_loop.model(inputs)
        else:
            preds = boxs_loop.model(inputs)

        top1, top5 = _accuracy(preds, labels)
        meters.top1.update(top1)
        meters.top5.update(top5)

        inputs, labels = prefetcher.next()

        num_val = len(boxs_loop.data.val)
        if _should_summarize(num_val, meters.step, flags):
            lr = boxs_loop.optimizer.param_groups[0]['lr']
            _log_metrics('val', meters, num_val, lr, flags)

    return meters.top1.avg


# NOTE(brendan): https://github.com/NVIDIA/apex/blob/3c7a0e4442c41bc5afa183fe4dd0672a729a3ec9/examples/imagenet/main_fp16_optimizer.py#L78
def _fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


def _get_data_loader(split, drop_last, shuffle, batch_size, input_size, flags):
    """Create and return dataset/loader for split."""
    return torch.utils.data.dataloader.DataLoader(
        dataset=dataset.H5Dataset(flags.h5_file, input_size, split),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=flags.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        collate_fn=_fast_collate)


def _get_loaders(batch_size, input_size, flags):
    train_loader = _get_data_loader('train',
                                    True,
                                    True,
                                    batch_size,
                                    input_size,
                                    flags)
    val_loader = _get_data_loader('val',
                                  False,
                                  False,
                                  batch_size,
                                  input_size,
                                  flags)

    return Loaders(train=train_loader, val=val_loader)


def train(flags):
    """Entry point for model training and validation."""
    torch.backends.cudnn.benchmark = True

    model_table = {
        'MobileNetV2': MobileNetV2,
        'MobileNetV2IB': MobileNetV2IB,
    }
    model = model_table[flags.model_name](input_size=flags.input_size,
                                          scale=flags.scale,
                                          grp_fact=flags.grp_fact)
    model = torch.nn.DataParallel(model).cuda()
    if flags.use_fp16:
        model = network_to_half(model)

    criterion = torch.nn.CrossEntropyLoss().cuda()

    flags.lr = flags.lr * flags.batch_size / 256.0
    optimizer = torch.optim.SGD(model.parameters(),
                                flags.lr,
                                momentum=flags.momentum,
                                weight_decay=flags.weight_decay,
                                nesterov=True)
    if flags.use_fp16:
        optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)

    input_size = flags.input_size
    loaders = _get_loaders(flags.batch_size, input_size, flags)
    boxs_loop = BoxsLoop(criterion=criterion,
                         data=loaders,
                         model=model,
                         optimizer=optimizer)

    latest_ckpt_path = f'{flags.log_dir}/latest_checkpoint'
    if (flags.checkpoint_path is None) and os.path.exists(latest_ckpt_path):
        with open(latest_ckpt_path, 'r') as f:
            flags.checkpoint_path = f'{flags.log_dir}/{f.read().strip()}'
        logging.log(f'Loading latest checkpoint {flags.checkpoint_path}',
                    flags.log_file_path)
    epoch = load_checkpoint(boxs_loop,
                            flags.checkpoint_path,
                            flags.log_file_path)
    epoch = 0 if epoch is None else (epoch + 1)

    best_prec1 = None
    best_ckpt_path = f'{flags.log_dir}/best_checkpoint'
    if os.path.exists(best_ckpt_path):
        with open(best_ckpt_path, 'r') as f:
            best_prec1 = f.read().split(',')[-1]
        # TODO(brendan): best_prec1 should be saved as a float, not list
        best_prec1 = float(best_prec1)

    for epoch in range(epoch, flags.max_epochs):
        logging.log(f'=> Epochs {epoch}', flags.log_file_path)

        _train_single_epoch(boxs_loop, epoch, flags)

        with torch.no_grad():
            prec1 = _validation(boxs_loop, epoch, flags)

        if (best_prec1 is None) or (prec1 > best_prec1):
            logging.log(f'new best! epoch: {epoch} prec1: {prec1}',
                        flags.log_file_path)
            save_checkpoint(boxs_loop, epoch, flags, 'best.pth.tar')
            with open(best_ckpt_path, 'w') as f:
                f.write(f'{epoch},{prec1}')
            best_prec1 = prec1

        save_checkpoint(boxs_loop, epoch, flags)


if __name__ == '__main__':
    train(config.parse_args())
