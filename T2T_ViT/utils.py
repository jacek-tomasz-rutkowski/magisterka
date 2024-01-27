# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

'''
- load_for_transfer_learning: load pretrained paramters to model in transfer learning
- get_mean_and_std: calculate the mean and std value of dataset.
- msr_init: net parameter initialization.
- progress_bar: progress bar mimic xlua.progress.
'''
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

_logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: Path, model: nn.Module, ignore_keys: list[str] = [], device="cpu") -> None:
    """
    Load a checkpoint onto a model.

    Ignored keys are kept unchanged in the model (typically ["head.weight", "head.bias"] for transfer learning).
    """

    state_dict = torch.load(checkpoint_path, map_location=device)

    # Fix nested state_dict-s.
    for k in ["state_dict_ema", "state_dict", "model_ema", "model", "net"]:
        if k in state_dict:
            state_dict = state_dict[k]
    some_key = next(iter(state_dict.keys()))
    if some_key.startswith("module."):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    # Resize position embedding if necessary.
    if state_dict["pos_embed"].shape != model.pos_embed.shape:
        state_dict["pos_embed"] = resize_pos_embed(state_dict["pos_embed"], model.pos_embed)

    # Handle ignored keys.
    if ignore_keys:
        model_state_dict = model.state_dict()
        for k in ignore_keys:
            if k in state_dict:
                state_dict[k] = model_state_dict[k]

    model.load_state_dict(state_dict, strict=True)
    model.to(device)


def resize_pos_embed(posemb: torch.Tensor, posemb_new: torch.Tensor) -> torch.Tensor:
    # example: 224:(14x14+1)-> 384: (24x24+1)
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]
    if True:
        # posemb_tok is for cls token, posemb_grid for the following tokens
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))     # 14
    gs_new = int(math.sqrt(ntok_new))             # 24
    _logger.info('Position embedding grid-size from %s to %s', gs_old, gs_new)
    # [1, 196, dim]->[1, 14, 14, dim]->[1, dim, 14, 14]
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    # [1, dim, 14, 14] -> [1, dim, 24, 24]
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bicubic')
    # [1, dim, 24, 24] -> [1, 24*24, dim]
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)   # [1, 24*24+1, dim]
    return posemb


def load_state_dict(checkpoint_path, model, use_ema=False, num_classes=1000, del_posemb=False):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            state_dict = {k.removeprefix('module.'): v for k, v in checkpoint[state_dict_key].items()}
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        if num_classes != 1000:
            # completely discard fully connected for all other differences between pretrained and created model
            # del state_dict['head' + '.weight']
            # del state_dict['head' + '.bias']
            state_dict['head' + '.weight'] = model.state_dict()['head' + '.weight']
            state_dict['head' + '.bias'] = model.state_dict()['head' + '.bias']

        if del_posemb:
            del state_dict['pos_embed']

        old_posemb = state_dict['pos_embed']

        if model.pos_embed.shape != old_posemb.shape:  # need resize the position embedding by interpolate
            new_posemb = resize_pos_embed(old_posemb, model.pos_embed)
            state_dict['pos_embed'] = new_posemb

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_for_transfer_learning(model, checkpoint_path, use_ema=False, strict=True, num_classes=1000):
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)


def get_mean_and_std(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    '''Compute the mean and std value of dataset, as a pair of tensors of shape (3,).'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))  # type: ignore
    std.div_(len(dataset))  # type: ignore
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


# _, term_width_str = os.popen('stty size', 'r').read().split()
# term_width = int(term_width_str)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    # for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
    #     sys.stdout.write(' ')

    # # Go back to the center of the bar.
    # for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
    #     sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds: float) -> str:
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f
