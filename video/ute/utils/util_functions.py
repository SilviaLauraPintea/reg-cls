#!/usr/bin/env python

"""Utilities for the project"""

__all__ = [
    "cls_from_filename",
    "plot_data_hist",
    "accuracy_fn",
    "join_data",
    "Averaging",
    "adjust_lr",
    "timing",
    "dir_check",
]

import numpy as np
import time
from collections import defaultdict
import os

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger


def cls_from_filename(model_name):
    splits = str(model_name).split("_")

    assert len(splits) >= 2, "There should be more than 2 splis in " + model_name
    return int(splits[-1])


def plot_data_hist(bins, hist, title, xaxis, yaxis, outfile, width=20.5):
    import matplotlib.pyplot as plt

    fig, (ax) = plt.subplots(
        1,
    )
    fig.tight_layout(pad=5.0)
    ax.bar(bins, hist, alpha=0.7, width=width)
    ax.set_title(title)
    ax.set_ylabel(yaxis)
    ax.set_xlabel(xaxis)
    # ax.set_yscale('log')
    plt.savefig(
        os.path.join(outfile),
        format="pdf",
        pad_inches=-2,
        transparent=False,
        dpi=300,
    )
    # plt.show()
    plt.close()


def accuracy_fn(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = min(max(topk), output.shape[1])
    batch_norm = 100.0 / float(target.size(0))

    val_inds = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = val_inds.indices.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        a_topk = correct_k * batch_norm
        res.append(a_topk)
    return res


class Averaging(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def join_data(data1, data2, f):
    """Simple use of numpy functions vstack and hstack even if data not a tuple

    Args:
        data1 (arr): array or None to be in front of
        data2 (arr): tuple of arrays to join to data1
        f: vstack or hstack from numpy

    Returns:
        Joined data with provided method.
    """
    if isinstance(data2, tuple):
        data2 = f(data2)
    if data1 is not None:
        data2 = f((data1, data2))
    return data2


def adjust_lr(optimizer, lr):
    """Decrease learning rate by 0.1 during training"""
    lr = lr * 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def timing(f):
    """Wrapper for functions to measure time"""

    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.debug(
            "%s took %0.3f ms ~ %0.3f min ~ %0.3f sec"
            % (f, (time2 - time1) * 1000.0, (time2 - time1) / 60.0, (time2 - time1))
        )
        return ret

    return wrap


def dir_check(path):
    """If folder given path does not exist it is created"""
    if not os.path.exists(path):
        os.mkdir(path)
