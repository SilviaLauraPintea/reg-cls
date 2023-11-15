#!/usr/bin/env python

"""
Update parameters which directly depends on the dataset.
"""

import os
import os.path as ops

from ute.utils.arg_pars import opt
from ute.utils.util_functions import dir_check
from ute.utils.logging_setup import path_logger, update_opt_str
import torch


def update():
    opt.data = ops.join(ops.join(opt.dataset_root, "data"), "features")
    opt.gt = ops.join(ops.join(opt.dataset_root, "data"), "groundTruth")
    opt.mapping_dir = ops.join(ops.join(opt.dataset_root, "data"), "mapping")
    opt.output_dir = ops.join(opt.dataset_root, "output")

    dir_check(opt.output_dir)

    if torch.cuda.is_available():
        opt.device = "cuda"
    opt.embed_dim = 20

    update_opt_str()
    logger = path_logger()

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug("%s: %s" % (arg, getattr(opt, arg)))
