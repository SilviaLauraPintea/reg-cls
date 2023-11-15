#!/usr/bin/env python

"""Logger parameters for the entire process.
"""

import logging
import datetime
import sys
import re
import os
from os.path import join

from ute.utils.arg_pars import opt


logger = logging.getLogger("basic")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

filename = sys.argv[0]
search = re.search(r"\/*(\w*).py", filename)
filename = search.group(1)


def update_opt_str():
    global opt, logger
    logs_args_map = {
        "model_name": "",
        "epochs": "ep",
        "embed_dim": "dim",
        "lr": "lr",
    }
    log_str = ""
    logs_args = ["subaction"] + sorted(logs_args_map)
    logs_args_map["subaction"] = ""
    for arg in logs_args:
        attr = getattr(opt, arg)
        arg = logs_args_map[arg]
        if isinstance(attr, bool):
            if attr:
                attr = arg
            else:
                attr = "!" + arg
        else:
            attr = "%s%s" % (arg, str(attr))
        log_str += "%s_" % attr

    opt.log_str = log_str

    vars_iter = list(vars(opt))
    for arg in sorted(vars_iter):
        logger.debug("%s: %s" % (arg, getattr(opt, arg)))


def path_logger():
    global logger
    if not os.path.exists(join(opt.dataset_root, "logs")):
        os.mkdir(join(opt.dataset_root, "logs"))
    path_logging = join(
        opt.dataset_root,
        "logs",
        "%s%s(%s)" % (opt.log_str, filename, str(datetime.datetime.now())),
    )
    fh = logging.FileHandler(path_logging, mode="w")
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelno)s - %(filename)s - " "%(funcName)s - %(message)s"
    )
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger
