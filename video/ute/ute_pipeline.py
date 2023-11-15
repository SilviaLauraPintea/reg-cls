"""
Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

import torch
import numpy as np
import random


from ute.corpus import Corpus
from ute.utils.logging_setup import logger, update_opt_str
from ute.utils.arg_pars import opt


from ute.utils.util_functions import timing


def set_seed():
    """Make everything deterministic using a predefined seed."""

    seed = int(opt.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def all_actions(actions):
    """Runs the code by training a regression MLP per subaction, for all actions."""

    # Sets the randomization seeds
    set_seed()

    return_stat_all = {"test_loss": 0}
    lr_init = opt.lr

    print("\n\n>>>>>>>>>> [ALL OPTIONS]:", opt)

    # Loop over actions
    for action in actions:
        opt.subaction = action

        # If the model is resumed
        if not opt.resume:
            opt.lr = lr_init
        update_opt_str()

        # Get the model statistics
        return_stat_single = temp_embed()
        return_stat_all["test_loss"] += return_stat_single["test_loss"] / len(actions)

    # Print all statistics
    logger.debug(return_stat_all)


@timing
def temp_embed():
    """Loads all the videos for one subaction and trains a regression MLP."""

    # Sets the randomization seeds
    set_seed()

    # Loads all videos, features, and gt
    corpus = Corpus(subaction=opt.subaction)
    logger.debug("Corpus with poses created")

    # Trains or loads a new model and uses it to extract temporal embeddings for each video
    corpus.regression_traineval()
    return corpus._return_stat
