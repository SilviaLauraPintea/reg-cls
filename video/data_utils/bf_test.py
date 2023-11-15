#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

import sys
import os

sys.path.append(os.path.abspath(".").split("data_utils")[0])

from ute.utils.arg_pars import opt
from data_utils.BF_utils.update_argpars import update
from ute.ute_pipeline import temp_embed, all_actions

if __name__ == "__main__":
    opt.dataset_root = "./data_utils"
    opt.ext = "txt"
    opt.feature_dim = 64

    opt.test_models = []
    opt.test_models.append(
        opt.model_name + "_" + opt.split + "_" + opt.subaction + "_" + str(opt.cls_num)
    )
    opt.model_name += "_" + opt.split + "_" + opt.subaction + "_" + str(opt.cls_num)

    # load an already trained model (stored in the models directory in dataset_root)
    opt.test_model = True
    opt.loaded_model_name = "%s.pth.tar"

    # update log name and absolute paths
    update()
    temp_embed()
