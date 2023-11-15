#!/usr/bin/env python

"""Implementation and improvement of the paper:
Unsupervised learning and segmentation of complex activities from video.
"""

import sys
import os
import numpy as np

sys.path.append(os.path.abspath(".").split("data_utils")[0])

from ute.utils.arg_pars import opt

from data_utils.update_argpars import update
from ute.ute_pipeline import temp_embed


if __name__ == "__main__":
    opt.dataset_root = "./data_utils"
    opt.ext = "txt"
    opt.feature_dim = 64
    opt.resume = False
    opt.test_model = False
    update()

    model_name = opt.model_name
    if opt.subaction == "all":
        actions = [
            "coffee",
            "cereals",
            "tea",
            "milk",
            "juice",
            "sandwich",
            "scrambledegg",
            "friedegg",
            "salat",
            "pancake",
        ]
        all_losses = []
        for act in actions:
            opt.subaction = act
            opt.model_name = (
                model_name
                + "_"
                + opt.split
                + "_"
                + opt.subaction
                + "_"
                + str(opt.cls_num)
            )
            test_loss = temp_embed()
            all_losses.append(test_loss["test_loss"])

        arr_rmse = np.array(all_losses)
        print(
            "All test losses (RMSE): \n",
            arr_rmse,
            " | ",
            arr_rmse.mean(),
            " +/- ",
            arr_rmse.std(),
        )

    else:
        temp_embed()
