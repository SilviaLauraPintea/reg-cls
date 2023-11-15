#!/usr/bin/env python

"""
Module with Corpus class. There are methods for each step of the alg for the
whole video collection of one complex activity. See pipeline.
"""

import pickle5 as pickle
import numpy as np
import os
import os.path as ops
import torch
import re
import matplotlib.pyplot as plt
import time

from ute.video import Video
from ute.models import mlp
from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import join_data, cls_from_filename

from ute.models.dataset_loader import load_reltime
from ute.models.training_embed import load_model, training, test


class Corpus(object):
    def __init__(self, subaction="coffee"):
        """
        Args:
            subaction: current name of complex activity
        """
        self._return_stat = {}
        self._subaction = subaction

        # Multiprocessing for sampling activities for each video
        self._videos = []
        self._init_videos()

    def is_action_file(self, split, filename):
        """
        Checks if a filename should be added to the data list of files.
        Uses the standard splits from https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/.
            - s1 test IDs: [03-15]
            - s2 test IDs: [16-28]
            - s3 test IDs: [29-41]
            - s4 test IDs: [42-54]
        """

        isin = False

        # If the filename is not in part of the subaction
        if opt.subaction not in filename:  # If not the right action
            isin = False
        else:
            # Add all split IDs to create the standard splits
            all_splits = []
            if split.startswith("s1"):
                test_id_start = 3
                test_id_end = 16
                for i in range(test_id_start, test_id_end):
                    all_splits.append("P%02d" % i)
            elif split.startswith("s2"):
                test_id_start = 16
                test_id_end = 29
                for i in range(test_id_start, test_id_end):
                    all_splits.append("P%02d" % i)
            elif split.startswith("s3"):
                test_id_start = 29
                test_id_end = 42
                for i in range(test_id_start, test_id_end):
                    all_splits.append("P%02d" % i)
            elif split.startswith("s4"):
                test_id_start = 42
                test_id_end = 55
                for i in range(test_id_start, test_id_end):
                    all_splits.append("P%02d" % i)

            # If we want the test split
            if opt.test_model:
                isin = False
                for ids in all_splits:
                    # Other actions that the one
                    if filename.startswith(ids):
                        isin = True
                        break
            # IF we want the trainval split
            else:
                isin = True
                for ids in all_splits:
                    # Other actions than the one
                    if filename.startswith(ids):
                        isin = False
                        break
            del all_splits
        return isin

    def _init_videos(self):
        self._videos = []

        """Initializes the video class with a list of video file names."""
        logger.debug(".")

        stats_path = os.path.join(
            opt.output_dir,
            opt.split
            + "_"
            + opt.subaction
            + ("_test" if opt.test_model else "_train")
            + "_dataset"
            + ("_rel" if opt.reltime else "_abs")
            + ".pkl",
        )

        if os.path.exists(stats_path):
            print("Loading dataset from ....", stats_path)
            with open(stats_path, "rb") as handle:
                self._videos = pickle.load(handle)
        else:
            for root, dirs, files in os.walk(opt.data):
                if not files:
                    continue

                for filename in files:
                    # Pick only certain files (ex: just concerning coffee in s2)
                    if self.is_action_file(opt.split, filename):

                        # use extracted videos from pretrained on gt embedding
                        path = os.path.join(root, filename)
                        try:
                            video = Video(path=path)
                            self._videos.append(video)
                        except AssertionError:
                            logger.debug("Assertion Error: %s" % path)
                            continue

            # Dump the videos
            with open(stats_path, "wb") as handle:
                pickle.dump(
                    self._videos,
                    handle,
                )

    def reset(self):
        self.__init__()

    def get_videos(self):
        for video in self._videos:
            yield video

    def video_byidx(self, idx):
        return np.asarray(self._videos)[idx]

    def __len__(self):
        return len(self._videos)

    def regression_traineval(self):
        """Trains / evaluates the MLP on the dataset."""

        # Train the model
        opt.test_model = False
        self.reset()
        self.train_mlp()

        # Test the model
        opt.test_model = True
        self.reset()
        opt.test_models = [opt.model_name]
        self.test_mlp()
        opt.test_model = False

    def train_mlp(self):
        """Trains the MLP on the feature data."""
        opt.test_model = False

        train_loader = load_reltime(
            videos=self._videos,
            is_train=True,
            shuffle=True,
            use_set=("train" if opt.testset_name.endswith("val") else "all"),
        )

        # Define the MLP model
        model, optimizer = mlp.create_model()

        # Train the MLP on the training data
        training(
            train_loader,
            opt.epochs,
            save=opt.save_model,
            model=model,
            loss=model.loss(),
            optimizer=optimizer,
            name=opt.model_name,
        )

    def test_mlp(self):
        """Loss per model and averaged across models."""
        opt.test_model = True
        return_stat_all = {"test_loss": 0}

        # Loop over all models to be tested
        for model_path in opt.test_models:
            opt.cls_num = cls_from_filename(model_path)

            model, optimizer = mlp.create_model()
            model.load_state_dict(load_model(model_path))

            test_loader = load_reltime(
                videos=self._videos,
                is_train=(not opt.test_model),
                shuffle=(not opt.test_model),
                use_set=opt.testset_name,
            )

            # Get the model, loss and optimizer
            self._embedding = model
            test_loss = test(
                test_loader,
                save=False,
                model=model,
                loss=model.loss(),
                name=opt.model_name,
            )
            logger.debug(
                "Test loss ["
                + opt.loss
                + "]: "
                + str(test_loss)
                + " for model "
                + model_path
            )
            return_stat_all["test_loss"] += test_loss / len(opt.test_models)

        # Return the average across models
        self._return_stat = {"test_loss": return_stat_all["test_loss"]}

        logger.debug(self._return_stat)
