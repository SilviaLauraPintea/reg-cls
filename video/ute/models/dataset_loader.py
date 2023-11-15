#!/usr/bin/env python

"""Creating dataset out of video features for different models.
"""

from torch.utils.data import Dataset
import torch
import numpy as np
import os
import pickle5 as pickle
import math
import copy

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import plot_data_hist, join_data

from ute.utils.histeq import cls_equalized_ranges


class FeatureDataset(Dataset):
    """A feature dataset containing only the features."""

    def __init__(self, videos, is_train):
        logger.debug("Creating feature dataset")

        self._features = None
        self._gt = None

        self._videos = videos
        self._is_train = is_train

    def __len__(self):
        return len(self._gt)

    def __getitem__(self, idx):
        idx = idx
        gt_item = self._gt[idx]
        features = self._features[idx]
        return np.asarray(features), gt_item


class RelTimeDataset(FeatureDataset):
    """
    A time dataset that has video frame features and frame progressions as labels.
    Args:
        videos: A list of video objects
        features: Associated features with each video object
        is_train: If it is a training set or an evaluation set
        use_set: Which set to use train/val/all (=trainval or test)
    """

    def __init__(self, videos, is_train, use_set):
        logger.debug("Relative time labels")
        super().__init__(videos, is_train)

        self._max_time = 0
        self._min_time = 0

        # Loads or re-assembles the dataset from files.
        self.get_dataset()

        # Pick the set we want: train/val/all (=trainval or test)
        self.pick_set(use_set)

        # Gets the target statistics
        self._is_train = is_train
        if use_set.startswith("all"):
            if not self._is_train:
                use_set = "test"
            else:
                use_set = "trainval"

        # Define class statistics and ranges
        if opt.cls_num > 0:
            self.cls_data_stats(
                split=use_set, minval=self._min_time, maxval=self._max_time, plot=True
            )

    def get_dataset(self):
        """Loads or re-assembles the dataset from files."""

        stats_path = os.path.join(
            opt.output_dir,
            opt.split
            + "_"
            + opt.subaction
            + ("_test" if opt.test_model else "_train")
            + "_final_dataset"
            + ("_rel" if opt.reltime else "_abs")
            + ".pkl",
        )

        if os.path.exists(stats_path):
            print("Loading dataset from ....", stats_path)
            with open(stats_path, "rb") as handle:
                (
                    self._features,
                    self._gt,
                    self._max_time,
                    self._min_time,
                ) = pickle.load(handle)
        else:
            self._max_time = 0
            self._min_time = 0

            for video in self._videos:
                self._max_time = max(self._max_time, video.n_frames)

                # Concatenate the Gt
                time_label = np.asarray(video._temp).reshape((-1, 1))
                self._gt = join_data(self._gt, time_label, np.vstack)

                # Concatenate the features to the rest of the features
                self._features = join_data(self._features, video.features(), np.vstack)

            assert self._gt.shape[0] == self._features.shape[0]

            # Save the dataset locally with features and GT
            with open(stats_path, "wb") as handle:
                dataset = (
                    self._features,
                    self._gt,
                    self._max_time,
                    self._min_time,
                )
                pickle.dump(
                    dataset,
                    handle,
                )

        # We do not need the videos anymore
        del self._videos

    def __getitem__(self, idx):
        """Returns one item for in the batch."""
        gt_item = self._gt[idx]
        features = self._features[idx]
        cls = self.cls_one_label(gt_item)
        return np.asarray(features), gt_item, cls

    def cls_data_stats(self, split, minval, maxval, plot):
        """Get the class data statistics.
        Args:
            - split: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        # Specifics of this dataset
        if opt.reltime:
            minval = 0.0
            maxval = 1.0

        self.cls_ranges = np.linspace(minval, maxval, opt.cls_num + 1)
        self.cls_equalize = opt.cls_equalize

        # Load the training stats if they are there
        stats = self.cls_stats(split, minval, maxval, plot)

        # Load the training stats if they are there
        if self.cls_equalize:
            self.cls_equ_stats(stats, split, minval, maxval, plot)

    def cls_stats(self, split, minval, maxval, plot):
        """First just compute the histogram of samples per class, class ranges, and the number of samples per bin.
        Args:
            - split: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        # Load the training stats if they are there
        stats_path = os.path.join(
            opt.output_dir,
            "target_stats_"
            + split
            + opt.subaction
            + "_"
            + str(opt.cls_num)
            + "_"
            + opt.split
            + ("_rel" if opt.reltime else "_abs")
            + ".pkl",
        )

        if os.path.exists(stats_path):
            with open(stats_path, "rb") as handle:
                stats = pickle.load(handle)
        else:
            bincounts = np.zeros((opt.cls_num,))
            for idx in range(len(self._gt)):
                gt_item = np.asarray(self._gt[idx]).astype(np.float32)

                # Discretize into bins
                indices = (
                    np.digitize(
                        gt_item.reshape(
                            -1,
                        ),
                        bins=self.cls_ranges,
                        right=False,
                    )
                    - 1
                )
                # For the last value: right=False skips the last value
                indices[indices == opt.cls_num] = opt.cls_num - 1

                hist = np.bincount(indices, minlength=opt.cls_num)
                bincounts += hist

            stats = {
                "bincounts": bincounts,
                "minval": minval,
                "maxval": maxval,
                "cls_ranges": self.cls_ranges,
            }
            with open(stats_path, "wb") as handle:
                pickle.dump(
                    stats,
                    handle,
                )

        # Plot stuff if we need it for later to see what's in the data
        if plot:
            plot_data_hist(
                self.cls_ranges[1:],
                stats["bincounts"],
                title="Data distribution " + split,
                yaxis="Counts per frame",
                xaxis="Frame number / Frame %",
                outfile=os.path.join(
                    opt.output_dir, "data" + split + str(opt.cls_num) + ".pdf"
                ),
            )
        return stats

    def cls_equ_stats(self, stats, split, minval, maxval, plot):
        """Given the histogram of samples per class, class ranges, equalize the histograms and define new ranges.
        Args:
            - stats: the dataset statistics (samples per class)
            - split: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        cls_path = os.path.join(
            opt.output_dir,
            "cls_equalized_"
            + split
            + opt.subaction
            + "_"
            + str(opt.cls_num)
            + "_"
            + opt.split
            + ("_rel" if opt.reltime else "_abs")
            + ".pkl",
        )
        # If equalized classes are there
        if os.path.exists(cls_path):
            with open(cls_path, "rb") as handle:
                (self.cls_keep_prob, self.cls_ranges) = pickle.load(handle)
        else:
            pre_stats = stats["bincounts"]
            pre_ranges = self.cls_ranges
            self.cls_ranges, stats["bincounts"] = cls_equalized_ranges(
                hist=pre_stats, ranges=pre_ranges
            )

            # We do not want to have less than 2 bins
            if (self.cls_ranges.size - 1) < 2:
                stats["bincounts"] = pre_stats
                self.cls_ranges = pre_ranges

            # Define per class keep probabilities
            hist = stats["bincounts"]
            minimum = hist.min()
            self.cls_keep_prob = float(minimum) / hist.astype(float)

            if plot:
                plot_data_hist(
                    self.cls_ranges[1:],
                    hist,
                    title="Equalized train-distribution",
                    yaxis="Counts over progress %",
                    xaxis="Progres %",
                    outfile=os.path.join(opt.output_dir, "equalized_" + split + ".pdf"),
                )
                plot_data_hist(
                    self.cls_ranges[1:],
                    self.cls_keep_prob,
                    title="Class probabilities",
                    yaxis="Keep prob per progress %",
                    xaxis="Progress %",
                    outfile=os.path.join(
                        opt.output_dir, "video_keep_prob_" + split + ".pdf"
                    ),
                )
            with open(cls_path, "wb") as handle:
                pickle.dump(
                    (self.cls_keep_prob, self.cls_ranges),
                    handle,
                )

        opt.cls_ranges = self.cls_ranges
        self.nclasses = self.cls_ranges.size - 1

    def cls_one_label(self, sample):
        """Returns the class label for one sample (target)."""

        if opt.cls_num <= 0:
            return 0

        # Define labels per sample
        label = (
            np.digitize(
                sample.reshape(
                    -1,
                ),
                bins=self.cls_ranges,
                right=False,
            )
            - 1
        )
        # For the right value: the end of the last bin
        label[label == opt.cls_num] = opt.cls_num - 1

        if self.cls_equalize:
            chance = np.random.rand(label[label != -1].size)
            label_prob = self.cls_keep_prob[label[label != -1]]
            wheres = np.where(chance >= label_prob)
            label[wheres] = -1
        return torch.Tensor(label).long()

    def __len__(self):
        return len(self._gt)

    def pick_set(self, use_set):
        """Splits the data into train val and test sets."""

        np.random.seed(0)
        if use_set.endswith("val"):
            indx = np.arange(0, self._features.shape[0])
            np.random.shuffle(indx)
            val_indx = indx[0 : self._features.shape[0] // 3]
            self._features = copy.deepcopy(self._features[val_indx, :])
            self._gt = copy.deepcopy(self._gt[val_indx, :])
        elif use_set.endswith("train"):
            indx = np.arange(0, self._features.shape[0])
            np.random.shuffle(indx)
            train_indx = indx[self._features.shape[0] // 3 : self._features.shape[0]]
            self._features = copy.deepcopy(self._features[train_indx])
            self._gt = copy.deepcopy(self._gt[train_indx])


def load_reltime(videos, is_train, shuffle, use_set):
    """Loads the features and ground truth from the videos and creates a data loader for it."""

    logger.debug("load data with temporal labels as ground truth")

    dataset = RelTimeDataset(videos, is_train, use_set)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers
    )
    return dataloader
