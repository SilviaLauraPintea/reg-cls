import torch
import numpy as np
from PIL import Image
import os
import math
import logging
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from nyu_transform import (
    Lighting,
    Normalize,
    ToTensor,
    Scale,
    CenterCrop,
    RandomRotate,
    ColorJitter,
    RandomHorizontalFlip,
)
from scipy.ndimage import convolve1d
from util import get_lds_kernel_window, plot_data_hist
import pickle
from histeq import cls_equalized_ranges
import copy
import torch.nn as nn

# for data loading efficiency
TRAIN_BUCKET_NUM = [
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    25848691,
    24732940,
    53324326,
    69112955,
    54455432,
    95637682,
    71403954,
    117244217,
    84813007,
    126524456,
    84486706,
    133130272,
    95464874,
    146051415,
    146133612,
    96561379,
    138366677,
    89680276,
    127689043,
    81608990,
    119121178,
    74360607,
    106839384,
    97595765,
    66718296,
    90661239,
    53103021,
    83340912,
    51365604,
    71262770,
    42243737,
    65860580,
    38415940,
    53647559,
    54038467,
    28335524,
    41485143,
    32106001,
    35936734,
    23966211,
    32018765,
    19297203,
    31503743,
    21681574,
    16363187,
    25743420,
    12769509,
    17675327,
    13147819,
    15798560,
    9547180,
    14933200,
    9663019,
    12887283,
    11803562,
    7656609,
    11515700,
    7756306,
    9046228,
    5114894,
    8653419,
    6859433,
    8001904,
    6430700,
    3305839,
    6318461,
    3486268,
    5621065,
    4030498,
    3839488,
    3220208,
    4483027,
    2555777,
    4685983,
    3145082,
    2951048,
    2762369,
    2367581,
    2546089,
    2343867,
    2481579,
    1722140,
    3018892,
    2325197,
    1952354,
    2047038,
    1858707,
    2052729,
    1348558,
    2487278,
    1314198,
    3338550,
    1132666,
]


class depthDataset(Dataset):
    def __init__(
        self,
        data_dir,
        csv_file,
        is_train,
        cls_num,
        use_set,  # train / val / all
        mask_file=None,
        transform=None,
        args=None,
    ):
        self.is_train = is_train
        self.data_dir = data_dir
        self.cls_equalize = args.cls_equalize

        # Read the data file paths
        frame = pd.read_csv(csv_file, header=None)
        self.mask = (
            torch.tensor(np.load(mask_file), dtype=torch.bool)
            if mask_file is not None
            else None
        )

        # Picking the train/val/test sets
        self.use_set = use_set
        if not self.use_set.endswith("all"):
            dfs = self.define_train_val(frame)
            if self.use_set.endswith("val"):
                self.frame = dfs["val"]
            elif self.use_set.endswith("train"):
                self.frame = dfs["train"]
        else:
            self.frame = frame

        # Always equalize the val set to make it balanced
        if self.use_set.endswith("val"):
            self.cls_equalize = True

        self.transform = transform
        self.bucket_weights = self._get_bucket_weights(args) if args.fds else None

        # Gets the target statistics
        self.nclasses = cls_num
        if self.nclasses > 0:
            self.cls_data_stats(args, use_set)

    def define_train_val(self, frame):
        """Splits the training dirs into train and val.
        Args:
            - frame: the frame names (pandas df) belonging to the training set.
        """

        fpath = os.path.join(self.data_dir, "train_val_dfs.pkl")

        if os.path.exists(fpath):
            try:
                dfs = pd.read_pickle(fpath)
            except:
                with open(fpath, "rb") as handle:
                    dfs = pickle.load(handle)
        else:
            np.random.seed(0)
            all_dirs = []
            for i in range(0, len(frame)):
                image_name = frame.iloc[i, 0]
                dir_split = image_name.split("/")[2]
                all_dirs.append(dir_split)

            all_udirs = np.unique(all_dirs)
            indexes = np.arange(0, len(all_udirs))
            np.random.shuffle(indexes)

            train_dirs = list(all_udirs[indexes[0 : 4 * len(all_udirs) // 5]])
            val_dirs = list(
                all_udirs[indexes[4 * len(all_udirs) // 5 : len(all_udirs)]]
            )

            traindf = copy.deepcopy(frame)
            valdf = copy.deepcopy(frame)
            for i in range(0, len(frame)):
                image_name = frame.iloc[i, 0]
                dir_split = image_name.split("/")[2]
                if dir_split in train_dirs:
                    valdf.drop(i, inplace=True)
                elif dir_split in val_dirs:
                    traindf.drop(i, inplace=True)
                    print(len(traindf))

            # Write the class stats locally
            dfs = {"val": valdf, "train": traindf}
            with open(fpath, "wb") as handle:
                pickle.dump(
                    dfs,
                    handle,
                    # protocol=pickle.HIGHEST_PROTOCOL
                )
        return dfs

    def read_data(self, idx):
        """Reads the data entry and index <idx>."""

        image_name = self.frame.iloc[idx, 0]
        depth_name = self.frame.iloc[idx, 1]

        image_name = os.path.join(self.data_dir, "/".join(image_name.split("/")[1:]))
        depth_name = os.path.join(self.data_dir, "/".join(depth_name.split("/")[1:]))

        image = Image.open(image_name)
        depth = Image.open(depth_name)
        sample = {"image": image, "depth": depth}

        # This is for getting depth at the right resolution
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_bucket_weights(self, args):
        assert args.reweight in {"none", "inverse", "sqrt_inv"}
        assert (
            args.reweight != "none" if args.lds else True
        ), "Set reweight to 'sqrt_inv' or 'inverse' (default) when using LDS"
        if args.reweight == "none":
            return None
        logging.info(f"Using re-weighting: [{args.reweight.upper()}]")

        if args.lds:
            value_lst = TRAIN_BUCKET_NUM[args.bucket_start :]
            lds_kernel_window = get_lds_kernel_window(
                args.lds_kernel, args.lds_ks, args.lds_sigma
            )
            logging.info(
                f"Using LDS: [{args.lds_kernel.upper()}] ({args.lds_ks}/{args.lds_sigma})"
            )
            if args.reweight == "sqrt_inv":
                value_lst = np.sqrt(value_lst)
            smoothed_value = convolve1d(
                np.asarray(value_lst), weights=lds_kernel_window, mode="reflect"
            )
            smoothed_value = [smoothed_value[0]] * args.bucket_start + list(
                smoothed_value
            )
            scaling = np.sum(TRAIN_BUCKET_NUM) / np.sum(
                np.array(TRAIN_BUCKET_NUM) / np.array(smoothed_value)
            )
            bucket_weights = [
                np.float32(scaling / smoothed_value[bucket])
                for bucket in range(args.bucket_num)
            ]
        else:
            value_lst = [
                TRAIN_BUCKET_NUM[args.bucket_start]
            ] * args.bucket_start + TRAIN_BUCKET_NUM[args.bucket_start :]
            if args.reweight == "sqrt_inv":
                value_lst = np.sqrt(value_lst)
            scaling = np.sum(TRAIN_BUCKET_NUM) / np.sum(
                np.array(TRAIN_BUCKET_NUM) / np.array(value_lst)
            )
            bucket_weights = [
                np.float32(scaling / value_lst[bucket])
                for bucket in range(args.bucket_num)
            ]
        return bucket_weights

    def cls_data_stats(self, args, use_set, minval=0.0, maxval=10.0, plot=True):
        """Get the class data statistics.
        Args:
            - args: the parse arguments
            - use_set: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        # Specifics of this dataset
        self.minval = minval
        self.maxval = maxval
        self.cls_ranges = np.linspace(minval, maxval, args.cls_num + 1)

        # Loads the class statistics
        print(
            "--------------------------------- DATA STATS -----------------------------------------------"
        )
        stats = self.cls_stats(args, use_set, minval, maxval, plot)

        # Load the equalized statistics if needed
        if self.cls_equalize:

            print(
                "--------------------------------- EQU DATA STATS -----------------------------------------------"
            )
            self.cls_equ_stats(stats, args, use_set, minval, maxval, plot)

    def cls_stats(self, args, use_set, minval, maxval, plot):
        """First just compute the histogram of samples per class, class ranges, and the number of samples per bin.
        Args:
            - args: the parse arguments
            - use_set: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        if use_set.endswith("all"):
            if self.is_train:
                use_set = "trainall"
            else:
                use_set = "test"

        stats_path = os.path.join(
            self.data_dir, "target_stats_" + use_set + str(args.cls_num) + ".pkl"
        )

        # Load the training stats if they are there
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "rb") as handle:
                    stats = pickle.load(handle)
            except:
                import pickle5 as pkl5

                with open(stats_path, "rb") as handle:
                    stats = pkl5.load(handle)

        # Compute it if it does not exist
        else:
            bincounts = np.zeros((args.cls_num,))
            for i in range(0, len(self.frame)):
                sample = self.read_data(i)

                # Define bins and compute histogram
                depth = sample["depth"]
                assert depth.min() >= minval and depth.max() <= maxval

                # Discretize into bins
                indices = (
                    np.digitize(
                        depth.reshape(
                            -1,
                        ),
                        bins=self.cls_ranges,
                        right=False,
                    )
                    - 1
                )
                # For the end of the last bin
                indices[indices == args.cls_num] = args.cls_num - 1

                hist = np.bincount(indices, minlength=args.cls_num)
                bincounts += hist
            stats = {
                "bincounts": bincounts,
                "minval": minval,
                "maxval": maxval,
                "bin_ranges": self.cls_ranges,
            }

            # Write the class stats locally
            with open(stats_path, "wb") as handle:
                pickle.dump(
                    stats,
                    handle,
                    # protocol=pickle.HIGHEST_PROTOCOL
                )

        # If plot for data analysis purposes
        if plot:
            plot_data_hist(
                self.cls_ranges[1:],
                stats["bincounts"],
                title="Data distribution " + use_set,
                yaxis="Counts over depth",
                xaxis="Depth",
                outfile="logs/data_" + use_set + str(args.cls_num) + ".pdf",
            )

        return stats

    def cls_equ_stats(self, stats, args, use_set, minval, maxval, plot):
        """Given the histogram of samples per class, class ranges, equalize the histograms and define new ranges.
        Args:
            - stats: the dataset statistics (samples per class)
            - args: the parse arguments
            - use_set: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        if use_set.endswith("all"):
            if self.is_train:
                use_set = "trainall"
            else:
                use_set = "test"
        cls_path = os.path.join(
            self.data_dir, "cls_equalized_" + use_set + str(args.cls_num) + ".pkl"
        )

        # If equalized classes are there
        if os.path.exists(cls_path):
            try:
                with open(cls_path, "rb") as handle:
                    (self.cls_keep_prob, self.cls_ranges) = pickle.load(handle)
            except:
                import pickle5 as pkl5

                with open(cls_path, "rb") as handle:
                    (self.cls_keep_prob, self.cls_ranges) = pkl5.load(handle)

        # Else create it on the train data
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

            # Plot for data analysis purpose:
            if plot:
                plot_data_hist(
                    self.cls_ranges[1:],
                    stats["bincounts"],
                    title="Equalized distribution " + use_set,
                    yaxis="Counts over depth",
                    xaxis="Depth",
                    outfile="logs/equalized_" + use_set + str(args.cls_num) + ".pdf",
                )
                plot_data_hist(
                    self.cls_ranges[1:],
                    self.cls_keep_prob,
                    title="Cls probs " + use_set,
                    yaxis="Keep prob per depth",
                    xaxis="Depth",
                    outfile="logs/cls_keep_prob_"
                    + use_set
                    + str(args.cls_num)
                    + ".pdf",
                )

            # Write the stats locally to not recompute them
            with open(cls_path, "wb") as handle:
                pickle.dump(
                    (self.cls_keep_prob, self.cls_ranges),
                    handle,
                    # protocol=pickle.HIGHEST_PROTOCOL,
                )

        # Redefine class ranges and number of classes
        args.cls_ranges = self.cls_ranges
        self.nclasses = self.cls_ranges.size - 1

    def cls_one_label(self, orig_sample):
        """Get the label for one sample given its targets.
        Args:
            - orig_sample: the original targets.
        """

        if self.nclasses <= 0:
            return 0

        # The training targets are 2x smaller than the predictions.
        if self.is_train:
            sample = nn.functional.interpolate(
                orig_sample.unsqueeze(0),
                size=[orig_sample.shape[1] * 2, orig_sample.shape[2] * 2],
                mode="bilinear",
                align_corners=True,
            )
        else:
            sample = orig_sample

        # Define labels per sample
        label = (
            np.digitize(
                sample.view(
                    -1,
                ),
                bins=self.cls_ranges,
                right=False,
            )
            - 1
        )
        # For the end of the last bin
        label[label == self.nclasses] = self.nclasses - 1

        # Ignore samples per bin with a certain probability
        if self.cls_equalize:
            chance = np.random.rand(label[label != -1].size)
            label_prob = self.cls_keep_prob[label[label != -1]]
            wheres = np.where(chance >= label_prob)
            label[wheres] = -1
        return torch.Tensor(label).long()

    def get_bin_idx(self, x):
        return min(int(x * np.float32(10)), 99)

    def _get_weights(self, depth):
        sp = depth.shape
        if self.bucket_weights is not None:
            depth = depth.view(-1).cpu().numpy()
            assert depth.dtype == np.float32
            weights = np.array(
                list(map(lambda v: self.bucket_weights[self.get_bin_idx(v)], depth))
            )
            weights = torch.tensor(weights, dtype=torch.float32).view(*sp)
        else:
            weights = torch.tensor([np.float32(1.0)], dtype=torch.float32).repeat(*sp)
        return weights

    def __getitem__(self, idx):
        sample = self.read_data(idx)

        sample["weight"] = self._get_weights(sample["depth"])
        sample["idx"] = idx

        if self.mask is not None:
            sample["mask"] = self.mask[idx].unsqueeze(0)
        else:
            sample["mask"] = 0

        # Get the quantified depth
        sample["depth_cls"] = self.cls_one_label(sample["depth"])
        return sample

    def __len__(self):
        return len(self.frame)


def getTrainingData(args, cls_num, use_set, batch_size):
    __imagenet_pca = {
        "eigval": torch.Tensor([0.2175, 0.0188, 0.0045]),
        "eigvec": torch.Tensor(
            [
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ]
        ),
    }
    __imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    if use_set.endswith("val"):
        is_train = False
        transform = transforms.Compose(
            [
                Scale(240),
                CenterCrop([304, 228], [304, 338]),
                ToTensor(is_test=False),
                Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
            ]
        )
    else:
        is_train = True
        transform = transforms.Compose(
            [
                Scale(240),
                RandomHorizontalFlip(),
                RandomRotate(5),
                CenterCrop([304, 228], [152, 114]),
                ToTensor(is_test=False),
                Lighting(0.1, __imagenet_pca["eigval"], __imagenet_pca["eigvec"]),
                ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                ),
                Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
            ]
        )

    transformed_training = depthDataset(
        data_dir=args.data_dir,
        csv_file=os.path.join(args.data_dir, "nyu2_train.csv"),
        is_train=is_train,
        cls_num=cls_num,
        use_set=use_set,
        transform=transform,
        args=args,
    )

    dataloader_training = DataLoader(
        transformed_training,
        batch_size,
        shuffle=is_train,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    return dataloader_training


def getTrainingFDSData(args, cls_num, use_set, batch_size):
    __imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    target_size = [152, 114]
    transformed_training = depthDataset(
        data_dir=args.data_dir,
        csv_file=os.path.join(args.data_dir, "nyu2_train_FDS_subset.csv"),
        is_train=False,
        cls_num=cls_num,
        use_set=use_set,
        transform=transforms.Compose(
            [
                Scale(240),
                CenterCrop([304, 228], target_size),
                ToTensor(),
                Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
            ]
        ),
        args=args,
    )

    dataloader_training = DataLoader(
        transformed_training,
        batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    return dataloader_training


def getTestingData(args, cls_num, use_set, batch_size):

    __imagenet_stats = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

    transformed_testing = depthDataset(
        data_dir=args.data_dir,
        csv_file=os.path.join(args.data_dir, "nyu2_test.csv"),
        mask_file=os.path.join(args.data_dir, "test_balanced_mask.npy"),
        is_train=False,
        cls_num=cls_num,
        use_set=use_set,
        transform=transforms.Compose(
            [
                Scale(240),
                CenterCrop([304, 228], [304, 228]),
                ToTensor(is_test=True),
                Normalize(__imagenet_stats["mean"], __imagenet_stats["std"]),
            ]
        ),
        args=args,
    )

    dataloader_testing = DataLoader(
        transformed_testing, batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    return dataloader_testing
