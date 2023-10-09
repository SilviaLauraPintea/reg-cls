import torch
import os
import numpy as np
from PIL import Image
from scipy.ndimage import convolve1d
from torch.utils import data
import torchvision.transforms as transforms

from utils import get_lds_kernel_window, plot_data_hist
from histeq import cls_equalized_ranges
import pickle


class IMDBWIKI(data.Dataset):
    def __init__(self, args, df, cls_num, split="train", cls_keep_prob=None):
        self.df = df
        self.data_dir = args.data_dir
        self.img_size = args.img_size
        self.split = split

        self.weights = self._prepare_weights(
            reweight=args.reweight,
            lds=args.lds,
            lds_kernel=args.lds_kernel,
            lds_ks=args.lds_ks,
            lds_sigma=args.lds_sigma,
        )

        # Gets the target statistics
        self.nclasses = cls_num
        if self.nclasses > 0:
            self.cls_data_stats(args, split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        img = Image.open(os.path.join(self.data_dir, row["path"])).convert("RGB")
        transform = self.get_transform()
        img = transform(img)

        label = np.asarray([row["age"]]).astype("float32")
        cls_label = self.cls_one_label(label)

        weight = (
            np.asarray([self.weights[index]]).astype("float32")
            if self.weights is not None
            else np.asarray([np.float32(1.0)])
        )

        return img, label, cls_label, weight

    def cls_data_stats(self, args, split, minval=0.0, maxval=100.0, plot=True):
        """Get the class data statistics.
        Args:
            - args: the parse arguments
            - split: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        # Specifics of this dataset
        self.cls_equalize = args.cls_equalize
        self.cls_ranges = np.linspace(minval, maxval, args.cls_num + 1)

        # Load the training stats if they are there
        stats = self.cls_stats(args, split, minval, maxval, plot)

        # Load the training stats if they are there
        if self.cls_equalize:
            self.cls_equ_stats(stats, args, split, minval, maxval, plot)

    def cls_stats(self, args, split, minval, maxval, plot):
        """First just compute the histogram of samples per class, class ranges, and the number of samples per bin.
        Args:
            - args: the parse arguments
            - split: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        stats_path = os.path.join(
            self.data_dir, "target_stats_" + split + str(args.cls_num) + ".pkl"
        )

        # Try to open the data stats
        if os.path.exists(stats_path):
            try:
                with open(stats_path, "rb") as handle:
                    stats = pickle.load(handle)
            except:
                import pickle5 as pkl5

                with open(stats_path, "rb") as handle:
                    stats = pkl5.load(handle)

        # Else recompute them and save them
        else:
            bincounts = np.zeros((args.cls_num,))
            for i in range(0, len(self.df)):
                row = self.df.iloc[i]
                label = np.asarray([row["age"]]).astype("float32")

                # Discretize into bins
                indices = (
                    np.digitize(
                        label.reshape(
                            -1,
                        ),
                        bins=self.cls_ranges,
                        right=False,
                    )
                    - 1
                )
                # For the last value of the last bin
                indices[indices == args.cls_num] = args.cls_num - 1

                hist = np.bincount(indices, minlength=args.cls_num)
                bincounts += hist

            stats = {
                "bincounts": bincounts,
                "minval": minval,
                "maxval": maxval,
                "bin_ranges": self.cls_ranges,
            }

            with open(stats_path, "wb") as handle:
                pickle.dump(
                    stats,
                    handle,
                    # protocol=pickle.HIGHEST_PROTOCOL
                )

        # Plot stuff if we need it for later to see what's in the data
        if plot:
            plot_data_hist(
                self.cls_ranges[1:],
                stats["bincounts"],
                title="Data distribution " + split,
                yaxis="Counts over age",
                xaxis="Age",
                outfile="logs/data" + split + str(args.cls_num) + ".pdf",
            )
        return stats

    def cls_equ_stats(self, stats, args, split, minval, maxval, plot):
        """Given the histogram of samples per class, class ranges, equalize the histograms and define new ranges.
        Args:
            - stats: the dataset statistics (samples per class)
            - args: the parse arguments
            - split: train/val/test
            - minval: the minimum possible target value
            - maxval: the maximum possible target value
            - plot: if plots should be made of the class histograms
        """

        # Define the labels given the data stats
        cls_path = os.path.join(
            self.data_dir, "cls_equalized_" + split + str(args.cls_num) + ".pkl"
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
                hist=stats["bincounts"], ranges=self.cls_ranges
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
                    stats["bincounts"],
                    title="Equalized distribution " + split,
                    yaxis="Counts over age",
                    xaxis="Age",
                    outfile="logs/equalized_data" + split + str(args.cls_num) + ".pdf",
                )
                plot_data_hist(
                    self.cls_ranges[1:],
                    self.cls_keep_prob,
                    title="Equalized distribution " + split,
                    yaxis="Keep prob per age",
                    xaxis="Age",
                    outfile="logs/cls_keep_prob" + split + str(args.cls_num) + ".pdf",
                )
            with open(cls_path, "wb") as handle:
                pickle.dump(
                    (self.cls_keep_prob, self.cls_ranges),
                    handle,
                )

        # Redefine class ranges and number of classes
        args.cls_ranges = self.cls_ranges
        self.nclasses = self.cls_ranges.size - 1

    def cls_one_label(self, sample):
        """Get the label for one sample given its targets.
        Args:
            - sample: the original targets.
        """

        if self.nclasses <= 0:
            return 0

        # Define labels per sample
        label = (
            np.digitize(
                sample,
                bins=self.cls_ranges,
                right=False,
            )
            - 1
        )
        # For the last value of the last bin
        label[label == self.nclasses] = self.nclasses - 1

        # Drop labels with a class probability (this is 1 sample)
        if self.cls_equalize and label != -1:
            chance = np.random.rand(1)
            if chance >= self.cls_keep_prob[label]:
                label[0] = -1
        return torch.Tensor(label).long()

    def get_transform(self):
        if self.split == "train":
            transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.RandomCrop(self.img_size, padding=16),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize((self.img_size, self.img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        return transform

    def _prepare_weights(
        self,
        reweight,
        max_target=121,
        lds=False,
        lds_kernel="gaussian",
        lds_ks=5,
        lds_sigma=2,
    ):
        assert reweight in {"none", "inverse", "sqrt_inv"}
        assert (
            reweight != "none" if lds else True
        ), "Set reweight to 'sqrt_inv' (default) or 'inverse' when using LDS"

        value_dict = {x: 0 for x in range(max_target)}
        labels = self.df["age"].values
        for label in labels:
            value_dict[min(max_target - 1, int(label))] += 1
        if reweight == "sqrt_inv":
            value_dict = {k: np.sqrt(v) for k, v in value_dict.items()}
        elif reweight == "inverse":
            value_dict = {
                k: np.clip(v, 5, 1000) for k, v in value_dict.items()
            }  # clip weights for inverse re-weight
        num_per_label = [
            value_dict[min(max_target - 1, int(label))] for label in labels
        ]
        if not len(num_per_label) or reweight == "none":
            return None
        print(f"Using re-weighting: [{reweight.upper()}]")

        if lds:
            lds_kernel_window = get_lds_kernel_window(lds_kernel, lds_ks, lds_sigma)
            print(f"Using LDS: [{lds_kernel.upper()}] ({lds_ks}/{lds_sigma})")
            smoothed_value = convolve1d(
                np.asarray([v for _, v in value_dict.items()]),
                weights=lds_kernel_window,
                mode="constant",
            )
            num_per_label = [
                smoothed_value[min(max_target - 1, int(label))] for label in labels
            ]

        weights = [np.float32(1 / x) for x in num_per_label]
        scaling = len(weights) / np.sum(weights)
        weights = [scaling * x for x in weights]
        return weights
