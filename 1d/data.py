import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from histeq import cls_equalized_ranges


class SynthDataset(Dataset):
    """Define the dataset."""

    def __init__(
        self,
        args,
        batch_size,
        setname,
        num_segment,
        seg_ranges=None,
    ):

        np.random.seed(args.random_seed)
        self.batch_size_ = batch_size
        self.setname = setname
        self.num_segment = num_segment
        self.seg_ranges = seg_ranges
        self.do_equ = args.equalize_segments

        # Get the raw data
        self.get_raw_data(args)

    def get_labels(self, args):
        if self.do_equ:
            if not self.is_test:
                self.seg_ranges, histeq = cls_equalized_ranges(
                    arr=self.target_, nbins=self.num_segment
                )
                args.orig_num_segment = args.num_segment
                self.num_segment = self.seg_ranges.size - 1
                args.num_segment = self.seg_ranges.size - 1

                minimum = histeq.min()
                self.seg_keep_prob = float(minimum) / histeq.astype(float)

        else:
            self.seg_ranges = np.linspace(
                self.target_.min(), self.target_.max(), self.num_segment + 1
            )
            args.orig_num_segment = args.num_segment

        # Define labels per sample
        self.label_ = np.zeros(self.target_.shape)
        for i in range(0, self.num_segment):
            wheres = np.where(
                (self.target_ >= self.seg_ranges[i])
                & (self.target_ < self.seg_ranges[i + 1])
            )
            self.label_[wheres] = i

        wheres = np.where((self.target_ >= self.seg_ranges[self.num_segment]))
        self.label_[wheres] = self.num_segment - 1

    def get_raw_data(self, args):
        if self.setname.startswith("train"):
            with open(args.path_train, "rb") as f:
                (self.data_, self.target_) = pickle.load(f)
            self.is_test = False
        elif self.setname.startswith("val"):
            with open(args.path_val, "rb") as f:
                (self.data_, self.target_) = pickle.load(f)
            self.is_test = True
        elif self.setname.startswith("test"):
            with open(args.path_test, "rb") as f:
                (self.data_, self.target_) = pickle.load(f)
            self.is_test = True

        self.get_labels(args)

    def __len__(self):
        return self.data_.shape[0]

    def __dim__(self):
        return self.data_.shape[1]

    def clip_labels(self, label):
        """
        Sample data with per class keep probability.
        We ignore during training the class 'num_segment'.
        """
        if self.do_equ and not self.is_test:
            chance = np.random.rand(label.size)
            label_prob = self.seg_keep_prob[label]
            wheres = np.where(chance >= label_prob)
            label[wheres] = -1
            return label
        else:
            return label

    def __getitem__(self, idx):
        """
        Gets an item of this dataset.
        """
        label = self.label_[idx : idx + 1].astype(int)
        label = self.clip_labels(label)

        return (
            torch.from_numpy(self.data_[idx, :]).double(),
            torch.from_numpy(self.target_[idx : idx + 1]).double(),
            torch.from_numpy(label).long(),
        )
