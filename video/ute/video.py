#!/usr/bin/env python

""" Module with class for single video. """

from collections import Counter
import numpy as np
import math as m
import os
from os.path import join

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import dir_check


class Video(object):
    """Single video class"""

    def __init__(self, path, name=""):
        """
        Args:
            path (str): path to video representation
            reset (bool): necessity of holding features in each instance
            name (str): short name without any extension
        """
        self.path = path
        self.name = name
        self.n_frames = 0

        # load the features
        self._features = None
        self.features()

        # temporal labels
        self._temp = None
        self._init_temporal_labels()

    def features(self):
        """Load features given path, if haven't done before"""

        if self._features is None:
            if opt.ext == "npy":
                self._features = np.load(self.path)
            else:
                self._features = np.loadtxt(self.path)
            self.n_frames = self._features.shape[0]
        return self._features

    def _init_temporal_labels(self):
        """Get the temporal labels"""

        self._temp = np.zeros(self.n_frames)
        for frame_idx in range(self.n_frames):
            if opt.reltime:
                self._temp[frame_idx] = frame_idx / float(self.n_frames)
            else:
                self._temp[frame_idx] = frame_idx

    def reset(self):
        """If features from here won't be in use anymore"""
        self._temp = None
        self._features = None
