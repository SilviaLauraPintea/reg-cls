import math
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
from utils import check_rootfolders

# These change per function.
A = 0.5
B = 1.0
C = 12.0
D = 3.0


def func(x, noise_level_y=0):
    """
    Some non-linear function with parameters A, B, C, D.
    :noise_level_y: If noise is added to the function on the y-axis.
    :x: The extent of the function.
    """
    global A, B, C, D
    f = A * np.sin(C * x) + B * np.sin(D * x)

    if noise_level_y > 0:
        noise = np.random.normal(loc=0.0, scale=noise_level_y, size=f.shape)
        f += noise
    return f


def get_uniform(max_bins, all_samples, xlow=-1.0, xhigh=1.0, noise_level_y=0):
    """
    Get uniform samples between certain x-ranges.
    :max_bins: Maximum bins to use.
    :all_samples: Number of total samples.
    :xlow: Lower x-boundary.
    :xhigh: Highest x-boundary.
    :noise_level_y: If noise is added to the function on the y-axis.
    """
    x = np.random.uniform(low=xlow, high=xhigh, size=(100000, 1))
    assert len(np.unique(x)) == 100000
    y = func(x, noise_level_y)
    yminmax = [y.min(), y.max()]

    # Add minimum samples per bin
    y_keep = []
    x_keep = []
    min_spls = all_samples // max_bins + 1
    bins = np.linspace(yminmax[0], yminmax[1], max_bins + 1)
    for i in range(0, max_bins):
        wheres = np.where((y >= bins[i]) & (y <= bins[i + 1]))
        y_bin = y[wheres]
        x_bin = x[wheres]
        try:  # Some bins may not have samples
            samples = np.random.randint(0, y_bin.shape[0], min_spls)
            y_keep.append(y_bin[samples])
            x_keep.append(x_bin[samples])
        except:
            y_bin = []
            x_bin = []

    # Stack and return
    y_stack = np.concatenate(tuple(y_keep)).reshape(-1, 1)
    x_stack = np.concatenate(tuple(x_keep)).reshape(-1, 1)

    return x_stack, y_stack


def get_imbalanced(
    std,
    max_bins,
    min_spls,
    all_samples,
    xlow=-1.0,
    xhigh=1.0,
    ymean=None,
    noise_level_y=0,
):
    """
    Sample imbalanced samples with a certain range.
    :std: The ratio of the range to sample from.
    :max_bins: Maximum number of bins.
    :min_spls: Minimum number of samples per bin.
    :all_samples: Number of total samples.
    :xlow: Lower x-boundary.
    :xhigh: Highest x-boundary.
    :ymean: The y-value (peak) around which we sample.
    :noise_level_y: If noise is added to the function on the y-axis.
    """
    x = np.random.uniform(low=xlow, high=xhigh, size=(100000, 1))
    assert len(np.unique(x)) == 100000
    y = func(x, noise_level_y)
    yminmax = [y.min(), y.max()]

    y_keep = []
    x_keep = []

    # Add the samples from the current range
    var = std * (yminmax[1] - yminmax[0])
    if ymean is None:  # Sample ymean
        ymean = np.random.uniform(low=yminmax[0], high=yminmax[1])

    # Given y-mean
    if ymean > yminmax[0] and ymean < yminmax[1]:  # If in the current range
        spls = (all_samples - max_bins * min_spls) // max_bins * 2
        bins = np.linspace(ymean - var, ymean + var, max_bins + 1)
        for i in range(0, max_bins):
            wheres = np.where((y >= bins[i]) & (y <= bins[i + 1]))
            y_extra = y[wheres]
            x_extra = x[wheres]
            try:  # Some bins may not have samples
                samples = np.random.randint(0, y_extra.shape[0], spls)
                y_keep.append(y_extra[samples])
                x_keep.append(x_extra[samples])
            except:
                y_extra = []
                x_extra = []

        # Remove if we oversamples
        pre_x_stack = np.concatenate(tuple(x_keep)).reshape(-1, 1)
        pre_y_stack = np.concatenate(tuple(y_keep)).reshape(-1, 1)
        ridx = np.arange(0, pre_x_stack.shape[0])
        np.random.shuffle(ridx)
        x_keep = []
        y_keep = []
        x_keep = [pre_x_stack[ridx[0 : all_samples - max_bins - min_spls]]]
        y_keep = [pre_y_stack[ridx[0 : all_samples - max_bins - min_spls]]]

    # Add minimum samples per bin
    bins = np.linspace(yminmax[0], yminmax[1], max_bins + 1)
    for i in range(0, max_bins):
        wheres = np.where((y >= bins[i]) & (y <= bins[i + 1]))
        y_bin = y[wheres]
        x_bin = x[wheres]
        try:  # Some bins may not have samples
            samples = np.random.randint(0, y_bin.shape[0], min_spls + 1)
            y_keep.append(y_bin[samples].reshape(-1, 1))
            x_keep.append(x_bin[samples].reshape(-1, 1))
        except:
            y_bin = []
            x_bin = []

    # Stack and return
    y_stack = np.concatenate(tuple(y_keep)).reshape(-1, 1)
    x_stack = np.concatenate(tuple(x_keep)).reshape(-1, 1)

    return x_stack, y_stack


def get_ood_imbalanced_mean(x_train_ranges, x_val_ranges, x_test_ranges, yminmax):
    """
    Picks a y-value to use as a sampling peak (mean) that is all sets.
    :x_train_ranges: The training ranges.
    :x_val_ranges: The validation ranges.
    :x_test_ranges: The test ranges.
    :yminmax: The min/max y values.
    """

    is_in = False
    while not is_in:
        # Find a y-mean
        ymean = np.random.uniform(low=yminmax[0], high=yminmax[1])

        train_is_in = 0
        val_is_in = 0
        test_is_in = 0
        is_in_train = False
        is_in_test = False
        is_in_val = False
        for r in range(0, x_train_ranges.shape[1]):
            tr_xlow = x_train_ranges[0, r]
            tr_xhigh = x_train_ranges[1, r]
            tr_x = np.random.uniform(low=tr_xlow, high=tr_xhigh, size=(100000, 1))
            assert len(np.unique(tr_x)) == 100000
            tr_y = func(tr_x)
            tr_yminmax = [tr_y.min(), tr_y.max()]
            # If in at least one range
            if ymean >= tr_yminmax[0] and ymean <= tr_yminmax[1]:
                is_in_train = True
                train_is_in += 1

            te_xlow = x_test_ranges[0, r]
            te_xhigh = x_test_ranges[1, r]
            te_x = np.random.uniform(low=te_xlow, high=te_xhigh, size=(100000, 1))
            assert len(np.unique(te_x)) == 100000
            te_y = func(te_x)
            te_yminmax = [te_y.min(), te_y.max()]
            # If in at least one range
            if ymean >= te_yminmax[0] and ymean <= te_yminmax[1]:
                is_in_test = True
                test_is_in += 1

            va_xlow = x_val_ranges[0, r]
            va_xhigh = x_val_ranges[1, r]
            va_x = np.random.uniform(low=va_xlow, high=va_xhigh, size=(100000, 1))
            assert len(np.unique(va_x)) == 100000
            va_y = func(va_x)
            va_yminmax = [va_y.min(), va_y.max()]
            # If in at least one range
            if ymean >= va_yminmax[0] and ymean <= va_yminmax[1]:
                is_in_val = True
                val_is_in += 1

        is_in = is_in_test and is_in_train and is_in_val

    return ymean, train_is_in, val_is_in, test_is_in


def sample_sets(idx, x, noise_level_y):
    """
    Given a set of x-values and their corresponding indices, split it into training/validation/test.
    :idx: The data samples indices.
    :x: The x-values.
    :noise_level_y: If noise is added to the function on the y-axis.
    """
    train_data = {"x": None, "y": None}
    val_data = {"x": None, "y": None}
    test_data = {"x": None, "y": None}

    idx_tr = idx[0 : idx.shape[0] // 3]
    idx_va = idx[idx.shape[0] // 3 : (2 * idx.shape[0]) // 3]
    idx_te = idx[(2 * idx.shape[0]) // 3 : idx.shape[0]]

    train_data["x"] = x[idx_tr]
    val_data["x"] = x[idx_va]
    test_data["x"] = x[idx_te]

    train_data["y"] = func(train_data["x"], noise_level_y=noise_level_y)
    val_data["y"] = func(val_data["x"], noise_level_y=noise_level_y)
    test_data["y"] = func(test_data["x"], noise_level_y=noise_level_y)

    return train_data, val_data, test_data


def sample_noisey(std, seed, case, min_spls, all_samples, max_bins, noise_level=0.1):
    """
    Get the noisy dataset.
    :std: The ratio of the range to sample from.
    :seed: The random seed to use for sampling.
    :case: The data sampling case: uniform / imbalanced (std)
    :min_spls: Minimum number of samples per bin.
    :all_samples: Number of total samples.
    :max_bins: Maximum number of bins.
    :noise_level: The amount of noise to add to the targets on the y-axis.
    """
    # Get min max
    x = np.random.uniform(low=-1.0, high=1.0, size=(all_samples, 1))
    assert len(np.unique(x)) == all_samples
    f = func(x)
    yminmax = [f.min(), f.max()]

    # Set the seed
    np.random.seed(seed)

    train_data = {}
    val_data = {}
    test_data = {}
    if case.startswith("uniform"):
        x, _ = get_uniform(
            max_bins=max_bins, all_samples=all_samples, noise_level_y=noise_level
        )
        ridx = np.arange(0, x.shape[0])
        np.random.shuffle(ridx)
        train_data, val_data, test_data = sample_sets(
            ridx, x, noise_level_y=noise_level
        )

    else:  # For all the other imbalanced samplings
        x, _ = get_imbalanced(
            std,
            max_bins=max_bins,
            min_spls=min_spls,
            all_samples=all_samples,
        )
        ridx = np.arange(0, x.shape[0])
        np.random.shuffle(ridx)
        train_data, val_data, test_data = sample_sets(
            ridx, x, noise_level_y=noise_level
        )

    return train_data, val_data, test_data, yminmax


def sample_ood(std, seed, case, min_spls, all_samples, max_bins):
    """
    Get the ood dataset.
    :std: The ratio of the range to sample from.
    :seed: The random seed to use for sampling.
    :case: The data sampling case: uniform / imbalanced (std)
    :min_spls: Minimum number of samples per bin.
    :all_samples: Number of total samples.
    :max_bins: Maximum number of bins.
    """
    # Get min max
    x = np.random.uniform(low=-1.0, high=1.0, size=(all_samples, 1))
    assert len(np.unique(x)) == all_samples
    y = func(x)
    yminmax = [y.min(), y.max()]

    # Set the seed for the intervals
    np.random.seed(seed)

    train_data = {"x": None, "y": None}
    val_data = {"x": None, "y": None}
    test_data = {"x": None, "y": None}

    # Define the training and test ranges
    range_size = 1.0 / 4.5
    range_size_var = range_size / 4.0
    x_intervals = np.arange(-1.0, 1.0 + range_size_var, range_size)
    x_ranges = np.array([x_intervals[:-1], x_intervals[1:]])
    # Add noise to the ranges
    x_ranges[1, :] += np.abs(np.random.normal(0, range_size_var, x_ranges.shape[1]))
    x_ranges[1, :] = np.minimum(x_ranges[1], 1.0)

    # Divide these ranges between train/val and test.
    idx_test = (np.arange(0, x_ranges.shape[1] // 3)) * 3 + 2
    idx_val = (np.arange(0, x_ranges.shape[1] // 3)) * 3 + 1
    idx_train = (np.arange(0, x_ranges.shape[1] // 3)) * 3
    assert (
        idx_test.shape[0] == idx_val.shape[0] and idx_val.shape[0] == idx_train.shape[0]
    )

    x_train_ranges = x_ranges[:, idx_train]
    x_val_ranges = x_ranges[:, idx_val]
    x_test_ranges = x_ranges[:, idx_test]

    # Set the seed for the data
    np.random.seed(seed)
    if case.startswith("uniform"):
        train_data["x"] = []
        val_data["x"] = []
        test_data["x"] = []

        for r in range(0, idx_val.shape[0]):
            tr_xlow = x_train_ranges[0, r]
            tr_xhigh = x_train_ranges[1, r]
            tr_x, _ = get_uniform(
                max_bins=max_bins,
                all_samples=all_samples // (3 * idx_val.shape[0]),
                xlow=tr_xlow,
                xhigh=tr_xhigh,
            )
            train_data["x"].append(tr_x)

            te_xlow = x_test_ranges[0, r]
            te_xhigh = x_test_ranges[1, r]
            te_x, _ = get_uniform(
                max_bins=max_bins,
                all_samples=all_samples // (3 * idx_val.shape[0]),
                xlow=te_xlow,
                xhigh=te_xhigh,
            )
            test_data["x"].append(te_x)

            va_xlow = x_val_ranges[0, r]
            va_xhigh = x_val_ranges[1, r]
            va_x, _ = get_uniform(
                max_bins=max_bins,
                all_samples=all_samples // (3 * idx_val.shape[0]),
                xlow=va_xlow,
                xhigh=va_xhigh,
            )
            val_data["x"].append(va_x)

        train_data["x"] = np.vstack(train_data["x"]).reshape(-1, 1)
        val_data["x"] = np.vstack(val_data["x"]).reshape(-1, 1)
        test_data["x"] = np.vstack(test_data["x"]).reshape(-1, 1)

        train_data["y"] = func(train_data["x"])
        val_data["y"] = func(val_data["x"])
        test_data["y"] = func(test_data["x"])

    else:  # For all the other imbalanced samplings
        # Get a y-mean that is both in train and test
        ymean, train_is_in, val_is_in, test_is_in = get_ood_imbalanced_mean(
            x_train_ranges, x_val_ranges, x_test_ranges, yminmax
        )

        train_data["x"] = []
        val_data["x"] = []
        test_data["x"] = []

        for r in range(0, x_train_ranges.shape[1]):
            tr_xlow = x_train_ranges[0, r]
            tr_xhigh = x_train_ranges[1, r]
            tr_x, _ = get_imbalanced(
                std,
                max_bins=max_bins,
                min_spls=min_spls // x_train_ranges.shape[1],
                all_samples=all_samples // (3 * train_is_in),
                xlow=tr_xlow,
                xhigh=tr_xhigh,
                ymean=ymean,
            )
            train_data["x"].append(tr_x)

            va_xlow = x_val_ranges[0, r]
            va_xhigh = x_val_ranges[1, r]
            va_x, _ = get_imbalanced(
                std,
                max_bins=max_bins,
                min_spls=min_spls // x_val_ranges.shape[1],
                all_samples=all_samples // (3 * val_is_in),
                xlow=va_xlow,
                xhigh=va_xhigh,
                ymean=ymean,
            )
            val_data["x"].append(va_x)

            te_xlow = x_test_ranges[0, r]
            te_xhigh = x_test_ranges[1, r]
            te_x, _ = get_imbalanced(
                std,
                max_bins=max_bins,
                min_spls=min_spls // x_test_ranges.shape[1],
                all_samples=all_samples // (3 * test_is_in),
                xlow=te_xlow,
                xhigh=te_xhigh,
                ymean=ymean,
            )
            test_data["x"].append(te_x)

        train_data["x"] = np.vstack(train_data["x"]).reshape(-1, 1)
        val_data["x"] = np.vstack(val_data["x"]).reshape(-1, 1)
        test_data["x"] = np.vstack(test_data["x"]).reshape(-1, 1)

        train_data["y"] = func(train_data["x"])
        val_data["y"] = func(val_data["x"])
        test_data["y"] = func(test_data["x"])
    return train_data, val_data, test_data, yminmax


def sample_clean(std, seed, case, min_spls, all_samples, max_bins):
    """
    Get the clean dataset.
    :std: The ratio of the range to sample from.
    :seed: The random seed to use for sampling.
    :case: The data sampling case: uniform / imbalanced (std)
    :min_spls: Minimum number of samples per bin.
    :all_samples: Number of total samples.
    :max_bins: Maximum number of bins.
    """

    # Get min max
    x = np.random.uniform(low=-1.0, high=1.0, size=(all_samples, 1))
    assert len(np.unique(x)) == all_samples
    f = func(x)
    yminmax = [f.min(), f.max()]

    # Set the seed
    np.random.seed(seed)

    train_data = {}
    val_data = {}
    test_data = {}

    if case.startswith("uniform"):
        x, y = get_uniform(max_bins=max_bins, all_samples=all_samples)
        ridx = np.arange(0, x.shape[0])
        np.random.shuffle(ridx)
        train_data, val_data, test_data = sample_sets(ridx, x, noise_level_y=0)

    else:  # For all the other imbalanced samplings
        x, y = get_imbalanced(
            std, max_bins=max_bins, min_spls=min_spls, all_samples=all_samples
        )
        ridx = np.arange(0, x.shape[0])
        np.random.shuffle(ridx)
        train_data, val_data, test_data = sample_sets(ridx, x, noise_level_y=0)

    return train_data, val_data, test_data, yminmax


def just_write(train_data, val_data, test_data, yminmax, case, index, findex, path):
    """
    Just writes the datasets locally into pickles.
    :train_data: Training dataset.
    :val_data: Validation dataset.
    :test_data: Test dataset.
    :yminmax: The y min and max values for plotting.
    :case: The case (uniform/imbalanced) to write in the file name.
    :index: The dataset index (We generate the dataset with 5 different seeds).
    :findex: The function index (We sample 10 functions).
    :path: Path to save the pickle.
    """
    file_name = case + str(index) + "fct" + str(findex)

    print(
        case + " sizes: ",
        train_data["x"].shape,
        val_data["x"].shape,
        test_data["x"].shape,
        train_data["y"].shape,
        val_data["y"].shape,
        test_data["y"].shape,
    )

    plotdata(
        train_data,
        val_data,
        test_data,
        path,
        name=case + str(index) + "fct" + str(findex),
        yminmax=yminmax,
    )

    with open(path + file_name + "_train.pkl", "wb") as f:
        pickle.dump((train_data["x"], train_data["y"]), f)

    with open(path + file_name + "_val.pkl", "wb") as f:
        pickle.dump((val_data["x"], val_data["y"]), f)

    with open(path + file_name + "_test.pkl", "wb") as f:
        pickle.dump((test_data["x"], test_data["y"]), f)


def sample_case(
    case, sampling_func, std, seed, min_spls, all_samples, max_bins, index, findex, path
):
    """
    Calls the dataset sampling function (clean/noisy/ood) to create the datasets.
    :case: The data sampling case: uniform / imbalanced (std)
    :sampling_func: The sampling function to be used: <sample_clean>, <sample_noisey>, <sample_ood>
    :std: The ratio of the range to sample from.
    :seed: The random seed to use for sampling.
    :min_spls: Minimum number of samples per bin.
    :all_samples: Number of total samples.
    :max_bins: Maximum number of bins.
    :index: The dataset index (We generate the dataset with 5 different seeds).
    :findex: The function index (We sample 10 functions).
    :path: Path to save the pickles.
    """

    train_data, val_data, test_data, yminmax = sampling_func(
        std=std,
        seed=seed,
        case=case,
        min_spls=min_spls,
        all_samples=all_samples,
        max_bins=max_bins,
    )
    just_write(
        train_data,
        val_data,
        test_data,
        yminmax,
        case=case,
        index=index,
        findex=findex,
        path=path,
    )


def write_data(sampling_func, path, min_spls, all_samples, max_bins):
    """
    Creates the datasets.
    :sampling_func: The sampling function to be used: <sample_clean>, <sample_noisey>, <sample_ood>
    :path: Path to save the pickles.
    :min_spls: Minimum number of samples per bin.
    :all_samples: Number of total samples.
    :max_bins: Maximum number of bins.
    """

    # Generate 5 seeds
    np.random.seed(0)
    random.seed(0)
    seeds = np.random.randint(0, high=10000, size=(5,))

    # Generate 10 functions
    for j in range(0, 10):
        # Define the function parameters
        global A, B, C, D
        A = random.random() * 1.0
        B = 1.5 - A
        C = random.randint(9, 30)
        D = C // 3

        # For each function and each seed generate a dataset
        for i in range(seeds.shape[0]):
            # get the uniform sampling
            sample_case(
                case="uniform",
                sampling_func=sampling_func,
                std=0,
                seed=seeds[i],
                min_spls=min_spls,
                all_samples=all_samples,
                max_bins=max_bins,
                index=i,
                findex=j,
                path=path,
            )

            # mildly imbalanced
            sample_case(
                case="mild",
                sampling_func=sampling_func,
                std=0.3,
                seed=seeds[i],
                min_spls=min_spls,
                all_samples=all_samples,
                max_bins=max_bins,
                index=i,
                findex=j,
                path=path,
            )

            # moderately imbalanced
            sample_case(
                case="moderate",
                sampling_func=sampling_func,
                std=0.1,
                seed=seeds[i],
                min_spls=min_spls,
                all_samples=all_samples,
                max_bins=max_bins,
                index=i,
                findex=j,
                path=path,
            )

            # severely imbalanced
            sample_case(
                case="severe",
                sampling_func=sampling_func,
                std=0.03,
                seed=seeds[i],
                min_spls=min_spls,
                all_samples=all_samples,
                max_bins=max_bins,
                index=i,
                findex=j,
                path=path,
            )


def plotdata(
    train_data,
    val_data,
    test_data,
    outdir,
    name,
    xminmax=[-1, 1],
    yminmax=[-2.8, 2.8],
    num_bins=4,
):
    """
    Plot the generated dataset.
    :train_data: Training dataset.
    :val_data: Validation dataset.
    :test_data: Test dataset.
    :outdir: Where to save the figure.
    :name: The name of the figure file.
    :xminmax: The x min and max values for plotting.
    :yminmax: The y min and max values for plotting.
    :num_bins: The number of bins for plotting.
    """
    plt.rcParams.update({"font.size": 6})

    fig, (ax) = plt.subplots(2, 3)
    fig.tight_layout(pad=5.0)
    ax = ax.flatten()

    bins = np.linspace(yminmax[0], yminmax[1], num_bins + 1)
    alphas = np.linspace(0.2, 0.8, num_bins)

    for i in range(0, num_bins):
        ax[0].axhspan(bins[i], bins[i + 1], facecolor="gray", alpha=alphas[-i - 1])
    idx = np.random.randint(0, train_data["x"].shape[0], 1000).tolist()

    # Scatter train data
    ax[0].scatter(train_data["x"][idx], train_data["y"][idx], s=10, alpha=0.5)
    ax[0].set_ylim(yminmax[0], yminmax[1])
    ax[0].set_xlim(xminmax[0], xminmax[1])
    ax[0].set_title(f"Train")
    ax[0].set_ylabel("f(x)")
    ax[0].set_xlabel("x")

    for i in range(0, num_bins):
        ax[1].axhspan(bins[i], bins[i + 1], facecolor="gray", alpha=alphas[-i - 1])
    idx = np.random.randint(0, val_data["x"].shape[0], 1000).tolist()

    # Scatter the test data
    ax[1].scatter(val_data["x"][idx], val_data["y"][idx], s=10, alpha=0.5)
    ax[1].set_ylim(yminmax[0], yminmax[1])
    ax[1].set_xlim(xminmax[0], xminmax[1])
    ax[1].set_title(f"Val")
    ax[1].set_ylabel("f(x)")
    ax[1].set_xlabel("x")

    for i in range(0, num_bins):
        ax[2].axhspan(bins[i], bins[i + 1], facecolor="gray", alpha=alphas[-i - 1])
    idx = np.random.randint(0, test_data["x"].shape[0], 1000).tolist()

    # Scatter the test data
    ax[2].scatter(test_data["x"][idx], test_data["y"][idx], s=10, alpha=0.5)
    ax[2].set_ylim(yminmax[0], yminmax[1])
    ax[2].set_xlim(xminmax[0], xminmax[1])
    ax[2].set_title(f"Test")
    ax[2].set_ylabel("f(x)")
    ax[2].set_xlabel("x")

    count, bins, ignored = ax[3].hist(
        train_data["y"], 254, orientation="horizontal", density=False
    )
    ax[3].set_ylim(yminmax[0], yminmax[1])
    ax[3].set_title(f"train-distribution")
    ax[3].set_ylabel("f(x)")
    ax[3].set_xlabel("log-counts of f(x)")
    ax[3].set_xscale("log")

    count, bins, ignored = ax[4].hist(
        val_data["y"], 254, orientation="horizontal", density=False
    )
    ax[4].set_ylim(yminmax[0], yminmax[1])
    ax[4].set_title(f"val-distribution")
    ax[4].set_ylabel("f(x)")
    ax[4].set_xlabel("log-counts of f(x)")
    ax[4].set_xscale("log")

    count, bins, ignored = ax[5].hist(
        test_data["y"], 254, orientation="horizontal", density=False
    )
    ax[5].set_ylim(yminmax[0], yminmax[1])
    ax[5].set_title(f"test-distribution")
    ax[5].set_ylabel("f(x)")
    ax[5].set_xlabel("log-counts of f(x)")
    ax[5].set_xscale("log")

    plt.savefig(
        os.path.join(outdir, name + ".pdf"),
        format="pdf",
        pad_inches=-2,
        transparent=False,
        dpi=300,
    )
    # plt.show()
    plt.close()


if __name__ == "__main__":
    # write data clean
    check_rootfolders("data/", "")

    sampling_func = sample_clean
    path = "data/clean/"
    check_rootfolders(path, "")
    write_data(sampling_func, path, min_spls=5, all_samples=60000, max_bins=1024)

    # write noisy-y data
    sampling_func = sample_noisey
    path = "data/noisey_0.1/"
    check_rootfolders(path, "")
    write_data(sampling_func, path, min_spls=5, all_samples=60000, max_bins=1024)

    # write ood data
    sampling_func = sample_ood
    path = "data/ood/"
    check_rootfolders(path, "")
    write_data(sampling_func, path, min_spls=5, all_samples=60000, max_bins=1024)
