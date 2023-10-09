import numpy as np
import math
from matplotlib import pyplot as plt


def removeStartZeros(hist, ranges):
    """Remove starting zeros from the histogram and ranges, if any."""

    i = 0
    while hist[i] == 0:
        i += 1

    if i > 0 and i < len(hist):
        new_hist = hist[i:]
        new_ranges = [x for n in (ranges[0:1], ranges[i + 1 :]) for x in n]
        return new_hist, np.array(new_ranges)
    return hist, ranges


def doEq(hist, nbins):
    """
    Returns an equalized histogram and the mapping function (Code adapted from JC van Gemert).
    Args:
        hist: Unequalized histogram.
        nbins: The number of bins.
    Returns:
        q: The mapping function.
        eqHist: The equalized histogram.

    Note: The bins are indexed from '0' to 'nbins-1' (not '1' to 'nbins').
    """
    nrPixels = hist.sum()

    cumulHist = 0
    q = np.zeros(nbins, dtype=int)
    eqHist = np.zeros(nbins)

    for i in range(nbins):
        cumulHist += hist[i]

        # Define q[i] to be zero for possible empty bins in the beginning
        # (if cumulHist = 0 then the -1 would index them at position -1)
        if cumulHist > 0:
            wantedCumulBin = nbins / nrPixels * cumulHist
            q[i] = max(round(wantedCumulBin) - 1, 0)
            eqHist[q[i]] += hist[i]

    print(
        "Mapping function: ",
        q,
        "\nInitial histogram: ",
        hist,
        "\nEqualized histogram: ",
        eqHist,
    )

    # If the cummulative hist starts with 0-s then ignore them.
    return q, eqHist


def change_ranges(q, ranges):
    """
    Changes the bin ranges using the mapping function q, such that the histogram is equalized.

    Args:
        q: The mapping function.
        ranges: The initial histogram bin ranges.
    Returns:
        new_ranges: The equalized new bin ranges.
    """
    # Get unique bins
    uniquebins = np.unique(q)

    ranges1 = dict.fromkeys(uniquebins, None)
    ranges2 = np.array([ranges[:-1], ranges[1:]])

    for i, u in enumerate(uniquebins.tolist()):
        repeats = np.where(q == u)
        mini = repeats[0].min()
        maxi = repeats[0].max()
        ranges1[u] = [ranges2[0, mini], ranges2[1, maxi]]
    list_ranges = np.array(list(ranges1.values()))

    # Merge the 2 rows:
    new_ranges = list(list_ranges[:, 0])
    new_ranges.append(list_ranges[-1, 1])

    print("Old bin ranges:", ranges, "\nNew bin ranges: ", new_ranges)
    return new_ranges


def cls_equalized_ranges(hist, ranges):
    """
    Given a histogram and its bin ranges is returns an equalized histogram together with its new ranges.

    Args:
        hist: The original unequalized histogram.
        ranges: The initial histogram bin ranges.
    Returns:

        new_ranges: The equalized new bin ranges.
    """
    # First remove trailing zero bins
    hist, ranges = removeStartZeros(hist, ranges)

    # Define the equalization function
    q, histeq = doEq(hist, nbins=ranges.size - 1)

    # Define the equalized bin ranges
    new_ranges = np.array(change_ranges(q, ranges))

    return new_ranges, histeq[histeq != 0]


if __name__ == "__main__":

    # Create a dummy array
    arr = np.random.normal(loc=1.0, scale=3, size=100)

    # Computer the histogram of the array
    hist, ranges = np.histogram(arr, bins=16)

    fig, (ax) = plt.subplots(1, 2)
    ax[0].bar(np.arange(0, 16), hist, label="Original histogram", color="red")
    ax[0].legend()

    # Equalize the histogram and return the mapping function
    new_ranges, new_hist = cls_equalized_ranges(hist, ranges)

    ax[1].bar(
        np.arange(new_ranges.size - 1),
        new_hist,
        label="Equalized histogram",
        color="green",
    )
    ax[1].legend()
    plt.show()
