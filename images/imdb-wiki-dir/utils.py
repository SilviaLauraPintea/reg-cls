import os
import shutil
import torch
import logging
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import triang


def check_rootfolders(root_model, store_name=""):
    """
    Create log and model folder
    """

    if len(store_name) < 1:  # assumes that the path is gives with the file
        splits = root_model.split("/")
        assert len(splits) > 1
        root_model = splits[0]
        for i in range(1, len(splits) - 1):
            root_model = os.path.join(root_model, splits[i])

    folders_util = [root_model, os.path.join(root_model, store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print("creating folder " + folder)
            os.mkdir(folder)


def cls_num_from_filename(filename):
    """Extracts the number of classes from the checkpoint name (useful for equalized classes)."""

    splits = str(filename).split("_")

    assert len(splits) >= 2, "There should be more than 2 splis in " + filename
    return int(splits[-2])


def plot_data_hist(bins, hist, title, xaxis, yaxis, outfile, width=1.1):
    """Plots class histograms."""

    import matplotlib.pyplot as plt

    fig, (ax) = plt.subplots(
        1,
    )
    fig.tight_layout(pad=5.0)
    ax.bar(bins, hist, alpha=0.7, width=width)
    ax.set_title(title)
    ax.set_ylabel(yaxis)
    ax.set_xlabel(xaxis)

    check_rootfolders(outfile, store_name="")
    plt.savefig(
        os.path.join(outfile),
        format="pdf",
        pad_inches=-2,
        transparent=False,
        dpi=300,
    )
    plt.close()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = min(max(topk), output.shape[1])
    batch_norm = 100.0 / float(target.size(0))

    val_inds = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = val_inds.indices.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        a_topk = correct_k * batch_norm
        res.append(a_topk)
    return res


class AverageMeter(object):
    """Updates the metrics"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1.0):
        self.val = float(val)
        self.sum += val * n
        self.count += n
        self.avg = float(self.sum) / float(self.count)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def query_yes_no(question):
    """Ask a yes/no question via input() and return their answer."""
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [Y/n] "

    while True:
        print(question + prompt, end=":")
        choice = input().lower()
        if choice == "":
            return valid["y"]
        elif choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def prepare_folders(args):
    folders_util = [args.store_root, os.path.join(args.store_root, args.store_name)]
    if (
        os.path.exists(folders_util[-1])
        and not args.resume
        and not args.pretrained
        and not args.evaluate
    ):
        """
        if query_yes_no("overwrite previous folder: {} ?".format(folders_util[-1])):
            shutil.rmtree(folders_util[-1])
            print(folders_util[-1] + " removed.")
        else:
            raise RuntimeError(
                "Output folder {} already exists".format(folders_util[-1])
            )
        """
        print("Output folder {} already exists".format(folders_util[-1]))

    for folder in folders_util:
        if not os.path.exists(folder):
            print(f"===> Creating folder: {folder}")
            os.mkdir(folder)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.0
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(args, state, is_best, prefix=""):
    filename = f"{args.store_root}/{args.store_name}/{prefix}ckpt.pth.tar"
    torch.save(state, filename)
    if is_best:
        logging.info("===> Saving current best checkpoint...")
        shutil.copyfile(filename, filename.replace("pth.tar", "best.pth.tar"))


def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.0).any():
        valid = v1 != 0.0
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[
            valid
        ]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ["gaussian", "triang", "laplace"]
    half_ks = (ks - 1) // 2
    if kernel == "gaussian":
        base_kernel = [0.0] * half_ks + [1.0] + [0.0] * half_ks
        kernel_window = gaussian_filter1d(base_kernel, sigma=sigma) / max(
            gaussian_filter1d(base_kernel, sigma=sigma)
        )
    elif kernel == "triang":
        kernel_window = triang(ks)
    else:
        laplace = lambda x: np.exp(-abs(x) / sigma) / (2.0 * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1))
        )

    return kernel_window
