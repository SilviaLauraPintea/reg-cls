import argparse
import torch
import numpy as np
import random
import os

parser = argparse.ArgumentParser(description="")
parser.add_argument("--seed", type=int, default=0, help="seed")

# training/optimization related
parser.add_argument("--print_freq", type=int, default=1000, help="logging frequency")
parser.add_argument("--num_workers", default=2, type=int, help="for the data loader")
parser.add_argument(
    "--epochs", default=10, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--lr", "--learning-rate", default=0.001, type=float, help="initial learning rate"
)
parser.add_argument(
    "--weight_decay",
    "--wd",
    default=1e-4,
    type=float,
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--batch_size", default=4, type=int, help="batch size number"
)  # 1 GPU - 8
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument("--store_root", type=str, default="checkpoint")
parser.add_argument("--store_name", type=str, default="nyud2")
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--data_dir", type=str, default="./data", help="data directory")
parser.add_argument(
    "--resume", action="store_true", default=False, help="whether to resume training"
)

# imbalanced related
# LDS
parser.add_argument(
    "--lds", action="store_true", default=False, help="whether to enable LDS"
)
parser.add_argument(
    "--lds_kernel",
    type=str,
    default="gaussian",
    choices=["gaussian", "triang", "laplace"],
    help="LDS kernel type",
)
parser.add_argument(
    "--lds_ks", type=int, default=5, help="LDS kernel size: should be odd number"
)
parser.add_argument(
    "--lds_sigma", type=float, default=2, help="LDS gaussian/laplace kernel sigma"
)
# FDS
parser.add_argument(
    "--fds", action="store_true", default=False, help="whether to enable FDS"
)
parser.add_argument(
    "--fds_kernel",
    type=str,
    default="gaussian",
    choices=["gaussian", "triang", "laplace"],
    help="FDS kernel type",
)
parser.add_argument(
    "--fds_ks", type=int, default=5, help="FDS kernel size: should be odd number"
)
parser.add_argument(
    "--fds_sigma", type=float, default=2, help="FDS gaussian/laplace kernel sigma"
)
parser.add_argument(
    "--start_update", type=int, default=0, help="which epoch to start FDS updating"
)
parser.add_argument(
    "--start_smooth",
    type=int,
    default=1,
    help="which epoch to start using FDS to smooth features",
)
parser.add_argument(
    "--bucket_num", type=int, default=100, help="maximum bucket considered for FDS"
)
parser.add_argument(
    "--bucket_start",
    type=int,
    default=7,
    help="minimum(starting) bucket for FDS, 7 for NYUDv2",
)
parser.add_argument("--fds_mmt", type=float, default=0.9, help="FDS momentum")


# ---CLS options-----------------
parser.add_argument(
    "--losstype",
    type=str,
    default="msecls",
    choices=["msecls", "mse"],
    help="Loss type",
)
parser.add_argument(
    "--cls_num", type=int, default=100, help="Number classes for classification"
)
parser.add_argument(
    "--cls_equalize",
    action="store_true",
    help="Equalize the classes, per sample",
)
# ---END CLS options-----------------
parser.add_argument("--test_mask", type=str, default="./data/test_balanced_mask.npy")


# re-weighting: SQRT_INV / INV
parser.add_argument(
    "--reweight",
    type=str,
    default="none",
    choices=["none", "inverse", "sqrt_inv"],
    help="cost-sensitive reweighting scheme",
)
# two-stage training: RRT
parser.add_argument(
    "--retrain_fc",
    action="store_true",
    default=False,
    help="whether to retrain last regression layer (regressor)",
)
parser.add_argument(
    "--pretrained",
    type=str,
    default="",
    help="pretrained checkpoint file path to load backbone weights for RRT",
)


def set_seed(seed):
    seed = int(seed)
    # make everything deterministic -> seed setup
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def args_store(args):
    if not args.lds and args.reweight != "none":
        args.store_name += f"_{args.reweight}"
    if args.lds:
        args.store_name += f"_lds_{args.lds_kernel[:3]}_{args.lds_ks}"
        if args.lds_kernel in ["gaussian", "laplace"]:
            args.store_name += f"_{args.lds_sigma}"
    if args.fds:
        args.store_name += f"_fds_{args.fds_kernel[:3]}_{args.fds_ks}"
        if args.fds_kernel in ["gaussian", "laplace"]:
            args.store_name += f"_{args.fds_sigma}"
        args.store_name += f"_{args.start_update}_{args.start_smooth}_{args.fds_mmt}"
    if args.retrain_fc:
        args.store_name += f"_retrain_fc"
    args.store_name += f"_lr_{args.lr}_bs_{args.batch_size}"

    args.store_name += "_seed" + str(args.seed) + "_" + args.opt
    if args.losstype.startswith("msecls"):
        args.cls_num_orig = args.cls_num
        args.store_name += (
            "__cls"
            + str(args.cls_num_orig)
            + ("_equ" if args.cls_equalize is True else "_noequ")
        )

    args.store_dir = os.path.join(args.store_root, args.store_name)

    if not args.resume:
        if os.path.exists(args.store_dir):
            """
            if query_yes_no('overwrite previous folder: {} ?'.format(args.store_dir)):
                shutil.rmtree(args.store_dir)
                print(args.store_dir + ' removed.')
            else:
                raise RuntimeError('Output folder {} already exists'.format(args.store_dir))
            """
            print("Output folder {} already exists".format(args.store_dir))
        else:
            print(f"===> Creating folder: {args.store_dir}")
            os.makedirs(args.store_dir)


args = parser.parse_args()
set_seed(args.seed)
args_store(args)
