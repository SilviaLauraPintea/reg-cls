import torch
import argparse
import pickle
import random
import numpy as np
from train_eval import run_train

parser = argparse.ArgumentParser()
parser.add_argument("--random-seed", type=int, default=0, help="Random seed")

parser.add_argument("--name", type=str, default="uniform0", help="Run name.")
parser.add_argument("--log-file", type=str, default="log.pkl", help="Run name.")
parser.add_argument(
    "--path-train",
    type=str,
    default="",
    help="Dataset location (pkl file) for training data.",
)
parser.add_argument(
    "--path-val",
    type=str,
    default="",
    help="Dataset location (pkl file) for validation data.",
)
parser.add_argument(
    "--path-test",
    type=str,
    default="",
    help="Dataset location (pkl file) for test data.",
)
parser.add_argument(
    "--optim",
    type=str,
    default="adam",
    choices=["adam"],
    help="Optimizer to be used",
)
parser.add_argument(
    "--num-segment",
    type=int,
    default=100,
    help="The number of segments/classes to use",
)

parser.add_argument(
    "--equalize_segments",
    action="store_true",
    help="If the segments/classes should be equalized",
)

parser.add_argument(
    "--bn", default=False, action="store_true", help="If we use bn or not"
)


parser.add_argument(
    "--num_layer",
    type=int,
    default=1,
    help="The number of layers of the MLP",
)

parser.add_argument(
    "--num_comp",
    type=int,
    default=15,  # +1 for bias
    help="Last hidden layer size for the MLP",
)

parser.add_argument(
    "--root_model", type=str, default="./checkpoint", help="Path to save the models."
)
parser.add_argument(
    "--store_name", type=str, default="", help="Name for the current run."
)
parser.add_argument(
    "--resume",
    default=False,
    action="store_true",
    help="resume from the latest checkpoint with the same name (default: none)",
)

parser.add_argument(
    "--val",
    default=True,
    action="store_true",
    help="Evaluate on the validation during training",
)
parser.add_argument(
    "--debug", default=False, action="store_true", help="Debugging enabled"
)

# MLP ---------------------------------------------------------
parser.add_argument(
    "--hidden_list",
    nargs="+",
    default=[6],
    help="List of hidden layer sizes in the MLP",
)

# Training ----------------------------------------------------
parser.add_argument("--lr", type=float, default=0.01, help="The initial learning rate")
parser.add_argument("--momentum", type=float, default=0.9, help="The momentum")

parser.add_argument("--weight_decay", type=float, default=1e-3, help="The weight decay")
parser.add_argument(
    "--lrdecay", type=float, default=0.1, help="The learning rate decay"
)

parser.add_argument(
    "--max_epochs",
    type=int,
    default=int(150),
    help="The training steps. Losses seem to flatten at 80 epochs; 150 to be sure.",
)
parser.add_argument("--batch_size", type=int, default=256, help="The batch size")
parser.add_argument("--test_batch_size", type=int, default=256, help="The batch size")
parser.add_argument("--workers", type=int, default=1, help="The number of workers")
parser.add_argument(
    "--erlambda",
    type=float,
    default=0,
    help="The lambda weight between reg and cls losses",
)

args = parser.parse_args()

# Use CUDA
args.use_cuda = torch.cuda.is_available()


def set_seed(seed):
    """
    Changes the random seed to the given seed everywhere.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Set the seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(args)
    args.store_name += (
        args.name + "_cls" + str(args.num_segment) + "_seed" + str(args.random_seed)
    )
    set_seed(args.random_seed)
    args.cls_lr = args.lr

    wandb = None
    if args.debug:
        import wandb

        wandb.init(project="1d", entity="silvas")  # change this to own
        wandb.run.name = args.store_name
    run_train(args, wandb)

    if args.debug:
        wandb.finish()
