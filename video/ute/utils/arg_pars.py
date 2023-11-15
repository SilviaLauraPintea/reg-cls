#!/usr/bin/env python

"""Hyper-parameters and logging set up

opt: include all hyper-parameters
logger: unified logger for the project
"""

import argparse
from os.path import join

parser = argparse.ArgumentParser()


# ---Added options------------------------------------------------
parser.add_argument(
    "--erlambda", type=float, default=1, help="The weight for the regression loss."
)
parser.add_argument(
    "--cls_num", type=int, default=100, help="Number classes for classification."
)
parser.add_argument(
    "--cls_equalize",
    action="store_true",
    help="Equalize the classes, per sample.",
)
parser.add_argument(
    "--loss", default="paper", help="loss to use: paper - mse(sum), l1, mse, rmse"
)
parser.add_argument("--split", default="s1", help="datasplit to use")
parser.add_argument("--testset_name", default="val", help="Evaluation dataset to use.")
parser.add_argument(
    "--reltime",
    action="store_true",
    help="If video time is relative or absolute",
)
# ---END added options--------------------------------------------

parser.add_argument("--optim", default="adam", help="optimizer to use: adam or sgd")
parser.add_argument(
    "--subaction", default="all", help="Subaction to filter the videos on."
)
parser.add_argument("--feature_dim", default=64, help="feature dimensionality")
parser.add_argument("--embed_dim", default=20, help="mlp embedding dimensionality")
parser.add_argument("--ext", default=".txt", help="extension of the feature files")
parser.add_argument(
    "--dataset_root", default="./data_utils/data", help="The path to the dataset"
)


# Hyperparams parameters for embeddings
parser.add_argument("--device", default="cpu", help="cpu | cuda")
parser.add_argument("--seed", default=0, help="Seed for random algorithms, everywhere")
parser.add_argument("--lr", default=1e-3, type=float, help="Initial learning rate")
parser.add_argument(
    "--lr_adj",
    default=True,
    type=bool,
    help="The lr will be multiplied by 0.1 in the middle",
)
parser.add_argument("--momentum", default=0.9, help="Momentum term")
parser.add_argument(
    "--weight_decay",
    default=1.0e-4,
    help="Regularization constant for l_2 regularizer of W",
)
parser.add_argument(
    "--batch_size", default=40, help="Batch size for training embedding (default: 40)"
)
parser.add_argument(
    "--num_workers", default=4, help="Number of threads for dataloading"
)
parser.add_argument(
    "--epochs", default=40, type=int, help="Number of epochs for training embedding"
)

# Local saving and checkpointing
parser.add_argument(
    "--save_model", default=True, type=bool, help="Save embedding model after training"
)
parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume model for embeddings, if true the number of epochs should be updated",
)
parser.add_argument(
    "--test_model",
    action="store_true",
    help="If the model should be loaded and evaluated.",
)
parser.add_argument(
    "--model_name",
    default="",
    help="The name of the checkpoint file to store the model.",
)
parser.add_argument(
    "--loaded_model_name",
    default="",
    help="The name of the checkpoint file to load the model.",
)
parser.add_argument(
    "--output_dir", default="", help="The name of the output directory."
)


opt = parser.parse_args()
opt.cls_num_orig = opt.cls_num
