import numpy as np
import random
import argparse
import time
import os
import shutil
import logging
import torch
import torch.backends.cudnn as cudnn
import loaddata
from tqdm import tqdm
from models import modules, net, resnet
from util import (
    query_yes_no,
    AverageMeter,
    accuracy,
    cls_num_from_filename,
    save_checkpoint,
)
from test_cls import test
from tensorboardX import SummaryWriter
import torch.nn as nn

parser = argparse.ArgumentParser(description="")
# ---added options-----------------
parser.add_argument(
    "--erlambda",
    type=float,
    default=1.0,
    help="The MSE weight: it should be anywhere between 1e-3 and 1e+3",
)
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument(
    "--losstype",
    type=str,
    default="msecls",
    choices=["msecls", "mse"],
    help="Loss type: use 'mse' for regression only, use 'msecls' for regression+classification.",
)
parser.add_argument(
    "--cls_num", type=int, default=0, help="Number classes for the classification head."
)

parser.add_argument(
    "--balance_data",
    action="store_true",
    help="Balance all training data for both regression and classification.",
)

parser.add_argument(
    "--cls_equalize",
    action="store_true",
    help="Equalize the samples per class for the classification head.",
)
parser.add_argument(
    "--test_set",
    type=str,
    default="test",
    help="Which set to evaluate on at the end of the training: 'test' or 'val' (For parameter search).",
)
parser.add_argument(
    "--batch_acu",
    default=1,
    type=int,
    help="The code uses batch accumulation steps to mimic larger batch sizes.",
)  # 1 GPU - 8
# ---added options-----------------

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
    "--lr", "--learning-rate", default=0.0001, type=float, help="initial learning rate"
)
parser.add_argument(
    "--weight_decay",
    "--wd",
    default=1e-4,
    type=float,
    help="weight decay (default: 1e-4)",
)
parser.add_argument(
    "--batch_size", default=32, type=int, help="batch size number"
)  # 1 GPU - 8
parser.add_argument("--opt", type=str, default="adam")
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


def define_model(args, addcls, nclasses):
    """Defines the model.
    Args:
        - args: the input parse arguments
        - addcls: bool, true if also a classification head should be used
        - nclasses: the number of classes to use in the classification head.
    """
    original_model = resnet.resnet50(pretrained=True)
    Encoder = modules.E_resnet(original_model)
    model = net.model(
        args,
        Encoder,
        num_features=2048,
        block_channel=[256, 512, 1024, 2048],
        addcls=addcls,
        cls_num=nclasses,
    )
    print("> Model ", model)
    model = torch.nn.DataParallel(model).cuda()
    return model


def args_store(args):
    """Pre-process the arguments to make them more complete."""
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

    args.store_name += (
        "_seed" + str(args.seed) + "_lambda" + str(args.erlambda) + "_" + args.losstype
    )
    if args.losstype.startswith("msecls") or args.cls_num > 0:
        args.store_name += (
            "_cls_"
            + str(args.cls_num)
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


def start_train(args):
    """Defines the model, optimizers, dataloaders, and starts the training."""
    # Define args storing name and path

    # Adjust the learning rate with the lambda factor.
    args_store(args)

    # Logging and stuff
    logging.root.handlers = []
    log_file = os.path.join(args.store_dir, "training_log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    # First define the data so the number of classes may change
    if args.test_set.endswith("test"):
        # Use 4/5 of training data for train
        train_loader = loaddata.getTrainingData(
            args, use_set="train", cls_num=args.cls_num, batch_size=args.batch_size
        )
        train_fds_loader = loaddata.getTrainingFDSData(
            args, use_set="train", cls_num=args.cls_num, batch_size=args.batch_size
        )
        # Use 1/5 for val
        val_loader = loaddata.getTrainingData(
            args, cls_num=args.cls_num, use_set="val", batch_size=1
        )
        # Use the test set
        test_loader = loaddata.getTestingData(
            args,
            cls_num=0,
            use_set="all",
            batch_size=1,
        )  # why is the batch size 1?
    else:
        # Use 4/5 of training data for train
        train_loader = loaddata.getTrainingData(
            args, cls_num=args.cls_num, use_set="train", batch_size=args.batch_size
        )
        train_fds_loader = loaddata.getTrainingFDSData(
            args, cls_num=args.cls_num, use_set="train", batch_size=args.batch_size
        )
        # Use 1/5 for val
        val_loader = loaddata.getTrainingData(
            args, cls_num=args.cls_num, use_set="val", batch_size=1
        )
        # Report last numbers of val (1/5 of training data)
        test_loader = loaddata.getTrainingData(
            args, cls_num=args.cls_num, use_set="val", batch_size=1
        )

    # Then define the model
    model = define_model(
        args, args.losstype.startswith("msecls"), train_loader.dataset.nclasses
    )
    logging.info(args)

    # If resume read the classes from the model
    if args.resume:
        nclasses = args.cls_num
        if args.losstype.startswith("msecls"):
            nclasses = cls_num_from_filename(args.model_name)
        # Then redefine the model cause the classes may have changed
        model = define_model(args, args.losstype.startswith("msecls"), nclasses)

        # Read all the rest of the suff
        model_state = torch.load(os.path.join(args.store_dir, args.model_name))
        logging.info(
            f"Loading checkpoint from {os.path.join(args.store_dir, args.model_name)}"
            f" (Epoch [{model_state['epoch']}], RMSE: {model_state['error']:.3f})"
        )
        model.module.load_state_dict(model_state["state_dict"])
        args.start_epoch = model_state["epoch"] + 1
        epoch_best = model_state["epoch"]
        error_best = model_state["error"]
        metric_dict_best = model_state["metric"]

    if args.retrain_fc:
        assert os.path.isfile(
            args.pretrained
        ), f"No checkpoint found at '{args.pretrained}'"
        model_state = torch.load(args.pretrained, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in model_state["state_dict"].items():
            if "R" not in k and "Rcls" not in k:
                new_state_dict[k] = v
        model.module.load_state_dict(new_state_dict, strict=False)
        logging.info(
            f"===> Pretrained weights found in total: [{len(list(new_state_dict.keys()))}]"
        )
        logging.info(f"===> Pre-trained model loaded: {args.pretrained}")
        for name, param in model.named_parameters():
            if "R" not in k and "Rcls" not in k:
                param.requires_grad = False
        logging.info(
            f"Only optimize parameters: {[n for n, p in model.named_parameters() if p.requires_grad]}"
        )

    cudnn.benchmark = True
    if not args.retrain_fc:
        if args.opt.startswith("adam"):
            optimizer = torch.optim.Adam(
                model.module.parameters(), args.lr, weight_decay=args.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                model.module.parameters(),
                args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
    else:
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        if args.opt.startswith("adam"):
            optimizer = torch.optim.Adam(
                parameters, args.lr, weight_decay=args.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                parameters, args.lr, momentum=0.9, weight_decay=args.weight_decay
            )

    # Call the training loop with everything
    error_best = {"val": 1e5}
    metric_dict_best = {"val": {}}
    epoch_best = {"val": -1}
    train_loop(
        error_best,
        metric_dict_best,
        epoch_best,
        args,
        optimizer,
        train_loader,
        train_fds_loader,
        test_loader,
        val_loader,
        model,
        logging,
    )


def train_loop(
    error_best,
    metric_dict_best,
    epoch_best,
    args,
    optimizer,
    train_loader,
    train_fds_loader,
    test_loader,
    val_loader,
    model,
    logging,
):
    """The main training loop over epoch that evaluates every epoch on the validation set.
    Args:
        - error_best: the best error on val across epochs.
        - metric_dict_best: the metric of the best model across epochs
        - epoch_best: the epoch of the best model.
        - args: the input parse arguments
        - optimizer: the optimizer to use.
        - train_loader: the training dataloader
        - train_fds_loader: the training dataloader for fds
        - test_loader: the test dataloader
        - val_loader: the val dataloader
        - model: the model to be trained
        - logging: the printing routine
    """
    writer = SummaryWriter(args.store_dir)

    # This is the main loop over epochs
    checkpoint_best = (
        "checkpoint_best_"
        + args.losstype
        + "_lambda"
        + str(args.erlambda)
        + "_"
        + str(args.cls_num)
        + "_.pth.tar"
    )

    # Loop over epochs ----------------------------------------------
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # Train the model on the dataset
        train(
            train_loader,
            train_fds_loader,
            model,
            optimizer,
            epoch,
            writer,
            losstype=args.losstype,
            erlambda=args.erlambda,
            cls_num=args.cls_num,
            print_freq=args.print_freq,
        )

        # Evaluate the model on the validation set
        error_val, metric_dict_val = test(
            val_loader,
            model,
            losstype=args.losstype,
            args=args,
            isval=True,
        )

        # Keep the best model on the validation set.
        if error_val < error_best["val"]:
            error_best["val"] = error_val
            metric_dict_best["val"] = metric_dict_val
            epoch_best["val"] = epoch
            logging.info(
                f"Saving checkpoint to {os.path.join(args.store_dir, checkpoint_best)}..."
            )
            save_checkpoint(
                model.module.state_dict(),
                epoch,
                error_val,
                metric_dict_val,
                args.store_dir,
                "val_" + checkpoint_best,
                cls_num=args.cls_num,
            )
        save_checkpoint(
            model.module.state_dict(),
            epoch,
            error_val,
            metric_dict_val,
            args.store_dir,
            cls_num=args.cls_num,
        )

    # ----------------------------------------------
    save_checkpoint(
        state_dict=model.module.state_dict(),
        epoch=epoch,
        error=error_val,
        metric_dict=metric_dict_val,
        store_dir=args.store_dir,
        cls_num=args.cls_num,
    )
    logging.info(
        "Best val epoch: {epoch_best}; Val-RMSE {error_best:.4f}\t".format(
            epoch_best=epoch_best["val"], error_best=error_best["val"]
        )
    )

    # Start the testing on the best saved model
    logging.info("***** TEST RESULTS *****")
    checkpoint = torch.load(os.path.join(args.store_dir, "val_" + checkpoint_best))
    model.module.load_state_dict(checkpoint["state_dict"])
    logging.info(
        f"Loaded best model, epoch {checkpoint['epoch']}, best val loss {checkpoint['error']:.4f}"
    )
    error, metric_dict_best = test(
        test_loader,
        model,
        losstype=args.losstype,
        args=args,
        isval=(not args.test_set.endswith("test")),
    )

    logging.info("***** FINAL RESULTS *****")
    if args.test_set.endswith("test"):
        for shot in ["Overall", "Many", "Medium", "Few"]:
            logging.info(
                f" * {shot}: RMSE {metric_dict_best[shot.lower()]['RMSE']:.3f}\t"
                f"ABS_REL {metric_dict_best[shot.lower()]['ABS_REL']:.3f}\t"
                f"LG10 {metric_dict_best[shot.lower()]['LG10']:.3f}\t"
                f"MAE {metric_dict_best[shot.lower()]['MAE']:.3f}\t"
                f"DELTA1 {metric_dict_best[shot.lower()]['DELTA1']:.3f}\t"
                f"DELTA2 {metric_dict_best[shot.lower()]['DELTA2']:.3f}\t"
                f"DELTA3 {metric_dict_best[shot.lower()]['DELTA3']:.3f}\t"
                f"NUM {metric_dict_best[shot.lower()]['NUM']}"
            )
    writer.close()


def train(
    train_loader,
    train_fds_loader,
    model,
    optimizer,
    epoch,
    writer,
    losstype,
    cls_num,
    erlambda,
    print_freq,
):
    """Performs just a single pass over the training data.
    Args:
        - train_loader: the training dataloader
        - train_fds_loader: the training dataloader for fds
        - model: the model to be trained
        - optimizer: the optimizer to use.
        - writer: the printing routine
        - losstype: the loss type to be optimized (mse or msecls)
        - cls_num: the number of classes in the classification head
        - erlambds: the weight between the regression head and classification head
        - print_freq: the printing frequency
    """
    batch_time = AverageMeter()
    losses = AverageMeter()

    model.train()

    # Define the classification loss
    cls_loss_criterion = None
    cls_string = ""
    if losstype.startswith("msecls") and args.cls_num > 0:
        cls_loss_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        accuracies = AverageMeter()
        losses_cls = AverageMeter()

    end = time.time()
    # Loop over the training data  ----------------------------------------------
    for i, sample_batched in enumerate(train_loader):
        image, depth, depth_cls, weight = (
            sample_batched["image"].cuda(),
            sample_batched["depth"].cuda(),
            sample_batched["depth_cls"].cuda(),
            sample_batched["weight"].cuda(),
        )

        if args.fds:
            output_reg, output_cls, feature = model(image, depth, epoch)
        else:
            output_reg, output_cls = model(image, depth, epoch)

        # Hack to balanced the regression as well, if needed
        if args.balance_data:
            depth = nn.functional.interpolate(
                depth,
                size=[depth.size(2) * 2, depth.size(3) * 2],
                mode="bilinear",
                align_corners=True,
            )
            output_reg = nn.functional.interpolate(
                output_reg,
                size=[output_reg.size(2) * 2, output_reg.size(3) * 2],
                mode="bilinear",
                align_corners=True,
            )
            weight = 1.0
            depth = depth.view(depth_cls.shape)[depth_cls != -1]
            output_reg = output_reg.view(depth_cls.shape)[depth_cls != -1]

        # Compute the reg loss
        loss = torch.mean(((output_reg - depth) ** 2) * weight)
        losses.update(loss.item(), max(1, image.size(0)))

        # Adds a classification loss
        if cls_loss_criterion is not None:
            depth_cls = depth_cls.reshape(
                -1,
            ).contiguous()

            # loss needs to apply everywhere
            output_cls = output_cls[depth_cls != -1]
            depth_cls = depth_cls[depth_cls != -1]

            loss_cls = cls_loss_criterion(input=output_cls, target=depth_cls)
            losses_cls.update(loss_cls.item(), image.size(0))
            batch_accuracy = accuracy(output_cls, depth_cls)

            accuracies.update(batch_accuracy[0], image.size(0))
            cls_string = "Acc {acc.val:.4f} % ({acc.avg:.4f} %)\t Loss cls {lossc.val:.4f} ({lossc.avg:.4f})\t".format(
                acc=accuracies, lossc=losses_cls
            )

            # Add the cls-loss to the reg-loss
            if erlambda > 1:
                totalloss = loss * erlambda + loss_cls
            else:
                totalloss = loss + loss_cls / erlambda
        else:
            totalloss = loss

        # Normalize the loss, if gradients are accumulated over batches
        totalloss = totalloss / float(args.batch_acu)
        totalloss.backward()

        if ((i + 1) % args.batch_acu == 0) or (i + 1 == len(train_loader)):
            # Update Optimizer
            optimizer.step()
            # Zero grads at the end
            optimizer.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        writer.add_scalar("data/loss", loss.item(), i + epoch * len(train_loader))
        if i % print_freq == 0:
            logging.info(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=losses
                )
                + cls_string
            )
    # ----------------------------------------------------------

    if args.fds and epoch >= args.start_update:
        logging.info(
            f"Starting Creating Epoch [{epoch}] features of subsampled training data..."
        )
        encodings, depths = [], []
        with torch.no_grad():
            for i, sample_batched in enumerate(tqdm(train_fds_loader)):
                image, depth = (
                    sample_batched["image"].cuda(),
                    sample_batched["depth"].cuda(),
                )
                _, _, feature = model(image, depth, epoch)
                encodings.append(feature.data.cpu())
                depths.append(depth.data.cpu())
        encodings, depths = torch.cat(encodings, 0), torch.cat(depths, 0)
        logging.info(
            f"Created Epoch [{epoch}] features of subsampled training data (size: {encodings.size(0)})!"
        )
        model.module.R.FDS.update_last_epoch_stats(epoch)
        model.module.R.FDS.update_running_stats(encodings, depths, epoch)


def set_seed(seed):
    """Define the random seed for all the libraries"""
    seed = int(seed)
    # make everything deterministic -> seed setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def adjust_learning_rate(optimizer, epoch):
    """Adjust the learning rate per epoch."""
    lr = args.lr * (0.1 ** (epoch // 5))

    print("Current lr is .. ", lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    global args
    args = parser.parse_args()
    set_seed(args.seed)
    start_train(args)
