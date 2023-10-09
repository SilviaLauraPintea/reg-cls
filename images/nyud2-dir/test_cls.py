import numpy as np
import os
import logging
import argparse
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import random
import loaddata
from models import modules, net, resnet
from util import Evaluator, accuracy, AverageMeter, cls_num_from_filename


def set_seed(seed):
    """Adjust the learning rate per epoch."""
    seed = int(seed)
    # make everything deterministic -> seed setup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_model",
        type=str,
        default="checkpoint/nyud2_lr_5e-05_bs_16_seed0_lambda1.0_cls_100_equ/checkpoint_best_mse_lambda1.0_100_.pth.tar",
        help="evaluation model path",
    )
    parser.add_argument("--data_dir", type=str, default="./data", help="data directory")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--fds", action="store_true", default=False, help="whether to enable FDS"
    )

    # Added parse option with cls --------------------------------
    parser.add_argument("--test_set", type=str, default="test")
    parser.add_argument(
        "--losstype",
        type=str,
        default="mse",
        choices=["msecls", "msecls-bal", "mse"],
        help="loss type",
    )
    parser.add_argument(
        "--cls_num", type=int, default=0, help="Number classes for classification"
    )
    parser.add_argument(
        "--cls_equalize",
        action="store_true",
        help="Equalize the classes, per sample",
    )
    # Added parse option with cls --------------------------------

    args = parser.parse_args()
    set_seed(args.seed)

    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # First define the data, then define the model to add the actual classes
    test_loader = loaddata.getTestingData(args, cls_num=0, use_set="all", batch_size=1)

    # Define the model with the updated number of classes
    assert os.path.isfile(
        args.eval_model
    ), f"No checkpoint found at '{args.eval_model}'"
    nclasses = cls_num_from_filename(args.eval_model)
    model = define_model(args, args.losstype.startswith("msecls"), nclasses)

    # If the number of classes is different this may give an error
    model_state = torch.load(args.eval_model)
    logging.info(f"Loading checkpoint from {args.eval_model}")
    model.module.load_state_dict(model_state["state_dict"], strict=False)
    logging.info("Loaded successfully!")

    # Run the test loop
    test(test_loader, model, losstype=args.losstype, args=args, isval=False)


def test(test_loader, model, losstype, args, isval):
    """Test the model on the test data.
    Args:
        - test_loader: the test dataloader
        - model: the trained model
        - losstype: 'mse' or 'msecls' (to know how many outputs if gives)
        - args: the parse arguments
        - isval: if the model is evaluated on the validation or on the test set.
    """

    model.eval()
    logging.info("Starting testing...")

    evaluator = Evaluator()
    losses = AverageMeter()

    with torch.no_grad():

        # Loop over the val/test samples --------------------------------------------
        for i, sample_batched in enumerate(test_loader):
            image, depth, depth_cls, mask = (
                sample_batched["image"].cuda(),
                sample_batched["depth"].cuda(),
                sample_batched["depth_cls"].cuda(),
                sample_batched["mask"].cuda(),
            )

            # Forward pass
            output_reg, output_cls = model(image)

            # Compute the loss: the predicted depth is smaller
            output_reg = nn.functional.interpolate(
                output_reg,
                size=[depth.size(2), depth.size(3)],
                mode="bilinear",
                align_corners=True,
            )

            # Hack to balanced the validation data (validation is not balanced because is a subset of train)
            if isval is True:
                depth = depth.view(depth_cls.shape)[depth_cls != -1]
                output_reg = output_reg.view(depth_cls.shape)[depth_cls != -1]

            # The balanced test loss
            loss = torch.mean(((output_reg[mask] - depth[mask]) ** 2))
            losses.update(loss.item(), image.size(0))

            if args.test_set.endswith("test"):
                evaluator(output_reg[mask], depth[mask])
        # --------------------------------------------------------------

    error = 0
    metric_dict = {}
    if not isval:
        logging.info("[TEST] RMSE loss {lavg:.4f}\t".format(lavg=math.sqrt(losses.avg)))
        logging.info("[TEST] Finished. Start printing statistics below...")
        metric_dict = evaluator.evaluate_shot()
        error = metric_dict["overall"]["RMSE"]
    else:
        logging.info("[VAL] RMSE loss {lavg:.4f}\t".format(lavg=math.sqrt(losses.avg)))
        error = math.sqrt(losses.avg)
        metric_dict["overall"] = {"RMSE": error}
    return error, metric_dict


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
        None,
        Encoder,
        num_features=2048,
        block_channel=[256, 512, 1024, 2048],
        addcls=addcls,
        cls_num=nclasses,
    )

    model = torch.nn.DataParallel(model).cuda()
    return model


if __name__ == "__main__":
    main()
