#!/usr/bin/env python
import os

"""Implementation of training and testing functions for embedding."""

__all__ = ["training", "load_model"]
__author__ = "Anna Kukleva"
__date__ = "August 2018"

import torch
import torch.backends.cudnn as cudnn
from os.path import join
import time
import numpy as np
import random
import math

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger
from ute.utils.util_functions import (
    Averaging,
    adjust_lr,
    dir_check,
    accuracy_fn,
    cls_from_filename,
    join_data,
)


def training(train_loader, epochs, save, **kwargs):
    """Training pipeline for embedding.

    Args:
        train_loader: iterator within dataset
        epochs: how much training epochs to perform
        save: if the model should be saved
        model: the mlp model
        loss: the loss function
        optimizer: the optimizer used
        name: the model name
    Returns:
        trained pytorch model
    """
    logger.debug("create model")

    # make everything deterministic -> seed setup
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    model = kwargs["model"]
    loss = kwargs["loss"]
    optimizer = kwargs["optimizer"]

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    losses_cls = None
    loss_cls = None
    accuracies = None
    if opt.cls_num > 0:
        losses_cls = Averaging()
        accuracies = Averaging()
        loss_cls = torch.nn.CrossEntropyLoss(ignore_index=-1)
    adjustable_lr = opt.lr

    logger.debug("epochs: %s", epochs)
    model.to(opt.device)
    epoch = 0
    # Loop over epochs -----------------------------------------------------
    for epoch in range(epochs):
        model.train()

        logger.debug("Epoch # %d" % epoch)
        if opt.lr_adj:
            if epoch % 30 == 0 and epoch > 0:
                adjustable_lr = adjust_lr(optimizer, adjustable_lr)
                logger.debug("lr: %f" % adjustable_lr)
        train_loop(
            model,
            loss,
            loss_cls,
            train_loader,
            optimizer,
            losses,
            losses_cls,
            accuracies,
            epoch,
            data_time,
            batch_time,
        )

        if save:
            save_model(model, epoch, optimizer, opt)

        logger.debug("loss: %f" % losses.avg)
        losses.reset()
        if opt.cls_num > 0:
            losses_cls.reset()
            accuracies.reset()
    # End loop over epochs -----------------------------------------------------
    if save:
        save_model(model, epoch, optimizer, opt)
    return model


def train_loop(
    model,
    loss,
    loss_cls,
    train_loader,
    optimizer,
    losses,
    losses_cls,
    accuracies,
    epoch,
    data_time,
    batch_time,
):
    """
    Performs just the training loop of the model.
    Args:
        model: the mlp model
        train_loader: the dataset loader
        optimizer: the optimizer
        loss: the loss function
        epoch: the current epoch.
        data_time: for printing purposes the time to read one batch
        batch_time: for printing purposes the time for one pass per batch
    """

    end = time.time()
    for i, (features, labels, cls_labels) in enumerate(train_loader):
        features = features.float().to(opt.device)
        labels = labels.float().to(opt.device)
        cls_labels = cls_labels.long().to(opt.device)

        data_time.update(time.time() - end)

        # Do the forward pass
        output_reg, output_cls = model(features)

        loss_values_reg = loss(output_reg.squeeze(), labels.squeeze())
        losses.update(loss_values_reg.item(), features.size(0))

        # The classification part
        if opt.cls_num > 0:
            acc = accuracy_fn(output=output_cls, target=cls_labels)

            loss_values_cls = loss_cls(
                input=output_cls,
                target=cls_labels.view(
                    -1,
                ),
            )
            accuracies.update(acc[0].item(), features.size(0))
            losses_cls.update(loss_values_cls.item(), features.size(0))
            cls_string = "Acc {acc.val:.3f} ({acc.avg:.4f})\t Cls-loss {closs.val:.4f} ({closs.avg:.4f})\t".format(
                acc=accuracies, closs=losses_cls
            )

            total_loss = opt.erlambda * loss_values_reg + loss_values_cls
        else:
            cls_string = ""
            total_loss = loss_values_reg

        # Do the backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0 and i:
            logger.debug(
                "Epoch: [{0}][{1}/{2}]\t"
                "Time {batch_time.val:.2f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.2f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.3f} ({loss.avg:.4f})\t"
                "Total loss {total_loss:.3f}\t".format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    total_loss=total_loss.item(),
                )
                + cls_string
            )


def save_model(model, epoch, optimizer, opt):
    """Save the model and the optimizer."""
    opt.resume_str = join(
        opt.dataset_root,
        "models",
        # "%s%s_%d.pth.tar" % (opt.log_str, opt.split, opt.cls_num),
        "%s.pth.tar" % (opt.model_name),
    )

    save_dict = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    dir_check(join(opt.dataset_root, "models"))
    torch.save(save_dict, opt.resume_str)


def test(test_loader, model, loss, **kwargs):
    """
    Evaluate a trained model.
    Args:
        test_loader: the test data loaders
        model: the model
        loss: the loss functions to evaluate
    """
    logger.debug("have model")

    # make everything deterministic -> seed setup
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True

    batch_time = Averaging()
    data_time = Averaging()
    losses = Averaging()

    end = time.time()
    if opt.cls_num > 0:
        losses_cls = Averaging()
        accuracies = Averaging()
        loss_cls = torch.nn.CrossEntropyLoss(ignore_index=-1)

    model.to(opt.device)
    model.eval()
    print("Testing model:", model, "|", opt.cls_num)

    # Loop over test samples ----------------------------------------------------
    for i, (features, labels, cls_labels) in enumerate(test_loader):
        features = features.float().to(opt.device)
        labels = labels.float().to(opt.device)
        cls_labels = cls_labels.long().to(opt.device)
        data_time.update(time.time() - end)

        # The forward pass through the model
        output_reg, output_cls = model(features)

        # Estimate the loss
        loss_values_reg = loss(output_reg.squeeze(), labels.squeeze())
        losses.update(loss_values_reg.item())

        # Classification part for printing
        if opt.cls_num > 0:
            acc = accuracy_fn(output=output_cls, target=cls_labels)

            loss_values_cls = loss_cls(
                input=output_cls,
                target=cls_labels.view(
                    -1,
                ),
            )
            accuracies.update(acc[0].item(), features.size(0))
            losses_cls.update(loss_values_cls.item(), features.size(0))
            cls_string = "Acc {acc.val:.4f} ({acc.avg:.4f})\t Cls-loss {closs.val:.4f} ({closs.avg:.4f})\t".format(
                acc=accuracies, closs=losses_cls
            )
        else:
            cls_string = ""
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            logger.debug(
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t".format(
                    i,
                    len(test_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                )
                + cls_string
            )
    # Loop over test samples ----------------------------------------------------
    logger.debug("loss: %f" % losses.avg)
    logger.debug(" \n \n MLP estimate [RMSE]: %f \n" % (losses.avg))
    return losses.avg


def load_model(model_path):
    """Loading the model"""
    resume_str = ("%s.pth.tar" % (model_path),)
    print(
        "Loading model from ",
        join(opt.dataset_root, "models", "%s" % resume_str),
        ".....",
    )
    opt.cls_num = cls_from_filename(model_path)

    if opt.device == "cpu":
        checkpoint = torch.load(
            join(opt.dataset_root, "models", "%s" % resume_str), map_location="cpu"
        )
    else:
        checkpoint = torch.load(join(opt.dataset_root, "models", "%s" % resume_str))

    checkpoint = checkpoint["state_dict"]
    logger.debug("loaded model: " + "%s" % resume_str)
    return checkpoint
