import pickle
import os
import sys
import time
import torch
from data import SynthDataset
from model import Reg, Reg_Cls
from torchsummary import summary
import numpy as np

from utils import (
    iter_print,
    update_metrics,
    save_checkpoint,
    check_rootfolders,
    AverageMeter,
)
from plot import plot_predictions


def resume(args, model_reg, model_regcls):
    """
    Resume training from stored weights.
    """

    if args.resume:
        # Resume the reg model
        filename = "%s/%s/ckpt.pth.tar" % (args.root_model, args.store_name)
        if os.path.isfile(filename):
            print(("=> loading checkpoint '{}'".format(filename)))
            checkpoint = torch.load(filename)
            model_reg.load_state_dict(checkpoint["state_dict_reg"], strict=False)
            model_regcls.load_state_dict(checkpoint["state_dict_regcls"], strict=False)
            print(("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"])))
        else:
            print(("=> no checkpoint found at '{}'".format(filename)))
    return model_reg, model_regcls


def def_datasets(args):
    """
    Create training and val, test datasets.
    The path to the correct pickled data files should be in the args.
    """

    train_dataset = SynthDataset(
        args=args,
        batch_size=args.batch_size,
        setname="train",
        num_segment=args.num_segment,
    )
    val_dataset = SynthDataset(
        args=args,
        batch_size=args.test_batch_size,
        setname="val",
        seg_ranges=train_dataset.seg_ranges,
        num_segment=args.num_segment,
    )
    test_dataset = SynthDataset(
        args=args,
        batch_size=args.test_batch_size,
        setname="test",
        seg_ranges=train_dataset.seg_ranges,
        num_segment=args.num_segment,
    )

    return train_dataset, val_dataset, test_dataset


def def_dataloaders(args, train_dataset, val_dataset, test_dataset):
    """
    Just defines the data loaders over the given datasets.
    """

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def def_models(args, train_dataset):
    """
    Define the models for regression and regression+classification.
    """

    model_reg = Reg(
        hidden_dims=args.hidden_list,
        input_dim=train_dataset.__dim__(),
        num_comp=args.num_comp,
        bn=args.bn,
    )
    model_regcls = Reg_Cls(
        hidden_dims=args.hidden_list,
        input_dim=train_dataset.__dim__(),
        num_comp=args.num_comp,
        num_class=args.num_segment,
        bn=args.bn,
    )

    # Resume models if any
    model_reg, model_regcls = resume(args, model_reg, model_regcls)

    # Cuda models
    if args.use_cuda:
        model_reg = torch.nn.DataParallel(model_reg.cuda())
        model_regcls = torch.nn.DataParallel(model_regcls.cuda())
    else:
        model_reg = torch.nn.DataParallel(model_reg)
        model_regcls = torch.nn.DataParallel(model_regcls)

    print("[Reg Model] definition:", model_reg)
    print("[Reg+Cls Model] definition:", model_regcls)
    # print(summary(model_reg.module.float(), (1, train_dataset.__dim__())))
    # print(summary(model_regcls.module.float(), (1, train_dataset.__dim__())))
    model_reg.module.double()
    model_regcls.module.double()

    return model_reg, model_regcls


def def_optimizer(args, model_reg, model_regcls):
    """
    Defined the optimizer, default is adam.
    """

    if args.optim.startswith("adam"):
        optimizer_reg = torch.optim.Adam(
            params=model_reg.module.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler_reg = torch.optim.lr_scheduler.StepLR(
            optimizer_reg, step_size=80, gamma=args.lrdecay
        )

        # Defined the optimizer
        optimizer_regcls = torch.optim.Adam(
            params=model_regcls.module.parameters(),
            lr=args.cls_lr,
            weight_decay=args.weight_decay,
        )
        scheduler_regcls = torch.optim.lr_scheduler.StepLR(
            optimizer_regcls, step_size=80, gamma=args.lrdecay
        )
    else:
        print("Not implemented")

    return optimizer_reg, scheduler_reg, optimizer_regcls, scheduler_regcls


def run_train(args, wandb):
    """
    Runs the main training loop with evaluation and all.
    """

    train_dataset, val_dataset, test_dataset = def_datasets(args)
    model_reg, model_regcls = def_models(args, train_dataset)
    train_loader, val_loader, test_loader = def_dataloaders(
        args, train_dataset, val_dataset, test_dataset
    )
    print("Loader lengths:", len(train_loader), len(val_loader), len(test_loader))

    optimizer_reg, scheduler_reg, optimizer_regcls, scheduler_regcls = def_optimizer(
        args, model_reg, model_regcls
    )

    # Train model
    best_loss_reg = 0.0
    best_loss_regcls = 0.0
    for e in range(0, args.max_epochs):
        # Get the validation loss during training
        if args.val:
            loss_val_reg, loss_val_regcls, prec_val = test_loop(
                args, val_loader, model_reg, model_regcls, e, args.max_epochs, args.lr
            )

        # Run 1 training loop over the data
        loss_train_reg, loss_train_regcls, prec_train = train_loop(
            args,
            train_loader,
            optimizer_reg,
            optimizer_regcls,
            model_reg,
            model_regcls,
            e,
            args.max_epochs,
            args.lr,
        )
        scheduler_reg.step()
        scheduler_regcls.step()

        # Store the best loss so far is we want to use it.
        if args.val:
            best_loss_reg = (
                loss_val_reg.avg if loss_val_reg.avg < best_loss_reg else best_loss_reg
            )
            best_loss_regcls = (
                loss_val_regcls.avg
                if loss_val_regcls.avg < best_loss_regcls
                else best_loss_regcls
            )

            # Do some logging of results
            if args.debug:
                wandb.log(
                    {
                        "sampling": (args.name).split("_")[0],
                        "setting": (args.name).split("_")[2],
                        "lr": args.lr,
                        "epoch": e,
                        "Train reg": loss_train_reg.avg,
                        "Val reg": loss_val_reg.avg,
                        "Train reg+cls": loss_train_regcls.avg,
                        "Val reg+cls": loss_val_regcls.avg,
                        "Top1 train": prec_train.avg,
                        "Top1 val": prec_val.avg,
                    }
                )

    # Eval the final model on the test dataset
    loss_test_reg, loss_test_regcls, prec_test = test_loop(
        args,
        test_loader,
        model_reg,
        model_regcls,
        args.max_epochs,
        args.max_epochs,
        args.lr,
    )

    # Log the final model
    log_data = {
        "name": args.name,
        "seed": args.random_seed,
        "cls": args.orig_num_segment,
        "cls_fact": args.num_segment,
        "train_reg_mse": loss_train_reg.avg,
        "train_regcls_mse": loss_train_regcls.avg,
        "val_reg_mse": loss_test_reg.avg,
        "val_regcls_mse": loss_test_regcls.avg,
        "top1_train": prec_train.avg,
        "top1_val": prec_test.avg,
    }
    print(log_data)
    check_rootfolders(args.log_file)
    with open(args.log_file, "ab+") as f:
        pickle.dump(log_data, f)

    # Save the model
    check_rootfolders(args.root_model, args.store_name)
    save_checkpoint(
        args.root_model,
        args.store_name,
        {
            "epoch": args.max_epochs,
            "arch": args.hidden_list,
            "state_dict_reg": model_reg.module.state_dict(),
            "state_dict_regcls": model_regcls.module.state_dict(),
            "optimizer_reg": optimizer_reg.state_dict(),
            "optimizer_regcls": optimizer_regcls.state_dict(),
            "loss_reg": loss_test_reg.avg,
            "loss_regcls": loss_test_regcls.avg,
        },
        False,
    )


def train_loop(
    args,
    train_loader,
    optimizer_reg,
    optimizer_regcls,
    model_reg,
    model_regcls,
    epoch,
    epochs,
    lr,
):
    """
    This is just the training part of the training loop.
    """
    meter_batch = AverageMeter()
    meter_data = AverageMeter()
    meter_losses_reg = AverageMeter()
    meter_losses_regcls = AverageMeter()
    meter_top1 = AverageMeter()
    meter_top5 = AverageMeter()
    end = time.time()

    # Loop over training data
    for (i, (data, target, label)) in enumerate(train_loader):
        # measure data loading time
        meter_data.update(time.time() - end)

        # set stuff on gpu
        if args.use_cuda:
            data = data.cuda()
            target = target.cuda()
            label = label.cuda()

        # forward pass
        output_reg = model_reg(data)
        output_regcls_reg, output_regcls_cls = model_regcls(data)

        # compute the losses
        loss_reg = model_reg.module.loss(output_reg, target)
        tloss_regcls, loss_regcls_reg, loss_regcls_cls = model_regcls.module.loss(
            output_regcls_reg, output_regcls_cls, target, label, weight=args.erlambda
        )

        # measure accuracy and record loss
        update_metrics(
            output=output_reg,
            target=target,
            losses=meter_losses_reg,
            loss=loss_reg.detach().cpu().numpy(),
        )
        update_metrics(
            output=output_regcls_cls,
            target=label,
            losses=meter_losses_regcls,
            loss=tloss_regcls.detach().cpu().numpy(),
            top1=meter_top1,
            top5=meter_top5,
        )

        # backward pass
        optimizer_reg.zero_grad()
        optimizer_regcls.zero_grad()

        loss_reg.backward()
        tloss_regcls.backward()

        optimizer_reg.step()
        optimizer_regcls.step()

        # measure elapsed time
        meter_batch.update(time.time() - end)
        end = time.time()

        # Print the metrics
        iter_print(
            text="Train reg ",
            epoch=epoch,
            epochs=epochs,
            lr=lr,
            lent=len(train_loader),
            batch_time=meter_batch,
            data_time=meter_data,
            losses=meter_losses_reg,
            itr=i,
            print_freq=50,
        )
        iter_print(
            text="Train reg_cls ",
            epoch=epoch,
            epochs=epochs,
            lr=lr,
            lent=len(train_loader),
            batch_time=meter_batch,
            data_time=meter_data,
            losses=meter_losses_regcls,
            itr=i,
            top1=meter_top1,
            top5=meter_top5,
            print_freq=50,
        )
    return meter_losses_reg, meter_losses_regcls, meter_top1


def test_loop(args, val_loader, model_reg, model_regcls, epoch, epochs, lr):
    """
    This does the validation part of the model.
    """

    meter_batch = AverageMeter()
    meter_data = AverageMeter()
    meter_losses_reg = AverageMeter()
    meter_losses_regcls = AverageMeter()
    meter_top1 = AverageMeter()
    meter_top5 = AverageMeter()
    end = time.time()

    alloutput_reg = []
    alloutput_regcls = []
    alldata = []
    alltarget = []
    # Loop over the test/val data
    for (i, (data, target, label)) in enumerate(val_loader):
        # measure data loading time
        meter_data.update(time.time() - end)

        # set stuff on the gpu
        if args.use_cuda:
            data = data.cuda()
            target = target.cuda()
            label = label.cuda()

        # forward pass
        output_reg = model_reg(data)
        output_regcls_reg, output_regcls_cls = model_regcls(data)

        # store some predictions for plotting
        if args.debug and epoch == epochs - 1:
            alloutput_reg.append(output_reg.detach().cpu().numpy())
            alloutput_regcls.append(output_regcls_reg.detach().cpu().numpy())
            alldata.append(data.detach().cpu().numpy())
            alltarget.append(target.detach().cpu().numpy())

        # compute the loss
        loss_reg = model_reg.module.loss(output_reg, target)
        _, loss_regcls_reg, _ = model_regcls.module.loss(
            output_regcls_reg, output_regcls_cls, target, label, weight=0
        )
        # measure accuracy and record loss
        update_metrics(
            output=output_reg,
            target=target,
            losses=meter_losses_reg,
            loss=loss_reg.detach().cpu().numpy(),
        )
        update_metrics(
            output=output_regcls_cls,
            target=label,
            losses=meter_losses_regcls,
            loss=loss_regcls_reg.detach().cpu().numpy(),
            top1=meter_top1,
            top5=meter_top5,
        )

        # measure elapsed time
        meter_batch.update(time.time() - end)
        end = time.time()

        # print current metrics
        iter_print(
            text="Val reg ",
            epoch=epoch,
            epochs=epochs,
            lr=lr,
            lent=len(val_loader),
            batch_time=meter_batch,
            data_time=meter_data,
            losses=meter_losses_reg,
            itr=i,
            print_freq=50,
        )
        iter_print(
            text="Val reg+cls ",
            epoch=epoch,
            epochs=epochs,
            lr=lr,
            lent=len(val_loader),
            batch_time=meter_batch,
            data_time=meter_data,
            losses=meter_losses_regcls,
            itr=i,
            top1=meter_top1,
            top5=meter_top5,
            print_freq=50,
        )

    # plot predictions
    if args.debug and epoch == epochs - 1:
        npout_reg = np.vstack(alloutput_reg)
        npout_regcls = np.vstack(alloutput_regcls)
        npdata = np.vstack(alldata)
        nptarget = np.vstack(alltarget)
        plot_predictions(args, npout_reg, npout_regcls, npdata, nptarget)

    return meter_losses_reg, meter_losses_regcls, meter_top1
