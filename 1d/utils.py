import numpy as np
import os
import torch
import math


def update_metrics(
    output,
    target,
    losses,
    loss,
    top1=None,
    top5=None,
):
    """
    Update the metrics over the iterations.
    """
    if (top1 is not None) and (top5 is not None):
        if len(output.shape) == 1:
            output = torch.sigmoid(output).unsqueeze(1)
            neg = 1.0 - output
            output = torch.cat((neg, output), dim=1)  # 0 - neg, 1 - pos
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        top1.update(prec1.item())
        top5.update(prec5.item())
    if losses is not None:
        losses.update(loss.item())


def epoch_print(text, epoch, epochs, lr, loss_train, loss_val, prec_train, prec_val):
    """
    Printing utility per iterations: it prints times, losses and precisions at certain epochs.
    """
    string = (
        text
        + " epoch: [{0}/{1}]\t"
        + "lr: {lr:.5f}\t"
        + "Train loss {loss_train.val:.4f} ({loss_train.avg:.4f})\t"
        + "Val loss {loss_val.val:.4f} ({loss_val.avg:.4f})\t"
        + "Train prec@1 {prec_train.val:.3f} ({prec_train.avg:.3f})\t"
        + "Val prec@1 {prec_val.val:.3f} ({prec_val.avg:.3f})\t"
    )

    output = string.format(
        epoch,
        epochs,
        lr=lr,
        loss_train=loss_train,
        loss_val=loss_val,
        prec_train=prec_train,
        prec_val=prec_val,
    )
    print(output)


def iter_print(
    text,
    lent,
    batch_time,
    data_time,
    losses,
    epoch,
    epochs,
    itr,
    top1=None,
    top5=None,
    lr=None,
    print_freq=50,
):
    """
    Printing utility per iterations: it prints times, losses and accuracies at certain steps.
    """
    perc_string = ""
    top_string = ""
    lr_string = ""
    if top1 is not None and top5 is not None:
        top_string = "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\tPrec@5 {top5.val:.3f} ({top5.avg:.3f})"
    else:
        perc_string = " | ({perc:.2f} %)"
    if lr is not None:
        lr_string = "lr: {lr:.5f}\t"

    if (epoch % print_freq) == 0 and itr == 0:
        string = (
            text
            + " Progress: Epoch [{0}/{1}] Iter [{2}/{3}]\t"
            + lr_string
            + "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            + "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
            + "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
            + perc_string
            + top_string
        )
        output = string.format(
            epoch + 1,
            epochs,
            itr + 1,
            lent,
            lr=lr,
            batch_time=batch_time,
            data_time=data_time,
            loss=losses,
            perc=losses.avg * 100.0,
            top1=top1,
            top5=top5,
        )
        print(output)


def save_checkpoint(root_model, store_name, state, is_best):
    """
    Save the model and state to the checkpoint
    """
    filename = "%s/%s/ckpt.pth.tar" % (root_model, store_name)
    torch.save(state, filename)
    if is_best:
        filename = "%s/%s/best.pth.tar" % (root_model, store_name)
        torch.save(state, filename)


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


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.
    """

    maxk = min(max(topk), output.shape[1])
    batch_size = target.size(0)

    val_inds = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = val_inds.indices.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
