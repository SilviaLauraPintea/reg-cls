#!/usr/bin/env python

"""Baseline for relative time embedding: learn regression model in terms of
relative time.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from ute.utils.arg_pars import opt
from ute.utils.logging_setup import logger


class MLPrelu(nn.Module):
    """MLP model with RELU activations instead of sigmoid (It works better on absolute progress prediction)."""

    def __init__(self):
        super(MLPrelu, self).__init__()

        self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.relu2 = nn.ReLU()
        self.fc_last = nn.Linear(opt.embed_dim, 1)

        # Add a classification head if classes are non-zero
        if opt.cls_num > 0:
            self.fc_last_cls = nn.Linear(opt.embed_dim, opt.cls_num)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))

        # Define the regression head
        x_reg = self.fc_last(x)

        x_cls = None
        if opt.cls_num > 0:
            x_cls = self.fc_last_cls(x)
        return x_reg, x_cls

    def embedded(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def loss(self):
        # At test time report standard RMSE
        if opt.test_model:
            self.loss_fn = RMSELoss()
        # At training time use the paper MSE (reduction: sum)
        else:
            if opt.loss.endswith("paper"):
                self.loss_fn = nn.MSELoss(reduction="sum")
            elif opt.loss.endswith("l1"):
                self.loss_fn = nn.L1Loss(reduction="mean")
            elif opt.loss.endswith("root"):
                self.loss_fn = RMSELoss()
            elif opt.loss.endswith("mse"):
                self.loss_fn = nn.MSELoss(reduction="mean")
        logger.debug(str(self.loss_fn))
        return self.loss_fn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="mean")

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class MLP(nn.Module):
    """MLP model with Sigmoid activations."""

    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(opt.feature_dim, opt.embed_dim * 2)
        self.fc2 = nn.Linear(opt.embed_dim * 2, opt.embed_dim)
        self.fc_last = nn.Linear(opt.embed_dim, 1)

        if opt.cls_num > 0:
            self.fc_last_cls = nn.Linear(opt.embed_dim, opt.cls_num)

        self._init_weights()

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))

        x_reg = self.fc_last(x)

        x_cls = None
        if opt.cls_num > 0:
            x_cls = self.fc_last_cls(x)
        return x_reg, x_cls

    def embedded(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def loss(self):
        # At test time report standard MSE/RMSE
        if opt.test_model:
            self.loss_fn = RMSELoss()
        # At training time use the paper MSE (reduction: sum)
        else:
            if opt.loss.endswith("paper"):
                self.loss_fn = nn.MSELoss(reduction="sum")
            elif opt.loss.endswith("l1"):
                self.loss_fn = nn.L1Loss(reduction="mean")
            elif opt.loss.endswith("root"):
                self.loss_fn = RMSELoss()
            elif opt.loss.endswith("mse"):
                self.loss_fn = nn.MSELoss(reduction="mean")
        logger.debug(str(self.loss_fn))
        return self.loss_fn


def create_model():
    """Create the mlp model and define the optimizer."""

    torch.manual_seed(opt.seed)

    if opt.model_name.endswith("relu"):
        model = MLPrelu().to(opt.device)
    else:
        model = MLP().to(opt.device)

    if opt.optim.startswith("adam"):
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay
        )
    logger.debug(str(model))
    logger.debug(str(optimizer))
    return model, optimizer
