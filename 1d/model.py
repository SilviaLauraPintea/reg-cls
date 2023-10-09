import torch
from torch import nn
import numpy as np


class Reg(nn.Module):
    def __init__(self, hidden_dims, input_dim, num_comp, bn):
        """Args:
        hidden_dim: Number of nodes for each layer: [num_layer1, num_layer2, ..]
        input_dim: The input dimensions
        num_comp: Size of last hidden layer
        """

        super(Reg, self).__init__()

        # The feature extractor
        self.mlist = torch.nn.ModuleList([])
        outdim = 0
        self.hidden_dims = hidden_dims.copy()  # otherwise we change the args
        self.hidden_dims.append(num_comp)  # Last layer is num_comp

        for i, hidden in enumerate(self.hidden_dims):
            outdim = hidden
            indim = input_dim if i == 0 else self.hidden_dims[i - 1]

            # FC layers
            self.mlist.append(nn.Linear(indim, outdim))

            ## Change the initializer for the linear layers
            torch.nn.init.xavier_uniform_(self.mlist[-1].weight)
            if self.mlist[-1].bias is not None:
                torch.nn.init.zeros_(self.mlist[-1].bias)

            # BN and non-linearity
            if bn:
                self.mlist.append(nn.BatchNorm1d(outdim))
            self.mlist.append(nn.ReLU(inplace=False))

        # The final output
        self.final_reg = nn.Linear(num_comp, 1)

        torch.nn.init.xavier_uniform_(self.final_reg.weight)
        if self.final_reg.bias is not None:
            torch.nn.init.zeros_(self.final_reg.bias)

        # Regression loss
        self.criterion = torch.nn.MSELoss()

    def loss(self, pred, target):
        loss_reg = self.criterion(
            input=pred,
            target=target.view(-1, 1).contiguous(),
        )
        return loss_reg

    def pre_forward(self, input):
        x = input.view(-1, input.shape[-1]).contiguous()
        for ii, m in enumerate(self.mlist):
            x = m(x)
        return x

    def forward(self, input):
        x = self.pre_forward(input)

        # Add the final layer
        x = self.final_reg(x)
        return x


# ----------------------------------------------------------------


class Reg_Cls(nn.Module):
    def __init__(self, hidden_dims, input_dim, num_comp, num_class, bn):
        """Args:
        hidden_dim: Number of nodes for each layer: [num_layer1, num_layer2, ..]
        input_dim: The input dimensions
        num_comp: Size of last hidden layer
        num_class: Number of classes to predict
        """
        super(Reg_Cls, self).__init__()

        # Define the regression backbone
        self.reg = Reg(hidden_dims, input_dim, num_comp, bn)

        # The final logistic regressor
        self.final_cls = nn.Linear(num_comp, num_class)

        torch.nn.init.xavier_uniform_(self.final_cls.weight)
        if self.final_cls.bias is not None:
            torch.nn.init.zeros_(self.final_cls.bias)

        # Classification loss
        self.criterion_cls = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.criterion_reg = torch.nn.MSELoss()

    def loss(self, pred_reg, pred_cls, target, label, weight):
        loss_reg = self.criterion_reg(
            input=pred_reg,
            target=target.view(-1, 1).contiguous(),
        )

        loss_cls = self.criterion_cls(
            input=pred_cls,
            target=label.view(-1).contiguous(),
        )

        # Use the lambda weight
        if weight!=0:
            total_loss = (weight * loss_reg + loss_cls)
        else:
            total_loss = loss_reg
        return total_loss, loss_reg, loss_cls

    def forward(self, input):
        x = self.reg.pre_forward(input)

        x_reg = self.reg.final_reg(x)
        x_cls = self.final_cls(x)

        return x_reg, x_cls
