import torch
import torch.nn as nn
from models import modules


class model(nn.Module):
    def __init__(self, args, Encoder, num_features, block_channel, addcls, cls_num):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)

        if addcls is True and cls_num > 0:
            self.R = modules.Rcls(args, block_channel, cls_num)
        else:
            self.R = modules.R(args, block_channel)

    def forward(self, x, depth=None, epoch=None):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(
            x_block1,
            x_block2,
            x_block3,
            x_block4,
            [x_decoder.size(2), x_decoder.size(3)],
        )

        # Depending on R, out = [x_reg, x_cls, [feat] ] or out = [x_reg, [feat]]
        out = self.R(torch.cat((x_decoder, x_mff), 1), depth, epoch)

        return out
