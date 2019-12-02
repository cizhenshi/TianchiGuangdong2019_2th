import torch
import torch.nn as nn


class Scale(nn.Module):

    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

class Scale_channel(nn.Module):
    def __init__(self, scale=1.0, channels=1):
        super(Scale_channel, self).__init__()
        self.scale = nn.Parameter(torch.ones(channels, dtype=torch.float)*scale)

    def forward(self, x):
        b, c, h, w = x.shape
        out = x.view(b, c, -1) * self.scale.view(-1, 1)
        out = out.view(b, c, h, w)
        return out
