# Based on [CycleGAN-VC](https://arxiv.org/abs/1711.11293)
#
# * padding
#   + no discription about "padding" in original articles
#   + reproductive implementation use "padding=same" [github](https://github.com/leimao/Voice_Converter_CycleGAN)
#   + no idea for last conv (512x6x16->Z) because height kernel is just but width is not. Now using hetero padding (0,1)

import torch.nn as nn
from mnet.networks.SeqUnitModule import SeqUnitModule
from mnet.networks.GatedLinearUnits import anyGLU
from mnet.networks.PixelShuffle import PixelShuffle1d

class GatedFullyConvNet1d(SeqUnitModule):
    def __init__(self):
        super(GatedFullyConvNet, self).__init__()
        unt = [
            anyGLU(nn.Conv1d, 24, 128, 15, stride=1, padding=7, bias=True),   # (N_batch, 128, T)
            anyGLU(Conv1dIN, 128, 256, 5, stride=2, padding=2), # (N_batch, 256, T/2)
            anyGLU(Conv1dIN, 256, 512, 5, stride=2, padding=2) # (N_batch, 512, T/4)
        ]
        # x6 loop
        unt.extend([ResidualBottleneckBlock() for i in range(6)])  # (N_batch, 512, T/4)
        unt.extend([
            anyGLU(Conv1dPSIN, 512, 1024, 5, stride=1, padding=2, r=2, Cfinal=512), # (N_batch, 1024 == 2x2x256, T/4) ->  (N_batch, 512, T/2)
            anyGLU(Conv1dPSIN, 512, 512, 5, stride=1, padding=2, r=2, Cfinal=256),  # (N_batch, 512 == 2x2x128, T/2) ->    (N_batch, 156, T)
            nn.Conv1d(256, 24, 15, stride=1, padding=7)                        # (N_batch, 24, T)
        ])
        # output: (N_batch, 24, T)
        self.units = nn.ModuleList(unt)

    # with print check
    # def forward(self, x):
    #     cnt = 0
    #     print(f"\ncount {cnt}: before apply {x.size()}")
    #     for f in self.units:
    #         x = f(x)
    #         print(f"count {cnt}: after apply {x.size()}\n")
    #         cnt = cnt + 1
    #     return x

class Conv1dIN(SeqUnitModule):
    """
    a 1D Convolution Unit with Instance Normalization.
    For arguments, please read PyTorch references.
    """
    def __init__(self, Cin, Cout, W, stride, padding):
        super(Conv1dIN, self).__init__()
        # normal conv
        self.units = nn.ModuleList([
            nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm1d(Cout)
        ])

class ResidualBottleneckBlock(nn.Module):
    def __init__(self):
        super(ResidualBottleneckBlock, self).__init__()
        self.units = nn.ModuleList([
            # (N_batch, 1024, T/4)
            anyGLU(Conv1dIN, 512, 1024, 3, stride=1, padding=1),
            # (N_batch, 512, T/4)
            nn.Conv1d(1024, 512, 3, stride=1, padding=1),
            nn.InstanceNorm1d(512)
        ])
        self.shortcut = nn.Sequential()

    def forward(self, x):
        y = x
        for f in self.units:
            y = f(y)
        y += self.shortcut(x)
        return y

class Conv1dPSIN(SeqUnitModule):
    """
    Gated Convolution layer with Instance Normalization

    Args:
        Cout    (int): output channel Number of "Convolution"
        r       (int): PixelShuffle scaling factor
        Cfinal  (int): final output Channel size (Conv + PS)
        others: check PyTorch Conv1d manual
    """
    def __init__(self, Cin, Cout, W, stride, padding, r, Cfinal):
        super(Conv1dPSIN, self).__init__()
        self.units = nn.ModuleList([
            nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding, bias=True),
            PixelShuffle1d(r),
            nn.InstanceNorm1d(Cfinal)
        ])
