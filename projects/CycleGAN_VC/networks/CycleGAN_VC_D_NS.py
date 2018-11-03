# Based on [CycleGAN-VC](https://arxiv.org/abs/1711.11293)
#
# * padding
#   + no discription about "padding" in original articles
#   + reproductive implementation use "padding=same" [github](https://github.com/leimao/Voice_Converter_CycleGAN)
#   + no idea for last conv (512x6x16->Z) because height kernel is just but width is not. Now using hetero padding (0,1)

import torch.nn as nn
from mnet.networks.SeqUnitModule import SeqUnitModule
from mnet.networks.GatedLinearUnits import anyGLU

class GatedCNN2D(SeqUnitModule):
    def __init__(self):
        super(GatedCNN2D, self).__init__()
        # input: (N_batch, 1, 24, T(128))
        self.units = nn.ModuleList([
            anyGLU(nn.Conv2d, 1, 128, (3, 3), (1, 2), padding=1, bias=True),  # (N_batch, 128, 24, T/2)
            anyGLU(Conv2dIN, 128, 256, (3, 3), (2, 2), padding=1),  # (N_batch, 256, 12, T/4)
            anyGLU(Conv2dIN, 256, 512, (3, 3), (2, 2), padding=1),  # (N_batch, 512, 6, T/8)
            anyGLU(Conv2dIN, 512, 1024, (6, 3), (1, 2), padding=(0, 1)),  # (N_batch, 1024, 1, 128/16==8)
            View(1024 * 1 * 8),
            nn.Linear(1024 * 1 * 8, 1),            # output: N_batchx1x1x1
            nn.Sigmoid()
        ])

    # with print check
    # def forward(self, x):
    #     cnt = 0
    #     print(f"\ncount {cnt}: before apply {x.size()}")
    #     for f in self.units:
    #         x = f(x)
    #         print(f"count {cnt}: after apply {x.size()}\n")
    #         cnt = cnt + 1
    #     return x

class Conv2dIN(SeqUnitModule):
    """
    a 2D Convolution Unit with Instance Normalization.
    For arguments, please read PyTorch references.
    """
    def __init__(self, Cin, Cout, K, stride, padding):
        super(Conv2dIN, self).__init__()
        # input: (N_batch, 1, 24, T(128))
        self.units = nn.ModuleList([
            nn.Conv2d(Cin, Cout, K, stride=stride, padding=padding, bias=True),
            nn.InstanceNorm2d(Cout)
        ])

class View(nn.Module):
    def __init__(self, sz):
        super(View, self).__init__()
        self.sz = sz
    def forward(self, input):
        batch_size = input.size()[0]
        return input.view(batch_size, self.sz)
