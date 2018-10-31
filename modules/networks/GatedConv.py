import torch.nn as nn
from GatedLinearUnits import anyGLU

class Conv2dBN(nn.Module):
    """
    a 2D Convolution Unit with Batch Normalization
    Args: Please read PyTorch references
    """
    def __init__(self, Cin, Cout, kernel, stride, padding):
        super(Conv2dBN, self).__init__()
        self.units = nn.ModuleList([
            nn.Conv2d(Cin, Cout, kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(Cout)
        ])

    def forward(self, x):
        for f in self.units:
            x = f(x)
        return x

class TransConv2dBN(nn.Module):
    """
    a 2D Convolution Unit with Batch Normalization

    Args:
      Cin     (int):
      Cout    (int):
      kernel (int or tuple): kernel size
      stride  (int or tuple): stride size
      padding (int or tuple): padding in both side
    """

    def __init__(self, Cin, Cout, kernel, stride, padding):
        super(TransConv2dBN, self).__init__()
        self.units = nn.ModuleList([
            nn.ConvTranspose2d(Cin, Cout, kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(Cout)
        ])

    def forward(self, x):
        for f in self.units:
            x = f(x)
        return x

class StarGANVC_G(nn.Module):
    def __init__(self):
        super(StarGANVC_G, self).__init__()
        self.units = nn.ModuleList([
            anyGLU(Conv2dBN,        1,  32, (3, 9), (1, 1), (1, 4)),
            anyGLU(Conv2dBN,       32,  64, (4, 8), (2, 2), (1, 3)),
            anyGLU(Conv2dBN,       64, 128, (4, 8), (2, 2), (1, 3)),
            anyGLU(Conv2dBN,      128,  64, (3, 5), (1, 1), (1, 2)),
            anyGLU(Conv2dBN,       64,   5, (9, 5), (9, 1), (0, 2)),
            anyGLU(TransConv2dBN,   5,  64, (9, 5), (9, 1), (0, 2)),
            anyGLU(TransConv2dBN,  64, 128, (3, 5), (1, 1), (1, 2)),
            anyGLU(TransConv2dBN, 128,  64, (4, 8), (2, 2), (1, 3)),
            anyGLU(TransConv2dBN,  64,  32, (4, 8), (2, 2), (1, 3)),
            nn.ConvTranspose2d(    32,   1, (3, 9), (1, 1), (1, 4))
        ])

    def forward(self, x):
        cnt = 0
        print(f"\ncount {cnt}: before apply {x.size()}")
        for f in self.units:
            x = f(x)
            print(f"count {cnt}: after apply {x.size()}\n")
            cnt = cnt + 1
        return x

import torch
input = torch.randn(1,1,36,512)
net = StarGANVC_G()
print(net(input))
