# should install mnet library
import torch.nn as nn
from mnet.networks.SeqUnitModule import SeqUnitModule
from mnet.networks.GatedLinearUnits import anyGLU

class StarGAN_C(SeqUnitModule):
    def __init__(self):
        super(StarGAN_D, self).__init__()
        self.units = nn.ModuleList([                            # ( 1, 36, 512)
            # slice?
            anyGLU(Conv2dBN,  1,  8, (4, 4), (2, 2), (1, 1)), # ( 8, 36, 512)
            anyGLU(Conv2dBN,  8, 16, (4, 4), (2, 2), (1, 1)), # (16, 36, 256)
            anyGLU(Conv2dBN, 16, 32, (4, 4), (2, 2), (1, 1)), # (32, 36, 128)
            anyGLU(Conv2dBN, 32, 16, (3, 4), (1, 2), (1, 1)), # (16, 36,  64)
            nn.Conv2d(       16,  4, (1, 4), (1, 1), (0, 2)), # ( 4,  1,  64)
            # softmax
            # product
        ])

    # def forward(self, x):
    #     cnt = 0
    #     print(f"\ncount {cnt}: before apply {x.size()}")
    #     for f in self.units:
    #         x = f(x)
    #         print(f"count {cnt}: after apply {x.size()}\n")
    #         cnt = cnt + 1
    #     return x

class Conv2dBN(SeqUnitModule):
    """
    a 2D Convolution Unit with Batch Normalization.
    For arguments, please read PyTorch references.
    """
    def __init__(self, Cin, Cout, kernel, stride, padding):
        super(Conv2dBN, self).__init__()
        self.units = nn.ModuleList([
            nn.Conv2d(Cin, Cout, kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(Cout)
        ])

if __name__ == "__main__":
    import torch
    input = torch.randn(1,1,36,512)
    net = StarGAN_VC_G()
    print(net(input))
