# should install mnet library
import torch.nn as nn
from mnet.networks.SeqUnitModule import SeqUnitModule
from mnet.networks.GatedLinearUnits import anyGLU

class StarGAN_VC_G(SeqUnitModule):
    def __init__(self):
        super(StarGAN_G, self).__init__()
        self.units = nn.ModuleList([                                 # (  1, 36, 512)
            anyGLU(Conv2dBN,        1,  32, (3, 9), (1, 1), (1, 4)), # ( 32, 36, 512)
            anyGLU(Conv2dBN,       32,  64, (4, 8), (2, 2), (1, 3)), # ( 64, 18, 256)
            anyGLU(Conv2dBN,       64, 128, (4, 8), (2, 2), (1, 3)), # (128,  9, 128)
            anyGLU(Conv2dBN,      128,  64, (3, 5), (1, 1), (1, 2)), # ( 64,  9, 128)
            anyGLU(Conv2dBN,       64,   5, (9, 5), (9, 1), (0, 2)), # (  5,  1, 128)
            anyGLU(TransConv2dBN,   5,  64, (9, 5), (9, 1), (0, 2)), # ( 64,  9, 128)
            anyGLU(TransConv2dBN,  64, 128, (3, 5), (1, 1), (1, 2)), # (128,  9, 128)
            anyGLU(TransConv2dBN, 128,  64, (4, 8), (2, 2), (1, 3)), # ( 64, 18, 256)
            anyGLU(TransConv2dBN,  64,  32, (4, 8), (2, 2), (1, 3)), # ( 32, 36, 512)
            nn.ConvTranspose2d(    32,   1, (3, 9), (1, 1), (1, 4))  # (  1, 36, 512)
        ])

    # def forward(self, x):
    #     cnt = 0
    #     print(f"\ncount {cnt}: before apply {x.size()}")
    #     for f in self.units:
    #         x = f(x)
    #         print(f"count {cnt}: after apply {x.size()}\n")
    #         cnt = cnt + 1
    #     return x



class TransConv2dBN(SeqUnitModule):
    """
    a 2D Transposed Convolution Unit with Batch Normalization.
    For arguments, please read PyTorch references.
    """
    def __init__(self, args):
        super(TransConv2dBN, self).__init__()
        self.units = nn.ModuleList([
            nn.ConvTranspose2d(Cin, Cout, kernel, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(Cout)
        ])

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
