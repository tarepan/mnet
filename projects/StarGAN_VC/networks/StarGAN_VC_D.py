# should install mnet library
import torch.nn as nn
from mnet.networks.SeqUnitModule import SeqUnitModule
from mnet.networks.GatedLinearUnits import anyGLU

class StarGAN_D(SeqUnitModule):
    def __init__(self):
        super(StarGAN_D, self).__init__()
        self.withClasses = nn.ModuleList([                            # ( 1, 36, 512)
            GatedConv2dBN_ax( 1+1, 32, ( 3, 9), ( 1, 1), (1, 4)), # (32, 36, 512)
            GatedConv2dBN_ax(32+1, 32, ( 3, 8), ( 1, 2), (1, 3)), # (32, 36, 256)
            GatedConv2dBN_ax(32+1, 32, ( 3, 8), ( 1, 2), (1, 3)), # (32, 36, 128)
            GatedConv2dBN_ax(32+1, 32, ( 3, 6), ( 1, 2), (1, 2)), # (32, 36,  64)
            classAdittion()
        ])
        self.woClasses = nn.ModuleList([
            nn.Conv2d(       32+1,  1, (36, 5), (36, 1), (0, 2)), # ( 1,  1,  64)
            nn.Sigmoid(),
        ])

    def forward(self, x, cls):
        for f in self.withClasses:
            x = f(x, cls)
        for f in self.woClasses:
            x = f(x)
        return x

class classAdittion(nn.Module):
    def __init__(self):
        super(classAdittion, self).__init__()

    def forward(self, input, cls):
        # concat (cls should be 1/x time of feature dim)
        cls_tiled = cls.repeat(1,1, input.size()[2]//cls.size()[2], input.size()[3])
        concated = torch.cat((input, cls_tiled), dim=1)
        # print(f"size: input/ {input.size()}, cls_tiled/ {cls_tiled.size()}, concated/ {concated.size()}")
        return concated


class GatedConv2dBN_ax(nn.Module):
    """
    a 2D Convolution Unit with GLU, Batch Normalization and auxility classes.
    For arguments, please read PyTorch references.
    """
    def __init__(self, Cin, Cout, kernel, stride, padding):
        super(GatedConv2dBN_ax, self).__init__()
        self.tiling = classAdittion()
        self.cnv = nn.Conv2d(Cin, Cout, kernel, stride=stride, padding=padding, bias=True)
        self.bn = nn.BatchNorm2d(Cout)
        self.cnv_gate = nn.Conv2d(Cin, Cout, kernel, stride=stride, padding=padding, bias=True)
        self.bn_gate = nn.BatchNorm2d(Cout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, cls):
        print(f"size: input/ {input.size()}, cls/ {cls.size()}")
        # concat (cls should be 1/x time of feature dim)
        concated = self.tiling(input, cls)
        # data
        conved = self.cnv(concated)
        batched = self.bn(conved)
        # Gate
        conved_gate = self.cnv_gate(concated)
        batched_gate = self.bn_gate(conved_gate)
        return batched * self.sigmoid(batched_gate)



if __name__ == "__main__":
    import torch
    from pathlib import Path
    input = torch.randn(1,1,36,512)
    cls = torch.randn(1,1, 2, 1)
    net = StarGAN_D()
    print(net(input, cls))
