import torch
import torch.nn as nn

class View(nn.Module):
    def __init__(self, sz):
        super(View, self).__init__()
        self.sz = sz
    def forward(self, input):
        batch_size = input.size()[0]
        return input.view(batch_size, self.sz)

class GatedConv2d(nn.Module):
    """
    Gated Convolution layer

    Args:
      Cin     (int):
      Cout    (int):
      K       (int or tuple): kernel
      stride  (int or tuple): stride
      padding (int or tuple): padding
    """

    def __init__(self, Cin, Cout, K, stride, padding):
        super(GatedConv2d, self).__init__()
        # normal conv
        # b ∈ R<sup>n</sup>, n is Number of output feature map. This is fullConv so cannot specify size of b -> (1,1)
        self.cnv = nn.Conv2d(Cin, Cout, K, stride=stride, padding=padding, bias=True)
        # for gating
        self.cnv_gate = nn.Conv2d(Cin, Cout, K, stride=stride, padding=padding, bias=True)
        # summation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        A = self.cnv(input)
        B = self.cnv_gate(input)
        return A * self.sigmoid(B)


class GatedConv2dIN(nn.Module):
    """
    Gated Convolution layer with Instance Normalization

    Args:
      Cin     (int):
      Cout    (int):
      K       (int or tuple): kernel
      stride  (int or tuple): stride
      padding (int or tuple): padding
    """

    def __init__(self, Cin, Cout, K, stride, padding):
        super(GatedConv2dIN, self).__init__()
        # normal conv
        self.cnv = nn.Conv2d(Cin, Cout, K, stride=stride, padding=padding, bias=True)
        self.inNorm = nn.InstanceNorm2d(Cout)
        # for gating
        self.cnv_gate = nn.Conv2d(Cin, Cout, K, stride=stride, padding=padding, bias=True)
        self.inNorm_gate = nn.InstanceNorm2d(Cout)
        # summation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        A = self.inNorm(self.cnv(input))
        B = self.inNorm_gate(self.cnv_gate(input))
        return A * self.sigmoid(B)


class GatedCNN(nn.Module):
    def __init__(self):
        super(GatedCNN, self).__init__()
        # input: (N_batch, 1, 24, T(128))
        self.units = nn.ModuleList([
            GatedConv2d(1, 128, (3, 3), (1, 2), padding=1),  # (N_batch, 128, 24, T/2)
            GatedConv2dIN(128, 256, (3, 3), (2, 2), padding=1),  # (N_batch, 256, 12, T/4)
            GatedConv2dIN(256, 512, (3, 3), (2, 2), padding=1),  # (N_batch, 512, 6, T/8)
            GatedConv2dIN(512, 1024, (6, 3), (1, 2), padding=(0, 1)),  # (N_batch, 1024, 1, 128/16==8)
            View(1024 * 1 * 8),
            nn.Linear(1024 * 1 * 8, 1)            # output: N_batchx1x1x1
        ])

    def forward(self, x):
        cnt = 0
        # print(f"\ncount {cnt}: before apply {x.size()}")
        for f in self.units:
            x = f(x)
            # print(f"count {cnt}: after apply {x.size()}\n")
            cnt = cnt + 1
        return x

# Based on [CycleGAN-VC](https://arxiv.org/abs/1711.11293)
#
# * padding
#   + no discription about "padding" in original articles
#   + reproductive implementation use "padding=same" [github](https://github.com/leimao/Voice_Converter_CycleGAN)
#   + no idea for last conv (512x6x16->Z) because height kernel is just but width is not. Now using hetero padding (0,1)
