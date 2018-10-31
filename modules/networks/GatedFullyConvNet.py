import torch
import torch.nn as nn

class PixelShuffle1d(nn.Module):
    def __init__(self, r):
        super(PixelShuffle1d, self).__init__()
        self.r = r

    def forward(self, input):
        return pixel_shuffle_1d(input, self.r)

def pixel_shuffle_1d(input, upscale_factor):
    """
    Args:
        input (tensor([N_batch, C, W]))
    Returns:
        (tensor[N_batch, C/r, r*W])
    """
    batch_size, channels, in_width = input.size()
    channels //= upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, upscale_factor,
        in_width)

    shuffle_out = input_view.permute(0, 1, 3, 2).contiguous()
    return shuffle_out.view(batch_size, channels, out_width)



class GatedConv(nn.Module):
    """
    Gated Convolution layer with Instance Normalization

    Args:
      Cin     (int):
      Cout    (int):
      W       (int): kernel width
      stride  (int): stride width
      padding (int): padding in both side
    """

    def __init__(self, Cin, Cout, W, stride, padding):
        super(GatedConv, self).__init__()
        # normal conv
        self.cnv = nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding, bias=True)
        # for gating
        self.cnv_gate = nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding, bias=True)
        # summation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        A = self.cnv(input)
        B = self.cnv_gate(input)
        return A * self.sigmoid(B)


class GatedConvIN(nn.Module):
    """
    Gated Convolution layer with Instance Normalization

    Args:
      Cin     (int):
      Cout    (int):
      W       (int): kernel width
      stride  (int): stride width
      padding (int): padding in both side
    """

    def __init__(self, Cin, Cout, W, stride, padding):
        super(GatedConvIN, self).__init__()
        # normal conv
        self.cnv = nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding, bias=True)
        self.inNorm = nn.InstanceNorm1d(Cout)
        # for gating
        self.cnv_gate = nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding, bias=True)
        self.inNorm_gate = nn.InstanceNorm1d(Cout)
        # summation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        A = self.inNorm(self.cnv(input))
        B = self.inNorm_gate(self.cnv_gate(input))
        return A * self.sigmoid(B)

class GatedConvPSIN(nn.Module):
    """
    Gated Convolution layer with Instance Normalization

    Args:
      Cin     (int):
      Cout    (int):
      W       (int): kernel width
      stride  (int): stride width
      padding (int): padding in both side
      r       (int): PixelShuffle scaling factor
      Cfinal  (int): output Channel size
    """

    def __init__(self, Cin, Cout, W, stride, padding, r, Cfinal):
        super(GatedConvPSIN, self).__init__()
        # normal conv
        self.cnv = nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding)
        self.ps = PixelShuffle1d(r)
        self.inNorm = nn.InstanceNorm1d(Cfinal)
        # for gating
        self.cnv_gate = nn.Conv1d(Cin, Cout, W, stride=stride, padding=padding)
        self.ps_gate = PixelShuffle1d(r)
        self.inNorm_gate = nn.InstanceNorm1d(Cfinal)
        # summation
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        A = self.inNorm(self.ps(self.cnv(input)))
        B = self.inNorm_gate(self.ps_gate(self.cnv_gate(input)))
        return A * self.sigmoid(B)

class ResidualBottleneckBlock(nn.Module):
    def __init__(self):
        super(ResidualBottleneckBlock, self).__init__()
        self.units = nn.ModuleList([
            # (N_batch, 1024, T/4)
            GatedConvIN(512, 1024, 3, stride=1, padding=1),
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


class GatedFullyConvNet(nn.Module):
    def __init__(self):
        super(GatedFullyConvNet, self).__init__()
        # input: (N_batch, 24, T)
        unt = [
            GatedConv(24, 128, 15, stride=1, padding=7),   # (N_batch, 128, T)
            GatedConvIN(128, 256, 5, stride=2, padding=2), # (N_batch, 256, T/2)
            GatedConvIN(256, 512, 5, stride=2, padding=2), # (N_batch, 512, T/4)
        ]
        # x6 loop
        unt.extend([ResidualBottleneckBlock() for i in range(6)])  # (N_batch, 512, T/4)
        unt.extend([
            GatedConvPSIN(512, 1024, 5, stride=1, padding=2, r=2, Cfinal=512), # (N_batch, 1024 == 2x2x256, T/4) ->  (N_batch, 512, T/2)
            GatedConvPSIN(512, 512, 5, stride=1, padding=2, r=2, Cfinal=256),  # (N_batch, 512 == 2x2x128, T/2) ->    (N_batch, 156, T)
            nn.Conv1d(256, 24, 15, stride=1, padding=7)                        # (N_batch, 24, T)
        ])
        # output: (N_batch, 24, T)
        self.units = nn.ModuleList(unt)

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
