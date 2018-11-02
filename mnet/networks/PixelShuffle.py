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
