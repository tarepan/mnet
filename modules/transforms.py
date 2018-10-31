import torch

class ToTensor(object):
    """
    Convert numpy.ndarray into Pytorch.Tensor
    """
    def __call__(self, sample):
        return torch.from_numpy(sample)

class ActivateRequiresGrad(object):
    """
    Activate tensor.requires_grad
    """
    def __call__(self, tensor):
        tensor.requires_grad = True
        return tensor

class Compose(object):
    """
    Compose transforms by sequential apply

    Args:
        transforms (list): list of callable transform
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
