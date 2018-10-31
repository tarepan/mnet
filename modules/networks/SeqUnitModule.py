import torch.nn as nn

class SeqUnitModule(nn.Module):
    """
    Sequential Units Module, which is used as base class for sequential "self.units" calculation (like nn.sequential)
    """
    def __init__(self, ):
        super(SeqUnitModule, self).__init__()
        # define units/layers with self.units = nn.ModuleList([nn.Module derivatives])

    def forward(self, x):
        for f in self.units:
            x = f(x)
        return x

class TemplateUnit(SeqUnitModule):
    def __init__(self, args):
        super(TemplateUnit, self).__init__()
        self.units = nn.ModuleList([
            nn.Conv2d(1,1,(2,2))
        ])
