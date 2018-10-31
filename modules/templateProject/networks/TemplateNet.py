# should install mnet library
import torch.nn as nn
from mnet.networks.SeqUnitModule import SeqUnitModule

class TemplateNet(SeqUnitModule):
    def __init__(self):
        super(TemplateNet, self).__init__()
        self.units = nn.ModuleList([
            nn.Conv2d(1,  32, (3, 9), (1, 1), (1, 4)),
            MyMmodule(32,  1, 5)
        ])

    # if you use SeqUnitModule and self.units, sequential unit calculation is implemented in parental SeqUnitModule
    # def forward(self, x):
    #     for f in self.units:
    #         x = f(x)
    #     return x

    # with print check
    # def forward(self, x):
    #     cnt = 0
    #     print(f"\ncount {cnt}: before apply {x.size()}")
    #     for f in self.units:
    #         x = f(x)
    #         print(f"count {cnt}: after apply {x.size()}\n")
    #         cnt = cnt + 1
    #     return x
