import torch.nn as nn

class anyGLU(nn.Module):
    """
    Convert any modules into Gated Linear Unit
    """
    def __init__(self, myModule, *modulePosArgs, **moduleKwdArgs):
        """
        Assign a module into this GLU unit

        Args:
            *modulePosArgs: positional Arguments of myModule
            **moduleKwdArgs: keyword Arguments of myModule
        """
        super(anyGLU, self).__init__()
        self.process = myModule(*modulePosArgs, **moduleKwdArgs)
        self.process_gate = myModule(*modulePosArgs, **moduleKwdArgs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        return self.process(input) * self.sigmoid(self.process_gate(input))
