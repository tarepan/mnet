# model sample: https://github.com/pytorch/examples/blob/master/mnist/main.py
import torch.nn as nn

class View(nn.Module):
  def __init__(self):
    super(View, self).__init__()
  def forward(self, x):
    return x.view(-1,320)

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.units = nn.ModuleList([
      nn.Conv2d(1, 10, (5,5)),
      nn.MaxPool2d((2,2)),
      nn.ReLU(),
      nn.Conv2d(10, 20, (5,5)),
      nn.Dropout2d(),
      nn.MaxPool2d((2,2)),
      nn.ReLU(),
      View(),
      nn.Linear(320, 50),
      nn.ReLU(),
      nn.Dropout(), # maybe too high
      nn.Linear(50, 10),
      nn.LogSoftmax(1)
    ])
  def forward(self, x):
    for f in self.units:
      x = f(x)
    return x
