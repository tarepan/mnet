# mnet - Modular neural NETwork library
mnet (**m**odular neural **net** work library) is modules for stress-free PyTorch coding.  
mnet divides complex neural network codes into common tiny building blocks (modules) and offer them as extendable library.  
As a library, mnet provides utility nn.Modules (e.g. Gated Linear Units), loaders and so on.  
Let's start stress-free PyTorch coding with mnet!!  

## How to use
by importing as library with pip as below,  
```bash
# with pipenv
pipenv install git+https://github.com/tarepan/mnet#egg=mnet
# or, with pip
pip install git+https://github.com/tarepan/mnet#egg=mnet
```
then, you can use useful modules.  
With mnet, you can make complex "Gated Convolutional Network" with simple code as
```python
from mnet import mnetBase, anyGLU
class GatedCNN(mnetBase):
  __init__(GatedCNN, self):
    units = nn.ModuleList([
      anyGLU(nn.Conv2d, kernel=3)
    ])
```
Super simple!!  

## Module lists
* mnetBase: automatic "forward" nn.Modules
* anyGLU: convert any nn.Modules into "Gated" version
* multiLoader: load random data from multiple "dataset" even if datasets have different number of data
