# mnet - Modular neural NETwork library
mnet (**m**odular neural **net** work library) is modules/templates for stress-free PyTorch coding.  
mnet divides complex neural network codes into common tiny building blocks (modules) and offer them as extendable library.    
As a library, mnet provides utility nn.Modules (e.g. Gated Linear Units).  
As a template, mnet provides well-structured format with one-line execution both in local and Google Colaboratory.  
Let's start stress-free PyTorch coding with mnet!!  

## mnet as utility module library
by importing as library with pip as below,  
```bash
# pipenv
pipenv install mnet
# pip
pip install mnet
```
then, you can use useful modules,
```python
from mnet import mnetBase, anyGLU
class MyNet(mnetBase):
  __init__(MyNet, self):
    units = nn.ModuleList([
      anyGLU(nn.Conv2d, kernel=3)
    ])
  # mnetBase provide automatic forward function!!  
  # If you needs forward by yourself, ofcource you can override forward
```

## mnet as templates
Clone this project as below.
```bash
git clone xxxx
```
then, make it to your network!!

if you want to run networks locally, what you should do is simply,
```
# pipenv
pipenv run python run_local.py
```
if you want to run networks in Google Colaboratory, in surprise, simply jump to  
```
```
then, it is run on Google Colaboratory!! Super easy!!

## 1-click run with Google Colaboratory
Jump to below link, then run all code (manually or "runtime"->"execute all cells")  
[Go to Colab](https://colab.research.google.com/github/tarepan/mnet/blob/master/mnet.ipynb)

## Arch.
* DNN
* other

preprocess should be independent from
