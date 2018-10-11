# module import rules: https://note.nkmk.me/python-import-usage/ (based on PEP8)
import time

import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from .train import train
from .test import test
from .networks.cnn import CNN
from .loaders import getLoaders
class Args:
  def __init__(self):
    self.batch_size = 500
    self.test_batch_size = 1000
    self.epochs = 10
    self.lr = 0.01
    self.momentum = 0.5
    self.no_cuda = False
    self.seed = 1
    self.log_interval = 10

def main():
    # acquire settings
    ## with Args lass
    args = Args()

    # settings
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}


    # initialization
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    writer = SummaryWriter("log")
#     writer = SummaryWriter("log/run1") # record instance

    # preprocess data and push them into loaders
    train_loader, test_loader = getLoaders(args, kwargs)

    # repeating traning and test
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == "__main__":
    main()
