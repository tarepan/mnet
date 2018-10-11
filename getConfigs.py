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

def getConfigs(type="f"):
    if(type=="f"):
        # main function as callee
        return Args()
    elif(type=="cmd"):
        # main function as __main__
        # Training settings
        parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
        parser.add_argument('--batch-size', type=int, default=64, metavar='N',
        help='input batch size for training (default: 64)')
        parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',
        help='number of epochs to train (default: 10)')
        parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
        help='learning rate (default: 0.01)')
        parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
        help='SGD momentum (default: 0.5)')
        parser.add_argument('--no-cuda', action='store_true', default=False,
        help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
        help='random seed (default: 1)')
        parser.add_argument('--log-interval', type=int, default=10, metavar='N',
        help='how many batches to wait before logging training status')
        return parser.parse_args()
    else:
