import torch
from torchvision import datasets, transforms

def getLoaders(args, kwargs):
    # data download and preprocessing for training
    train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
      transform=transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))
      ])),
      batch_size=args.batch_size, shuffle=True, **kwargs)
    # data download and preprocessing for test
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return train_loader, test_loader
