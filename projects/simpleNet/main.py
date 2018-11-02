import time
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from modules.trainings.train import train
from modules.tests.test import test
from modules.networks.cnn import CNN
from modules.loaders import getLoaders
from modules.getConfigs import getConfigs
from modules.saveLoad import resumeTraining, saveModels

def main():
    # settings
    args = getConfigs("f")
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # initialization & data preprocess
    model = CNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    writer = SummaryWriter("log")
    train_loader, test_loader = getLoaders(args, kwargs)
    resumeTraining([model], ["./trials/trial1/models/model.pth"])
    # repeating traning and test
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, writer)
        test(args, model, device, test_loader, epoch, writer)
        # save models
        saveModels([model], ["./trials/trial1/models/model.pth"])
    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == "__main__":
    main()
