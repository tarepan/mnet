import time
import itertools

from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.optim as optim

from mnet.getConfigs import getConfigs
from mnet.audioProcess.getAudioStats import getAudioStats
from mnet.audioProcess.AudioDataset import MonoAudioDataset
from mnet.saveLoad import resumeTraining, saveModels

# from .trains.train_CycleGAN import train
from .trains.train_CycleGAN_NS import train
from .tests.test_CycleGAN_VC import test
from .networks.CycleGAN_VC_G import GatedFullyConvNet1d as FullyGCNN
# from .networks.CycleGAN_VC_D import GatedCNN2D as GatedCNN
from .networks.CycleGAN_VC_D_NS import GatedCNN2D as GatedCNN

def main(args, train_a_dir, train_b_dir, modelBase, evalDirA, evalDirB):
    # initialization
    ## devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ## Nets
    net_G_A2B, net_G_B2A = FullyGCNN().to(device), FullyGCNN().to(device)
    net_D_A, net_D_B = GatedCNN().to(device), GatedCNN().to(device)
    resumeEpoch = resumeTraining([net_G_A2B, net_G_B2A, net_D_A, net_D_B],
        [modelBase/"net_G_A2B.pth", modelBase/"net_G_B2A.pth", modelBase/"net_D_A.pth", modelBase/"net_D_B.pth"],
        modelBase/"epoch.txt")
    ## Optimizers & LR schedulers
    opt_G = optim.Adam(itertools.chain(net_G_A2B.parameters(), net_G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    opt_D_A = optim.Adam(net_D_A.parameters(), lr=0.0001, betas=(0.5, 0.999))
    opt_D_B = optim.Adam(net_D_B.parameters(), lr=0.0001, betas=(0.5, 0.999))
    linearDecay = lambda epoch: 1 if epoch < 2500 else -40.5*10^-5*epoch + 2
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=linearDecay)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda=linearDecay)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda=linearDecay)
    ## TensorBoard
    writer = SummaryWriter("log")
    ## Data
    _, _, trainLoader_a = acquireF0statAndAudioLoader(train_a_dir, args.sampling_rate, args)
    _, _, trainLoader_b = acquireF0statAndAudioLoader(train_b_dir, args.sampling_rate, args)
    featStats_a = np.load(evalDirA/"featureStats"/"stats.npz")
    featStats_b = np.load(evalDirB/"featureStats"/"stats.npz")

    # Evaluation prep

    # repeating traning and test
    start = time.time()
    for epoch in range(resumeEpoch if type(resumeEpoch) is int else 1, args.epochs + 1):
    # for epoch in range(1, 10):
        train(args, net_G_A2B, net_G_B2A, net_D_A, net_D_B, device, trainLoader_a, trainLoader_b, opt_G, opt_D_A, opt_D_B, epoch, writer)
        lr_scheduler_G.step(), lr_scheduler_D_A.step(), lr_scheduler_D_B.step()
        if epoch % 10 == 0:
            test(args, net_G_A2B, net_G_B2A, net_D_A, net_D_B, device, args.sampling_rate, featStats_a, featStats_b, evalDirA, evalDirB, epoch, writer)
            saveModels([net_G_A2B, net_G_B2A, net_D_A, net_D_B],
                [modelBase/"net_G_A2B.pth", modelBase/"net_G_B2A.pth", modelBase/"net_D_A.pth", modelBase/"net_D_B.pth"],
                 epoch, modelBase/"epoch.txt")
    print(f"total time: {time.time() - start}")

from torch.utils.data import DataLoader
from mnet.transforms import Compose, ToTensor, ActivateRequiresGrad
from mnet.audioProcess.transforms import Clop
from mnet.audioProcess.transforms import ToNormedMCEPseq
from mnet.Dataset import NumpyDataset

def acquireF0statAndAudioLoader(train_dir, sampling_rate, args):
    if False:
        logF0_mean, logF0_std, MCEP_means, MCEP_stds = getAudioStats(train_dir, sampling_rate)

        vcc2016_normed_MCEPseqs = MonoAudioDataset(train_dir, sampling_rate, transform=Compose([
            ToNormedMCEPseq(sampling_rate, MCEP_means, MCEP_stds),
            Clop(),
            ToTensor(),
            ActivateRequiresGrad()
        ]))
        trainLoader = DataLoader(vcc2016_normed_MCEPseqs, batch_size=10, shuffle=False)
        return logF0_mean, logF0_std, trainLoader
    else:
        vcc2016_normed_MCEPseqs = NumpyDataset(train_dir/"MCEPseqs", transform=Compose([
            Clop(),
            ToTensor(),
            ActivateRequiresGrad()
        ]))
        loader = DataLoader(vcc2016_normed_MCEPseqs, batch_size=args.batch_size, shuffle=True)
        return None, None, loader


if __name__ == "__main__":
    from pathlib import Path
    root = Path(__file__).parent.parent.parent.resolve()
    train_a_dir = root/"data"/"vcc2016"/"vcc2016_training"/"SF1"
    train_b_dir = root/"data"/"vcc2016"/"vcc2016_training"/"TF2"

    modelBase = Path(__file__).parent.resolve()/"trials"/"trial1"/"models"
    print(f"modelbase: {modelBase}")
    evalDirA = root/"data"/"vcc2016"/"evaluation_all"/"SF1"
    evalDirB = root/"data"/"vcc2016"/"evaluation_all"/"TF2"

    args = getConfigs("f")
    args.batch_num_limit_train = 1
    args.batch_num_limit_test = 1

    main(args, train_a_dir, train_b_dir, modelBase, evalDirA, evalDirB)
