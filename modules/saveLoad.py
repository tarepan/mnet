import os
import torch
import numpy as np

def saveModels(models, paths, epoch, epochPath):
    """
    Save Network models and epoch into specified paths
    """
    for model, path in zip(models, paths):
        torch.save(model.state_dict(), path)
    # save epoch
    print(f"epochPath: {epochPath}")
    np.savetxt(epochPath, [epoch])

def loadModelParams(modelInstance, path):
    modelInstance.load_state_dict(torch.load(path))

def resumeTraining(modelInstances, paths, epochPath):
    """
    Resume training from checkpoint
    """
    for modelInstance, path in zip(modelInstances, paths):
        if os.path.isfile(path):
            loadModelParams(modelInstance, path)
            print("Resume! Model parameters loaded.")
        else:
            print("No checkpoint data. Start training from scratch.")
    # load and return epoch
    if os.path.isfile(epochPath):
        epc = np.loadtxt(epochPath, dtype="int32")
        print(f"resume from epoch {epc}")
        return epc.item()
    else:
        print("no resume epoch info. Start training from epoch 1.")
        return None
