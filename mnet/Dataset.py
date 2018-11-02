import os
import numpy as np
from functional import seq
from torch.utils.data import Dataset

class FileDataset(Dataset):
    """
    Dataset which load numpy npy dataset
    """

    def __init__(self, dirPath, recursive=False, transform=lambda ndarray: ndarray):
        """
        Args:
            dirPath (string): a Path of a directory which contains data
            recursive (bool): flag for recursive file finding
            transform (function): a transform function
        """
        self.dirPath = dirPath
        if(recursive == False):
            self.fileNames = seq(os.listdir(dirPath)).filter(lambda name: os.path.isfile(dirPath/f"{name}")).to_list()
        else:
            raise "Not yet implemented"
        self.transform = transform

    def __len__(self):
        """
        Return the size of this dataset (number of files)
        """
        return len(self.fileNames)

    def __getitem__(self, idx):
        """
        Load a file
        """
        """
        # Examples
        name = self.fileNames[idx]
        filePath = os.path.join(self.dirPath, name)
        ## load process ## nparray = np.load(filePath)
        return self.transform(nparray)
        """
        raise NotImplementedError()

class NumpyDataset(FileDataset):
    """
    Dataset which load numpy npy dataset
    """
    def __init__(self, dirPath, transform=lambda ndarray: ndarray):
        """
        Args:
            dirPath (string): a Path of a directory which contains .npy files
            transform (function): a transform function
        """
        super().__init__(dirPath, recursive=False, transform=transform)

    def __getitem__(self, idx):
        """
        Load a .npy file as a numpy.ndarray
        """
        name = self.fileNames[idx]
        filePath = self.dirPath/f"{name}"
        nparray = np.load(filePath)
        return self.transform(nparray)
