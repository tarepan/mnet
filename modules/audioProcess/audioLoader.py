import os
import librosa
import torch
from torch.utils.data import Dataset

class ToTensor(object):
    """
    Convert numpy.ndarray into Pytorch.Tensor
    """
    def __call__(self, sample):
        return torch.from_numpy(sample)

class VCC2016Dataset(Dataset):
    """
    Dataset of Voice Conversion Contest 2016 (VCC2016)
    """

    def __init__(self, dirPath, samplingRate, transform=None):
        """
        Args:
            dirPath (string): Path of directory which contain .wav audios
        """
        self.dirPath = dirPath
        self.fileNames = os.listdir(dirPath)
        self.samplingRate = samplingRate
        self.transform = transform
        # .map(lambda filePath: ))

    def __len__(self):
        """
        Return the size of VCC2016 dataset
        """
        return len(self.fileNames)

    def __getitem__(self, idx):
        """
        Load proper wav file as numpy.ndarray(1, T) waveform
        """
        name = self.fileNames[idx]
        filePath = os.path.join(self.dirPath, name)
        waveform = librosa.load(filePath, sr = self.samplingRate, mono = True)[0]
        print(f"\nloaded in __getitem__!! type is ${waveform.dtype}")
        print(waveform)

        if self.transform:
            waveform = self.transform(waveform)
            print("transformed!")
            print(waveform)
        return waveform
