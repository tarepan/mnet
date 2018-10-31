import os
import librosa
from torch.utils.data import Dataset
from ..Dataset import FileDataset

class MonoAudioDataset(FileDataset):
    """
    Monaural Audio Dataset
    """

    def __init__(self, dirPath, samplingRate, transform=lambda wave: wave):
        """
        Args:
            dirPath (string): a Path of a directory which contains audio files
            samplingRate (int): audio sampling rate
            transform (function): a transform function
        """
        super().__init__(dirPath, recursive=False, transform=transform)
        self.samplingRate = samplingRate

    def __getitem__(self, idx):
        """
        Load a audio file as a numpy.ndarray(T,) monaural waveform
        """
        name = self.fileNames[idx]
        filePath = os.path.join(self.dirPath, name)
        waveform = librosa.load(filePath, sr = self.samplingRate, mono = True)[0]
        return self.transform(waveform)
