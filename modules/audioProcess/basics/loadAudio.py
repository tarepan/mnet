import librosa
import numpy as np
import os

from functional import seq

def loadWavs(dirPath, sr):
    """
    Load all .wav files in wavDir directory with sr sampling rate

    Args:
        dirPath (str): directory path
        sr (int): sampling rate

    Returns:
        iterable: iterable waveforms (single datum: np.ndarray(1, T))
    """
    return (seq(os.listdir(dirPath))
        .map(lambda fileName: os.path.join(dirPath, fileName))
        .map(lambda filePath: librosa.load(filePath, sr = sr, mono = True)[0]))
