import os

import librosa
from functional import seq

def loadWavs(dirPath, sr):
    """
    Load all .wav files in wavDir directory with sr sampling rate.

    Args:
        dirPath (str): directory path
        sr (int): sampling rate

    Returns:
        ([np.ndarray(T,)]): waveform list
    """
    return (seq(os.listdir(dirPath))
        .map(lambda name: dirPath/f"{name}")
        .filter(lambda path: os.path.isfile(path))
        .map(lambda filePath: librosa.load(filePath, sr = sr, mono = True)[0])
        .to_list())

def loadWavsFromDirs(dirPathList, sr):
    """
    Load all .wav files in wavDir directory List with sr sampling rate

    Args:
        dirPathList ([str]): directory path list
        sr (int): sampling rate

    Returns:
        iterable: iterable waveforms (single datum: np.ndarray(1, T))
    """
    lists = (seq(dirPathList)
        .map(lambda dirPath: loadWavs(dirPath, sr))
        .to_list()
        )
    # super kuso-code
    flatten = []
    for ls in lists:
        flatten.extend(ls)
    return flatten
