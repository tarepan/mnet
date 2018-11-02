import numpy as np
import librosa
from .basics.getFeatures import convertWavIntoF0seqMCEPseq


class Clop(object):
    """
    Clop audio

    Args:
        waveform (numpy.ndarray(MCEPdim, T/frame_period))
    """
    def __init__(self):
        self.lgth = 128
    def __call__(self, waveform):
        in_h = 24
        in_w = waveform.shape[1]

        start = np.random.randint(0, (in_w-1) - self.lgth)
        clopped = waveform[:, start:start+self.lgth]
        return clopped

class ToNormedMCEPseq(object):
    """
    Convert waveform into a person-normalized MCEP sequence

    Args:
        MCEP_means
        MCEP_stds

    Returns:
        (numpy.ndarray) normalized MCEP sequence
    """
    def __init__(self, sampling_rate, MCEP_means, MCEP_stds):
        self.sampling_rate = sampling_rate
        self.MCEP_means = MCEP_means
        self.MCEP_stds = MCEP_stds

    def __call__(self, waveform):
        _, MCEPseq = convertWavIntoF0seqMCEPseq(waveform, self.sampling_rate)
        normedMCEPseq = (MCEPseq - self.MCEP_means)/self.MCEP_stds
        return normedMCEPseq


class Resample(object):
    """
    Resample waveform into target sampling rate

    Args:
        s_sr (int): source sampling rate
        t_sr (int): target sampling rate
    Returns:
        (numpy.ndarray) normalized MCEP sequence
    """
    def __init__(self, s_sr, t_sr):
        self.s_sr = s_sr
        self.t_sr = t_sr

    def __call__(self, waveform):
        return librosa.core.resample(waveform, self.s_sr, self.t_sr)
