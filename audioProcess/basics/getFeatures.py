import numpy as np
import pyworld

def convertWavIntoF0seqMCEPseq(wav, fs, frame_period = 5.0, MCEPdim = 24):
    """
    Extract a F0 sequence and a MCEP sequence from a single waveform

    Args:
        wav (np.ndarray(1,T)): waveform
        fs :
        frame_period (float): [ms]
        MCEPdim (int): dimension of Mel CEPstral analysis

    Returns:
        tuple: f0seq (np.ndarray(1, T/frame_period)) & MCEPseq (np.ndarray(MCEPdim, T/frame_period))
    """
    print("pyworld start")
    wav = wav.astype(np.float64) # np.ndarray -> np.ndarray(number is float64)
    f0seq, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    spetrogram = pyworld.cheaptrick(wav, f0seq, timeaxis, fs)
    MCEPseq = pyworld.code_spectral_envelope(spetrogram, fs, MCEPdim)
    print("pyworld end")
    return f0seq, MCEPseq.T
