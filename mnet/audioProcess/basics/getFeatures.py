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
    wav = wav.astype(np.float64) # np.ndarray -> np.ndarray(number is float64)
    f0seq, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    spetrogram = pyworld.cheaptrick(wav, f0seq, timeaxis, fs)
    MCEPseq = pyworld.code_spectral_envelope(spetrogram, fs, MCEPdim)
    return f0seq, MCEPseq.T.astype(np.float32)

def convertWavIntoFeatures(wav, fs, frame_period = 5.0, MCEPdim = 24):
    # basic features
    wav = wav.astype(np.float64) # np.ndarray -> np.ndarray(number is float64)
    f0seq, timeaxis = pyworld.harvest(wav, fs, frame_period = frame_period, f0_floor = 71.0, f0_ceil = 800.0)
    spectrogram = pyworld.cheaptrick(wav, f0seq, timeaxis, fs)
    MCEPseq = pyworld.code_spectral_envelope(spectrogram, fs, MCEPdim)
    APseq = pyworld.d4c(wav, f0seq, timeaxis, fs)
    # argumentation
    # print("wavIntoFeatures size")
    # print(f"f0seq: {f0seq.shape}, MCEPseq_before_T: {MCEPseq.shape}, APseq: {APseq.shape}")
    return f0seq, MCEPseq.T.astype(np.float32), APseq

def convertFeaturesIntoWav(f0seq, MCEPseq, APseq, fs, frame_period = 5.0):
    contNumpy_MCEPseq = np.ascontiguousarray(MCEPseq.T, dtype=np.float64)
    fftlen = pyworld.get_cheaptrick_fft_size(fs)
    spectrogram = pyworld.decode_spectral_envelope(contNumpy_MCEPseq, fs, fftlen)
    # print(f"dtypes. f0seq:{f0seq.dtype}, spectrogram:{spectrogram.dtype}, APseq:{APseq.dtype}")
    wav = pyworld.synthesize(f0seq, spectrogram, APseq, fs, frame_period)
    return wav.astype(np.float32)
