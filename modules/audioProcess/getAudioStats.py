from .basics.loadAudio import loadWavs
from .basics.getFeatures import convertWavIntoF0seqMCEPseq
from .basics.getFeatureStats import getLogF0Stat, getMCEPStat

def getAudioStats(wavDirPath, sampling_rate):
    """
    Calculate logF0 and MCEP's means/std

    Args:
        wavDirPath (str): path of audio directory
        sampling_rate (int): sampling rate of audios which are in wavDirPath directory

    Returns:
        logF0_mean, logF0_std, MCEP_mean, MCEP_std
    """
    # disc => np.ndarray(1,T)
    waveforms = loadWavs(wavDirPath, sampling_rate)
    # [np.ndarray(1,T)] => [(np.ndarray(1, frames), np.ndarray(24MCEPs, frames)]
    sets = waveforms.map(lambda waveform: convertWavIntoF0seqMCEPseq(waveform, sampling_rate))
    f0seqs = sets.map(lambda set: set[0]).to_list()
    MCEPseqs = sets.map(lambda set: set[1]).to_list()
    logF0_mean, logF0_std = getLogF0Stat(f0seqs)
    MCEP_mean, MCEP_std = getMCEPStat(MCEPseqs)
    return logF0_mean, logF0_std, MCEP_mean, MCEP_std
