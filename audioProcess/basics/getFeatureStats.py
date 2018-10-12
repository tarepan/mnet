import numpy as np

def getLogF0Stat(f0Seqs):
    """
    Acquire a F0 mean and std from array of f0

    Args:
        f0Seqs ([numpy.ndarray]): list of f0 sequence (each contents is f0 timeseries of a waveform, so numpy.ndarray(1,T))

    Returns:
        tuple: f0_mean & f0_std
    """
    # concatenation for calculation trick
    log_f0seq_concatenated = np.ma.log(np.concatenate(f0Seqs))
    log_f0s_mean = log_f0seq_concatenated.mean()
    log_f0s_std = log_f0seq_concatenated.std()
    return log_f0s_mean, log_f0s_std

def getMCEPStat(mceps):
    """
    Acquire MCEP means & stds per dimension of many MCEP sequences (== from many waveform)

    Args:
        mceps ([np.ndarray]): array of a MCEP sequence (if MCEP_dim == 24, single contents == numpy.ndarray(24, T))

    Returns:
        tuple: MCEP_means & MCEP_stds (if MCEP_dim == 24, means == numpy.ndarray(24,1))
    """
    mceps_concatenated = np.concatenate(mceps, axis = 1)
    mceps_mean = np.mean(mceps_concatenated, axis = 1, keepdims = True)
    mceps_std = np.std(mceps_concatenated, axis = 1, keepdims = True)

    return mceps_mean, mceps_std
