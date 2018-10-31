import os
import librosa
import numpy as np
from scipy.io.wavfile import read as readwav

def wave2rainbowgram(wav):
    """
    Convert a wavefrom into frequency-domain time series
    Args:
        wav (numpy.ndarray(n,)):
    Returns:
        logPower ():
        IF:
    """
    # Transform
    C = librosa.stft(wav, n_fft=2048, hop_length=512)

    # magnitude processing
    mag = np.abs(C) # (freq, time)
    ## intensity scaling
    logMag = np.log(mag)
    max, min = logMag.max(), logMag.min()
    mean = (max+min)/2
    normedLogMag = (logMag - mean)/(max - mean)
    ## frequency scaling
    melFilter = librosa.filters.mel(16000, 2048, 1025)
    for i in range(0, 1025):
        print(melFilter[i:i+1,:].max())
    print(f"mag dim: {normedLogMag.shape}")
    print(f"melFitler: {melFilter.shape}")
    print(melFilter)
    melProcessedMag = np.dot(melFilter, normedLogMag)
    print(f"dot produt dim: {melProcessedMag.shape}")
    # melLogScaledMag = np.pad(melProcessedMag, [(0,0), (0,2)], "constant", constant_values=-1)
    melLogScaledMag = normedLogMag
    # phase angle processing
    ## IF-nize
    phase_unwrapped = np.unwrap(np.angle(C))
    IF = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1] # finite difference
    ## intensity scaling
    scaled_IF = np.concatenate([phase_unwrapped[:, 0:1], IF], axis=1) / np.pi # (-pi ~ pi) => (-1, 1)
    ## frequency scaling
    return melLogScaledMag, scaled_IF


# path = "testaudio.wav"
path = "test2.mp3"
wave, sr = librosa.core.load(path, sr=16000)
wave = wave[:64000]
print(f"shape of wave: {wave.shape}")
processedMag, scaledIF = wave2rainbowgram(wave)
print("processedMag:")
print(processedMag)
print("partly")
print(processedMag[10:15, 100:115])
print(scaledIF)
print(f"IF max: {scaledIF.max()}, min: {scaledIF.min()}")
print(f"shape of processedMag: {processedMag.shape}")
# import matplotlib.pyplot as plt
#
# plt.imshow(IF)
# plt.show()
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.rcParams['svg.fonttype'] = 'none'
#
# # Plotting functions
# cdict  = {'red':  ((0.0, 0.0, 0.0),
# (1.0, 0.0, 0.0)),
#
# 'green': ((0.0, 0.0, 0.0),
# (1.0, 0.0, 0.0)),
#
# 'blue':  ((0.0, 0.0, 0.0),
# (1.0, 0.0, 0.0)),
#
# 'alpha':  ((0.0, 1.0, 1.0),
# (1.0, 0.0, 0.0))
# }
#
# my_mask = matplotlib.colors.LinearSegmentedColormap('MyMask', cdict)
# plt.register_cmap(cmap=my_mask)
#
def plot_rainbowgram(rainbowgrams, rows=2, cols=4, col_labels=[], row_labels=[]):
  """
    Plot rainbowgram
    Args:
        rainbowgrams ([(mag, IF)]): list of rainbowgram datum (tuple of power and IF)
  """
  # prepare graph overview
  fig, axes = plt.subplots(rows, cols, sharex=True, sharey=True)
  fig.subplots_adjust(left=0.1, right=0.9, wspace=0.05, hspace=0.1)
  # prepare subplot
  for i, path in enumerate(rainbowgrams):
    row = i / cols
    col = i % cols
    if rows == 1:
      ax = axes[col]
    elif cols == 1:
      ax = axes[row]
    else:
      ax = axes[row, col]

    ax.matshow(dphase[::-1, :], cmap=plt.cm.rainbow)
    ax.matshow(mag[::-1, :], cmap=my_mask)

    # cmap : Colormap
    ax.set_axis_bgcolor('white')
    ax.set_xticks([]); ax.set_yticks([])
    if col == 0 and row_labels:
      ax.set_ylabel(row_labels[row])
    if row == rows-1 and col_labels:
      ax.set_xlabel(col_labels[col])
