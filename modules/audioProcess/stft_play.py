import os
import time
import librosa
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['svg.fonttype'] = 'none'
import numpy as np
from scipy.io.wavfile import read as readwav

# Constants
n_fft = 1024
hop_length = 256
SR = 16000
over_sample = 4
res_factor = 0.8
octaves = 6
notes_per_octave=10



def note_specgram(path, distPath, ax, peak=70.0, use_cqt=True, use_mask=True):
  # Add several samples together
  audio, sr = librosa.core.load(path, 16000)
  audio = audio.astype(np.float32)
  stt = time.time()
  if use_cqt:
    C = librosa.cqt(audio, sr=sr, hop_length=hop_length,
                      bins_per_octave=int(notes_per_octave*over_sample),
                      n_bins=int(octaves * notes_per_octave * over_sample),
                      filter_scale=res_factor,
                      fmin=librosa.note_to_hz('C2'))
  else:
    C = librosa.stft(audio, n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)
  print(f"stft time: {time.time() - stt}[sec]")
  mag, phase = librosa.core.magphase(C)
  phase_angle = np.angle(phase)
  reconstructed = librosa.core.istft(C, hop_length=hop_length, win_length=n_fft, center=True)
  librosa.output.write_wav(distPath, reconstructed, sr)
  # phase_unwrapped = np.unwrap(phase_angle)
  # dphase = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
  # dphase = np.concatenate([phase_unwrapped[:, 0:1], dphase], axis=1) / np.pi
  # mag = (librosa.logamplitude(mag**2, amin=1e-13, top_db=peak, ref_power=np.max) / peak) + 1
  # mag = librosa.core.amplitude_to_db(mag**2, amin=1e-13, top_db=peak)
  # [matplotlib.pyplot.matshow](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.matshow.html)

note_specgram("correct_m1.wav", "restore_m1.wav", None, use_cqt=False)
