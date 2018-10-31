import pyworld as pw
import librosa

# filepath = "./data/vcc2016/vcc2016_training/smallSM1/wavs/100001.wav"
filepath = "./data/vcc2016/vcc2016_training/SM1/100006.wav"
waveform = librosa.core.load(filepath, sr = 16000, mono = True, dtype="float64")[0]
print(waveform)

fs = 16000
_f0, t = pw.dio(waveform, fs)    # raw pitch extractor
f0 = pw.stonemask(waveform, _f0, t, fs)  # pitch refinement
sp = pw.cheaptrick(waveform, f0, t, fs)  # extract smoothed spectrogram

# MCEPseq = pw.code_spectral_envelope(sp, fs, 24)
# fftlen = pw.get_cheaptrick_fft_size(fs)
# sp = pw.decode_spectral_envelope(MCEPseq, fs, fftlen)

ap = pw.d4c(waveform, f0, t, fs)         # extract aperiodicity
y = pw.synthesize(f0, sp, ap, fs)



print(y)
librosa.output.write_wav("./re.wav", y.astype("float32"), fs)
