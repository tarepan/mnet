import os
import time
import numpy as np
from functional import seq
import librosa

from modules.audioProcess.AudioDataset import MonoAudioDataset
from modules.audioProcess.transforms import ToNormedMCEPseq, Resample
from modules.audioProcess.getAudioStats import getAudioStats, waves2stats
from modules.audioProcess.basics.loadAudio import loadWavsFromDirs
"""
Load, convert to MCEP, then save
"""
"""
use MonoAudioDataset and extract each files from it then save
"""

def preprocess(originDirPath, sampling_rate, distDirPath):
    print("Start Preprocessing...")
    start = time.time()
    logF0_mean, logF0_std, MCEP_means, MCEP_stds = getAudioStats(originDirPath, sampling_rate)
    featurePath = os.path.join(distDirPath, "featureStats", "stats.npz")
    np.savez(featurePath, logF0_mean=logF0_mean, logF0_std=logF0_std, MCEP_mean=MCEP_means, MCEP_std=MCEP_stds)
    print("feature stat saved!!")
    normed_MCEPseqs = MonoAudioDataset(originDirPath, sampling_rate, transform=ToNormedMCEPseq(sampling_rate, MCEP_means, MCEP_stds))
    for idx, MCEPseq in enumerate(normed_MCEPseqs):
        filePath = os.path.join(distDirPath, "MCEPseqs", normed_MCEPseqs.fileNames[idx])
        np.save(filePath+".npy", MCEPseq)
        # print("MCEPseq saved")
    print(f"Preprocessed!! {time.time() - start}")

def preprocessSingleSets(wavDirPaths, sr, distDirPathsStats, distDirPathsNormedMCEPseqs):
    print("Start Preprocessing...")
    start = time.time()

    waves = loadWavsFromDirs(wavDirPaths, sr)
    print(f"shape of wave list {len(waves)}")
    logF0_mean, logF0_std, MCEP_means, MCEP_stds = waves2stats(waves, sr)
    _ = (seq(distDirPathsStats)
        .map(lambda distDirPath: f"{distDirPath}/stats.npz")
        .map(lambda filePath: np.savez(filePath, logF0_mean=logF0_mean, logF0_std=logF0_std, MCEP_mean=MCEP_means, MCEP_std=MCEP_stds))
        .to_list())
    print("feature stat saved!!")

    for wavDirPath, distDirPath in zip(wavDirPaths, distDirPathsNormedMCEPseqs):
        normed_MCEPseqs = MonoAudioDataset(wavDirPath, sr, transform=ToNormedMCEPseq(sr, MCEP_means, MCEP_stds))
        for idx, MCEPseq in enumerate(normed_MCEPseqs):
            filePath = f"{distDirPath}/{normed_MCEPseqs.fileNames[idx]}"
            np.save(filePath, MCEPseq)
    print("MCEPseq acquisition & normalization finished!!")
    print(f"Preprocessed!! {time.time() - start}")


def resampling(wavDirPath, s_sr, t_sr, distDirPath):
    resampled_waves = MonoAudioDataset(wavDirPath, s_sr, transform=Resample(s_sr, t_sr))
    for idx, wave in enumerate(resampled_waves):
        filePath = f"{distDirPath}/{resampled_waves.fileNames[idx]}"
        librosa.output.write_wav(filePath, wave, t_sr)


if __name__ == "__main__":
    preprocessSingleSets(
        [
            "./data/vcc2016/evaluation_all/smallSF1/wavs",
            "./data/vcc2016/evaluation_all/smallSM1/wavs"
        ],
        16000,
        [
            "./data/vcc2016/evaluation_all/smallSF1/featureStats"
        ],[
            "./data/vcc2016/evaluation_all/smallSF1/MCEPseqs",
            "./data/vcc2016/evaluation_all/smallSM1/MCEPseqs"
    ])
