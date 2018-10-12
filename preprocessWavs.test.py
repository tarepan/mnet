from audioProcess.getAudioStats import getAudioStats

def test():
    # train_A_dir = "./data/vcc2016/vcc2016_training/smallSF1"
    train_A_dir = "./data/vcc2016/vcc2016_training/SF1"
    sampling_rate = 16000
    logF0_mean, logF0_std, MCEP_means, MCEP_stds = getAudioStats(train_A_dir, sampling_rate)
    print(f"LogF0\nmean: {logF0_mean}, std: {logF0_std}");
    print(f"MCEP means:")
    print(MCEP_means)
    print(f"MCEP stds:")
    print(MCEP_stds)

if __name__ == "__main__":
    test()
